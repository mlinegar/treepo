"""Built-in neural-operator families for :mod:`treepo.methods`.

``neural_operator`` is the generic family. It delegates named operator kinds
to ``neuralop.models`` when available and keeps ``operator_kind='conv1d'`` as
a tiny local baseline. The public ``family='fno'`` route is the concrete
Fourier neural-operator route over the same shared runtime. Use
``family='neural_operator'`` when selecting an operator kind explicitly, for
example ``operator_kind='fno'`` or the ``operator_kind='fourier'`` alias.
Dataset-specific structure belongs in tree fixtures or registered downstream
families; this module only owns the generic neural-operator method surface.

This module is intentionally lean: it holds the family runtime and the two
build helpers, and delegates the messy work to focused siblings —

* ``_fno_config``: config dataclasses and coercion.
* ``_fno_neuralop``: torch/neuralop discovery.
* ``_fno_models``: the f/g torch model.
* ``_fno_encoding``: leaf extraction and embedding (data-prep).
* ``_fno_targets``: supervision target extraction (data-prep).
* ``_fno_transition``: numeric transition-state supervision (data-prep).
* ``_fno_statistic``: the composable-statistic adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.llm.embedding import EmbeddingClient, HashingEmbeddingClient
from treepo.methods._fno_config import (
    FNOFamilyConfig,
    NeuralOperatorFamilyConfig,
    _clamp,
    _coerce_config,
    _config_payload,
    _normalize_operator_kind,
    _tensor_payload,
)
from treepo.methods._fno_encoding import (
    _coerce_embedding,
    _encode_numeric_leaf_features,
    _leaf_texts,
    _leaf_token_groups,
    _tree_sequence_cache_key,
)
from treepo.methods._fno_models import _TreeFGModel
from treepo.methods._fno_neuralop import _require_torch, _validate_operator_kind
from treepo.methods._fno_statistic import _NeuralOperatorStatistic
from treepo.methods._fno_targets import _target_rows
from treepo.methods._fno_transition import (
    _numeric_transition_state_loss,
    _numeric_transition_state_targets,
)


class NeuralOperatorFamily:
    """Generic neural-operator ``FamilyRuntime`` for tree root-score prediction."""

    name = "neural_operator"
    artifact_kind = "treepo_neural_operator"
    config_cls = NeuralOperatorFamilyConfig

    def __init__(
        self,
        config: NeuralOperatorFamilyConfig | None = None,
        *,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config or self.config_cls()
        self.operator_kind = _normalize_operator_kind(self.config.operator_kind)
        _validate_operator_kind(self.operator_kind, family_name=self.name)
        self.embedding_client = embedding_client or HashingEmbeddingClient(
            dim=self.config.embedding_dim,
            salt=self.config.embedding_salt,
        )
        self._torch = _require_torch()
        self._torch.manual_seed(int(self.config.seed))
        self._device = self._torch.device(str(self.config.device))
        self._model = None
        self._output_dim: int | None = None
        self._target_center = None
        self._target_scale = None
        self._last_artifact: dict[str, Any] | None = None
        # Encoding cache keyed by tree identity. Each entry pins the tree
        # objects it was built from, so the id()-based key stays valid for as
        # long as the entry lives; the cache is bounded, evicting oldest first.
        self._encoding_cache: dict[tuple[Any, ...], tuple[Any, Any, tuple[Any, ...]]] = {}
        self._encoding_cache_max_entries = 16

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del f_init, g, output_dir
        return self._train_side(kind="f", traces=traces, iteration=iteration)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del g_init, f, output_dir
        return self._train_side(kind="g", traces=traces, iteration=iteration)

    def _train_side(self, *, kind: str, traces: Sequence[Any], iteration: int) -> Mapping[str, Any]:
        train_g = kind == "g"
        trees, targets = _target_rows(traces, self.config)
        x, lengths = self._encode_trees(trees)
        y = self._torch.tensor(targets, dtype=self._torch.float32, device=self._device)
        self._ensure_model(output_dim=int(y.shape[1]))
        y_train = self._normalized_targets(y)
        last_loss = self._train_supervised(
            x,
            lengths,
            y_train,
            train_f=not train_g,
            train_g=train_g,
            trees=trees if train_g else None,
        )
        artifact = self._artifact_payload(
            kind=kind,
            iteration=iteration,
            n_train=int(x.shape[0]),
            loss=last_loss,
        )
        self._last_artifact = artifact
        return artifact

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[Any | None]:
        del f, g
        tree_list = list(trees or [])
        if not tree_list:
            return []
        if self._model is None:
            return [None] * len(tree_list)
        self._ensure_model(output_dim=self._output_dim)
        assert self._model is not None
        self._model.eval()
        x, lengths = self._encode_trees(tree_list)
        with self._torch.no_grad():
            raw = self._model(x, lengths)
            raw_values = self._denormalized_predictions(raw).detach().cpu().tolist()
        values = [row if isinstance(row, list) else [row] for row in raw_values]
        if (self._output_dim or 1) == 1:
            return [_clamp(float(row[0]), self.config.target_min, self.config.target_max) for row in values]
        return [
            [_clamp(float(value), self.config.target_min, self.config.target_max) for value in row]
            for row in values
        ]

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if kind in {"f", "g"} and not isinstance(artifact, Mapping):
            raise TypeError(f"family={self.name!r} {kind} artifact must be a mapping")

    def as_statistic(self, *, f: Any = None, g: Any = None) -> Any:
        del f, g
        if self._model is None:
            return None
        return _NeuralOperatorStatistic(self)

    def _artifact_kind_for_operator(self) -> str:
        if self.operator_kind == "fno":
            return "treepo_fno"
        return self.artifact_kind

    def _artifact_payload(self, *, kind: str, iteration: int, n_train: int, loss: float | None) -> dict[str, Any]:
        base_kind = self._artifact_kind_for_operator()
        return {
            "kind": base_kind if kind == "f" else f"{base_kind}_g",
            "trained": kind,
            "operator_kind": self.operator_kind,
            "iteration": int(iteration),
            "n_train": int(n_train),
            "loss": loss,
            "normalize_targets": bool(self.config.normalize_targets),
            "target_center": _tensor_payload(self._target_center),
            "target_scale": _tensor_payload(self._target_scale),
            "numeric_transition_state_weight": float(self.config.numeric_transition_state_weight),
            "output_dim": int(self._output_dim or 1),
            "target_key": self.config.target_key,
            "target_vector_key": self.config.target_vector_key,
            "config": _config_payload(self.config),
        }

    def _ensure_model(self, output_dim: int | None = None) -> None:
        if output_dim is not None:
            output_dim = max(1, int(output_dim))
            if self._output_dim is not None and self._output_dim != output_dim:
                raise ValueError(
                    f"family={self.name!r} output_dim changed from {self._output_dim} to {output_dim}"
                )
            self._output_dim = output_dim
        if self._output_dim is None:
            self._output_dim = 1
        if self._model is not None:
            return
        _validate_operator_kind(self.operator_kind, family_name=self.name)
        self._model = _TreeFGModel(
            operator_kind=self.operator_kind,
            config=self.config,
            torch=self._torch,
            output_dim=int(self._output_dim),
        ).to(self._device)

    def _normalized_targets(self, y: Any) -> Any:
        if not bool(self.config.normalize_targets):
            self._target_center = self._torch.zeros((int(y.shape[1]),), dtype=y.dtype, device=y.device)
            self._target_scale = self._torch.ones((int(y.shape[1]),), dtype=y.dtype, device=y.device)
            return y
        center = y.mean(dim=0)
        scale = y.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        self._target_center = center.detach()
        self._target_scale = scale.detach()
        return (y - center) / scale

    def _denormalized_predictions(self, y: Any) -> Any:
        if self._target_center is None or self._target_scale is None:
            return y
        return y * self._target_scale.to(device=y.device, dtype=y.dtype) + self._target_center.to(device=y.device, dtype=y.dtype)

    def _train_supervised(
        self,
        x: Any,
        lengths: Any,
        y: Any,
        *,
        train_f: bool,
        train_g: bool,
        trees: Sequence[Any] | None = None,
    ) -> float | None:
        assert self._model is not None
        self._set_trainable(train_f=train_f, train_g=train_g)
        params = [param for param in self._model.parameters() if bool(param.requires_grad)]
        if not params:
            return None
        self._model.train()
        opt = self._torch.optim.Adam(params, lr=float(self.config.learning_rate))
        batch_size = max(1, int(self.config.batch_size))
        epochs = max(1, int(self.config.epochs_per_iteration))
        last_loss = None
        n = int(x.shape[0])
        state_targets = (
            _numeric_transition_state_targets(trees, self.config, torch=self._torch, device=self._device)
            if train_g and trees is not None and float(self.config.numeric_transition_state_weight) > 0.0
            else None
        )
        for _epoch in range(epochs):
            order = self._torch.randperm(n, device=self._device)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if state_targets is None:
                    pred = self._model(x[idx], lengths[idx])
                    state_loss = None
                else:
                    pred, traces = self._model.forward_with_trace(x[idx], lengths[idx])
                    selected = [state_targets[int(i)] for i in idx.detach().cpu().tolist()]
                    state_loss = _numeric_transition_state_loss(
                        traces,
                        selected,
                        torch=self._torch,
                        device=self._device,
                        dtype=pred.dtype,
                    )
                loss = self._torch.nn.functional.mse_loss(pred, y[idx])
                if state_loss is not None:
                    loss = loss + float(self.config.numeric_transition_state_weight) * state_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().cpu())
        self._set_trainable(train_f=True, train_g=True)
        return last_loss

    def _set_trainable(self, *, train_f: bool, train_g: bool) -> None:
        assert self._model is not None
        for param in self._model.parameters():
            param.requires_grad = False
        modules = []
        if train_g:
            modules.extend([self._model.leaf_operator, self._model.merge])
        if train_f:
            modules.append(self._model.readout)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    def _encode_trees(self, trees: Sequence[Any]) -> tuple[Any, Any]:
        cache_key = _tree_sequence_cache_key(trees, dim=int(self.config.embedding_dim), device=str(self._device))
        cached = self._encoding_cache.get(cache_key)
        if cached is not None:
            return cached[0], cached[1]
        numeric_groups = [_leaf_token_groups(tree) for tree in trees]
        if numeric_groups and all(group is not None for group in numeric_groups):
            encoded = _encode_numeric_leaf_features(
                numeric_groups,
                dim=int(self.config.embedding_dim),
                torch=self._torch,
                device=self._device,
            )
        else:
            leaf_groups = [_leaf_texts(tree) for tree in trees]
            lengths = [max(1, len(group)) for group in leaf_groups]
            max_leaves = max(1, max(lengths))
            matrices = []
            for group in leaf_groups:
                texts = group or [""]
                vectors = self.embedding_client.embed_texts(texts)
                matrix = [_coerce_embedding(vec, int(self.config.embedding_dim)) for vec in vectors]
                while len(matrix) < max_leaves:
                    matrix.append([0.0] * int(self.config.embedding_dim))
                matrices.append(matrix[:max_leaves])
            x = self._torch.tensor(matrices, dtype=self._torch.float32, device=self._device)
            length_tensor = self._torch.tensor(lengths, dtype=self._torch.long, device=self._device)
            encoded = (x, length_tensor)
        while len(self._encoding_cache) >= self._encoding_cache_max_entries:
            self._encoding_cache.pop(next(iter(self._encoding_cache)))
        self._encoding_cache[cache_key] = (encoded[0], encoded[1], tuple(trees))
        return encoded


class FNOFamily(NeuralOperatorFamily):
    """Concrete Fourier neural-operator ``FamilyRuntime``."""

    name = "fno"
    artifact_kind = "treepo_fno"
    config_cls = FNOFamilyConfig


def build_neural_operator_family(backend_config: Mapping[str, Any]) -> NeuralOperatorFamily:
    """Build the generic neural-operator family from method ``backend_config``."""

    payload = dict(backend_config or {})
    raw_config = (
        payload.get("neural_operator_config")
        if "neural_operator_config" in payload
        else payload.get("no_config", payload.get("fno_config"))
    )
    config = _coerce_config(raw_config, payload, config_cls=NeuralOperatorFamilyConfig)
    embedding_client = payload.get("embedding_client")
    if embedding_client is not None and not hasattr(embedding_client, "embed_texts"):
        raise TypeError("backend_config['embedding_client'] must provide embed_texts(texts)")
    return NeuralOperatorFamily(config, embedding_client=embedding_client)


def build_fno_family(backend_config: Mapping[str, Any]) -> FNOFamily:
    """Build the concrete FNO family from method ``backend_config``."""

    payload = dict(backend_config or {})
    raw_config = (
        payload.get("fno_config")
        if "fno_config" in payload
        else payload.get("neural_operator_config", payload.get("no_config"))
    )
    config = _coerce_config(raw_config, payload, config_cls=FNOFamilyConfig)
    requested = _normalize_operator_kind(config.operator_kind)
    if requested != "fno":
        raise ValueError(
            "family='fno' only supports operator_kind='fno'; use "
            "family='neural_operator' for other operator kinds."
        )
    config.operator_kind = "fno"
    embedding_client = payload.get("embedding_client")
    if embedding_client is not None and not hasattr(embedding_client, "embed_texts"):
        raise TypeError("backend_config['embedding_client'] must provide embed_texts(texts)")
    return FNOFamily(config, embedding_client=embedding_client)


__all__ = [
    "FNOFamily",
    "FNOFamilyConfig",
    "NeuralOperatorFamily",
    "NeuralOperatorFamilyConfig",
    "build_fno_family",
    "build_neural_operator_family",
]
