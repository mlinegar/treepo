"""Built-in neural-operator families for :mod:`treepo.methods`.

``neural_operator`` is the generic family. It delegates named operator kinds
to ``neuralop.models`` when available and keeps ``operator_kind='conv1d'`` as
a tiny local baseline. The public ``family='fno'`` route is the concrete
Fourier neural-operator route over the same shared runtime. Use
``family='neural_operator'`` when selecting an operator kind explicitly, for
example ``operator_kind='fno'``.
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
    _numeric_transition_law_rows,
    _numeric_transition_state_targets,
)
from treepo.local_law import LawKind
from treepo.objective import LOCAL_LAW_ESTIMATOR_ORACLE_STATE, ObjectiveSpec


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
        self._objective: ObjectiveSpec | None = None
        self._objective_law_shares: dict[str, float] = {}
        self._objective_gamma_depth: float = 1.0
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
        del g
        self._maybe_warmstart(f_init)
        return self._train_side(kind="f", traces=traces, iteration=iteration, output_dir=output_dir)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del f
        self._maybe_warmstart(g_init)
        return self._train_side(kind="g", traces=traces, iteration=iteration, output_dir=output_dir)

    def _maybe_warmstart(self, artifact: Any) -> None:
        """Load model weights from a prior artifact into a fresh family.

        Artifacts returned by ``_train_side`` carry a ``weights_path``; handing
        one back (``initial_artifacts`` in a spec, or ``f=``/``g=`` at scoring
        time) must reconstruct the trained model, not be silently ignored.
        A live in-run model always wins: within one alternating run the state
        already reflects every completed iteration.
        """

        if self._model is not None or not isinstance(artifact, Mapping):
            return
        path = artifact.get("weights_path")
        if not path:
            return
        weights_file = Path(str(path))
        if not weights_file.exists():
            raise ValueError(
                f"family={self.name!r} warmstart artifact points at missing "
                f"weights_path {str(weights_file)!r}"
            )
        artifact_operator = artifact.get("operator_kind")
        if artifact_operator is not None and str(artifact_operator) != self.operator_kind:
            raise ValueError(
                f"family={self.name!r} warmstart operator_kind mismatch: artifact "
                f"has {artifact_operator!r}, family is {self.operator_kind!r}"
            )
        self._ensure_model(output_dim=int(artifact.get("output_dim") or 1))
        assert self._model is not None
        # Artifacts persist only tensor entries (see _train_side), so the safe
        # weights-only loader applies; non-tensor module extras are rebuilt by
        # construction, hence strict=False.
        state = self._torch.load(weights_file, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state, strict=False)
        for attr, key in (("_target_center", "target_center"), ("_target_scale", "target_scale")):
            payload = artifact.get(key)
            if payload is not None:
                setattr(
                    self,
                    attr,
                    self._torch.tensor(payload, dtype=self._torch.float32, device=self._device),
                )

    def configure_objective(self, objective: ObjectiveSpec | None) -> None:
        """Adopt a resolved ``ObjectiveSpec`` as the executed training objective.

        With a law-bearing spec, every training step minimizes the convex
        combination ``root_share * root_mse + sum_c share_c * law_c``, where
        each law channel is the depth-discounted canonical objective from
        :mod:`treepo.training.local_law` over exact numeric transition-state
        rows. The supervision targets are exact, so the spec must declare
        ``local_law_estimator='oracle_state'``; the canonical corrected
        estimator then degenerates to the oracle term (observed, propensity 1).
        Mutually exclusive with the legacy additive
        ``numeric_transition_state_weight`` knob: a configured objective must
        fully describe the executed loss.
        """
        if objective is None:
            self._objective = None
            self._objective_law_shares = {}
            self._objective_gamma_depth = 1.0
            return
        if not isinstance(objective, ObjectiveSpec):
            raise TypeError(
                f"family={self.name!r} objective must be an ObjectiveSpec; "
                f"got {type(objective).__name__}"
            )
        if float(self.config.numeric_transition_state_weight) > 0.0:
            raise ValueError(
                f"family={self.name!r} got both an objective spec and "
                "numeric_transition_state_weight > 0; the objective is the "
                "single weight source — drop the legacy knob"
            )
        shares: dict[str, float] = {}
        gamma = 1.0
        law_weight = float(objective.local_law_weight or 0.0)
        if law_weight > 0.0:
            if str(objective.local_law_estimator) != LOCAL_LAW_ESTIMATOR_ORACLE_STATE:
                raise ValueError(
                    f"family={self.name!r} trains laws against exact numeric "
                    "transition states; declare local_law_estimator="
                    f"'oracle_state' (got {objective.local_law_estimator!r})"
                )
            weights = dict(objective.local_law_component_weights or {})
            if float(weights.get(LawKind.C2_IDEMPOTENCE.value, 0.0)) > 0.0:
                raise ValueError(
                    f"family={self.name!r} has no C2 (on-range idempotence) law "
                    "surface; set its component weight to 0"
                )
            total = float(sum(float(v) for v in weights.values()))
            shares = {
                str(name): law_weight * float(value) / total
                for name, value in weights.items()
                if float(value) > 0.0
            }
            gamma = float(objective.gamma_depth)
        self._objective = objective
        self._objective_law_shares = shares
        self._objective_gamma_depth = gamma

    def _train_side(
        self,
        *,
        kind: str,
        traces: Sequence[Any],
        iteration: int,
        output_dir: Path | None = None,
    ) -> Mapping[str, Any]:
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
            trees=trees if (train_g or self._objective_law_shares) else None,
        )
        weights_path: Path | None = None
        if output_dir is not None and self._model is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            weights_path = out / f"{kind}_weights_iter{int(iteration):02d}.pt"
            # Persist tensors only (parameters/buffers): keeps the file loadable
            # under torch's safe weights-only unpickler.
            tensor_state = {
                key: value
                for key, value in self._model.state_dict().items()
                if self._torch.is_tensor(value)
            }
            self._torch.save(tensor_state, weights_path)
        artifact = self._artifact_payload(
            kind=kind,
            iteration=iteration,
            n_train=int(x.shape[0]),
            loss=last_loss,
            weights_path=weights_path,
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
        # f and g artifacts each snapshot the whole shared model at their own
        # training iteration; scoring must resume from the newest snapshot.
        candidates = [
            artifact
            for artifact in (f, g)
            if isinstance(artifact, Mapping) and artifact.get("weights_path")
        ]
        if candidates:
            self._maybe_warmstart(
                max(candidates, key=lambda artifact: int(artifact.get("iteration") or 0))
            )
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
        if isinstance(artifact, Mapping):
            path = artifact.get("weights_path")
            if path and not Path(str(path)).exists():
                raise ValueError(
                    f"family={self.name!r} {kind} artifact weights_path "
                    f"{str(path)!r} does not exist"
                )

    def as_statistic(self, *, f: Any = None, g: Any = None) -> Any:
        del f, g
        if self._model is None:
            return None
        return _NeuralOperatorStatistic(self)

    def _artifact_kind_for_operator(self) -> str:
        if self.operator_kind == "fno":
            return "treepo_fno"
        return self.artifact_kind

    def _artifact_payload(
        self,
        *,
        kind: str,
        iteration: int,
        n_train: int,
        loss: float | None,
        weights_path: Path | None = None,
    ) -> dict[str, Any]:
        base_kind = self._artifact_kind_for_operator()
        return {
            "kind": base_kind if kind == "f" else f"{base_kind}_g",
            "trained": kind,
            "weights_path": str(weights_path) if weights_path is not None else None,
            "operator_kind": self.operator_kind,
            "iteration": int(iteration),
            "n_train": int(n_train),
            "loss": loss,
            "normalize_targets": bool(self.config.normalize_targets),
            "target_center": _tensor_payload(self._target_center),
            "target_scale": _tensor_payload(self._target_scale),
            "numeric_transition_state_weight": float(self.config.numeric_transition_state_weight),
            "objective_executed": self._objective is not None,
            "objective": self._objective.to_dict() if self._objective is not None else None,
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
        # The law signal enters training in one of two mutually exclusive ways:
        # a configured ObjectiveSpec drives the convex assembly on every step,
        # while the legacy additive knob applies to g-steps only.
        law_in_loss = bool(self._objective_law_shares) or (
            self._objective is None
            and train_g
            and float(self.config.numeric_transition_state_weight) > 0.0
        )
        state_targets = (
            _numeric_transition_state_targets(trees, self.config, torch=self._torch, device=self._device)
            if law_in_loss and trees is not None
            else None
        )
        if self._objective_law_shares and state_targets is None:
            raise ValueError(
                f"family={self.name!r} objective declares a local-law weight but "
                "the training trees carry no numeric transition supervision "
                "(tree metadata needs n_states and vocabulary_size)"
            )
        for _epoch in range(epochs):
            order = self._torch.randperm(n, device=self._device)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if state_targets is None:
                    pred = self._model(x[idx], lengths[idx])
                    law_rows = None
                else:
                    pred, traces = self._model.forward_with_trace(x[idx], lengths[idx])
                    selected = [state_targets[int(i)] for i in idx.detach().cpu().tolist()]
                    law_rows = _numeric_transition_law_rows(
                        traces,
                        selected,
                        torch=self._torch,
                        device=self._device,
                        dtype=pred.dtype,
                    )
                root_loss = self._torch.nn.functional.mse_loss(pred, y[idx])
                loss = self._assemble_loss(root_loss, law_rows)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().cpu())
        self._set_trainable(train_f=True, train_g=True)
        return last_loss

    def _assemble_loss(self, root_loss: Any, law_rows: Any | None) -> Any:
        """Combine root and law terms under the executed objective.

        ObjectiveSpec path: ``root_share * root + sum_c share_c * law_c`` with
        each channel routed through the canonical depth-discounted objective.
        Legacy path: ``root + numeric_transition_state_weight * node_mean`` —
        the law term is the unweighted node mean, the sampling contract's
        audit estimand.
        """
        # Imported lazily to keep `import treepo.methods.fno` free of torch;
        # this module only touches torch after family construction.
        from treepo.training.local_law import local_law_objective_from_losses

        torch = self._torch
        if self._objective is None:
            if law_rows is None:
                return root_loss
            proxy, _depths, _is_leaf = law_rows
            return root_loss + float(self.config.numeric_transition_state_weight) * proxy.mean()
        loss = float(self._objective.root_share) * root_loss
        if not self._objective_law_shares:
            return loss
        if law_rows is None:
            raise ValueError(
                f"family={self.name!r} objective declares a local-law weight but "
                "the training batch produced no law rows"
            )
        proxy, depths, is_leaf = law_rows
        channel_masks = {
            LawKind.C1_LEAF.value: is_leaf,
            LawKind.C3_MERGE.value: ~is_leaf,
        }
        for name, share in self._objective_law_shares.items():
            mask = channel_masks[name]
            if not bool(mask.any()):
                raise ValueError(
                    f"family={self.name!r} objective weights law channel {name!r} "
                    "but the training batch has no such nodes; set its component "
                    "weight to 0"
                )
            masked = proxy[mask]
            # Exact-state supervision: the corrected estimator degenerates to
            # the oracle term (observed rows, propensity 1, oracle == proxy).
            channel = local_law_objective_from_losses(
                proxy_loss=masked,
                oracle_loss=masked,
                observed=torch.ones(masked.shape[0], dtype=torch.bool, device=masked.device),
                propensity=torch.ones(masked.shape[0], dtype=masked.dtype, device=masked.device),
                depths=depths[mask],
                gamma_depth=float(self._objective_gamma_depth),
            )
            loss = loss + float(share) * channel
        return loss

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
    config = _coerce_config(
        payload.get("neural_operator_config"), payload, config_cls=NeuralOperatorFamilyConfig
    )
    embedding_client = payload.get("embedding_client")
    if embedding_client is not None and not hasattr(embedding_client, "embed_texts"):
        raise TypeError("backend_config['embedding_client'] must provide embed_texts(texts)")
    return NeuralOperatorFamily(config, embedding_client=embedding_client)


def build_fno_family(backend_config: Mapping[str, Any]) -> FNOFamily:
    """Build the concrete FNO family from method ``backend_config``."""

    payload = dict(backend_config or {})
    config = _coerce_config(payload.get("fno_config"), payload, config_cls=FNOFamilyConfig)
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
