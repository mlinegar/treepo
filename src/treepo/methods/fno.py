"""Built-in neural-operator families for :mod:`treepo.methods`.

``neural_operator`` is the generic family. It delegates named operator kinds
to ``neuralop.models`` when available and keeps ``operator_kind='conv1d'`` as
a tiny local baseline. ``fno`` is the short convenience family name for the
FNO path. Dataset-specific structure belongs in tree fixtures or registered
downstream families; this module only owns the generic neural-operator method
surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from treepo.llm.embedding import EmbeddingClient, HashingEmbeddingClient

_LOCAL_OPERATOR_KINDS = frozenset({"conv1d"})
_NEURALOP_KIND_ALIASES = {
    "codano": "CODANO",
    "fno": "FNO",
    "fno_gno": "FNOGNO",
    "fnogno": "FNOGNO",
    "gino": "GINO",
    "tfno": "TFNO",
    "uno": "UNO",
    "uqno": "UQNO",
}
_SEQUENCE_INCOMPATIBLE_NEURALOP_KINDS = frozenset({"codano", "fno_gno", "fnogno", "gino"})
_SEQUENCE_COMPATIBLE_NEURALOP_KINDS = frozenset({"fno", "tfno", "uno"})


@dataclass
class NeuralOperatorFamilyConfig:
    """Config for the generic neural-operator root-score family."""

    operator_kind: str = "fno"
    embedding_dim: int = 32
    hidden_channels: int = 16
    n_modes: int = 8
    n_layers: int = 2
    conv_kernel_size: int = 3
    head_hidden_dim: int = 32
    operator_kwargs: Mapping[str, Any] = field(default_factory=dict)
    learning_rate: float = 1e-3
    epochs_per_iteration: int = 8
    batch_size: int = 8
    seed: int = 0
    device: str = "cpu"
    normalize_targets: bool = True
    target_key: str | None = None
    target_vector_key: str | None = None
    target_keys: Sequence[str] | None = None
    target_dim: int | None = None
    target_min: float | None = None
    target_max: float | None = None
    embedding_salt: str = "treepo_neural_operator"
    use_numeric_leaf_features: bool = True
    numeric_transition_state_weight: float = 0.0
    numeric_transition_count_scale: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class FNOFamilyConfig(NeuralOperatorFamilyConfig):
    """Config for the short ``family='fno'`` route."""

    operator_kind: str = "fno"
    embedding_salt: str = "treepo_fno"


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
        self._target_mean = 0.0
        self._target_center = None
        self._target_scale = None
        self._last_artifact: dict[str, Any] | None = None
        self._encoding_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del f_init, g
        trees, targets = _target_rows(traces, self.config)
        x, lengths = self._encode_trees(trees)
        y = self._torch.tensor(targets, dtype=self._torch.float32, device=self._device)
        self._ensure_model(output_dim=int(y.shape[1]))
        y_train = self._normalized_targets(y)
        last_loss = self._train_supervised(
            x,
            lengths,
            y_train,
            train_f=True,
            train_g=False,
        )
        artifact = self._artifact_payload(
            kind="f",
            iteration=iteration,
            n_train=int(x.shape[0]),
            loss=last_loss,
        )
        self._last_artifact = artifact
        return artifact

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del g_init, f
        trees, targets = _target_rows(traces, self.config)
        x, lengths = self._encode_trees(trees)
        y = self._torch.tensor(targets, dtype=self._torch.float32, device=self._device)
        self._ensure_model(output_dim=int(y.shape[1]))
        y_train = self._normalized_targets(y)
        last_loss = self._train_supervised(
            x,
            lengths,
            y_train,
            train_f=False,
            train_g=True,
            trees=trees,
        )
        artifact = self._artifact_payload(
            kind="g",
            iteration=iteration,
            n_train=int(x.shape[0]),
            loss=last_loss,
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
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
            "target_keys": list(self.config.target_keys or ()),
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
            self._target_mean = float(y.mean().detach().cpu())
            self._target_center = self._torch.zeros((int(y.shape[1]),), dtype=y.dtype, device=y.device)
            self._target_scale = self._torch.ones((int(y.shape[1]),), dtype=y.dtype, device=y.device)
            return y
        center = y.mean(dim=0)
        scale = y.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        self._target_mean = float(y.mean().detach().cpu())
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
            return cached
        if bool(self.config.use_numeric_leaf_features):
            numeric_groups = [_leaf_token_groups(tree) for tree in trees]
            if numeric_groups and all(group is not None for group in numeric_groups):
                encoded = _encode_numeric_leaf_features(
                    numeric_groups,
                    dim=int(self.config.embedding_dim),
                    torch=self._torch,
                    device=self._device,
                )
                self._encoding_cache[cache_key] = encoded
                return encoded
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
        self._encoding_cache[cache_key] = encoded
        return encoded


class FNOFamily(NeuralOperatorFamily):
    """Short ``family='fno'`` route for ``operator_kind='fno'``."""

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
    """Build the short FNO family from method ``backend_config``."""

    payload = dict(backend_config or {})
    requested = _normalize_operator_kind(payload.get("operator_kind", "fno"))
    if requested != "fno":
        raise ValueError(
            "family='fno' only supports operator_kind='fno'; use "
            "family='neural_operator' for other operator kinds."
        )
    payload["operator_kind"] = "fno"
    config = _coerce_config(payload.get("fno_config"), payload, config_cls=FNOFamilyConfig)
    embedding_client = payload.get("embedding_client")
    if embedding_client is not None and not hasattr(embedding_client, "embed_texts"):
        raise TypeError("backend_config['embedding_client'] must provide embed_texts(texts)")
    return FNOFamily(config, embedding_client=embedding_client)


_ConfigT = TypeVar("_ConfigT", bound=NeuralOperatorFamilyConfig)


def _coerce_config(
    raw: Any,
    backend_config: Mapping[str, Any],
    *,
    config_cls: type[_ConfigT],
) -> _ConfigT:
    if isinstance(raw, config_cls):
        base = raw
    elif isinstance(raw, NeuralOperatorFamilyConfig):
        base = config_cls(**_known_config_keys(_config_payload(raw), config_cls=config_cls))
    elif isinstance(raw, Mapping):
        base = config_cls(**_known_config_keys(raw, config_cls=config_cls))
    elif is_dataclass(raw):
        base = config_cls(**_known_config_keys(getattr(raw, "__dict__", {}), config_cls=config_cls))
    elif raw is None:
        base = config_cls(**_known_config_keys(backend_config, config_cls=config_cls))
    else:
        raise TypeError(
            "backend_config config must be NeuralOperatorFamilyConfig/FNOFamilyConfig "
            f"or mapping; got {type(raw).__name__}"
        )
    overrides = _known_config_keys(backend_config, config_cls=config_cls)
    if not overrides:
        return base
    data = _config_payload(base)
    data.update(overrides)
    return config_cls(**_known_config_keys(data, config_cls=config_cls))


def _known_config_keys(values: Mapping[str, Any], *, config_cls: type[Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(config_cls)}
    return {str(k): v for k, v in dict(values or {}).items() if str(k) in allowed}


def _config_payload(config: NeuralOperatorFamilyConfig) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def _tensor_payload(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
    except Exception:
        return None


def _normalize_operator_kind(value: Any) -> str:
    return str(value or "fno").strip().lower().replace("-", "_")


def _validate_operator_kind(operator_kind: str, *, family_name: str) -> None:
    if operator_kind in _LOCAL_OPERATOR_KINDS:
        return
    if _neuralop_model_class(operator_kind, required=False) is not None:
        return
    supported = ", ".join(sorted((*_LOCAL_OPERATOR_KINDS, *_available_neuralop_kinds())))
    raise ValueError(
        f"family={family_name!r} does not support operator_kind={operator_kind!r}; "
        f"supported operator_kind values: {supported}"
    )



class _TreeFGModel:
    def __new__(
        cls,
        *,
        operator_kind: str,
        config: NeuralOperatorFamilyConfig,
        torch: Any,
        output_dim: int,
    ) -> Any:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden = max(1, int(config.hidden_channels))
                self.leaf_operator = _build_leaf_operator(
                    operator_kind=operator_kind,
                    config=config,
                    torch=torch,
                )
                self.merge = torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden, max(1, int(config.head_hidden_dim))),
                    torch.nn.GELU(),
                    torch.nn.Linear(max(1, int(config.head_hidden_dim)), hidden),
                )
                self.readout = torch.nn.Sequential(
                    torch.nn.Linear(hidden, max(1, int(config.head_hidden_dim))),
                    torch.nn.GELU(),
                    torch.nn.Linear(max(1, int(config.head_hidden_dim)), max(1, int(output_dim))),
                )

            def forward(self, x: Any, lengths: Any) -> Any:
                return self.forward_with_trace(x, lengths)[0]

            def forward_with_trace(self, x: Any, lengths: Any) -> tuple[Any, list[Any]]:
                leaf_states = self.leaf_operator(x)
                if int(leaf_states.shape[0]) > 0 and bool(torch.all(lengths == lengths[0]).detach().cpu().item()):
                    length = max(1, int(lengths[0].detach().cpu().item()))
                    roots, trace = self._compose_batch_with_trace(leaf_states[:, :length, :])
                    return self.readout(roots), [trace[idx] for idx in range(int(trace.shape[0]))]
                roots = []
                traces = []
                raw_lengths = lengths.detach().cpu().tolist()
                for idx, raw_length in enumerate(raw_lengths):
                    length = max(1, int(raw_length))
                    root, trace = self._compose_with_trace(leaf_states[idx, :length, :])
                    roots.append(root)
                    traces.append(trace)
                return self.readout(torch.stack(roots, dim=0)), traces

            def _compose(self, states: Any) -> Any:
                return self._compose_with_trace(states)[0]

            def _compose_batch_with_trace(self, states: Any) -> tuple[Any, Any]:
                trace_parts = [states]
                while int(states.shape[1]) > 1:
                    n_states = int(states.shape[1])
                    pair_count = n_states // 2
                    left = states[:, 0 : pair_count * 2 : 2, :]
                    right = states[:, 1 : pair_count * 2 : 2, :]
                    pair_inputs = torch.cat([left, right], dim=-1)
                    merged = self.merge(pair_inputs.reshape(-1, int(pair_inputs.shape[-1]))).reshape(
                        int(states.shape[0]),
                        pair_count,
                        -1,
                    )
                    trace_parts.append(merged)
                    if n_states % 2:
                        states = torch.cat([merged, states[:, -1:, :].clone()], dim=1)
                    else:
                        states = merged
                return states[:, 0, :], torch.cat(trace_parts, dim=1)

            def _compose_with_trace(self, states: Any) -> tuple[Any, Any]:
                trace_parts = [states]
                while int(states.shape[0]) > 1:
                    n_states = int(states.shape[0])
                    pair_count = n_states // 2
                    left = states[0 : pair_count * 2 : 2]
                    right = states[1 : pair_count * 2 : 2]
                    merged = self.merge(torch.cat([left, right], dim=-1))
                    trace_parts.append(merged)
                    if n_states % 2:
                        states = torch.cat([merged, states[-1:].clone()], dim=0)
                    else:
                        states = merged
                return states.squeeze(0), torch.cat(trace_parts, dim=0)

        return _Model()


def _build_leaf_operator(*, operator_kind: str, config: NeuralOperatorFamilyConfig, torch: Any) -> Any:
    if operator_kind == "conv1d":
        return _LeafConv1D(config=config, torch=torch)
    return _LeafNeuralOp(
        operator_kind=operator_kind,
        config=config,
        torch=torch,
        model_cls=_neuralop_model_class(operator_kind, required=True),
    )


class _LeafNeuralOp:
    def __new__(
        cls,
        *,
        operator_kind: str,
        config: NeuralOperatorFamilyConfig,
        torch: Any,
        model_cls: Any,
    ) -> Any:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.operator = model_cls(
                    **_neuralop_constructor_kwargs(
                        operator_kind=operator_kind,
                        config=config,
                        model_cls=model_cls,
                    )
                )

            def forward(self, x: Any) -> Any:
                y = self.operator(x.transpose(1, 2))
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if y.ndim == 3:
                    return y.transpose(1, 2)
                if y.ndim == 2:
                    return y.unsqueeze(1)
                if y.ndim > 3:
                    y = y.reshape(y.shape[0], y.shape[1], -1)
                    return y.transpose(1, 2)
                return y.reshape(y.shape[0], 1, -1)

        return _Model()


class _LeafConv1D:
    def __new__(cls, *, config: NeuralOperatorFamilyConfig, torch: Any) -> Any:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                in_channels = max(1, int(config.embedding_dim))
                hidden = max(1, int(config.hidden_channels))
                kernel = _odd_kernel_size(config.conv_kernel_size)
                layers: list[Any] = []
                for layer_idx in range(max(1, int(config.n_layers))):
                    left = in_channels if layer_idx == 0 else hidden
                    layers.append(
                        torch.nn.Conv1d(
                            left,
                            hidden,
                            kernel_size=kernel,
                            padding=kernel // 2,
                        )
                    )
                    layers.append(torch.nn.GELU())
                self.operator = torch.nn.Sequential(*layers)

            def forward(self, x: Any) -> Any:
                return self.operator(x.transpose(1, 2)).transpose(1, 2)

        return _Model()


def _build_root_model(*, operator_kind: str, config: NeuralOperatorFamilyConfig, torch: Any) -> Any:
    if operator_kind == "conv1d":
        return _RootConv1D(config=config, torch=torch)
    return _RootNeuralOp(
        operator_kind=operator_kind,
        config=config,
        torch=torch,
        model_cls=_neuralop_model_class(operator_kind, required=True),
    )


class _RootNeuralOp:
    def __new__(
        cls,
        *,
        operator_kind: str,
        config: NeuralOperatorFamilyConfig,
        torch: Any,
        model_cls: Any,
    ) -> Any:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.operator = model_cls(
                    **_neuralop_constructor_kwargs(
                        operator_kind=operator_kind,
                        config=config,
                        model_cls=model_cls,
                    )
                )
                self.head = _root_head(config=config, torch=torch)

            def forward(self, x: Any) -> Any:
                y = self.operator(x.transpose(1, 2))
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if y.ndim >= 3:
                    dims = tuple(range(2, int(y.ndim)))
                    z = y.mean(dim=dims)
                elif y.ndim == 2:
                    z = y
                else:
                    z = y.reshape(y.shape[0], -1)
                return self.head(z).squeeze(-1)

        return _Model()


class _RootConv1D:
    def __new__(cls, *, config: NeuralOperatorFamilyConfig, torch: Any) -> Any:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                in_channels = max(1, int(config.embedding_dim))
                hidden = max(1, int(config.hidden_channels))
                kernel = _odd_kernel_size(config.conv_kernel_size)
                layers: list[Any] = []
                for layer_idx in range(max(1, int(config.n_layers))):
                    left = in_channels if layer_idx == 0 else hidden
                    layers.append(
                        torch.nn.Conv1d(
                            left,
                            hidden,
                            kernel_size=kernel,
                            padding=kernel // 2,
                        )
                    )
                    layers.append(torch.nn.GELU())
                self.operator = torch.nn.Sequential(*layers)
                self.head = _root_head(config=config, torch=torch)

            def forward(self, x: Any) -> Any:
                z = self.operator(x.transpose(1, 2)).mean(dim=-1)
                return self.head(z).squeeze(-1)

        return _Model()


def _root_head(*, config: NeuralOperatorFamilyConfig, torch: Any) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(max(1, int(config.hidden_channels)), max(1, int(config.head_hidden_dim))),
        torch.nn.GELU(),
        torch.nn.Linear(max(1, int(config.head_hidden_dim)), 1),
    )


def _odd_kernel_size(value: Any) -> int:
    kernel = max(1, int(value))
    return kernel if kernel % 2 == 1 else kernel + 1


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when deps absent
        raise ImportError(
            "family='neural_operator' requires PyTorch. Install the default "
            "treepo package dependencies or add torch to this environment."
        ) from exc
    return torch


def _available_neuralop_kinds() -> tuple[str, ...]:
    try:
        import neuralop.models as models
    except ImportError:
        return tuple()
    names: list[str] = []
    for name in dir(models):
        if name.startswith("_"):
            continue
        obj = getattr(models, name)
        if isinstance(obj, type) and str(getattr(obj, "__module__", "")).startswith("neuralop"):
            names.append(_normalize_operator_kind(name))
    return tuple(sorted(set(names)))


def _neuralop_model_class(operator_kind: str, *, required: bool) -> Any:
    normalized = _normalize_operator_kind(operator_kind)
    try:
        import neuralop.models as models
    except ImportError as exc:  # pragma: no cover - exercised when deps absent
        if not required:
            return None
        raise ImportError(
            f"operator_kind={operator_kind!r} requires neuraloperator. Install the default "
            "treepo package dependencies or add neuraloperator to this environment."
        ) from exc
    exact_name = _NEURALOP_KIND_ALIASES.get(normalized)
    if exact_name is not None and hasattr(models, exact_name):
        return getattr(models, exact_name)
    for name in dir(models):
        if _normalize_operator_kind(name) == normalized:
            obj = getattr(models, name)
            if isinstance(obj, type):
                return obj
    if required:
        supported = ", ".join(sorted((*_LOCAL_OPERATOR_KINDS, *_available_neuralop_kinds())))
        raise ValueError(
            f"operator_kind={operator_kind!r} is not available from neuralop.models; "
            f"supported operator_kind values: {supported}"
        )
    return None


def _neuralop_constructor_kwargs(
    *,
    operator_kind: str,
    config: NeuralOperatorFamilyConfig,
    model_cls: Any,
) -> dict[str, Any]:
    import inspect

    raw_kwargs = dict(config.operator_kwargs or {})
    signature = inspect.signature(model_cls)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    normalized = _normalize_operator_kind(operator_kind)
    if normalized in _SEQUENCE_INCOMPATIBLE_NEURALOP_KINDS:
        supported = ", ".join(sorted((*_LOCAL_OPERATOR_KINDS, *_SEQUENCE_COMPATIBLE_NEURALOP_KINDS)))
        raise ValueError(
            f"operator_kind={operator_kind!r} is available from neuralop.models, "
            "but treepo's built-in neural_operator family accepts one embedded "
            f"leaf-sequence tensor. Use one of {supported}, or register a "
            "downstream family for geometry/query neural operators."
        )
    in_channels = max(1, int(config.embedding_dim))
    hidden_channels = max(1, int(config.hidden_channels))
    n_layers = max(1, int(config.n_layers))
    n_modes = (max(1, int(config.n_modes)),)
    dense_defaults = {
        "in_channels": in_channels,
        "out_channels": hidden_channels,
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
        "n_modes": n_modes,
    }
    extended_defaults = {
        **dense_defaults,
        "fno_in_channels": in_channels,
        "fno_hidden_channels": hidden_channels,
        "fno_n_layers": n_layers,
        "fno_n_modes": n_modes,
    }
    if normalized == "uno":
        extended_defaults.update(
            {
                "lifting_channels": hidden_channels,
                "projection_channels": hidden_channels,
                "uno_out_channels": [hidden_channels] * n_layers,
                "uno_n_modes": [n_modes] * n_layers,
                "uno_scalings": [[1.0] * len(n_modes)] * n_layers,
                "horizontal_skips_map": {},
            }
        )
    kwargs = dict(dense_defaults) if accepts_kwargs else {
        key: value for key, value in extended_defaults.items() if key in signature.parameters
    }
    kwargs.update(raw_kwargs)
    required_missing = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if required_missing:
        missing = ", ".join(required_missing)
        raise ValueError(
            f"operator_kind={operator_kind!r} needs operator_kwargs for required "
            f"constructor argument(s): {missing}"
        )
    return kwargs




def _leaf_token_groups(tree: Any) -> list[Any] | None:
    leaves = getattr(tree, "leaves", None)
    if not leaves and callable(getattr(tree, "get_leaves", None)):
        leaves = tree.get_leaves()
    raw_groups = leaves if leaves else (tree,)
    groups: list[Any] = []
    for leaf in raw_groups:
        tokens = getattr(leaf, "tokens", None)
        if tokens is None and isinstance(leaf, Mapping):
            tokens = leaf.get("tokens")
        if tokens is None:
            return None
        try:
            group = tokens if hasattr(tokens, "dtype") else [int(token) for token in tokens]
        except (TypeError, ValueError):
            return None
        groups.append(group)
    return groups or [[]]


def _encode_numeric_leaf_features(
    groups: Sequence[Sequence[Any] | None],
    *,
    dim: int,
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    import numpy as np

    dim = max(1, int(dim))
    materialized = []
    for group in groups:
        if group is None or len(group) == 0:
            materialized.append([()])
        else:
            materialized.append(list(group))
    lengths = [max(1, len(group)) for group in materialized]
    max_leaves = max(1, max(lengths))
    features = np.zeros((len(materialized) * max_leaves, dim), dtype=np.float32)
    for tree_idx, group in enumerate(materialized):
        for leaf_idx, tokens in enumerate(group[:max_leaves]):
            if len(tokens) == 0:
                continue
            row = tree_idx * max_leaves + leaf_idx
            _add_numeric_sequence_features(features[row], tokens, dim=dim, np=np)
    x_flat = torch.as_tensor(features, dtype=torch.float32, device=device)
    denom = torch.clamp(x_flat.sum(dim=-1, keepdim=True), min=1.0)
    x_flat = x_flat / torch.sqrt(denom)
    x = x_flat.reshape(len(materialized), max_leaves, dim)
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    return x, length_tensor


def _numeric_transition_state_targets(
    trees: Sequence[Any] | None,
    config: NeuralOperatorFamilyConfig,
    *,
    torch: Any,
    device: Any,
) -> list[Any] | None:
    if trees is None:
        return None
    out = []
    for tree in trees:
        spec = _numeric_transition_spec(tree, config)
        groups = _leaf_token_groups(tree)
        if spec is None or groups is None:
            return None
        n_states, bucket, count_scale = spec
        rows = _numeric_transition_rows(
            groups,
            n_states=n_states,
            bucket=bucket,
            count_scale=count_scale,
        )
        out.append(torch.tensor(rows, dtype=torch.float32, device=device))
    return out


def _numeric_transition_state_loss(
    traces: Sequence[Any],
    targets: Sequence[Any],
    *,
    torch: Any,
    device: Any,
    dtype: Any,
) -> Any | None:
    if traces and targets and len(traces) == len(targets):
        first_trace_shape = tuple(int(x) for x in traces[0].shape)
        first_target_shape = tuple(int(x) for x in targets[0].shape)
        if (
            first_trace_shape[0] == first_target_shape[0]
            and all(tuple(int(x) for x in trace.shape) == first_trace_shape for trace in traces)
            and all(tuple(int(x) for x in target.shape) == first_target_shape for target in targets)
        ):
            pred = torch.stack(list(traces), dim=0)
            target = torch.stack(
                [target.to(device=device, dtype=dtype) for target in targets],
                dim=0,
            )
            d = min(int(pred.shape[-1]), int(target.shape[-1]))
            if d > 0:
                return torch.nn.functional.mse_loss(pred[:, :, :d], target[:, :, :d])
    losses = []
    for pred, target in zip(traces, targets):
        target = target.to(device=device, dtype=dtype)
        n = min(int(pred.shape[0]), int(target.shape[0]))
        d = min(int(pred.shape[1]), int(target.shape[1]))
        if n <= 0 or d <= 0:
            continue
        losses.append(torch.nn.functional.mse_loss(pred[:n, :d], target[:n, :d]))
    if not losses:
        return None
    return torch.stack(losses).mean()


def _numeric_transition_spec(
    tree: Any,
    config: NeuralOperatorFamilyConfig,
) -> tuple[int, int, float] | None:
    meta = getattr(tree, "metadata", None) or {}
    if not isinstance(meta, Mapping):
        return None
    n_states = _optional_positive_int(meta.get("n_states"))
    vocabulary_size = _optional_positive_int(meta.get("vocabulary_size"))
    if n_states is None or vocabulary_size is None:
        return None
    bucket = max(1, int(vocabulary_size) // int(n_states))
    raw_scale = config.numeric_transition_count_scale
    if raw_scale is None:
        raw_scale = meta.get("doc_tokens") or 1.0
    count_scale = max(1.0, float(raw_scale))
    return int(n_states), int(bucket), float(count_scale)


def _numeric_transition_rows(
    groups: Sequence[Sequence[int]],
    *,
    n_states: int,
    bucket: int,
    count_scale: float,
) -> list[list[float]]:
    cur = [_numeric_transition_leaf_state(tokens, n_states=n_states, bucket=bucket) for tokens in groups]
    rows = [_numeric_transition_vector(state, n_states=n_states, count_scale=count_scale) for state in cur]
    while len(cur) > 1:
        next_level = []
        for idx in range(0, len(cur) - 1, 2):
            merged = _numeric_transition_merge_state(cur[idx], cur[idx + 1])
            next_level.append(merged)
            rows.append(_numeric_transition_vector(merged, n_states=n_states, count_scale=count_scale))
        if len(cur) % 2:
            next_level.append(cur[-1])
        cur = next_level
    return rows


def _numeric_transition_leaf_state(
    tokens: Sequence[int],
    *,
    n_states: int,
    bucket: int,
) -> tuple[float, int, int]:
    states = [
        min(max(0, int(token) // max(1, int(bucket))), int(n_states) - 1)
        for token in tokens
    ]
    if not states:
        return 0.0, 0, 0
    count = sum(1 for left, right in zip(states, states[1:]) if int(left) != int(right))
    return float(count), int(states[0]), int(states[-1])


def _numeric_transition_merge_state(
    left: tuple[float, int, int],
    right: tuple[float, int, int],
) -> tuple[float, int, int]:
    left_count, left_first, left_last = left
    right_count, right_first, right_last = right
    join = 1.0 if int(left_last) != int(right_first) else 0.0
    return float(left_count + right_count + join), int(left_first), int(right_last)


def _numeric_transition_vector(
    state: tuple[float, int, int],
    *,
    n_states: int,
    count_scale: float,
) -> list[float]:
    count, first, last = state
    vec = [float(count) / max(1.0, float(count_scale))]
    first_oh = [0.0] * int(n_states)
    last_oh = [0.0] * int(n_states)
    if 0 <= int(first) < int(n_states):
        first_oh[int(first)] = 1.0
    if 0 <= int(last) < int(n_states):
        last_oh[int(last)] = 1.0
    return vec + first_oh + last_oh


def _optional_positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _tree_sequence_cache_key(trees: Sequence[Any], *, dim: int, device: str) -> tuple[Any, ...]:
    if not trees:
        return ("empty", int(dim), str(device))
    ids = tuple(id(tree) for tree in trees)
    return (len(ids), ids[0], ids[-1], hash(ids), int(dim), str(device))


def _add_numeric_sequence_features(row: Any, tokens: Sequence[int], *, dim: int, np: Any) -> None:
    token_arr = np.asarray(tokens, dtype=np.uint64)
    if token_arr.size == 0:
        return
    n = int(token_arr.size)
    _add_reserved_numeric_feature(row, 0, float(np.sqrt(max(1, n))))
    _add_hashed_counts(row, _hash_numeric_array(token_arr, dim=dim, salt=0x9E3779B185EBCA87, np=np), weight=1.0, np=np)
    for shift, weight in ((4, 0.5), (8, 0.75), (12, 0.35)):
        coarse = np.right_shift(token_arr, np.uint64(shift))
        _add_hashed_counts(
            row,
            _hash_numeric_array(coarse, dim=dim, salt=0xC2B2AE3D27D4EB4F + shift, np=np),
            weight=weight,
            np=np,
        )
    length_bucket = _hash_numeric_value(n, dim=dim, salt=0xA24BAED4963EE407)
    row[length_bucket] += float(np.sqrt(max(1, n)))
    for slot_idx, shift in enumerate((4, 8, 12), start=1):
        coarse = np.right_shift(token_arr, np.uint64(shift))
        first_bucket = _hash_numeric_value(int(coarse[0]), dim=dim, salt=0x165667B19E3779F9 + shift)
        last_bucket = _hash_numeric_value(int(coarse[-1]), dim=dim, salt=0x85EBCA77C2B2AE63 + shift)
        row[first_bucket] += 2.0
        row[last_bucket] += 2.0
        if n > 1:
            changes = int(np.count_nonzero(coarse[1:] != coarse[:-1]))
            _add_reserved_numeric_feature(row, slot_idx, float(changes))
            change_bucket = _hash_numeric_value(shift, dim=dim, salt=0x27D4EB2F165667C5)
            row[change_bucket] += float(changes)
            pair_values = (coarse[:-1] << np.uint64(32)) ^ coarse[1:]
            changed_pairs = pair_values[coarse[1:] != coarse[:-1]]
            if changed_pairs.size:
                _add_hashed_counts(
                    row,
                    _hash_numeric_array(changed_pairs, dim=dim, salt=0x94D049BB133111EB + shift, np=np),
                    weight=0.25,
                    np=np,
                )


def _add_reserved_numeric_feature(row: Any, index: int, value: float) -> None:
    if 0 <= int(index) < int(row.shape[0]):
        row[int(index)] += float(value)


def _add_hashed_counts(row: Any, buckets: Any, *, weight: float, np: Any) -> None:
    counts = np.bincount(buckets.astype(np.int64), minlength=int(row.shape[0])).astype(np.float32)
    row += counts[: int(row.shape[0])] * float(weight)


def _hash_numeric_array(values: Any, *, dim: int, salt: int, np: Any) -> Any:
    arr = np.asarray(values, dtype=np.uint64) ^ np.uint64(salt)
    arr ^= arr >> np.uint64(33)
    arr *= np.uint64(0xFF51AFD7ED558CCD)
    arr ^= arr >> np.uint64(33)
    arr *= np.uint64(0xC4CEB9FE1A85EC53)
    arr ^= arr >> np.uint64(33)
    dim = max(1, int(dim))
    reserved = _numeric_hash_reserved_slots(dim)
    width = max(1, dim - reserved)
    out = np.mod(arr, np.uint64(width)).astype(np.int64)
    if dim > reserved:
        out += int(reserved)
    return out


def _hash_numeric_value(value: int, *, dim: int, salt: int) -> int:
    dim = max(1, int(dim))
    x = (int(value) ^ int(salt)) & ((1 << 64) - 1)
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & ((1 << 64) - 1)
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & ((1 << 64) - 1)
    x ^= x >> 33
    reserved = _numeric_hash_reserved_slots(dim)
    width = max(1, dim - reserved)
    out = int(x % width)
    return out + reserved if dim > reserved else int(x % dim)


def _numeric_hash_reserved_slots(dim: int) -> int:
    dim = max(1, int(dim))
    return min(8, dim) if dim >= 16 else min(4, dim)

def _leaf_texts(tree: Any) -> list[str]:
    leaves = getattr(tree, "leaves", None)
    if not leaves and callable(getattr(tree, "get_leaves", None)):
        leaves = tree.get_leaves()
    if leaves:
        return [_object_text(leaf) for leaf in leaves]
    return [_object_text(tree)]


def _object_text(value: Any) -> str:
    for attr in ("text", "content", "summary", "tokens"):
        candidate = getattr(value, attr, None)
        if candidate is not None:
            return _text_from_value(candidate)
    if isinstance(value, Mapping):
        for key in ("text", "content", "summary", "tokens"):
            if key in value:
                return _text_from_value(value[key])
    return _text_from_value(value)


def _text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key}:{_text_from_value(val)}" for key, val in sorted(value.items()))
    if isinstance(value, Sequence):
        return " ".join(str(item) for item in value)
    return str(value)


def _coerce_embedding(vector: Sequence[float], dim: int) -> list[float]:
    values = [float(x) for x in vector]
    dim = max(1, int(dim))
    if len(values) < dim:
        values.extend([0.0] * (dim - len(values)))
    return values[:dim]



def _target_rows(
    traces: Sequence[Any],
    config: NeuralOperatorFamilyConfig,
) -> tuple[list[Any], list[list[float]]]:
    rows: list[tuple[Any, list[float]]] = []
    for tree in traces:
        target = _target_vector(tree, config)
        if target is not None:
            rows.append((tree, target))
    if not rows:
        raise ValueError(
            "family='neural_operator' needs training trees with scalar or vector "
            "target metadata ('teacher_score_native', backend_config['target_key'], "
            "backend_config['target_keys'], or backend_config['target_vector_key'])."
        )
    width = len(rows[0][1])
    if width <= 0:
        raise ValueError("target vectors must be non-empty")
    for _tree, target in rows:
        if len(target) != width:
            raise ValueError("all target vectors must have the same length")
    return [tree for tree, _target in rows], [target for _tree, target in rows]


def _target_vector(tree: Any, config: NeuralOperatorFamilyConfig) -> list[float] | None:
    if config.target_keys:
        values = [_value_by_key(tree, key) for key in config.target_keys]
        if any(value is None for value in values):
            return None
        return [float(value) for value in values if value is not None]
    if config.target_vector_key:
        values = _vector_by_key(tree, config.target_vector_key)
        if values is None:
            return None
        if config.target_dim is not None and len(values) != int(config.target_dim):
            raise ValueError(
                f"target_vector_key={config.target_vector_key!r} produced {len(values)} values; "
                f"expected target_dim={int(config.target_dim)}"
            )
        return values
    if config.target_dim and int(config.target_dim) > 1:
        values = _vector_by_key(tree, "topic_proportions")
        if values is not None and len(values) == int(config.target_dim):
            return values
    score = _target_score(tree, config.target_key)
    return None if score is None else [float(score)]


def _value_by_key(tree: Any, key: str | None) -> float | None:
    if not key:
        return None
    meta = getattr(tree, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    if key in meta:
        return _safe_float(meta.get(key))
    return _safe_float(getattr(tree, str(key), None))


def _vector_by_key(tree: Any, key: str | None) -> list[float] | None:
    if not key:
        return None
    meta = getattr(tree, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    value = meta.get(key) if key in meta else getattr(tree, str(key), None)
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        out = [float(item) for item in value]
    except TypeError:
        return None
    return out if out else None


def _target_score(tree: Any, target_key: str | None) -> float | None:
    meta = getattr(tree, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    keys = [target_key] if target_key else []
    keys.extend(
        [
            "teacher_score_native",
            "teacher_score_1_7",
            "expert_score_for_objective",
            "expert_score_native",
            "expert_score_1_7",
        ]
    )
    for key in keys:
        if not key:
            continue
        score = _safe_float(meta.get(key))
        if score is not None:
            return score
    return _safe_float(getattr(tree, "document_score", None))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None:
        value = max(float(lower), value)
    if upper is not None:
        value = min(float(upper), value)
    return float(value)


__all__ = [
    "FNOFamily",
    "FNOFamilyConfig",
    "NeuralOperatorFamily",
    "NeuralOperatorFamilyConfig",
    "build_fno_family",
    "build_neural_operator_family",
]
