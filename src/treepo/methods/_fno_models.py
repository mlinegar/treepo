"""Torch model definitions for neural-operator method families.

Every built-in neural-operator kind implements the same theorem-facing contract:
one learned ``g`` is reused for the leaf build and every internal reduction call.
Fixed input adapters distinguish a raw leaf from a pair of stored states; they do
not introduce leaf-only or merge-only learned parameters.
"""

from __future__ import annotations

from typing import Any

from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_neuralop import (
    _neuralop_constructor_kwargs,
    _neuralop_model_class,
)

_SHARED_G_INPUT_CHANNELS = 4
_SHARED_G_ARCHITECTURE_VERSION = "single_shared_pair_g_v3"


class _SharedPairG:
    """One learned operator used as both ``g(x, empty)`` and ``g(s_l, s_r)``.

    The learned core always receives four channels over the embedding axis:
    left content, right content, left occupancy, and right occupancy.  A leaf
    uses an empty right slot with occupancy zero; a merge with an actually-zero
    right state still has occupancy one and is therefore a distinct input.

    The deterministic masked-mean bypass gives a coherent zero-residual
    initialization: ``g(x, empty) == x`` and ``g(s, t) == (s + t) / 2``.  The
    same residual operator and parameters modify both call domains.
    """

    def __new__(
        cls,
        *,
        operator_kind: str,
        config: NeuralOperatorFamilyConfig,
        torch: Any,
    ) -> Any:
        class _Model(torch.nn.Module):
            architecture_version = _SHARED_G_ARCHITECTURE_VERSION
            input_channels = _SHARED_G_INPUT_CHANNELS

            def __init__(self) -> None:
                super().__init__()
                hidden = max(1, int(config.hidden_channels))
                self.operator = _build_shared_operator(
                    operator_kind=operator_kind,
                    config=config,
                    torch=torch,
                    in_channels=_SHARED_G_INPUT_CHANNELS,
                    out_channels=hidden,
                )
                self.residual_head = torch.nn.Conv1d(hidden, 1, kernel_size=1)
                torch.nn.init.zeros_(self.residual_head.weight)
                if self.residual_head.bias is not None:
                    torch.nn.init.zeros_(self.residual_head.bias)

            def pack_inputs(self, left: Any, right: Any | None = None) -> Any:
                """Return the canonical pair input without applying learned g."""

                if left.ndim != 2:
                    raise ValueError(
                        f"shared g expects state rows shaped (B, D); got {tuple(left.shape)}"
                    )
                if right is not None and tuple(right.shape) != tuple(left.shape):
                    raise ValueError(
                        "shared g left/right state rows must have identical shapes; "
                        f"got {tuple(left.shape)} and {tuple(right.shape)}"
                    )
                empty = torch.zeros_like(left)
                right_content = empty if right is None else right
                left_present = torch.ones_like(left)
                right_present = empty if right is None else torch.ones_like(left)
                return torch.stack(
                    [left, right_content, left_present, right_present],
                    dim=1,
                )

            def forward(self, left: Any, right: Any | None = None) -> Any:
                if left.ndim < 2:
                    raise ValueError(
                        "shared g expects tensors with a final state dimension; "
                        f"got {tuple(left.shape)}"
                    )
                original_shape = tuple(left.shape)
                state_dim = int(original_shape[-1])
                left_rows = left.reshape(-1, state_dim)
                right_rows = None
                if right is not None:
                    if tuple(right.shape) != original_shape:
                        raise ValueError(
                            "shared g left/right states must have identical shapes; "
                            f"got {original_shape} and {tuple(right.shape)}"
                        )
                    right_rows = right.reshape(-1, state_dim)
                packed = self.pack_inputs(left_rows, right_rows)
                hidden = self.operator(packed)
                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]
                if hidden.ndim != 3:
                    hidden = hidden.reshape(int(packed.shape[0]), -1, state_dim)
                if int(hidden.shape[-1]) != state_dim:
                    raise ValueError(
                        f"operator_kind={operator_kind!r} changed the shared-g spatial "
                        f"extent from {state_dim} to {int(hidden.shape[-1])}; built-in "
                        "tree reduction requires extent-preserving operators"
                    )
                residual = self.residual_head(hidden).squeeze(1)
                baseline = left_rows if right_rows is None else 0.5 * (left_rows + right_rows)
                return (baseline + residual).reshape(original_shape)

            def encode_leaf(self, content: Any) -> Any:
                """Compatibility alias for ``g(content, empty)``."""

                return self.forward(content)

            def merge(self, left: Any, right: Any) -> Any:
                """Compatibility alias for ``g(left, right)``."""

                return self.forward(left, right)

        return _Model()


class _UnifiedGTreeModel:
    """Pairwise tree model with one registered and repeatedly applied ``g``."""

    def __new__(
        cls,
        *,
        operator_kind: str,
        config: NeuralOperatorFamilyConfig,
        torch: Any,
        output_dim: int,
    ) -> Any:
        class _Model(torch.nn.Module):
            architecture_version = _SHARED_G_ARCHITECTURE_VERSION
            shared_g_parameterization = "one_module_reused_at_leaf_and_merge"
            f_module_names = ("readout",)
            g_module_names = ("g",)
            bounded_output = bool(
                operator_kind == "fno" or config.law_state_root_readout == "first_coordinate"
            )

            def __init__(self) -> None:
                super().__init__()
                dim = max(1, int(config.embedding_dim))
                self.g_mode = "learned"
                self.g = _SharedPairG(
                    operator_kind=operator_kind,
                    config=config,
                    torch=torch,
                )
                self.readout = torch.nn.Sequential(
                    torch.nn.Linear(dim, max(1, int(config.head_hidden_dim))),
                    torch.nn.GELU(),
                    torch.nn.Linear(
                        max(1, int(config.head_hidden_dim)),
                        max(1, int(output_dim)),
                    ),
                )
                if operator_kind == "fno":
                    final = self.readout[-1]
                    torch.nn.init.zeros_(final.weight)
                    torch.nn.init.zeros_(final.bias)

            def set_g_mode(self, mode: str) -> None:
                resolved = str(mode).strip().lower()
                if resolved not in {"identity", "learned"}:
                    raise ValueError(f"unsupported neural-operator g execution mode {mode!r}")
                self.g_mode = resolved

            def _encode_leaves(self, x: Any, lengths: Any | None = None) -> Any:
                """Apply the same g used at merges to every real raw leaf."""

                batch = int(x.shape[0])
                max_leaves = int(x.shape[1])
                dim = int(x.shape[2])
                flat = x.reshape(batch * max_leaves, dim)
                if lengths is None:
                    real = torch.ones(
                        batch * max_leaves,
                        dtype=torch.bool,
                        device=x.device,
                    )
                else:
                    real = (
                        torch.arange(max_leaves, device=x.device)[None, :] < lengths[:, None]
                    ).reshape(-1)
                rows = flat[real]
                states = torch.zeros_like(flat)
                if int(rows.shape[0]) > 0:
                    if self.g_mode == "identity":
                        states[real] = rows
                    else:
                        states[real] = self.g(rows)
                return states.reshape(batch, max_leaves, dim)

            def _merge_rows(self, left: Any, right: Any) -> Any:
                return self.g(left, right)

            def _read(self, states: Any) -> Any:
                if config.law_state_root_readout == "first_coordinate":
                    return states[..., :1]
                raw = self.readout(states)
                return torch.sigmoid(raw) if operator_kind == "fno" else raw

            def leaf_operator(self, x: Any) -> Any:
                """Compatibility alias; learned execution is ``self.g``."""

                return self._encode_leaves(x)

            def merge(self, pair: Any) -> Any:
                """Compatibility alias over a last-axis concatenated pair."""

                dim = int(pair.shape[-1]) // 2
                if dim <= 0 or int(pair.shape[-1]) != 2 * dim:
                    raise ValueError("merge expects a last-axis concatenation of two states")
                return self.g(pair[..., :dim], pair[..., dim:])

            def forward(self, x: Any, lengths: Any) -> Any:
                return self._forward(x, lengths, collect_trace=False)[0]

            def forward_with_trace(self, x: Any, lengths: Any) -> tuple[Any, list[Any]]:
                pred, traces, _leaf_states = self._forward(
                    x,
                    lengths,
                    collect_trace=True,
                )
                return pred, traces

            def forward_rollup(
                self,
                x: Any,
                lengths: Any,
                weights: Any,
                *,
                collect_trace: bool = False,
            ) -> tuple[Any, list[Any]]:
                if collect_trace:
                    _pred, traces, leaf_states = self._forward(
                        x,
                        lengths,
                        collect_trace=True,
                    )
                else:
                    leaf_states = self._encode_leaves(x, lengths)
                    traces = []
                leaf_preds = self._read(leaf_states)
                return (leaf_preds * weights.unsqueeze(-1)).sum(dim=1), traces

            def _forward(
                self,
                x: Any,
                lengths: Any,
                *,
                collect_trace: bool,
            ) -> tuple[Any, list[Any], Any]:
                leaf_states = self._encode_leaves(x, lengths)
                if int(leaf_states.shape[0]) > 0 and bool(
                    torch.all(lengths == lengths[0]).detach().cpu().item()
                ):
                    length = max(1, int(lengths[0].detach().cpu().item()))
                    roots, trace = self._compose_batch(
                        leaf_states[:, :length, :],
                        collect_trace=collect_trace,
                    )
                    traces = (
                        [trace[idx] for idx in range(int(trace.shape[0]))] if collect_trace else []
                    )
                    return self._read(roots), traces, leaf_states
                roots = []
                traces = []
                for idx, raw_length in enumerate(lengths.detach().cpu().tolist()):
                    length = max(1, int(raw_length))
                    root, trace = self._compose(
                        leaf_states[idx, :length, :],
                        collect_trace=collect_trace,
                    )
                    roots.append(root)
                    if collect_trace:
                        traces.append(trace)
                return self._read(torch.stack(roots, dim=0)), traces, leaf_states

            def _compose_batch(self, states: Any, *, collect_trace: bool) -> tuple[Any, Any]:
                trace_parts = [states] if collect_trace else None
                while int(states.shape[1]) > 1:
                    n_states = int(states.shape[1])
                    pair_count = n_states // 2
                    left = states[:, 0 : pair_count * 2 : 2, :]
                    right = states[:, 1 : pair_count * 2 : 2, :]
                    batch = int(states.shape[0])
                    merged = self._merge_rows(
                        left.reshape(batch * pair_count, -1),
                        right.reshape(batch * pair_count, -1),
                    ).reshape(batch, pair_count, -1)
                    if trace_parts is not None:
                        trace_parts.append(merged)
                    if n_states % 2:
                        states = torch.cat([merged, states[:, -1:, :]], dim=1)
                    else:
                        states = merged
                trace = torch.cat(trace_parts, dim=1) if trace_parts is not None else None
                return states[:, 0, :], trace

            def _compose(self, states: Any, *, collect_trace: bool) -> tuple[Any, Any]:
                trace_parts = [states] if collect_trace else None
                while int(states.shape[0]) > 1:
                    n_states = int(states.shape[0])
                    pair_count = n_states // 2
                    left = states[0 : pair_count * 2 : 2]
                    right = states[1 : pair_count * 2 : 2]
                    merged = self._merge_rows(left, right)
                    if trace_parts is not None:
                        trace_parts.append(merged)
                    if n_states % 2:
                        states = torch.cat([merged, states[-1:]], dim=0)
                    else:
                        states = merged
                trace = torch.cat(trace_parts, dim=0) if trace_parts is not None else None
                return states.squeeze(0), trace

        return _Model()


def _build_shared_operator(
    *,
    operator_kind: str,
    config: NeuralOperatorFamilyConfig,
    torch: Any,
    in_channels: int,
    out_channels: int,
) -> Any:
    if operator_kind == "conv1d":
        hidden = max(1, int(config.hidden_channels))
        kernel = _odd_kernel_size(config.conv_kernel_size)
        layers: list[Any] = []
        for layer_idx in range(max(1, int(config.n_layers))):
            left = int(in_channels) if layer_idx == 0 else hidden
            layers.append(
                torch.nn.Conv1d(
                    left,
                    hidden,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
            )
            layers.append(torch.nn.GELU())
        if int(out_channels) != hidden:
            layers.append(torch.nn.Conv1d(hidden, int(out_channels), kernel_size=1))
        return torch.nn.Sequential(*layers)

    model_cls = _neuralop_model_class(operator_kind, required=True)
    kwargs = _neuralop_constructor_kwargs(
        operator_kind=operator_kind,
        config=config,
        model_cls=model_cls,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
    )
    # The pair schema is structural, not a tunable operator_kwarg.  Re-assert
    # it after user extras were merged by the generic constructor helper.
    for key, value in (
        ("in_channels", int(in_channels)),
        ("out_channels", int(out_channels)),
        ("fno_in_channels", int(in_channels)),
    ):
        if key in kwargs:
            kwargs[key] = value
    return model_cls(**kwargs)


def _odd_kernel_size(value: Any) -> int:
    kernel = max(1, int(value))
    return kernel if kernel % 2 == 1 else kernel + 1


__all__ = [
    "_SHARED_G_ARCHITECTURE_VERSION",
    "_SHARED_G_INPUT_CHANNELS",
    "_SharedPairG",
    "_UnifiedGTreeModel",
    "_odd_kernel_size",
]
