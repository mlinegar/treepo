"""Torch model definitions for neural-operator method families.

Model logic only: the f/g tree model (``_TreeFGModel``) that composes leaf
states pairwise up to the root, and the leaf operators it builds (a neuralop
model or the local ``conv1d`` baseline). Torch is only imported through the
``torch`` handle passed in by the family, keeping ``import treepo`` light.
"""

from __future__ import annotations

from typing import Any

from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_neuralop import (
    _neuralop_constructor_kwargs,
    _neuralop_model_class,
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


def _odd_kernel_size(value: Any) -> int:
    kernel = max(1, int(value))
    return kernel if kernel % 2 == 1 else kernel + 1


__all__ = [
    "_build_leaf_operator",
    "_LeafConv1D",
    "_LeafNeuralOp",
    "_odd_kernel_size",
    "_TreeFGModel",
]
