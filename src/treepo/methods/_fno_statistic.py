"""Composable-statistic adapter over a trained neural-operator family.

Exposes a trained family through the ``ComposableStatistic`` surface
(``encode_leaf`` / ``merge`` / ``readout`` / ``local_law_rows``) so tree tooling
can drive it directly. This is an adapter: it holds a reference to the family
and reads its model, rather than owning any training state itself.
"""

from __future__ import annotations

from typing import Any, Sequence

from treepo.local_law import LawKind, LocalLawAuditRow
from treepo.methods._fno_config import _clamp
from treepo.methods._fno_encoding import _leaf_token_groups
from treepo.methods._fno_transition import (
    _numeric_transition_state_targets,
    _pairwise_merge_depths,
    _tree_row_id,
)
from treepo.statistic import StatisticInfo


class _NeuralOperatorStatistic:
    """ComposableStatistic wrapper over a trained neural-operator family."""

    def __init__(self, family: Any) -> None:
        self.family = family
        self.info = StatisticInfo(
            name=str(family.name),
            state_kind=str(family.operator_kind),
            exact=False,
            supports_local_laws=True,
            metadata={
                "operator_kind": str(family.operator_kind),
                "output_dim": int(family._output_dim or 1),
                "target_key": family.config.target_key,
                "target_vector_key": family.config.target_vector_key,
                "numeric_transition_state_weight": float(family.config.numeric_transition_state_weight),
                "trained": family._model is not None,
            },
        )

    def encode_leaf(self, leaf: Any) -> Any:
        family = self.family
        family._ensure_model(output_dim=family._output_dim)
        assert family._model is not None
        proxy_tree = _LeafOnlyTree(leaf)
        x, _lengths = family._encode_trees([proxy_tree])
        family._model.eval()
        with family._torch.no_grad():
            states = family._model.leaf_operator(x)
        return states[0, 0].detach().clone()

    def merge(self, left: Any, right: Any) -> Any:
        family = self.family
        family._ensure_model(output_dim=family._output_dim)
        assert family._model is not None
        left_t = self._state_tensor(left)
        right_t = self._state_tensor(right)
        with family._torch.no_grad():
            merged = family._model.merge(family._torch.cat([left_t, right_t], dim=-1))
        return merged.squeeze(0).detach().clone()

    def readout(self, state: Any, query: Any = None) -> Any:
        del query
        family = self.family
        family._ensure_model(output_dim=family._output_dim)
        assert family._model is not None
        state_t = self._state_tensor(state)
        with family._torch.no_grad():
            raw = family._model.readout(state_t)
            values = family._denormalized_predictions(raw).detach().cpu().tolist()
        row = values[0] if values and isinstance(values[0], list) else values
        if (family._output_dim or 1) == 1:
            return _clamp(float(row[0] if isinstance(row, list) else row), family.config.target_min, family.config.target_max)
        return [
            _clamp(float(value), family.config.target_min, family.config.target_max)
            for value in list(row)
        ]

    def encode_tree(self, tree: Any) -> Any:
        family = self.family
        family._ensure_model(output_dim=family._output_dim)
        assert family._model is not None
        x, lengths = family._encode_trees([tree])
        family._model.eval()
        with family._torch.no_grad():
            leaf_states = family._model.leaf_operator(x)
            length = max(1, int(lengths[0].detach().cpu().item()))
            root, _ = family._model._compose(leaf_states[0, :length, :], collect_trace=False)
        return root.detach().clone()

    def predict_tree(self, tree: Any) -> Any:
        return self.readout(self.encode_tree(tree))

    def local_law_rows(
        self,
        units: Sequence[Any],
        *,
        query: Any = None,
        oracle: Any = None,
    ) -> Sequence[LocalLawAuditRow]:
        del query, oracle
        family = self.family
        if family._model is None:
            return ()
        trees = list(units or ())
        if not trees:
            return ()
        targets = _numeric_transition_state_targets(
            trees,
            family.config,
            torch=family._torch,
            device=family._device,
        )
        if targets is None:
            return ()
        x, lengths = family._encode_trees(trees)
        family._model.eval()
        with family._torch.no_grad():
            _raw, traces = family._model.forward_with_trace(x, lengths)
        rows: list[LocalLawAuditRow] = []
        for tree_idx, (tree, pred_trace, target_trace) in enumerate(zip(trees, traces, targets)):
            leaf_count = len(_leaf_token_groups(tree) or ())
            if int(pred_trace.shape[0]) != int(target_trace.shape[0]):
                raise ValueError(
                    "local-law state audit node count mismatch: model trace has "
                    f"{int(pred_trace.shape[0])} nodes, targets have {int(target_trace.shape[0])}"
                )
            n_rows = int(pred_trace.shape[0])
            if n_rows != max(1, 2 * leaf_count - 1):
                raise ValueError(
                    f"local-law state audit expected {max(1, 2 * leaf_count - 1)} trace rows "
                    f"for {leaf_count} leaves, model trace has {n_rows}"
                )
            depths = _pairwise_merge_depths(leaf_count)
            for node_idx in range(n_rows):
                d = min(int(pred_trace.shape[1]), int(target_trace.shape[1]))
                if d <= 0:
                    continue
                loss = float(
                    family._torch.nn.functional.mse_loss(
                        pred_trace[node_idx, :d],
                        target_trace[node_idx, :d].to(device=family._device, dtype=pred_trace.dtype),
                    ).detach().cpu()
                )
                law_kind = LawKind.C1_LEAF if node_idx < leaf_count else LawKind.C3_MERGE
                rows.append(
                    LocalLawAuditRow(
                        row_id=f"{_tree_row_id(tree, tree_idx)}:state:{node_idx}",
                        law_kind=law_kind,
                        proxy_loss=loss,
                        oracle_loss=loss,
                        observed=True,
                        propensity=1.0,
                        depth=depths[node_idx] if node_idx < len(depths) else 0,
                        metadata={
                            "statistic": self.info.name,
                            "state_kind": self.info.state_kind,
                            "check": "numeric_transition_state",
                            "tree_index": int(tree_idx),
                            "node_index": int(node_idx),
                            "target_dim": int(target_trace.shape[1]),
                            "learned_dim": int(pred_trace.shape[1]),
                        },
                    )
                )
        return tuple(rows)

    def _state_tensor(self, value: Any) -> Any:
        family = self.family
        if hasattr(value, "detach"):
            tensor = value.to(device=family._device, dtype=family._torch.float32)
        else:
            tensor = family._torch.tensor(value, dtype=family._torch.float32, device=family._device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor


class _LeafOnlyTree:
    def __init__(self, leaf: Any) -> None:
        self.leaves = (leaf,)


__all__ = ["_LeafOnlyTree", "_NeuralOperatorStatistic"]
