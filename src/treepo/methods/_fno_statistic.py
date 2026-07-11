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
    _numeric_transition_law_rows,
    _numeric_transition_state_targets,
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

    def node_readouts(self, trees: Sequence[Any]) -> list[dict[str, Any]]:
        """Return ``f`` readouts for every node of every tree, in trace order.

        Applies the readout head to each state the merge trace produced, so
        the rows show the prediction forming up the tree: leaves first, then
        each merge level, root last. The final row per tree equals
        ``predict_tree``. Row shape: ``tree_id``, ``node_index``, ``value``
        (scalar, or a list for vector targets).
        """

        family = self.family
        if family._model is None:
            return []
        tree_list = list(trees or ())
        if not tree_list:
            return []
        x, lengths = family._encode_trees(tree_list)
        family._model.eval()
        rows: list[dict[str, Any]] = []
        with family._torch.no_grad():
            _raw, traces = family._model.forward_with_trace(x, lengths)
            for tree_idx, (tree, trace) in enumerate(zip(tree_list, traces)):
                raw = family._model.readout(trace)
                values = family._denormalized_predictions(raw).detach().cpu().tolist()
                for node_idx, row in enumerate(values):
                    row = row if isinstance(row, list) else [row]
                    clamped = [
                        _clamp(float(value), family.config.target_min, family.config.target_max)
                        for value in row
                    ]
                    rows.append(
                        {
                            "tree_id": _tree_row_id(tree, tree_idx),
                            "node_index": int(node_idx),
                            "value": clamped[0] if (family._output_dim or 1) == 1 else clamped,
                        }
                    )
        return rows

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
            for tree, pred_trace in zip(trees, traces):
                leaf_count = len(_leaf_token_groups(tree) or ())
                n_rows = int(pred_trace.shape[0])
                if n_rows != max(1, 2 * leaf_count - 1):
                    raise ValueError(
                        f"local-law state audit expected {max(1, 2 * leaf_count - 1)} trace rows "
                        f"for {leaf_count} leaves, model trace has {n_rows}"
                    )
            law_rows = _numeric_transition_law_rows(
                traces,
                targets,
                torch=family._torch,
                device=family._device,
                dtype=traces[0].dtype,
            )
            if law_rows is None:
                return ()
            loss_tensor, depth_tensor, leaf_tensor = law_rows
            # One host sync for the whole audit batch; never per node.
            losses = loss_tensor.detach().cpu().tolist()
            depths = depth_tensor.detach().cpu().tolist()
            leaf_flags = leaf_tensor.detach().cpu().tolist()
        rows: list[LocalLawAuditRow] = []
        cursor = 0
        for tree_idx, (tree, pred_trace, target_trace) in enumerate(zip(trees, traces, targets)):
            n_rows = int(pred_trace.shape[0])
            d = min(int(pred_trace.shape[1]), int(target_trace.shape[1]))
            if n_rows <= 0 or d <= 0:
                continue
            tree_id = _tree_row_id(tree, tree_idx)
            for node_idx in range(n_rows):
                loss = float(losses[cursor])
                is_leaf_row = bool(leaf_flags[cursor])
                law_kind = LawKind.C1_LEAF if is_leaf_row else LawKind.C3_MERGE
                depth = int(depths[cursor])
                cursor += 1
                rows.append(
                    LocalLawAuditRow(
                        row_id=f"{tree_id}:state:{node_idx}",
                        law_kind=law_kind,
                        proxy_loss=loss,
                        oracle_loss=loss,
                        observed=True,
                        propensity=1.0,
                        depth=depth,
                        metadata={
                            "statistic": self.info.name,
                            "state_kind": self.info.state_kind,
                            "check": "numeric_transition_state",
                            "law_facet": (
                                "c1_sufficiency" if is_leaf_row else "c3b_compositionality"
                            ),
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
