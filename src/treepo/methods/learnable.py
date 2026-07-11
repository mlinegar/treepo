"""Minimal demonstrative family that uses local-law math as a training signal.

The point is to prove the local-law arithmetic *drives a training step*
directly. The family is a single learnable scalar ``c``
(the predicted root score for every tree). ``train_f`` builds
:class:`LocalLawAuditRow` objects from training trees and calls
:func:`local_law_objective_summary` in ``sampled_ipw`` mode — the
returned ``objective`` *is* the trained ``f``.

This is the Horvitz-Thompson / Hájek estimator of the mean root score
under unequal sampling propensities. It is the simplest training
problem where IPW correction matters: with confounded propensities
(high-value trees sampled more often), the naive observed mean is
biased; the IPW estimate is unbiased. Tests demonstrate both behaviors.

Real backends (FNO, learned merges) plug in the same way from downstream
packages: their ``train_f`` would build per-node rows from a forward pass and
call the same objective. The math is identical; only the rows differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from treepo.local_law import (
    LawKind,
    LocalLawAuditRow,
    local_law_objective_summary,
)
from treepo.statistic import StatisticInfo
from treepo.tree import tree_root_target, tree_row_id


class LearnableConstantFamily:
    """f is a single scalar; trained via IPW-corrected mean over sampled
    training trees. g passes through unchanged.

    Each training tree carries:

    - a root target, read through the canonical
      :func:`treepo.tree.tree_root_target` convention
      (``"teacher_score_native"`` metadata first, then the ``document_score``
      / ``root_label`` fields bundle trees expose),
    - ``"observed"`` metadata (whether the oracle was sampled for this tree),
    - ``"propensity"`` metadata (the design sampling probability).

    Test trees only need ``"split"`` metadata for the evaluator.
    """

    name = "learnable_constant"

    def __init__(self, *, gamma_depth: float = 1.0) -> None:
        self._gamma_depth = float(gamma_depth)
        self.last_trained_f: float | None = None
        self.last_objective_summary: Mapping[str, Any] | None = None

    # ------------------------ FamilyRuntime API ------------------------- #

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> float:
        rows = self._rows_from_trees(traces)
        summary = local_law_objective_summary(
            rows,
            objective_mode="sampled_ipw",
            gamma_depth=self._gamma_depth,
        )
        learned = float(summary.objective)
        self.last_trained_f = learned
        self.last_objective_summary = summary.to_dict()
        return learned

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        return g_init  # g passes through unchanged; only f carries state

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> List[Optional[float]]:
        # f is the raw trained scalar (in-memory) at evaluation time.
        if f is None:
            return [None] * len(trees)
        try:
            value = float(f)
        except (TypeError, ValueError):
            return [None] * len(trees)
        return [value] * len(trees)

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        return None

    def as_statistic(self, *, f: Any = None, g: Any = None) -> Any:
        del g
        value: float | None
        try:
            value = float(f) if f is not None else self.last_trained_f
        except (TypeError, ValueError):
            value = self.last_trained_f
        if value is None:
            return None
        return _ConstantStatistic(value=float(value))

    # ------------------------------ helpers ------------------------------ #

    def _rows_from_trees(self, trees: Sequence[Any]) -> List[LocalLawAuditRow]:
        rows: List[LocalLawAuditRow] = []
        for idx, tree in enumerate(trees):
            meta = getattr(tree, "metadata", None) or {}
            observed = bool(meta.get("observed", True))
            propensity = float(meta.get("propensity", 1.0))
            oracle = tree_root_target(tree) if observed else None
            # Sampled-IPW mode only reads oracle_loss / propensity / observed.
            # We use ``oracle_loss`` as the per-tree target value (the loss
            # is min-zero at the trained constant by construction).
            rows.append(
                LocalLawAuditRow(
                    row_id=f"row_{idx}",
                    law_kind=LawKind.C1_LEAF,
                    proxy_loss=0.0,
                    oracle_loss=oracle,
                    observed=observed,
                    propensity=propensity,
                    node_weight=1.0,
                    depth=0,
                )
            )
        return rows


class _ConstantStatistic:
    """ComposableStatistic surface for the trained constant.

    The state is the constant itself, so C2 idempotence and C3b merge
    compositionality hold by construction — the rows certify that with exact
    0/1 indicator checks. C1 sufficiency is the substantive check: the
    constant leaf state must carry each tree's score, so its loss is the
    squared error against the teacher score under the logged sampling design.
    """

    def __init__(self, *, value: float) -> None:
        self.value = float(value)
        self.info = StatisticInfo(
            name="learnable_constant",
            state_kind="constant",
            exact=True,
            supports_local_laws=True,
            metadata={"value": float(value)},
        )

    def encode_leaf(self, leaf: Any) -> float:
        del leaf
        return self.value

    def merge(self, left: Any, right: Any) -> float:
        del left, right
        return self.value

    def readout(self, state: Any, query: Any = None) -> float:
        del query
        return float(state)

    def predict_tree(self, tree: Any) -> float:
        del tree
        return self.value

    def local_law_rows(
        self,
        units: Sequence[Any],
        *,
        query: Any = None,
        oracle: Any = None,
    ) -> Sequence[LocalLawAuditRow]:
        del query, oracle
        rows: list[LocalLawAuditRow] = []
        for idx, tree in enumerate(list(units or ())):
            tree_id = tree_row_id(tree, idx, fallback_prefix="tree")
            base_metadata = {"statistic": self.info.name, "state_kind": "constant"}
            merged = self.merge(self.value, self.value)
            rows.append(
                LocalLawAuditRow(
                    row_id=f"{tree_id}:idempotence",
                    law_kind=LawKind.C2_IDEMPOTENCE,
                    proxy_loss=0.0 if merged == self.value else 1.0,
                    oracle_loss=0.0 if merged == self.value else 1.0,
                    observed=True,
                    propensity=1.0,
                    metadata={**base_metadata, "check": "self_merge_identity", "law_facet": "c2_idempotence"},
                )
            )
            composed = self.readout(self.merge(self.encode_leaf(None), self.encode_leaf(None)))
            rows.append(
                LocalLawAuditRow(
                    row_id=f"{tree_id}:composition",
                    law_kind=LawKind.C3_MERGE,
                    proxy_loss=0.0 if composed == self.value else 1.0,
                    oracle_loss=0.0 if composed == self.value else 1.0,
                    observed=True,
                    propensity=1.0,
                    metadata={**base_metadata, "check": "composed_readout_identity", "law_facet": "c3b_compositionality"},
                )
            )
            meta = getattr(tree, "metadata", None) or {}
            teacher = tree_root_target(tree)
            if teacher is None:
                continue
            loss = float((self.value - float(teacher)) ** 2)
            observed = bool(meta.get("observed", True))
            rows.append(
                LocalLawAuditRow(
                    row_id=f"{tree_id}:sufficiency",
                    law_kind=LawKind.C1_LEAF,
                    proxy_loss=loss,
                    oracle_loss=loss if observed else None,
                    observed=observed,
                    propensity=float(meta.get("propensity", 1.0)),
                    metadata={**base_metadata, "check": "teacher_agreement", "law_facet": "c1_sufficiency"},
                )
            )
        return tuple(rows)


__all__ = ["LearnableConstantFamily"]
