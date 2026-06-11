"""Minimal demonstrative family that uses local-law math as a training signal.

The point is to prove the local-law arithmetic *drives a training step*, not
just sits as post-hoc audit. The family is a single learnable scalar ``c``
(the predicted root score for every tree). ``train_f`` builds
:class:`LocalLawAuditRow` objects from training trees and calls
:func:`local_law_objective_summary` in ``sampled_ipw`` mode — the
returned ``objective`` *is* the trained ``f``.

This is the Horvitz-Thompson / Hájek estimator of the mean root score
under unequal sampling propensities. It is the simplest training
problem where IPW correction matters: with confounded propensities
(high-value trees sampled more often), the naive observed mean is
biased; the IPW estimate is unbiased. Tests demonstrate both behaviors.

Real backends (FNO, learned merges) plug in the same way — their
``train_f`` would build per-node rows from a forward pass and call the
same objective. The math is identical; only the rows differ. v1 does
not yet ship a learned FNO that uses this surface (that wiring lives in
``treepo._research.ctreepo.fno_family`` today, not in ``treepo.methods``), but the
demonstrative family + this module's signature show what the contract
looks like.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from treepo.local_law import (
    LocalLawAuditRow,
    local_law_objective_summary,
)


class LearnableConstantFamily:
    """f is a single scalar; trained via IPW-corrected mean over sampled
    training trees. g is unused.

    Each training tree's ``metadata`` must carry:

    - ``"teacher_score_1_7"`` (the oracle root score),
    - ``"observed"`` (whether the oracle was sampled for this tree),
    - ``"propensity"`` (the design sampling probability).

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
        return g_init  # g is structural, not trained

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

    # ------------------------------ helpers ------------------------------ #

    def _rows_from_trees(self, trees: Sequence[Any]) -> List[LocalLawAuditRow]:
        rows: List[LocalLawAuditRow] = []
        for idx, tree in enumerate(trees):
            meta = getattr(tree, "metadata", None) or {}
            observed = bool(meta.get("observed", True))
            propensity = float(meta.get("propensity", 1.0))
            oracle_value = meta.get("teacher_score_1_7") if observed else None
            oracle = float(oracle_value) if oracle_value is not None else None
            # Sampled-IPW mode only reads oracle_loss / propensity / observed.
            # We use ``oracle_loss`` as the per-tree target value (the loss
            # is min-zero at the trained constant by construction).
            rows.append(
                LocalLawAuditRow(
                    row_id=f"row_{idx}",
                    law_kind="c1_leaf",
                    proxy_loss=0.0,
                    oracle_loss=oracle,
                    observed=observed,
                    propensity=propensity,
                    node_weight=1.0,
                    depth=0,
                )
            )
        return rows


__all__ = ["LearnableConstantFamily"]
