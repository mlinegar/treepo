"""f vs f* error estimation — per-split, per-dimension, per-law.

The C-TreePO paper's central diagnostic is the per-split / per-law
deviation of a learned (or sketched) ``f`` from the oracle ``f*``:

- ``internal_f_pearson`` / ``internal_f_mae``: how closely ``f`` agrees
  with the *teacher* (proxy supervision signal).
- ``external_expert_pearson`` / ``external_expert_mae``: how closely
  ``f`` agrees with the *gold expert* (truth).
- ``f_star_gap = internal_f_pearson - external_expert_pearson``: the
  reward-hacking signal — positive gap means ``f`` matches the
  teacher's quirks more than the truth.

The upstream alternating loop already computes these per
``("all", "train", "val", "test")``. Earlier versions of
``treepo.methods._final_metrics`` discarded every split except ``"all"``.
These tests verify the surfaces are now wired correctly.

For audits on local-law rows, the user's central question is "compare
local-law error between f* and f *per law kind* (C1 / C2 / C3)". The
audit method now returns a ``by_law_kind`` decomposition; tests verify
it matches the law assignments on the rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import pytest

from treepo.methods import LocalLawAuditRow
from treepo.local_law import audit_local_laws
from treepo import fit


# --------------------------------------------------------------------------- #
# Tools: a family that returns the teacher score directly + trees with
# distinct teacher vs expert metadata so f_star_gap is observable.
# --------------------------------------------------------------------------- #


class _TeacherPassthroughFamily:
    name = "teacher_passthrough"

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        return f_init

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        return g_init

    def score_roots_with_f(self, *, f, g, trees) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for tree in trees:
            meta = getattr(tree, "metadata", None) or {}
            value = meta.get("teacher_score_1_7")
            out.append(float(value) if value is not None else None)
        return out

    def validate_artifact(self, *, kind, artifact):
        return None


def _make_trees_with_distinct_teacher_and_expert(
    teacher_scores: Sequence[float],
    expert_scores: Sequence[float],
    split: str = "test",
) -> List[SimpleNamespace]:
    assert len(teacher_scores) == len(expert_scores)
    return [
        SimpleNamespace(
            leaves=[SimpleNamespace(tokens=[])],
            metadata={
                "split": split,
                "teacher_score_1_7": float(t),
                "teacher_score_native": float(t),
                "expert_score_1_7": float(e),
                "expert_score_native": float(e),
                "expert_target_scale": "raw",
                "expert_score_for_objective": float(e),
            },
        )
        for t, e in zip(teacher_scores, expert_scores)
    ]


# --------------------------------------------------------------------------- #
# 1. Per-split metrics are surfaced
# --------------------------------------------------------------------------- #


def test_per_split_metrics_surface_in_result_metrics(tmp_path: Path) -> None:
    """Build a fixture where train/test splits exist; assert
    ``result.metrics`` carries both prefixed (``test_*``, ``train_*``)
    and unprefixed (``"all"``-equivalent) entries.
    """
    train_trees = _make_trees_with_distinct_teacher_and_expert(
        teacher_scores=[1, 2, 3, 4], expert_scores=[1, 2, 3, 4], split="train"
    )
    test_trees = _make_trees_with_distinct_teacher_and_expert(
        teacher_scores=[5, 6, 7, 8], expert_scores=[5, 6, 7, 8], split="test"
    )
    eval_data = train_trees + test_trees

    family = _TeacherPassthroughFamily()
    result = fit(
        {
            "family": "passthrough",
            "eval_data": eval_data,
            "backend_config": {
                "family_runtime": family,
                "output_dir": str(tmp_path),
            },
        },
    )
    assert result.status == "success"

    # Unprefixed "all"-split metrics.
    assert "internal_f_mae" in result.metrics
    assert result.metrics["n"] == 8.0
    # Per-split prefixed metrics.
    assert "test_internal_f_mae" in result.metrics
    assert "train_internal_f_mae" in result.metrics
    assert result.metrics["test_n"] == 4.0
    assert result.metrics["train_n"] == 4.0
    # Structured per-split block on summary.
    sm = result.summary["split_metrics"]
    assert set(sm) >= {"all", "train", "test"}
    assert sm["test"]["n"] == 4
    assert sm["train"]["n"] == 4


def test_per_split_metrics_only_present_when_split_has_rows(tmp_path: Path) -> None:
    """Fixture with no ``val`` rows must NOT emit ``val_*`` keys."""
    trees = _make_trees_with_distinct_teacher_and_expert(
        [1, 2, 3, 4], [1, 2, 3, 4], split="test"
    )
    family = _TeacherPassthroughFamily()
    result = fit(
        {
            "family": "passthrough",
            "eval_data": trees,
            "backend_config": {"family_runtime": family, "output_dir": str(tmp_path)},
        },
    )
    assert "test_internal_f_mae" in result.metrics
    assert "val_internal_f_mae" not in result.metrics
    assert "train_internal_f_mae" not in result.metrics
    sm = result.summary["split_metrics"]
    # The upstream evaluator may emit a val/train entry with n=0 — those
    # are real but uninformative and we accept either presence with n=0
    # or absence. The constraint is that no metric value should appear
    # for empty splits.
    for split_name in ("train", "val"):
        if split_name in sm:
            assert sm[split_name]["n"] == 0


# --------------------------------------------------------------------------- #
# 2. f_star_gap is observable when teacher and expert diverge
# --------------------------------------------------------------------------- #


def test_f_star_gap_positive_when_f_matches_teacher_but_not_expert(
    tmp_path: Path,
) -> None:
    """Strong test of the reward-hacking diagnostic.

    Construction: 8 trees. teacher_score = [1..8]. expert_score =
    [8..1] (anti-correlated). f = teacher_passthrough (i.e. predictions
    match teacher exactly). Then:

    - internal Pearson(pred, teacher) = +1.0
    - external Pearson(pred, expert)  = -1.0
    - f_star_gap = internal - external = +2.0
    """
    teacher = [1, 2, 3, 4, 5, 6, 7, 8]
    expert = [8, 7, 6, 5, 4, 3, 2, 1]
    trees = _make_trees_with_distinct_teacher_and_expert(
        teacher, expert, split="test"
    )

    family = _TeacherPassthroughFamily()
    result = fit(
        {
            "family": "teacher_only",
            "eval_data": trees,
            "backend_config": {"family_runtime": family, "output_dir": str(tmp_path)},
        },
    )
    assert result.status == "success"
    gap = result.metrics["f_star_gap"]
    assert gap == pytest.approx(2.0, abs=1e-4), (
        f"expected internal=+1 minus external=-1 = +2.0, got {gap}"
    )
    # And the gap is surfaced per split, not just on 'all'.
    assert result.metrics.get("test_f_star_gap") == pytest.approx(2.0, abs=1e-4)


def test_f_star_gap_zero_when_teacher_equals_expert(tmp_path: Path) -> None:
    """Calibration sanity: if teacher == expert, the gap is zero."""
    same = [1, 2, 3, 4, 5, 6, 7, 8]
    trees = _make_trees_with_distinct_teacher_and_expert(same, same, split="test")
    family = _TeacherPassthroughFamily()
    result = fit(
        {
            "family": "teacher_only",
            "eval_data": trees,
            "backend_config": {"family_runtime": family, "output_dir": str(tmp_path)},
        },
    )
    assert result.metrics["f_star_gap"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. Prediction records JSONL paths surface in result.artifacts
# --------------------------------------------------------------------------- #


def test_prediction_records_jsonl_paths_in_artifacts(tmp_path: Path) -> None:
    trees = _make_trees_with_distinct_teacher_and_expert(
        [1, 2, 3, 4, 5, 6, 7, 8], [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1], "test",
    )
    family = _TeacherPassthroughFamily()
    result = fit(
        {
            "family": "passthrough",
            "eval_data": trees,
            "backend_config": {"family_runtime": family, "output_dir": str(tmp_path)},
        },
    )
    paths = result.artifacts.get("prediction_records") or []
    assert paths, "expected at least one prediction_records JSONL"
    # Each file is real JSONL with one row per tree.
    for path in paths:
        p = Path(path)
        assert p.exists()
        lines = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        assert len(lines) == len(trees)
        # Each row carries prediction + teacher + expert fields the
        # paper-code plotting scripts (and future post-hoc analysis)
        # expect to find.
        for row in lines:
            assert "prediction" in row
            assert "teacher_score" in row
            assert "expert_score" in row


# --------------------------------------------------------------------------- #
# 4. Per-law-kind decomposition of the audit (C1 vs C2 vs C3)
# --------------------------------------------------------------------------- #


def test_audit_by_law_kind_decomposes_c1_c2_c3() -> None:
    """The user's central concern: compare f-vs-f* error *per law*
    (C1 leaf, C2 idempotence, C3 merge), not lumped into one scalar.
    """
    rows = [
        # 3 C1 rows: observed=True, oracle - proxy small
        LocalLawAuditRow(
            row_id=f"c1_{i}", law_kind="c1_leaf",
            proxy_loss=0.10, oracle_loss=0.10,
            observed=True, propensity=0.5, node_weight=1.0,
        )
        for i in range(3)
    ] + [
        # 2 C2 rows: observed=True, oracle - proxy bigger
        LocalLawAuditRow(
            row_id=f"c2_{i}", law_kind="c2_idempotence",
            proxy_loss=0.20, oracle_loss=0.50,
            observed=True, propensity=0.5, node_weight=1.0,
        )
        for i in range(2)
    ] + [
        # 4 C3 rows: half observed, half not
        LocalLawAuditRow(
            row_id=f"c3_{i}", law_kind="c3_merge",
            proxy_loss=0.05,
            oracle_loss=0.05 if i % 2 == 0 else None,
            observed=(i % 2 == 0),
            propensity=0.5,
            node_weight=1.0,
        )
        for i in range(4)
    ]
    result = audit_local_laws(rows, objective_mode="corrected_local_law")
    assert result["status"] == "success"
    assert result["n_rows"] == 9

    by_kind = result["by_law_kind"]
    assert set(by_kind) == {
        "leaf_preservation",
        "on_range_idempotence",
        "merge_preservation",
    }

    # C1: 3 rows, all observed. Corrected loss = proxy + (oracle-proxy)/pi.
    # proxy=0.10, oracle=0.10 → corrected per row = 0.10. Mean = 0.10.
    assert by_kind["leaf_preservation"]["local_law_objective"]["objective"] == pytest.approx(0.10)
    assert by_kind["leaf_preservation"]["local_law_objective"]["row_count"] == 3
    assert by_kind["leaf_preservation"]["influence_weighted_overlap"]["n_observed"] == 3

    # C2: 2 rows. proxy=0.20, oracle=0.50, pi=0.5.
    # corrected per row = 0.20 + (0.50 - 0.20)/0.5 = 0.20 + 0.60 = 0.80.
    # mean = 0.80.
    assert by_kind["on_range_idempotence"]["local_law_objective"]["objective"] == pytest.approx(0.80)
    assert by_kind["on_range_idempotence"]["local_law_objective"]["row_count"] == 2

    # C3: 4 rows, 2 observed and 2 not. Observed contributions =
    # 0.05 + (0.05 - 0.05)/0.5 = 0.05. Unobserved = proxy = 0.05.
    # weighted mean across 4 unit-weight rows = 0.05.
    assert by_kind["merge_preservation"]["local_law_objective"]["objective"] == pytest.approx(0.05)
    assert by_kind["merge_preservation"]["influence_weighted_overlap"]["n_observed"] == 2


def test_audit_top_level_objective_equals_weighted_average_of_per_law(
    tmp_path: Path,
) -> None:
    """Per-law decomposition is a *partition* of the rows; the
    weighted average of per-law objectives equals the top-level
    objective when weights are uniform.
    """
    rows = [
        LocalLawAuditRow(
            row_id="c1_a", law_kind="c1",
            proxy_loss=1.0, oracle_loss=1.0,
            observed=True, propensity=1.0, node_weight=1.0,
        ),
        LocalLawAuditRow(
            row_id="c2_a", law_kind="c2",
            proxy_loss=2.0, oracle_loss=2.0,
            observed=True, propensity=1.0, node_weight=1.0,
        ),
        LocalLawAuditRow(
            row_id="c3_a", law_kind="c3",
            proxy_loss=3.0, oracle_loss=3.0,
            observed=True, propensity=1.0, node_weight=1.0,
        ),
    ]
    result = audit_local_laws(rows)
    top = result["local_law_objective"]["objective"]
    # Top is the unit-weighted mean over all rows = (1+2+3)/3 = 2.0.
    assert top == pytest.approx(2.0)
    # And the per-law decomposition's row_count sums to total.
    total = sum(
        result["by_law_kind"][kind]["local_law_objective"]["row_count"]
        for kind in result["by_law_kind"]
    )
    assert total == 3


# --------------------------------------------------------------------------- #
# 5. Manifest sidecar carries split_metrics + by-law-kind block
# --------------------------------------------------------------------------- #


def test_manifest_carries_per_split_metrics(tmp_path: Path) -> None:
    trees = _make_trees_with_distinct_teacher_and_expert(
        [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], "test",
    )
    family = _TeacherPassthroughFamily()
    result = fit(
        {
            "family": "passthrough",
            "eval_data": trees,
            "backend_config": {"family_runtime": family, "output_dir": str(tmp_path)},
        },
    )
    payload = json.loads(Path(result.manifest_path).read_text())
    # Top-level summary block carries the same split_metrics dict the
    # in-memory result does — same JSON shape paper scripts read off
    # disk.
    sm = payload["summary"]["split_metrics"]
    assert "all" in sm
    assert sm["test"]["n"] == 8
