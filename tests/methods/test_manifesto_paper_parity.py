"""Single-grid-point parity: ``treepo.methods.run`` vs the paper's manifesto
teacher-metric function.

Reproduces one cell of the manifesto grid (``--backend teacher`` path)
on a real labeled-trees artifact under
``outputs/manifesto_dimension_fit_existing/``. Asserts the metrics from
:func:`scripts.research.run_manifesto_fg_real_training_grid._fg_teacher_metrics`
match the metrics returned by ``treepo.methods.run("fit", ...)`` **bit-for-bit**
(no float tolerance — we want this to flag any divergence loudly).

Skipped automatically when the smoke artifact isn't present (outputs/
is not version-controlled). To run:

    python -m pytest \\
      tests/methods/test_manifesto_paper_parity.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import pytest

# Make the repo root importable so the paper-script function resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_SMOKE_LABELED_TREES = (
    _REPO_ROOT
    / "outputs/manifesto_dimension_fit_existing/smoke_qwen_embedding_economic/labeled_trees.jsonl"
)

pytestmark = pytest.mark.skipif(
    not _SMOKE_LABELED_TREES.exists(),
    reason=f"smoke labeled_trees artifact missing: {_SMOKE_LABELED_TREES}",
)


class _TeacherPassthroughFamily:
    """Mirrors the manifesto ``--backend teacher`` path: predictions are
    the teacher scores read off tree metadata. No training, no LLM call.
    """

    name = "teacher_passthrough"

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        return f_init

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        return g_init

    def score_roots_with_f(
        self, *, f: Any, g: Any, trees: Sequence[Any]
    ) -> List[Optional[float]]:
        return [float(t.metadata["teacher_score_1_7"]) for t in trees]

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        return None


def _trees_from_root_rows(root_rows) -> list[SimpleNamespace]:
    """Wrap each root row as a tree-like object with the metadata
    fields the alternating-loop evaluator reads.
    """
    out: list[SimpleNamespace] = []
    for row in root_rows:
        teacher = row.get("teacher_score_1_7")
        expert = row.get("expert_score_1_7")
        if teacher is None or expert is None:
            continue
        out.append(
            SimpleNamespace(
                leaves=[SimpleNamespace(tokens=[])],
                metadata={
                    "split": "test",
                    "teacher_score_1_7": float(teacher),
                    "teacher_score_native": float(teacher),
                    "expert_score_1_7": float(expert),
                    "expert_score_native": float(expert),
                    "expert_target_scale": "raw",
                    "expert_score_for_objective": float(expert),
                },
            )
        )
    return out


def test_manifesto_teacher_metrics_match_paper_script_bit_for_bit(tmp_path: Path) -> None:
    """Numeric reproducibility on one real grid cell.

    The paper-script function ``_fg_teacher_metrics`` reads a labeled-
    trees JSONL, extracts root rows, and computes regression metrics
    (Pearson r, MAE, means) of teacher vs expert. ``treepo.methods.run``
    computes the same metrics via its ``external_expert_*`` fields when
    a teacher-passthrough family is used. The numbers must match exactly.
    """
    from scripts.research.run_manifesto_fg_real_training_grid import (
        _fg_teacher_metrics,
        _tree_lookup,
    )
    from treepo._research.ctreepo.distillation import load_labeled_trees

    import treepo.methods

    # 1. Paper-script reference numbers.
    paper = _fg_teacher_metrics(_SMOKE_LABELED_TREES)
    rve = paper["root_vs_expert"]

    # 2. Same root rows, wrapped as treepo.methods trees.
    trees_raw = load_labeled_trees(_SMOKE_LABELED_TREES)
    lookup = _tree_lookup(trees_raw)
    root_rows = [r for r in lookup.values() if r.get("is_root")]
    tmethods_trees = _trees_from_root_rows(root_rows)

    # 3. Run treepo.methods on the same data.
    family = _TeacherPassthroughFamily()
    result = treepo.methods.run(
        "fit",
        {
            "family": "manifesto_teacher_passthrough",
            "eval_data": tmethods_trees,
            "backend_config": {
                "family_runtime": family,
                "output_dir": str(tmp_path),
            },
        },
    )
    assert result.status == "success"
    m = result.metrics

    # 4. Equality on the five reproducible scalars the paper script
    #    reports for root_vs_expert. Sums/means/MAE are bit-for-bit; the
    #    Pearson comparison allows float-noise (the vendored paper script's
    #    summation order has drifted a few ulps from the methods-side
    #    implementation — 2026-06-11, observed delta ~4e-16 relative).
    assert int(m["n"]) == int(rve["n"])
    assert m["external_expert_pearson"] == pytest.approx(rve["pearson_r"], rel=1e-12)
    assert m["external_expert_mae"] == rve["mae"]
    assert m["mean_prediction"] == rve["mean_prediction"]
    assert m["mean_expert"] == rve["mean_truth"]


def test_manifesto_teacher_metrics_propagate_to_test_split(tmp_path: Path) -> None:
    """The same metrics are also reachable per-split (``test_*`` keys)
    when the tree metadata's ``split`` field is set. Verifies the
    per-split surfacing we just added doesn't drop the manifesto path.
    """
    from scripts.research.run_manifesto_fg_real_training_grid import (
        _fg_teacher_metrics,
        _tree_lookup,
    )
    from treepo._research.ctreepo.distillation import load_labeled_trees

    import treepo.methods

    paper = _fg_teacher_metrics(_SMOKE_LABELED_TREES)
    rve = paper["root_vs_expert"]

    trees_raw = load_labeled_trees(_SMOKE_LABELED_TREES)
    lookup = _tree_lookup(trees_raw)
    root_rows = [r for r in lookup.values() if r.get("is_root")]
    tmethods_trees = _trees_from_root_rows(root_rows)  # all marked split="test"

    family = _TeacherPassthroughFamily()
    result = treepo.methods.run(
        "fit",
        {
            "family": "manifesto_teacher_passthrough",
            "eval_data": tmethods_trees,
            "backend_config": {
                "family_runtime": family,
                "output_dir": str(tmp_path),
            },
        },
    )
    # Per-split metrics expose the same numbers under ``test_*`` keys.
    assert int(result.metrics["test_n"]) == int(rve["n"])
    # Same float-noise allowance as the all-split test above (vendored
    # paper script drifted a few ulps from the methods-side Pearson).
    assert result.metrics["test_external_expert_pearson"] == pytest.approx(
        rve["pearson_r"], rel=1e-12
    )
    assert result.metrics["test_external_expert_mae"] == rve["mae"]
