"""Tree-structured DSPy program for RILE with per-node predictions.

Combines the existing manifesto DSPy signatures (`RILESummarize`,
`RILEMerge`, `RILEScoreSignature` from `src.tasks.manifesto.pipeline`)
into a single `RileTreeProgram(dspy.Module)` that, given a
`RILETreeScaffold`:

1. Applies the summarizer at every leaf → per-leaf summary
2. Applies the merger up a balanced binary tree → per-internal-node summary
3. Applies the scorer at every node (leaves + internals + root) → per-node RILE
4. Additionally applies the scorer to a swapped-merge probe at the root
   (merge(b, a) rather than merge(a, b)) for the C2 commutativity check.

Returns a `TreeRilePredictions` that the GEPA / GRPO metric consumes via
`tree_rile_reward_for_rollout`.

Usage — inside a trainer that has already configured `dspy.configure(lm=...)`:

    program = RileTreeProgram()
    rollout = program.rollout(scaffold)
    score_with_feedback = dspy_gepa_metric_from_rollout(rollout)

The program is a single `dspy.Module` with three `dspy.ChainOfThought`
sub-modules so GEPA can reflect on and rewrite all three sub-prompts
independently.
"""
from __future__ import annotations

from typing import Any

from treepo._research.unified_g_v1.realdoc.rile_tree import (
    PerNodeRilePrediction,
    RILETreeScaffold,
)
from treepo._research.unified_g_v1.training.trainers.rile_tree_feedback import TreeRilePredictions

try:  # pragma: no cover - exercised when dspy is installed.
    import dspy as _dspy_module
except ImportError:  # pragma: no cover - test environments may not ship dspy.
    _dspy_module = None


_DSPY_PROGRAM_BASE = _dspy_module.Module if _dspy_module is not None else object


def _lazy_dspy():
    import dspy

    return dspy


def _build_signatures():
    # Import here to defer the `dspy` + manifesto-pipeline dependency to
    # call time. The tests in this module don't need the real signatures.
    from treepo._research.tasks.manifesto.pipeline import (
        RILEMerge,
        RILEScoreSignature,
        RILESummarize,
    )
    from treepo._research.tasks.manifesto.rubrics import (
        RILE_PRESERVATION_RUBRIC,
        RILE_TASK_CONTEXT,
    )

    return (
        RILESummarize,
        RILEMerge,
        RILEScoreSignature,
        RILE_PRESERVATION_RUBRIC,
        RILE_TASK_CONTEXT,
    )


class RileTreeProgram(_DSPY_PROGRAM_BASE):
    """Tree-structured DSPy program over leaves, merges, and root.

    Not a `dspy.Module` directly — constructing one requires the real
    `dspy` package on the import path. Instantiate inside a trainer that
    has already run `dspy.configure(lm=...)`. Each attribute below is a
    `dspy.ChainOfThought(Signature)` instance.
    """

    def __init__(self) -> None:
        dspy = _lazy_dspy()
        if _dspy_module is not None:
            super().__init__()
        (
            RILESummarize,
            RILEMerge,
            RILEScoreSignature,
            rubric,
            task_context,
        ) = _build_signatures()
        self.summarizer = dspy.ChainOfThought(RILESummarize)
        self.merger = dspy.ChainOfThought(RILEMerge)
        self.scorer = dspy.ChainOfThought(RILEScoreSignature)
        self._rubric = rubric
        self._task_context = task_context

    # ----------------------------- helpers --------------------------------

    def summarize(self, text: str) -> str:
        out = self.summarizer(rubric=self._rubric, text=str(text))
        return str(getattr(out, "summary", "") or "")

    def merge(self, left_summary: str, right_summary: str) -> str:
        out = self.merger(
            rubric=self._rubric,
            summary1=str(left_summary),
            summary2=str(right_summary),
        )
        return str(getattr(out, "merged_summary", "") or "")

    def score(self, summary: str) -> float:
        out = self.scorer(task_context=self._task_context, summary=str(summary))
        raw = getattr(out, "score", None)
        try:
            return max(-100.0, min(100.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0

    # ----------------------------- rollout --------------------------------

    def forward(self, scaffold: RILETreeScaffold) -> TreeRilePredictions:
        return self.rollout(scaffold)

    def rollout(self, scaffold: RILETreeScaffold) -> TreeRilePredictions:
        """Run the full tree on one doc.

        Returns per-leaf, per-merge, and root RILE predictions, plus a
        single swapped-merge probe at the root for C2.
        """
        # Leaves: summarize + score.
        leaf_summaries: list[str] = []
        leaf_predictions: list[PerNodeRilePrediction] = []
        for leaf in scaffold.leaves:
            summary = self.summarize(leaf.text)
            leaf_summaries.append(summary)
            leaf_predictions.append(
                PerNodeRilePrediction(
                    node_index=int(leaf.index),
                    predicted_rile=float(self.score(summary)),
                )
            )

        # Internals: merge according to the scaffold's precomputed order,
        # then score. The scaffold's `internals` already carries the
        # balanced-binary-tree pairing.
        n_leaves = len(scaffold.leaves)
        all_summaries: dict[int, str] = {
            int(leaf.index): leaf_summaries[i]
            for i, leaf in enumerate(scaffold.leaves)
        }
        merge_predictions: list[PerNodeRilePrediction] = []
        for internal in scaffold.internals:
            left_sum = all_summaries.get(int(internal.left), "")
            right_sum = all_summaries.get(int(internal.right), "")
            merged = self.merge(left_sum, right_sum)
            all_summaries[int(internal.index)] = merged
            merge_predictions.append(
                PerNodeRilePrediction(
                    node_index=int(internal.index),
                    predicted_rile=float(self.score(merged)),
                )
            )

        # Root prediction = the last internal's score (or the single leaf's
        # score if there are no merges).
        if scaffold.internals:
            root_prediction = merge_predictions[-1].predicted_rile
        elif leaf_predictions:
            root_prediction = leaf_predictions[-1].predicted_rile
        else:
            root_prediction = 0.0

        # C2 probe: swap the root merge's inputs and re-score. Only
        # meaningful when there's at least one internal node.
        commutativity_pairs: list[tuple[float, float]] = []
        if scaffold.internals:
            root_internal = scaffold.internals[-1]
            left_sum = all_summaries.get(int(root_internal.left), "")
            right_sum = all_summaries.get(int(root_internal.right), "")
            swapped_merge = self.merge(right_sum, left_sum)
            swapped_score = float(self.score(swapped_merge))
            commutativity_pairs.append(
                (float(root_prediction), float(swapped_score))
            )

        return TreeRilePredictions(
            scaffold=scaffold,
            root_prediction=float(root_prediction),
            leaf_predictions=tuple(leaf_predictions),
            merge_predictions=tuple(merge_predictions),
            commutativity_pairs=tuple(commutativity_pairs),
        )
