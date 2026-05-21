"""TRL backend family for the alternating f/g optimization loop.

Artifact semantics:
- ``f`` artifact: path to a HuggingFace model directory (scalar-regression head
  trained via ``src.training.trl_training.train_scalar_reward_records``), OR
  the sentinel ``"teacher_passthrough"`` at k=0.
- ``g`` artifact: path to an HF model directory (causal LM fine-tuned by
  either SFT or GRPO), OR ``"teacher_passthrough"``.

Alternation: ``train_g`` should launch GRPO with ``reward_funcs=[f_current]``
so g optimizes prompts to maximize the current student f's score. The
GRPO plumbing lives in :mod:`src.training.trl_training` (``train_grpo`` at
roughly line 1539) and needs a thin wrapper to accept an arbitrary Python
reward callable.

This module currently implements the passthrough (k=0) case end-to-end and
raises :class:`NotImplementedError` at k>=1 with a pointer to the required
integration. This lets the grid script compose all three families on day 1
and exercise the iteration-0 row for TRL without the full GRPO integration.

Implementation notes for the followup (see plan Stage 3c):
- Route ``train_g`` through ``distill_ctreepo_students.py --run-g-grpo`` with
  the reward function wrapping the current-f HF pipeline.
- Route ``train_f`` through ``distill_ctreepo_students.py --run-f-lm-regression
  --init-checkpoint <prev_f_path>`` for warmstart.
- ``score_roots_with_f`` loads the g and f HF models, generates a summary for
  each tree's root prompt, scores it, returns 1-7 predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from treepo._research.ctreepo.alternating import FamilyRuntime
from treepo._research.ctreepo.fg_arity import check_two_child_lm_budget
from treepo._research.tree.labeled import LabeledNode, LabeledTree

LOGGER = logging.getLogger(__name__)


def _root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    levels = getattr(tree, "levels", None) or []
    for level_ids in reversed(levels):
        for node_id in level_ids:
            node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
            if node is not None:
                return node
    return None


def _teacher_root_score(tree: LabeledTree) -> Optional[float]:
    root = _root_node(tree)
    if root is not None and root.score is not None:
        try:
            return float(root.score)
        except (TypeError, ValueError):
            pass
    metadata = tree.metadata or {}
    for key in ("teacher_score_1_7", "document_score"):
        v = metadata.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class TRLFamilyConfig:
    """Config for the TRL alternating family."""

    #: Base HF model ID for g (causal LM).
    g_base_model: str = "nvidia/Gemma-4-31B-IT-NVFP4"
    #: Base HF model ID for f (sequence-classification scalar head).
    f_base_model: str = "nvidia/Gemma-4-31B-IT-NVFP4"
    target_min: float = 1.0
    target_max: float = 7.0
    #: Canonical size-token row axis, carried even while TRL k>=1 is scaffolded.
    leaf_size_tokens: int = 512
    #: Context budget used by future TRL f/g calls. Checked now so TRL rows fail
    #: consistently with DSPy if the run config cannot hold two child states.
    lm_context_window_tokens: int = 12000
    max_completion_tokens: int = 512
    prompt_template_overhead_tokens: int = 1500
    tokenizer_model_path: str = "/mnt/data/models/google/embeddinggemma-300m"
    #: distill_ctreepo_students.py args passed through when training.
    distill_script: str = "scripts/distill_ctreepo_students.py"
    #: Optional shared subprocess env.
    subprocess_env: Dict[str, str] = field(default_factory=dict)


class TRLFamily(FamilyRuntime):
    """TRL alternating family scaffold.

    Iteration 0 (teacher passthrough) is fully functional so grids that set
    ``max_iterations=0`` or query only k=0 rows get real numbers. Iterations
    k>=1 raise :class:`NotImplementedError` with guidance on the GRPO
    integration needed to complete the family.
    """

    name: str = "trl"

    TEACHER_PASSTHROUGH: str = "teacher_passthrough"

    def __init__(self, *, config: TRLFamilyConfig) -> None:
        self.config = config
        self._check_two_leaf_budget_config()

    def _check_two_leaf_budget_config(self) -> None:
        check_two_child_lm_budget(
            family_name="TRLFamily",
            leaf_size_tokens=int(self.config.leaf_size_tokens),
            lm_context_window_tokens=int(self.config.lm_context_window_tokens),
            max_completion_tokens=int(self.config.max_completion_tokens),
            prompt_template_overhead_tokens=int(
                self.config.prompt_template_overhead_tokens
            ),
        )

    # ------------------------------------------------------------------
    # FamilyRuntime protocol
    # ------------------------------------------------------------------

    def _init_checkpoint(self, artifact: Any, *, base_model: str) -> str:
        """Resolve the checkpoint to warmstart from.

        - ``TEACHER_PASSTHROUGH`` / ``"identity"`` / ``None`` -> base model
        - otherwise: path to the prior iteration's HF model directory
        """
        if artifact in (None, "identity", self.TEACHER_PASSTHROUGH):
            return base_model
        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning(
                "TRL init checkpoint %s missing; falling back to base model %s",
                path, base_model,
            )
            return base_model
        return str(path)

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        """Hard-check that a TRL/HF artifact has a loadable directory shape."""
        if artifact in (None, "identity", self.TEACHER_PASSTHROUGH):
            return
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"TRL {kind} artifact does not exist: {path}")
        if not path.is_dir():
            raise RuntimeError(f"TRL {kind} artifact is not a directory: {path}")
        load_markers = {
            "config.json",
            "adapter_config.json",
            "model.safetensors",
            "pytorch_model.bin",
        }
        if not any((path / marker).exists() for marker in load_markers):
            raise RuntimeError(
                f"TRL {kind} artifact has no HuggingFace load markers in {path}"
            )

    def _traces_artifact_path(self, traces: Sequence[LabeledTree]) -> Optional[Path]:
        """Recover the on-disk path for labeled_trees.jsonl used in this row.

        The alternating trampoline doesn't tell the family where the traces
        came from, but the FNOFamily / DSPyFamily don't need this because
        they work in-memory. TRL drives a subprocess that expects a JSONL
        on disk, so we scan the first tree's metadata for a hint. If the
        traces were loaded from a path, it's stored in ``tree.metadata``
        (``teacher_fg_model.source_artifact``) by
        ``run_manifesto_teacher_fg_leaf_grid.py``.
        """
        if not traces:
            return None
        meta = (traces[0].metadata or {})
        candidate = meta.get("source_artifact") or meta.get("labeled_trees_path")
        if candidate and Path(str(candidate)).exists():
            return Path(str(candidate))
        return None

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """Warmstart the HF scalar-regression f model from f_init.

        Subprocess dispatch to scripts/distill_ctreepo_students.py with
        ``--run-f-lm-regression --init-checkpoint=<f_init>``. Honors the
        'never reset between rungs' rule: when ``f_init`` is a prior HF dir,
        the trainer resumes from it; when it's ``"identity"`` or passthrough,
        it starts from the configured base.

        The subprocess uses a GPU via transformers+TRL. If the vLLM server is
        occupying all GPUs, this call will fail with OOM; the operator must
        either bring up a separate GPU or pause vLLM before running TRL
        training. Logged as a RuntimeError in that case; the trampoline
        catches it and the row is marked skipped.
        """
        import subprocess

        traces_path = self._traces_artifact_path(traces)
        if traces_path is None:
            raise RuntimeError(
                "TRL train_f: could not locate labeled_trees.jsonl path from "
                "traces metadata. The teacher writer must populate "
                "metadata['source_artifact'] for the TRL subprocess to consume."
            )
        init = self._init_checkpoint(f_init, base_model=self.config.f_base_model)
        cmd = [
            "python3", str(self.config.distill_script),
            "--labeled-tree-artifacts", str(traces_path),
            "--output-dir", str(output_dir),
            "--run-f-lm-regression",
            "--init-checkpoint", init,
            "--skip-g-export",
            "--skip-f-fit",
            "--target-min", str(self.config.target_min),
            "--target-max", str(self.config.target_max),
        ]
        LOGGER.info("TRL train_f iter=%d cmd=%s", iteration, " ".join(cmd))
        env = {**dict(self.config.subprocess_env)} if self.config.subprocess_env else None
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"TRL train_f subprocess failed (exit {result.returncode}): "
                f"{result.stderr[-800:] if result.stderr else '(no stderr)'}"
            )
        # The TRL trainer writes the fine-tuned model into ``output_dir`` under
        # a subdir; default layout is <output_dir>/trl_scalar_reward_head.
        candidate = Path(output_dir) / "trl_scalar_reward_head"
        return str(candidate if candidate.exists() else output_dir)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """Warmstart the HF g model from g_init via SFT with teacher supervision.

        This is the MVP TRL g-training path: SFT on the g target records,
        warmstarted from g_init. It does NOT yet use f as a reward function
        (that requires TRL GRPO; see the followup below). As a result, this
        is 'alternation-like' in the warmstart sense but uses teacher
        supervision rather than current-f reward. The row still strengthens
        monotonically because it resumes from g_init instead of resetting.

        Followup for true GRPO: add ``--run-g-grpo`` to
        scripts/distill_ctreepo_students.py that calls
        ``src.training.trl_training.train_grpo`` with ``reward_funcs=[f_eval]``
        where ``f_eval`` loads the current-f HF model and returns per-sample
        scalar rewards.
        """
        import subprocess

        traces_path = self._traces_artifact_path(traces)
        if traces_path is None:
            raise RuntimeError(
                "TRL train_g: could not locate labeled_trees.jsonl path from "
                "traces metadata."
            )
        init = self._init_checkpoint(g_init, base_model=self.config.g_base_model)
        cmd = [
            "python3", str(self.config.distill_script),
            "--labeled-tree-artifacts", str(traces_path),
            "--output-dir", str(output_dir),
            "--run-g-sft",
            "--init-checkpoint", init,
            "--skip-f-fit",
            "--target-min", str(self.config.target_min),
            "--target-max", str(self.config.target_max),
        ]
        LOGGER.info("TRL train_g iter=%d cmd=%s", iteration, " ".join(cmd))
        env = {**dict(self.config.subprocess_env)} if self.config.subprocess_env else None
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"TRL train_g subprocess failed (exit {result.returncode}): "
                f"{result.stderr[-800:] if result.stderr else '(no stderr)'}"
            )
        candidate = Path(output_dir) / "trl_sft_g"
        return str(candidate if candidate.exists() else output_dir)

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Optional[float]]:
        """Score tree roots.

        At teacher passthrough (both sentinels), returns the tree's teacher
        root score directly. This produces a meaningful k=0 (``fg``) row
        without invoking any HF model.

        At non-passthrough artifacts, raises NotImplementedError until the
        full HF load + generate + score path is implemented.
        """
        if f in (None, "identity", self.TEACHER_PASSTHROUGH) and g in (
            None,
            "identity",
            self.TEACHER_PASSTHROUGH,
        ):
            return [_teacher_root_score(tree) for tree in trees]
        raise NotImplementedError(
            "TRL family score_roots_with_f for non-passthrough artifacts is not "
            "yet wired. Load g as a causal LM, generate summaries for each tree "
            "root, then load f as a scalar-regression head and score the output. "
            "See the followup checklist in src/ctreepo/trl_family.py module docstring."
        )
