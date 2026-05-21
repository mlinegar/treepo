"""Joint six-dimension DSPy family for the alternating f/g ladder.

This is the vector analogue of :mod:`src.ctreepo.dspy_family`: one ``g``
program produces summaries under the Manifesto joint rubric, and one shared
``JointDimensionScorer`` scores those summaries across all Benoit dimensions.
The alternating trampoline receives macro metrics, while per-dimension metrics
are preserved in ``SplitMetrics.per_dimension``.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from treepo._research.ctreepo.dspy_family import (
    DSPyFamily,
    DSPyFamilyConfig,
    _clamp01,
    _parse_first_float,
    _root_node,
)
from treepo._research.tree.labeled import LabeledNode, LabeledTree

LOGGER = logging.getLogger(__name__)


DIMENSION_ORDER = (
    "economic",
    "social",
    "immigration",
    "eu",
    "environment",
    "decentralization",
)


@dataclass
class JointDSPyFamilyConfig(DSPyFamilyConfig):
    """Config for a shared six-dimension DSPy f/g family."""

    dimension: str = "combined"
    dimensions: Sequence[str] = field(default_factory=lambda: DIMENSION_ORDER)
    #: Default warm-start for the shared f scorer. Empty string forces a bare
    #: ``JointDimensionScorer``.
    f_init_path: Optional[str] = "outputs/phase2/joint_gepa/optimized_program.json"


def _summary_target(node: LabeledNode, *, include_identity_targets: bool = False) -> Optional[str]:
    metadata = dict(node.metadata or {})
    for key in (
        "teacher_summary",
        "teacher_leaf_summary",
        "teacher_merge_summary",
        "target_summary",
        "summary",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if include_identity_targets and str(node.text or "").strip():
        return str(node.text)
    return None


def _finite_mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _node_dimension_scores(node: LabeledNode) -> Dict[str, float]:
    out: Dict[str, float] = {}
    raw = getattr(node, "dimension_scores", None)
    if isinstance(raw, Mapping):
        for dim, value in raw.items():
            parsed = _parse_first_float(value)
            if parsed is not None:
                out[str(dim)] = float(parsed)
    metadata = dict(node.metadata or {})
    for key in ("teacher_dimension_scores_1_7", "dimension_scores"):
        raw_meta = metadata.get(key)
        if isinstance(raw_meta, Mapping):
            for dim, value in raw_meta.items():
                parsed = _parse_first_float(value)
                if parsed is not None:
                    out.setdefault(str(dim), float(parsed))
    return out


def _tree_teacher_dimension_scores(tree: LabeledTree) -> Dict[str, float]:
    root = _root_node(tree)
    if root is not None:
        scores = _node_dimension_scores(root)
        if scores:
            return scores
    metadata = dict(tree.metadata or {})
    raw = metadata.get("teacher_dimension_scores_1_7") or metadata.get("dimension_scores")
    if isinstance(raw, Mapping):
        return {
            str(dim): float(value)
            for dim, value in raw.items()
            if _parse_first_float(value) is not None
        }
    return {}


def _tree_expert_dimension_scores(tree: LabeledTree) -> Dict[str, float]:
    metadata = dict(tree.metadata or {})
    raw = metadata.get("expert_dimension_scores_1_7") or metadata.get("expert_means")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(dim): float(value)
        for dim, value in raw.items()
        if _parse_first_float(value) is not None
    }


class JointDSPyFamily(DSPyFamily):
    """DSPy alternating family with one joint g and one shared vector f."""

    name: str = "dspy"

    def __init__(self, *, config: JointDSPyFamilyConfig) -> None:
        super().__init__(config=config)
        self.config: JointDSPyFamilyConfig = config

    def _active_dimensions(self) -> List[str]:
        dims = [str(dim) for dim in self.config.dimensions if str(dim)]
        return [dim for dim in DIMENSION_ORDER if dim in set(dims)]

    def _g_signature(self):
        import dspy

        from treepo._research.tasks.manifesto.dimensions import get_joint_rubric

        instructions = (
            "Generate a summary of the given political text that preserves all "
            "information relevant to all six Benoit manifesto policy dimensions. "
            "The same summary will later be scored separately on economic, social, "
            "immigration, EU, environment, and decentralization scales, so keep "
            "concrete evidence for every dimension and do not collapse the text "
            "to a single left-right score.\n\n"
            f"{get_joint_rubric()}"
        )

        class CTreePOJointGSignature(dspy.Signature):
            __doc__ = instructions

            prompt: str = dspy.InputField(desc="Input text or child summaries to summarize")
            completion: str = dspy.OutputField(desc="All-dimension score-preserving summary")

        return CTreePOJointGSignature

    @staticmethod
    def _leaf_reduction_prompt(node: LabeledNode) -> str:
        return (
            "Summarize the following leaf span for all-dimension "
            "score-preserving C-TreePO distillation.\n\nLEAF:\n"
            f"{node.text}"
        )

    @staticmethod
    def _merge_reduction_prompt(left_state: str, right_state: str) -> str:
        return (
            "Merge the two child summaries into one all-dimension "
            "score-preserving C-TreePO parent summary.\n\nLEFT:\n"
            f"{left_state}\n\nRIGHT:\n{right_state}"
        )

    def _default_f_init_path(self) -> Optional[Path]:
        if self.config.f_init_path is not None:
            if not self.config.f_init_path:
                return None
            return Path(self.config.f_init_path)
        return None

    def _new_joint_scorer(self, *, max_output_tokens: Optional[int] = None):
        from treepo._research.tasks.manifesto.joint_scorer import JointDimensionScorer

        return JointDimensionScorer(
            use_cot=False,
            max_output_tokens=max_output_tokens or int(self.config.max_completion_tokens),
        )

    def _load_f_program(self, artifact: Any):
        if artifact == self.TEACHER_PASSTHROUGH:
            return self.TEACHER_PASSTHROUGH
        if artifact == "identity":
            LOGGER.warning(
                "Legacy joint DSPy f_init='identity' means pretuned_scorer, not "
                "true identity/teacher passthrough."
            )
            artifact = self.PRETUNED_SCORER
        if artifact == self.BARE_SCORER:
            return self._new_joint_scorer()
        if artifact in (None, self.PRETUNED_SCORER):
            scorer = self._new_joint_scorer()
            default_path = self._default_f_init_path()
            if default_path is not None and default_path.exists():
                try:
                    scorer.load(str(default_path))
                    LOGGER.info("Loaded joint scorer from %s", default_path)
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to load joint scorer from %s: %s; using bare JointDimensionScorer",
                        default_path,
                        exc,
                    )
            return scorer

        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning("Joint DSPy f artifact %s missing; using bare scorer", path)
            return self._new_joint_scorer()
        if path.is_dir() and (path / "program.pkl").exists():
            import dspy

            return dspy.load(str(path))
        scorer = self._new_joint_scorer()
        scorer.load(str(path))
        return scorer

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if artifact in (
            None,
            "identity",
            self.RAW_CONCAT,
            self.TEACHER_PASSTHROUGH,
            self.PRETUNED_SCORER,
            self.BARE_SCORER,
        ):
            return
        kind = str(kind)
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"Joint DSPy {kind} artifact does not exist: {path}")
        if kind == "f":
            if path.is_dir():
                if not (path / "program.pkl").exists():
                    raise RuntimeError(
                        f"Joint DSPy f program directory is missing program.pkl: {path}"
                    )
                import dspy

                loaded = dspy.load(str(path))
                if not callable(loaded):
                    raise RuntimeError(f"Joint DSPy f program is not callable: {path}")
                return
            scorer = self._new_joint_scorer()
            scorer.load(str(path))
            if not (
                callable(getattr(scorer, "scorer", None))
                or callable(getattr(scorer, "score", None))
            ):
                raise RuntimeError(f"Joint DSPy f state is not callable: {path}")
            return
        if kind == "g":
            return super().validate_artifact(kind=kind, artifact=artifact)
        raise ValueError(f"unknown DSPy artifact kind: {kind!r}")

    def _task_context(self, dim: str) -> str:
        from treepo._research.tasks.manifesto.dimensions import PolicyDimension
        from treepo._research.tasks.manifesto.scoring_contexts import get_scoring_context

        return str(get_scoring_context(PolicyDimension(str(dim))))

    def _apply_f_raw_for_dim(
        self,
        f_program: Any,
        *,
        response: str,
        dim: str,
    ) -> Optional[float]:
        if f_program == self.TEACHER_PASSTHROUGH:
            return None
        self._assert_lm_input_budget(
            label=f"joint f inference ({dim})",
            fields={"task_context": self._task_context(dim), "summary": response},
        )
        import dspy

        lm = self._ensure_lm()
        try:
            with dspy.context(lm=lm):
                result = f_program(summary=response, task_context=self._task_context(dim))
            raw = result.get("score") if isinstance(result, dict) else getattr(result, "score", None)
        except Exception as exc:
            LOGGER.warning("JointDimensionScorer call failed for %s: %s", dim, exc)
            return None
        parsed = _parse_first_float(raw)
        if parsed is None:
            return None
        return max(float(self.config.target_min), min(float(self.config.target_max), float(parsed)))

    def _apply_f_normalized_for_dim(
        self,
        f_program: Any,
        *,
        response: str,
        dim: str,
    ) -> Optional[float]:
        raw = self._apply_f_raw_for_dim(f_program, response=response, dim=dim)
        if raw is None:
            return None
        span = max(1e-9, float(self.config.target_max) - float(self.config.target_min))
        return _clamp01((float(raw) - float(self.config.target_min)) / span)

    def _joint_f_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        active = set(self._active_dimensions())
        span = max(1e-9, float(self.config.target_max) - float(self.config.target_min))
        for tree in trees:
            split = str((tree.metadata or {}).get("split", "") or "")
            for node in tree.nodes.values():
                target_text = _summary_target(
                    node,
                    include_identity_targets=bool(self.config.include_identity_targets),
                )
                if not target_text:
                    continue
                scores = _node_dimension_scores(node)
                for dim in self._active_dimensions():
                    if dim not in active or dim not in scores:
                        continue
                    raw = float(scores[dim])
                    records.append(
                        {
                            "response": str(target_text),
                            "dimension": dim,
                            "task_context": self._task_context(dim),
                            "score": raw,
                            "score_normalized": _clamp01((raw - float(self.config.target_min)) / span),
                            "metadata": {
                                "doc_id": tree.doc_id,
                                "node_id": node.node_id,
                                "split": split,
                                "level": int(node.level),
                                "law_role": "leaf_f" if int(node.level) == 0 else "merge_f",
                                "target_score_raw": raw,
                                "target_dimension": dim,
                            },
                        }
                    )
        return records

    def _joint_g_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        active = set(self._active_dimensions())
        for tree in trees:
            split = str((tree.metadata or {}).get("split", "") or "")
            for node in tree.nodes.values():
                target = _summary_target(
                    node,
                    include_identity_targets=bool(self.config.include_identity_targets),
                )
                if not target:
                    continue
                scores = {
                    dim: float(value)
                    for dim, value in _node_dimension_scores(node).items()
                    if dim in active
                }
                if not scores:
                    continue
                if int(node.level) == 0:
                    prompt = (
                        "Summarize the following leaf span for all-dimension "
                        "score-preserving C-TreePO distillation.\n\nLEAF:\n"
                        f"{node.text}"
                    )
                else:
                    left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
                    right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
                    left_text = _summary_target(left, include_identity_targets=True) if left is not None else ""
                    right_text = _summary_target(right, include_identity_targets=True) if right is not None else ""
                    prompt = (
                        "Merge the two child summaries into one all-dimension "
                        "score-preserving C-TreePO parent summary.\n\nLEFT:\n"
                        f"{left_text}\n\nRIGHT:\n{right_text}"
                    )
                records.append(
                    {
                        "prompt": prompt,
                        "completion": target,
                        "metadata": {
                            "doc_id": tree.doc_id,
                            "node_id": node.node_id,
                            "split": split,
                            "level": int(node.level),
                            "is_leaf": int(node.level) == 0,
                            "target_dimension_scores_raw": scores,
                        },
                    }
                )
        return records

    def _check_joint_f_record_budgets(self, records: Sequence[Mapping[str, Any]]) -> None:
        for idx, row in enumerate(records):
            self._assert_lm_input_budget(
                label=f"joint f training record {idx}",
                fields={
                    "task_context": str(row.get("task_context") or ""),
                    "summary": str(row.get("response") or ""),
                },
            )

    def _check_joint_g_record_budgets(self, records: Sequence[Mapping[str, Any]]) -> None:
        for idx, row in enumerate(records):
            self._assert_lm_input_budget(
                label=f"joint g training record {idx}",
                fields={
                    "prompt": str(row.get("prompt") or ""),
                    "completion": str(row.get("completion") or ""),
                },
            )

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        import dspy

        f_program = self._load_f_program(f_init)
        if f_program == self.TEACHER_PASSTHROUGH:
            f_program = self._load_f_program("identity")

        records = self._joint_f_records(traces)
        self._check_joint_f_record_budgets(records)
        train_examples = [
            dspy.Example(
                summary=str(row.get("response") or ""),
                task_context=str(row.get("task_context") or ""),
                score=str(float(row.get("score", 4.0))),
            ).with_inputs("summary", "task_context")
            for row in records
        ]
        if not train_examples:
            LOGGER.warning("No joint f training examples; skipping compile")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = Path(output_dir) / "f_dspy_noop.json"
            path.write_text("{}\n", encoding="utf-8")
            return str(path)

        lo = float(self.config.target_min)
        hi = float(self.config.target_max)
        span = max(1e-9, hi - lo)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            target = _parse_first_float(getattr(gold, "score", None))
            raw_score = getattr(pred, "score", None)
            if raw_score is None and isinstance(pred, dict):
                raw_score = pred.get("score")
            predicted_raw = _parse_first_float(raw_score)
            if target is None or predicted_raw is None:
                return 0.0
            return max(0.0, 1.0 - abs(float(predicted_raw) - float(target)) / span)

        lm = self._ensure_lm()
        with dspy.context(lm=lm):
            compiled = self._compile(
                program=f_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = Path(output_dir) / f"f_joint_dspy_iter_{iteration:02d}"
        compiled.save(str(artifact_path), save_program=True)
        return str(artifact_path)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        import dspy

        f_program = self._load_f_program(f)
        if f_program == self.TEACHER_PASSTHROUGH:
            f_program = self._load_f_program("identity")

        g_program = self._load_g_program(g_init)
        if g_program in (self.TEACHER_PASSTHROUGH, self.RAW_CONCAT):
            g_program = dspy.Predict(self._g_signature())

        records = self._joint_g_records(traces)
        self._check_joint_g_record_budgets(records)
        train_examples = [
            dspy.Example(
                prompt=str(row.get("prompt") or ""),
                completion=str(row.get("completion") or ""),
                node_id=str((row.get("metadata") or {}).get("node_id") or ""),
                target_scores_json=json.dumps(
                    (row.get("metadata") or {}).get("target_dimension_scores_raw") or {},
                    sort_keys=True,
                ),
            ).with_inputs("prompt")
            for row in records
        ]
        if not train_examples:
            LOGGER.warning("No joint g training examples; skipping compile")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = Path(output_dir) / "g_dspy_noop.json"
            path.write_text("{}\n", encoding="utf-8")
            return str(path)

        lo = float(self.config.target_min)
        hi = float(self.config.target_max)
        span = max(1e-9, hi - lo)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            summary = str(getattr(pred, "completion", "") or "")
            if not summary:
                return 0.0
            try:
                targets = json.loads(str(getattr(gold, "target_scores_json", "{}") or "{}"))
            except json.JSONDecodeError:
                targets = {}
            rewards: List[float] = []
            for dim in self._active_dimensions():
                if dim not in targets:
                    continue
                predicted_norm = self._apply_f_normalized_for_dim(
                    f_program,
                    response=summary,
                    dim=dim,
                )
                target_raw = _parse_first_float(targets.get(dim))
                if predicted_norm is None or target_raw is None:
                    continue
                target_norm = _clamp01((float(target_raw) - lo) / span)
                rewards.append(max(0.0, 1.0 - abs(float(predicted_norm) - target_norm)))
            return float(sum(rewards) / len(rewards)) if rewards else 0.0

        lm = self._ensure_lm()
        with dspy.context(lm=lm):
            compiled = self._compile(
                program=g_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = Path(output_dir) / f"g_joint_dspy_iter_{iteration:02d}.json"
        compiled.save(str(artifact_path))
        return str(artifact_path)

    def score_roots_with_f_by_dimension(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> Dict[str, List[Optional[float]]]:
        f_program = self._load_f_program(f)
        g_program = self._load_g_program(g)
        active_dims = self._active_dimensions()
        out: Dict[str, List[Optional[float]]] = {dim: [] for dim in active_dims}

        def score_tree(tree: LabeledTree) -> Dict[str, Optional[float]]:
            summary = self._reduce_tree_with_g(g_program, tree)
            teacher_scores = _tree_teacher_dimension_scores(tree)
            row: Dict[str, Optional[float]] = {}
            for dim in active_dims:
                if f_program == self.TEACHER_PASSTHROUGH or not summary:
                    row[dim] = teacher_scores.get(dim)
                else:
                    row[dim] = self._apply_f_raw_for_dim(
                        f_program, response=summary, dim=dim
                    )
            return row

        tree_list = list(trees)
        max_workers = max(1, min(len(tree_list), int(self.config.num_threads or 1)))
        if max_workers == 1:
            rows = [score_tree(tree) for tree in tree_list]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                rows = list(pool.map(score_tree, tree_list))
        for row in rows:
            for dim in active_dims:
                out[dim].append(row.get(dim))
        return out

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Optional[float]]:
        by_dim = self.score_roots_with_f_by_dimension(f=f, g=g, trees=trees)
        rows: List[Optional[float]] = []
        for idx in range(len(trees)):
            values = [
                float(preds[idx])
                for preds in by_dim.values()
                if idx < len(preds) and preds[idx] is not None
            ]
            rows.append(_finite_mean(values))
        return rows
