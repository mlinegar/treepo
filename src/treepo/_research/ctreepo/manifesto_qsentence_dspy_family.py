"""DSPy f/g family for Manifesto Project quasi-sentence supervision.

This family keeps the alternating ladder contract from ``DSPyFamily`` and
``JointDSPyFamily`` but swaps the supervision source: every labeled node target
is derived exactly from descendant Manifesto Project CMP quasi-sentence labels.
The learned ``g`` program produces compact CMP policy states; the learned ``f``
program predicts compact aggregate targets from those states.
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
    _root_text,
)
from treepo._research.tasks.manifesto.span_targets import (
    COMPACT_TARGET_DIMENSIONS,
    parse_compact_scores_json,
)
from treepo._research.tree.labeled import LabeledNode, LabeledTree

LOGGER = logging.getLogger(__name__)


try:  # Keep import of this module cheap for tests that do not exercise DSPy.
    import dspy as _dspy
except Exception:  # pragma: no cover - exercised only in non-DSPy envs.
    _dspy = None


QSENTENCE_TARGET_SIGNATURE_DOC = (
    "Predict the compact Manifesto Project CMP target vector from a candidate "
    "policy state summary. Return only a JSON object with keys "
    "`rile`, `domain_1`, `domain_2`, `domain_3`, `domain_4`, `domain_5`, "
    "`domain_6`, and `domain_7`. Each value must be a number in [0, 1]. "
    "`rile` is normalized from raw RILE [-100,100] into [0,1]; each domain "
    "value is the share of non-headline quasi-sentences in that CMP domain."
)


if _dspy is not None:

    class ManifestoQSentenceTargetSignature(_dspy.Signature):
        __doc__ = QSENTENCE_TARGET_SIGNATURE_DOC

        summary: str = _dspy.InputField(desc="Compact policy state or summary to score")
        scores_json: str = _dspy.OutputField(
            desc=(
                "Strict JSON object with numeric [0,1] keys: rile, domain_1, "
                "domain_2, domain_3, domain_4, domain_5, domain_6, domain_7"
            )
        )


    class ManifestoQSentenceTargetScorer(_dspy.Module):
        """DSPy module for compact quasi-sentence aggregate target prediction."""

        def __init__(self, *, max_output_tokens: int = 256) -> None:
            super().__init__()
            self.max_output_tokens = int(max_output_tokens)
            self.predictor = _dspy.Predict(ManifestoQSentenceTargetSignature)

        def load_state(self, state: Any) -> None:
            compat_state = dict(state)
            if "predictor" not in compat_state and "scores_json" in compat_state:
                compat_state["predictor"] = compat_state["scores_json"]
            super().load_state(compat_state)

        def forward(self, summary: str) -> dict[str, Any]:
            predictor = getattr(self, "predictor", None)
            if not callable(predictor):
                predictor = _dspy.Predict(ManifestoQSentenceTargetSignature)
                self.predictor = predictor
            result = predictor(
                summary=str(summary or ""),
                config={"max_tokens": int(self.max_output_tokens)},
            )
            raw = str(getattr(result, "scores_json", "") or "")
            scores = parse_compact_scores_json(raw)
            if not scores:
                scores = parse_compact_scores_json(str(result))
            return {
                "scores_json": json.dumps(scores, sort_keys=True) if scores else raw,
                "scores": scores,
            }

else:
    ManifestoQSentenceTargetSignature = None

    class ManifestoQSentenceTargetScorer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("dspy is required for ManifestoQSentenceTargetScorer")


@dataclass
class ManifestoQSentenceDSPyFamilyConfig(DSPyFamilyConfig):
    """Config for the quasi-sentence compact-target DSPy family."""

    dimension: str = "manifesto_qsentence"
    target_dimensions: Sequence[str] = field(
        default_factory=lambda: tuple(COMPACT_TARGET_DIMENSIONS)
    )
    # Empty default means "start from a bare compact-target scorer" rather than
    # inheriting the Benoit dimension scorer warm start from ``DSPyFamily``.
    f_init_path: Optional[str] = ""


def _active_target_dimensions(config: ManifestoQSentenceDSPyFamilyConfig) -> List[str]:
    requested = [str(dim) for dim in config.target_dimensions if str(dim)]
    requested_set = set(requested)
    return [dim for dim in COMPACT_TARGET_DIMENSIONS if dim in requested_set]


def _summary_target(
    node: Optional[LabeledNode],
    *,
    include_identity_targets: bool = False,
) -> Optional[str]:
    if node is None:
        return None
    metadata = dict(node.metadata or {})
    for key in (
        "target_summary",
        "teacher_summary",
        "teacher_leaf_summary",
        "teacher_merge_summary",
        "summary",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if include_identity_targets and str(node.text or "").strip():
        return str(node.text)
    return None


def _prediction_scores(pred: Any) -> Dict[str, float]:
    if isinstance(pred, Mapping):
        if isinstance(pred.get("scores"), Mapping):
            parsed = parse_compact_scores_json(pred.get("scores"))
            if parsed:
                return parsed
        parsed = parse_compact_scores_json(pred.get("scores_json"))
        if parsed:
            return parsed
        return parse_compact_scores_json(pred)
    raw_scores = getattr(pred, "scores", None)
    if isinstance(raw_scores, Mapping):
        parsed = parse_compact_scores_json(raw_scores)
        if parsed:
            return parsed
    parsed = parse_compact_scores_json(getattr(pred, "scores_json", None))
    if parsed:
        return parsed
    return parse_compact_scores_json(str(pred))


def _node_target_scores(node: Optional[LabeledNode]) -> Dict[str, float]:
    if node is None:
        return {}
    out: Dict[str, float] = {}
    raw = getattr(node, "dimension_scores", None)
    if isinstance(raw, Mapping):
        for dim, value in raw.items():
            parsed = _parse_first_float(value)
            if parsed is not None:
                out[str(dim)] = _clamp01(float(parsed))
    metadata = dict(node.metadata or {})
    for key in ("target_dimension_scores_0_1", "teacher_dimension_scores_1_7", "dimension_scores"):
        raw_meta = metadata.get(key)
        if isinstance(raw_meta, Mapping):
            for dim, value in raw_meta.items():
                parsed = _parse_first_float(value)
                if parsed is not None:
                    out.setdefault(str(dim), _clamp01(float(parsed)))
    if not out:
        summary = _summary_target(node, include_identity_targets=False)
        out.update(parse_compact_scores_json(summary))
    return out


def _tree_target_scores(tree: LabeledTree) -> Dict[str, float]:
    root = _root_node(tree)
    if root is not None:
        scores = _node_target_scores(root)
        if scores:
            return scores
    metadata = dict(tree.metadata or {})
    for key in ("target_dimension_scores_0_1", "teacher_dimension_scores_1_7", "dimension_scores"):
        raw = metadata.get(key)
        if isinstance(raw, Mapping):
            parsed = parse_compact_scores_json(raw)
            if parsed:
                return parsed
            out = {}
            for dim, value in raw.items():
                parsed_value = _parse_first_float(value)
                if parsed_value is not None:
                    out[str(dim)] = _clamp01(float(parsed_value))
            if out:
                return out
    score = _parse_first_float(getattr(tree, "document_score", None))
    return {"rile": _clamp01(float(score))} if score is not None else {}


def _scores_json(scores: Mapping[str, Any], dims: Sequence[str]) -> str:
    payload = {}
    for dim in dims:
        parsed = _parse_first_float(scores.get(dim))
        if parsed is not None:
            payload[str(dim)] = _clamp01(float(parsed))
    return json.dumps(payload, sort_keys=True)


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


class ManifestoQSentenceDSPyFamily(DSPyFamily):
    """Alternating DSPy family for compact CMP quasi-sentence targets."""

    name: str = "dspy"

    def __init__(self, *, config: ManifestoQSentenceDSPyFamilyConfig) -> None:
        super().__init__(config=config)
        self.config: ManifestoQSentenceDSPyFamilyConfig = config

    def _active_dimensions(self) -> List[str]:
        return _active_target_dimensions(self.config)

    def _g_signature(self):
        import dspy

        instructions = (
            "Generate a compact CMP policy state for Manifesto Project "
            "quasi-sentence distillation. Inputs are either raw quasi-sentence "
            "text spans or child CMP policy states. Preserve enough information "
            "for a downstream scorer to recover normalized RILE and the salience "
            "shares for CMP domains 1 through 7. Prefer concise JSON-like state "
            "over prose. Do not invent policy evidence not supported by the input."
        )

        class CTreePOQSentenceGSignature(dspy.Signature):
            __doc__ = instructions

            prompt: str = dspy.InputField(desc="Raw quasi-sentence span or child states")
            completion: str = dspy.OutputField(desc="Compact CMP policy-bearing state")

        return CTreePOQSentenceGSignature

    def _new_target_scorer(self, *, max_output_tokens: Optional[int] = None) -> Any:
        return ManifestoQSentenceTargetScorer(
            max_output_tokens=max_output_tokens or int(self.config.max_completion_tokens)
        )

    def _default_f_init_path(self) -> Optional[Path]:
        if self.config.f_init_path is None or not str(self.config.f_init_path):
            return None
        return Path(str(self.config.f_init_path))

    def _load_f_program(self, artifact: Any) -> Any:
        if artifact == self.TEACHER_PASSTHROUGH:
            return self.TEACHER_PASSTHROUGH
        if artifact in (None, "identity"):
            scorer = self._new_target_scorer()
            default_path = self._default_f_init_path()
            if default_path is not None and default_path.exists():
                if default_path.is_dir() and (default_path / "program.pkl").exists():
                    import dspy

                    return dspy.load(str(default_path))
                try:
                    scorer.load(str(default_path))
                    LOGGER.info("Loaded q-sentence compact scorer from %s", default_path)
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to load q-sentence compact scorer from %s: %s; "
                        "using a bare scorer",
                        default_path,
                        exc,
                    )
            return scorer
        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning(
                "Q-sentence DSPy f artifact %s missing; using a bare scorer", path
            )
            return self._new_target_scorer()
        if path.is_dir() and (path / "program.pkl").exists():
            import dspy

            return dspy.load(str(path))
        scorer = self._new_target_scorer()
        scorer.load(str(path))
        return scorer

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if artifact in (None, "identity", self.TEACHER_PASSTHROUGH):
            return
        kind = str(kind)
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"Q-sentence DSPy {kind} artifact does not exist: {path}")
        if kind == "f":
            if path.is_dir():
                if not (path / "program.pkl").exists():
                    raise RuntimeError(
                        f"Q-sentence DSPy f program directory is missing program.pkl: {path}"
                    )
                import dspy

                loaded = dspy.load(str(path))
                if not callable(loaded):
                    raise RuntimeError(f"Q-sentence DSPy f program is not callable: {path}")
                return
            scorer = self._new_target_scorer()
            scorer.load(str(path))
            if not callable(getattr(scorer, "predictor", None)):
                raise RuntimeError(f"Q-sentence DSPy f state is not callable: {path}")
            return
        if kind == "g":
            return super().validate_artifact(kind=kind, artifact=artifact)
        raise ValueError(f"unknown DSPy artifact kind: {kind!r}")

    def _apply_f_scores(self, f_program: Any, *, response: str) -> Dict[str, float]:
        if f_program == self.TEACHER_PASSTHROUGH:
            return {}
        self._assert_lm_input_budget(
            label="q-sentence f inference",
            fields={"summary": str(response or "")},
        )
        import dspy

        lm = self._ensure_lm()
        try:
            with dspy.context(lm=lm):
                result = f_program(summary=str(response or ""))
            return _prediction_scores(result)
        except Exception as exc:
            LOGGER.warning("Q-sentence compact target scorer call failed: %s", exc)
            return {}

    def _leaf_prompt(self, node: LabeledNode) -> str:
        return (
            "Convert this Manifesto Project quasi-sentence span into a compact "
            "CMP policy state. Preserve the CMP code signal, RILE direction, "
            "and domain salience.\n\nSPAN:\n"
            f"{str(node.text or '')}"
        )

    def _merge_prompt(self, *, left_state: str, right_state: Optional[str]) -> str:
        if right_state is None:
            return (
                "Promote this only child CMP policy state as the parent state. "
                "Do not duplicate its counts or salience.\n\nCHILD_STATE:\n"
                f"{left_state}"
            )
        return (
            "Merge these two child CMP policy states into one compact parent "
            "state. Preserve aggregate RILE direction and CMP domain salience.\n\n"
            f"LEFT_STATE:\n{left_state}\n\nRIGHT_STATE:\n{right_state}"
        )

    def _g_prompt_for_node(self, tree: LabeledTree, node: LabeledNode) -> str:
        if int(node.level) == 0:
            return self._leaf_prompt(node)
        left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
        right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
        left_text = _summary_target(left, include_identity_targets=True) or ""
        right_text: Optional[str]
        if right is None or (left is not None and right.node_id == left.node_id):
            right_text = None
        else:
            right_text = _summary_target(right, include_identity_targets=True) or ""
        return self._merge_prompt(left_state=left_text, right_state=right_text)

    def _qsentence_f_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        dims = self._active_dimensions()
        records: List[Dict[str, Any]] = []
        for tree in trees:
            split = str((tree.metadata or {}).get("split", "") or "")
            for node in tree.nodes.values():
                summary = _summary_target(
                    node,
                    include_identity_targets=bool(self.config.include_identity_targets),
                )
                if not summary:
                    continue
                scores = _node_target_scores(node)
                if not scores:
                    continue
                records.append(
                    {
                        "summary": summary,
                        "scores_json": _scores_json(scores, dims),
                        "metadata": {
                            "doc_id": tree.doc_id,
                            "node_id": node.node_id,
                            "split": split,
                            "level": int(node.level),
                            "law_role": "leaf_f" if int(node.level) == 0 else "merge_f",
                            "target_scores": {
                                dim: float(scores[dim]) for dim in dims if dim in scores
                            },
                        },
                    }
                )
        return records

    def _qsentence_g_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        dims = self._active_dimensions()
        records: List[Dict[str, Any]] = []
        for tree in trees:
            split = str((tree.metadata or {}).get("split", "") or "")
            for node in tree.nodes.values():
                target = _summary_target(
                    node,
                    include_identity_targets=bool(self.config.include_identity_targets),
                )
                if not target:
                    continue
                scores = _node_target_scores(node)
                if not scores:
                    continue
                records.append(
                    {
                        "prompt": self._g_prompt_for_node(tree, node),
                        "completion": target,
                        "metadata": {
                            "doc_id": tree.doc_id,
                            "node_id": node.node_id,
                            "split": split,
                            "level": int(node.level),
                            "is_leaf": int(node.level) == 0,
                            "target_scores": {
                                dim: float(scores[dim]) for dim in dims if dim in scores
                            },
                        },
                    }
                )
        return records

    def _check_qsentence_f_record_budgets(self, records: Sequence[Mapping[str, Any]]) -> None:
        for idx, row in enumerate(records):
            self._assert_lm_input_budget(
                label=f"q-sentence f training record {idx}",
                fields={
                    "summary": str(row.get("summary") or ""),
                    "scores_json": str(row.get("scores_json") or ""),
                },
            )

    def _check_qsentence_g_record_budgets(self, records: Sequence[Mapping[str, Any]]) -> None:
        for idx, row in enumerate(records):
            self._assert_lm_input_budget(
                label=f"q-sentence g training record {idx}",
                fields={
                    "prompt": str(row.get("prompt") or ""),
                    "completion": str(row.get("completion") or ""),
                },
            )

    def _score_vector_reward(
        self,
        *,
        predicted: Mapping[str, Any],
        target: Mapping[str, Any],
    ) -> float:
        rewards: List[float] = []
        for dim in self._active_dimensions():
            if dim not in target or dim not in predicted:
                continue
            p = _parse_first_float(predicted.get(dim))
            t = _parse_first_float(target.get(dim))
            if p is None or t is None:
                continue
            rewards.append(max(0.0, 1.0 - abs(_clamp01(p) - _clamp01(t))))
        return float(sum(rewards) / len(rewards)) if rewards else 0.0

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

        records = self._qsentence_f_records(traces)
        self._check_qsentence_f_record_budgets(records)
        train_examples = [
            dspy.Example(
                summary=str(row.get("summary") or ""),
                scores_json=str(row.get("scores_json") or "{}"),
            ).with_inputs("summary")
            for row in records
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        if not train_examples:
            LOGGER.warning("No q-sentence f training examples; saving bare scorer")
            artifact_path = Path(output_dir) / f"f_qsentence_dspy_iter_{iteration:02d}"
            f_program.save(str(artifact_path), save_program=True)
            return str(artifact_path)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            target = parse_compact_scores_json(getattr(gold, "scores_json", "{}"))
            predicted = _prediction_scores(pred)
            return self._score_vector_reward(predicted=predicted, target=target)

        lm = self._ensure_lm()
        with dspy.context(lm=lm):
            compiled = self._compile(
                program=f_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        artifact_path = Path(output_dir) / f"f_qsentence_dspy_iter_{iteration:02d}"
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
        if g_program == self.TEACHER_PASSTHROUGH:
            g_program = dspy.Predict(self._g_signature())

        records = self._qsentence_g_records(traces)
        self._check_qsentence_g_record_budgets(records)
        train_examples = [
            dspy.Example(
                prompt=str(row.get("prompt") or ""),
                completion=str(row.get("completion") or ""),
                target_scores_json=json.dumps(
                    (row.get("metadata") or {}).get("target_scores") or {},
                    sort_keys=True,
                ),
            ).with_inputs("prompt")
            for row in records
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        if not train_examples:
            LOGGER.warning("No q-sentence g training examples; saving bare g")
            artifact_path = Path(output_dir) / f"g_qsentence_dspy_iter_{iteration:02d}.json"
            g_program.save(str(artifact_path))
            return str(artifact_path)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            summary = str(getattr(pred, "completion", "") or "")
            if not summary:
                return 0.0
            try:
                target = json.loads(str(getattr(gold, "target_scores_json", "{}") or "{}"))
            except json.JSONDecodeError:
                target = {}
            predicted = self._apply_f_scores(f_program, response=summary)
            return self._score_vector_reward(predicted=predicted, target=target)

        lm = self._ensure_lm()
        with dspy.context(lm=lm):
            compiled = self._compile(
                program=g_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        artifact_path = Path(output_dir) / f"g_qsentence_dspy_iter_{iteration:02d}.json"
        compiled.save(str(artifact_path))
        return str(artifact_path)

    def _generate_root_state(self, *, g_program: Any, tree: LabeledTree) -> str:
        root = _root_node(tree)
        if root is None:
            return str(tree.document_text or "")
        if g_program == self.TEACHER_PASSTHROUGH:
            return _summary_target(root, include_identity_targets=True) or _root_text(tree)

        state_by_node: Dict[str, str] = {}
        for level_ids in getattr(tree, "levels", None) or []:
            for node_id in level_ids:
                node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                if node is None:
                    continue
                if int(node.level) == 0:
                    prompt = self._leaf_prompt(node)
                else:
                    left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
                    right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
                    left_state = (
                        state_by_node.get(str(left.node_id))
                        if left is not None
                        else ""
                    )
                    if right is None or (left is not None and right.node_id == left.node_id):
                        right_state = None
                    else:
                        right_state = state_by_node.get(str(right.node_id), "")
                    prompt = self._merge_prompt(
                        left_state=str(left_state or ""),
                        right_state=right_state,
                    )
                generated = self._apply_g(g_program, prompt=prompt)
                fallback = _summary_target(node, include_identity_targets=True) or str(node.text or "")
                state_by_node[str(node.node_id)] = generated or fallback
        return (
            state_by_node.get(str(root.node_id))
            or _summary_target(root, include_identity_targets=True)
            or _root_text(tree)
        )

    def score_roots_with_f_by_dimension(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> Dict[str, List[Optional[float]]]:
        f_program = self._load_f_program(f)
        g_program = self._load_g_program(g)
        dims = self._active_dimensions()
        out: Dict[str, List[Optional[float]]] = {dim: [] for dim in dims}

        def score_tree(tree: LabeledTree) -> Dict[str, Optional[float]]:
            target_scores = _tree_target_scores(tree)
            summary = self._generate_root_state(g_program=g_program, tree=tree)
            if f_program == self.TEACHER_PASSTHROUGH or not summary:
                return {dim: target_scores.get(dim) for dim in dims}
            predicted = self._apply_f_scores(f_program, response=summary)
            return {dim: predicted.get(dim) for dim in dims}

        tree_list = list(trees)
        max_workers = max(1, min(len(tree_list), int(self.config.num_threads or 1)))
        if max_workers == 1:
            rows = [score_tree(tree) for tree in tree_list]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                rows = list(pool.map(score_tree, tree_list))
        for row in rows:
            for dim in dims:
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
            rows.append(_mean(values))
        return rows
