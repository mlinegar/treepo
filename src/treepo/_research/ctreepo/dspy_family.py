"""DSPy backend family for the alternating f/g optimization loop.

In this family, ``f`` and ``g`` are each compiled DSPy programs:
- ``g`` = ``dspy.Predict(CTreePOGSignature)`` compiled via MIPROv2/GEPA/etc.
- ``f`` = ``dspy.Predict(CTreePOFSignature)`` likewise.

Both serialize to JSON on disk; the family's artifact type is the JSON path.

Key alternation detail: when training ``g``, the optimizer's metric is
``f_current(response=candidate_summary).score`` — higher is better. This means
DSPy's optimizer (GEPA/MIPRO) searches prompts for g that maximize the
current student f's score, not string similarity to the teacher summary.
This is what makes alternation alternation rather than parallel fit.

Initialization convention:
- ``f_init = "pretuned_scorer"`` / ``None`` -> load the dimension's pre-tuned
  scorer artifact when available, with a bare scorer fallback.
- ``f_init = "bare_scorer"`` -> start from an unoptimized scorer.
- ``f_init = "teacher_passthrough"`` -> read teacher scores directly from tree
  metadata at k=0; training upgrades this sentinel to a bare optimizable scorer.
- ``g_init = "raw_concat"`` / legacy ``"identity"`` -> reduce the supplied
  tree bottom-up by concatenating leaf/child states, never reading cached
  root summaries.
- ``g_init = "teacher_passthrough"`` -> explicit compatibility mode that reads
  the tree's cached teacher/root summary at k=0.

Warmstart from a prior compiled iterate (passing ``dspy.Program.load(prev)``
as ``student=`` to ``optimizer.compile``) is NOT yet wired in this first pass;
the DSPy optimizer API varies across versions and the lift warrants its own
integration check. Left as a followup.
"""

from __future__ import annotations

import json
import logging
import math
import re
import hashlib
import threading
import inspect
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from treepo._research.training.optimization.gepa import GEPA_STRONG_DEFAULT_KWARGS
from treepo._research.core.batch_transport import (
    DEFAULT_BATCH_MAX_CONCURRENT,
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_ROUTING_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
    normalize_base_urls,
)
from treepo._research.ctreepo.alternating import FamilyRuntime
from treepo._research.ctreepo.fg_arity import check_two_child_lm_budget
from treepo._research.tree.labeled import LabeledNode, LabeledTree
from treepo._research.tree.state_tree import (
    StateTree,
    explicit_oracle_trace_kwargs,
    local_law_trace_metadata,
    state_tree_skeleton_from_labeled_tree,
    state_tree_trace_metrics,
    update_state_tree_node,
    write_state_trees_jsonl,
)
from treepo.paths import default_tokenizer_path

LOGGER = logging.getLogger(__name__)
DEFAULT_LOCAL_LAW_WEIGHT_WITH_ANCHORS = 0.25


def _root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    levels = getattr(tree, "levels", None) or []
    for level_ids in reversed(levels):
        for node_id in level_ids:
            node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
            if node is not None:
                return node
    return None


def _labeled_nodes_by_level(tree: LabeledTree) -> List[LabeledNode]:
    nodes: List[LabeledNode] = []
    seen: set[str] = set()
    for level_ids in list(getattr(tree, "levels", None) or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
            if node is not None and str(node.node_id) not in seen:
                nodes.append(node)
                seen.add(str(node.node_id))
    raw_nodes = getattr(tree, "nodes", {}) or {}
    node_values = raw_nodes.values() if isinstance(raw_nodes, Mapping) else raw_nodes
    for node in sorted(
        list(node_values),
        key=lambda item: (int(getattr(item, "level", 0)), str(getattr(item, "node_id", ""))),
    ):
        if str(node.node_id) not in seen:
            nodes.append(node)
            seen.add(str(node.node_id))
    return nodes


def _root_text(tree: LabeledTree) -> str:
    root = _root_node(tree)
    if root is None:
        return ""
    meta = root.metadata or {}
    return str(
        meta.get("teacher_summary")
        or meta.get("target_summary")
        or root.text
        or tree.document_text
        or ""
    )


def _teacher_root_score(tree: LabeledTree) -> Optional[float]:
    root = _root_node(tree)
    if root is not None and root.score is not None:
        try:
            return float(root.score)
        except (TypeError, ValueError):
            pass
    metadata = tree.metadata or {}
    for key in ("teacher_score_1_7", "document_score"):
        val = metadata.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _parse_first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class _TwoPhaseEvaluate:
    """DSPy Evaluate-compatible candidate evaluator for g training.

    DSPy's stock evaluator computes each item as ``program(example)`` followed
    immediately by ``metric(example, prediction)``. For g training our metric
    calls f, so that interleaves long g generations with shorter f scoring
    requests. This evaluator keeps the same score semantics but batches the
    two traffic classes separately: first all g generations, then all f scores.
    """

    def __init__(
        self,
        *,
        devset: list[Any],
        metric: Callable[..., float] | None = None,
        num_threads: int | None = None,
        display_progress: bool = False,
        display_table: bool | int = False,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        failure_score: float = 0.0,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
        **kwargs: Any,
    ) -> None:
        if "return_outputs" in kwargs:
            raise ValueError("`return_outputs` is no longer supported.")
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.failure_score = float(failure_score)
        self.save_as_csv = save_as_csv
        self.save_as_json = save_as_json

    def __call__(
        self,
        program: Any,
        metric: Callable[..., float] | None = None,
        devset: list[Any] | None = None,
        num_threads: int | None = None,
        display_progress: bool | None = None,
        display_table: bool | int | None = None,
        callback_metadata: dict[str, Any] | None = None,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
    ) -> Any:
        import dspy
        from dspy.evaluate.evaluate import EvaluationResult
        from dspy.primitives.prediction import Prediction
        from dspy.utils.parallelizer import ParallelExecutor

        active_metric = metric if metric is not None else self.metric
        active_devset = devset if devset is not None else self.devset
        active_threads = num_threads if num_threads is not None else self.num_threads
        active_progress = (
            display_progress if display_progress is not None else self.display_progress
        )
        if active_metric is None:
            raise ValueError("Two-phase DSPy evaluation requires a metric.")

        examples = list(active_devset)
        if not examples:
            return EvaluationResult(score=0.0, results=[])

        gen_executor = ParallelExecutor(
            num_threads=active_threads,
            disable_progress_bar=not active_progress,
            max_errors=(self.max_errors if self.max_errors is not None else dspy.settings.max_errors),
            provide_traceback=self.provide_traceback,
            compare_results=False,
        )

        def generate(example: Any) -> Any:
            return program(**example.inputs())

        predictions = gen_executor.execute(generate, examples)
        predictions = [
            Prediction() if prediction is None else prediction for prediction in predictions
        ]

        score_executor = ParallelExecutor(
            num_threads=active_threads,
            disable_progress_bar=not active_progress,
            max_errors=(self.max_errors if self.max_errors is not None else dspy.settings.max_errors),
            provide_traceback=self.provide_traceback,
            compare_results=False,
        )

        def score_item(item: tuple[Any, Any]) -> float:
            example, prediction = item
            try:
                return float(active_metric(example, prediction))
            except Exception:
                raise

        scores = score_executor.execute(score_item, list(zip(examples, predictions, strict=True)))
        cleaned_scores = [
            self.failure_score if score is None else float(score) for score in scores
        ]
        results = [
            (example, prediction, score)
            for example, prediction, score in zip(
                examples, predictions, cleaned_scores, strict=True
            )
        ]
        ncorrect = float(sum(cleaned_scores))
        ntotal = len(examples)
        logging.getLogger("dspy.evaluate.evaluate").info(
            "Average Metric: %s / %s (%s%%)",
            ncorrect,
            ntotal,
            round(100 * ncorrect / ntotal, 1),
        )
        return EvaluationResult(score=ncorrect, results=results)


@dataclass
class DSPyFamilyConfig:
    """Config for the DSPy alternating family."""

    #: DSPy optimizer to use ("gepa", "mipro", "bootstrap_fewshot", ...).
    #: Paper canonical: gepa (matches ``OptimizationConfig.optimizer_type``).
    optimizer: str = "gepa"
    #: Budget for GEPA/MIPRO's `auto` knob ("light", "medium", "heavy").
    #: Paper canonical: heavy (matches ``OptimizationConfig.gepa_auto``).
    budget: str = "heavy"
    num_threads: int = 128
    #: Expert/objective target range. For Benoit Environment native-scale
    #: experiments this is 0-10, while the scorer still emits 1-7.
    target_min: float = 1.0
    target_max: float = 7.0
    #: Bounds for DimensionScorer outputs. Benoit's scoring prompts remain
    #: seven-point for all dimensions, even when expert targets are native.
    scorer_output_min: float = 1.0
    scorer_output_max: float = 7.0
    #: LM configuration dict. By default this is routed through
    #: ``BatchedDSPyLM`` so DSPy optimizer traffic shares the repo's central
    #: async batching/client pool; set ``lm_transport="litellm"`` to use
    #: plain ``dspy.LM`` for debugging.
    lm_config: Dict[str, Any] = field(default_factory=dict)
    lm_transport: str = "batch"
    batch_max_concurrent: int = DEFAULT_BATCH_MAX_CONCURRENT
    batch_size: int = DEFAULT_BATCH_SIZE
    batch_timeout: float = DEFAULT_BATCH_TIMEOUT_SECONDS
    batch_request_timeout: float = DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS
    batch_await_response_timeout: Optional[float] = None
    batch_routing_policy: str = DEFAULT_BATCH_ROUTING_POLICY
    #: Optional MIPRO controls. When either num_trials or num_candidates is set,
    #: we run MIPRO in manual mode (auto=None), which is useful for avoiding
    #: DSPy's serial bootstrapped-demo search on large local-vLLM runs.
    mipro_num_candidates: Optional[int] = None
    mipro_num_trials: Optional[int] = None
    mipro_max_bootstrapped_demos: Optional[int] = None
    mipro_max_labeled_demos: Optional[int] = None
    mipro_minibatch_size: int = 35
    mipro_minibatch_full_eval_steps: int = 5
    #: Optional fast-breadth cap applied after record filtering and before
    #: DSPy examples are constructed. <=0 means use all filtered records.
    max_train_records: Optional[int] = None
    record_sample_seed: int = 0
    #: Include train+val examples even when identity targets would cause
    #: circular supervision (inherited from existing record builders).
    include_identity_targets: bool = False
    #: Size of every leaf in tokens. This is the load-bearing axis of the
    #: size-based restructure: leaves are exactly this many EmbeddingGemma
    #: tokens (except possibly the last leaf of a document). g at merge
    #: time must hold 2 × this many tokens. Set at family construction.
    leaf_size_tokens: int = 512
    #: Total LM context window in tokens. The budget check asserts:
    #:   ``2 * leaf_size_tokens + max_completion_tokens + prompt_template_overhead_tokens``
    #:   ``<= lm_context_window_tokens``.
    #: Default 32000 matches the production Gemma-4-31B-IT-NVFP4 vLLM server's
    #: ``--max-model-len 32768`` (1024 of headroom for safety).
    lm_context_window_tokens: int = 32000
    #: Upper bound on the LM's generated output per call (tokens). Must satisfy
    #: ``max_completion_tokens >= 2 * leaf_size_tokens`` (two-leaf concat invariant).
    #: Default 1024 is the paper canonical for leaf_size_tokens=512.
    max_completion_tokens: int = 1024
    #: Conservative estimate of DSPy signature template + demo stacking
    #: overhead (tokens) that sits on top of the raw prompt + completion in
    #: the actual LM request. Tune by inspection if MIPRO reports OOC.
    prompt_template_overhead_tokens: int = 1500
    #: Tokenizer used for exact record-level and inference-time budget checks.
    tokenizer_model_path: str = field(default_factory=default_tokenizer_path)
    #: Policy dimension (economic, social, immigration, eu, environment,
    #: decentralization). Drives the rubric / scoring context injected into
    #: the DSPy signatures so the student has teacher-level task framing
    #: even when unoptimized.
    problem_id: str = "manifesto_benoit"
    dimension: str = "economic"
    #: Path to a pre-tuned DSPy scorer (``DimensionScorer`` compiled via GEPA
    #: v2). Loaded as ``f_init`` at k=0 so the starting point is teacher-grade
    #: (~0.83 Pearson on Benoit summaries) instead of a bare ``dspy.Predict``.
    #: When ``None``, defaults to ``outputs/phase1_gepa_v2_rank/<dim>/optimized_scorer.json``.
    #: Set to empty string to force a bare scorer (not recommended).
    f_init_path: Optional[str] = None
    #: Explicit initial f mode. ``pretuned_scorer`` is the historical behavior
    #: formerly mislabeled as ``identity``.
    f_init_mode: str = "pretuned_scorer"
    #: Root label sources used for DSPy distillation records.  Empty means the
    #: training objective is local-law/node-only.
    root_label_sources: Sequence[str] = field(default_factory=tuple)
    root_label_target: str = "expert"
    #: Canonical local-law objective mass λ. When full-doc anchors are enabled,
    #: the root share is ``1 - local_law_weight``.
    local_law_weight: Optional[float] = None
    node_weight_normalization: str = "per_tree"
    #: Caller-side pre-filter: scripts should drop eval trees whose
    #: ``document_text`` length exceeds this BEFORE passing them to the family.
    #: Only relevant at k=0 with ``g="raw_concat"`` (where ``f`` sees the
    #: whole concatenated doc); after train_g, ``g`` produces short summaries
    #: so this stops mattering. ``None`` = no pre-filter.
    max_input_chars: Optional[int] = None
    #: Extra kwargs forwarded to ``dspy.GEPA(**kwargs)`` when ``optimizer="gepa"``.
    #: Sourced from ``src/training/optimization/gepa.py::GEPA_STRONG_DEFAULT_KWARGS``
    #: so this dataclass and ``OptimizationConfig`` cannot drift apart.
    #: Per-call kwargs (metric, reflection_lm, auto, num_threads) are layered
    #: on top in ``DSPyFamily._build_optimizer`` and take precedence.
    gepa_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(GEPA_STRONG_DEFAULT_KWARGS)
    )

    def __post_init__(self) -> None:
        root_label_sources = tuple(str(source).strip() for source in self.root_label_sources if str(source).strip())
        if self.local_law_weight is not None:
            local_value = float(self.local_law_weight)
            if not math.isfinite(local_value) or local_value < 0.0 or local_value > 1.0:
                raise ValueError(
                    f"local_law_weight must be in [0, 1], got {self.local_law_weight!r}"
                )
            if not root_label_sources and not math.isclose(local_value, 1.0):
                raise ValueError("empty root_label_sources requires local_law_weight=1.0")


class DSPyFamily(FamilyRuntime):
    """DSPy alternating family: text-in/text-out g, text-in/scalar-out f.

    Artifact semantics:
    - ``f`` artifact: path to a saved DSPy Program JSON, OR the sentinel string
      ``"teacher_passthrough"`` which reads teacher scores directly from the
      tree at k=0.
    - ``g`` artifact: path to a saved DSPy Program JSON, the ``"raw_concat"``
      sentinel for bottom-up concatenation, or the explicit
      ``"teacher_passthrough"`` compatibility sentinel.
    """

    name: str = "dspy"

    TEACHER_PASSTHROUGH: str = "teacher_passthrough"
    RAW_CONCAT: str = "raw_concat"
    PRETUNED_SCORER: str = "pretuned_scorer"
    BARE_SCORER: str = "bare_scorer"

    def __init__(self, *, config: DSPyFamilyConfig) -> None:
        self.config = config
        self._lm = None
        self._lm_lock = threading.Lock()
        self._dspy_cache_configured = False
        self._last_full_tree_traces: List[StateTree[Any, Any]] = []
        # Pure config check: must hold for every iteration, so evaluate once.
        self._check_two_leaf_budget_config()

    # ------------------------------------------------------------------
    # Budget enforcement (two-leaf concatenation invariant)
    # ------------------------------------------------------------------

    def _check_two_leaf_budget_config(self) -> None:
        """Raise immediately if the LM can't hold 2 × leaf_size_tokens of input
        AND emit at least 2 × leaf_size_tokens of output (so g can pass through
        a literal concatenation of two children if it wants to). No-truncation
        invariant expressed as a pure config relationship.
        """
        check_two_child_lm_budget(
            family_name="DSPyFamily",
            leaf_size_tokens=int(self.config.leaf_size_tokens),
            lm_context_window_tokens=int(self.config.lm_context_window_tokens),
            max_completion_tokens=int(self.config.max_completion_tokens),
            prompt_template_overhead_tokens=int(
                self.config.prompt_template_overhead_tokens
            ),
        )

    def _available_input_budget(self) -> int:
        return (
            int(self.config.lm_context_window_tokens)
            - int(self.config.max_completion_tokens)
            - int(self.config.prompt_template_overhead_tokens)
        )

    def _count_tokens(self, text: str) -> int:
        from treepo._research.preprocessing.leaf_size_utils import count_tokens

        return int(
            count_tokens(
                str(text or ""),
                model_path=str(self.config.tokenizer_model_path),
            )
        )

    def _assert_lm_input_budget(
        self,
        *,
        label: str,
        fields: Mapping[str, str],
    ) -> None:
        """Hard-error before any DSPy optimizer / LM call that would truncate.

        This is intentionally record-level: the fast config check catches the
        canonical two-leaf case, while this scans the actual prompt/response
        text that DSPy will send or place in demos.
        """
        counts = {
            str(key): self._count_tokens(str(value or ""))
            for key, value in dict(fields).items()
        }
        total = int(sum(counts.values()))
        budget = int(self._available_input_budget())
        if total > budget:
            raise RuntimeError(
                f"DSPy no-truncation guard failed for {label}: actual input "
                f"tokens={total}, available budget={budget} "
                f"(lm_context_window_tokens={self.config.lm_context_window_tokens} "
                f"- max_completion_tokens={self.config.max_completion_tokens} "
                f"- prompt_template_overhead_tokens="
                f"{self.config.prompt_template_overhead_tokens}); "
                f"field_counts={counts}. Reduce leaf_size_tokens, shorten "
                "teacher summaries, or run with a larger verified LM context."
            )

    def _check_training_record_budgets(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        role: str,
    ) -> None:
        for idx, row in enumerate(records):
            prompt = str(row.get("prompt") or "")
            if role == "f":
                self._assert_lm_input_budget(
                    label=f"f training record {idx}",
                    fields={
                        "prompt": prompt,
                        "response": str(row.get("response") or ""),
                    },
                )
            elif role == "g":
                self._assert_lm_input_budget(
                    label=f"g training record {idx}",
                    fields={
                        "prompt": prompt,
                        "completion": str(row.get("completion") or ""),
                    },
                )
            else:
                raise ValueError(f"unknown DSPy budget role: {role!r}")

    # ------------------------------------------------------------------
    # LM / signature helpers
    # ------------------------------------------------------------------

    def _ensure_lm(self) -> Any:
        import dspy

        if self._lm is not None:
            return self._lm
        with self._lm_lock:
            if self._lm is not None:
                return self._lm
            return self._ensure_lm_unlocked(dspy=dspy)

    def _ensure_lm_unlocked(self, *, dspy: Any) -> Any:
        if not self.config.lm_config:
            raise ValueError(
                "DSPyFamilyConfig.lm_config must be populated before training; "
                "set model / api_base / api_key via DSPyFamilyConfig(lm_config=...)"
            )
        if not self._dspy_cache_configured:
            # MIPRO/GEPA already parallelize candidate evaluation. DSPy's
            # sqlite-backed disk cache can exhaust file descriptors and lock
            # under long local-vLLM runs, so keep cache state in-process here.
            try:
                dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
            except Exception as exc:
                LOGGER.warning("Failed to configure DSPy cache: %s", exc)
            self._dspy_cache_configured = True
        lm_config = dict(self.config.lm_config)
        transport = str(getattr(self.config, "lm_transport", "batch") or "batch").lower()
        if transport in {"batch", "batched", "centralized", "centralised"}:
            from treepo._research.core.dspy_batch_client import BatchedDSPyLM

            model = str(lm_config.pop("model", "") or "")
            api_base = lm_config.pop("api_base", None)
            api_bases = lm_config.pop("api_bases", None)
            api_key = str(lm_config.pop("api_key", "EMPTY") or "EMPTY")
            max_tokens = lm_config.pop("max_tokens", None)
            temperature = lm_config.pop("temperature", None)
            cache = bool(lm_config.pop("cache", True))
            if not model:
                raise ValueError(
                    "Batched DSPy LM transport requires lm_config['model']; "
                    "set --dspy-model or use lm_transport='litellm'."
                )
            base_list = normalize_base_urls(api_base=api_base, api_bases=api_bases)
            if not base_list:
                raise ValueError(
                    "Batched DSPy LM transport requires lm_config['api_base'] or "
                    "lm_config['api_bases']; set --dspy-api-base or use "
                    "lm_transport='litellm'."
                )

            self._lm = BatchedDSPyLM(
                model=model,
                api_bases=base_list,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                cache=cache,
                max_concurrent=int(self.config.batch_max_concurrent),
                batch_size=int(self.config.batch_size),
                batch_timeout=float(self.config.batch_timeout),
                request_timeout=float(self.config.batch_request_timeout),
                await_response_timeout=self.config.batch_await_response_timeout,
                routing_policy=str(self.config.batch_routing_policy),
                **lm_config,
            )
            LOGGER.info(
                "Using batched DSPy LM transport: model=%s endpoints=%d "
                "max_concurrent=%d batch_size=%d batch_timeout=%.3f routing=%s",
                model,
                len(base_list),
                int(self.config.batch_max_concurrent),
                int(self.config.batch_size),
                float(self.config.batch_timeout),
                str(self.config.batch_routing_policy),
            )
        elif transport in {"litellm", "direct", "plain", "dspy"}:
            self._lm = dspy.LM(**lm_config)
            LOGGER.info("Using direct DSPy LM transport via dspy.LM")
        else:
            raise ValueError(
                f"Unsupported DSPy LM transport {self.config.lm_transport!r}; "
                "expected 'batch' or 'litellm'."
            )
        return self._lm

    def _task_adapter(self):
        from treepo._research.ctreepo.problem_adapters import dspy_task_adapter

        return dspy_task_adapter(
            problem_id=str(self.config.problem_id),
            dimension=str(self.config.dimension),
            f_init_path=self.config.f_init_path,
        )

    def _g_signature(self):
        import dspy

        instructions = self._task_adapter().summary_instructions()

        class CTreePOGSignature(dspy.Signature):
            __doc__ = instructions

            prompt: str = dspy.InputField(desc="Input text or child summary/summaries to summarize")
            completion: str = dspy.OutputField(desc="Dimension-preserving summary")

        return CTreePOGSignature

    def _default_f_init_path(self) -> Optional[Path]:
        """Resolve the default GEPA-v2 tuned scorer path for the configured dimension.

        Returns ``None`` if the artifact is absent (forces a bare scorer fallback).
        """
        if self.config.f_init_path is not None:
            if not self.config.f_init_path:
                return None  # explicit empty string = do not load
            return Path(self.config.f_init_path)
        root = Path(__file__).resolve().parent.parent.parent
        return self._task_adapter().default_f_init_path(repo_root=root)

    def _new_dimension_scorer(self, *, max_output_tokens: Optional[int] = None):
        """Instantiate a fresh :class:`DimensionScorer` for the configured dimension."""
        return self._task_adapter().new_scorer(
            max_output_tokens=max_output_tokens or int(self.config.max_completion_tokens),
        )

    def _explicit_f_init_path_configured(self) -> bool:
        return self.config.f_init_path is not None and bool(str(self.config.f_init_path))

    @staticmethod
    def _program_forward_parameters(program: Any) -> set[str]:
        target = getattr(program, "forward", None)
        if target is None:
            target = program
        try:
            return set(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            return set()

    def _program_accepts_summary_input(self, program: Any) -> bool:
        return "summary" in self._program_forward_parameters(program)

    def _program_accepts_full_doc_input(self, program: Any) -> bool:
        params = self._program_forward_parameters(program)
        return {"dimension", "task_context", "document"}.issubset(params)

    def _full_doc_f_task_context(self) -> str:
        return self._task_adapter().full_doc_f_task_context()

    def _adapt_full_doc_f_program(self, program: Any, *, source: Path) -> Any:
        """Expose a saved full-document f program as the tree ladder's f(summary) scorer."""

        import dspy

        dimension = str(self.config.dimension)
        task_context = self._full_doc_f_task_context()
        max_output_tokens = int(
            getattr(program, "max_output_tokens", int(self.config.max_completion_tokens))
            or int(self.config.max_completion_tokens)
        )

        class FullDocFAsSummaryScorer(dspy.Module):
            def __init__(
                self,
                *,
                full_doc_program: Any,
                dimension: str,
                task_context: str,
                max_output_tokens: int,
                source_artifact: str,
            ) -> None:
                super().__init__()
                self.dimension = str(dimension)
                self.task_context = str(task_context)
                self.max_output_tokens = int(max_output_tokens)
                self.source_artifact = str(source_artifact)
                predictor = getattr(full_doc_program, "predictor", None)
                if callable(predictor):
                    self.predictor = predictor
                else:
                    self.full_doc_program = full_doc_program

            def forward(self, summary: str, task_context: Optional[str] = None) -> Any:
                ctx = self.task_context if task_context is None else str(task_context)
                if callable(getattr(self, "predictor", None)):
                    result = self.predictor(
                        dimension=self.dimension,
                        task_context=ctx,
                        document=str(summary or ""),
                        config={"max_tokens": int(self.max_output_tokens)},
                    )
                    raw = str(getattr(result, "score", "") or "")
                    return dspy.Prediction(
                        score=raw,
                        parsed_score=_parse_first_float(raw),
                    )
                return self.full_doc_program(
                    dimension=self.dimension,
                    task_context=ctx,
                    document=str(summary or ""),
                )

        LOGGER.info(
            "Adapting full-document DSPy f program from %s to tree f(summary) "
            "interface for dimension=%s",
            source,
            dimension,
        )
        return FullDocFAsSummaryScorer(
            full_doc_program=program,
            dimension=dimension,
            task_context=task_context,
            max_output_tokens=max_output_tokens,
            source_artifact=str(source),
        )

    def _maybe_adapt_loaded_f_program(self, program: Any, *, source: Path) -> Any:
        if self._program_accepts_summary_input(program):
            return program
        if self._program_accepts_full_doc_input(program):
            return self._adapt_full_doc_f_program(program, source=source)
        return program

    def _load_f_program(self, artifact: Any):
        """Return a loaded ``DimensionScorer``, or the ``TEACHER_PASSTHROUGH`` sentinel.

        Semantics:
        - ``None``, ``"pretuned_scorer"``: load the dimension's GEPA-v2 optimized
          scorer if present; otherwise return a bare ``DimensionScorer``. This
          is the default f_init and makes k=0 a teacher-grade scorer (not a bare
          ``Predict``).
        - ``"bare_scorer"``: return a fresh unoptimized ``DimensionScorer``.
        - legacy ``"identity"``: accepted as a readable alias for
          ``"pretuned_scorer"`` with a warning. New manifests should not emit it.
        - ``TEACHER_PASSTHROUGH``: explicit passthrough — callers read teacher
          root scores from tree metadata. No LM call.
        - path to a scorer JSON: load the scorer from that path.
        """
        if artifact == self.TEACHER_PASSTHROUGH:
            return self.TEACHER_PASSTHROUGH
        if artifact == "identity":
            LOGGER.warning(
                "Legacy DSPy f_init='identity' means pretuned_scorer, not true "
                "identity/teacher passthrough. Emitting explicit f_init modes in "
                "new manifests avoids this ambiguity."
            )
            artifact = self.PRETUNED_SCORER
        if artifact == self.BARE_SCORER:
            return self._new_dimension_scorer()
        if artifact in (None, self.PRETUNED_SCORER):
            scorer = self._new_dimension_scorer()
            default_path = self._default_f_init_path()
            explicit_path = self._explicit_f_init_path_configured()
            if default_path is not None:
                if not default_path.exists():
                    if explicit_path:
                        raise RuntimeError(
                            f"Configured DSPy f warm-start artifact does not exist: {default_path}"
                        )
                    LOGGER.info(
                        "No pretuned scorer found for dimension=%s at %s; using bare DimensionScorer",
                        self.config.dimension,
                        default_path,
                    )
                    return scorer
                if default_path.is_dir() and (default_path / "program.pkl").exists():
                    import dspy

                    program = dspy.load(str(default_path))
                    return self._maybe_adapt_loaded_f_program(program, source=default_path)
                try:
                    scorer.load(str(default_path))
                    LOGGER.info(
                        "Loaded GEPA-v2 tuned scorer from %s for dimension=%s",
                        default_path, self.config.dimension,
                    )
                except Exception as exc:
                    if explicit_path:
                        raise RuntimeError(
                            f"Configured DSPy f warm-start artifact could not be loaded: "
                            f"{default_path}"
                        ) from exc
                    LOGGER.warning(
                        "Failed to load GEPA scorer from %s: %s; using bare DimensionScorer",
                        default_path, exc,
                    )
            else:
                LOGGER.info(
                    "No pretuned scorer found for dimension=%s; using bare DimensionScorer",
                    self.config.dimension,
                )
            return scorer
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"DSPy f artifact does not exist: {path}")
        if path.is_dir() and (path / "program.pkl").exists():
            import dspy

            program = dspy.load(str(path))
            return self._maybe_adapt_loaded_f_program(program, source=path)
        scorer = self._new_dimension_scorer()
        try:
            scorer.load(str(path))
        except KeyError as exc:
            raise RuntimeError(
                f"DSPy f artifact {path} is not compatible with DimensionScorer "
                f"state loading"
            ) from exc
        return scorer

    def _load_g_program(self, artifact: Any):
        if artifact == self.TEACHER_PASSTHROUGH:
            return self.TEACHER_PASSTHROUGH
        if artifact in (None, "identity", self.RAW_CONCAT):
            return self.RAW_CONCAT
        import dspy

        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning("DSPy g artifact %s missing; falling back to raw_concat", path)
            return self.RAW_CONCAT
        program = dspy.Predict(self._g_signature())
        program.load(str(path))
        return program

    def _fallback_trainable_f_init(self) -> str:
        mode = str(getattr(self.config, "f_init_mode", self.PRETUNED_SCORER) or self.PRETUNED_SCORER)
        if mode == self.TEACHER_PASSTHROUGH:
            return self.BARE_SCORER
        if mode == "identity":
            return self.PRETUNED_SCORER
        if mode in {self.PRETUNED_SCORER, self.BARE_SCORER}:
            return mode
        return self.BARE_SCORER

    def _apply_g(self, g_program: Any, *, prompt: str) -> str:
        """Run g on a prompt and return the generated completion text."""
        if g_program == self.TEACHER_PASSTHROUGH:
            return ""  # sentinel; caller falls back to the tree's teacher_summary
        if g_program == self.RAW_CONCAT:
            return str(prompt or "")
        self._assert_lm_input_budget(
            label="g inference prompt",
            fields={"prompt": prompt},
        )
        import dspy

        lm = self._ensure_lm()
        try:
            with dspy.context(lm=lm):
                result = g_program(prompt=prompt)
            return str(getattr(result, "completion", "") or "")
        except Exception as exc:
            LOGGER.warning("DSPy g call failed: %s", exc)
            return ""

    def _apply_f_normalized(
        self,
        f_program: Any,
        *,
        prompt: str = "",
        response: str,
    ) -> Optional[float]:
        """Score ``response`` with the DimensionScorer; return normalized [0,1].

        ``prompt`` is accepted for backward compatibility but ignored — the
        DimensionScorer carries the dimension's scoring context internally
        (rubric, 1-7 scale, expert framing) via its frozen task_context.
        This is the fix for the previous bug where passing g's long
        "Summarize..." prompt inflated the LM call to 10977 tokens.
        """
        if f_program == self.TEACHER_PASSTHROUGH:
            return None
        # Only the response (the candidate summary) is the variable input here.
        self._assert_lm_input_budget(
            label="f inference (summary only)",
            fields={"summary": response},
        )
        import dspy

        lm = self._ensure_lm()
        try:
            with dspy.context(lm=lm):
                # DimensionScorer returns {"score": float|None, "reasoning": str}.
                # Use the module call path so compiled DSPy programs run with
                # their optimizer-managed state instead of bypassing wrappers.
                result = f_program(summary=response)
            raw = result.get("score") if isinstance(result, dict) else getattr(result, "score", None)
        except Exception as exc:
            LOGGER.warning("DimensionScorer call failed: %s", exc)
            return None
        if raw is None:
            return None
        return self._normalize_scorer_output(raw)

    @staticmethod
    def _leaf_reduction_prompt(node: LabeledNode) -> str:
        return (
            "Summarize the following leaf span for score-preserving "
            "C-TreePO distillation.\n\nLEAF:\n"
            f"{node.text}"
        )

    @staticmethod
    def _merge_reduction_prompt(left_state: str, right_state: str) -> str:
        return (
            "Merge the two child summaries into one score-preserving "
            "C-TreePO parent summary.\n\nLEFT:\n"
            f"{left_state}\n\nRIGHT:\n{right_state}"
        )

    def _reduce_tree_with_g(self, g_program: Any, tree: LabeledTree) -> str:
        """Generate the root state by applying trained g bottom-up.

        The trained path intentionally consumes only the supplied tree object:
        leaf calls see raw/external leaf text, merge calls see generated child
        states, and cached teacher root summaries are ignored. The
        teacher-passthrough sentinel remains the explicit compatibility mode.
        """
        if g_program == self.TEACHER_PASSTHROUGH:
            return _root_text(tree)
        if g_program == self.RAW_CONCAT:
            states: Dict[str, str] = {}
            levels = list(getattr(tree, "levels", None) or [])
            if not levels:
                return str(tree.document_text or "")
            for level_ids in levels:
                for node_id in level_ids:
                    node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                    if node is None:
                        continue
                    is_leaf = int(node.level) == 0 or (
                        not node.left_child_id and not node.right_child_id
                    )
                    if is_leaf:
                        states[str(node.node_id)] = str(node.text or "")
                    else:
                        states[str(node.node_id)] = "\n\n".join(
                            part
                            for part in (
                                states.get(str(node.left_child_id or ""), ""),
                                states.get(str(node.right_child_id or ""), ""),
                            )
                            if part
                        ) or str(node.text or "")
            root = _root_node(tree)
            if root is not None:
                return str(states.get(str(root.node_id)) or "")
            if states:
                return next(reversed(states.values()))
            return str(tree.document_text or "")

        states: Dict[str, str] = {}
        levels = list(getattr(tree, "levels", None) or [])
        if not levels:
            prompt = str(tree.document_text or "")
            return self._apply_g(g_program, prompt=prompt) or prompt

        for level_ids in levels:
            for node_id in level_ids:
                node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                if node is None:
                    continue
                is_leaf = int(node.level) == 0 or (
                    not node.left_child_id and not node.right_child_id
                )
                if is_leaf:
                    prompt = self._leaf_reduction_prompt(node)
                    fallback = str(node.text or "")
                else:
                    left_state = states.get(str(node.left_child_id or ""), "")
                    right_state = states.get(str(node.right_child_id or ""), "")
                    prompt = self._merge_reduction_prompt(left_state, right_state)
                    fallback = "\n\n".join(
                        part for part in (left_state, right_state) if part
                    )
                    if not fallback:
                        fallback = str(node.text or "")
                generated = str(self._apply_g(g_program, prompt=prompt) or "").strip()
                states[str(node.node_id)] = generated or fallback

        root = _root_node(tree)
        if root is not None:
            return str(states.get(str(root.node_id)) or "")
        if states:
            return next(reversed(states.values()))
        return str(tree.document_text or "")

    def _full_tree_trace_with_g_states(
        self,
        *,
        g_program: Any,
        tree: LabeledTree,
    ) -> Tuple[StateTree[Any, Any], Dict[str, str], Optional[LabeledNode]]:
        """Run g over every node and return the trace skeleton plus node states."""

        trace = state_tree_skeleton_from_labeled_tree(
            tree,
            method_family="dspy",
            state_kind="summary_text",
            split=str((tree.metadata or {}).get("split", "") or ""),
        )
        states: Dict[str, str] = {}
        levels = list(getattr(tree, "levels", None) or [])
        if g_program == self.TEACHER_PASSTHROUGH:
            for node in getattr(tree, "nodes", {}).values():
                states[str(node.node_id)] = str(
                    (node.metadata or {}).get("teacher_summary")
                    or (node.metadata or {}).get("target_summary")
                    or node.text
                    or ""
                )
        elif g_program == self.RAW_CONCAT:
            for level_ids in levels:
                for node_id in level_ids:
                    node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                    if node is None:
                        continue
                    is_leaf = int(node.level) == 0 or (
                        not node.left_child_id and not node.right_child_id
                    )
                    if is_leaf:
                        states[str(node.node_id)] = str(node.text or "")
                    else:
                        states[str(node.node_id)] = "\n\n".join(
                            part
                            for part in (
                                states.get(str(node.left_child_id or ""), ""),
                                states.get(str(node.right_child_id or ""), ""),
                            )
                            if part
                        ) or str(node.text or "")
        else:
            for level_ids in levels:
                for node_id in level_ids:
                    node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                    if node is None:
                        continue
                    is_leaf = int(node.level) == 0 or (
                        not node.left_child_id and not node.right_child_id
                    )
                    if is_leaf:
                        prompt = self._leaf_reduction_prompt(node)
                        fallback = str(node.text or "")
                    else:
                        left_state = states.get(str(node.left_child_id or ""), "")
                        right_state = states.get(str(node.right_child_id or ""), "")
                        prompt = self._merge_reduction_prompt(left_state, right_state)
                        fallback = "\n\n".join(part for part in (left_state, right_state) if part)
                        if not fallback:
                            fallback = str(node.text or "")
                    generated = str(self._apply_g(g_program, prompt=prompt) or "").strip()
                    states[str(node.node_id)] = generated or fallback

        root = _root_node(tree)
        return trace, states, root

    def _node_trace_update_payload(
        self,
        *,
        f_program: Any,
        tree: LabeledTree,
        root: Optional[LabeledNode],
        states: Mapping[str, str],
        node: LabeledNode,
    ) -> Tuple[str, str, Dict[str, Any]]:
        summary = str(states.get(str(node.node_id), node.text or "") or "")
        if f_program == self.TEACHER_PASSTHROUGH:
            pred_source = (
                _teacher_root_score(tree)
                if root is not None and str(node.node_id) == str(root.node_id)
                else node.score
            )
            if pred_source is None:
                pred_source = node.score
            pred_raw = float(pred_source)
            pred_norm = self._normalize_scorer_output(pred_raw)
        else:
            pred_norm = self._apply_f_normalized(f_program, response=summary)
            pred_raw = (
                self._target_from_scorer_norm(float(pred_norm))
                if pred_norm is not None
                else float("nan")
            )
        target_raw = float(node.score)
        finite_pred = math.isfinite(float(pred_raw))
        proxy_loss = float((float(pred_raw) - target_raw) ** 2) if finite_pred else None
        is_root = root is not None and str(node.node_id) == str(root.node_id)
        law_channel = "root" if is_root else ("leaf" if int(node.level) == 0 else "merge")
        oracle_kwargs = explicit_oracle_trace_kwargs(getattr(node, "metadata", {}) or {})
        law_metadata = local_law_trace_metadata(
            prediction=float(pred_raw) if finite_pred else None,
            proxy_target=float(target_raw),
            proxy_loss=proxy_loss,
            oracle_target=oracle_kwargs["oracle_target"],
            oracle_loss=oracle_kwargs["oracle_loss"],
            observed=bool(oracle_kwargs["observed"]),
            sampled=bool(oracle_kwargs["sampled"]),
            propensity=oracle_kwargs["propensity"],
            law_channel=law_channel,
            state_kind="summary_text",
            label_source=str(oracle_kwargs["label_source"] or "proxy_score"),
        )
        return (
            str(node.node_id),
            summary,
            {
                "prediction": float(pred_raw) if finite_pred else None,
                "scorer_output": float(pred_raw) if finite_pred else None,
                "prediction_normalized": (
                    None if pred_norm is None else float(pred_norm)
                ),
                "target": float(target_raw),
                "target_score": float(target_raw),
                **law_metadata,
            },
        )

    @staticmethod
    def _apply_trace_update(
        trace: StateTree[Any, Any],
        *,
        node_id: str,
        summary: str,
        metadata: Mapping[str, Any],
    ) -> None:
        update_state_tree_node(
            trace,
            str(node_id),
            rendered=summary,
            state=summary,
            metadata=dict(metadata),
        )

    def _full_tree_trace_with_g_and_f(
        self,
        *,
        f_program: Any,
        g_program: Any,
        tree: LabeledTree,
    ) -> StateTree[Any, Any]:
        """Run g/f over every node and return a rich full-tree trace."""

        trace, states, root = self._full_tree_trace_with_g_states(
            g_program=g_program,
            tree=tree,
        )
        for node in _labeled_nodes_by_level(tree):
            node_id, summary, metadata = self._node_trace_update_payload(
                f_program=f_program,
                tree=tree,
                root=root,
                states=states,
                node=node,
            )
            self._apply_trace_update(
                trace,
                node_id=node_id,
                summary=summary,
                metadata=metadata,
            )
        return trace

    def _normalize_scorer_output(self, value: Any) -> Optional[float]:
        raw = _parse_first_float(value)
        if raw is None:
            return None
        lo = float(self.config.scorer_output_min)
        hi = float(self.config.scorer_output_max)
        span = max(1e-9, hi - lo)
        return _clamp01((float(raw) - lo) / span)

    def _target_from_scorer_norm(self, value: float) -> float:
        span = float(self.config.target_max) - float(self.config.target_min)
        return float(self.config.target_min) + span * _clamp01(float(value))

    # ------------------------------------------------------------------
    # Record building (mirrors existing helpers in the grid script)
    # ------------------------------------------------------------------

    def _g_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        from treepo._research.ctreepo.distillation import build_g_sft_records

        return build_g_sft_records(
            list(trees),
            include_identity_targets=self.config.include_identity_targets,
            target_min=float(self.config.target_min),
            target_max=float(self.config.target_max),
            scorer_output_min=float(self.config.scorer_output_min),
            scorer_output_max=float(self.config.scorer_output_max),
            root_label_sources=tuple(self.config.root_label_sources or ()),
            root_label_target=str(self.config.root_label_target),
            local_law_weight=self.config.local_law_weight,
            node_weight_normalization=str(self.config.node_weight_normalization),
        )

    def _f_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        from treepo._research.ctreepo.distillation import build_f_lm_regression_records

        return build_f_lm_regression_records(
            list(trees),
            include_identity_targets=self.config.include_identity_targets,
            target_min=float(self.config.target_min),
            target_max=float(self.config.target_max),
            scorer_output_min=float(self.config.scorer_output_min),
            scorer_output_max=float(self.config.scorer_output_max),
            root_label_sources=tuple(self.config.root_label_sources or ()),
            root_label_target=str(self.config.root_label_target),
            local_law_weight=self.config.local_law_weight,
            node_weight_normalization=str(self.config.node_weight_normalization),
        )

    def _effective_local_law_weight(self) -> float:
        if not tuple(self.config.root_label_sources or ()):
            return 1.0
        if self.config.local_law_weight is None:
            return float(DEFAULT_LOCAL_LAW_WEIGHT_WITH_ANCHORS)
        return float(self.config.local_law_weight)

    def _effective_root_anchor_weight(self) -> float:
        if not tuple(self.config.root_label_sources or ()):
            return 0.0
        return float(1.0 - self._effective_local_law_weight())

    def _record_weight(self, row: Mapping[str, Any]) -> float:
        raw = row.get("weight")
        if raw is None:
            raw = (row.get("metadata") or {}).get("example_weight") if isinstance(row.get("metadata"), Mapping) else None
        try:
            weight = float(1.0 if raw is None else raw)
        except (TypeError, ValueError):
            weight = 1.0
        return max(0.0, weight)

    def _filter_weighted_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        role: str,
    ) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        skipped_zero = 0
        skipped_raw_budget = 0
        for row in records:
            payload = dict(row)
            weight = self._record_weight(payload)
            if weight <= 0.0:
                skipped_zero += 1
                continue
            meta = dict(payload.get("metadata") or {})
            is_raw_anchor = (
                str(meta.get("anchor_text_source") or "") == "raw_document"
                and str(meta.get("law_role") or "").startswith("full_doc_")
            )
            if is_raw_anchor:
                try:
                    self._check_training_record_budgets([payload], role=role)
                except RuntimeError as exc:
                    skipped_raw_budget += 1
                    LOGGER.warning("Skipping over-budget raw full-doc %s anchor: %s", role, exc)
                    continue
            payload["weight"] = float(weight)
            meta["example_weight"] = float(weight)
            payload["metadata"] = meta
            kept.append(payload)
        if skipped_zero or skipped_raw_budget:
            LOGGER.info(
                "Filtered %s records: kept=%d skipped_zero_weight=%d skipped_raw_budget=%d",
                role,
                len(kept),
                skipped_zero,
                skipped_raw_budget,
            )
        return kept

    def _record_identity(self, row: Mapping[str, Any], *, role: str) -> str:
        meta = dict(row.get("metadata") or {})
        parts = [
            str(self.config.record_sample_seed),
            str(self.config.dimension),
            str(role),
            str(meta.get("law_role") or ""),
            str(meta.get("doc_id") or ""),
            str(meta.get("node_id") or ""),
            str(meta.get("tree_id") or ""),
            str(row.get("prompt") or row.get("response") or "")[:512],
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8", errors="replace")).hexdigest()

    def _stable_record_sort_key(self, row: Mapping[str, Any], *, role: str) -> Tuple[str, str, str]:
        meta = dict(row.get("metadata") or {})
        return (
            str(meta.get("doc_id") or ""),
            str(meta.get("node_id") or ""),
            self._record_identity(row, role=role),
        )

    def _select_stratified_records(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        limit: int,
        role: str,
    ) -> List[Dict[str, Any]]:
        if limit <= 0 or not records:
            return []
        if len(records) <= limit:
            return sorted(records, key=lambda row: self._stable_record_sort_key(row, role=role))

        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in records:
            meta = dict(row.get("metadata") or {})
            key = (str(meta.get("law_role") or "unknown"), str(meta.get("doc_id") or ""))
            groups.setdefault(key, []).append(row)
        for key in list(groups):
            groups[key] = sorted(
                groups[key],
                key=lambda row: self._stable_record_sort_key(row, role=role),
            )
        ordered_keys = sorted(
            groups,
            key=lambda key: hashlib.sha256(
                "\x1f".join(
                    [
                        str(self.config.record_sample_seed),
                        str(self.config.dimension),
                        str(role),
                        key[0],
                        key[1],
                    ]
                ).encode("utf-8", errors="replace")
            ).hexdigest(),
        )

        selected: List[Dict[str, Any]] = []
        offset = 0
        while len(selected) < limit:
            progressed = False
            for key in ordered_keys:
                bucket = groups[key]
                if offset < len(bucket):
                    selected.append(bucket[offset])
                    progressed = True
                    if len(selected) >= limit:
                        break
            if not progressed:
                break
            offset += 1
        return selected

    def _cap_training_records(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        role: str,
    ) -> List[Dict[str, Any]]:
        raw_cap = self.config.max_train_records
        if raw_cap is None or int(raw_cap) <= 0:
            return list(records)
        cap = int(raw_cap)
        if len(records) <= cap:
            return list(records)

        expected_anchor_role = f"full_doc_{role}_anchor"
        anchors: List[Dict[str, Any]] = []
        leaves: List[Dict[str, Any]] = []
        merges: List[Dict[str, Any]] = []
        other: List[Dict[str, Any]] = []
        for row in records:
            law_role = str((row.get("metadata") or {}).get("law_role") or "unknown")
            if law_role == expected_anchor_role:
                anchors.append(row)
            elif law_role.startswith("leaf_"):
                leaves.append(row)
            elif law_role.startswith("merge_"):
                merges.append(row)
            else:
                other.append(row)

        selected = self._select_stratified_records(anchors, limit=cap, role=role)
        remaining = cap - len(selected)
        if remaining <= 0:
            LOGGER.info(
                "Capped DSPy %s records from %d to %d; kept anchors only because cap was exhausted",
                role,
                len(records),
                len(selected),
            )
            return selected

        leaf_quota = remaining // 2
        merge_quota = remaining - leaf_quota
        selected_leaves = self._select_stratified_records(leaves, limit=leaf_quota, role=role)
        selected_merges = self._select_stratified_records(merges, limit=merge_quota, role=role)
        selected.extend(selected_leaves)
        selected.extend(selected_merges)

        leftover_capacity = cap - len(selected)
        if leftover_capacity > 0:
            leftover: List[Dict[str, Any]] = []
            selected_ids = {id(row) for row in selected}
            for row in list(leaves) + list(merges) + list(other):
                if id(row) not in selected_ids:
                    leftover.append(row)
            selected.extend(
                self._select_stratified_records(leftover, limit=leftover_capacity, role=role)
            )

        LOGGER.info(
            "Capped DSPy %s records from %d to %d "
            "(anchors=%d/%d leaves=%d/%d merges=%d/%d other_capacity=%d)",
            role,
            len(records),
            len(selected),
            len([row for row in selected if str((row.get("metadata") or {}).get("law_role") or "") == expected_anchor_role]),
            len(anchors),
            len([row for row in selected if str((row.get("metadata") or {}).get("law_role") or "").startswith("leaf_")]),
            len(leaves),
            len([row for row in selected if str((row.get("metadata") or {}).get("law_role") or "").startswith("merge_")]),
            len(merges),
            max(0, leftover_capacity),
        )
        return selected

    def _full_doc_anchor_enabled(self) -> bool:
        return bool(tuple(self.config.root_label_sources or ()))

    def _assert_full_doc_anchor_records_present(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        role: str,
    ) -> None:
        if not self._full_doc_anchor_enabled():
            return
        if self._effective_root_anchor_weight() <= 0.0:
            return
        expected = f"full_doc_{role}_anchor"
        count = 0
        by_role: Dict[str, int] = {}
        for row in records:
            meta = dict(row.get("metadata") or {})
            law_role = str(meta.get("law_role") or "unknown")
            by_role[law_role] = int(by_role.get(law_role, 0)) + 1
            if law_role == expected:
                count += 1
        if count > 0:
            return
        raise RuntimeError(
            "Full-document anchor objective is enabled but no "
            f"{expected} records reached DSPy {role} training. "
            f"root_label_sources={tuple(self.config.root_label_sources or ())!r} "
            f"root_label_target={self.config.root_label_target!r} "
            f"local_law_weight={self.config.local_law_weight!r} "
            f"available_law_roles={by_role}. "
            "Check that teacher traces contain stored root summaries and "
            "expert/root targets, or set ROOT_LABEL_SOURCES='' for "
            "teacher-only reproduction."
        )

    def _training_record_summary(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        role: str,
        pre_cap_records: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
            by_role: Dict[str, Dict[str, float]] = {}
            by_anchor_source: Dict[str, int] = {}
            by_target_source: Dict[str, int] = {}
            doc_ids = set()
            observed_targets = 0
            total_weight = 0.0
            for row in rows:
                weight = self._record_weight(row)
                total_weight += float(weight)
                meta = dict(row.get("metadata") or {})
                doc_id = meta.get("doc_id")
                if doc_id is not None:
                    doc_ids.add(str(doc_id))
                law_role = str(meta.get("law_role") or "unknown")
                bucket = by_role.setdefault(law_role, {"count": 0.0, "weight": 0.0})
                bucket["count"] += 1.0
                bucket["weight"] += float(weight)
                anchor_source = meta.get("anchor_text_source")
                if anchor_source is not None:
                    key = str(anchor_source)
                    by_anchor_source[key] = int(by_anchor_source.get(key, 0)) + 1
                target_source = meta.get("target_source")
                if target_source is not None:
                    key = str(target_source)
                    by_target_source[key] = int(by_target_source.get(key, 0)) + 1
                if bool(meta.get("observed_target")):
                    observed_targets += 1
            return {
                "count": int(len(rows)),
                "tree_count": int(len(doc_ids)),
                "total_weight": float(total_weight),
                "by_law_role": {
                    key: {"count": int(value["count"]), "weight": float(value["weight"])}
                    for key, value in sorted(by_role.items())
                },
                "by_anchor_text_source": by_anchor_source,
                "by_target_source": by_target_source,
                "observed_target_count": int(observed_targets),
            }

        post = summarize(records)
        pre = summarize(pre_cap_records if pre_cap_records is not None else records)
        cap_value = self.config.max_train_records
        cap_applied = pre["count"] != post["count"]
        return {
            "role": str(role),
            **post,
            "record_cap": {
                "max_train_records": None if cap_value is None or int(cap_value) <= 0 else int(cap_value),
                "applied": bool(cap_applied),
                "pre_cap_count": int(pre["count"]),
                "post_cap_count": int(post["count"]),
                "pre_cap_by_law_role": pre["by_law_role"],
                "post_cap_by_law_role": post["by_law_role"],
                "sample_seed": int(self.config.record_sample_seed),
            },
            "objective": {
                "root_label_sources": [str(source) for source in tuple(self.config.root_label_sources or ())],
                "root_label_target": str(self.config.root_label_target),
                "root_share": float(self._effective_root_anchor_weight()),
                "local_law_weight": float(self._effective_local_law_weight()),
                "local_law_component_weights": {
                    "teacher_node": float(self._effective_local_law_weight()),
                },
                "node_weight_normalization": str(self.config.node_weight_normalization),
                "target_min": float(self.config.target_min),
                "target_max": float(self.config.target_max),
                "scorer_output_min": float(self.config.scorer_output_min),
                "scorer_output_max": float(self.config.scorer_output_max),
            },
        }

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _compile(
        self,
        *,
        program: Any,
        metric: Callable[..., float],
        trainset: Sequence[Any],
        valset: Sequence[Any],
        two_phase_evaluate: bool = False,
    ) -> Any:
        import dspy

        optimizer_name = self.config.optimizer.strip().lower()
        if optimizer_name == "gepa":
            # gepa_kwargs (strong defaults) are baked into DSPyFamilyConfig as
            # field defaults sourced from GEPA_STRONG_DEFAULT_KWARGS. Per-call
            # kwargs (metric, reflection_lm, auto, num_threads) layer on top.
            gepa_kwargs = dict(self.config.gepa_kwargs or {})
            gepa_kwargs.update({
                "metric": metric,
                "reflection_lm": dspy.settings.lm,
                "auto": self.config.budget,
                "num_threads": int(self.config.num_threads),
            })
            optimizer = dspy.GEPA(**gepa_kwargs)
        elif optimizer_name == "mipro":
            manual_mipro = (
                self.config.mipro_num_candidates is not None
                or self.config.mipro_num_trials is not None
            )
            zero_demo_mipro = (
                self.config.mipro_max_bootstrapped_demos == 0
                and self.config.mipro_max_labeled_demos == 0
            )
            mipro_num_trials = self.config.mipro_num_trials
            if manual_mipro and mipro_num_trials is None:
                mipro_num_trials = max(1, int(self.config.mipro_num_candidates or 1))
            optimizer = dspy.MIPROv2(
                metric=metric,
                auto=None if manual_mipro else self.config.budget,
                num_threads=int(self.config.num_threads),
                num_candidates=self.config.mipro_num_candidates,
                max_bootstrapped_demos=(
                    4
                    if self.config.mipro_max_bootstrapped_demos is None
                    else int(self.config.mipro_max_bootstrapped_demos)
                ),
                max_labeled_demos=(
                    4
                    if self.config.mipro_max_labeled_demos is None
                    else int(self.config.mipro_max_labeled_demos)
                ),
            )
            if zero_demo_mipro:
                def _skip_fewshot_bootstrap(_optimizer: Any, program: Any, trainset: list, seed: int, teacher: Any) -> None:
                    LOGGER.info(
                        "Skipping MIPRO few-shot bootstrap because both "
                        "mipro_max_bootstrapped_demos and mipro_max_labeled_demos are 0"
                    )
                    return None

                # DSPy still runs a serial BootstrapFewShot pass for instruction
                # proposal when both demo caps are zero. For explicit 0/0 runs,
                # make that setting literal so the parallel evaluation phase can
                # start immediately.
                optimizer._bootstrap_fewshot_examples = types.MethodType(  # type: ignore[attr-defined]
                    _skip_fewshot_bootstrap,
                    optimizer,
                )
        elif optimizer_name in {"bootstrap_fewshot", "bootstrap"}:
            # Cap demo stacking to keep context under budget. DSPy's default
            # of max_bootstrapped_demos=4 stacks large demo (prompt, response,
            # score) tuples — for merge-level records, each demo alone can be
            # >1k tokens and four of them overflow a 12k context window.
            optimizer = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=2,
                max_labeled_demos=2,
            )
        else:
            raise ValueError(f"unsupported DSPy optimizer: {self.config.optimizer!r}")
        try:
            compile_kwargs = {
                "trainset": list(trainset),
                "valset": list(valset),
            }
            if optimizer_name == "mipro":
                if mipro_num_trials is not None:
                    compile_kwargs["num_trials"] = int(mipro_num_trials)
                if self.config.mipro_max_bootstrapped_demos is not None:
                    compile_kwargs["max_bootstrapped_demos"] = int(
                        self.config.mipro_max_bootstrapped_demos
                    )
                if self.config.mipro_max_labeled_demos is not None:
                    compile_kwargs["max_labeled_demos"] = int(
                        self.config.mipro_max_labeled_demos
                    )
                compile_kwargs["minibatch_size"] = int(self.config.mipro_minibatch_size)
                compile_kwargs["minibatch_full_eval_steps"] = int(
                    self.config.mipro_minibatch_full_eval_steps
                )
                if (
                    self.config.mipro_max_bootstrapped_demos == 0
                    and self.config.mipro_max_labeled_demos == 0
                ):
                    compile_kwargs["fewshot_aware_proposer"] = False
            if bool(two_phase_evaluate) and optimizer_name == "mipro":
                import dspy.teleprompt.mipro_optimizer_v2 as mipro_optimizer_v2

                old_evaluate = getattr(mipro_optimizer_v2, "Evaluate")
                LOGGER.info("Using two-phase DSPy g evaluation")
                try:
                    mipro_optimizer_v2.Evaluate = _TwoPhaseEvaluate
                    return optimizer.compile(program, **compile_kwargs)
                finally:
                    mipro_optimizer_v2.Evaluate = old_evaluate
            return optimizer.compile(program, **compile_kwargs)
        except TypeError:
            return optimizer.compile(program, trainset=list(trainset))

    # ------------------------------------------------------------------
    # FamilyRuntime protocol
    # ------------------------------------------------------------------

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """**Strengthen** the current f via MIPRO/GEPA over its existing state.

        Warmstart invariant (see feedback_never_reset_between_rungs.md):
        the current ``f_init`` (which is a ``DimensionScorer`` — typically
        already loaded with GEPA-v2 state for the configured dimension) is
        passed as the ``program`` to ``optimizer.compile(...)`` so each rung
        refines the previous, never resets it.

        Supervision records come from ``build_f_lm_regression_records`` —
        they carry ``response`` = the node's summary and ``score`` = the
        teacher's 1-7 score normalized to [0, 1].

        Metric: rewards the predicted score matching the target score —
        ``1 - |pred - target|``. This is a distillation-style metric that
        correctly rewards fidelity, not high absolute scores.
        """
        import dspy

        # Warmstart: load the current f program (DimensionScorer) so the
        # optimizer refines IT rather than starting from a fresh bare Predict.
        f_program = self._load_f_program(f_init)
        if f_program == self.TEACHER_PASSTHROUGH:
            # Upgrade passthrough to a loaded DimensionScorer so we have
            # something to optimize.
            f_program = self._load_f_program(self._fallback_trainable_f_init())

        records = self._filter_weighted_records(self._f_records(traces), role="f")
        pre_cap_records = list(records)
        records = self._cap_training_records(pre_cap_records, role="f")
        self._assert_full_doc_anchor_records_present(records, role="f")
        self._check_training_record_budgets(records, role="f")
        output_dir.mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"f_training_records_summary_iter_{iteration:02d}.json").write_text(
            json.dumps(
                self._training_record_summary(records, role="f", pre_cap_records=pre_cap_records),
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        train_examples = [
            dspy.Example(
                summary=str(row.get("response") or ""),
                score=str(float(row.get("score", 0.5))),
                weight=float(row.get("weight", 1.0)),
            ).with_inputs("summary")
            for row in records
        ]
        if not train_examples:
            LOGGER.warning("No f training examples; skipping f compile")
            path = Path(output_dir) / "f_dspy_noop.json"
            path.write_text("{}\n", encoding="utf-8")
            return str(path)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            target = _parse_first_float(getattr(gold, "score", None))
            # DimensionScorer.forward returns a dict; dspy Module calls return
            # a Prediction wrapper whose ``.score`` attribute holds the raw
            # 1-7 string. Parse whichever form appears and normalize to [0,1]
            # so the diff is comparable to the target.
            raw_score = getattr(pred, "score", None)
            if raw_score is None and isinstance(pred, dict):
                raw_score = pred.get("score")
            predicted_raw = _parse_first_float(raw_score)
            if target is None or predicted_raw is None:
                return 0.0
            predicted_norm = self._normalize_scorer_output(predicted_raw)
            if predicted_norm is None:
                return 0.0
            target_norm = _clamp01(target)
            weight = _parse_first_float(getattr(gold, "weight", None))
            return float(1.0 if weight is None else max(0.0, weight)) * max(
                0.0,
                1.0 - abs(predicted_norm - target_norm),
            )

        lm = self._ensure_lm()
        with dspy.context(lm=lm):
            compiled = self._compile(
                program=f_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        artifact_path = Path(output_dir) / f"f_dspy_iter_{iteration:02d}"
        compiled.save(str(artifact_path), save_program=True)
        return str(artifact_path)

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        """Hard-check that a returned artifact can be reloaded by this family."""
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
            raise RuntimeError(f"DSPy {kind} artifact does not exist: {path}")
        import dspy

        if path.is_dir():
            if not (path / "program.pkl").exists():
                raise RuntimeError(
                    f"DSPy {kind} program directory is missing program.pkl: {path}"
                )
            program = dspy.load(str(path))
            if kind == "f" and not callable(getattr(program, "predictor", None)):
                raise RuntimeError(
                    f"DSPy f program at {path} does not expose a callable predictor"
                )
            return
        if kind == "f":
            scorer = self._new_dimension_scorer()
            scorer.load(str(path))
            if not callable(getattr(scorer, "predictor", None)):
                raise RuntimeError(
                    f"DSPy f state at {path} does not expose a callable predictor"
                )
            return
        if kind == "g":
            program = dspy.Predict(self._g_signature())
            program.load(str(path))
            return
        raise ValueError(f"unknown DSPy artifact kind: {kind!r}")

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """**Strengthen** the current g using f_current as the scoring judge.

        Two invariants:

        1. **Warmstart** (see feedback_never_reset_between_rungs.md): the
           current ``g_init`` program is the ``program`` arg to
           ``optimizer.compile`` — not a fresh ``Predict``.
        2. **Agreement metric** (not raw f-score): the reward for a candidate
           summary is ``1 - |f_current(candidate) - target| / scale`` where
           ``target`` is the ground-truth score known for the node (teacher
           score on the node's span, or expert score at the root). This
           rewards *fidelity* — g should produce summaries that let f
           correctly recover the known score — not summaries that merely
           make f output a high absolute number (which would just be reward
           hacking).
        """
        import dspy

        f_program = self._load_f_program(f)
        if f_program == self.TEACHER_PASSTHROUGH:
            # Upgrade to the loaded DimensionScorer so we have a real judge.
            f_program = self._load_f_program(self._fallback_trainable_f_init())

        # Warmstart: load current g.
        g_program = self._load_g_program(g_init)
        if g_program in (self.TEACHER_PASSTHROUGH, self.RAW_CONCAT):
            g_program = dspy.Predict(self._g_signature())

        records = self._filter_weighted_records(self._g_records(traces), role="g")
        pre_cap_records = list(records)
        records = self._cap_training_records(pre_cap_records, role="g")
        self._assert_full_doc_anchor_records_present(records, role="g")
        self._check_training_record_budgets(records, role="g")
        output_dir.mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"g_training_records_summary_iter_{iteration:02d}.json").write_text(
            json.dumps(
                self._training_record_summary(records, role="g", pre_cap_records=pre_cap_records),
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        # Build target lookup so the metric can find each example's ground-
        # truth target score by node_id. Record metadata carries the raw
        # teacher score in "target_score_raw"; fall back to 4.0 (dimension
        # midpoint) when absent.
        target_by_node: Dict[str, float] = {}
        for row in records:
            meta = row.get("metadata") or {}
            node_id = str(meta.get("node_id") or "")
            raw = meta.get("target_score_raw")
            if node_id and raw is not None:
                try:
                    target_by_node[node_id] = float(raw)
                except (TypeError, ValueError):
                    pass

        train_examples = [
            dspy.Example(
                prompt=str(row.get("prompt") or ""),
                completion=str(row.get("completion") or ""),
                node_id=str((row.get("metadata") or {}).get("node_id") or ""),
                target_raw=float(
                    (row.get("metadata") or {}).get("target_score_raw") or 4.0
                ),
                target_normalized=float(
                    (row.get("metadata") or {}).get("target_score_normalized") or 0.5
                ),
                weight=float(row.get("weight", 1.0)),
            ).with_inputs("prompt")
            for row in records
        ]
        if not train_examples:
            LOGGER.warning("No g training examples; skipping g compile")
            path = Path(output_dir) / "g_dspy_noop.json"
            path.write_text("{}\n", encoding="utf-8")
            return str(path)

        lm = self._ensure_lm()
        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            summary = str(getattr(pred, "completion", "") or "")
            if not summary:
                return 0.0
            if f_program == self.TEACHER_PASSTHROUGH:
                # Fallback: lexical similarity to teacher's reference summary.
                from difflib import SequenceMatcher

                reference = str(getattr(gold, "completion", "") or "")
                if not reference:
                    return 0.0
                weight = _parse_first_float(getattr(gold, "weight", None))
                return float(1.0 if weight is None else max(0.0, weight)) * float(
                    SequenceMatcher(None, reference, summary).ratio()
                )

            # Ground-truth target for this example, already normalized by the
            # row builder on its own scale: teacher nodes use the scorer's
            # 1-7 bounds, expert anchors use the native expert bounds.
            target_norm = _parse_first_float(getattr(gold, "target_normalized", None))
            if target_norm is None:
                target_raw = _parse_first_float(getattr(gold, "target_raw", None))
                lo = float(self.config.target_min)
                hi = float(self.config.target_max)
                span = max(1e-9, hi - lo)
                if target_raw is None:
                    node_id = str(getattr(gold, "node_id", "") or "")
                    target_raw = target_by_node.get(node_id)
                if target_raw is None:
                    target_raw = (lo + hi) / 2.0
                target_norm = _clamp01((float(target_raw) - lo) / span)
            else:
                target_norm = _clamp01(target_norm)

            # Score the candidate summary with the current f. _apply_f_normalized
            # already normalizes DimensionScorer's 1-7 output to [0, 1].
            predicted_norm = self._apply_f_normalized(
                f_program, response=summary
            )
            if predicted_norm is None:
                return 0.0
            # Reward = how close f's score on the candidate comes to the known
            # target. Maximum = 1 (perfect agreement), minimum = 0.
            weight = _parse_first_float(getattr(gold, "weight", None))
            return float(1.0 if weight is None else max(0.0, weight)) * max(
                0.0,
                1.0 - abs(float(predicted_norm) - target_norm),
            )

        with dspy.context(lm=lm):
            compiled = self._compile(
                program=g_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
                two_phase_evaluate=True,
            )
        artifact_path = Path(output_dir) / f"g_dspy_iter_{iteration:02d}.json"
        compiled.save(str(artifact_path))
        return str(artifact_path)

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Optional[float]]:
        """Score each tree's root by reducing the supplied tree with g, then f.

        Teacher-passthrough sentinels use the tree's existing teacher summary
        and/or teacher score. This makes the k=0 (identity) row meaningful
        without an LM call: predictions equal the teacher's root scores.
        Trained g programs do not read cached teacher root summaries; they
        generate leaf states and internal merge states from the supplied tree.
        """
        f_program = self._load_f_program(f)
        g_program = self._load_g_program(g)

        def score_tree(tree: LabeledTree) -> Optional[float]:
            if f_program == self.TEACHER_PASSTHROUGH:
                raw = _teacher_root_score(tree)
                return None if raw is None else float(raw)
            summary = self._reduce_tree_with_g(g_program, tree)
            if not str(summary or "").strip():
                return None
            pred_norm = self._apply_f_normalized(f_program, response=str(summary))
            if pred_norm is None:
                return None
            return float(self._target_from_scorer_norm(float(pred_norm)))

        tree_list = list(trees)
        max_workers = max(1, min(len(tree_list), int(self.config.num_threads or 1)))
        if max_workers == 1:
            return [score_tree(tree) for tree in tree_list]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(score_tree, tree_list))

    def full_tree_traces_with_f_g(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[StateTree[Any, Any]]:
        """Score every node and return full summary/score traces."""

        f_program = self._load_f_program(f)
        g_program = self._load_g_program(g)
        tree_list = list(trees)
        max_workers = max(1, min(len(tree_list), int(self.config.num_threads or 1)))
        if max_workers == 1:
            prepared = [
                self._full_tree_trace_with_g_states(g_program=g_program, tree=tree)
                for tree in tree_list
            ]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                prepared = list(
                    pool.map(
                        lambda tree: self._full_tree_trace_with_g_states(
                            g_program=g_program, tree=tree
                        ),
                        tree_list,
                    )
                )
        traces = [item[0] for item in prepared]

        tasks: List[Tuple[int, LabeledTree, StateTree[Any, Any], Dict[str, str], Optional[LabeledNode], LabeledNode]] = []
        for trace_idx, (tree, (trace, states, root)) in enumerate(zip(tree_list, prepared)):
            for node in _labeled_nodes_by_level(tree):
                tasks.append((trace_idx, tree, trace, states, root, node))

        def score_node(
            item: Tuple[
                int,
                LabeledTree,
                StateTree[Any, Any],
                Dict[str, str],
                Optional[LabeledNode],
                LabeledNode,
            ],
        ) -> Tuple[int, str, str, Dict[str, Any]]:
            trace_idx, tree, _trace, states, root, node = item
            node_id, summary, metadata = self._node_trace_update_payload(
                f_program=f_program,
                tree=tree,
                root=root,
                states=states,
                node=node,
            )
            return trace_idx, node_id, summary, metadata

        node_workers = max(1, min(len(tasks), int(self.config.num_threads or 1)))
        if node_workers == 1:
            updates = [score_node(item) for item in tasks]
        else:
            with ThreadPoolExecutor(max_workers=node_workers) as pool:
                updates = list(pool.map(score_node, tasks))
        for trace_idx, node_id, summary, metadata in updates:
            self._apply_trace_update(
                traces[int(trace_idx)],
                node_id=node_id,
                summary=summary,
                metadata=metadata,
            )
        self._last_full_tree_traces = list(traces)
        return traces

    def export_last_full_tree_traces(
        self,
        output_root: str | Path,
        *,
        split: str = "predict",
    ) -> Dict[str, Any]:
        """Persist the most recent full-tree traces emitted by inference."""

        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        trace_path = root / f"full_tree_traces_{split}.jsonl"
        metrics_path = root / f"full_tree_metrics_{split}.json"
        write_state_trees_jsonl(self._last_full_tree_traces, trace_path)
        metrics = state_tree_trace_metrics(self._last_full_tree_traces)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "full_tree_traces_jsonl": str(trace_path),
            "full_tree_metrics_json": str(metrics_path),
        }
