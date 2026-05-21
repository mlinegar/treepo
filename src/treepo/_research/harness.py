"""
Lightweight harness for tree-structured document audit with formal certificates.

This is the public API for ThinkingTrees. It composes the existing tree builder,
auditor, and IPW estimators behind a single TreeAudit class.

Usage:
    from treepo._research.harness import TreeAudit, AuditBudget

    # Production: batched async requests, saturates GPU
    audit = TreeAudit(
        llm_endpoint="http://localhost:8000/v1",
        oracle=lambda text: my_model.score(text),
    )
    result = audit.run_sync(["document text here..."])
    print(result.certificate.to_dict())

    # Commercial API with lower concurrency
    audit = TreeAudit(
        llm_endpoint="https://api.openai.com/v1",
        oracle=my_oracle,
        model="gpt-4o",
        max_concurrent=20,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from treepo._research.core.data_models import Tree
from treepo._research.core.llm_client import LLMClient, LLMConfig, create_summarizer
from treepo._research.core.scoring import SimilarityScorer, BoundedScale, UNIT_SCALE
from treepo._research.core.strategy import (
    BatchedStrategy,
    CallableStrategy,
    TournamentStrategy,
    TournamentConfig,
)
from treepo._research.tree.builder import TreeBuilder, BuildConfig, BuildResult
from treepo._research.tree.auditor import Auditor, AuditConfig, AuditReport, SamplingStrategy
from treepo._research.tree.ipw import (
    TreeSample,
    NodeType,
    ipw_violation_rate,
    ipw_union_bound,
    ipw_violation_empirical_bernstein_ci,
    ipw_preference_empirical_bernstein_ci,
    effective_sample_size,
    analyze_tree_samples,
)
from treepo._research.training.supervision import (
    BinaryComparison,
    SupervisionDataset,
    save_supervision_artifact_bundle,
)
from treepo._research.feedback.types import FeedbackDataset, FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AuditBudget:
    """Controls the audit scope and statistical targets."""

    delta: float = 0.05
    """Confidence parameter. Certificate holds with probability >= 1 - delta."""

    epsilon: float = 0.10
    """Target violation bound. Drives sample budget when set."""

    sample_budget: int = 20
    """Max nodes to audit per tree (overridden by epsilon/delta if both set)."""

    audit_idempotence: bool = True
    """Check re-summarization stability (C2)."""

    audit_substitution: bool = True
    """Check leaf boundary consistency (C3 Case A)."""

    sampling_strategy: str = "random"
    """Sampling strategy: "random", "level_weighted", or "content_weighted"."""

    content_weight_concentration: float = 2.0
    """α exponent for content-weighted PPS: weight_i = info_score_i^α."""

    content_weight_propensity_floor: float = 0.01
    """Minimum inclusion probability for content-weighted sampling."""


@dataclass
class AuditCertificate:
    """JSON-serializable audit certificate with formal guarantees."""

    guarantee_level: str
    """One of: EXACT (all rates 0), UNION_BOUND (nonzero rates), EMPIRICAL (CI-based)."""

    violation_bound: float
    """Upper bound on Pr[root violation]: N*p_suff + M*p_merge + (R-1)*p_idem."""

    confidence: float
    """Confidence level: 1 - delta."""

    sufficiency_rate: float
    """IPW-corrected leaf sufficiency violation rate."""

    merge_rate: float
    """IPW-corrected Lean-L2 / paper-C3 merge violation rate."""

    idempotence_rate: float
    """IPW-corrected idempotence violation rate."""

    ci_low: float
    """Lower bound of empirical Bernstein CI on overall violation rate."""

    ci_high: float
    """Upper bound of empirical Bernstein CI on overall violation rate."""

    n_documents: int
    n_nodes_audited: int
    n_leaves_total: int
    n_merges_total: int
    effective_sample_size: float
    token_usage: Dict[str, int]
    model_id: str
    timestamp: str
    seed: Optional[int] = None
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AuditCertificate:
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> AuditCertificate:
        return cls.from_dict(json.loads(s))


@dataclass
class HarnessResult:
    """Complete result of a TreeAudit run."""

    certificate: AuditCertificate
    trees: List[Tree]
    supervision: SupervisionDataset
    trace: List[Dict[str, Any]]
    audit_reports: List[AuditReport]
    feedback: Optional[FeedbackDataset] = None
    """Generalized feedback collected via FeedbackCollectors (if configured)."""

    def save(self, output_dir: str) -> None:
        """Save all artifacts to a directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Certificate
        (out / "certificate.json").write_text(self.certificate.to_json())

        save_supervision_artifact_bundle(
            self.supervision,
            supervision_path=out / "supervision.json",
        )

        # Feedback
        if self.feedback and len(self.feedback) > 0:
            self.feedback.save(out / "feedback.json")

        # Trace
        with (out / "trace.jsonl").open("w") as f:
            for event in self.trace:
                f.write(json.dumps(event) + "\n")

        # Audit reports
        (out / "audit_reports.json").write_text(
            json.dumps(
                [
                    report.to_dict() if hasattr(report, "to_dict") else {}
                    for report in self.audit_reports
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        # Trees
        tree_dir = out / "trees"
        tree_dir.mkdir(exist_ok=True)
        for i, tree in enumerate(self.trees):
            tree.save(str(tree_dir / f"tree_{i:04d}.json"))


# ---------------------------------------------------------------------------
# TreeAudit: the main harness
# ---------------------------------------------------------------------------

class TreeAudit:
    """
    Lightweight harness for auditable tree-structured document processing.

    Composes TreeBuilder + Auditor + IPW estimators behind a single interface.
    Accepts any OpenAI-compatible LLM endpoint and any user-defined oracle
    function. Produces an AuditCertificate with formal statistical guarantees
    and optionally collects IPW-weighted preference pairs for DPO/GRPO training.

    By default, uses AsyncBatchLLMClient for high-throughput request pooling
    (up to max_concurrent requests in flight) and BatchTreeOrchestrator for
    cross-document pipelined tree building with no level-synchronization barriers.

    Args:
        llm_endpoint: URL of an OpenAI-compatible API (vLLM, SGLang, OpenAI, etc.).
        oracle: Function (text) -> float scoring text on [0, 1].
        oracle_scale: BoundedScale for oracle values. Defaults to [0, 1].
        context_limit: Model context window in tokens.
        budget: Audit scope and statistical parameters.
        rubric: Task rubric for summarization prompts.
        model: Model name. "default" auto-detects from the endpoint.
        api_key: API key for Authorization header. "EMPTY" for local vLLM/SGLang
            servers, or a real key for OpenAI/Anthropic commercial APIs.
        max_summary_tokens: Max tokens per summarization response.
        chunk_chars: Target characters per leaf chunk.
        max_concurrent: Max in-flight requests to the LLM server. Use 200
            for local vLLM/SGLang, ~20 for rate-limited commercial APIs.
        batch_size: Requests collected before dispatch. Default 50.
        tournament_k: Candidates per tournament. 0 = no tournament (faster).
        judge: Pairwise judge for tournament. Required if tournament_k > 0.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        llm_endpoint: str = "http://localhost:8000/v1",
        oracle: Optional[Callable[[str], float]] = None,
        oracle_scale: BoundedScale = UNIT_SCALE,
        context_limit: int = 32768,
        budget: Optional[AuditBudget] = None,
        rubric: str = "",
        model: str = "default",
        api_key: str = "EMPTY",
        max_summary_tokens: int = 500,
        chunk_chars: int = 2000,
        max_concurrent: int = 200,
        batch_size: int = 50,
        tournament_k: int = 0,
        judge: Any = None,
        seed: Optional[int] = None,
        feedback_collectors: Optional[List[Any]] = None,
        _client_override: Any = None,
    ):
        self.budget = budget or AuditBudget()
        self.rubric = rubric
        self.seed = seed
        self.tournament_k = tournament_k
        self._feedback_collectors = feedback_collectors or []
        self._run_id = str(uuid.uuid4())[:12]
        self._chunk_chars = chunk_chars
        self._max_summary_tokens = max_summary_tokens

        # --- Execution mode ---
        # _client_override: sync mock client for testing (uses CallableStrategy)
        # otherwise: AsyncBatchLLMClient for production (uses BatchedStrategy)
        self._use_batch = _client_override is None
        self._llm_endpoint = llm_endpoint
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self._batch_size = batch_size

        if _client_override is not None:
            # Test / sync mode
            self._sync_client = _client_override
            self._model_id = getattr(_client_override, "config", LLMConfig()).model
            summarizer_fn = create_summarizer(client=self._sync_client)
            base_strategy = CallableStrategy(summarizer=summarizer_fn)
            if tournament_k > 0 and judge is not None:
                self._strategy = TournamentStrategy(
                    base=base_strategy, judge=judge,
                    config=TournamentConfig(k=tournament_k),
                )
            else:
                self._strategy = base_strategy
            self._summarizer_fn = summarizer_fn
        else:
            # Production mode -- batch client created in run()
            self._sync_client = None
            self._model_id = model
            self._judge = judge
            self._strategy = None  # built in run() after batch client starts
            self._summarizer_fn = None

        # --- Oracle scorer ---
        if oracle is not None:
            self._scorer = SimilarityScorer(
                value_extractor=oracle,
                scale=oracle_scale,
                name="oracle",
                cache_size=4096,
            )
        else:
            self._scorer = None

        # --- Audit config ---
        self._audit_config = AuditConfig(
            sample_budget=self.budget.sample_budget,
            sampling_strategy=SamplingStrategy(self.budget.sampling_strategy),
            discrepancy_threshold=0.1,
            audit_idempotence=self.budget.audit_idempotence,
            audit_substitution=self.budget.audit_substitution,
            target_epsilon=self.budget.epsilon,
            target_delta=self.budget.delta,
            random_seed=seed,
            content_weight_concentration=self.budget.content_weight_concentration,
            content_weight_propensity_floor=self.budget.content_weight_propensity_floor,
        )
        computed_budget = self._audit_config.compute_sample_budget_for_guarantee()
        if computed_budget > self._audit_config.sample_budget:
            self._audit_config.sample_budget = computed_budget

    async def run(
        self,
        documents: List[str],
        feedback_signals: Optional[List[Any]] = None,
    ) -> HarnessResult:
        """
        Process documents, build trees, audit, and emit certificate.

        Args:
            documents: Document texts to process.
            feedback_signals: Optional per-document feedback signals (e.g., from
                VLM visual segmentation). Each entry is a list of
                ``ChunkFeedbackSignal`` or None. When provided, signals are
                injected into the build config for adaptive chunking.

        In production mode (default), uses AsyncBatchLLMClient for high-
        throughput request pooling and BatchTreeOrchestrator for cross-document
        pipelined execution. In test mode (_client_override), falls back to
        sequential per-document processing.
        """
        if self._use_batch:
            return await self._run_batched(documents, feedback_signals)
        else:
            return await self._run_sequential(documents, feedback_signals)

    def run_sync(
        self,
        documents: List[str],
        feedback_signals: Optional[List[Any]] = None,
    ) -> HarnessResult:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(documents, feedback_signals=feedback_signals))

    # ------------------------------------------------------------------
    # Production path: batched async
    # ------------------------------------------------------------------

    async def _run_batched(
        self,
        documents: List[str],
        feedback_signals: Optional[List[Any]] = None,
    ) -> HarnessResult:
        """Build trees with full async batching, then audit."""
        from treepo._research.core.batch_processor import AsyncBatchLLMClient
        from treepo._research.core.batch_orchestrator import BatchTreeOrchestrator

        # Per-document feedback signals not yet supported in batch mode.
        # Log a warning if provided; they will be ignored.
        if feedback_signals and any(s is not None for s in feedback_signals):
            logger.warning(
                "Per-document feedback_signals are not yet supported in batched mode. "
                "Use sequential mode (set _client_override) or pass a single shared "
                "signal set via BuildConfig for batched VLM workflows."
            )

        batch_client = AsyncBatchLLMClient(
            base_url=self._llm_endpoint,
            max_concurrent=self._max_concurrent,
            batch_size=self._batch_size,
            model=None if self._model_id == "default" else self._model_id,
            api_key=self._api_key,
        )

        try:
            await batch_client.start()
            self._model_id = batch_client.model or self._model_id

            # Create batched strategy
            base_strategy = BatchedStrategy(
                client=batch_client,
                max_tokens=self._max_summary_tokens,
            )
            if self.tournament_k > 0 and self._judge is not None:
                strategy = TournamentStrategy(
                    base=base_strategy, judge=self._judge,
                    config=TournamentConfig(k=self.tournament_k),
                )
            else:
                strategy = base_strategy

            # Build all trees with cross-document pipelining
            build_config = BuildConfig(
                max_chunk_chars=self._chunk_chars,
                chunk_strategy="axis",
            )

            t0 = time.monotonic()
            orchestrator = BatchTreeOrchestrator(
                strategy=strategy, config=build_config,
            )
            build_results = await orchestrator.process_documents(
                documents=documents,
                rubric=self.rubric,
                get_text_fn=lambda doc: doc,
            )
            build_elapsed = time.monotonic() - t0

            logger.info(
                "Built %d trees in %.1fs (batched, %d max concurrent)",
                len(build_results), build_elapsed, self._max_concurrent,
            )

            # Create a sync summarizer for audit idempotence/substitution checks
            # (these re-summarize nodes and need a callable summarizer)
            sync_client = LLMClient(LLMConfig(
                base_url=self._llm_endpoint,
                model=self._model_id,
                api_key=self._api_key,
            ))
            summarizer_fn = create_summarizer(client=sync_client)

            # Audit + aggregate
            return self._audit_and_aggregate(
                documents, build_results, summarizer_fn,
                batch_stats=batch_client.stats,
            )

        finally:
            await batch_client.stop()

    # ------------------------------------------------------------------
    # Test / sync path: sequential
    # ------------------------------------------------------------------

    async def _run_sequential(
        self,
        documents: List[str],
        feedback_signals: Optional[List[Any]] = None,
    ) -> HarnessResult:
        """Build trees one at a time (test/small-run mode)."""
        build_results: List[BuildResult] = []
        for doc_idx, doc_text in enumerate(documents):
            # Per-document feedback signals for adaptive chunking
            doc_signals = None
            if feedback_signals and doc_idx < len(feedback_signals):
                doc_signals = feedback_signals[doc_idx]

            builder = TreeBuilder(
                strategy=self._strategy,
                config=BuildConfig(
                    max_chunk_chars=self._chunk_chars,
                    chunk_strategy="axis",
                    chunk_feedback_signals=doc_signals,
                ),
            )

            # Reset tournament segment counter per document so segment IDs
            # (leaf_0, leaf_1, ...) stay aligned with tree node IDs.
            if hasattr(self._strategy, "reset_counter"):
                self._strategy.reset_counter()
            result = await builder.build(doc_text, self.rubric)
            build_results.append(result)

        return self._audit_and_aggregate(
            documents, build_results, self._summarizer_fn,
        )

    # ------------------------------------------------------------------
    # Shared: audit all trees and build certificate
    # ------------------------------------------------------------------

    def _audit_and_aggregate(
        self,
        documents: List[str],
        build_results: List[BuildResult],
        summarizer_fn: Optional[Callable],
        batch_stats: Any = None,
    ) -> HarnessResult:
        """Audit trees from build results and produce HarnessResult."""
        trees: List[Tree] = []
        all_binary_comparisons: List[BinaryComparison] = []
        all_samples: List[TreeSample] = []
        audit_reports: List[AuditReport] = []
        trace: List[Dict[str, Any]] = []
        total_leaves = 0
        total_merges = 0

        # Create auditor (needs scorer + optional summarizer for C2/C3a)
        auditor = None
        if self._scorer is not None:
            auditor = Auditor(
                oracle=self._scorer,
                config=self._audit_config,
                summarizer=summarizer_fn,
            )

        for doc_idx, result in enumerate(build_results):
            t0 = time.monotonic()
            doc_id = f"doc_{doc_idx:04d}"
            tree = result.tree
            trees.append(tree)

            binary_projection = result.supervision.project_binary(projection="adjacent")
            if binary_projection.comparisons:
                for pref in binary_projection.comparisons:
                    pref.source_doc_id = doc_id
                all_binary_comparisons.extend(binary_projection.comparisons)

            n_leaves = tree.leaf_count
            n_merges = max(0, n_leaves - 1)
            total_leaves += n_leaves
            total_merges += n_merges

            # Inject per-document content_weights for CONTENT_WEIGHTED sampling
            if auditor is not None and result.content_weights:
                auditor.config.content_weights = result.content_weights

            # Audit
            report: Optional[AuditReport] = None
            if auditor is not None:
                report = auditor.audit_tree(tree)
                report.source_doc_id = doc_id
                audit_reports.append(report)
                all_samples.extend(report.to_tree_samples())

            # Propagate design-time inclusion probabilities to binary supervision rows
            if report is not None and binary_projection.comparisons and report.inclusion_probability_map:
                from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
                from treepo._research.tree.compositional_learning import SHARED_SAMPLED_QUERY_POLICY_NAME

                prob_map = report.inclusion_probability_map
                observation_ids_by_unit = {
                    str(observation.unit_id): str(observation.observation_id)
                    for observation in (report.logged_observations or [])
                }
                for pref in binary_projection.comparisons:
                    example_id = pref.source_example_id or ""
                    # Strip doc_id prefix if present (batched mode tags: "doc_id:leaf_N")
                    node_ref = example_id.rsplit(":", 1)[-1] if ":" in example_id else example_id
                    prop = prob_map.get(node_ref)
                    if prop is not None:
                        base_sampling = getattr(pref, "sampling", None)
                        if not isinstance(base_sampling, SamplingMetadata):
                            base_sampling = SamplingMetadata(unit_kind=ObservationUnitKind.PAIR)
                        pref.sampling = base_sampling.with_updates(
                            document_propensity=1.0,
                            unit_propensity=float(prop),
                            label_propensity=1.0,
                            sampling_scheme=report.sampling_strategy,
                            policy_name=SHARED_SAMPLED_QUERY_POLICY_NAME,
                            unit_kind=ObservationUnitKind.PAIR,
                            supports_ipw_estimation=True,
                        )
                        if node_ref in observation_ids_by_unit:
                            pref.source_observation_ids = [observation_ids_by_unit[node_ref]]

            elapsed = time.monotonic() - t0
            trace.append({
                "run_id": self._run_id,
                "doc_id": doc_id,
                "doc_idx": doc_idx,
                "n_leaves": n_leaves,
                "n_merges": n_merges,
                "n_nodes_audited": report.nodes_audited if report else 0,
                "n_failures": report.nodes_failed if report else 0,
                "latency_s": round(elapsed, 3),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

            logger.info(
                "doc %d/%d: %d leaves, %d audited, %d failures (%.1fs)",
                doc_idx + 1, len(build_results), n_leaves,
                report.nodes_audited if report else 0,
                report.nodes_failed if report else 0,
                elapsed,
            )

        # Token usage: from batch stats if available, else from sync client
        token_usage = {}
        if batch_stats is not None:
            token_usage = {
                "prompt_tokens": batch_stats.prompt_tokens,
                "completion_tokens": batch_stats.completion_tokens,
                "total_tokens": batch_stats.total_tokens,
                "total_requests": batch_stats.total_requests,
            }
        elif self._sync_client is not None:
            token_usage = self._sync_client.get_usage()

        certificate = self._build_certificate(
            all_samples, total_leaves, total_merges,
            len(build_results), token_usage,
        )

        # --- Generalized feedback collection ---
        feedback_dataset = None
        if self._feedback_collectors:
            feedback_dataset = self._collect_feedback(
                build_results, audit_reports, trees,
            )

        return HarnessResult(
            certificate=certificate,
            trees=trees,
            supervision=SupervisionDataset(
                comparative_judgments=[
                    pair.to_comparative_judgment() for pair in all_binary_comparisons
                ]
            ),
            trace=trace,
            audit_reports=audit_reports,
            feedback=feedback_dataset,
        )

    # ------------------------------------------------------------------
    # Certificate builder
    # ------------------------------------------------------------------

    def _build_certificate(
        self,
        samples: List[TreeSample],
        total_leaves: int,
        total_merges: int,
        n_documents: int,
        token_usage: Dict[str, int],
    ) -> AuditCertificate:
        """Build an AuditCertificate from aggregated IPW samples."""
        n_audited = len(samples)
        delta = self.budget.delta

        if n_audited == 0:
            return AuditCertificate(
                guarantee_level="NONE",
                violation_bound=float("inf"),
                confidence=1.0 - delta,
                sufficiency_rate=float("nan"),
                merge_rate=float("nan"),
                idempotence_rate=float("nan"),
                ci_low=0.0,
                ci_high=1.0,
                n_documents=n_documents,
                n_nodes_audited=0,
                n_leaves_total=total_leaves,
                n_merges_total=total_merges,
                effective_sample_size=0.0,
                token_usage=token_usage,
                model_id=self._model_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                seed=self.seed,
                run_id=self._run_id,
            )

        p_suff = ipw_violation_rate(samples, node_type=NodeType.LEAF)
        p_merge = ipw_violation_rate(samples, node_type=NodeType.MERGE)
        p_idem = ipw_violation_rate(samples, node_type=NodeType.RESUMMARY)

        bound = ipw_union_bound(samples, num_leaves=total_leaves, num_merges=total_merges)
        ci_low, ci_high = ipw_violation_empirical_bernstein_ci(samples, delta=delta)
        neff = effective_sample_size(samples)

        if p_suff == 0.0 and p_merge == 0.0 and p_idem == 0.0:
            level = "EXACT"
        elif ci_high > 0:
            level = "EMPIRICAL"
        else:
            level = "UNION_BOUND"

        return AuditCertificate(
            guarantee_level=level,
            violation_bound=bound,
            confidence=1.0 - delta,
            sufficiency_rate=p_suff,
            merge_rate=p_merge,
            idempotence_rate=p_idem,
            ci_low=ci_low,
            ci_high=ci_high,
            n_documents=n_documents,
            n_nodes_audited=n_audited,
            n_leaves_total=total_leaves,
            n_merges_total=total_merges,
            effective_sample_size=neff,
            token_usage=token_usage,
            model_id=self._model_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=self.seed,
            run_id=self._run_id,
        )

    # ------------------------------------------------------------------
    # Generalized feedback collection
    # ------------------------------------------------------------------

    def _collect_feedback(
        self,
        build_results: List[BuildResult],
        audit_reports: List[AuditReport],
        trees: List[Tree],
    ) -> FeedbackDataset:
        """Create FeedbackRequests from audited nodes and route through collectors.

        For each audited node, creates a FeedbackRequest carrying the node's
        content, rubric, and propensity metadata. Routes through all configured
        feedback collectors and aggregates responses into a FeedbackDataset.
        """
        from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
        from treepo._research.tree.compositional_learning import SHARED_SAMPLED_QUERY_POLICY_NAME

        dataset = FeedbackDataset()
        req_counter = 0

        for doc_idx, tree in enumerate(trees):
            doc_id = f"doc_{doc_idx:04d}"
            report = audit_reports[doc_idx] if doc_idx < len(audit_reports) else None
            prob_map = report.inclusion_probability_map if report else {}

            # Create feedback requests for leaf nodes (auditable units)
            for node in tree.leaves():
                req_counter += 1
                node_prop = prob_map.get(node.id, 1.0)
                request = FeedbackRequest(
                    request_id=f"fb_{self._run_id}_{req_counter}",
                    text_a=node.summary if hasattr(node, "summary") else str(node),
                    original_text=node.text if hasattr(node, "text") else "",
                    rubric=self.rubric,
                    node_id=node.id,
                    tree_id=f"{doc_id}_tree",
                    source_doc_id=doc_id,
                    law_type="sufficiency",
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=float(node_prop),
                        label_propensity=1.0,
                        sampling_scheme=self.budget.sampling_strategy,
                        policy_name=SHARED_SAMPLED_QUERY_POLICY_NAME,
                        unit_kind=ObservationUnitKind.PAIR,
                        supports_ipw_estimation=True,
                    ),
                )

                for collector in self._feedback_collectors:
                    try:
                        response = collector.collect(request)
                        dataset.add(request, response)
                    except Exception as e:
                        logger.warning(
                            "Feedback collector failed for node %s: %s",
                            node.id, e,
                        )

        if len(dataset) > 0:
            logger.info(
                "Collected %d feedback items via %d collectors",
                len(dataset), len(self._feedback_collectors),
            )

        return dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_feedback_from_samples(
    samples: List[Any],
) -> tuple:
    """
    Extract texts and feedback signals from DocumentSample-like objects.

    Convenience helper for PDF workflows where VLM segmentation produces
    visual_feedback_signals in sample metadata.

    Args:
        samples: List of DocumentSample objects (or plain strings).

    Returns:
        Tuple of (texts, feedback_signals) suitable for ``TreeAudit.run()``.
    """
    texts: List[str] = []
    signals: List[Any] = []
    for sample in samples:
        if isinstance(sample, str):
            texts.append(sample)
            signals.append(None)
        else:
            texts.append(getattr(sample, "text", str(sample)))
            meta = getattr(sample, "metadata", {})
            if isinstance(meta, dict):
                signals.append(meta.get("visual_feedback_signals"))
            else:
                signals.append(None)
    return texts, signals


def _parse_host(url: str) -> str:
    """Extract host from URL like http://localhost:8000/v1."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.hostname or "localhost"


def _parse_port(url: str) -> int:
    """Extract port from URL like http://localhost:8000/v1."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.port or 8000
