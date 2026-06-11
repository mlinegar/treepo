"""
OPS-style oracle simulations for a Markov changepoint-count target.

This module is designed to match the Lean OPS semantics (C1/C2/C3 ≈ L1/L3/L2):
we audit leaf- and merge-level oracle preservation for a *tree reduction*,
and we separate three error sources that matter for the paper:

1) Approximation bias: insufficient sketch state (chunking loss).
2) Estimation error: finite training docs + finite oracle labels.
3) Selection bias: adaptive node sampling (corrected by IPW / DSL-style AIPW).

Oracle:
    f⋆(x) = number of changepoints (# adjacent regime flips) in a span.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for markov_changepoint_ops_count_simulation. "
        "Install with: uv sync --extra torch"
    ) from e

from treepo._research.core.logged_supervision import (
    LoggedLabelObservation,
    ObservationUnitKind,
    SamplingMetadata,
    summarize_logged_observations,
    write_logged_observations_jsonl,
)
from treepo._research.tree.compositional_learning import shared_logged_substructure_observation
from treepo._research.tree.ipw import (
    CertificateEnvelope,
    NodeType,
    TreeSample,
    compute_certificate,
    effective_sample_size,
    empirical_bernstein_ci,
    hajek_ht_comparison,
    horvitz_thompson_mean,
    max_weight,
)
from treepo._research.tree.markov_boundary_honesty_simulation import _make_transition_matrices
from treepo._research.tree.markov_changepoint_honesty_simulation import (
    ChangepointMarkovDoc,
    MarkovChangepointConfig as _GeneratorConfig,
    _sample_num_segments,
    _sample_segment_lengths,
    _sample_segment_regimes,
    generate_changepoint_docs,
)
from treepo._research.tree.ctreepo_model import CTreePOConfig, CTreePOModel, normalize_target
from treepo._research.core.ops_checks import EvidenceStatus, LawKind
from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    ORACLE_OBSERVATION_DESIGN_PARAMETER_FIELDS,
    PUBLIC_CONTRACT_LEGACY_FIELDS,
    assert_public_contract_clean,
    canonical_law_set_id,
)
from treepo._research.ctreepo.sim.composite_objective import (
    CompositeObjectiveSpec,
    OBJECTIVE_ESTIMATOR_KEYS,
    evaluate_composite_objective_from_metrics,
    objective_estimator_alias,
    resolve_root_local_objective_weights,
    scalarize_objective_estimates,
)
from treepo._research.ctreepo.sim.core.markov_capability import markov_theorem_score
from treepo._research.ctreepo.sim.core.markov_law_stress import (
    VALID_EXACT_FAMILIES,
    VALID_LAW_PACKAGES,
    markov_law_bundle_score,
)
from treepo._research.ctreepo.sim.core.theorem_feature_route import (
    DEFAULT_THEOREM_FEATURE_ADAPTER,
    valid_theorem_feature_adapters,
)
from treepo._research.ctreepo.sim.core.training_selection import (
    TrainingSelectionMetadata,
    clone_module_state,
    improved_metric,
    restore_module_state,
)
from treepo._research.ctreepo.sim.learning_problem import attach_local_law_learning_problem
from treepo._research.ctreepo.sim.local_law_learnability import (
    DownstreamMetrics,
    GArtifact,
    LocalLawCounterexampleEvaluation,
    LocalLawMetrics,
    LocalLawPolicyEvaluation,
    LocalLawRunSummary,
    PolicyRole,
    SupportBudgetSummary,
    artifact_index,
    write_json_g_artifact,
    write_npz_g_artifact,
)
from treepo._research.training.supervision import (
    DenseSupervisionExample,
    DenseScalarRidgeTrainingConfig,
    DenseScalarTrainingConfig,
    OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
    OPTIMIZER_FAMILY_GRADIENT_DENSE,
    OPTIMIZER_FAMILY_TREE_ENSEMBLE,
    REPRESENTATION_DENSE_FEATURE_VECTOR,
    SupervisionDataset,
    TARGET_SCALAR,
    build_dense_full_document_supervision_dataset,
    build_dense_sampled_substructure_supervision_dataset,
    dense_scalar_rows,
    dense_scalar_rows_to_numpy,
    fit_dense_scalar_ridge_regressor,
    fit_dense_scalar_regressor,
    predict_dense_scalar_ridge_regressor,
    predict_dense_scalar_regressor,
    supervision_training_contract,
)
from treepo._research.core.unified_runtime import (
    VALID_GPU_RUNTIME_BUCKET_MODES,
    VALID_GPU_RUNTIME_DATA_MODES,
)
from treepo._research.core.local_law_adjustment import (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
    VALID_LOCAL_LAW_OBJECTIVE_MODES,
    normalize_local_law_objective_mode,
)


ScheduleName = str
VALID_SCHEDULES: Tuple[ScheduleName, ...] = ("balanced", "left_to_right", "right_to_left")
ModelFamilyName = str
VALID_MODEL_FAMILIES: Tuple[ModelFamilyName, ...] = ("neural", "additive", "fno")
GeneratorProfileName = str
VALID_GENERATOR_PROFILES: Tuple[GeneratorProfileName, ...] = (
    "piecewise_markov",
    "piecewise_palette",
    "piecewise_disjoint_palette",
    "hazard_aliased",
    "hazard_topic",
)
DocSequenceObjectiveName = str
VALID_DOC_SEQUENCE_OBJECTIVES: Tuple[DocSequenceObjectiveName, ...] = (
    "count_ce_only",
    "count_ce_plus_scalar_mse",
)
DocSequenceFNOPoolingName = str
VALID_DOC_SEQUENCE_FNO_POOLING: Tuple[DocSequenceFNOPoolingName, ...] = ("mean", "sum")
LocalLawObjectiveModeName = str
TreeDocumentLossNormalizationModeName = str
VALID_TREE_DOCUMENT_LOSS_NORMALIZATION_MODES: Tuple[
    TreeDocumentLossNormalizationModeName, ...
] = (
    "auto",
    "batch_docs",
    "supervised_docs",
)
TreeRootSupervisionKind = str
VALID_TREE_ROOT_SUPERVISION_KINDS: Tuple[TreeRootSupervisionKind, ...] = (
    "mse",
    "count_ce",
)
TreeCheckpointMetricName = str
VALID_TREE_CHECKPOINT_METRICS: Tuple[TreeCheckpointMetricName, ...] = (
    "val_root_mae",
    "val_leaf_codec_direct",
    "val_theorem_bootstrap_direct",
    "val_exact_sketch_direct",
    "val_task_root_exact_sketch_direct",
)
LeafSupervisionKindName = str
VALID_LEAF_SUPERVISION_KINDS: Tuple[LeafSupervisionKindName, ...] = (
    "count_only",
    "bounded_full_sketch",
    "full_sketch",
)
TreeLocalWeightingModeName = str
VALID_TREE_LOCAL_WEIGHTING_MODES: Tuple[TreeLocalWeightingModeName, ...] = (
    "subset_mean",
    "fixed_k_hajek",
    "span_mass_ipw_sum",
)
TreeSupervisionSourceName = str
VALID_TREE_SUPERVISION_SOURCES: Tuple[TreeSupervisionSourceName, ...] = (
    "rate",
    "manifest",
)
TreeExactCollapseModeName = str
VALID_TREE_EXACT_COLLAPSE_MODES: Tuple[TreeExactCollapseModeName, ...] = (
    "",
    "official_fno_one_tree_identity",
    "official_fno_runtime_identity",
)
ComparisonModeName = str
VALID_COMPARISON_MODES: Tuple[ComparisonModeName, ...] = (
    "legacy",
    "comparable",
    "exact_collapse",
)
TreeTrainingScheduleName = str
VALID_TREE_TRAINING_SCHEDULES: Tuple[TreeTrainingScheduleName, ...] = (
    "single_stage",
    "two_stage",
)
TreeStage1EvalModeName = str
VALID_TREE_STAGE1_EVAL_MODES: Tuple[TreeStage1EvalModeName, ...] = (
    "per_epoch",
    "end_only",
)
TreeBatchPackModeName = str
VALID_TREE_BATCH_PACK_MODES: Tuple[TreeBatchPackModeName, ...] = (
    "structure_bucket",
    "fixed_fused",
)
TreeTaskHeadModeName = str
VALID_TREE_TASK_HEAD_MODES: Tuple[TreeTaskHeadModeName, ...] = (
    "full_state_scalar",
    "theorem_feature_scalar",
)
TreeTheoremSurfaceModeName = str
VALID_TREE_THEOREM_SURFACE_MODES: Tuple[TreeTheoremSurfaceModeName, ...] = (
    "shared_bottleneck",
    "shared_feature",
    "shared_feature_adapters",
    "factorized_score_fiber",
    "carrier_projection",
    "opaque_carrier_exact_sketch",
    "slotwise",
    "learned_projection",
)
TreeScoreMergeModeName = str
VALID_TREE_SCORE_MERGE_MODES: Tuple[TreeScoreMergeModeName, ...] = (
    "gated_affine",
    "exact_projected_sketch",
)
TreePhiAlignmentLossName = str
VALID_TREE_PHI_ALIGNMENT_LOSSES: Tuple[TreePhiAlignmentLossName, ...] = (
    "cosine_mse",
)
TreeTheoremCountHeadModeName = str
VALID_TREE_THEOREM_COUNT_HEAD_MODES: Tuple[TreeTheoremCountHeadModeName, ...] = (
    "scalar_mse",
    "support_classifier",
    "hybrid_ordinal",
)
TreeSummarySpecRootModeName = str
VALID_TREE_SUMMARY_SPEC_ROOT_MODES: Tuple[TreeSummarySpecRootModeName, ...] = (
    "task_split_ablation",
    "theorem_primary",
    "unified_f",
    "factored_theorem_readout",
)
AlignedSketchSurfaceName = str
VALID_ALIGNED_SKETCH_SURFACES: Tuple[AlignedSketchSurfaceName, ...] = (
    "",
    "decoded_markov_sketch",
)
SummarySpecName = str
VALID_SUMMARY_SPEC_NAMES: Tuple[SummarySpecName, ...] = (
    "",
    "markov_count_sketch",
)
DiagnosticDetailModeName = str
VALID_DIAGNOSTIC_DETAIL_MODES: Tuple[DiagnosticDetailModeName, ...] = (
    "summary",
    "debug_raw",
)
PosttrainDiagnosticsModeName = str
VALID_POSTTRAIN_DIAGNOSTICS_MODES: Tuple[PosttrainDiagnosticsModeName, ...] = (
    "",
    "full",
    "minimal",
)
InternalSupervisionKindName = str
VALID_INTERNAL_SUPERVISION_KINDS: Tuple[InternalSupervisionKindName, ...] = (
    "none",
    "count_only",
    "bounded_full_sketch",
    "full_sketch",
)
BudgetedDocConsumptionModeName = str
VALID_BUDGETED_DOC_CONSUMPTION_MODES: Tuple[BudgetedDocConsumptionModeName, ...] = (
    "",
    "root_only",
    "doc_sequence",
    "full_doc_only",
)
BudgetedLocalSplitModeName = str
VALID_BUDGETED_LOCAL_SPLIT_MODES: Tuple[BudgetedLocalSplitModeName, ...] = (
    "",
    "balanced",
    "leaf_heavy",
    "internal_heavy",
    "leaf_only",
    "depth_equal_nonroot",
    "inactive_for_family",
)
BudgetedAllocationPolicyName = str
VALID_BUDGETED_ALLOCATION_POLICIES: Tuple[BudgetedAllocationPolicyName, ...] = (
    "",
    "breadth_first",
    "depth_first",
)
BUDGETED_SAMPLING_SCHEME_RANDOM_WITHOUT_REPLACEMENT = (
    "seeded_random_without_replacement"
)
DocTransformerHeadFamilyName = str
VALID_DOC_TRANSFORMER_HEAD_FAMILIES: Tuple[DocTransformerHeadFamilyName, ...] = (
    "pooled_count_classifier",
    "boundary_sum_count_hybrid",
)
GuidanceOverrideModeName = str
VALID_GUIDANCE_OVERRIDE_MODES: Tuple[GuidanceOverrideModeName, ...] = ("reset", "adjust")
AuditPolicyName = str
VALID_AUDIT_POLICIES: Tuple[AuditPolicyName, ...] = (
    "all",
    "fixed",
    "fraction",
    "sqrt",
    "log2",
)
C3AuditStrategyName = str
VALID_C3_AUDIT_STRATEGIES: Tuple[C3AuditStrategyName, ...] = (
    "uniform",
    "top_span",
    "span_weighted",
    "hybrid_top_span",
)
DEFAULT_NORMALIZED_LOCAL_LAW_WEIGHT = 0.25
StateT = TypeVar("StateT")


def audit_sample_count(
    internal_nodes: int,
    *,
    policy: AuditPolicyName,
    fixed_nodes: int = 0,
    fraction: float = 1.0,
    scale: float = 1.0,
) -> int:
    """
    How many realized internal nodes to label (per doc), matching learned_sketch_simulation semantics.
    """

    n = int(max(0, internal_nodes))
    if n <= 0:
        return 0

    pol = str(policy)
    if pol == "all":
        q = n
    elif pol == "fixed":
        q = int(max(0, fixed_nodes))
    elif pol == "fraction":
        q = int(math.ceil(float(fraction) * float(n)))
    elif pol == "sqrt":
        q = int(math.ceil(float(scale) * math.sqrt(float(n))))
    elif pol == "log2":
        q = int(math.ceil(float(scale) * math.log2(float(n) + 1.0)))
    else:
        raise ValueError(
            f"unsupported audit policy: {policy!r}; expected one of {VALID_AUDIT_POLICIES}"
        )
    return int(max(0, min(n, q)))


def leaf_sample_count(leaves: int, *, rate: float) -> int:
    """How many realized leaf nodes to label (per doc)."""

    n = int(max(0, leaves))
    if n <= 0:
        return 0
    r = float(rate)
    if r <= 0.0:
        return 0
    if r >= 1.0:
        return n
    q = int(math.ceil(r * float(n)))
    return int(max(1, min(n, q)))


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _changepoint_count(regimes: Sequence[int]) -> int:
    """Count regime transitions in a flat sequence.

    Thin re-export of the canonical oracle now registered as
    ``oracle:markov_changepoint_count`` in :mod:`src.ctreepo.oracles.markov`.
    """
    from treepo._research.ctreepo.oracles.markov import markov_changepoint_count

    return markov_changepoint_count(regimes)


def _oracle_count(doc: ChangepointMarkovDoc, *, start: int, end: int) -> int:
    """Apply the changepoint oracle to a doc slice.

    Thin re-export delegating to
    :func:`src.ctreepo.oracles.markov.markov_changepoint_count_for_doc`.
    """
    from treepo._research.ctreepo.oracles.markov import markov_changepoint_count_for_doc

    return markov_changepoint_count_for_doc(doc, start=int(start), end=int(end))


def _leaf_spans(n_tokens: int, *, leaf_tokens: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = int(n_tokens)
    step = int(max(1, leaf_tokens))
    i = 0
    while i < n:
        j = min(n, i + step)
        spans.append((i, j))
        i = j
    return spans


def _normalize_budgeted_doc_consumption_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in VALID_BUDGETED_DOC_CONSUMPTION_MODES:
        raise ValueError(
            "doc_consumption_mode="
            f"{value!r} unsupported; expected one of "
            f"{VALID_BUDGETED_DOC_CONSUMPTION_MODES}"
        )
    return mode


def _normalize_budgeted_local_split_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in VALID_BUDGETED_LOCAL_SPLIT_MODES:
        raise ValueError(
            "local_split_mode="
            f"{value!r} unsupported; expected one of "
            f"{VALID_BUDGETED_LOCAL_SPLIT_MODES}"
        )
    return mode


def _normalize_budgeted_allocation_policy(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in VALID_BUDGETED_ALLOCATION_POLICIES:
        raise ValueError(
            "local_allocation_policy="
            f"{value!r} unsupported; expected one of "
            f"{VALID_BUDGETED_ALLOCATION_POLICIES}"
        )
    return mode


def _local_split_shares(local_split_mode: str) -> Tuple[float, float]:
    mode = _normalize_budgeted_local_split_mode(local_split_mode)
    if mode in {"", "balanced"}:
        return 0.5, 0.5
    if mode == "leaf_heavy":
        return 0.75, 0.25
    if mode == "internal_heavy":
        return 0.25, 0.75
    if mode == "leaf_only":
        return 1.0, 0.0
    if mode == "depth_equal_nonroot":
        # The package builder resolves depth-equal mass targets into compatible
        # leaf/internal label rates and max_internal_depth. At manifest time we
        # only need a stable leaf-vs-internal allocation split for explicit unit
        # selection, so we split evenly across the two non-root supervision
        # families here.
        return 0.5, 0.5
    if mode == "inactive_for_family":
        return 0.0, 0.0
    raise ValueError(f"unhandled local_split_mode={local_split_mode!r}")


def _balanced_merge_leaf_ranges(n_leaves: int) -> List[Tuple[int, int, int]]:
    """Return ``(start_leaf_idx, end_leaf_idx, depth)`` for each merge node.

    ``depth`` is 1-indexed: depth 1 = first round of pairwise leaf merges,
    depth 2 = merging pairs of depth-1 nodes, etc.
    """
    spans: List[Tuple[int, int]] = [(i, i + 1) for i in range(int(n_leaves))]
    merges: List[Tuple[int, int, int]] = []
    depth = 0
    while len(spans) > 1:
        depth += 1
        nxt: List[Tuple[int, int]] = []
        i = 0
        while i < len(spans):
            if i + 1 >= len(spans):
                nxt.append(spans[i])
                i += 1
                continue
            merged = (int(spans[i][0]), int(spans[i + 1][1]))
            merges.append((merged[0], merged[1], depth))
            nxt.append(merged)
            i += 2
        spans = nxt
    return merges


def _doc_leaf_and_internal_spans(
    *,
    n_tokens: int,
    leaf_tokens: int,
    max_internal_depth: int = 0,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return ``(leaf_spans, internal_spans)``.

    If *max_internal_depth* > 0, only internal merge nodes whose depth in the
    balanced binary reduction tree is ``<= max_internal_depth`` are included.
    ``max_internal_depth=0`` (default) means no depth filtering.
    """
    leaf_spans = _leaf_spans(int(n_tokens), leaf_tokens=int(leaf_tokens))
    if not leaf_spans:
        return [], []
    merge_ranges = _balanced_merge_leaf_ranges(len(leaf_spans))
    if int(max_internal_depth) > 0:
        merge_ranges = [
            (s, e, d) for s, e, d in merge_ranges if int(d) <= int(max_internal_depth)
        ]
    internal_spans: List[Tuple[int, int]] = []
    for start_leaf_idx, end_leaf_idx, _depth in merge_ranges:
        start = int(leaf_spans[int(start_leaf_idx)][0])
        end = int(leaf_spans[int(end_leaf_idx) - 1][1])
        internal_spans.append((start, end))
    return leaf_spans, internal_spans


def _sample_units_without_replacement(
    *,
    units_by_doc: Mapping[int, Sequence[Tuple[int, int, float]]],
    budget: int,
    rng: random.Random,
) -> Tuple[Dict[int, List[int]], float]:
    population: List[Tuple[int, int]] = []
    for doc_idx, units in units_by_doc.items():
        for unit_idx, _span_tokens, _mass in units:
            population.append((int(doc_idx), int(unit_idx)))
    total_population = int(len(population))
    if int(budget) <= 0 or total_population <= 0:
        return {}, 0.0
    rng.shuffle(population)
    sample_count = int(min(int(budget), total_population))
    selected: Dict[int, List[int]] = {}
    for doc_idx, unit_idx in population[:sample_count]:
        selected.setdefault(int(doc_idx), []).append(int(unit_idx))
    return (
        {
            int(doc_idx): sorted(int(value) for value in indices)
            for doc_idx, indices in selected.items()
            if indices
        },
        (
            1.0
            if sample_count >= total_population
            else float(sample_count) / float(max(1, total_population))
        ),
    )


def build_budgeted_train_supervision_manifest(
    *,
    docs: Sequence[ChangepointMarkovDoc],
    config: OPSCountConfig,
    baseline_family: str,
    seed: int,
) -> BudgetedTrainSupervisionManifest | None:
    requested_total_calls = int(config.budget_total_calls)
    if requested_total_calls <= 0 and float(config.budget_total_calls_per_doc) > 0.0:
        requested_total_calls = int(
            round(float(config.budget_total_calls_per_doc) * float(len(docs)))
        )
    if requested_total_calls <= 0 or not docs:
        return None

    family = str(baseline_family or "").strip().lower()
    doc_only_family = family in {
        "official_fno",
        "official_fno_sumlen",
        "tree_doc_ridge",
    }
    full_doc_budget_share = float(config.full_doc_budget_share)
    if doc_only_family and abs(full_doc_budget_share - 1.0) > 1e-12:
        raise ValueError(
            f"{family} requires full_doc_budget_share=1.0, got {full_doc_budget_share!r}"
        )

    requested_total_calls = int(max(0, requested_total_calls))
    requested_full_doc_calls = int(
        max(0, min(requested_total_calls, round(full_doc_budget_share * requested_total_calls)))
    )
    requested_local_calls = int(max(0, requested_total_calls - requested_full_doc_calls))

    doc_mode = _normalize_budgeted_doc_consumption_mode(config.doc_consumption_mode)
    local_split_mode = _normalize_budgeted_local_split_mode(config.local_split_mode)
    allocation_policy = _normalize_budgeted_allocation_policy(
        config.local_allocation_policy or "breadth_first"
    )
    if not doc_mode:
        doc_mode = "full_doc_only" if doc_only_family else "root_only"
    if not local_split_mode:
        local_split_mode = "inactive_for_family" if doc_only_family else "balanced"
    if doc_only_family:
        requested_local_calls = 0
        local_split_mode = "inactive_for_family"

    if allocation_policy != "breadth_first":
        raise ValueError(
            "budgeted supervision currently supports only "
            "local_allocation_policy='breadth_first'"
        )

    rng = random.Random(int(seed))
    doc_order = list(range(len(docs)))
    rng.shuffle(doc_order)
    selected_doc_indices = set(
        int(idx) for idx in doc_order[: min(len(doc_order), int(requested_full_doc_calls))]
    )

    leaf_units_by_doc: Dict[int, List[Tuple[int, int, float]]] = {}
    internal_units_by_doc: Dict[int, List[Tuple[int, int, float]]] = {}
    for doc_idx, doc in enumerate(docs):
        doc_tokens = int(getattr(doc, "n_tokens", len(doc.tokens)))
        leaf_spans, internal_spans = _doc_leaf_and_internal_spans(
            n_tokens=int(doc_tokens),
            leaf_tokens=int(config.fixed_leaf_tokens),
            max_internal_depth=int(config.max_internal_depth),
        )
        leaf_units: List[Tuple[int, int, float]] = []
        for unit_idx, (start, end) in enumerate(leaf_spans):
            span_tokens = int(max(0, int(end) - int(start)))
            mass = (
                float(span_tokens) / float(max(1, doc_tokens))
                if doc_tokens > 0
                else 0.0
            )
            leaf_units.append((int(unit_idx), int(span_tokens), float(mass)))
        internal_units: List[Tuple[int, int, float]] = []
        for unit_idx, (start, end) in enumerate(internal_spans):
            span_tokens = int(max(0, int(end) - int(start)))
            mass = (
                float(span_tokens) / float(max(1, doc_tokens))
                if doc_tokens > 0
                else 0.0
            )
            internal_units.append((int(unit_idx), int(span_tokens), float(mass)))
        rng.shuffle(leaf_units)
        rng.shuffle(internal_units)
        leaf_units_by_doc[int(doc_idx)] = leaf_units
        internal_units_by_doc[int(doc_idx)] = internal_units

    leaf_share, internal_share = _local_split_shares(local_split_mode)
    requested_leaf_calls = int(round(float(requested_local_calls) * float(leaf_share)))
    requested_leaf_calls = int(min(requested_leaf_calls, requested_local_calls))
    requested_internal_calls = int(max(0, requested_local_calls - requested_leaf_calls))
    selected_leaf_indices, leaf_unit_propensity = _sample_units_without_replacement(
        units_by_doc=leaf_units_by_doc,
        budget=int(requested_leaf_calls),
        rng=rng,
    )
    selected_internal_indices, internal_unit_propensity = _sample_units_without_replacement(
        units_by_doc=internal_units_by_doc,
        budget=int(requested_internal_calls),
        rng=rng,
    )

    doc_plans: List[BudgetedTrainSupervisionDocPlan] = []
    document_mass_total = 0.0
    leaf_mass_total = 0.0
    internal_mass_total = 0.0
    document_calls_total = 0
    leaf_calls_total = 0
    internal_calls_total = 0
    touched_docs_total = 0
    for doc_idx, doc in enumerate(docs):
        doc_tokens = int(getattr(doc, "n_tokens", len(doc.tokens)))
        leaf_spans, internal_spans = _doc_leaf_and_internal_spans(
            n_tokens=int(doc_tokens),
            leaf_tokens=int(config.fixed_leaf_tokens),
            max_internal_depth=int(config.max_internal_depth),
        )
        document_mode = str(doc_mode) if int(doc_idx) in selected_doc_indices else ""
        leaf_indices = tuple(int(v) for v in selected_leaf_indices.get(int(doc_idx), []))
        internal_indices = tuple(
            int(v) for v in selected_internal_indices.get(int(doc_idx), [])
        )
        document_mass = 1.0 if document_mode else 0.0
        leaf_propensity = (
            float(leaf_unit_propensity) if len(leaf_spans) > 0 else 0.0
        )
        leaf_mass = float(
            sum(
                float(max(0, leaf_spans[int(idx)][1] - leaf_spans[int(idx)][0]))
                / float(max(1, doc_tokens))
                for idx in leaf_indices
                if 0 <= int(idx) < len(leaf_spans)
            )
        )
        internal_propensity = (
            float(internal_unit_propensity) if len(internal_spans) > 0 else 0.0
        )
        internal_mass = float(
            sum(
                float(max(0, internal_spans[int(idx)][1] - internal_spans[int(idx)][0]))
                / float(max(1, doc_tokens))
                for idx in internal_indices
                if 0 <= int(idx) < len(internal_spans)
            )
        )
        raw_call_cost = int(bool(document_mode)) + int(len(leaf_indices)) + int(
            len(internal_indices)
        )
        if raw_call_cost > 0:
            touched_docs_total += 1
        document_calls_total += int(bool(document_mode))
        leaf_calls_total += int(len(leaf_indices))
        internal_calls_total += int(len(internal_indices))
        document_mass_total += float(document_mass)
        leaf_mass_total += float(leaf_mass)
        internal_mass_total += float(internal_mass)
        doc_plans.append(
            BudgetedTrainSupervisionDocPlan(
                doc_index=int(doc_idx),
                doc_tokens=int(doc_tokens),
                document_mode=str(document_mode),
                leaf_indices=tuple(int(v) for v in leaf_indices),
                internal_indices=tuple(int(v) for v in internal_indices),
                raw_call_cost=int(raw_call_cost),
                document_mass=float(document_mass),
                leaf_mass=float(leaf_mass),
                internal_mass=float(internal_mass),
                leaf_propensity=float(leaf_propensity),
                internal_propensity=float(internal_propensity),
                effective_full_doc_mass=float(
                    float(document_mass) + float(leaf_mass) + float(internal_mass)
                ),
            )
        )

    budget_total_calls_used = int(
        document_calls_total + leaf_calls_total + internal_calls_total
    )
    effective_full_doc_mass_total = float(
        document_mass_total + leaf_mass_total + internal_mass_total
    )
    return BudgetedTrainSupervisionManifest(
        budget_total_calls=int(requested_total_calls),
        budget_total_calls_per_doc=float(
            float(requested_total_calls) / float(max(1, len(docs)))
        ),
        budget_total_calls_used=int(budget_total_calls_used),
        budget_utilization=float(
            float(budget_total_calls_used) / float(max(1, requested_total_calls))
        ),
        full_doc_budget_share=float(full_doc_budget_share),
        full_doc_calls_requested=int(requested_full_doc_calls),
        full_doc_calls_total=int(document_calls_total),
        local_calls_requested=int(requested_local_calls),
        local_calls_total=int(leaf_calls_total + internal_calls_total),
        doc_consumption_mode=str(doc_mode),
        local_split_mode=str(local_split_mode),
        local_allocation_policy=str(allocation_policy),
        sampling_scheme=str(BUDGETED_SAMPLING_SCHEME_RANDOM_WITHOUT_REPLACEMENT),
        doc_touch_rate=float(float(touched_docs_total) / float(max(1, len(docs)))),
        mean_labels_per_touched_doc=float(
            float(budget_total_calls_used) / float(max(1, touched_docs_total))
        ),
        touched_docs_total=int(touched_docs_total),
        effective_full_doc_mass_total=float(effective_full_doc_mass_total),
        effective_full_doc_mass_per_doc=float(
            effective_full_doc_mass_total / float(max(1, len(docs)))
        ),
        document_mass_share=float(
            float(document_mass_total) / float(max(1e-12, effective_full_doc_mass_total))
        ),
        leaf_mass_share=float(
            float(leaf_mass_total) / float(max(1e-12, effective_full_doc_mass_total))
        ),
        internal_mass_share=float(
            float(internal_mass_total) / float(max(1e-12, effective_full_doc_mass_total))
        ),
        document_call_share=float(
            float(document_calls_total) / float(max(1, budget_total_calls_used))
        ),
        leaf_call_share=float(
            float(leaf_calls_total) / float(max(1, budget_total_calls_used))
        ),
        internal_call_share=float(
            float(internal_calls_total) / float(max(1, budget_total_calls_used))
        ),
        actual_doc_tokens_mean=float(
            float(sum(int(plan.doc_tokens) for plan in doc_plans))
            / float(max(1, len(doc_plans)))
        ),
        actual_doc_tokens_unique=tuple(
            sorted({int(plan.doc_tokens) for plan in doc_plans})
        ),
        supervision_source="budget_call_allocator",
        local_estimand_mode="",
        mass_target_per_doc=float("nan"),
        requested_root_mass_per_doc=float(
            float(requested_full_doc_calls) / float(max(1, len(docs)))
        ),
        realized_root_mass_per_doc=float(
            float(document_mass_total) / float(max(1, len(docs)))
        ),
        realized_leaf_mass_per_doc=float(
            float(leaf_mass_total) / float(max(1, len(docs)))
        ),
        realized_internal_mass_per_doc=float(
            float(internal_mass_total) / float(max(1, len(docs)))
        ),
        leaf_propensity_mean=float(
            float(
                sum(float(plan.leaf_propensity) for plan in doc_plans)
                / float(max(1, len(doc_plans)))
            )
        ),
        internal_propensity_mean=float(
            float(
                sum(float(plan.internal_propensity) for plan in doc_plans)
                / float(max(1, len(doc_plans)))
            )
        ),
        local_weight_ess=0.0,
        local_weight_max=0.0,
        doc_plans=tuple(doc_plans),
    )


def budgeted_manifest_plan_maps(
    manifest: BudgetedTrainSupervisionManifest | None,
) -> Tuple[Dict[int, str], Dict[int, Tuple[int, ...]], Dict[int, Tuple[int, ...]]]:
    if manifest is None:
        return {}, {}, {}
    document_modes: Dict[int, str] = {}
    leaf_indices: Dict[int, Tuple[int, ...]] = {}
    internal_indices: Dict[int, Tuple[int, ...]] = {}
    for plan in manifest.doc_plans:
        if str(plan.document_mode).strip():
            document_modes[int(plan.doc_index)] = str(plan.document_mode)
        if plan.leaf_indices:
            leaf_indices[int(plan.doc_index)] = tuple(
                int(value) for value in plan.leaf_indices
            )
        if plan.internal_indices:
            internal_indices[int(plan.doc_index)] = tuple(
                int(value) for value in plan.internal_indices
            )
    return document_modes, leaf_indices, internal_indices


def _generator_profile_key(profile: str) -> str:
    key = str(profile or "piecewise_markov").strip().lower() or "piecewise_markov"
    if key not in VALID_GENERATOR_PROFILES:
        raise ValueError(
            f"generator_profile={profile!r} unsupported; expected one of {VALID_GENERATOR_PROFILES}"
        )
    return key


def _make_aliased_transition_matrices(
    *,
    n_regimes: int,
    vocab_size: int,
) -> np.ndarray:
    n = int(n_regimes)
    v = int(vocab_size)
    if n <= 0 or v <= 0:
        raise ValueError("n_regimes and vocab_size must be positive")
    eye = np.eye(v, dtype=np.float64)
    uniform = np.full((v, v), 1.0 / float(v), dtype=np.float64)
    coprime_steps = [step for step in range(1, v) if math.gcd(step, v) == 1]
    if not coprime_steps:
        coprime_steps = [1]

    mats = np.empty((n, v, v), dtype=np.float64)
    for regime in range(n):
        step = int(coprime_steps[regime % len(coprime_steps)])
        reverse_step = int(coprime_steps[(regime + 1) % len(coprime_steps)])
        perm = np.zeros((v, v), dtype=np.float64)
        perm[np.arange(v), (np.arange(v) + step) % v] = 1.0
        reverse_perm = np.zeros((v, v), dtype=np.float64)
        reverse_perm[np.arange(v), (np.arange(v) - reverse_step) % v] = 1.0

        phase = float(regime) / float(max(1, n - 1)) if n > 1 else 0.5
        stay_weight = 0.30 + 0.18 * phase
        forward_weight = 0.44 - 0.10 * phase
        reverse_weight = 0.18 + 0.06 * (1.0 - phase)
        uniform_weight = 1.0 - stay_weight - forward_weight - reverse_weight
        mat = (
            float(stay_weight) * eye
            + float(forward_weight) * perm
            + float(reverse_weight) * reverse_perm
            + float(max(0.0, uniform_weight)) * uniform
        )
        mat = np.maximum(mat, 1e-12)
        mat /= np.sum(mat, axis=1, keepdims=True)
        mats[regime] = mat
    return mats


def _sample_hazard_segment_lengths(
    *,
    n_tokens: int,
    segment_regimes: Sequence[int],
    min_seg_len: int,
    max_seg_len: int,
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    n_segments = int(len(segment_regimes))
    if n_segments <= 0:
        raise ValueError("segment_regimes must be non-empty")
    n_regimes = int(max(1, max(int(x) for x in segment_regimes) + 1))
    if n_segments == 1:
        return (int(n_tokens),)

    mean_lengths = np.linspace(
        float(max(1, int(min_seg_len))),
        float(max(int(min_seg_len), int(max_seg_len))),
        num=n_regimes,
        dtype=np.float64,
    )
    lengths: List[int] = []
    remaining_tokens = int(n_tokens)
    remaining_segments = int(n_segments)

    for raw_regime in segment_regimes[:-1]:
        regime = int(raw_regime)
        min_allowed = max(
            int(min_seg_len),
            remaining_tokens - (remaining_segments - 1) * int(max_seg_len),
        )
        max_allowed = min(
            int(max_seg_len),
            remaining_tokens - (remaining_segments - 1) * int(min_seg_len),
        )
        if min_allowed > max_allowed:
            raise ValueError("Infeasible hazard segment-length draw under current constraints.")
        mean_len = float(mean_lengths[regime])
        hazard = 1.0 / float(max(1.0, mean_len))
        seg_len: int | None = None
        for _ in range(64):
            candidate = int(rng.geometric(hazard))
            if min_allowed <= candidate <= max_allowed:
                seg_len = int(candidate)
                break
        if seg_len is None:
            seg_len = int(np.clip(int(round(mean_len)), min_allowed, max_allowed))
        lengths.append(int(seg_len))
        remaining_tokens -= int(seg_len)
        remaining_segments -= 1

    final_len = int(remaining_tokens)
    if final_len < int(min_seg_len) or final_len > int(max_seg_len):
        raise ValueError("hazard segment sampler produced invalid final segment length")
    lengths.append(final_len)
    return tuple(lengths)


def _make_palette_emissions(
    *,
    n_regimes: int,
    vocab_size: int,
) -> np.ndarray:
    n = int(n_regimes)
    v = int(vocab_size)
    palette_size = max(2, min(v, max(2, v // max(1, n))))
    step = max(1, v // max(1, n))
    out = np.empty((n, v), dtype=np.float64)
    for regime in range(n):
        probs = np.full((v,), 0.15 / float(max(1, v - palette_size)), dtype=np.float64)
        active = [int((regime * step + offset) % v) for offset in range(palette_size)]
        for idx in active:
            probs[int(idx)] = 0.85 / float(palette_size)
        probs = np.maximum(probs, 1e-12)
        probs /= float(np.sum(probs))
        out[regime] = probs
    return out


def _make_disjoint_palette_emissions(
    *,
    n_regimes: int,
    vocab_size: int,
) -> np.ndarray:
    n = int(n_regimes)
    v = int(vocab_size)
    if n <= 0 or v <= 0:
        raise ValueError("n_regimes and vocab_size must be positive")
    if v < n:
        raise ValueError(
            "piecewise_disjoint_palette requires vocab_size >= n_regimes so each regime "
            "can own at least one observed token"
        )
    token_blocks = np.array_split(np.arange(v, dtype=np.int64), n)
    out = np.zeros((n, v), dtype=np.float64)
    for regime, block in enumerate(token_blocks):
        if block.size <= 0:
            raise ValueError("disjoint palette construction produced an empty regime block")
        out[int(regime), block] = 1.0 / float(block.size)
    return out


def _sample_segment_regimes_with_constraints(
    *,
    n_segments: int,
    n_regimes: int,
    rng: np.random.Generator,
    min_distinct_regimes_per_doc: Optional[int] = None,
    max_distinct_regimes_per_doc: Optional[int] = None,
) -> Tuple[int, ...]:
    configured_min = (
        None
        if min_distinct_regimes_per_doc is None
        else int(min_distinct_regimes_per_doc)
    )
    configured_max = (
        None
        if max_distinct_regimes_per_doc is None
        else int(max_distinct_regimes_per_doc)
    )
    if configured_min is None and configured_max is None:
        return _sample_segment_regimes(
            n_segments=int(n_segments),
            n_regimes=int(n_regimes),
            rng=rng,
        )

    if int(n_segments) <= 0:
        return tuple()
    feasible_max = int(
        min(
            int(n_regimes),
            int(n_segments),
            int(n_regimes) if configured_max is None else int(configured_max),
        )
    )
    feasible_min = int(
        max(
            2 if int(n_segments) > 1 else 1,
            1 if configured_min is None else int(configured_min),
        )
    )
    if feasible_min > feasible_max:
        raise ValueError(
            "distinct-regime segment sampling is infeasible with "
            f"n_segments={n_segments}, n_regimes={n_regimes}, "
            f"min_distinct_regimes_per_doc={configured_min}, "
            f"max_distinct_regimes_per_doc={configured_max}"
        )

    target_distinct = int(rng.integers(feasible_min, feasible_max + 1))
    selected = [int(x) for x in rng.choice(int(n_regimes), size=target_distinct, replace=False)]
    if int(target_distinct) == 1:
        return (int(selected[0]),)

    seq: List[int] = []
    used: set[int] = set()

    def _backtrack(prev: Optional[int]) -> bool:
        pos = int(len(seq))
        if pos >= int(n_segments):
            return int(len(used)) == int(target_distinct)
        remaining_after = int(n_segments) - pos - 1
        candidate_indices = rng.permutation(len(selected)).tolist()
        for idx in candidate_indices:
            candidate = int(selected[int(idx)])
            if prev is not None and int(candidate) == int(prev):
                continue
            used_after = set(used)
            used_after.add(int(candidate))
            unused_after = int(target_distinct) - int(len(used_after))
            if unused_after > remaining_after:
                continue
            seq.append(int(candidate))
            added = int(candidate) not in used
            if added:
                used.add(int(candidate))
            if _backtrack(int(candidate)):
                return True
            seq.pop()
            if added:
                used.remove(int(candidate))
        return False

    if not _backtrack(None):
        raise RuntimeError(
            "failed to sample a non-adjacent segment regime sequence with the requested "
            "distinct-regime constraints"
        )
    return tuple(int(x) for x in seq)


def _generate_piecewise_palette_docs(
    config: OPSCountConfig,
    *,
    n_docs: int,
    seed: int,
    emissions: np.ndarray,
) -> Tuple[ChangepointMarkovDoc, ...]:
    rng = np.random.default_rng(int(seed))
    docs: List[ChangepointMarkovDoc] = []
    for _ in range(n_docs):
        n_tokens = int(rng.integers(int(config.min_tokens), int(config.max_tokens) + 1))
        n_segments = _sample_num_segments(
            n_tokens=n_tokens,
            config=_GeneratorConfig(
                n_regimes=int(config.n_regimes),
                vocab_size=int(config.vocab_size),
                min_tokens=int(config.min_tokens),
                max_tokens=int(config.max_tokens),
                min_segments=int(config.min_segments),
                max_segments=int(config.max_segments),
                min_seg_len=int(config.min_seg_len),
                max_seg_len=int(config.max_seg_len),
                train_docs=0,
                test_docs=int(n_docs),
                seed=int(seed),
                sinkhorn_iters=int(config.sinkhorn_iters),
                transition_log_std=float(config.transition_log_std),
                use_cuda=False,
            ),
            rng=rng,
        )
        seg_lengths = _sample_segment_lengths(
            n_tokens=n_tokens,
            n_segments=n_segments,
            min_seg_len=int(config.min_seg_len),
            max_seg_len=int(config.max_seg_len),
            rng=rng,
        )
        seg_regimes = _sample_segment_regimes_with_constraints(
            n_segments=n_segments,
            n_regimes=int(config.n_regimes),
            rng=rng,
            min_distinct_regimes_per_doc=config.min_distinct_regimes_per_doc,
            max_distinct_regimes_per_doc=config.max_distinct_regimes_per_doc,
        )
        token_regimes = np.empty(int(n_tokens), dtype=np.int64)
        tokens = np.empty(int(n_tokens), dtype=np.int64)
        idx = 0
        for seg_len, regime in zip(seg_lengths, seg_regimes):
            regime_id = int(regime)
            token_regimes[idx : idx + int(seg_len)] = regime_id
            draw = rng.choice(
                int(config.vocab_size),
                size=int(seg_len),
                replace=True,
                p=emissions[regime_id],
            )
            tokens[idx : idx + int(seg_len)] = np.asarray(draw, dtype=np.int64)
            idx += int(seg_len)
        transition_regimes = token_regimes[1:].copy()
        boundaries = np.nonzero(token_regimes[:-1] != token_regimes[1:])[0].astype(np.int64)
        docs.append(
            ChangepointMarkovDoc(
                tokens=tuple(int(x) for x in tokens.tolist()),
                token_regimes=tuple(int(x) for x in token_regimes.tolist()),
                transition_regimes=tuple(int(x) for x in transition_regimes.tolist()),
                true_boundaries=tuple(int(x) for x in boundaries.tolist()),
            )
        )
    return tuple(docs)


def _generate_hazard_topic_docs(
    config: OPSCountConfig,
    *,
    n_docs: int,
    seed: int,
    emissions: np.ndarray,
) -> Tuple[ChangepointMarkovDoc, ...]:
    """Pure hazard-based topic model.

    At each token position, with probability ``hazard_rate`` the current
    topic switches to a uniformly random *different* topic.  The number
    of changepoints is therefore stochastic, driven by the hazard rate
    and document length.  ``hazard_rate`` can be set explicitly via
    ``hazard_switch_prob``; otherwise it is derived from the config's
    segment-length bounds as ``2 / (min_seg_len + max_seg_len)``.
    """
    rng = np.random.default_rng(int(seed))
    n_regimes = int(config.n_regimes)
    vocab_size = int(config.vocab_size)
    configured_switch_prob = float(
        getattr(config, "hazard_switch_prob", float("nan"))
    )
    if math.isfinite(configured_switch_prob):
        hazard_rate = float(min(1.0, max(0.0, configured_switch_prob)))
    else:
        mean_seg_len = 0.5 * (float(config.min_seg_len) + float(config.max_seg_len))
        hazard_rate = 1.0 / max(1.0, mean_seg_len)
    docs: List[ChangepointMarkovDoc] = []
    for _ in range(int(n_docs)):
        n_tokens = int(
            rng.integers(int(config.min_tokens), int(config.max_tokens) + 1)
        )
        # Build per-token regime sequence via hazard process.
        token_regimes = np.empty(n_tokens, dtype=np.int64)
        current_regime = int(rng.integers(0, n_regimes))
        token_regimes[0] = current_regime
        for t in range(1, n_tokens):
            if rng.random() < hazard_rate:
                # Switch to a different topic uniformly at random.
                candidates = [r for r in range(n_regimes) if r != current_regime]
                current_regime = int(rng.choice(candidates)) if candidates else current_regime
            token_regimes[t] = current_regime
        # Emit tokens from the per-regime emission distributions.
        tokens = np.empty(n_tokens, dtype=np.int64)
        for regime_id in range(n_regimes):
            mask = token_regimes == regime_id
            count = int(mask.sum())
            if count > 0:
                tokens[mask] = rng.choice(
                    vocab_size, size=count, replace=True, p=emissions[regime_id],
                )
        transition_regimes = token_regimes[1:].copy()
        boundaries = np.nonzero(
            token_regimes[:-1] != token_regimes[1:]
        )[0].astype(np.int64)
        docs.append(
            ChangepointMarkovDoc(
                tokens=tuple(int(x) for x in tokens.tolist()),
                token_regimes=tuple(int(x) for x in token_regimes.tolist()),
                transition_regimes=tuple(int(x) for x in transition_regimes.tolist()),
                true_boundaries=tuple(int(x) for x in boundaries.tolist()),
            )
        )
    return tuple(docs)


def _generate_hazard_aliased_docs(
    config: OPSCountConfig,
    *,
    n_docs: int,
    seed: int,
    transitions: np.ndarray,
) -> Tuple[ChangepointMarkovDoc, ...]:
    rng = np.random.default_rng(int(seed))
    cdfs = np.cumsum(np.asarray(transitions, dtype=np.float64), axis=2)
    cdfs[:, :, -1] = 1.0

    docs: List[ChangepointMarkovDoc] = []
    for _ in range(n_docs):
        n_tokens = int(rng.integers(int(config.min_tokens), int(config.max_tokens) + 1))
        n_segments = _sample_num_segments(
            n_tokens=n_tokens,
            config=_GeneratorConfig(
                n_regimes=int(config.n_regimes),
                vocab_size=int(config.vocab_size),
                min_tokens=int(config.min_tokens),
                max_tokens=int(config.max_tokens),
                min_segments=int(config.min_segments),
                max_segments=int(config.max_segments),
                min_seg_len=int(config.min_seg_len),
                max_seg_len=int(config.max_seg_len),
                train_docs=0,
                test_docs=int(n_docs),
                seed=int(seed),
                sinkhorn_iters=int(config.sinkhorn_iters),
                transition_log_std=float(config.transition_log_std),
                use_cuda=False,
            ),
            rng=rng,
        )
        seg_regimes = _sample_segment_regimes_with_constraints(
            n_segments=n_segments,
            n_regimes=int(config.n_regimes),
            rng=rng,
            min_distinct_regimes_per_doc=config.min_distinct_regimes_per_doc,
            max_distinct_regimes_per_doc=config.max_distinct_regimes_per_doc,
        )
        seg_lengths = _sample_hazard_segment_lengths(
            n_tokens=n_tokens,
            segment_regimes=seg_regimes,
            min_seg_len=int(config.min_seg_len),
            max_seg_len=int(config.max_seg_len),
            rng=rng,
        )

        token_regimes = np.empty(int(n_tokens), dtype=np.int64)
        idx = 0
        for seg_len, regime in zip(seg_lengths, seg_regimes):
            token_regimes[idx : idx + int(seg_len)] = int(regime)
            idx += int(seg_len)

        tokens = np.empty(int(n_tokens), dtype=np.int64)
        tokens[0] = int(rng.integers(0, int(config.vocab_size)))
        transition_regimes = token_regimes[1:].copy()
        for t in range(int(n_tokens) - 1):
            regime = int(transition_regimes[t])
            row = cdfs[regime, int(tokens[t])]
            u = float(rng.random())
            nxt = int(np.searchsorted(row, u, side="right"))
            if nxt >= int(config.vocab_size):
                nxt = int(config.vocab_size) - 1
            tokens[t + 1] = nxt

        boundaries = np.nonzero(token_regimes[:-1] != token_regimes[1:])[0].astype(np.int64)
        docs.append(
            ChangepointMarkovDoc(
                tokens=tuple(int(x) for x in tokens.tolist()),
                token_regimes=tuple(int(x) for x in token_regimes.tolist()),
                transition_regimes=tuple(int(x) for x in transition_regimes.tolist()),
                true_boundaries=tuple(int(x) for x in boundaries.tolist()),
            )
        )
    return tuple(docs)


def _build_generator_transitions(
    config: OPSCountConfig,
    *,
    seed: int,
) -> np.ndarray:
    profile = _generator_profile_key(str(config.generator_profile))
    if profile == "piecewise_palette":
        return _make_palette_emissions(
            n_regimes=int(config.n_regimes),
            vocab_size=int(config.vocab_size),
        )
    if profile in {"piecewise_disjoint_palette", "hazard_topic"}:
        return _make_disjoint_palette_emissions(
            n_regimes=int(config.n_regimes),
            vocab_size=int(config.vocab_size),
        )
    if profile == "hazard_aliased":
        return _make_aliased_transition_matrices(
            n_regimes=int(config.n_regimes),
            vocab_size=int(config.vocab_size),
        )
    rng = np.random.default_rng(int(seed))
    return _make_transition_matrices(
        n_classes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )


def _generate_ops_count_docs(
    config: OPSCountConfig,
    *,
    n_docs: int,
    seed: int,
    transitions: np.ndarray,
) -> Tuple[ChangepointMarkovDoc, ...]:
    profile = _generator_profile_key(str(config.generator_profile))
    if profile in {"piecewise_palette", "piecewise_disjoint_palette"}:
        return _generate_piecewise_palette_docs(
            config,
            n_docs=int(n_docs),
            seed=int(seed),
            emissions=transitions,
        )
    if profile == "hazard_topic":
        return _generate_hazard_topic_docs(
            config,
            n_docs=int(n_docs),
            seed=int(seed),
            emissions=transitions,
        )
    if profile == "hazard_aliased":
        return _generate_hazard_aliased_docs(
            config,
            n_docs=int(n_docs),
            seed=int(seed),
            transitions=transitions,
        )
    gen_cfg = _GeneratorConfig(
        n_regimes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_len=int(config.min_seg_len),
        max_seg_len=int(config.max_seg_len),
        train_docs=0,
        test_docs=int(n_docs),
        seed=int(seed),
        sinkhorn_iters=int(config.sinkhorn_iters),
        transition_log_std=float(config.transition_log_std),
        use_cuda=False,
    )
    return generate_changepoint_docs(gen_cfg, transitions=transitions)


@dataclass(frozen=True)
class OPSCountConfig:
    problem_id: str = "markov_ops_count"
    method_id: str = "tree_neural"
    law_set_id: str = "all"
    # Document generator.
    n_regimes: int = 4
    vocab_size: int = 96
    generator_profile: GeneratorProfileName = "piecewise_markov"
    min_tokens: int = 384
    max_tokens: int = 384
    min_segments: int = 12
    max_segments: int = 24
    min_seg_len: int = 8
    max_seg_len: int = 32
    hazard_switch_prob: float = float("nan")
    min_distinct_regimes_per_doc: Optional[int] = None
    max_distinct_regimes_per_doc: Optional[int] = None
    sinkhorn_iters: int = 30
    transition_log_std: float = 1.25

    # Realized partition (fixed leaves).
    fixed_leaf_tokens: int = 16

    # Training / eval sizes.
    train_docs: int = 1000
    val_docs: int = 0
    test_docs: int = 1000
    data_seed: Optional[int] = None
    model_seed: Optional[int] = None
    val_seed_offset: int = 5_000
    test_seed_offset: int = 10_000

    # Learned sketch settings.
    model_family: ModelFamilyName = (
        "neural"  # "neural"/"fno" → FNOCountSketch; "additive" → AdditiveCountSketch.
    )
    feature_mode: str = (
        "full"  # "full" uses latent regime features; token_* modes use observed-token features only.
    )
    state_dim: int = 32
    hidden_dim: int = 128
    n_epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-5
    # Legacy direct term weights. The generic baseline is intentionally root-only:
    # pre-local-law runs should correspond to zero theorem-term weight unless the
    # caller opts into local supervision explicitly.
    #
    # If `local_law_weight` is set, these are only used to preserve old configs
    # in summaries and are not the active objective.
    c3_weight: float = 0.0
    c2_weight: float = 0.0
    leaf_weight: float = 0.0
    law_package: str = ""
    # Formal theorem-facing parameterization of the local-law bundle:
    # either (1 - λ) * root_objective + λ * equal_active_local_laws, or a
    # normalized explicit root/C1/C2/C3 convex objective when λ is unset.
    # `task_objective_weight` is accepted only in explicit-weight mode.
    local_law_weight: Optional[float] = None
    task_objective_weight: Optional[float] = None
    c1_relative_weight: float = 1.0
    c2_relative_weight: float = 1.0
    c3_relative_weight: float = 1.0
    # Legacy direct root-term weight. In the normalized theorem-facing parameterization,
    # the active root weight is `1 - local_law_weight`; this field is retained for
    # backward-compatible legacy runs and reporting.
    root_weight: float = 1.0
    # RL-style depth discount factor for local-law supervision (Lean:
    # DiscountedTreeMetaObjective).  depth_discount_gamma=1.0 reproduces the
    # current flat weighting.  Lower values down-weight deeper nodes: leaf
    # supervision at depth d receives gamma^d, merge supervision likewise.
    # gamma=0.0 zeroes out all local-law terms (root-only training).
    depth_discount_gamma: float = 1.0
    # When True, decode_summary/encode_summary are learned MLPs (same in/out
    # dim) so the C2 re-summarization cycle is non-trivial.  When False,
    # decode/encode are identity (backward-compatible).
    # Lean L3: for Z in range(g), re-summarizing Z preserves oracle value.
    c2_learned_resummary: bool = False
    # Proxy-only associativity regularizer; not a Lean local law.
    schedule_consistency_weight: float = 0.0
    grad_clip_norm: float = 1.0
    exact_family: str = ""

    # Node-label budgets (oracle queries).
    audit_policy: AuditPolicyName = "fraction"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 0.2
    audit_scale: float = 1.0
    c3_audit_strategy: C3AuditStrategyName = "uniform"
    c3_include_root: bool = True
    leaf_query_rate: float = 1.0
    include_root_query: bool = True
    # Inference-time oracle visibility sweep on realized internal nodes.
    eval_guidance_qs: Tuple[float, ...] = tuple()
    eval_guidance_trials: int = 0
    eval_guidance_seed_offset: int = 100_000
    eval_guidance_include_root: bool = True
    guidance_override_mode: GuidanceOverrideModeName = (
        "reset"  # for neural sketches: reset vs adjust along readout
    )

    # Simple classical baseline (doc-level root regression).
    include_rf_root_baseline: bool = False
    include_doc_level_baseline: bool = False
    include_doc_sequence_baseline: bool = False
    include_doc_transformer_baseline: bool = False
    include_fno_baseline: bool = False
    include_deeponet_baseline: bool = False
    include_mlp_bigram_baseline: bool = False
    include_cnn1d_baseline: bool = False
    include_doc_level_ridge_baseline: bool = False
    include_leaf_ridge_tree_baseline: bool = False
    include_leaf_knn_tree_baseline: bool = False
    include_leaf_endpoint_table_tree_baseline: bool = False
    include_leaf_dt_tree_baseline: bool = False
    include_leaf_rf_tree_baseline: bool = False
    include_sampled_leaf_pool_ridge_baseline: bool = False
    include_sampled_leaf_pool_rf_baseline: bool = False
    official_fno_preserve_requested_leaf_tokens: bool = False
    preserve_requested_leaf_tokens: bool = False
    comparison_mode: ComparisonModeName = "legacy"
    rf_n_estimators: int = 200
    rf_max_depth: int = 16
    rf_min_samples_leaf: int = 5
    doc_level_ridge_alpha: float = 1.0
    doc_level_ridge_breakdown_orders: Tuple[int, ...] = tuple()
    doc_sequence_objective: DocSequenceObjectiveName = "count_ce_only"
    doc_sequence_fno_pooling: DocSequenceFNOPoolingName = "mean"
    doc_sequence_fno_concat_length_feature: bool = False
    doc_sequence_fno_include_transition_channel: bool = False
    tree_document_loss_normalization_mode: TreeDocumentLossNormalizationModeName = "auto"
    doc_transformer_head_family: DocTransformerHeadFamilyName = "boundary_sum_count_hybrid"
    doc_transformer_layers: int = 0
    leaf_knn_neighbors: int = 32
    sampled_leaf_pool_leaf_counts: Tuple[int, ...] = tuple()
    sampled_leaf_pool_seed_offset: int = 200_000

    # FNO tree-merge model hyperparameters (used when model_family="neural" or "fno").
    fno_width: int = 64
    fno_n_modes: int = 8
    fno_n_layers: int = 2
    tree_leaf_fno_width: Optional[int] = None
    tree_leaf_fno_n_modes: Optional[int] = None
    tree_leaf_fno_n_layers: Optional[int] = None
    tree_leaf_fno_pooling: Optional[str] = None
    tree_root_supervision_kind: TreeRootSupervisionKind = "mse"
    tree_checkpoint_metric: TreeCheckpointMetricName = "val_root_mae"
    tree_stage1_checkpoint_metric: TreeCheckpointMetricName = "val_root_mae"
    tree_stage1_eval_mode: TreeStage1EvalModeName = "per_epoch"
    tree_stage1_screen_doc_limit: int = 0
    tree_stage1_final_exact_doc_limit: int = 0
    exact_metric_selection_doc_limit: int = 0
    exact_metric_selection_interval: int = 1
    exact_metric_final_doc_limit: int = 0
    tree_exact_eval_max_docs: int = 0
    tree_posttrain_train_doc_limit: int = 0
    tree_batch_pack_mode: TreeBatchPackModeName = "structure_bucket"
    tree_batch_runtime_mode: str = "legacy"
    tree_model_version: str = "legacy"
    tree_batch_token_budget: int = 0
    tree_batch_node_budget: int = 0
    tree_batch_autotune: bool = True
    tree_batch_structural_pad_limit: float = 0.5
    tree_batch_auto_queue_min_docs: int = 8
    tree_batch_auto_queue_min_fill_ratio: float = 0.5
    tree_eval_workers_per_mig: int = 0
    gpu_runtime_data_mode: str = "resident"
    gpu_runtime_bucket_mode: str = "exact_then_bucketed"
    gpu_runtime_preload_splits: Tuple[str, ...] = ("train", "val", "test")
    gpu_runtime_preload_targets: bool = True
    gpu_runtime_workers_per_mig: int = 1
    gpu_runtime_allow_multi_worker_screen: bool = True
    gpu_runtime_capacity_workers_per_mig: int = 2
    tree_stage1_artifact_dir: str = ""
    tree_stage1_artifact_root: str = ""
    tree_stage1_resume_if_available: bool = True
    prepared_data_root: str = ""
    prepared_data_allow_create: bool = True
    prepared_data_signature: str = ""
    diagnostic_detail_mode: DiagnosticDetailModeName = "summary"
    posttrain_diagnostics_mode: PosttrainDiagnosticsModeName = ""
    raw_diagnostic_artifact_dir: str = ""
    tree_stage1_root_weight: float = 0.0
    tree_join_bit_weight: float = 0.0
    tree_training_schedule: TreeTrainingScheduleName = "two_stage"
    tree_stage1_epochs: int = 12
    tree_stage2_epochs: int = 20
    tree_task_head_mode: TreeTaskHeadModeName = "full_state_scalar"
    tree_theorem_surface_mode: TreeTheoremSurfaceModeName = "slotwise"
    tree_theorem_count_head_mode: TreeTheoremCountHeadModeName = "scalar_mse"
    tree_theorem_count_ordinal_weight: float = 1.0
    tree_theorem_count_scalar_aux_weight: float = 0.25
    tree_theorem_count_threshold_balance: bool = True
    tree_summary_spec_root_mode: TreeSummarySpecRootModeName = "task_split_ablation"
    tree_theorem_feature_dim: int = 48
    tree_theorem_feature_hidden_dim: int = 256
    tree_merge_hidden_dim: int = 0
    tree_theorem_score_dim: int = 0
    tree_theorem_fiber_dim: int = 0
    tree_theorem_aux_dim: int = 0
    tree_phi_compose_weight: float = 1.0
    tree_phi_contrastive_weight: float = 0.25
    tree_phi_alignment_loss: TreePhiAlignmentLossName = "cosine_mse"
    tree_c2_mode: str = "reconstruction"
    oracle_metric_name: str = ""
    oracle_same_threshold: float = 0.0
    oracle_diff_threshold: float = 0.0
    theorem_feature_adapter: str = DEFAULT_THEOREM_FEATURE_ADAPTER
    theorem_pair_same_threshold: float | None = None
    theorem_pair_diff_threshold: float | None = None
    aligned_sketch_surface: AlignedSketchSurfaceName = ""
    summary_spec_name: SummarySpecName = ""
    slot_count: int = 0
    tree_theorem_count_dim: int = 0
    tree_theorem_first_dim: int = 0
    tree_theorem_last_dim: int = 0
    tree_local_weighting_mode: TreeLocalWeightingModeName = "fixed_k_hajek"
    tree_supervision_source: TreeSupervisionSourceName = "rate"
    tree_exact_collapse_mode: TreeExactCollapseModeName = ""
    leaf_supervision_kind: LeafSupervisionKindName = "full_sketch"
    leaf_label_rate: float = 1.0
    internal_supervision_kind: InternalSupervisionKindName = "none"
    internal_label_rate: float = 0.0
    leaf_exact_supervision: bool = False
    endpoint_loss_scale: float = 1.0

    # Unified local-law objective for FNO tree supervision.
    ipw_leaf_sample_rate: float = 1.0
    ipw_internal_sample_rate: float = 1.0
    local_law_objective_mode: LocalLawObjectiveModeName = LOCAL_LAW_OBJECTIVE_CORRECTED
    use_residual_decomposition: bool = True
    root_only_train_fraction: float = 0.0
    # Fraction of training docs routed through the in-model full-document doc-sequence objective.
    doc_sequence_train_fraction: float = 0.0
    # Corpus-level training supervision budget.
    budget_total_calls: int = 0
    budget_total_calls_per_doc: float = 0.0
    mass_target_per_doc: float = float("nan")
    full_doc_budget_share: float = 1.0
    doc_consumption_mode: BudgetedDocConsumptionModeName = ""
    package_semantics: str = ""
    local_split_mode: BudgetedLocalSplitModeName = ""
    local_allocation_policy: BudgetedAllocationPolicyName = ""
    max_internal_depth: int = 0  # 0 = unlimited; >0 limits internal nodes to this tree depth

    # Evaluation / audit thresholds.
    violation_tau: float = 0.0
    suite_role: str = ""
    artifact_dir: str = ""
    tree_progress_snapshot_interval: int = 10
    save_logged_observations: bool = False

    # Runtime.
    seed: int = 0
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    torch_threads: int = 0


@dataclass(frozen=True)
class SketchMetrics:
    root_mae: float
    root_median_abs_error: float
    root_p95_abs_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float
    leaf_mae: float
    leaf_violation_rate: float
    c2_idempotence_mae: float
    c2_r1_mae: float
    c2_r2_mae: float
    c2_r4_mae: float
    resummary_root_drift_r1: float
    resummary_root_drift_r2: float
    resummary_root_drift_r4: float
    merge_mae: float
    merge_violation_rate: float
    n_docs: int
    c2_state_replay_mse: float = 0.0
    c2_bottleneck_reconstruction_mse: float = 0.0
    root_mse: float = float("nan")
    condition_root_mae: Dict[str, float] = field(default_factory=dict)
    condition_root_n_docs: Dict[str, int] = field(default_factory=dict)
    condition_root_macro_mae: float = float("nan")
    condition_root_worst_mae: float = float("nan")

    @property
    def c2_count_drift_r1_mae(self) -> float:
        return float(self.c2_r1_mae)

    @property
    def c2_count_drift_r2_mae(self) -> float:
        return float(self.c2_r2_mae)

    @property
    def c2_count_drift_r4_mae(self) -> float:
        return float(self.c2_r4_mae)

    @property
    def c2_root_count_drift_r1_mae(self) -> float:
        return float(self.resummary_root_drift_r1)

    @property
    def c2_root_count_drift_r2_mae(self) -> float:
        return float(self.resummary_root_drift_r2)

    @property
    def c2_root_count_drift_r4_mae(self) -> float:
        return float(self.resummary_root_drift_r4)


def _sketch_metric_alias_payload(metrics: SketchMetrics) -> Dict[str, float]:
    return {
        "c2_count_drift_r1_mae": float(metrics.c2_count_drift_r1_mae),
        "c2_count_drift_r2_mae": float(metrics.c2_count_drift_r2_mae),
        "c2_count_drift_r4_mae": float(metrics.c2_count_drift_r4_mae),
        "c2_root_count_drift_r1_mae": float(metrics.c2_root_count_drift_r1_mae),
        "c2_root_count_drift_r2_mae": float(metrics.c2_root_count_drift_r2_mae),
        "c2_root_count_drift_r4_mae": float(metrics.c2_root_count_drift_r4_mae),
    }


@dataclass(frozen=True)
class TrainingGeometry:
    mean_tokens: float
    mean_leaves: float
    mean_internal_nodes: float
    mean_leaf_labels: float
    mean_internal_labels: float
    mean_queries_per_doc: float
    root_queries_total: int
    leaf_labels_total: int
    internal_labels_total: int
    total_queries_estimate: int


@dataclass(frozen=True)
class BudgetedTrainSupervisionDocPlan:
    doc_index: int
    doc_tokens: int
    document_mode: str = ""
    leaf_indices: Tuple[int, ...] = tuple()
    internal_indices: Tuple[int, ...] = tuple()
    raw_call_cost: int = 0
    document_mass: float = 0.0
    leaf_mass: float = 0.0
    internal_mass: float = 0.0
    leaf_propensity: float = 0.0
    internal_propensity: float = 0.0
    effective_full_doc_mass: float = 0.0


@dataclass(frozen=True)
class BudgetedTrainSupervisionManifest:
    budget_total_calls: int
    budget_total_calls_per_doc: float
    budget_total_calls_used: int
    budget_utilization: float
    full_doc_budget_share: float
    full_doc_calls_requested: int
    full_doc_calls_total: int
    local_calls_requested: int
    local_calls_total: int
    doc_consumption_mode: str
    local_split_mode: str
    local_allocation_policy: str
    sampling_scheme: str
    doc_touch_rate: float
    mean_labels_per_touched_doc: float
    touched_docs_total: int
    effective_full_doc_mass_total: float
    effective_full_doc_mass_per_doc: float
    document_mass_share: float
    leaf_mass_share: float
    internal_mass_share: float
    document_call_share: float
    leaf_call_share: float
    internal_call_share: float
    actual_doc_tokens_mean: float = 0.0
    actual_doc_tokens_unique: Tuple[int, ...] = tuple()
    supervision_source: str = ""
    local_estimand_mode: str = ""
    package_semantics: str = ""
    mass_target_per_doc: float = float("nan")
    requested_root_mass_per_doc: float = 0.0
    realized_root_mass_per_doc: float = 0.0
    realized_leaf_mass_per_doc: float = 0.0
    realized_internal_mass_per_doc: float = 0.0
    leaf_propensity_mean: float = 0.0
    internal_propensity_mean: float = 0.0
    local_weight_ess: float = 0.0
    local_weight_max: float = 0.0
    doc_plans: Tuple[BudgetedTrainSupervisionDocPlan, ...] = tuple()


@dataclass(frozen=True)
class EstimatorDiagnostics:
    true_mean: float
    naive_bias: float
    ipw_bias: float
    dsl_bias: float
    ipw_var: float
    dsl_var: float


@dataclass(frozen=True)
class TrainFitDiagnostics:
    train_loss_final: float
    train_loss_curve: Tuple[float, ...]
    epochs_completed: int
    selection_metric_curve: Tuple[float, ...] = tuple()
    selection_mode: str = "final_epoch_no_validation"
    selection_split: str = "config"
    selection_metric_name: str = "train_loss_final"
    selection_metric_value: float = float("nan")
    best_epoch: int = 0
    train_exact_match_rate: float = float("nan")
    val_exact_match_rate: float = float("nan")
    test_exact_match_rate: float = float("nan")
    stage1_selection_metric_curve: Tuple[float, ...] = tuple()
    stage2_selection_metric_curve: Tuple[float, ...] = tuple()
    stage1_selection_metric_name: str = ""
    stage2_selection_metric_name: str = ""
    training_schedule: str = ""


@dataclass(frozen=True)
class ObjectiveMetrics:
    optimization_total_loss: float
    optimization_root_loss: float
    optimization_leaf_loss: float
    optimization_c2_loss: float
    optimization_merge_loss: float
    optimization_schedule_consistency_loss: float
    raw_total_loss: float
    raw_root_loss: float
    raw_leaf_loss: float
    raw_c2_loss: float
    raw_merge_loss: float
    raw_schedule_consistency_loss: float
    n_docs: int


@dataclass(frozen=True)
class OPSCountSummary:
    config: Dict[str, object]
    training_geometry: Dict[str, float | int]
    objective: Dict[str, object]
    metrics: Dict[str, object]
    estimator_diagnostics: Dict[str, float]
    local_law_learnability: Dict[str, object] = field(default_factory=dict)
    g_artifacts: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = _markov_public_summary_payload(self)
        assert_public_contract_clean(payload, surface="markov ops-count summary")
        return json.dumps(payload, indent=2, sort_keys=True)


def _markov_public_summary_payload(summary: OPSCountSummary) -> Dict[str, object]:
    """Return the public Markov summary with legacy axis/objective keys removed."""

    def clean(value: object) -> object:
        if isinstance(value, Mapping):
            out: Dict[str, object] = {}
            for raw_key, raw_child in dict(value).items():
                key = str(raw_key)
                if key == "family":
                    out.setdefault("method_id", clean(raw_child))
                    continue
                if key == "dgp":
                    out.setdefault("problem_id", clean(raw_child))
                    continue
                if key == "law_package":
                    try:
                        out.setdefault(
                            "law_set_id",
                            canonical_law_set_id(str(raw_child), allow_aliases=True),
                        )
                    except Exception:
                        out.setdefault("law_set_id", str(raw_child))
                    continue
                if key.startswith(("law_c", "local_law_c", "objective_local_law_c")):
                    continue
                if key.startswith("tree_") and key.endswith("_weight"):
                    continue
                if key in PUBLIC_CONTRACT_LEGACY_FIELDS:
                    continue
                if key in ORACLE_OBSERVATION_DESIGN_PARAMETER_FIELDS:
                    continue
                out[key] = clean(raw_child)
            return out
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        if isinstance(value, str):
            return (
                value.replace("law_package", "law_set_id")
                .replace("root_only_reference_package", "root_only_reference_law_set_id")
                .replace("all_laws_reference_package", "all_laws_reference_law_set_id")
            )
        return value

    objective = clean(summary.objective)
    if isinstance(objective, Mapping):
        component_weights = dict(objective.get("local_law_component_weights", {}) or {})
        if component_weights:
            objective = {
                **dict(objective),
                "local_law_component_weights": {
                    str(k): float(v) for k, v in component_weights.items()
                },
            }

    payload = {
        "config": clean(summary.config),
        "training_geometry": clean(summary.training_geometry),
        "objective": objective,
        "metrics": clean(summary.metrics),
        "estimator_diagnostics": clean(summary.estimator_diagnostics),
    }
    if summary.local_law_learnability:
        payload["local_law_learnability"] = clean(summary.local_law_learnability)
    if summary.g_artifacts:
        payload["g_artifacts"] = clean(summary.g_artifacts)
    return payload


@dataclass(frozen=True)
class MarkovOPSDataBundle:
    train_docs: Tuple[ChangepointMarkovDoc, ...]
    val_docs: Tuple[ChangepointMarkovDoc, ...]
    test_docs: Tuple[ChangepointMarkovDoc, ...]
    train_corpus_signature: str
    val_corpus_signature: str
    test_corpus_signature: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "train_docs": [_markov_doc_signature_payload(doc) for doc in self.train_docs],
            "val_docs": [_markov_doc_signature_payload(doc) for doc in self.val_docs],
            "test_docs": [_markov_doc_signature_payload(doc) for doc in self.test_docs],
            "train_corpus_signature": str(self.train_corpus_signature),
            "val_corpus_signature": str(self.val_corpus_signature),
            "test_corpus_signature": str(self.test_corpus_signature),
            "metadata": dict(getattr(self, "metadata", {}) or {}),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path).endswith(".pkl"):
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MarkovOPSDataBundle":
        def _load_docs(key: str) -> Tuple[ChangepointMarkovDoc, ...]:
            raw_docs = list(payload.get(key) or [])
            docs: List[ChangepointMarkovDoc] = []
            for raw in raw_docs:
                item = dict(raw or {})
                docs.append(
                    ChangepointMarkovDoc(
                        tokens=tuple(int(x) for x in (item.get("tokens") or [])),
                        token_regimes=tuple(int(x) for x in (item.get("token_regimes") or [])),
                        transition_regimes=tuple(
                            int(x) for x in (item.get("transition_regimes") or [])
                        ),
                        true_boundaries=tuple(int(x) for x in (item.get("true_boundaries") or [])),
                    )
                )
            return tuple(docs)

        return cls(
            train_docs=_load_docs("train_docs"),
            val_docs=_load_docs("val_docs"),
            test_docs=_load_docs("test_docs"),
            train_corpus_signature=str(payload.get("train_corpus_signature", "")),
            val_corpus_signature=str(payload.get("val_corpus_signature", "")),
            test_corpus_signature=str(payload.get("test_corpus_signature", "")),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: Path) -> "MarkovOPSDataBundle":
        if str(path).endswith(".pkl"):
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _markov_doc_signature_payload(doc: ChangepointMarkovDoc) -> Dict[str, object]:
    return {
        "tokens": [int(x) for x in doc.tokens],
        "token_regimes": [int(x) for x in doc.token_regimes],
        "transition_regimes": [int(x) for x in doc.transition_regimes],
        "true_boundaries": [int(x) for x in doc.true_boundaries],
    }


def _markov_corpus_signature(docs: Sequence[ChangepointMarkovDoc]) -> str:
    h = hashlib.sha256()
    for doc in docs:
        h.update(
            json.dumps(
                _markov_doc_signature_payload(doc),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def _condition_error_diagnostics(
    errors: Sequence[float] | np.ndarray,
    condition_ids: Sequence[str] | None,
) -> Dict[str, object]:
    err = np.asarray([float(value) for value in errors], dtype=np.float64).reshape(-1)
    ids = [str(value) for value in list(condition_ids or [])]
    if err.size == 0 or len(ids) != int(err.size):
        return {
            "condition_root_mae": {},
            "condition_root_n_docs": {},
            "condition_root_macro_mae": float("nan"),
            "condition_root_worst_mae": float("nan"),
        }
    by_condition: Dict[str, List[float]] = {}
    for condition_id, value in zip(ids, err.tolist()):
        by_condition.setdefault(str(condition_id), []).append(float(value))
    condition_mae = {
        key: float(np.mean(np.asarray(values, dtype=np.float64)))
        for key, values in sorted(by_condition.items())
    }
    condition_n_docs = {
        key: int(len(values)) for key, values in sorted(by_condition.items())
    }
    mae_values = np.asarray(list(condition_mae.values()), dtype=np.float64)
    return {
        "condition_root_mae": condition_mae,
        "condition_root_n_docs": condition_n_docs,
        "condition_root_macro_mae": float(np.mean(mae_values)) if mae_values.size else float("nan"),
        "condition_root_worst_mae": float(np.max(mae_values)) if mae_values.size else float("nan"),
    }


def _condition_ids_for_split(
    data_bundle: MarkovOPSDataBundle,
    split: str,
    n_docs: int,
) -> tuple[str, ...]:
    metadata = dict(getattr(data_bundle, "metadata", {}) or {})
    condition_ids_by_split = dict(metadata.get("condition_ids") or {})
    raw_ids = list(condition_ids_by_split.get(str(split), []) or [])
    if len(raw_ids) < int(n_docs):
        return tuple()
    return tuple(str(value) for value in raw_ids[: int(n_docs)])


def _root_count_diagnostics(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    condition_ids: Sequence[str] | None = None,
) -> Dict[str, object]:
    if not docs:
        return {
            "n_docs": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "n_unique": 0,
            "is_constant": False,
            "histogram": {},
            "quantiles": {},
            "global_mean_baseline_mae": float("nan"),
            "condition_mean_baseline_mae": float("nan"),
            "condition_macro_mean_baseline_mae": float("nan"),
            "mean_guess_gap": float("nan"),
            "condition_diagnostics": {},
        }
    counts = np.asarray(
        [float(_oracle_count(doc, start=0, end=len(doc.tokens))) for doc in docs],
        dtype=np.float64,
    )
    unique = np.unique(counts)
    hist_values, hist_counts = np.unique(np.rint(counts).astype(np.int64), return_counts=True)
    global_mean = float(np.mean(counts))
    global_mean_baseline_mae = float(np.mean(np.abs(counts - global_mean)))
    diagnostics: Dict[str, object] = {
        "n_docs": int(counts.size),
        "min": float(np.min(counts)),
        "max": float(np.max(counts)),
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "n_unique": int(unique.size),
        "is_constant": bool(unique.size <= 1),
        "histogram": {
            str(int(value)): int(count)
            for value, count in zip(hist_values.tolist(), hist_counts.tolist())
        },
        "quantiles": {
            "p00": float(np.percentile(counts, 0.0)),
            "p25": float(np.percentile(counts, 25.0)),
            "p50": float(np.percentile(counts, 50.0)),
            "p75": float(np.percentile(counts, 75.0)),
            "p90": float(np.percentile(counts, 90.0)),
            "p95": float(np.percentile(counts, 95.0)),
            "p100": float(np.percentile(counts, 100.0)),
        },
        "global_mean_baseline_mae": float(global_mean_baseline_mae),
        "condition_mean_baseline_mae": float("nan"),
        "condition_macro_mean_baseline_mae": float("nan"),
        "mean_guess_gap": float("nan"),
        "condition_diagnostics": {},
    }
    ids = [str(value) for value in list(condition_ids or [])]
    if len(ids) == int(counts.size):
        by_condition: Dict[str, List[float]] = {}
        for condition_id, count in zip(ids, counts.tolist()):
            by_condition.setdefault(str(condition_id), []).append(float(count))
        condition_abs: List[float] = []
        condition_mean_baselines: List[float] = []
        condition_payload: Dict[str, object] = {}
        for condition_id, values in sorted(by_condition.items()):
            arr = np.asarray(values, dtype=np.float64)
            cond_mean = float(np.mean(arr))
            cond_abs = np.abs(arr - cond_mean)
            condition_abs.extend(float(value) for value in cond_abs.tolist())
            condition_mean_baselines.append(float(np.mean(cond_abs)))
            c_hist_values, c_hist_counts = np.unique(
                np.rint(arr).astype(np.int64),
                return_counts=True,
            )
            condition_payload[str(condition_id)] = {
                "n_docs": int(arr.size),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(cond_mean),
                "std": float(np.std(arr)),
                "n_unique": int(np.unique(arr).size),
                "is_constant": bool(np.unique(arr).size <= 1),
                "mean_baseline_mae": float(np.mean(cond_abs)),
                "histogram": {
                    str(int(value)): int(count)
                    for value, count in zip(c_hist_values.tolist(), c_hist_counts.tolist())
                },
            }
        condition_mean_baseline_mae = (
            float(np.mean(np.asarray(condition_abs, dtype=np.float64)))
            if condition_abs
            else float("nan")
        )
        condition_macro_mean_baseline_mae = (
            float(np.mean(np.asarray(condition_mean_baselines, dtype=np.float64)))
            if condition_mean_baselines
            else float("nan")
        )
        diagnostics.update(
            {
                "condition_mean_baseline_mae": float(condition_mean_baseline_mae),
                "condition_macro_mean_baseline_mae": float(condition_macro_mean_baseline_mae),
                "mean_guess_gap": (
                    float(global_mean_baseline_mae - condition_mean_baseline_mae)
                    if math.isfinite(condition_mean_baseline_mae)
                    else float("nan")
                ),
                "condition_diagnostics": condition_payload,
            }
        )
    return diagnostics


def build_markov_changepoint_ops_count_data_bundle(config: OPSCountConfig) -> MarkovOPSDataBundle:
    seeds = _resolve_runtime_seeds(config)
    transitions = _build_generator_transitions(
        config,
        seed=int(seeds["effective_data_seed"]),
    )
    train_docs = _generate_ops_count_docs(
        config,
        n_docs=int(config.train_docs),
        seed=int(seeds["effective_data_seed"]),
        transitions=transitions,
    )
    val_docs = _generate_ops_count_docs(
        config,
        n_docs=int(config.val_docs),
        seed=int(seeds["effective_val_seed"]),
        transitions=transitions,
    )
    test_docs = _generate_ops_count_docs(
        config,
        n_docs=int(config.test_docs),
        seed=int(seeds["effective_test_seed"]),
        transitions=transitions,
    )
    return MarkovOPSDataBundle(
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
        train_corpus_signature=_markov_corpus_signature(train_docs),
        val_corpus_signature=_markov_corpus_signature(val_docs),
        test_corpus_signature=_markov_corpus_signature(test_docs),
        metadata={},
    )


def _metrics_with_split_prefix(
    metrics: SketchMetrics,
    *,
    prefix: str,
    target_scale: Optional[float] = None,
) -> Dict[str, float | int]:
    payload: Dict[str, float | int] = {
        f"{prefix}_root_mae": float(metrics.root_mae),
        f"{prefix}_root_mse": float(metrics.root_mse),
        f"{prefix}_root_median_abs_error": float(metrics.root_median_abs_error),
        f"{prefix}_root_p95_abs_error": float(metrics.root_p95_abs_error),
        f"{prefix}_schedule_spread_mean": float(metrics.schedule_spread_mean),
        f"{prefix}_schedule_spread_p95": float(metrics.schedule_spread_p95),
        f"{prefix}_leaf_mae": float(metrics.leaf_mae),
        f"{prefix}_leaf_violation_rate": float(metrics.leaf_violation_rate),
        f"{prefix}_c2_count_drift_r1_mae": float(metrics.c2_count_drift_r1_mae),
        f"{prefix}_c2_count_drift_r2_mae": float(metrics.c2_count_drift_r2_mae),
        f"{prefix}_c2_count_drift_r4_mae": float(metrics.c2_count_drift_r4_mae),
        f"{prefix}_c2_root_count_drift_r1_mae": float(metrics.c2_root_count_drift_r1_mae),
        f"{prefix}_c2_root_count_drift_r2_mae": float(metrics.c2_root_count_drift_r2_mae),
        f"{prefix}_c2_root_count_drift_r4_mae": float(metrics.c2_root_count_drift_r4_mae),
        f"{prefix}_c2_idempotence_mae": float(metrics.c2_idempotence_mae),
        f"{prefix}_c2_r1_mae": float(metrics.c2_r1_mae),
        f"{prefix}_c2_r2_mae": float(metrics.c2_r2_mae),
        f"{prefix}_c2_r4_mae": float(metrics.c2_r4_mae),
        f"{prefix}_resummary_root_drift_r1": float(metrics.resummary_root_drift_r1),
        f"{prefix}_resummary_root_drift_r2": float(metrics.resummary_root_drift_r2),
        f"{prefix}_resummary_root_drift_r4": float(metrics.resummary_root_drift_r4),
        f"{prefix}_merge_mae": float(metrics.merge_mae),
        f"{prefix}_merge_violation_rate": float(metrics.merge_violation_rate),
        f"{prefix}_n_docs": int(metrics.n_docs),
        f"{prefix}_c2_state_replay_mse": float(metrics.c2_state_replay_mse),
        f"{prefix}_c2_bottleneck_reconstruction_mse": float(metrics.c2_bottleneck_reconstruction_mse),
    }
    scale = float(target_scale) if target_scale is not None else float("nan")
    if math.isfinite(scale) and scale > 0.0:
        payload.update(
            {
                f"{prefix}_root_mae_n": float(metrics.root_mae) / scale,
                f"{prefix}_schedule_spread_mean_n": float(metrics.schedule_spread_mean) / scale,
                f"{prefix}_c1_leaf_mae_n": float(metrics.leaf_mae) / scale,
                f"{prefix}_c2_count_drift_r1_mae_n": float(metrics.c2_count_drift_r1_mae)
                / scale,
                f"{prefix}_c2_count_drift_r2_mae_n": float(metrics.c2_count_drift_r2_mae)
                / scale,
                f"{prefix}_c2_count_drift_r4_mae_n": float(metrics.c2_count_drift_r4_mae)
                / scale,
                f"{prefix}_c2_root_count_drift_r1_mae_n": float(
                    metrics.c2_root_count_drift_r1_mae
                )
                / scale,
                f"{prefix}_c2_root_count_drift_r2_mae_n": float(
                    metrics.c2_root_count_drift_r2_mae
                )
                / scale,
                f"{prefix}_c2_root_count_drift_r4_mae_n": float(
                    metrics.c2_root_count_drift_r4_mae
                )
                / scale,
                f"{prefix}_c2_idempotence_mae_n": float(metrics.c2_idempotence_mae) / scale,
                f"{prefix}_c2_r1_mae_n": float(metrics.c2_r1_mae) / scale,
                f"{prefix}_c2_r2_mae_n": float(metrics.c2_r2_mae) / scale,
                f"{prefix}_c2_r4_mae_n": float(metrics.c2_r4_mae) / scale,
                f"{prefix}_resummary_root_drift_r1_n": float(metrics.resummary_root_drift_r1)
                / scale,
                f"{prefix}_resummary_root_drift_r2_n": float(metrics.resummary_root_drift_r2)
                / scale,
                f"{prefix}_resummary_root_drift_r4_n": float(metrics.resummary_root_drift_r4)
                / scale,
                f"{prefix}_c3_merge_mae_n": float(metrics.merge_mae) / scale,
                f"{prefix}_c2_state_replay_mse_n": float(metrics.c2_state_replay_mse)
                / scale,
            }
        )
    return payload


def _objective_with_split_prefix(
    metrics: ObjectiveMetrics,
    *,
    prefix: str,
) -> Dict[str, float | int]:
    return {
        # Backward-compatible aliases: these remain the weighted objective used in optimization.
        f"{prefix}_objective_full_labels": float(metrics.optimization_total_loss),
        f"{prefix}_objective_root_term": float(metrics.optimization_root_loss),
        f"{prefix}_objective_task_objective_term": float(metrics.optimization_root_loss),
        f"{prefix}_objective_leaf_term": float(metrics.optimization_leaf_loss),
        f"{prefix}_objective_c2_term": float(metrics.optimization_c2_loss),
        f"{prefix}_objective_merge_term": float(metrics.optimization_merge_loss),
        f"{prefix}_objective_schedule_consistency_term": float(
            metrics.optimization_schedule_consistency_loss
        ),
        f"{prefix}_optimization_objective_full_labels": float(metrics.optimization_total_loss),
        f"{prefix}_optimization_objective_root_term": float(metrics.optimization_root_loss),
        f"{prefix}_optimization_objective_task_objective_term": float(
            metrics.optimization_root_loss
        ),
        f"{prefix}_optimization_objective_leaf_term": float(metrics.optimization_leaf_loss),
        f"{prefix}_optimization_objective_c2_term": float(metrics.optimization_c2_loss),
        f"{prefix}_optimization_objective_merge_term": float(metrics.optimization_merge_loss),
        f"{prefix}_optimization_objective_schedule_consistency_term": float(
            metrics.optimization_schedule_consistency_loss
        ),
        f"{prefix}_unweighted_objective_full_labels": float(metrics.raw_total_loss),
        f"{prefix}_unweighted_objective_root_term": float(metrics.raw_root_loss),
        f"{prefix}_unweighted_objective_task_objective_term": float(metrics.raw_root_loss),
        f"{prefix}_unweighted_objective_leaf_term": float(metrics.raw_leaf_loss),
        f"{prefix}_unweighted_objective_c2_term": float(metrics.raw_c2_loss),
        f"{prefix}_unweighted_objective_merge_term": float(metrics.raw_merge_loss),
        f"{prefix}_unweighted_objective_schedule_consistency_term": float(
            metrics.raw_schedule_consistency_loss
        ),
        f"{prefix}_objective_n_docs": int(metrics.n_docs),
    }


def _full_tree_ipw_with_split_prefix(
    metrics: Mapping[str, Any],
    *,
    prefix: str,
) -> Dict[str, float | int]:
    payload = dict(metrics or {})
    if not payload:
        return {}
    return {
        f"{prefix}_full_node_exact_mean_loss": float(
            payload.get("full_node_exact_mean_loss", float("nan"))
        ),
        f"{prefix}_sampled_node_naive_mean_loss": float(
            payload.get("sampled_node_naive_mean_loss", float("nan"))
        ),
        f"{prefix}_sampled_node_naive_signed_error": float(
            payload.get("sampled_node_naive_signed_error", float("nan"))
        ),
        f"{prefix}_sampled_node_naive_abs_error": float(
            payload.get("sampled_node_naive_abs_error", float("nan"))
        ),
        f"{prefix}_sampled_node_ht_mean_loss": float(
            payload.get("sampled_node_ht_mean_loss", float("nan"))
        ),
        f"{prefix}_sampled_node_ht_signed_error": float(
            payload.get("sampled_node_ht_signed_error", float("nan"))
        ),
        f"{prefix}_sampled_node_ht_abs_error": float(
            payload.get("sampled_node_ht_abs_error", float("nan"))
        ),
        f"{prefix}_sampled_node_hajek_mean_loss": float(
            payload.get("sampled_node_hajek_mean_loss", float("nan"))
        ),
        f"{prefix}_sampled_node_hajek_signed_error": float(
            payload.get("sampled_node_hajek_signed_error", float("nan"))
        ),
        f"{prefix}_sampled_node_hajek_abs_error": float(
            payload.get("sampled_node_hajek_abs_error", float("nan"))
        ),
        f"{prefix}_document_top_loss": float(payload.get("document_top_loss", float("nan"))),
        f"{prefix}_document_top_mae": float(payload.get("document_top_mae", float("nan"))),
        f"{prefix}_document_vs_root_node_target_gap_mae": float(
            payload.get("document_vs_root_node_target_gap_mae", float("nan"))
        ),
        f"{prefix}_document_vs_root_node_prediction_gap_mae": float(
            payload.get("document_vs_root_node_prediction_gap_mae", float("nan"))
        ),
        f"{prefix}_full_tree_population_size": int(payload.get("population_size", 0)),
        f"{prefix}_full_tree_sampled_nodes": int(payload.get("sampled_nodes", 0)),
        f"{prefix}_full_tree_sampled_fraction": float(
            payload.get("sampled_fraction", float("nan"))
        ),
        f"{prefix}_full_tree_effective_sample_size": float(
            payload.get("effective_sample_size", float("nan"))
        ),
        f"{prefix}_full_tree_max_weight": float(payload.get("max_weight", float("nan"))),
    }


def _objective_estimator_with_split_prefix(
    estimator_payload: Mapping[str, Any],
    *,
    prefix: str,
) -> Dict[str, object]:
    payload = dict(estimator_payload or {})
    if not payload:
        return {}
    base_name = str(payload.get("objective_name", "configured_objective"))
    out: Dict[str, object] = {
        f"{prefix}_objective_estimator_payload": payload,
        f"{prefix}_objective_name": base_name,
        f"{prefix}_objective_selection_metric_name": str(
            payload.get("selection_metric_name", "")
        ),
        f"{prefix}_objective_selection_estimator": str(
            payload.get("selection_estimator", "exact")
        ),
        f"{prefix}_objective_selection_metric_value": float(
            payload.get("selection_metric_value", float("nan"))
        ),
        f"{prefix}_objective_available_estimators": list(
            payload.get("available_estimators", [])
        ),
    }
    for estimator in OBJECTIVE_ESTIMATOR_KEYS:
        alias = objective_estimator_alias(base_name, estimator)
        if alias in payload:
            out[f"{prefix}_{alias}"] = float(payload[alias])
    width_key = f"{base_name}_eb_width"
    if width_key in payload:
        out[f"{prefix}_{width_key}"] = float(payload[width_key])
    selection_value_key = f"{base_name}_selection_value"
    if selection_value_key in payload:
        out[f"{prefix}_{selection_value_key}"] = float(payload[selection_value_key])
    if "estimator_diagnostics" in payload:
        out[f"{prefix}_objective_estimator_diagnostics"] = dict(
            payload.get("estimator_diagnostics", {}) or {}
        )
    return out


def _markov_local_metrics(metrics: SketchMetrics, *, target_scale: float) -> LocalLawMetrics:
    scale = float(max(1.0, target_scale))
    return LocalLawMetrics(
        c1=float(metrics.leaf_mae) / scale,
        c2=float(metrics.c2_count_drift_r1_mae) / scale,
        c3=float(metrics.merge_mae) / scale,
        combined=float(
            markov_law_bundle_score(
                c1=float(metrics.leaf_mae) / scale,
                c2=float(metrics.c2_count_drift_r1_mae) / scale,
                c3=float(metrics.merge_mae) / scale,
            )
        ),
        root_error=float(metrics.root_mae) / scale,
        schedule_spread=float(metrics.schedule_spread_mean) / scale,
        c1_violation_rate=float(metrics.leaf_violation_rate),
        c3_violation_rate=float(metrics.merge_violation_rate),
    )


def _markov_downstream_metrics(
    metrics: SketchMetrics,
    *,
    target_scale: float,
) -> DownstreamMetrics:
    scale = float(max(1.0, target_scale))
    return DownstreamMetrics(
        root_error=float(metrics.root_mae) / scale,
        schedule_spread=float(metrics.schedule_spread_mean) / scale,
    )


def _maybe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(float(out)) else float(default)


def _constant_estimator_map(value: float) -> Dict[str, float]:
    return {str(name): float(value) for name in OBJECTIVE_ESTIMATOR_KEYS}


def _markov_normalized_ci_and_coverage(
    samples: Sequence[TreeSample],
    *,
    exact_value: float,
    raw_values_population: Sequence[float],
    population_size: float,
    delta: float,
) -> Dict[str, float]:
    raw_arr = np.asarray(raw_values_population, dtype=np.float64)
    lo = float(np.min(raw_arr)) if raw_arr.size > 0 else 0.0
    hi = float(np.max(raw_arr)) if raw_arr.size > 0 else 1.0
    scale = max(1e-12, hi - lo)
    if raw_arr.size == 0 or scale <= 1e-12:
        return {
            "value_min": float(lo),
            "value_max": float(hi),
            "scale": float(scale),
            "ht_mean": float(exact_value),
            "hajek": float(exact_value),
            "ht_abs_error": 0.0,
            "hajek_abs_error": 0.0,
            "eb_lo": float(exact_value),
            "eb_hi": float(exact_value),
            "eb_width": 0.0,
            "eb_contains_exact": 1.0,
            "effective_sample_size": float(effective_sample_size(samples)),
            "max_weight": float(max_weight(samples)),
            "sample_count": float(len(samples)),
            "weight_sum": float(sum(s.weight for s in samples)),
        }

    normalized_samples = [
        TreeSample(
            doc_id=str(sample.doc_id),
            node_id=str(sample.node_id),
            node_type=sample.node_type,
            violation=int(sample.violation),
            preference_loss=float(np.clip((float(sample.preference_loss) - lo) / scale, 0.0, 1.0)),
            sampling=sample.sampling,
            metadata=dict(sample.metadata),
        )
        for sample in samples
    ]
    comp = hajek_ht_comparison(
        normalized_samples,
        lambda s: float(s.preference_loss),
        population_size=float(population_size),
    )
    ci = empirical_bernstein_ci(
        normalized_samples,
        lambda s: float(s.preference_loss),
        float(delta),
        value_min=0.0,
        value_max=1.0,
    )
    exact_norm = (float(exact_value) - lo) / scale
    return {
        "value_min": float(lo),
        "value_max": float(hi),
        "scale": float(scale),
        "ht_mean": float(lo + scale * float(comp["ht_mean"])),
        "hajek": float(lo + scale * float(comp["hajek"])),
        "ht_abs_error": abs(float(lo + scale * float(comp["ht_mean"])) - float(exact_value)),
        "hajek_abs_error": abs(float(lo + scale * float(comp["hajek"])) - float(exact_value)),
        "eb_lo": float(lo + scale * float(ci[0])),
        "eb_hi": float(lo + scale * float(ci[1])),
        "eb_width": float(scale * max(0.0, float(ci[1]) - float(ci[0]))),
        "eb_contains_exact": (
            1.0 if float(ci[0]) - 1e-12 <= exact_norm <= float(ci[1]) + 1e-12 else 0.0
        ),
        "effective_sample_size": float(effective_sample_size(samples)),
        "max_weight": float(max_weight(samples)),
        "sample_count": float(len(samples)),
        "weight_sum": float(comp["weight_sum"]),
    }


def _markov_internal_inclusion_probabilities(
    n_internal: int,
    *,
    q: int,
    strategy: C3AuditStrategyName,
    merge_sizes: Sequence[int],
    include_root: bool,
) -> Tuple[Optional[np.ndarray], str]:
    n = int(max(0, n_internal))
    qq = int(max(0, q))
    if n <= 0:
        return np.zeros((0,), dtype=np.float64), "empty"
    if qq <= 0:
        return np.zeros((n,), dtype=np.float64), "zero_queries"
    if qq >= n:
        return np.ones((n,), dtype=np.float64), "all"

    strat = str(strategy)
    if strat not in VALID_C3_AUDIT_STRATEGIES:
        return None, "unsupported_strategy"

    probs = np.zeros((n,), dtype=np.float64)
    selected: set[int] = set()
    if include_root:
        selected.add(int(n - 1))
        probs[int(n - 1)] = 1.0
    if len(selected) >= qq:
        return probs, "deterministic_support"

    available = [i for i in range(n) if i not in selected]
    need = int(qq - len(selected))
    if need <= 0:
        return probs, "deterministic_support"

    if strat == "uniform":
        if not available:
            return probs, "uniform"
        p = float(min(1.0, float(need) / float(len(available))))
        for idx in available:
            probs[int(idx)] = float(p)
        return probs, "uniform"

    ranked = sorted(
        available,
        key=lambda i: (int(merge_sizes[i]) if i < len(merge_sizes) else 0, int(i)),
        reverse=True,
    )
    if strat == "top_span":
        for idx in ranked[:need]:
            probs[int(idx)] = 1.0
        return None, "top_span_partial_support"

    if strat == "hybrid_top_span":
        top_need = min(len(ranked), max(1, need // 2))
        deterministic = ranked[:top_need]
        for idx in deterministic:
            probs[int(idx)] = 1.0
        remaining_need = int(need - top_need)
        rem = [i for i in available if i not in set(deterministic)]
        if remaining_need >= len(rem):
            for idx in rem:
                probs[int(idx)] = 1.0
        elif remaining_need > 0 and rem:
            p = float(min(1.0, float(remaining_need) / float(len(rem))))
            for idx in rem:
                probs[int(idx)] = float(p)
        return probs, "hybrid_top_span"

    return None, "span_weighted_first_order_unavailable"


def _markov_objective_estimator_payload(
    model: object,
    docs: Sequence,
    *,
    device: torch.device,
    objective_summary: Mapping[str, Any],
    exact_objective: ObjectiveMetrics,
    leaf_query_rate: float,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    c3_audit_strategy: C3AuditStrategyName,
    c3_include_root: bool,
    config: Optional["OPSCountConfig"] = None,
    objective_ci_delta: float = 0.05,
    seed: int = 0,
    encode_leaf_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    if len(docs) == 0:
        return {}

    objective_spec = _markov_composite_objective_spec(objective_summary)
    population_size = float(len(docs))
    rng = random.Random(int(seed))
    leaf_population_values: List[float] = []
    leaf_samples: List[TreeSample] = []
    merge_population_values: List[float] = []
    merge_samples: List[TreeSample] = []
    leaf_support_ok = True
    merge_support_ok = True
    merge_support_mode = "all"

    model.eval()
    with torch.no_grad():
        for doc_idx, doc in enumerate(docs):
            if encode_leaf_fn is not None:
                states = encode_leaf_fn(doc)
            else:
                leaf_feats = _to_device(doc.leaf_features, device=device)
                states = [model.encode_leaf(x) for x in leaf_feats]
            _root_state, merge_states = model._merge_states(
                states,
                schedule="balanced",
                collect_merge_states=True,
            )

            n_leaf = int(len(states))
            if n_leaf > 0:
                leaf_losses: List[float] = []
                for st, truth in zip(states, doc.leaf_counts):
                    pred_leaf = model.predict_norm_from_state(st)
                    true_leaf = torch.tensor(
                        float(truth) / float(model.target_scale),
                        device=pred_leaf.device,
                        dtype=pred_leaf.dtype,
                    )
                    leaf_losses.append(
                        float(F.mse_loss(pred_leaf, true_leaf, reduction="mean").detach().cpu())
                    )
                scaled_leaf_values = [float(loss) / float(n_leaf) for loss in leaf_losses]
                leaf_population_values.extend(float(v) for v in scaled_leaf_values)
                q_leaf = leaf_sample_count(n_leaf, rate=float(leaf_query_rate))
                if q_leaf <= 0:
                    leaf_support_ok = False
                else:
                    if q_leaf >= n_leaf:
                        leaf_indices = list(range(n_leaf))
                    else:
                        leaf_indices = rng.sample(range(n_leaf), k=int(q_leaf))
                    pi_leaf = float(min(1.0, float(q_leaf) / float(n_leaf)))
                    if pi_leaf <= 0.0:
                        leaf_support_ok = False
                    for idx in leaf_indices:
                        leaf_samples.append(
                            TreeSample(
                                doc_id=f"markov_eval_{doc_idx}",
                                node_id=f"leaf_{idx}",
                                node_type=NodeType.LEAF,
                                violation=0,
                                preference_loss=float(scaled_leaf_values[int(idx)]),
                                sampling=SamplingMetadata(
                                    unit_propensity=float(pi_leaf),
                                    unit_kind=ObservationUnitKind.LEAF,
                                ),
                            )
                        )

            n_internal = int(len(merge_states))
            if n_internal > 0:
                merge_losses: List[float] = []
                for idx, st in enumerate(merge_states):
                    if idx >= len(doc.merge_counts_balanced):
                        break
                    pred = model.predict_norm_from_state(st)
                    truth = torch.tensor(
                        float(doc.merge_counts_balanced[idx]) / float(model.target_scale),
                        device=pred.device,
                        dtype=pred.dtype,
                    )
                    merge_losses.append(
                        float(F.mse_loss(pred, truth, reduction="mean").detach().cpu())
                    )
                if merge_losses:
                    scaled_merge_values = [
                        float(loss) / float(len(merge_losses)) for loss in merge_losses
                    ]
                    merge_population_values.extend(float(v) for v in scaled_merge_values)
                    q_internal = audit_sample_count(
                        n_internal,
                        policy=str(audit_policy),
                        fixed_nodes=int(audit_fixed_nodes),
                        fraction=float(audit_fraction),
                        scale=float(audit_scale),
                    )
                    inclusion_probs, support_mode = _markov_internal_inclusion_probabilities(
                        n_internal,
                        q=int(q_internal),
                        strategy=str(c3_audit_strategy),
                        merge_sizes=doc.merge_sizes_balanced,
                        include_root=bool(c3_include_root),
                    )
                    merge_support_mode = str(support_mode)
                    if inclusion_probs is None:
                        merge_support_ok = False
                    else:
                        sampled_internal = _sample_internal_audit_indices(
                            n_internal,
                            k=int(q_internal),
                            strategy=str(c3_audit_strategy),
                            merge_sizes=doc.merge_sizes_balanced,
                            include_root=bool(c3_include_root),
                            rng=rng,
                        )
                        if sampled_internal is None:
                            internal_indices = list(range(n_internal))
                        else:
                            internal_indices = list(sampled_internal)
                        if np.any(inclusion_probs <= 0.0):
                            merge_support_ok = False
                        for idx in internal_indices:
                            if idx >= len(scaled_merge_values):
                                continue
                            pi_merge = float(inclusion_probs[int(idx)])
                            if pi_merge <= 0.0:
                                continue
                            merge_samples.append(
                                TreeSample(
                                    doc_id=f"markov_eval_{doc_idx}",
                                    node_id=f"merge_{idx}",
                                    node_type=NodeType.MERGE,
                                    violation=0,
                                    preference_loss=float(scaled_merge_values[int(idx)]),
                                    sampling=SamplingMetadata(
                                        unit_propensity=float(pi_merge),
                                        unit_kind=ObservationUnitKind.MERGE,
                                    ),
                                )
                            )

    task_estimates = _constant_estimator_map(float(exact_objective.raw_root_loss))
    local_law_estimates: Dict[str, Dict[str, float]] = {
        LAW_ID_LEAF_PRESERVATION: {"exact": float(exact_objective.raw_leaf_loss)},
        LAW_ID_ON_RANGE_IDEMPOTENCE: _constant_estimator_map(float(exact_objective.raw_c2_loss)),
        LAW_ID_MERGE_PRESERVATION: {"exact": float(exact_objective.raw_merge_loss)},
    }
    proxy_estimates = {
        "schedule_consistency": _constant_estimator_map(
            float(exact_objective.raw_schedule_consistency_loss)
        )
    }
    estimator_diagnostics: Dict[str, Any] = {
        "population_size_docs": float(population_size),
        "leaf_support_ok": bool(leaf_support_ok),
        "merge_support_ok": bool(merge_support_ok),
        "merge_support_mode": str(merge_support_mode),
    }

    if leaf_support_ok and leaf_samples:
        leaf_eval = _markov_normalized_ci_and_coverage(
            leaf_samples,
            exact_value=float(exact_objective.raw_leaf_loss),
            raw_values_population=leaf_population_values,
            population_size=float(population_size),
            delta=float(objective_ci_delta),
        )
        local_law_estimates[LAW_ID_LEAF_PRESERVATION].update(
            {
                "ht": float(leaf_eval.get("ht_mean", float("nan"))),
                "hajek": float(leaf_eval.get("hajek", float("nan"))),
                "eb_lo": float(leaf_eval.get("eb_lo", float("nan"))),
                "eb_hi": float(leaf_eval.get("eb_hi", float("nan"))),
            }
        )
        estimator_diagnostics[LAW_ID_LEAF_PRESERVATION] = dict(leaf_eval)
    else:
        estimator_diagnostics[LAW_ID_LEAF_PRESERVATION] = {
            "sample_count": float(len(leaf_samples)),
            "effective_sample_size": float(effective_sample_size(leaf_samples)),
            "max_weight": float(max_weight(leaf_samples)),
        }

    if merge_support_ok and merge_samples:
        merge_eval = _markov_normalized_ci_and_coverage(
            merge_samples,
            exact_value=float(exact_objective.raw_merge_loss),
            raw_values_population=merge_population_values,
            population_size=float(population_size),
            delta=float(objective_ci_delta),
        )
        local_law_estimates[LAW_ID_MERGE_PRESERVATION].update(
            {
                "ht": float(merge_eval.get("ht_mean", float("nan"))),
                "hajek": float(merge_eval.get("hajek", float("nan"))),
                "eb_lo": float(merge_eval.get("eb_lo", float("nan"))),
                "eb_hi": float(merge_eval.get("eb_hi", float("nan"))),
            }
        )
        estimator_diagnostics[LAW_ID_MERGE_PRESERVATION] = dict(merge_eval)
    else:
        estimator_diagnostics[LAW_ID_MERGE_PRESERVATION] = {
            "sample_count": float(len(merge_samples)),
            "effective_sample_size": float(effective_sample_size(merge_samples)),
            "max_weight": float(max_weight(merge_samples)),
            "support_mode": str(merge_support_mode),
        }

    prefer_hajek = False
    if float(objective_spec.local_law_weights.get(LAW_ID_LEAF_PRESERVATION, 0.0)) > 0.0 or float(
        objective_spec.local_law_weights.get(LAW_ID_MERGE_PRESERVATION, 0.0)
    ) > 0.0:
        c1_ready = (
            float(objective_spec.local_law_weights.get(LAW_ID_LEAF_PRESERVATION, 0.0)) <= 0.0
            or math.isfinite(_maybe_float(local_law_estimates[LAW_ID_LEAF_PRESERVATION].get("hajek")))
        )
        c3_ready = (
            float(objective_spec.local_law_weights.get(LAW_ID_MERGE_PRESERVATION, 0.0)) <= 0.0
            or math.isfinite(_maybe_float(local_law_estimates[LAW_ID_MERGE_PRESERVATION].get("hajek")))
        )
        prefer_hajek = bool(c1_ready and c3_ready and (c1_ready or c3_ready))

    payload = scalarize_objective_estimates(
        objective_spec,
        task_estimates=task_estimates,
        local_law_estimates=local_law_estimates,
        proxy_estimates=proxy_estimates,
        selection_preference=("hajek" if prefer_hajek else "exact"),
    )
    logged_observations = _samples_to_logged_observations(
        leaf_samples,
        supervision_signal_name="c1",
    )
    logged_observations.extend(
        _samples_to_logged_observations(
            merge_samples,
            supervision_signal_name="c3",
        )
    )
    payload["estimator_diagnostics"] = estimator_diagnostics
    payload["logged_observations_summary"] = (
        summarize_logged_observations(logged_observations)
        if logged_observations
        else {
            "count": 0,
            "unit_kinds": [],
            "supports_ipw_estimation": False,
            "joint_propensity_min": 1.0,
            "joint_propensity_max": 1.0,
            "joint_propensity_mean": 1.0,
        }
    )
    payload["logged_observations"] = [observation.to_dict() for observation in logged_observations]
    return payload


def _markov_composite_objective_spec(
    objective_summary: Mapping[str, Any],
) -> CompositeObjectiveSpec:
    composite = dict(objective_summary.get("composite_objective", {}) or {})
    return CompositeObjectiveSpec(
        name=str(composite.get("name", "configured_objective")),
        selection_metric_name=str(composite.get("selection_metric_name", "configured_objective")),
        root_metric_name=str(composite.get("root_metric_name", "root_error")),
        root_share=float(composite.get("root_share", 0.0)),
        local_law_component_weights={
            str(name): float(value)
            for name, value in dict(
                composite.get("local_law_component_weights", {}) or {}
            ).items()
        },
        auxiliary_diagnostic_weights={
            str(name): float(value)
            for name, value in dict(
                composite.get("auxiliary_diagnostic_weights", {}) or {}
            ).items()
        },
        weighting_scheme=str(composite.get("weighting_scheme", "explicit_weighted_sum")),
        root_share_source=str(composite.get("root_share_source", "")),
        metadata={
            "root_metric_name": "root_error",
            "local_law_metric_names": {
                LAW_ID_LEAF_PRESERVATION: "c1",
                LAW_ID_ON_RANGE_IDEMPOTENCE: "c2",
                LAW_ID_MERGE_PRESERVATION: "c3",
            },
            "proxy_metric_names": {
                "schedule_consistency": "schedule_spread",
            },
        },
    )


def _markov_objective_metrics(
    *,
    local_metrics: Mapping[str, object],
    downstream_metrics: Mapping[str, object],
    objective_summary: Mapping[str, Any],
    split_name: str,
    split_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    composite = dict(objective_summary.get("composite_objective", {}) or {})
    objective_name = str(composite.get("name", composite.get("selection_metric_name", "configured_objective")))
    selection_metric_name = str(composite.get("selection_metric_name", objective_name))
    task_metric_name = str(objective_summary.get("task_objective_name", "root_error"))
    full_objective_value = float("nan")
    task_objective_value = float("nan")
    task_objective_term = float("nan")
    local_law_objective_value = float("nan")
    local_law_objective_term = float("nan")
    proxy_objective_value = float("nan")
    proxy_objective_term = float("nan")
    value_source = "recomputed_from_normalized_metrics"
    estimator_payload: Dict[str, Any] = {}
    root_share = float("nan")
    local_law_weight = float("nan")
    if split_payload is not None:
        payload = dict(split_payload)
        estimator_payload = dict(payload.get(f"{split_name}_objective_estimator_payload", {}) or {})
        full_objective_value = _maybe_float(payload.get(f"{split_name}_objective_full_labels"))
        task_objective_value = _maybe_float(
            payload.get(f"{split_name}_unweighted_objective_task_objective_term")
        )
        task_objective_term = _maybe_float(
            payload.get(f"{split_name}_objective_task_objective_term")
        )
        local_law_objective_value = float(
            sum(
                _maybe_float(
                    payload.get(f"{split_name}_unweighted_objective_{suffix}"),
                    0.0,
                )
                for suffix in ("leaf_term", "c2_term", "merge_term")
            )
        )
        local_law_objective_term = float(
            sum(
                _maybe_float(payload.get(f"{split_name}_objective_{suffix}"), 0.0)
                for suffix in ("leaf_term", "c2_term", "merge_term")
            )
        )
        proxy_objective_value = _maybe_float(
            payload.get(f"{split_name}_unweighted_objective_schedule_consistency_term"),
            0.0,
        )
        proxy_objective_term = _maybe_float(
            payload.get(f"{split_name}_objective_schedule_consistency_term"),
            0.0,
        )
        if estimator_payload:
            selection_metric_name = str(
                estimator_payload.get("selection_metric_name", selection_metric_name)
            )
        if math.isfinite(full_objective_value):
            value_source = "reported_split_payload"
    if not math.isfinite(full_objective_value):
        objective_spec = _markov_composite_objective_spec(objective_summary)
        objective_eval = evaluate_composite_objective_from_metrics(
            objective_spec,
            metrics={
                "root_error": float(downstream_metrics.get("root_error", float("nan"))),
                "c1": float(local_metrics.get("c1", float("nan"))),
                "c2": float(local_metrics.get("c2", float("nan"))),
                "c3": float(local_metrics.get("c3", float("nan"))),
                "schedule_spread": float(downstream_metrics.get("schedule_spread", float("nan"))),
            },
        )
        full_objective_value = float(objective_eval.total)
        root_share = float(objective_spec.normalized_task_share())
        local_law_weight = float(objective_spec.local_law_weight())
        if not math.isfinite(task_objective_value):
            task_objective_value = float(objective_eval.task_raw)
        if not math.isfinite(task_objective_term):
            task_objective_term = float(objective_eval.task_term)
        if not math.isfinite(local_law_objective_value):
            local_law_term_sum = float(sum(float(v) for v in objective_eval.local_law_terms.values()))
            local_law_objective_value = (
                float(local_law_term_sum / local_law_weight)
                if local_law_weight > 0.0
                else 0.0
            )
        if not math.isfinite(local_law_objective_term):
            local_law_objective_term = float(
                sum(float(v) for v in objective_eval.local_law_terms.values())
            )
        if not math.isfinite(proxy_objective_value):
            proxy_objective_value = float(sum(float(v) for v in objective_eval.proxy_raw.values()))
        if not math.isfinite(proxy_objective_term):
            proxy_objective_term = float(sum(float(v) for v in objective_eval.proxy_terms.values()))
    root_share_from_summary = float(objective_summary.get("root_share", 0.0))
    local_law_weight_total = float(
        sum(float(v) for v in dict(composite.get("local_law_component_weights", {}) or {}).values())
    )
    proxy_weight_total = float(
        sum(float(v) for v in dict(composite.get("auxiliary_proxy_input_masses", {}) or {}).values())
    )
    total_weight_without_proxy = float(
        composite.get(
            "total_weight_without_proxy",
            objective_summary.get("optimization_weight_mass_no_proxy", float("nan")),
        )
    )
    normalized_task_share = float(composite.get("root_share", root_share_from_summary))
    normalized_local_law_share = float(
        composite.get("local_law_weight", objective_summary.get("local_law_weight", local_law_weight_total))
    )
    if not math.isfinite(root_share):
        root_share = normalized_task_share
    if not math.isfinite(local_law_weight):
        local_law_weight = normalized_local_law_share
    out = {
        "objective_name": objective_name,
        "selection_metric_name": selection_metric_name,
        "weighting_scheme": str(objective_summary.get("weighting_scheme", "")),
        "task_metric_name": task_metric_name,
        "root_input_mass": root_share_from_summary,
        "local_law_weight_total": local_law_weight_total,
        "proxy_weight_total": proxy_weight_total,
        "total_weight_without_proxy": total_weight_without_proxy,
        "local_law_weight": float(local_law_weight),
        "root_share": float(root_share),
        "full_objective_value": float(full_objective_value),
        "task_objective_value": float(task_objective_value),
        "task_objective_term": float(task_objective_term),
        "regular_objective_value": float(task_objective_value),
        "regular_objective_term": float(task_objective_term),
        "local_law_objective_value": float(local_law_objective_value),
        "local_law_objective_term": float(local_law_objective_term),
        "proxy_objective_value": float(proxy_objective_value),
        "proxy_objective_term": float(proxy_objective_term),
        "value_source": str(value_source),
        "local_law_component_weights": {
            str(name): float(value)
            for name, value in dict(composite.get("local_law_component_weights", {}) or {}).items()
        },
        "auxiliary_proxy_input_masses": {
            str(name): float(value)
            for name, value in dict(composite.get("auxiliary_proxy_input_masses", {}) or {}).items()
        },
    }
    if estimator_payload:
        out["selection_estimator"] = str(estimator_payload.get("selection_estimator", "exact"))
        out["selection_metric_value"] = float(
            estimator_payload.get("selection_metric_value", float("nan"))
        )
        out["available_estimators"] = list(estimator_payload.get("available_estimators", []))
        out["estimator_components"] = dict(
            estimator_payload.get("estimator_components", {}) or {}
        )
        for estimator in OBJECTIVE_ESTIMATOR_KEYS:
            alias = objective_estimator_alias(objective_name, estimator)
            if alias in estimator_payload:
                out[str(alias)] = float(estimator_payload[alias])
            for key in (
                f"{alias}_root_share",
                f"{alias}_local_law_weight",
            ):
                if key in estimator_payload:
                    out[str(key)] = float(estimator_payload[key])
        width_key = f"{objective_name}_eb_width"
        if width_key in estimator_payload:
            out[width_key] = float(estimator_payload[width_key])
        selection_value_key = f"{objective_name}_selection_value"
        if selection_value_key in estimator_payload:
            out[selection_value_key] = float(estimator_payload[selection_value_key])
        if "estimator_diagnostics" in estimator_payload:
            out["estimator_diagnostics"] = dict(
                estimator_payload.get("estimator_diagnostics", {}) or {}
            )
    return out


def _markov_split_id(*, split: str, seed: int, n_docs: int) -> str:
    return f"markov:{str(split)}:seed={int(seed)}:docs={int(n_docs)}"


def _markov_artifact_dir(config: OPSCountConfig) -> Optional[Path]:
    text = str(config.artifact_dir).strip()
    if not text:
        return None
    path = Path(text)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _samples_to_logged_observations(
    samples: Sequence[TreeSample],
    *,
    supervision_signal_name: str,
) -> List[LoggedLabelObservation[Any]]:
    signal_to_law_kind = {
        "c1": LawKind.L1_LEAF,
        "c3": LawKind.L2_MERGE,
    }
    observations: List[LoggedLabelObservation[Any]] = []
    for sample in samples:
        observations.append(
            shared_logged_substructure_observation(
                document_id=str(sample.doc_id),
                unit_id=str(sample.node_id),
                unit_kind=sample.sampling.unit_kind or (
                    ObservationUnitKind.LEAF
                    if sample.node_type == NodeType.LEAF
                    else ObservationUnitKind.MERGE
                ),
                label=float(sample.preference_loss),
                application_name="markov_ops_count",
                supervision_signal_name=supervision_signal_name,
                law_kind=signal_to_law_kind.get(supervision_signal_name),
                sampling=sample.sampling,
                context=dict(sample.metadata or {}),
            )
        )
    return observations


def _markov_analytic_artifact(
    *,
    output_dir: Optional[Path],
    artifact_id: str,
    name: str,
    role: PolicyRole,
    config: OPSCountConfig,
    target_scale: float,
    notes: str,
    targeted_laws: Optional[Sequence[str]] = None,
) -> Optional[GArtifact]:
    if output_dir is None:
        return None
    return write_json_g_artifact(
        output_dir=output_dir,
        artifact_id=str(artifact_id),
        name=str(name),
        role=role,
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        payload={
            "model_family": "analytic",
            "feature_mode": str(config.feature_mode),
            "n_regimes": int(config.n_regimes),
            "target_scale": float(target_scale),
            "state_layout": str(name),
            "merge_semantics": str(notes),
            "resummary_semantics": (
                "identity" if str(name) != "flip_R2" else "toggle-flip bit on each resummary"
            ),
            "targeted_laws": [str(x) for x in (targeted_laws or [])],
        },
        metadata={
            "suite_role": str(config.suite_role),
            "law_package": str(config.law_package),
            "exact_family": str(config.exact_family),
        },
    )


def _markov_model_artifact(
    *,
    output_dir: Optional[Path],
    artifact_id: str,
    role: PolicyRole,
    name: str,
    model: object,
    config: OPSCountConfig,
    target_scale: float,
) -> Optional[GArtifact]:
    if output_dir is None:
        return None

    arrays: Dict[str, np.ndarray] = {}
    manifest_payload: Dict[str, Any] = {
        "model_family": str(config.model_family),
        "feature_mode": str(config.feature_mode),
        "n_regimes": int(config.n_regimes),
        "target_scale": float(target_scale),
        "state_layout": "",
        "merge_semantics": "",
        "readout_semantics": "",
        "resummary_semantics": "decode_summary/encode_summary over full sketch state summary",
    }

    if hasattr(model, "encoder") and isinstance(getattr(model, "encoder"), nn.Linear):
        enc = getattr(model, "encoder")
        arrays["encoder_weight"] = enc.weight.detach().cpu().numpy()
        arrays["encoder_bias"] = enc.bias.detach().cpu().numpy()
        manifest_payload["state_layout"] = "normalized_count + first_endpoint + last_endpoint"
        manifest_payload["merge_semantics"] = (
            "additive count merge with explicit boundary correction when endpoints differ"
        )
        manifest_payload["readout_semantics"] = "identity normalized count readout"
    elif hasattr(model, "fno_encoder") and hasattr(model, "merger") and getattr(model, "merger") is not None:
        # FNOCountSketch: FNO leaf encoder + learned merger/readout
        leaf_proj = getattr(model, "leaf_proj")
        mer = getattr(model, "merger")
        readout = getattr(model, "readout")
        sum_enc = getattr(model, "summary_encoder")
        arrays["token_embedding_weight"] = model.token_embedding.weight.detach().cpu().numpy()
        arrays["leaf_proj_linear0_weight"] = leaf_proj[0].weight.detach().cpu().numpy()
        arrays["leaf_proj_linear0_bias"] = leaf_proj[0].bias.detach().cpu().numpy()
        arrays["leaf_proj_linear1_weight"] = leaf_proj[2].weight.detach().cpu().numpy()
        arrays["leaf_proj_linear1_bias"] = leaf_proj[2].bias.detach().cpu().numpy()
        arrays["merger_linear0_weight"] = mer[0].weight.detach().cpu().numpy()
        arrays["merger_linear0_bias"] = mer[0].bias.detach().cpu().numpy()
        arrays["merger_linear1_weight"] = mer[3].weight.detach().cpu().numpy()
        arrays["merger_linear1_bias"] = mer[3].bias.detach().cpu().numpy()
        arrays["readout_linear0_weight"] = readout[0].weight.detach().cpu().numpy()
        arrays["readout_linear0_bias"] = readout[0].bias.detach().cpu().numpy()
        arrays["readout_linear1_weight"] = readout[2].weight.detach().cpu().numpy()
        arrays["readout_linear1_bias"] = readout[2].bias.detach().cpu().numpy()
        arrays["summary_linear0_weight"] = sum_enc[0].weight.detach().cpu().numpy()
        arrays["summary_linear0_bias"] = sum_enc[0].bias.detach().cpu().numpy()
        arrays["summary_linear1_weight"] = sum_enc[2].weight.detach().cpu().numpy()
        arrays["summary_linear1_bias"] = sum_enc[2].bias.detach().cpu().numpy()
        # Endpoint projection weights (learned from FNO output at boundary tokens)
        arrays["first_endpoint_proj_weight"] = model.first_endpoint_proj.weight.detach().cpu().numpy()
        arrays["first_endpoint_proj_bias"] = model.first_endpoint_proj.bias.detach().cpu().numpy()
        arrays["last_endpoint_proj_weight"] = model.last_endpoint_proj.weight.detach().cpu().numpy()
        arrays["last_endpoint_proj_bias"] = model.last_endpoint_proj.bias.detach().cpu().numpy()
        manifest_payload["state_layout"] = "fno_latent_state + first_endpoint + last_endpoint"
        manifest_payload["merge_semantics"] = (
            "learned merger over left/right FNO latent states plus boundary endpoints"
        )
        manifest_payload["readout_semantics"] = "sigmoid(readout(fno_latent_state)) * target_scale"
    else:
        return None

    return write_npz_g_artifact(
        output_dir=output_dir,
        artifact_id=str(artifact_id),
        name=str(name),
        role=role,
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        manifest_payload=manifest_payload,
        arrays=arrays,
        metadata={
            "suite_role": str(config.suite_role),
            "law_package": str(config.law_package),
            "exact_family": str(config.exact_family),
        },
    )


def _policy_split_payload(
    metrics: SketchMetrics,
    *,
    target_scale: float,
    split_name: str,
    objective_summary: Mapping[str, Any],
    split_payload: Optional[Mapping[str, Any]] = None,
    config: Optional["OPSCountConfig"] = None,
) -> Dict[str, Any]:
    local_metrics = _markov_local_metrics(
        metrics,
        target_scale=float(target_scale),
    ).to_dict()
    downstream_metrics = _markov_downstream_metrics(
        metrics,
        target_scale=float(target_scale),
    ).to_dict()
    return {
        "local_law_metrics": local_metrics,
        "downstream_metrics": downstream_metrics,
        "objective_metrics": _markov_objective_metrics(
            local_metrics=local_metrics,
            downstream_metrics=downstream_metrics,
            objective_summary=objective_summary,
            split_name=str(split_name),
            split_payload=split_payload,
        ),
    }


def _build_markov_local_law_learnability(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    target_scale: float,
    objective_summary: Mapping[str, Any],
    geom: TrainingGeometry,
    exact: SketchMetrics,
    leaf_bucket: SketchMetrics,
    undersupported: SketchMetrics,
    flip_r2: SketchMetrics,
    current_name: str,
    current_role: str,
    current_train: Optional[SketchMetrics],
    current_val: Optional[SketchMetrics],
    current_test: SketchMetrics,
    current_selection_metric_name: str,
    current_selection_metric: float,
    current_train_payload: Optional[Mapping[str, Any]] = None,
    current_val_payload: Optional[Mapping[str, Any]] = None,
    current_test_payload: Optional[Mapping[str, Any]] = None,
    model: Optional[object] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    artifact_dir = _markov_artifact_dir(config)
    artifacts: List[GArtifact] = []
    oracle_artifact = _markov_analytic_artifact(
        output_dir=artifact_dir,
        artifact_id="oracle_g",
        name="oracle_g",
        role=PolicyRole.ORACLE_G,
        config=config,
        target_scale=float(target_scale),
        notes="exact changepoint count sketch with endpoints and count",
        targeted_laws=[],
    )
    if oracle_artifact is not None:
        artifacts.append(oracle_artifact)
    current_artifact = (
        _markov_model_artifact(
            output_dir=artifact_dir,
            artifact_id=str(current_role),
            role=PolicyRole(str(current_role)),
            name=str(current_name),
            model=model,
            config=config,
            target_scale=float(target_scale),
        )
        if model is not None
        else None
    )
    if current_artifact is not None:
        artifacts.append(current_artifact)
    counterexample_specs = [
        (
            "leaf_bucket",
            leaf_bucket,
            ["C1"],
            "leaf summaries collapse to a bucket identity and break leaf preservation",
        ),
        (
            "count_only",
            undersupported,
            ["C3"],
            "count-only merge omits boundary state and breaks merge preservation",
        ),
        (
            "flip_R2",
            flip_r2,
            ["C2"],
            "resummary toggles a hidden flip bit and breaks idempotence",
        ),
    ]
    counterexamples: List[LocalLawCounterexampleEvaluation] = []
    for name, metrics, targeted_laws, notes in counterexample_specs:
        artifact = _markov_analytic_artifact(
            output_dir=artifact_dir,
            artifact_id=str(name),
            name=str(name),
            role=PolicyRole.COUNTEREXAMPLE_G,
            config=config,
            target_scale=float(target_scale),
            notes=str(notes),
            targeted_laws=targeted_laws,
        )
        if artifact is not None:
            artifacts.append(artifact)
        counterexamples.append(
            LocalLawCounterexampleEvaluation(
                name=str(name),
                role=PolicyRole.COUNTEREXAMPLE_G,
                targeted_laws=[str(x) for x in targeted_laws],
                artifact_id=(artifact.artifact_id if artifact is not None else None),
                metrics={
                    "test": _policy_split_payload(
                        metrics,
                        target_scale=float(target_scale),
                        split_name="test",
                        objective_summary=objective_summary,
                        config=config,
                    )
                },
                metadata={"note": str(notes)},
            )
        )

    policy_role = PolicyRole(str(current_role))
    split_metrics: Dict[str, Dict[str, Any]] = {
        "test": _policy_split_payload(
            current_test,
            target_scale=float(target_scale),
            split_name="test",
            objective_summary=objective_summary,
            split_payload=current_test_payload,
            config=config,
        )
    }
    if current_train is not None:
        split_metrics["train"] = _policy_split_payload(
            current_train,
            target_scale=float(target_scale),
            split_name="train",
            objective_summary=objective_summary,
            split_payload=current_train_payload,
            config=config,
        )
    if current_val is not None:
        split_metrics["val"] = _policy_split_payload(
            current_val,
            target_scale=float(target_scale),
            split_name="val",
            objective_summary=objective_summary,
            split_payload=current_val_payload,
            config=config,
        )

    policies = {
        "oracle_g": LocalLawPolicyEvaluation(
            name="oracle_g",
            role=PolicyRole.ORACLE_G,
            artifact_id=(oracle_artifact.artifact_id if oracle_artifact is not None else None),
            split_metrics={
                "test": _policy_split_payload(
                    exact,
                    target_scale=float(target_scale),
                    split_name="test",
                    objective_summary=objective_summary,
                    config=config,
                )
            },
            metadata={"law_package": "exact"},
        ),
        str(current_name): LocalLawPolicyEvaluation(
            name=str(current_name),
            role=policy_role,
            artifact_id=(current_artifact.artifact_id if current_artifact is not None else None),
            selection_metric_value=float(current_selection_metric),
            split_metrics=split_metrics,
            metadata={
                "law_package": str(config.law_package),
                "model_family": str(config.model_family),
            },
        ),
    }
    test_estimator_payload = dict(
        dict(current_test_payload or {}).get("test_objective_estimator_payload", {}) or {}
    )
    selected_logged_observations = [
        LoggedLabelObservation.from_dict(row)
        for row in list(test_estimator_payload.get("logged_observations", []) or [])
    ]
    logged_observation_artifacts: Dict[str, Any] = {}
    logged_observations_summary = dict(test_estimator_payload.get("logged_observations_summary", {}) or {})
    if not logged_observations_summary:
        logged_observations_summary = (
            summarize_logged_observations(selected_logged_observations)
            if selected_logged_observations
            else {
                "count": 0,
                "unit_kinds": [],
                "supports_ipw_estimation": bool(
                    dict(current_test_payload or {}).get("ipw_evaluation")
                ),
                "joint_propensity_min": 1.0,
                "joint_propensity_max": 1.0,
                "joint_propensity_mean": 1.0,
            }
        )
    if selected_logged_observations and artifact_dir is not None and bool(config.save_logged_observations):
        artifact = write_logged_observations_jsonl(
            artifact_dir / "local_law_logged_observations.jsonl",
            selected_logged_observations,
            channel_name="sampled_substructure_supervision",
        )
        logged_observation_artifacts = {artifact.channel_name: artifact.to_dict()}

    summary = LocalLawRunSummary(
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        oracle_name="changepoint_count_exact_summary",
        study_role=str(current_role),
        split_ids={
            "train": _markov_split_id(
                split="train",
                seed=int(seeds["effective_data_seed"]),
                n_docs=int(config.train_docs),
            ),
            "val": _markov_split_id(
                split="val",
                seed=int(seeds["effective_val_seed"]),
                n_docs=int(config.val_docs),
            ),
            "test": _markov_split_id(
                split="test",
                seed=int(seeds["effective_test_seed"]),
                n_docs=int(config.test_docs),
            ),
        },
        support_budget=SupportBudgetSummary(
            train_docs=int(config.train_docs),
            val_docs=int(config.val_docs),
            test_docs=int(config.test_docs),
            leaf_query_rate=float(config.leaf_query_rate),
            internal_query_rate=(
                float(geom.mean_internal_labels) / float(max(1.0, geom.mean_internal_nodes))
            ),
            root_query_rate=1.0 if bool(config.include_root_query) else 0.0,
            mean_leaf_labels_per_doc=float(geom.mean_leaf_labels),
            mean_internal_labels_per_doc=float(geom.mean_internal_labels),
            mean_queries_per_doc=float(geom.mean_queries_per_doc),
            total_queries_estimate=float(geom.total_queries_estimate),
            metadata={
                "audit_fraction": float(config.audit_fraction),
                "audit_policy": str(config.audit_policy),
            },
        ),
        selection={
            "selection_split": "val" if int(config.val_docs) > 0 else "config",
            "selection_metric": (
                "configured_exact_family"
                if str(current_role) == PolicyRole.COUNTEREXAMPLE_G.value
                else str(current_selection_metric_name or "configured_objective")
            ),
            "selected_candidate": str(current_name),
            "uses_test_metrics": False,
            "selection_reason": (
                "fixed by configured law_package"
                if str(current_role) != PolicyRole.COUNTEREXAMPLE_G.value
                else "fixed by configured exact_family"
            ),
        },
        policies=policies,
        counterexamples=counterexamples,
        thresholds={
            "c1_tau": float(config.violation_tau) / float(max(1.0, target_scale)),
            "c2_tau": float(config.violation_tau) / float(max(1.0, target_scale)),
            "c3_tau": float(config.violation_tau) / float(max(1.0, target_scale)),
        },
        suite_role=str(config.suite_role),
        logged_observation_artifacts=logged_observation_artifacts,
        metadata={
            "law_package": str(config.law_package),
            "exact_family": str(config.exact_family),
            "root_only_reference_package": "root_only",
            "all_laws_reference_package": "all_laws_plus_sched",
            "configured_objective_name": str(
                dict(objective_summary.get("composite_objective", {}) or {}).get(
                    "selection_metric_name",
                    "configured_objective",
                )
            ),
            "objective": dict(objective_summary.get("composite_objective", {}) or {}),
            "logged_observations_summary": logged_observations_summary,
        },
    )
    summary = attach_local_law_learning_problem(summary)
    return summary.to_dict(), artifact_index(artifacts)


def _resolve_runtime_seeds(config: OPSCountConfig) -> Dict[str, int]:
    shared_seed = int(config.seed)
    data_seed = int(shared_seed if config.data_seed is None else config.data_seed)
    model_seed = int(shared_seed if config.model_seed is None else config.model_seed)
    val_seed = int(data_seed + int(config.val_seed_offset))
    test_seed = int(data_seed + int(config.test_seed_offset))
    return {
        "seed": int(shared_seed),
        "effective_data_seed": int(data_seed),
        "effective_model_seed": int(model_seed),
        "effective_val_seed": int(val_seed),
        "effective_test_seed": int(test_seed),
    }


def _markov_active_laws(package: str) -> Tuple[str, ...]:
    law_sets = {
        "root_only": (),
        "c1_only": (LAW_ID_LEAF_PRESERVATION,),
        "c2_only": (LAW_ID_ON_RANGE_IDEMPOTENCE,),
        "c3_only": (LAW_ID_MERGE_PRESERVATION,),
        "c1c3": (LAW_ID_LEAF_PRESERVATION, LAW_ID_MERGE_PRESERVATION),
        "all_laws": (
            LAW_ID_LEAF_PRESERVATION,
            LAW_ID_ON_RANGE_IDEMPOTENCE,
            LAW_ID_MERGE_PRESERVATION,
        ),
        "sched_only": (),
        "all_laws_plus_sched": (
            LAW_ID_LEAF_PRESERVATION,
            LAW_ID_ON_RANGE_IDEMPOTENCE,
            LAW_ID_MERGE_PRESERVATION,
        ),
    }
    return tuple(
        law_sets.get(
            str(package or "all_laws").strip().lower(),
            (
                LAW_ID_LEAF_PRESERVATION,
                LAW_ID_ON_RANGE_IDEMPOTENCE,
                LAW_ID_MERGE_PRESERVATION,
            ),
        )
    )


def _law_package_weights(config: OPSCountConfig) -> Optional[Dict[str, float]]:
    package = str(config.law_package or "").strip().lower()
    if not package:
        return None
    if package not in VALID_LAW_PACKAGES:
        raise ValueError(
            f"law_package={package!r} unsupported; expected one of {VALID_LAW_PACKAGES}"
        )

    active = _markov_active_laws(package)
    if config.local_law_weight is not None:
        resolved = resolve_root_local_objective_weights(
            local_law_weight=float(config.local_law_weight),
            active_laws=active,
            objective_context="markov law_package",
        )
    elif active:
        raise ValueError(
            "law_package selects active local laws but local_law_weight was not "
            "supplied; provide lambda explicitly or use explicit root/law weights"
        )
    else:
        resolved = resolve_root_local_objective_weights(
            local_law_weight=0.0,
            active_laws=active,
            objective_context="markov law_package",
        )
    weights = {
        LAW_ID_LEAF_PRESERVATION: 0.0,
        LAW_ID_ON_RANGE_IDEMPOTENCE: 0.0,
        LAW_ID_MERGE_PRESERVATION: 0.0,
        "schedule": 0.0,
    }
    for key, share in dict(resolved.local_law_shares).items():
        weights[str(key)] = float(share)
    if package in {"sched_only", "all_laws_plus_sched"}:
        sched_scale = (
            float(config.schedule_consistency_weight)
            if float(config.schedule_consistency_weight) > 0.0
            else 0.1
        )
        weights["schedule"] = float(max(0.0, sched_scale))
    weights["local_law_weight"] = float(resolved.local_law_weight)
    return weights


def _resolve_local_law_weights(config: OPSCountConfig) -> Dict[str, float | str]:
    collapse_mode = str(getattr(config, "tree_exact_collapse_mode", "") or "").strip()
    if collapse_mode in {
        "official_fno_one_tree_identity",
        "official_fno_runtime_identity",
    }:
        configured_task_weight = (
            float(max(0.0, config.task_objective_weight))
            if config.task_objective_weight is not None
            else 1.0
        )
        return {
            "parameterization": "exact_collapse_root_only_identity",
            "weighting_scheme": "exact_collapse_root_only_identity",
            "law_set_id": "root_only",
            "local_law_weight": 0.0,
            "root_share": 1.0,
            "optimization_root_weight": 1.0,
            "configured_root_share": float(configured_task_weight),
            "root_share_source": "exact_collapse_root_only_identity",
            "local_law_c1_weight": 0.0,
            "local_law_c2_weight": 0.0,
            "local_law_c3_weight": 0.0,
            "local_law_c1_share": 0.0,
            "local_law_c2_share": 0.0,
            "local_law_c3_share": 0.0,
            "optimization_weight_mass_no_proxy": 1.0,
            "legacy_leaf_weight": float(max(0.0, config.leaf_weight)),
            "legacy_c2_weight": float(max(0.0, config.c2_weight)),
            "legacy_c3_weight": float(max(0.0, config.c3_weight)),
            "proxy_schedule_consistency_weight": 0.0,
        }

    legacy_c1 = float(config.leaf_weight)
    legacy_c2 = float(config.c2_weight)
    legacy_c3 = float(config.c3_weight)
    configured_task_weight = (
        float(max(0.0, config.task_objective_weight))
        if config.task_objective_weight is not None
        else None
    )
    package = str(config.law_package or "").strip().lower()
    if package and package not in VALID_LAW_PACKAGES:
        raise ValueError(
            f"law_package={package!r} unsupported; expected one of {VALID_LAW_PACKAGES}"
        )
    active_laws = _markov_active_laws(package or "all_laws")
    raw_law_weights = {
        LAW_ID_LEAF_PRESERVATION: float(max(0.0, legacy_c1)),
        LAW_ID_ON_RANGE_IDEMPOTENCE: float(max(0.0, legacy_c2)),
        LAW_ID_MERGE_PRESERVATION: float(max(0.0, legacy_c3)),
    }
    if config.local_law_weight is not None:
        if configured_task_weight is not None:
            raise ValueError("local_law_weight cannot be combined with task_objective_weight")
        if max(0.0, legacy_c1) > 0.0 or max(0.0, legacy_c2) > 0.0 or max(0.0, legacy_c3) > 0.0:
            raise ValueError("local_law_weight cannot be combined with explicit law weights")
        lambda_raw = float(config.local_law_weight)
        if (
            lambda_raw > 0.0
            and (
                not math.isclose(float(config.c1_relative_weight), 1.0)
                or not math.isclose(float(config.c2_relative_weight), 1.0)
                or not math.isclose(float(config.c3_relative_weight), 1.0)
            )
        ):
            raise ValueError("lambda mode uses equal active-law weights; relative weights are not supported")
        resolved = resolve_root_local_objective_weights(
            local_law_weight=lambda_raw,
            active_laws=active_laws,
            objective_context="markov objective",
        )
        effective_c1 = float(resolved.local_law_shares.get(LAW_ID_LEAF_PRESERVATION, 0.0))
        effective_c2 = float(resolved.local_law_shares.get(LAW_ID_ON_RANGE_IDEMPOTENCE, 0.0))
        effective_c3 = float(resolved.local_law_shares.get(LAW_ID_MERGE_PRESERVATION, 0.0))
        local_weight = float(resolved.local_law_weight)
        if local_weight > 0.0:
            c1_share = float(effective_c1 / local_weight)
            c2_share = float(effective_c2 / local_weight)
            c3_share = float(effective_c3 / local_weight)
        else:
            c1_share = 0.0
            c2_share = 0.0
            c3_share = 0.0
        proxy_weight = float(max(0.0, config.schedule_consistency_weight))
        if package in {"sched_only", "all_laws_plus_sched"} and proxy_weight <= 0.0:
            proxy_weight = 0.1
        parameterization = "law_set_lambda" if package else "lambda"
        weighting_scheme = str(resolved.weighting_scheme)
        optimization_root_weight = float(resolved.root_share)
        root_share_source = "derived_from_local_law_weight"
    else:
        inactive_laws = set(raw_law_weights.keys()) - set(active_laws)
        if package and any(float(raw_law_weights[name]) > 0.0 for name in inactive_laws):
            raise ValueError("law_package explicit mode cannot set weights for inactive laws")
        resolved = resolve_root_local_objective_weights(
            local_law_weight=None,
            active_laws=active_laws,
            explicit_root_weight=(
                float(config.root_weight) if configured_task_weight is None else configured_task_weight
            ),
            explicit_law_weights={
                name: raw_law_weights[name] for name in active_laws if name in raw_law_weights
            },
            objective_context="markov objective",
        )
        optimization_root_weight = float(resolved.root_share)
        effective_c1 = float(resolved.local_law_shares.get(LAW_ID_LEAF_PRESERVATION, 0.0))
        effective_c2 = float(resolved.local_law_shares.get(LAW_ID_ON_RANGE_IDEMPOTENCE, 0.0))
        effective_c3 = float(resolved.local_law_shares.get(LAW_ID_MERGE_PRESERVATION, 0.0))
        local_weight = float(resolved.local_law_weight)
        if local_weight > 0.0:
            c1_share = float(effective_c1 / local_weight)
            c2_share = float(effective_c2 / local_weight)
            c3_share = float(effective_c3 / local_weight)
        else:
            c1_share = c2_share = c3_share = 0.0
        proxy_weight = float(max(0.0, config.schedule_consistency_weight))
        if package in {"sched_only", "all_laws_plus_sched"} and proxy_weight <= 0.0:
            proxy_weight = 0.1
        parameterization = "explicit_normalized_weights"
        weighting_scheme = str(resolved.weighting_scheme)
        root_share_source = (
            "explicit_root_share"
            if configured_task_weight is not None
            else "explicit_root_weight"
        )

    return {
        "parameterization": str(parameterization),
        "weighting_scheme": str(weighting_scheme),
        "law_set_id": canonical_law_set_id(
            str(config.law_package or "all_laws"),
            allow_aliases=True,
        ),
        "local_law_weight": float(local_weight),
        "root_share": float(optimization_root_weight),
        "optimization_root_weight": float(optimization_root_weight),
        "configured_root_share": (
            float(configured_task_weight) if configured_task_weight is not None else None
        ),
        "root_share_source": str(root_share_source),
        "local_law_c1_weight": float(effective_c1),
        "local_law_c2_weight": float(effective_c2),
        "local_law_c3_weight": float(effective_c3),
        "local_law_c1_share": float(c1_share),
        "local_law_c2_share": float(c2_share),
        "local_law_c3_share": float(c3_share),
        "optimization_weight_mass_no_proxy": float(
            optimization_root_weight + effective_c1 + effective_c2 + effective_c3
        ),
        "legacy_leaf_weight": float(legacy_c1),
        "legacy_c2_weight": float(legacy_c2),
        "legacy_c3_weight": float(legacy_c3),
        "proxy_schedule_consistency_weight": float(proxy_weight),
    }


def _build_objective_summary(config: OPSCountConfig) -> Dict[str, Any]:
    resolved = _resolve_local_law_weights(config)
    root_share = float(resolved["root_share"])
    root_active = bool(config.include_root_query) and root_share > 0.0
    proxy_weight = float(resolved["proxy_schedule_consistency_weight"])
    uses_weighted_neural_objective = str(config.model_family) in ("neural", "fno")
    weighting_scheme = str(resolved["weighting_scheme"])
    if weighting_scheme == "normalized_explicit_weights":
        objective_formula = (
            "`normalized_root_weight * task + sum_i normalized_law_weight_i * law_i`"
        )
    else:
        objective_formula = "`(1 - lambda) * task + lambda * equal_active_local_laws`"
    composite_spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        root_metric_name="root_count_mse",
        root_share=float(resolved["root_share"]),
        local_law_component_weights={
            LAW_ID_LEAF_PRESERVATION: float(resolved["local_law_c1_weight"]),
            LAW_ID_ON_RANGE_IDEMPOTENCE: float(resolved["local_law_c2_weight"]),
            LAW_ID_MERGE_PRESERVATION: float(resolved["local_law_c3_weight"]),
        },
        auxiliary_diagnostic_weights={"schedule_consistency": float(proxy_weight)},
        weighting_scheme=str(weighting_scheme),
        root_share_source=str(resolved["root_share_source"]),
        metadata={
            "problem_id": str(config.problem_id),
            "method_id": str(config.method_id),
            "law_set_id": str(resolved["law_set_id"]),
            "root_metric_name": "root_count_mse",
            "parameterization": str(resolved["parameterization"]),
            "local_law_weight": float(resolved["local_law_weight"]),
        },
    )
    theorem_terms = [
        {
            "law_kind": LawKind.L1_LEAF.value,
            "paper_condition": LawKind.L1_LEAF.paper_condition,
            "lean_name": LawKind.L1_LEAF.lean_name,
            "name": "leaf_preservation",
            "weight": float(resolved["local_law_c1_weight"]),
            "share_within_local_law": float(resolved["local_law_c1_share"]),
            "active": uses_weighted_neural_objective
            and float(resolved["local_law_c1_weight"]) > 0.0,
            "evidence_status": EvidenceStatus.THEOREM_BACKED.value,
        },
        {
            "law_kind": LawKind.L2_MERGE.value,
            "paper_condition": LawKind.L2_MERGE.paper_condition,
            "lean_name": LawKind.L2_MERGE.lean_name,
            "name": "merge_preservation",
            "weight": float(resolved["local_law_c3_weight"]),
            "share_within_local_law": float(resolved["local_law_c3_share"]),
            "active": uses_weighted_neural_objective
            and float(resolved["local_law_c3_weight"]) > 0.0,
            "evidence_status": EvidenceStatus.THEOREM_BACKED.value,
        },
        {
            "law_kind": LawKind.L3_IDEMPOTENCE.value,
            "paper_condition": LawKind.L3_IDEMPOTENCE.paper_condition,
            "lean_name": LawKind.L3_IDEMPOTENCE.lean_name,
            "name": "idempotence",
            "weight": float(resolved["local_law_c2_weight"]),
            "share_within_local_law": float(resolved["local_law_c2_share"]),
            "active": uses_weighted_neural_objective
            and float(resolved["local_law_c2_weight"]) > 0.0,
            "evidence_status": EvidenceStatus.THEOREM_BACKED.value,
        },
    ]
    proxy_terms = [
        {
            "name": "schedule_consistency",
            "weight": float(proxy_weight),
            "active": uses_weighted_neural_objective and float(proxy_weight) > 0.0,
            "evidence_status": EvidenceStatus.PROXY_ONLY.value,
            "notes": "Associativity proxy over schedule spread; not a Lean local law.",
        }
    ]
    return {
        **resolved,
        "problem_id": str(config.problem_id),
        "method_id": str(config.method_id),
        "model_family": str(config.model_family),
        "training_scheme": (
            "weighted_neural_objective"
            if uses_weighted_neural_objective
            else "closed_form_label_fit"
        ),
        "task_objective_name": "root_count_mse",
        "root_share": float(root_share),
        "root_share_source": str(resolved["root_share_source"]),
        "task_objective_weight_source": "configured_objective_builder",
        "root_supervision_active": bool(uses_weighted_neural_objective and root_active),
        "task_supervision_active": bool(uses_weighted_neural_objective and root_active),
        "proxy_schedule_consistency_weight": float(proxy_weight),
        "local_law_active": bool(
            uses_weighted_neural_objective and float(resolved["local_law_weight"]) > 0.0
        ),
        "local_law_component_weights": {
            LAW_ID_LEAF_PRESERVATION: float(resolved["local_law_c1_weight"]),
            LAW_ID_ON_RANGE_IDEMPOTENCE: float(resolved["local_law_c2_weight"]),
            LAW_ID_MERGE_PRESERVATION: float(resolved["local_law_c3_weight"]),
        },
        "parameterization_overrides_legacy": bool(
            config.local_law_weight is not None or str(config.law_package or "").strip()
        ),
        "composite_objective": composite_spec.to_dict(),
        "theorem_terms": theorem_terms,
        "proxy_terms": proxy_terms,
        "formal_notes": (
            "The theorem-facing local-law bundle covers C1/L1 leaf preservation, "
            "C2/L3 on-range idempotence, and C3/L2 merge preservation. "
            f"The active task/local-law objective is {objective_formula}, with "
            "schedule_consistency reported separately as a proxy-only regularizer."
        ),
        "model_family_notes": (
            "These weights are active in the neural lane. The additive lane instead uses "
            "closed-form regression on the available C1/C3 labels and exact scalar re-summary for C2."
        ),
    }


def _validate_unified_fno_local_law_objective(
    config: OPSCountConfig,
    objective_summary: Mapping[str, Any],
) -> None:
    """Reject law-specific FNO weights that the bundled loss cannot honor."""

    if str(config.model_family) not in ("fno", "neural"):
        return
    local_law_weight = float(objective_summary.get("local_law_weight", 0.0) or 0.0)
    if local_law_weight <= 1e-12:
        return

    weights = (
        float(objective_summary.get("local_law_c1_weight", 0.0) or 0.0),
        float(objective_summary.get("local_law_c2_weight", 0.0) or 0.0),
        float(objective_summary.get("local_law_c3_weight", 0.0) or 0.0),
    )
    positive = [value for value in weights if value > 1e-12]
    if len(positive) == 3 and all(
        math.isclose(value, positive[0], rel_tol=1e-6, abs_tol=1e-9)
        for value in positive
    ):
        return

    raise ValueError(
        "FNO unified local-law training uses one bundled corrected_local_law loss; "
        "law-specific C1/C2/C3 weights or packages are not supported on this path. "
        "Use root_only, local_law_weight with all active laws, or explicit equal "
        "C1/C2/C3 shares."
    )


@dataclass(frozen=True)
class _ExactState:
    count: int
    first: int
    last: int


def _exact_from_span(doc: ChangepointMarkovDoc, span: Tuple[int, int]) -> _ExactState:
    start, end = span
    regs = doc.token_regimes[int(start) : int(end)]
    if len(regs) == 0:
        raise ValueError("empty span")
    return _ExactState(
        count=_changepoint_count(regs),
        first=int(regs[0]),
        last=int(regs[-1]),
    )


def _exact_merge(a: _ExactState, b: _ExactState) -> _ExactState:
    join = 0 if int(a.last) == int(b.first) else 1
    return _ExactState(
        count=int(a.count) + int(b.count) + int(join),
        first=int(a.first),
        last=int(b.last),
    )


@dataclass(frozen=True)
class _CountOnlyState:
    count: int


def _count_only_from_span(doc: ChangepointMarkovDoc, span: Tuple[int, int]) -> _CountOnlyState:
    start, end = span
    regs = doc.token_regimes[int(start) : int(end)]
    if len(regs) == 0:
        raise ValueError("empty span")
    return _CountOnlyState(count=_changepoint_count(regs))


def _count_only_merge(a: _CountOnlyState, b: _CountOnlyState) -> _CountOnlyState:
    return _CountOnlyState(count=int(a.count) + int(b.count))


@dataclass(frozen=True)
class _FlipState:
    count: int
    first: int
    last: int
    flipped: bool


def _flip_from_span(doc: ChangepointMarkovDoc, span: Tuple[int, int]) -> _FlipState:
    base = _exact_from_span(doc, span)
    return _FlipState(count=base.count, first=base.first, last=base.last, flipped=False)


def _flip_merge(a: _FlipState, b: _FlipState) -> _FlipState:
    base = _exact_merge(
        _ExactState(a.count, a.first, a.last), _ExactState(b.count, b.first, b.last)
    )
    return _FlipState(count=base.count, first=base.first, last=base.last, flipped=False)


def _flip_resummary(z: _FlipState) -> _FlipState:
    return _FlipState(
        count=int(z.count), first=int(z.first), last=int(z.last), flipped=not bool(z.flipped)
    )


def _flip_value(z: _FlipState) -> int:
    return int(z.count) + (1 if bool(z.flipped) else 0)


@dataclass(frozen=True)
class _LeafBucketState:
    exact_count: int
    readout_count: int
    first: int
    last: int


def _leaf_bucket_from_span(doc: ChangepointMarkovDoc, span: Tuple[int, int]) -> _LeafBucketState:
    base = _exact_from_span(doc, span)
    return _LeafBucketState(
        exact_count=int(base.count),
        readout_count=1,
        first=int(base.first),
        last=int(base.last),
    )


def _leaf_bucket_merge(a: _LeafBucketState, b: _LeafBucketState) -> _LeafBucketState:
    base = _exact_merge(
        _ExactState(a.exact_count, a.first, a.last),
        _ExactState(b.exact_count, b.first, b.last),
    )
    return _LeafBucketState(
        exact_count=int(base.count),
        readout_count=int(base.count),
        first=int(base.first),
        last=int(base.last),
    )


def _leaf_bucket_value(z: _LeafBucketState) -> int:
    return int(z.readout_count)


def _identity_resummary(z: StateT) -> StateT:
    return z


def _apply_resummary(z: StateT, *, rounds: int, resummary: Callable[[StateT], StateT]) -> StateT:
    cur = z
    for _ in range(int(max(0, rounds))):
        cur = resummary(cur)
    return cur


def _zero_sketch_metrics(*, n_docs: int) -> SketchMetrics:
    return SketchMetrics(
        root_mae=0.0,
        root_median_abs_error=0.0,
        root_p95_abs_error=0.0,
        schedule_spread_mean=0.0,
        schedule_spread_p95=0.0,
        leaf_mae=0.0,
        leaf_violation_rate=0.0,
        c2_idempotence_mae=0.0,
        c2_r1_mae=0.0,
        c2_r2_mae=0.0,
        c2_r4_mae=0.0,
        resummary_root_drift_r1=0.0,
        resummary_root_drift_r2=0.0,
        resummary_root_drift_r4=0.0,
        merge_mae=0.0,
        merge_violation_rate=0.0,
        n_docs=int(n_docs),
        c2_state_replay_mse=0.0,
    )


def _eval_structured_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
    from_span: Callable[[ChangepointMarkovDoc, Tuple[int, int]], StateT],
    merge: Callable[[StateT, StateT], StateT],
    value: Callable[[StateT], float],
    resummary: Callable[[StateT], StateT],
) -> SketchMetrics:
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []
    c2_r1_abs: List[float] = []
    c2_r2_abs: List[float] = []
    c2_r4_abs: List[float] = []
    root_drift_r1: List[float] = []
    root_drift_r2: List[float] = []
    root_drift_r4: List[float] = []

    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        spans = _leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        leaf_states = [from_span(doc, sp) for sp in spans]
        leaf_truth = [_oracle_count(doc, start=sp[0], end=sp[1]) for sp in spans]
        for st, truth in zip(leaf_states, leaf_truth):
            leaf_abs.append(abs(float(value(st)) - float(truth)))

        # Root predictions for schedule spread.
        roots: Dict[str, float] = {}
        balanced_root_state: Optional[StateT] = None
        balanced_states: List[StateT] = list(leaf_states)
        for sched in VALID_SCHEDULES:
            if str(sched) == "balanced":
                cur_s = list(leaf_states)
                cur_p = list(spans)
                while len(cur_s) > 1:
                    nxt_s: List[StateT] = []
                    nxt_p: List[Tuple[int, int]] = []
                    i = 0
                    while i < len(cur_s):
                        if i + 1 >= len(cur_s):
                            nxt_s.append(cur_s[i])
                            nxt_p.append(cur_p[i])
                            i += 1
                            continue
                        merged = merge(cur_s[i], cur_s[i + 1])
                        parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                        nxt_s.append(merged)
                        balanced_states.append(merged)
                        nxt_p.append(parent)
                        i += 2
                    cur_s, cur_p = nxt_s, nxt_p
                balanced_root_state = cur_s[0]
                roots[str(sched)] = float(value(cur_s[0]))
            elif str(sched) == "left_to_right":
                acc = leaf_states[0]
                for st in leaf_states[1:]:
                    acc = merge(acc, st)
                roots[str(sched)] = float(value(acc))
            elif str(sched) == "right_to_left":
                acc = leaf_states[-1]
                for st in reversed(leaf_states[:-1]):
                    acc = merge(st, acc)
                roots[str(sched)] = float(value(acc))
            else:
                raise ValueError(f"unsupported schedule: {sched!r}")

        truth_root = float(_oracle_count(doc, start=0, end=n_tok))
        pred = roots["balanced"]
        root_abs.append(abs(pred - truth_root))
        spreads.append(max(roots.values()) - min(roots.values()))
        if balanced_root_state is None:
            raise ValueError("balanced schedule must produce a root state")

        for state in balanced_states:
            base = float(value(state))
            c2_r1_abs.append(
                abs(float(value(_apply_resummary(state, rounds=1, resummary=resummary))) - base)
            )
            c2_r2_abs.append(
                abs(float(value(_apply_resummary(state, rounds=2, resummary=resummary))) - base)
            )
            c2_r4_abs.append(
                abs(float(value(_apply_resummary(state, rounds=4, resummary=resummary))) - base)
            )
        root_base = float(value(balanced_root_state))
        root_drift_r1.append(
            abs(
                float(value(_apply_resummary(balanced_root_state, rounds=1, resummary=resummary)))
                - root_base
            )
        )
        root_drift_r2.append(
            abs(
                float(value(_apply_resummary(balanced_root_state, rounds=2, resummary=resummary)))
                - root_base
            )
        )
        root_drift_r4.append(
            abs(
                float(value(_apply_resummary(balanced_root_state, rounds=4, resummary=resummary)))
                - root_base
            )
        )

        # C3 discrepancies (balanced schedule only).
        cur_s = list(leaf_states)
        cur_p = list(spans)
        while len(cur_s) > 1:
            nxt_s = []
            nxt_p = []
            i = 0
            while i < len(cur_s):
                if i + 1 >= len(cur_s):
                    nxt_s.append(cur_s[i])
                    nxt_p.append(cur_p[i])
                    i += 1
                    continue
                merged = merge(cur_s[i], cur_s[i + 1])
                parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                truth_parent = float(_oracle_count(doc, start=parent[0], end=parent[1]))
                merge_abs.append(abs(float(value(merged)) - truth_parent))
                nxt_s.append(merged)
                nxt_p.append(parent)
                i += 2
            cur_s, cur_p = nxt_s, nxt_p

    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    c2_r1_arr = np.asarray(c2_r1_abs, dtype=np.float64)
    c2_r2_arr = np.asarray(c2_r2_abs, dtype=np.float64)
    c2_r4_arr = np.asarray(c2_r4_abs, dtype=np.float64)
    root_drift_r1_arr = np.asarray(root_drift_r1, dtype=np.float64)
    root_drift_r2_arr = np.asarray(root_drift_r2, dtype=np.float64)
    root_drift_r4_arr = np.asarray(root_drift_r4, dtype=np.float64)

    tau = float(tau)
    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=(
            float(np.mean((leaf_abs_arr > tau).astype(np.float64))) if leaf_abs_arr.size else 0.0
        ),
        c2_idempotence_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r1_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r2_mae=float(np.mean(c2_r2_arr)) if c2_r2_arr.size else 0.0,
        c2_r4_mae=float(np.mean(c2_r4_arr)) if c2_r4_arr.size else 0.0,
        resummary_root_drift_r1=(
            float(np.mean(root_drift_r1_arr)) if root_drift_r1_arr.size else 0.0
        ),
        resummary_root_drift_r2=(
            float(np.mean(root_drift_r2_arr)) if root_drift_r2_arr.size else 0.0
        ),
        resummary_root_drift_r4=(
            float(np.mean(root_drift_r4_arr)) if root_drift_r4_arr.size else 0.0
        ),
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=(
            float(np.mean((merge_abs_arr > tau).astype(np.float64))) if merge_abs_arr.size else 0.0
        ),
        n_docs=int(len(docs)),
    )


def _eval_exact_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
) -> SketchMetrics:
    return _eval_structured_family(
        docs,
        leaf_tokens=int(leaf_tokens),
        tau=float(tau),
        from_span=_exact_from_span,
        merge=_exact_merge,
        value=lambda z: float(z.count),
        resummary=_identity_resummary,
    )


def _eval_count_only_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
) -> SketchMetrics:
    return _eval_structured_family(
        docs,
        leaf_tokens=int(leaf_tokens),
        tau=float(tau),
        from_span=_count_only_from_span,
        merge=_count_only_merge,
        value=lambda z: float(z.count),
        resummary=_identity_resummary,
    )


def _eval_leaf_bucket_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
) -> SketchMetrics:
    return _eval_structured_family(
        docs,
        leaf_tokens=int(leaf_tokens),
        tau=float(tau),
        from_span=_leaf_bucket_from_span,
        merge=_leaf_bucket_merge,
        value=lambda z: float(_leaf_bucket_value(z)),
        resummary=_identity_resummary,
    )


def _eval_flip_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
    rounds: int,
) -> SketchMetrics:
    base = _eval_structured_family(
        docs,
        leaf_tokens=int(leaf_tokens),
        tau=float(tau),
        from_span=_flip_from_span,
        merge=_flip_merge,
        value=lambda z: float(_flip_value(z)),
        resummary=_flip_resummary,
    )
    if int(rounds) <= 1:
        return base
    drift_by_round = {
        2: float(base.resummary_root_drift_r1),
        3: float(base.resummary_root_drift_r2),
        5: float(base.resummary_root_drift_r4),
    }
    root_round_drift = float(drift_by_round.get(int(rounds), base.resummary_root_drift_r1))
    return SketchMetrics(
        root_mae=float(root_round_drift),
        root_median_abs_error=float(root_round_drift),
        root_p95_abs_error=float(root_round_drift),
        schedule_spread_mean=float(base.schedule_spread_mean),
        schedule_spread_p95=float(base.schedule_spread_p95),
        leaf_mae=float(base.leaf_mae),
        leaf_violation_rate=float(base.leaf_violation_rate),
        c2_idempotence_mae=float(base.c2_idempotence_mae),
        c2_r1_mae=float(base.c2_r1_mae),
        c2_r2_mae=float(base.c2_r2_mae),
        c2_r4_mae=float(base.c2_r4_mae),
        resummary_root_drift_r1=float(base.resummary_root_drift_r1),
        resummary_root_drift_r2=float(base.resummary_root_drift_r2),
        resummary_root_drift_r4=float(base.resummary_root_drift_r4),
        merge_mae=float(base.merge_mae),
        merge_violation_rate=float(base.merge_violation_rate),
        n_docs=int(base.n_docs),
    )


def _span_features(
    doc: ChangepointMarkovDoc,
    span: Tuple[int, int],
    *,
    n_regimes: int,
    vocab_size: int,
    mode: str,
) -> torch.Tensor:
    start, end = span
    regs = np.asarray(doc.token_regimes[int(start) : int(end)], dtype=np.int64)
    toks = np.asarray(doc.tokens[int(start) : int(end)], dtype=np.int64)
    if regs.size == 0 or toks.size == 0:
        raise ValueError("empty span")
    n = int(n_regimes)
    v = int(vocab_size)

    if mode not in {"full", "no_endpoints", "token_full", "token_bow"}:
        raise ValueError(
            "unsupported feature_mode: "
            f"{mode!r} (expected 'full', 'no_endpoints', 'token_full', or 'token_bow')"
        )

    parts: List[np.ndarray] = []
    if mode == "full":
        first = np.zeros((n,), dtype=np.float32)
        last = np.zeros((n,), dtype=np.float32)
        first[int(regs[0])] = 1.0
        last[int(regs[-1])] = 1.0
        parts.extend([first, last])
    elif mode == "token_full":
        first_tok = np.zeros((v,), dtype=np.float32)
        last_tok = np.zeros((v,), dtype=np.float32)
        first_tok[int(toks[0])] = 1.0
        last_tok[int(toks[-1])] = 1.0
        unigram = np.bincount(toks, minlength=v).astype(np.float32, copy=False)
        unigram /= float(max(1, toks.size))
        pair_dim = int(v) * int(v)
        if toks.size >= 2:
            pair_idx = toks[:-1] * int(v) + toks[1:]
            bigram = np.bincount(pair_idx, minlength=pair_dim).astype(np.float32, copy=False)
            bigram /= float(max(1, toks.size - 1))
        else:
            bigram = np.zeros((pair_dim,), dtype=np.float32)
        parts.extend([first_tok, last_tok, unigram, bigram])
    elif mode == "token_bow":
        unigram = np.bincount(toks, minlength=v).astype(np.float32, copy=False)
        unigram /= float(max(1, toks.size))
        parts.append(unigram)

    if mode in {"full", "no_endpoints"}:
        trans = np.zeros((n, n), dtype=np.float32)
        if regs.size >= 2:
            for a, b in zip(regs[:-1], regs[1:]):
                trans[int(a), int(b)] += 1.0
            trans /= float(max(1, regs.size - 1))
        parts.append(trans.reshape(-1))

    # Length feature (helps disambiguate sparse leaves).
    parts.append(np.asarray([float(toks.size)], dtype=np.float32))

    feat = np.concatenate(parts, axis=0)
    return torch.tensor(feat, dtype=torch.float32)


@dataclass(frozen=True)
class _CountDoc:
    n_tokens: int
    leaf_features: Tuple[torch.Tensor, ...]  # CPU float32
    leaf_counts: Tuple[float, ...]
    merge_counts_balanced: Tuple[
        float, ...
    ]  # oracle counts for each realized merge (balanced order)
    merge_sizes_balanced: Tuple[int, ...]  # number of leaves under each realized merge
    root_count: float


@dataclass(frozen=True)
class _SampledLeafPoolDoc:
    n_tokens: int
    total_leaves: int
    feature_vector: np.ndarray
    root_count: float
    sampled_leaves: int
    sampled_tokens: int
    sampled_token_fraction: float


def _to_device(xs: Sequence[torch.Tensor], *, device: torch.device) -> List[torch.Tensor]:
    return [x.to(device=device) for x in xs]


def _prepare_count_docs(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    n_regimes: int,
    vocab_size: int,
    feature_mode: str,
) -> Tuple[_CountDoc, ...]:
    out: List[_CountDoc] = []
    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        spans = _leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        leaf_feats = tuple(
            _span_features(
                doc,
                sp,
                n_regimes=int(n_regimes),
                vocab_size=int(vocab_size),
                mode=str(feature_mode),
            )
            for sp in spans
        )
        leaf_counts = tuple(float(_oracle_count(doc, start=sp[0], end=sp[1])) for sp in spans)
        # Balanced merge labels (oracle on the realized internal nodes).
        cur_spans = list(spans)
        cur_sizes = [1 for _ in spans]
        merge_counts: List[float] = []
        merge_sizes: List[int] = []
        while len(cur_spans) > 1:
            nxt_spans: List[Tuple[int, int]] = []
            nxt_sizes: List[int] = []
            i = 0
            while i < len(cur_spans):
                if i + 1 >= len(cur_spans):
                    nxt_spans.append(cur_spans[i])
                    nxt_sizes.append(int(cur_sizes[i]))
                    i += 1
                    continue
                parent = (int(cur_spans[i][0]), int(cur_spans[i + 1][1]))
                parent_size = int(cur_sizes[i]) + int(cur_sizes[i + 1])
                merge_counts.append(float(_oracle_count(doc, start=parent[0], end=parent[1])))
                merge_sizes.append(int(parent_size))
                nxt_spans.append(parent)
                nxt_sizes.append(int(parent_size))
                i += 2
            cur_spans = nxt_spans
            cur_sizes = nxt_sizes
        root_count = float(_oracle_count(doc, start=0, end=n_tok))
        out.append(
            _CountDoc(
                n_tokens=int(n_tok),
                leaf_features=leaf_feats,
                leaf_counts=leaf_counts,
                merge_counts_balanced=tuple(merge_counts),
                merge_sizes_balanced=tuple(merge_sizes),
                root_count=float(root_count),
            )
        )
    return tuple(out)


def _prepare_doc_level_count_docs(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    n_regimes: int,
    vocab_size: int,
    feature_mode: str,
) -> Tuple[_CountDoc, ...]:
    out: List[_CountDoc] = []
    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        root_count = float(_oracle_count(doc, start=0, end=n_tok))
        out.append(
            _CountDoc(
                n_tokens=int(n_tok),
                leaf_features=(
                    _span_features(
                        doc,
                        (0, n_tok),
                        n_regimes=int(n_regimes),
                        vocab_size=int(vocab_size),
                        mode=str(feature_mode),
                    ),
                ),
                leaf_counts=(float(root_count),),
                merge_counts_balanced=tuple(),
                merge_sizes_balanced=tuple(),
                root_count=float(root_count),
            )
        )
    return tuple(out)


def _doc_level_supervision_dataset(
    docs: Sequence[_CountDoc],
    *,
    split: str,
    target_scale: float,
) -> SupervisionDataset:
    rows: List[DenseSupervisionExample] = []
    rubric = (
        "Predict the full-document changepoint count from a single dense document representation."
    )
    for index, doc in enumerate(docs):
        if not doc.leaf_features:
            continue
        feature_vector = (
            doc.leaf_features[0]
            .detach()
            .cpu()
            .to(dtype=torch.float32)
            .numpy()
            .astype(float)
            .tolist()
        )
        doc_id = f"{split}_doc_{index}"
        rows.append(
            DenseSupervisionExample(
                example_id=doc_id,
                features=feature_vector,
                scalar_target=float(doc.root_count),
                original_text=f"markov_changepoint_ops_count::{doc_id}",
                rubric=rubric,
                response="single_full_document_candidate",
                response_id=f"{doc_id}:single_candidate",
                reference_score=float(doc.root_count),
                source_doc_id=doc_id,
                truth_label_source="oracle",
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=1.0,
                    label_propensity=1.0,
                    sampling_scheme="full_document_supervision",
                    policy_name="all_documents",
                    unit_kind=ObservationUnitKind.DOCUMENT,
                    supports_ipw_estimation=True,
                ),
                metadata={
                    "dgp": "markov_changepoint_ops_count",
                    "input_view": "single_full_document_leaf",
                    "uses_tree_merges": False,
                    "n_tokens": int(doc.n_tokens),
                    "target_scale": float(target_scale),
                },
            )
        )
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="markov_ops_count",
        supervision_signal_name="document_level_target",
        response_signal_name="changepoint_count",
        law_type="document_level_target",
        split=str(split),
        response_signal_min=0.0,
        response_signal_max=float(target_scale),
        metadata={
            "dgp": "markov_changepoint_ops_count",
            "input_view": "single_full_document_leaf",
            "uses_tree_merges": False,
        },
    )


class DocSequenceBoundaryRegressor(nn.Module):
    def __init__(self, *, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_id = int(vocab_size)
        self.embedding = nn.Embedding(
            int(vocab_size) + 1,
            int(emb_dim),
            padding_idx=self.pad_id,
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(2 * int(emb_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.embedding(tokens)
        pair_feat = torch.cat(
            [
                emb[:, :-1, :],
                emb[:, 1:, :],
            ],
            dim=-1,
        )
        logits = self.boundary_head(pair_feat).squeeze(-1)
        boundary_mask = (token_mask[:, :-1] * token_mask[:, 1:]).to(dtype=logits.dtype)
        scores = torch.sigmoid(logits) * boundary_mask
        return torch.sum(scores, dim=1)


def _token_sequence_arrays(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not docs:
        return (
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    max_len = max(int(len(doc.tokens)) for doc in docs)
    toks = np.full((len(docs), max_len), int(pad_id), dtype=np.int64)
    mask = np.zeros((len(docs), max_len), dtype=np.float32)
    targets = np.zeros((len(docs),), dtype=np.float32)
    for idx, doc in enumerate(docs):
        n = int(len(doc.tokens))
        doc_tokens = np.asarray(doc.tokens, dtype=np.int64)
        if doc_tokens.size > 0 and (
            int(np.min(doc_tokens)) < 0 or int(np.max(doc_tokens)) >= int(pad_id)
        ):
            raise ValueError(
                f"encountered token ids outside [0, {int(pad_id)}) while building "
                f"full-document token arrays: min={int(np.min(doc_tokens))}, "
                f"max={int(np.max(doc_tokens))}"
            )
        toks[idx, :n] = doc_tokens
        mask[idx, :n] = 1.0
        targets[idx] = float(_oracle_count(doc, start=0, end=n))
    return toks, mask, targets


def _token_sequence_input_signature(
    tokens: np.ndarray,
    mask: np.ndarray,
) -> str:
    h = hashlib.sha256()
    toks_arr = np.asarray(tokens, dtype=np.int64, order="C")
    mask_arr = np.asarray(mask, dtype=np.float32, order="C")
    h.update(str(tuple(int(x) for x in toks_arr.shape)).encode("utf-8"))
    h.update(str(tuple(int(x) for x in mask_arr.shape)).encode("utf-8"))
    h.update(toks_arr.tobytes())
    h.update(mask_arr.tobytes())
    return h.hexdigest()


def _shared_full_sequence_input_signatures(
    *,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
    pad_id: int,
) -> Dict[str, str]:
    train_tokens, train_mask, _ = _token_sequence_arrays(train_docs, pad_id=int(pad_id))
    val_tokens, val_mask, _ = _token_sequence_arrays(val_docs, pad_id=int(pad_id))
    test_tokens, test_mask, _ = _token_sequence_arrays(test_docs, pad_id=int(pad_id))
    return {
        "train": _token_sequence_input_signature(train_tokens, train_mask),
        "val": _token_sequence_input_signature(val_tokens, val_mask),
        "test": _token_sequence_input_signature(test_tokens, test_mask),
    }


def _token_sequence_endpoint_targets(
    tokens: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    toks = np.asarray(tokens, dtype=np.int64)
    mask_arr = np.asarray(mask, dtype=np.float32)
    if toks.ndim != 2 or mask_arr.ndim != 2:
        raise ValueError("tokens and mask must be rank-2 arrays")
    first = np.zeros((int(toks.shape[0]),), dtype=np.int64)
    last = np.zeros((int(toks.shape[0]),), dtype=np.int64)
    for idx in range(int(toks.shape[0])):
        valid = np.flatnonzero(mask_arr[idx] > 0.0)
        if valid.size <= 0:
            continue
        first[idx] = int(toks[idx, int(valid[0])])
        last[idx] = int(toks[idx, int(valid[-1])])
    return first, last


class FullSequenceCTreePOOperator(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        token_embedding_dim: int,
        sketch_dim: int,
        hidden_dim: int,
        target_max: float,
        n_count_classes: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(vocab_size)
        self.token_embedding = nn.Embedding(
            int(vocab_size) + 1,
            int(token_embedding_dim),
            padding_idx=self.pad_id,
        )
        self.operator = CTreePOModel(
            CTreePOConfig(
                embedding_dim=int(token_embedding_dim),
                sketch_dim=int(sketch_dim),
                hidden_dim=int(hidden_dim),
                merge_type="residual_gated",
                head_names=("count",),
                target_min=0.0,
                target_max=float(target_max),
            )
        )
        head_hidden = int(max(64, hidden_dim // 2))
        self.count_classifier = nn.Sequential(
            nn.Linear(int(sketch_dim), head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, int(n_count_classes)),
        )
        self.count_classifier_skip = nn.Linear(int(sketch_dim), int(n_count_classes))
        self.first_token_classifier = nn.Sequential(
            nn.Linear(int(sketch_dim), head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, int(vocab_size)),
        )
        self.first_token_classifier_skip = nn.Linear(int(sketch_dim), int(vocab_size))
        self.last_token_classifier = nn.Sequential(
            nn.Linear(int(sketch_dim), head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, int(vocab_size)),
        )
        self.last_token_classifier_skip = nn.Linear(int(sketch_dim), int(vocab_size))

    def _merge_token_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a rank-1 sequence")
        if int(token_ids.numel()) <= 0:
            raise ValueError("token sequence must contain at least one token")
        token_emb = self.token_embedding(token_ids)
        states = self.operator.encode_leaf(token_emb)
        while int(states.shape[0]) > 1:
            n_pairs = int(states.shape[0] // 2)
            merged = self.operator.merge(
                states[: 2 * n_pairs : 2],
                states[1 : 2 * n_pairs : 2],
            )
            if int(states.shape[0]) % 2 == 1:
                states = torch.cat([merged, states[-1:]], dim=0)
            else:
                states = merged
        return states[0]

    def _merge_token_batch(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or token_mask.ndim != 2:
            raise ValueError("tokens and token_mask must both be rank-2")
        if int(tokens.shape[0]) <= 0:
            return torch.zeros(
                (0, int(self.operator.config.sketch_dim)),
                dtype=torch.float32,
                device=tokens.device,
            )
        token_emb = self.token_embedding(tokens)
        states = self.operator.encode_leaf(token_emb)
        active_mask = token_mask > 0.0
        zero_state = torch.zeros_like(states[:, :1, :])
        while int(states.shape[1]) > 1:
            if int(states.shape[1]) % 2 == 1:
                states = torch.cat([states, zero_state], dim=1)
                active_mask = torch.cat(
                    [
                        active_mask,
                        torch.zeros(
                            (int(active_mask.shape[0]), 1),
                            dtype=torch.bool,
                            device=active_mask.device,
                        ),
                    ],
                    dim=1,
                )
            left = states[:, 0::2, :]
            right = states[:, 1::2, :]
            left_mask = active_mask[:, 0::2]
            right_mask = active_mask[:, 1::2]
            merged = self.operator.merge(left, right)
            parent = torch.where(left_mask.unsqueeze(-1) & right_mask.unsqueeze(-1), merged, left)
            parent = torch.where((~left_mask).unsqueeze(-1) & right_mask.unsqueeze(-1), right, parent)
            states = parent
            active_mask = left_mask | right_mask
            zero_state = torch.zeros_like(states[:, :1, :])
        roots = states[:, 0, :]
        has_any = torch.any(token_mask > 0.0, dim=1, keepdim=True)
        return torch.where(has_any, roots, torch.zeros_like(roots))

    def encode_root_states(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or token_mask.ndim != 2:
            raise ValueError("tokens and token_mask must both be rank-2")
        return self._merge_token_batch(tokens, token_mask=token_mask)

    def predict_normalized(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        roots = self.encode_root_states(tokens, token_mask=token_mask)
        return self.operator.predict_normalized(roots, head="count").reshape(-1)

    def predict_count_logits(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        roots = self.encode_root_states(tokens, token_mask=token_mask)
        return self.count_classifier(roots) + self.count_classifier_skip(roots)

    def predict_endpoint_logits(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        roots = self.encode_root_states(tokens, token_mask=token_mask)
        first = self.first_token_classifier(roots) + self.first_token_classifier_skip(roots)
        last = self.last_token_classifier(roots) + self.last_token_classifier_skip(roots)
        return first, last


class FullSequenceBoundaryTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_positions: int,
        model_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        n_count_classes: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(vocab_size)
        self.token_embedding = nn.Embedding(
            int(vocab_size) + 1,
            int(model_dim),
            padding_idx=self.pad_id,
        )
        self.position_embedding = nn.Embedding(int(max_positions), int(model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(n_heads),
            dim_feedforward=int(hidden_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.output_norm = nn.LayerNorm(int(model_dim))
        pair_hidden = int(max(64, hidden_dim // 2))
        self.boundary_head = nn.Sequential(
            nn.Linear(2 * int(model_dim), pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, 1),
        )
        self.summary_norm = nn.LayerNorm(int(model_dim))
        self.count_classifier = nn.Sequential(
            nn.Linear(int(model_dim), pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, int(n_count_classes)),
        )
        self.count_classifier_skip = nn.Linear(int(model_dim), int(n_count_classes))
        self.first_token_classifier = nn.Sequential(
            nn.Linear(int(model_dim), pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, int(vocab_size)),
        )
        self.first_token_classifier_skip = nn.Linear(int(model_dim), int(vocab_size))
        self.last_token_classifier = nn.Sequential(
            nn.Linear(int(model_dim), pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, int(vocab_size)),
        )
        self.last_token_classifier_skip = nn.Linear(int(model_dim), int(vocab_size))

    def encode_sequence(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or token_mask.ndim != 2:
            raise ValueError("tokens and token_mask must both be rank-2")
        batch, width = int(tokens.shape[0]), int(tokens.shape[1])
        positions = torch.arange(width, device=tokens.device).unsqueeze(0).expand(batch, width)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        key_padding_mask = token_mask <= 0.0
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.output_norm(encoded)

    def pooled_summary(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode_sequence(tokens, token_mask=token_mask)
        weights = token_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        denom = torch.clamp(torch.sum(weights, dim=1), min=1.0)
        pooled = torch.sum(encoded * weights, dim=1) / denom
        return self.summary_norm(pooled)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode_sequence(tokens, token_mask=token_mask)
        pair_feat = torch.cat([encoded[:, :-1, :], encoded[:, 1:, :]], dim=-1)
        logits = self.boundary_head(pair_feat).squeeze(-1)
        boundary_mask = (token_mask[:, :-1] * token_mask[:, 1:]).to(dtype=logits.dtype)
        return torch.sum(torch.sigmoid(logits) * boundary_mask, dim=1)

    def predict_count_logits(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pooled_summary(tokens, token_mask=token_mask)
        return self.count_classifier(pooled) + self.count_classifier_skip(pooled)

    def predict_endpoint_logits(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pooled_summary(tokens, token_mask=token_mask)
        first = self.first_token_classifier(pooled) + self.first_token_classifier_skip(pooled)
        last = self.last_token_classifier(pooled) + self.last_token_classifier_skip(pooled)
        return first, last


def _eval_root_predictions(
    preds: Sequence[float],
    truths: Sequence[float],
    *,
    tau: float,
    condition_ids: Sequence[str] | None = None,
) -> SketchMetrics:
    if len(truths) == 0:
        return _zero_sketch_metrics(n_docs=0)
    y = np.asarray([float(value) for value in truths], dtype=np.float64)
    pred = np.asarray([float(value) for value in preds], dtype=np.float64)
    if y.shape != pred.shape:
        raise ValueError("root predictions must align with truths")
    abs_err = np.abs(pred - y)
    sq_err = (pred - y) ** 2
    tau_v = float(tau)
    condition_metrics = _condition_error_diagnostics(abs_err, condition_ids)
    _nan = float("nan")
    return SketchMetrics(
        root_mae=float(np.mean(abs_err)),
        root_mse=float(np.mean(sq_err)),
        root_median_abs_error=float(np.median(abs_err)),
        root_p95_abs_error=float(np.percentile(abs_err, 95.0)),
        schedule_spread_mean=_nan,
        schedule_spread_p95=_nan,
        leaf_mae=float(np.mean(abs_err)),
        leaf_violation_rate=float(np.mean((abs_err > tau_v).astype(np.float64))),
        # Local-law metrics are not applicable for root-only evaluation.
        c2_idempotence_mae=_nan,
        c2_r1_mae=_nan,
        c2_r2_mae=_nan,
        c2_r4_mae=_nan,
        resummary_root_drift_r1=_nan,
        resummary_root_drift_r2=_nan,
        resummary_root_drift_r4=_nan,
        merge_mae=_nan,
        merge_violation_rate=_nan,
        n_docs=int(len(truths)),
        condition_root_mae=dict(condition_metrics["condition_root_mae"]),
        condition_root_n_docs=dict(condition_metrics["condition_root_n_docs"]),
        condition_root_macro_mae=float(condition_metrics["condition_root_macro_mae"]),
        condition_root_worst_mae=float(condition_metrics["condition_root_worst_mae"]),
    )


def _exact_match_rate(
    preds: Sequence[float],
    truths: Sequence[float],
) -> float:
    if len(truths) == 0:
        return float("nan")
    y = np.asarray([float(value) for value in truths], dtype=np.float64)
    pred = np.asarray([float(value) for value in preds], dtype=np.float64)
    if y.shape != pred.shape:
        raise ValueError("exact-match predictions must align with truths")
    return float(np.mean((np.rint(pred) == np.rint(y)).astype(np.float64)))


def _eval_doc_level_dense_predictions(
    preds: Sequence[float],
    docs: Sequence[_CountDoc],
    *,
    tau: float,
) -> SketchMetrics:
    return _eval_root_predictions(
        preds,
        [float(doc.root_count) for doc in docs],
        tau=float(tau),
    )


def _doc_level_feature_matrix(docs: Sequence[_CountDoc]) -> np.ndarray:
    if not docs:
        raise ValueError("doc-level feature matrix requires at least one document")
    rows: List[np.ndarray] = []
    for doc in docs:
        if not doc.leaf_features:
            raise ValueError("doc-level baseline requires a full-document feature vector")
        rows.append(
            doc.leaf_features[0]
            .detach()
            .cpu()
            .to(dtype=torch.float32)
            .numpy()
            .astype(np.float64, copy=False)
        )
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _normalized_ngram_orders(raw_orders: Sequence[int]) -> Tuple[int, ...]:
    seen: set[int] = set()
    orders: List[int] = []
    for raw in raw_orders:
        value = int(raw)
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        orders.append(int(value))
    return tuple(sorted(orders))


def _ngram_order_label(order: int) -> str:
    value = int(order)
    if value == 1:
        return "unigram"
    if value == 2:
        return "bigram"
    if value == 3:
        return "trigram"
    return f"{value}gram"


def _token_ngram_count_vector(
    doc: ChangepointMarkovDoc,
    *,
    vocab_size: int,
    order: int,
) -> np.ndarray:
    toks = np.asarray(doc.tokens, dtype=np.int64)
    v = int(vocab_size)
    k = int(order)
    if k <= 0:
        raise ValueError("ngram order must be positive")
    dim = int(v**k)
    if dim > 250_000:
        raise ValueError(
            f"token ngram feature dimension too large for explicit order-{k} counts "
            f"(vocab_size={v}, dim={dim}); reduce vocab_size or omit this order"
        )
    out = np.zeros((dim,), dtype=np.float64)
    if toks.size < k:
        return out
    idx = np.zeros((int(toks.size - k + 1),), dtype=np.int64)
    for offset in range(k):
        idx *= int(v)
        idx += toks[offset : toks.size - k + offset + 1]
    counts = np.bincount(idx, minlength=dim).astype(np.float64, copy=False)
    out[: counts.shape[0]] = counts
    return out


def _doc_token_ngram_feature_matrix(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    vocab_size: int,
    orders: Sequence[int],
) -> np.ndarray:
    if not docs:
        raise ValueError("doc token ngram feature matrix requires at least one document")
    normalized_orders = _normalized_ngram_orders(orders)
    if not normalized_orders:
        raise ValueError("at least one positive ngram order is required")
    rows: List[np.ndarray] = []
    for doc in docs:
        parts = [
            _token_ngram_count_vector(
                doc,
                vocab_size=int(vocab_size),
                order=int(order),
            )
            for order in normalized_orders
        ]
        rows.append(np.concatenate(parts, axis=0).astype(np.float64, copy=False))
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _doc_root_targets(docs: Sequence[ChangepointMarkovDoc]) -> np.ndarray:
    return np.asarray(
        [
            float(_oracle_count(doc, start=0, end=int(len(doc.tokens))))
            for doc in docs
        ],
        dtype=np.float64,
    )


def _dense_doc_matrix_supervision_dataset(
    feature_matrix: np.ndarray,
    targets: Sequence[float],
    *,
    split: str,
    target_scale: float,
    input_view: str,
    metadata: Mapping[str, object] | None = None,
) -> SupervisionDataset:
    features = np.asarray(feature_matrix, dtype=np.float64)
    y = np.asarray([float(value) for value in targets], dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("feature_matrix must be rank-2")
    if int(features.shape[0]) != int(y.shape[0]):
        raise ValueError("feature_matrix rows must align with targets")
    rows: List[DenseSupervisionExample] = []
    rubric = "Predict the full-document changepoint count from full-document token ngram counts."
    extra_meta = dict(metadata or {})
    for index in range(int(features.shape[0])):
        doc_id = f"{split}_doc_{index}"
        rows.append(
            DenseSupervisionExample(
                example_id=doc_id,
                features=features[index].astype(float, copy=False).tolist(),
                scalar_target=float(y[index]),
                original_text=f"markov_changepoint_ops_count::{doc_id}",
                rubric=rubric,
                response="single_full_document_candidate",
                response_id=f"{doc_id}:single_candidate",
                reference_score=float(y[index]),
                source_doc_id=doc_id,
                truth_label_source="oracle",
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=1.0,
                    label_propensity=1.0,
                    sampling_scheme="full_document_supervision",
                    policy_name="all_documents",
                    unit_kind=ObservationUnitKind.DOCUMENT,
                    supports_ipw_estimation=True,
                ),
                metadata={
                    "dgp": "markov_changepoint_ops_count",
                    "input_view": str(input_view),
                    "uses_tree_merges": False,
                    "target_scale": float(target_scale),
                    **extra_meta,
                },
            )
        )
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="markov_ops_count",
        supervision_signal_name="document_level_target",
        response_signal_name="changepoint_count",
        law_type="document_level_target",
        split=str(split),
        response_signal_min=0.0,
        response_signal_max=float(target_scale),
        metadata={
            "dgp": "markov_changepoint_ops_count",
            "input_view": str(input_view),
            "uses_tree_merges": False,
            **extra_meta,
        },
    )


def _sampled_leaf_budget_values(config: OPSCountConfig) -> Tuple[int, ...]:
    seen: set[int] = set()
    out: List[int] = []
    for raw in tuple(config.sampled_leaf_pool_leaf_counts):
        value = int(raw)
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(int(value))
    return tuple(sorted(out))


def _leaf_core_feature_vector(
    feature: torch.Tensor,
    *,
    n_regimes: int,
    use_endpoints: bool,
) -> np.ndarray:
    if bool(use_endpoints):
        return (
            feature[2 * int(n_regimes) :]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
    return feature.detach().cpu().numpy().astype(np.float64, copy=False)


def _feature_leaf_token_count(feature: torch.Tensor) -> int:
    flat = feature.detach().cpu().reshape(-1)
    if flat.numel() <= 0:
        return 0
    return int(max(0.0, float(flat[-1].item())))


def _leaf_endpoint_table_feature_vector(
    feature: torch.Tensor,
    *,
    n_regimes: int,
    use_endpoints: bool,
) -> np.ndarray:
    leaf_len = int(_feature_leaf_token_count(feature))
    if bool(use_endpoints):
        first_id = int(torch.argmax(feature[: int(n_regimes)]).item())
        last_id = int(torch.argmax(feature[int(n_regimes) : 2 * int(n_regimes)]).item())
        return np.asarray([float(first_id), float(last_id), float(leaf_len)], dtype=np.float64)
    return np.asarray([float(leaf_len)], dtype=np.float64)


def _leaf_endpoint_table_key(features: np.ndarray) -> Tuple[int, int, int]:
    flat = np.asarray(features, dtype=np.float64).reshape(-1)
    if flat.size >= 3:
        return (
            int(round(float(flat[0]))),
            int(round(float(flat[1]))),
            int(round(float(flat[2]))),
        )
    if flat.size == 1:
        return (-1, -1, int(round(float(flat[0]))))
    raise ValueError("leaf endpoint table features must contain at least one value")


def _leaf_ridge_tree_supervision_dataset(
    docs: Sequence[_CountDoc],
    *,
    split: str,
    target_scale: float,
    n_regimes: int,
    use_endpoints: bool,
    leaf_query_rate: float,
    seed: int,
) -> SupervisionDataset:
    rows: List[DenseSupervisionExample] = []
    rubric = "Predict the leaf-local changepoint count from sampled leaf features."
    for doc_index, doc in enumerate(docs):
        n_leaf = int(len(doc.leaf_features))
        if n_leaf <= 0:
            continue
        q_leaf = leaf_sample_count(n_leaf, rate=float(leaf_query_rate))
        if q_leaf <= 0:
            continue
        rng = random.Random(int(seed) + 7_919 * int(doc_index + 1))
        if q_leaf >= n_leaf:
            leaf_idxs = list(range(n_leaf))
        else:
            leaf_idxs = sorted(int(i) for i in rng.sample(range(n_leaf), k=int(q_leaf)))
        unit_propensity = float(min(1.0, float(q_leaf) / float(max(1, n_leaf))))
        doc_id = f"{split}_doc_{doc_index}"
        for leaf_idx in leaf_idxs:
            rows.append(
                DenseSupervisionExample(
                    example_id=f"{doc_id}:leaf_{int(leaf_idx)}:tree_ridge",
                    features=_leaf_core_feature_vector(
                        doc.leaf_features[int(leaf_idx)],
                        n_regimes=int(n_regimes),
                        use_endpoints=bool(use_endpoints),
                    ).astype(float).tolist(),
                    scalar_target=float(doc.leaf_counts[int(leaf_idx)]) / float(target_scale),
                    original_text=f"markov_changepoint_ops_count::{doc_id}",
                    rubric=rubric,
                    response="sampled_leaf_tree_ridge_candidate",
                    response_id=f"{doc_id}:leaf_{int(leaf_idx)}:tree_ridge",
                    unit_kind=ObservationUnitKind.LEAF,
                    reference_score=float(doc.leaf_counts[int(leaf_idx)]) / float(target_scale),
                    source_doc_id=doc_id,
                    truth_label_source="oracle",
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=float(unit_propensity),
                        label_propensity=1.0,
                        joint_propensity=float(unit_propensity),
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name="uniform_leaf_query_budget",
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=True,
                        metadata={
                            "split": str(split),
                            "leaf_index": int(leaf_idx),
                            "leaf_query_rate": float(leaf_query_rate),
                        },
                    ),
                    metadata={
                        "dgp": "markov_changepoint_ops_count",
                        "input_view": "sampled_leaf_core_features",
                        "uses_tree_merges": True,
                        "leaf_index": int(leaf_idx),
                        "leaf_query_rate": float(leaf_query_rate),
                        "target_scale": float(target_scale),
                    },
                )
            )
    return build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="markov_ops_count",
        supervision_signal_name="substructure_level_target",
        response_signal_name="changepoint_count",
        law_type="c1",
        split=str(split),
        response_signal_min=0.0,
        response_signal_max=1.0,
        metadata={
            "dgp": "markov_changepoint_ops_count",
            "input_view": "sampled_leaf_core_features",
            "uses_tree_merges": True,
            "leaf_query_rate": float(leaf_query_rate),
            "target_scale": float(target_scale),
        },
    )


def _leaf_endpoint_table_supervision_dataset(
    docs: Sequence[_CountDoc],
    *,
    split: str,
    target_scale: float,
    n_regimes: int,
    use_endpoints: bool,
    leaf_query_rate: float,
    seed: int,
) -> SupervisionDataset:
    rows: List[DenseSupervisionExample] = []
    rubric = (
        "Predict the leaf-local changepoint count from a simple local signature: "
        "leaf endpoints and leaf length."
    )
    for doc_index, doc in enumerate(docs):
        n_leaf = int(len(doc.leaf_features))
        if n_leaf <= 0:
            continue
        q_leaf = leaf_sample_count(n_leaf, rate=float(leaf_query_rate))
        if q_leaf <= 0:
            continue
        rng = random.Random(int(seed) + 11_003 * int(doc_index + 1))
        if q_leaf >= n_leaf:
            leaf_idxs = list(range(n_leaf))
        else:
            leaf_idxs = sorted(int(i) for i in rng.sample(range(n_leaf), k=int(q_leaf)))
        unit_propensity = float(min(1.0, float(q_leaf) / float(max(1, n_leaf))))
        doc_id = f"{split}_doc_{doc_index}"
        for leaf_idx in leaf_idxs:
            rows.append(
                DenseSupervisionExample(
                    example_id=f"{doc_id}:leaf_{int(leaf_idx)}:endpoint_table",
                    features=_leaf_endpoint_table_feature_vector(
                        doc.leaf_features[int(leaf_idx)],
                        n_regimes=int(n_regimes),
                        use_endpoints=bool(use_endpoints),
                    )
                    .astype(float)
                    .tolist(),
                    scalar_target=float(doc.leaf_counts[int(leaf_idx)]) / float(target_scale),
                    original_text=f"markov_changepoint_ops_count::{doc_id}",
                    rubric=rubric,
                    response="sampled_leaf_endpoint_table_candidate",
                    response_id=f"{doc_id}:leaf_{int(leaf_idx)}:endpoint_table",
                    unit_kind=ObservationUnitKind.LEAF,
                    reference_score=float(doc.leaf_counts[int(leaf_idx)]) / float(target_scale),
                    source_doc_id=doc_id,
                    truth_label_source="oracle",
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=float(unit_propensity),
                        label_propensity=1.0,
                        joint_propensity=float(unit_propensity),
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name="uniform_leaf_query_budget",
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=True,
                        metadata={
                            "split": str(split),
                            "leaf_index": int(leaf_idx),
                            "leaf_query_rate": float(leaf_query_rate),
                        },
                    ),
                    metadata={
                        "dgp": "markov_changepoint_ops_count",
                        "input_view": "sampled_leaf_endpoints_length",
                        "uses_tree_merges": True,
                        "leaf_index": int(leaf_idx),
                        "leaf_query_rate": float(leaf_query_rate),
                        "target_scale": float(target_scale),
                    },
                )
            )
    return build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="markov_ops_count",
        supervision_signal_name="substructure_level_target",
        response_signal_name="changepoint_count",
        law_type="c1",
        split=str(split),
        response_signal_min=0.0,
        response_signal_max=1.0,
        metadata={
            "dgp": "markov_changepoint_ops_count",
            "input_view": "sampled_leaf_endpoints_length",
            "uses_tree_merges": True,
            "n_regimes": int(n_regimes),
            "use_endpoints": bool(use_endpoints),
            "leaf_query_rate": float(leaf_query_rate),
            "target_scale": float(target_scale),
        },
    )


def _sampled_leaf_pool_doc(
    doc: _CountDoc,
    *,
    leaf_budget: int,
    rng: random.Random,
) -> _SampledLeafPoolDoc:
    leaf = list(doc.leaf_features)
    if not leaf:
        raise ValueError("sampled leaf pool requires at least one leaf per doc")
    k = int(max(1, min(int(leaf_budget), len(leaf))))
    if k >= len(leaf):
        idxs = list(range(len(leaf)))
    else:
        idxs = sorted(int(i) for i in rng.sample(range(len(leaf)), k=k))
    feats = torch.stack([leaf[i] for i in idxs], dim=0).to(dtype=torch.float32, device="cpu")
    mean = feats.mean(dim=0)
    std = feats.std(dim=0, unbiased=False)
    sampled_tokens = int(sum(_feature_leaf_token_count(leaf[i]) for i in idxs))
    summary = torch.cat(
        [
            mean,
            std,
            torch.tensor([float(k), float(sampled_tokens)], dtype=torch.float32),
        ],
        dim=0,
    )
    return _SampledLeafPoolDoc(
        n_tokens=int(doc.n_tokens),
        total_leaves=int(len(leaf)),
        feature_vector=summary.detach().cpu().numpy().astype(np.float64, copy=False),
        root_count=float(doc.root_count),
        sampled_leaves=int(k),
        sampled_tokens=int(sampled_tokens),
        sampled_token_fraction=(
            float(sampled_tokens) / float(max(1, int(doc.n_tokens)))
        ),
    )


def _prepare_sampled_leaf_pool_docs(
    docs: Sequence[_CountDoc],
    *,
    leaf_budget: int,
    seed: int,
) -> Tuple[_SampledLeafPoolDoc, ...]:
    out: List[_SampledLeafPoolDoc] = []
    budget = int(leaf_budget)
    for index, doc in enumerate(docs):
        rng = random.Random(int(seed) + 7_919 * int(index + 1))
        out.append(
            _sampled_leaf_pool_doc(
                doc,
                leaf_budget=int(budget),
                rng=rng,
            )
        )
    return tuple(out)


def _sampled_leaf_pool_supervision_dataset(
    docs: Sequence[_SampledLeafPoolDoc],
    *,
    split: str,
    target_scale: float,
    leaf_budget: int,
) -> SupervisionDataset:
    rows: List[DenseSupervisionExample] = []
    rubric = (
        "Predict the full-document changepoint count from a pooled representation of randomly sampled leaves."
    )
    for index, doc in enumerate(docs):
        doc_id = f"{split}_doc_{index}"
        rows.append(
            DenseSupervisionExample(
                example_id=doc_id,
                features=np.asarray(doc.feature_vector, dtype=np.float64).astype(float).tolist(),
                scalar_target=float(doc.root_count),
                original_text=f"markov_changepoint_ops_count::{doc_id}",
                rubric=rubric,
                response="sampled_leaf_pool_candidate",
                response_id=f"{doc_id}:sampled_leaf_pool_candidate",
                reference_score=float(doc.root_count),
                source_doc_id=doc_id,
                truth_label_source="oracle",
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=1.0,
                    label_propensity=1.0,
                    sampling_scheme="sampled_leaf_pool_uniform",
                    policy_name="uniform_random_without_replacement",
                    unit_kind=ObservationUnitKind.DOCUMENT,
                    supports_ipw_estimation=False,
                ),
                metadata={
                    "dgp": "markov_changepoint_ops_count",
                    "input_view": "sampled_leaf_pool_uniform",
                    "uses_tree_merges": False,
                    "sample_leaf_budget": int(leaf_budget),
                    "sampled_leaves": int(doc.sampled_leaves),
                    "sampled_tokens": int(doc.sampled_tokens),
                    "sampled_token_fraction": float(doc.sampled_token_fraction),
                    "n_tokens": int(doc.n_tokens),
                },
            )
        )
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="markov_ops_count",
        supervision_signal_name="sampled_leaf_pool_target",
        response_signal_name="changepoint_count",
        law_type="document_level_target",
        split=str(split),
        response_signal_min=0.0,
        response_signal_max=float(target_scale),
        metadata={
            "dgp": "markov_changepoint_ops_count",
            "input_view": "sampled_leaf_pool_uniform",
            "uses_tree_merges": False,
            "sample_leaf_budget": int(leaf_budget),
        },
    )


def _sampled_leaf_pool_feature_matrix(docs: Sequence[_SampledLeafPoolDoc]) -> np.ndarray:
    if not docs:
        raise ValueError("sampled leaf pool feature matrix requires at least one document")
    return np.stack(
        [np.asarray(doc.feature_vector, dtype=np.float64) for doc in docs],
        axis=0,
    ).astype(np.float64, copy=False)


def _eval_root_only_predictions(
    preds: Sequence[float],
    root_counts: Sequence[float],
    *,
    n_docs: int,
    tau: float,
    condition_ids: Sequence[str] | None = None,
) -> SketchMetrics:
    if int(n_docs) <= 0:
        return _zero_sketch_metrics(n_docs=0)
    y = np.asarray([float(value) for value in root_counts], dtype=np.float64)
    pred = np.asarray([float(value) for value in preds], dtype=np.float64)
    if y.shape != pred.shape:
        raise ValueError("root-only predictions must align with targets")
    abs_err = np.abs(pred - y)
    tau_v = float(tau)
    condition_metrics = _condition_error_diagnostics(abs_err, condition_ids)
    return SketchMetrics(
        root_mae=float(np.mean(abs_err)),
        root_median_abs_error=float(np.median(abs_err)),
        root_p95_abs_error=float(np.percentile(abs_err, 95.0)),
        schedule_spread_mean=0.0,
        schedule_spread_p95=0.0,
        leaf_mae=float(np.mean(abs_err)),
        leaf_violation_rate=float(np.mean((abs_err > tau_v).astype(np.float64))),
        c2_idempotence_mae=0.0,
        c2_r1_mae=0.0,
        c2_r2_mae=0.0,
        c2_r4_mae=0.0,
        resummary_root_drift_r1=0.0,
        resummary_root_drift_r2=0.0,
        resummary_root_drift_r4=0.0,
        merge_mae=0.0,
        merge_violation_rate=0.0,
        n_docs=int(n_docs),
        condition_root_mae=dict(condition_metrics["condition_root_mae"]),
        condition_root_n_docs=dict(condition_metrics["condition_root_n_docs"]),
        condition_root_macro_mae=float(condition_metrics["condition_root_macro_mae"]),
        condition_root_worst_mae=float(condition_metrics["condition_root_worst_mae"]),
    )


def _sampled_leaf_pool_observation_summary(
    docs: Sequence[_SampledLeafPoolDoc],
) -> Dict[str, float]:
    if not docs:
        return {
            "sampled_leaves_mean": 0.0,
            "sampled_tokens_mean": 0.0,
            "sampled_token_fraction_mean": 0.0,
            "total_leaves_mean": 0.0,
            "total_tokens_mean": 0.0,
        }
    sampled_leaves = np.asarray([float(doc.sampled_leaves) for doc in docs], dtype=np.float64)
    sampled_tokens = np.asarray([float(doc.sampled_tokens) for doc in docs], dtype=np.float64)
    sampled_fraction = np.asarray(
        [float(doc.sampled_token_fraction) for doc in docs], dtype=np.float64
    )
    total_leaves = np.asarray([float(doc.total_leaves) for doc in docs], dtype=np.float64)
    total_tokens = np.asarray([float(doc.n_tokens) for doc in docs], dtype=np.float64)
    return {
        "sampled_leaves_mean": float(np.mean(sampled_leaves)),
        "sampled_tokens_mean": float(np.mean(sampled_tokens)),
        "sampled_token_fraction_mean": float(np.mean(sampled_fraction)),
        "total_leaves_mean": float(np.mean(total_leaves)),
        "total_tokens_mean": float(np.mean(total_tokens)),
    }


def _sample_internal_audit_indices(
    n_internal: int,
    *,
    k: int,
    strategy: C3AuditStrategyName,
    merge_sizes: Sequence[int],
    include_root: bool,
    rng: random.Random,
) -> Optional[set[int]]:
    """
    Sample realized internal nodes for C3 labels.

    Returns:
      - `None`: use all internal nodes
      - empty set: use none
      - non-empty set: selected indices
    """

    n = int(max(0, n_internal))
    q = int(max(0, k))
    if n <= 0 or q <= 0:
        return set()
    if q >= n:
        return None

    strat = str(strategy)
    if strat not in VALID_C3_AUDIT_STRATEGIES:
        raise ValueError(
            f"unsupported c3_audit_strategy: {strategy!r}; expected one of {VALID_C3_AUDIT_STRATEGIES}"
        )

    selected: set[int] = set()
    if include_root and n > 0:
        # In `_prepare_count_docs`, the root merge is appended last.
        selected.add(int(n - 1))
    if len(selected) >= q:
        return set(list(selected)[:q])

    available = [i for i in range(n) if i not in selected]
    need = int(q - len(selected))
    if need <= 0:
        return selected

    if strat == "uniform":
        selected.update(rng.sample(available, k=need))
        return selected

    if strat == "top_span":
        ranked = sorted(
            available,
            key=lambda i: (int(merge_sizes[i]) if i < len(merge_sizes) else 0, int(i)),
            reverse=True,
        )
        selected.update(ranked[:need])
        return selected

    if strat == "hybrid_top_span":
        ranked = sorted(
            available,
            key=lambda i: (int(merge_sizes[i]) if i < len(merge_sizes) else 0, int(i)),
            reverse=True,
        )
        top_need = min(len(ranked), max(1, need // 2))
        selected.update(ranked[:top_need])
        remaining_need = int(need - top_need)
        if remaining_need > 0:
            rem = [i for i in available if i not in selected]
            if remaining_need >= len(rem):
                selected.update(rem)
            else:
                selected.update(rng.sample(rem, k=remaining_need))
        return selected

    # Weighted without replacement (Efraimidis-Spirakis): larger spans are more likely.
    keys: List[Tuple[float, int]] = []
    for i in available:
        w = float(merge_sizes[i]) if i < len(merge_sizes) else 1.0
        w = max(1e-8, w)
        u = max(float(rng.random()), 1e-12)
        keys.append((u ** (1.0 / w), int(i)))
    keys.sort(reverse=True)
    selected.update(i for _k, i in keys[:need])
    return selected



class AdditiveCountSketch(nn.Module):
    """
    Structured sketch family for the Markov changepoint-count target.

    State layout:
      - normalized count scalar (R^1)
      - first regime one-hot (R^{n_regimes})
      - last regime one-hot (R^{n_regimes})

    Merge law (associative by construction):
      c(parent) = c(left) + c(right) + 1[last(left) != first(right)] / target_scale

    This family is intentionally "OPS-shaped": we *separate* endpoint transport (exact) from the
    (learned) scalar count, so that under full labels it can approach the exact ceiling.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        target_scale: float,
        n_regimes: int,
        use_endpoints: bool,
        c2_learned_resummary: bool = False,
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.n_regimes = int(n_regimes)
        self.use_endpoints = bool(use_endpoints)
        if self.n_regimes <= 0:
            raise ValueError("n_regimes must be positive")
        if self.target_scale <= 0:
            raise ValueError("target_scale must be positive")

        endpoint_dim = 2 * int(self.n_regimes) if self.use_endpoints else 0
        encoder_in = int(feature_dim) - int(endpoint_dim)
        if encoder_in <= 0:
            raise ValueError("feature_dim too small for endpoint stripping")

        # Linear leaf encoder -> scalar normalized count.
        # This matches the default DGP where the changepoint count is a linear functional of transition counts.
        self.encoder = nn.Linear(int(encoder_in), 1, bias=True)

        # Learned re-summarization cycle (Lean L3 / C2).  When active,
        # decode_summary and encode_summary are learned functions so the
        # round-trip is non-trivial.  C2 loss then tests whether re-summarizing
        # preserves oracle value — a functional property, not an architectural
        # constraint.
        self._c2_learned_resummary = bool(c2_learned_resummary)
        if self._c2_learned_resummary:
            sdim = self.summary_dim
            self.c2_decoder = nn.Sequential(
                nn.Linear(sdim, sdim),
                nn.Tanh(),
                nn.Linear(sdim, sdim),
            )
            self.c2_re_encoder = nn.Sequential(
                nn.Linear(sdim, sdim),
                nn.Tanh(),
                nn.Linear(sdim, sdim),
            )

    @property
    def summary_dim(self) -> int:
        return 1 + 2 * int(self.n_regimes)

    def _split_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(self.n_regimes)
        if state.shape[-1] != 1 + 2 * n:
            raise ValueError("unexpected state dimension for AdditiveCountSketch")
        count = state[..., 0]
        first = state[..., 1 : 1 + n]
        last = state[..., 1 + n : 1 + 2 * n]
        return count, first, last

    def encode_leaf(self, features: torch.Tensor) -> torch.Tensor:
        n = int(self.n_regimes)
        if self.use_endpoints:
            if features.shape[-1] < 2 * n:
                raise ValueError("leaf features missing endpoint slots")
            first = features[..., :n]
            last = features[..., n : 2 * n]
            core = features[..., 2 * n :]
        else:
            first = torch.zeros(
                (*features.shape[:-1], n), device=features.device, dtype=features.dtype
            )
            last = torch.zeros(
                (*features.shape[:-1], n), device=features.device, dtype=features.dtype
            )
            core = features

        count_norm = self.encoder(core).squeeze(-1)
        return torch.cat([count_norm.unsqueeze(-1), first, last], dim=-1)

    def predict_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        count, _first, _last = self._split_state(state)
        return count

    def predict_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict_norm_from_state(state) * float(self.target_scale)

    def decode_summary(self, state: torch.Tensor) -> torch.Tensor:
        if self._c2_learned_resummary:
            return self.c2_decoder(state)
        return state

    def encode_summary(self, summary: torch.Tensor) -> torch.Tensor:
        if self._c2_learned_resummary:
            return self.c2_re_encoder(summary)
        x = summary
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.shape[-1] == int(self.summary_dim):
            return x
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        zeros = torch.zeros(
            (*x.shape[:-1], 2 * int(self.n_regimes)),
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, zeros], dim=-1)

    def _merge_states(
        self,
        states: Sequence[torch.Tensor],
        *,
        schedule: ScheduleName,
        collect_merge_states: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if len(states) == 0:
            raise ValueError("need at least one state")
        if len(states) == 1:
            return states[0], []

        def _merge(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            left_c, left_first, left_last = self._split_state(left)
            right_c, right_first, right_last = self._split_state(right)
            if self.use_endpoints:
                same = torch.sum(left_last * right_first, dim=-1)  # one-hot dot
                join = 1.0 - same
                join_term = join / float(self.target_scale)
            else:
                join_term = torch.zeros_like(left_c)
            merged_c = left_c + right_c + join_term
            return torch.cat([merged_c.unsqueeze(-1), left_first, right_last], dim=-1)

        merged_states: List[torch.Tensor] = []
        if str(schedule) == "balanced":
            cur = list(states)
            while len(cur) > 1:
                nxt: List[torch.Tensor] = []
                i = 0
                while i < len(cur):
                    if i + 1 >= len(cur):
                        nxt.append(cur[i])
                        i += 1
                        continue
                    merged = _merge(cur[i], cur[i + 1])
                    if collect_merge_states:
                        merged_states.append(merged)
                    nxt.append(merged)
                    i += 2
                cur = nxt
            return cur[0], merged_states

        if str(schedule) in ("left_to_right", "right_to_left"):
            if str(schedule) == "left_to_right":
                acc = states[0]
                for st in states[1:]:
                    acc = _merge(acc, st)
                    if collect_merge_states:
                        merged_states.append(acc)
                return acc, merged_states

            acc = states[-1]
            for st in reversed(states[:-1]):
                acc = _merge(st, acc)
                if collect_merge_states:
                    merged_states.append(acc)
            return acc, merged_states

        raise ValueError(f"unsupported schedule: {schedule!r}")

    def forward_doc(
        self,
        leaf_features: Sequence[torch.Tensor],
        leaf_counts: Sequence[float],
        merge_counts_balanced: Sequence[float],
        *,
        schedule: ScheduleName,
        collect_leaf: bool,
        collect_c3: bool,
        collect_c2: bool,
        leaf_audit_indices: Optional[set[int]] = None,
        c3_audit_indices: Optional[set[int]] = None,
    ) -> Dict[str, torch.Tensor | float]:
        if len(leaf_features) == 0:
            raise ValueError("leaf_features must be non-empty")
        if len(leaf_features) != len(leaf_counts):
            raise ValueError("leaf_features and leaf_counts must align")

        states = [self.encode_leaf(x) for x in leaf_features]
        root_state, merge_states = self._merge_states(
            states,
            schedule=schedule,
            collect_merge_states=(collect_c3 or collect_c2) and str(schedule) == "balanced",
        )
        pred_norm = self.predict_norm_from_state(root_state)
        out: Dict[str, torch.Tensor | float] = {
            "pred_norm": pred_norm,
            "pred_count": self.predict_count_from_state(root_state),
        }

        if collect_leaf:
            leaf_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            leaf_count = 0
            for idx, (st, truth) in enumerate(zip(states, leaf_counts)):
                if leaf_audit_indices is not None and idx not in leaf_audit_indices:
                    continue
                pred_leaf = self.predict_norm_from_state(st)
                true_leaf = torch.tensor(
                    float(truth) / float(self.target_scale),
                    device=pred_norm.device,
                    dtype=pred_leaf.dtype,
                )
                leaf_loss = leaf_loss + F.mse_loss(pred_leaf, true_leaf, reduction="mean")
                leaf_count += 1
            out["leaf_loss"] = leaf_loss / float(max(1, leaf_count))
            out["leaf_count"] = float(leaf_count)
        else:
            out["leaf_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["leaf_count"] = 0.0

        if collect_c3:
            if str(schedule) != "balanced":
                raise ValueError("collect_c3 is only supported for balanced schedule")
            c3_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c3_count = 0
            for idx, st in enumerate(merge_states):
                if c3_audit_indices is not None and idx not in c3_audit_indices:
                    continue
                if idx >= len(merge_counts_balanced):
                    continue
                pred = self.predict_norm_from_state(st)
                truth = torch.tensor(
                    float(merge_counts_balanced[idx]) / float(self.target_scale),
                    device=pred_norm.device,
                    dtype=pred.dtype,
                )
                c3_loss = c3_loss + F.mse_loss(pred, truth, reduction="mean")
                c3_count += 1
            out["c3_loss"] = c3_loss / float(max(1, c3_count))
            out["c3_count"] = float(c3_count)
        else:
            out["c3_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["c3_count"] = 0.0
        if collect_c2:
            c2_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c2_count = 0
            candidate_states = list(states)
            candidate_states.extend(list(merge_states))
            if len(candidate_states) == 0:
                candidate_states = [root_state]
            for st in candidate_states:
                # Prediction-level C2 (Lean L3): re-summarizing preserves
                # oracle value.  predict(state) ≈ predict(encode(decode(state))).
                original_pred = self.predict_norm_from_state(st)
                resummary_state = self.encode_summary(self.decode_summary(st))
                resummary_pred = self.predict_norm_from_state(resummary_state)
                c2_loss = c2_loss + F.mse_loss(resummary_pred, original_pred.detach(), reduction="mean")
                c2_count += 1
            out["c2_loss"] = c2_loss / float(max(1, c2_count))
            out["c2_count"] = float(c2_count)
        else:
            out["c2_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["c2_count"] = 0.0
        return out


def _training_geometry(
    docs: Sequence[_CountDoc],
    *,
    policy: AuditPolicyName,
    fixed_nodes: int,
    fraction: float,
    scale: float,
    leaf_query_rate: float,
    include_root_query: bool,
) -> TrainingGeometry:
    if len(docs) == 0:
        return TrainingGeometry(
            mean_tokens=0.0,
            mean_leaves=0.0,
            mean_internal_nodes=0.0,
            mean_leaf_labels=0.0,
            mean_internal_labels=0.0,
            mean_queries_per_doc=0.0,
            root_queries_total=0,
            leaf_labels_total=0,
            internal_labels_total=0,
            total_queries_estimate=0,
        )

    toks: List[float] = []
    leaves: List[float] = []
    internals: List[float] = []
    leaf_labels: List[float] = []
    internal_labels: List[float] = []
    leaf_labels_total = 0
    internal_labels_total = 0

    for doc in docs:
        n_tok = int(doc.n_tokens)
        n_leaves = int(len(doc.leaf_features))
        n_internal = int(max(0, n_leaves - 1))
        q_leaf = leaf_sample_count(n_leaves, rate=float(leaf_query_rate))
        q_internal = audit_sample_count(
            n_internal,
            policy=str(policy),
            fixed_nodes=int(fixed_nodes),
            fraction=float(fraction),
            scale=float(scale),
        )
        toks.append(float(n_tok))
        leaves.append(float(n_leaves))
        internals.append(float(n_internal))
        leaf_labels.append(float(q_leaf))
        internal_labels.append(float(q_internal))
        leaf_labels_total += int(q_leaf)
        internal_labels_total += int(q_internal)

    n_docs = int(len(docs))
    root_queries_total = int(n_docs if include_root_query else 0)
    total = int(root_queries_total + leaf_labels_total + internal_labels_total)
    mean_leaf = float(np.mean(np.asarray(leaves, dtype=np.float64)))
    mean_leaf_labels = float(np.mean(np.asarray(leaf_labels, dtype=np.float64)))
    mean_internal = float(np.mean(np.asarray(internals, dtype=np.float64)))
    mean_internal_labels = float(np.mean(np.asarray(internal_labels, dtype=np.float64)))
    return TrainingGeometry(
        mean_tokens=float(np.mean(np.asarray(toks, dtype=np.float64))),
        mean_leaves=float(mean_leaf),
        mean_internal_nodes=float(mean_internal),
        mean_leaf_labels=float(mean_leaf_labels),
        mean_internal_labels=float(mean_internal_labels),
        mean_queries_per_doc=float(
            mean_leaf_labels + mean_internal_labels + (1.0 if include_root_query else 0.0)
        ),
        root_queries_total=int(root_queries_total),
        leaf_labels_total=int(leaf_labels_total),
        internal_labels_total=int(internal_labels_total),
        total_queries_estimate=int(total),
    )


def _resummary_summary_sequence(
    model: AdditiveCountSketch,
    state: torch.Tensor,
    *,
    depths: Sequence[int],
) -> Dict[int, torch.Tensor]:
    wanted = sorted({int(d) for d in depths if int(d) >= 1})
    if not wanted:
        return {}
    cur = model.decode_summary(state)
    out: Dict[int, torch.Tensor] = {}
    max_depth = int(max(wanted))
    for step in range(1, max_depth + 1):
        cur = model.decode_summary(model.encode_summary(cur))
        if step in wanted:
            out[int(step)] = cur
    return out


def _predict_count_from_summary(
    model: AdditiveCountSketch,
    summary: torch.Tensor,
) -> torch.Tensor:
    return model.predict_count_from_state(model.encode_summary(summary))




def _fit_additive_leaf_encoder_closed_form(
    model: AdditiveCountSketch,
    train_docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    c3_audit_strategy: C3AuditStrategyName,
    c3_include_root: bool,
    leaf_query_rate: float,
    seed: int,
) -> TrainFitDiagnostics:
    """
    Closed-form fit for the additive sketch's linear leaf encoder.

    Uses whatever oracle labels the run has "paid for":
      - leaf labels (C1) under `leaf_query_rate`
      - internal-node labels (C3) under `audit_policy`/`audit_fraction`/`c3_audit_strategy`

    Each internal-node label yields a linear equation in the leaf encoder weights because:
      1) internal counts are sums of leaf counts + join indicators, and
      2) join indicators are computable from endpoints when `feature_mode='full'`.
    """

    if len(train_docs) == 0:
        return TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=0,
            selection_metric_curve=(0.0,),
            selection_mode="closed_form_solution",
            selection_split="config",
            selection_metric_name="closed_form_train_mse",
            selection_metric_value=0.0,
            best_epoch=0,
        )

    rng = random.Random(int(seed))
    n = int(model.n_regimes)
    X_rows: List[np.ndarray] = []  # each row is [core_features..., bias_coeff]
    y_rows: List[float] = []

    def _leaf_core(feat: torch.Tensor) -> np.ndarray:
        if model.use_endpoints:
            return feat[2 * n :].detach().cpu().numpy().astype(np.float64)
        return feat.detach().cpu().numpy().astype(np.float64)

    def _balanced_merge_leaf_ranges(n_leaves: int) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = [(i, i + 1) for i in range(int(n_leaves))]
        merges: List[Tuple[int, int]] = []
        while len(spans) > 1:
            nxt: List[Tuple[int, int]] = []
            i = 0
            while i < len(spans):
                if i + 1 >= len(spans):
                    nxt.append(spans[i])
                    i += 1
                    continue
                merged = (int(spans[i][0]), int(spans[i + 1][1]))
                merges.append(merged)
                nxt.append(merged)
                i += 2
            spans = nxt
        return merges

    for doc in train_docs:
        n_leaf = int(len(doc.leaf_features))
        if n_leaf <= 0:
            continue

        # ----------------------------
        # Leaf labels (C1)
        # ----------------------------
        q_leaf = leaf_sample_count(n_leaf, rate=float(leaf_query_rate))
        if q_leaf > 0:
            if q_leaf >= n_leaf:
                leaf_idxs = list(range(n_leaf))
            else:
                leaf_idxs = rng.sample(range(n_leaf), k=int(q_leaf))
            for idx in leaf_idxs:
                core = _leaf_core(doc.leaf_features[int(idx)])
                X_rows.append(np.concatenate([core, np.asarray([1.0], dtype=np.float64)], axis=0))
                y_rows.append(float(doc.leaf_counts[int(idx)]) / float(model.target_scale))

        # ----------------------------
        # Internal labels (C3), balanced-merge order
        # ----------------------------
        n_internal = int(max(0, n_leaf - 1))
        if n_internal <= 0 or len(doc.merge_counts_balanced) == 0:
            continue
        ranges = _balanced_merge_leaf_ranges(n_leaf)
        if len(ranges) != n_internal:
            raise ValueError("internal merge range reconstruction failed")

        q_internal = audit_sample_count(
            n_internal,
            policy=str(audit_policy),
            fixed_nodes=int(audit_fixed_nodes),
            fraction=float(audit_fraction),
            scale=float(audit_scale),
        )
        internal_idxs = _sample_internal_audit_indices(
            n_internal,
            k=int(q_internal),
            strategy=str(c3_audit_strategy),
            merge_sizes=doc.merge_sizes_balanced,
            include_root=bool(c3_include_root),
            rng=rng,
        )
        if internal_idxs is None:
            internal_iter = range(n_internal)
        else:
            internal_iter = list(internal_idxs)

        # Precompute per-boundary join indicators (between adjacent leaves).
        if model.use_endpoints:
            first_ids: List[int] = []
            last_ids: List[int] = []
            for feat in doc.leaf_features:
                first_ids.append(int(torch.argmax(feat[:n]).item()))
                last_ids.append(int(torch.argmax(feat[n : 2 * n]).item()))
            join_flags = [0 if last_ids[i] == first_ids[i + 1] else 1 for i in range(n_leaf - 1)]
        else:
            join_flags = [0 for _ in range(max(0, n_leaf - 1))]

        core_cache = [_leaf_core(f) for f in doc.leaf_features]
        for idx in internal_iter:
            k = int(idx)
            if k < 0 or k >= n_internal:
                continue
            a, b = ranges[k]
            if not (0 <= a < b <= n_leaf):
                continue
            span_leaves = int(b - a)
            sum_core = np.sum(np.stack(core_cache[a:b], axis=0), axis=0)
            join_sum = int(sum(join_flags[a : b - 1])) if span_leaves >= 2 else 0
            y_internal_norm = float(doc.merge_counts_balanced[k]) / float(model.target_scale)
            y_target = float(y_internal_norm) - float(join_sum) / float(model.target_scale)

            X_rows.append(
                np.concatenate(
                    [sum_core, np.asarray([float(span_leaves)], dtype=np.float64)], axis=0
                )
            )
            y_rows.append(float(y_target))

    if not X_rows:
        return TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="closed_form_solution",
            selection_split="config",
            selection_metric_name="closed_form_train_mse",
            selection_metric_value=0.0,
            best_epoch=0,
        )

    X = np.stack(X_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.float64)
    beta, *_rest = np.linalg.lstsq(X, y, rcond=None)
    w = beta[:-1]
    b = beta[-1]

    with torch.no_grad():
        model.encoder.weight.copy_(
            torch.tensor(w.reshape(1, -1), device=device, dtype=torch.float32)
        )
        model.encoder.bias.copy_(torch.tensor([float(b)], device=device, dtype=torch.float32))

    preds = X @ beta
    mse = float(np.mean((preds - y) ** 2))
    return TrainFitDiagnostics(
        train_loss_final=float(mse),
        train_loss_curve=(float(mse),),
        epochs_completed=1,
        selection_metric_curve=(float(mse),),
        selection_mode="closed_form_solution",
        selection_split="config",
        selection_metric_name="closed_form_train_mse",
        selection_metric_value=float(mse),
        best_epoch=0,
    )


@torch.no_grad()
def _eval_learned_model(
    model: AdditiveCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    tau: float,
    condition_ids: Sequence[str] | None = None,
) -> SketchMetrics:
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    model.eval()
    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []
    c2_r1_abs: List[float] = []
    c2_r2_abs: List[float] = []
    c2_r4_abs: List[float] = []
    c2_bottleneck_recon: List[float] = []
    root_drift_r1: List[float] = []
    root_drift_r2: List[float] = []
    root_drift_r4: List[float] = []

    for doc in docs:
        leaf_feats = _to_device(doc.leaf_features, device=device)
        states = [model.encode_leaf(x) for x in leaf_feats]

        # Leaf C1.
        for st, truth in zip(states, doc.leaf_counts):
            pred = float(model.predict_count_from_state(st).detach().cpu())
            leaf_abs.append(abs(pred - float(truth)))

        # Merge C3 (balanced schedule, all internal nodes).
        _root_state, merge_states = model._merge_states(
            states,
            schedule="balanced",
            collect_merge_states=True,
        )
        for pred_st, truth in zip(merge_states, doc.merge_counts_balanced):
            pred = float(model.predict_count_from_state(pred_st).detach().cpu())
            merge_abs.append(abs(pred - float(truth)))

        c2_states = list(states)
        c2_states.extend(list(merge_states))
        if not c2_states:
            c2_states = [_root_state]
        for st in c2_states:
            base_summary = model.decode_summary(st)
            base_value = float(_predict_count_from_summary(model, base_summary).detach().cpu())
            replay = _resummary_summary_sequence(model, st, depths=(1, 2, 4))
            c2_r1_abs.append(
                abs(
                    float(_predict_count_from_summary(model, replay[1]).detach().cpu()) - base_value
                )
            )
            c2_r2_abs.append(
                abs(
                    float(_predict_count_from_summary(model, replay[2]).detach().cpu()) - base_value
                )
            )
            c2_r4_abs.append(
                abs(
                    float(_predict_count_from_summary(model, replay[4]).detach().cpu()) - base_value
                )
            )
            # Re-summarization reconstruction MSE: state-space round-trip fidelity.
            if model._c2_learned_resummary:
                recon = model.encode_summary(model.decode_summary(st))
                c2_bottleneck_recon.append(
                    float(F.mse_loss(recon, st, reduction="mean").detach().cpu())
                )

        # Root distortion + schedule spread.
        roots: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            root_state, _ = model._merge_states(states, schedule=sched, collect_merge_states=False)
            roots[str(sched)] = float(model.predict_count_from_state(root_state).detach().cpu())
        pred_root = roots["balanced"]
        root_abs.append(abs(pred_root - float(doc.root_count)))
        spreads.append(max(roots.values()) - min(roots.values()))
        root_replay = _resummary_summary_sequence(model, _root_state, depths=(1, 2, 4))
        root_base = float(
            _predict_count_from_summary(model, model.decode_summary(_root_state)).detach().cpu()
        )
        root_drift_r1.append(
            abs(
                float(_predict_count_from_summary(model, root_replay[1]).detach().cpu()) - root_base
            )
        )
        root_drift_r2.append(
            abs(
                float(_predict_count_from_summary(model, root_replay[2]).detach().cpu()) - root_base
            )
        )
        root_drift_r4.append(
            abs(
                float(_predict_count_from_summary(model, root_replay[4]).detach().cpu()) - root_base
            )
        )

    tau = float(tau)
    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    c2_r1_arr = np.asarray(c2_r1_abs, dtype=np.float64)
    c2_r2_arr = np.asarray(c2_r2_abs, dtype=np.float64)
    c2_r4_arr = np.asarray(c2_r4_abs, dtype=np.float64)
    root_drift_r1_arr = np.asarray(root_drift_r1, dtype=np.float64)
    root_drift_r2_arr = np.asarray(root_drift_r2, dtype=np.float64)
    root_drift_r4_arr = np.asarray(root_drift_r4, dtype=np.float64)
    condition_metrics = _condition_error_diagnostics(root_abs_arr, condition_ids)

    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=(
            float(np.mean((leaf_abs_arr > tau).astype(np.float64))) if leaf_abs_arr.size else 0.0
        ),
        c2_idempotence_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r1_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r2_mae=float(np.mean(c2_r2_arr)) if c2_r2_arr.size else 0.0,
        c2_r4_mae=float(np.mean(c2_r4_arr)) if c2_r4_arr.size else 0.0,
        resummary_root_drift_r1=(
            float(np.mean(root_drift_r1_arr)) if root_drift_r1_arr.size else 0.0
        ),
        resummary_root_drift_r2=(
            float(np.mean(root_drift_r2_arr)) if root_drift_r2_arr.size else 0.0
        ),
        resummary_root_drift_r4=(
            float(np.mean(root_drift_r4_arr)) if root_drift_r4_arr.size else 0.0
        ),
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=(
            float(np.mean((merge_abs_arr > tau).astype(np.float64))) if merge_abs_arr.size else 0.0
        ),
        n_docs=int(len(docs)),
        c2_bottleneck_reconstruction_mse=(
            float(np.mean(c2_bottleneck_recon)) if c2_bottleneck_recon else 0.0
        ),
        condition_root_mae=dict(condition_metrics["condition_root_mae"]),
        condition_root_n_docs=dict(condition_metrics["condition_root_n_docs"]),
        condition_root_macro_mae=float(condition_metrics["condition_root_macro_mae"]),
        condition_root_worst_mae=float(condition_metrics["condition_root_worst_mae"]),
    )


@torch.no_grad()
def _compute_dpo_preference_gap(
    model: AdditiveCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    beta: float = 1.0,
    n_pairs: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute DPO preference gap: tree-summary path vs full-document path.

    Validates the Lean chain: local laws hold -> zero distortion -> DPO equivalence.
    (Lean: dpo_gap_bounded in OPT/PreferenceBounds.lean)

    For random pairs (i, j) of documents:
      - "original" DPO loss uses true changepoint counts
      - "summary" DPO loss uses model's root predictions from tree reduction
      - gap = mean |L_original - L_summary|

    The Lean bound says: gap <= 2 * beta * L_pol * E[distortion].
    """
    if len(docs) < 2:
        return {"dpo_preference_gap": 0.0, "dpo_mean_distortion": 0.0, "dpo_n_pairs": 0}

    model.eval()
    rng = np.random.RandomState(seed)

    true_counts: List[float] = []
    pred_counts: List[float] = []
    for doc in docs:
        true_counts.append(float(doc.root_count))
        leaf_feats = _to_device(doc.leaf_features, device=device)
        states = [model.encode_leaf(x) for x in leaf_feats]
        root_state, _ = model._merge_states(states, schedule="balanced", collect_merge_states=False)
        pred_counts.append(float(model.predict_count_from_state(root_state).detach().cpu()))

    true_arr = np.asarray(true_counts, dtype=np.float64)
    pred_arr = np.asarray(pred_counts, dtype=np.float64)
    distortions = np.abs(pred_arr - true_arr)

    # Sample random pairs.
    actual_pairs = min(int(n_pairs), len(docs) * (len(docs) - 1) // 2)
    idx_i = rng.randint(0, len(docs), size=actual_pairs)
    idx_j = rng.randint(0, len(docs) - 1, size=actual_pairs)
    idx_j[idx_j >= idx_i] += 1  # avoid self-pairs

    dpo_gaps: List[float] = []
    for ii, jj in zip(idx_i, idx_j):
        # Original DPO: preference based on true counts.
        true_margin = float(beta) * (true_arr[ii] - true_arr[jj])
        loss_original = float(np.log1p(np.exp(-true_margin)))
        # Summary DPO: preference based on predicted counts.
        pred_margin = float(beta) * (pred_arr[ii] - pred_arr[jj])
        loss_summary = float(np.log1p(np.exp(-pred_margin)))
        dpo_gaps.append(abs(loss_original - loss_summary))

    mean_gap = float(np.mean(dpo_gaps)) if dpo_gaps else 0.0
    mean_distortion = float(np.mean(distortions))

    return {
        "dpo_preference_gap": mean_gap,
        "dpo_mean_distortion": mean_distortion,
        "dpo_n_pairs": int(actual_pairs),
        "dpo_beta": float(beta),
        "dpo_lipschitz_bound": 2.0 * float(beta) * mean_distortion,
    }


@torch.no_grad()
def _eval_objective_terms(
    model: AdditiveCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    leaf_weight: float,
    c2_weight: float,
    c3_weight: float,
    root_weight: float,
    schedule_consistency_weight: float,
    include_root_query: bool,
) -> ObjectiveMetrics:
    if len(docs) == 0:
        return ObjectiveMetrics(
            optimization_total_loss=0.0,
            optimization_root_loss=0.0,
            optimization_leaf_loss=0.0,
            optimization_c2_loss=0.0,
            optimization_merge_loss=0.0,
            optimization_schedule_consistency_loss=0.0,
            raw_total_loss=0.0,
            raw_root_loss=0.0,
            raw_leaf_loss=0.0,
            raw_c2_loss=0.0,
            raw_merge_loss=0.0,
            raw_schedule_consistency_loss=0.0,
            n_docs=0,
        )

    model.eval()
    optimization_total_terms: List[float] = []
    optimization_root_terms: List[float] = []
    optimization_leaf_terms: List[float] = []
    optimization_c2_terms: List[float] = []
    optimization_merge_terms: List[float] = []
    optimization_consistency_terms: List[float] = []
    raw_total_terms: List[float] = []
    raw_root_terms: List[float] = []
    raw_leaf_terms: List[float] = []
    raw_c2_terms: List[float] = []
    raw_merge_terms: List[float] = []
    raw_consistency_terms: List[float] = []

    for doc in docs:
        leaf_feats = _to_device(doc.leaf_features, device=device)
        out = model.forward_doc(
            leaf_feats,
            doc.leaf_counts,
            doc.merge_counts_balanced,
            schedule="balanced",
            collect_leaf=True,
            collect_c3=True,
            collect_c2=True,
            leaf_audit_indices=None,
            c3_audit_indices=None,
        )
        pred_norm = out["pred_norm"]
        leaf_loss_tensor = out["leaf_loss"]
        c2_loss_tensor = out["c2_loss"]
        c3_loss_tensor = out["c3_loss"]
        if not isinstance(pred_norm, torch.Tensor):
            raise TypeError("expected tensor pred_norm from forward_doc")
        if (
            not isinstance(leaf_loss_tensor, torch.Tensor)
            or not isinstance(c2_loss_tensor, torch.Tensor)
            or not isinstance(c3_loss_tensor, torch.Tensor)
        ):
            raise TypeError("expected tensor leaf/c2/c3 losses from forward_doc")

        true_norm = torch.tensor(
            float(doc.root_count) / float(getattr(model, "target_scale", 1.0)),
            device=device,
            dtype=pred_norm.dtype,
        )
        raw_root_term = (
            float(F.mse_loss(pred_norm, true_norm, reduction="mean").detach().cpu())
            if bool(include_root_query)
            else 0.0
        )
        raw_leaf_term = float(leaf_loss_tensor.detach().cpu())
        raw_c2_term = float(c2_loss_tensor.detach().cpu())
        raw_merge_term = float(c3_loss_tensor.detach().cpu())

        if float(schedule_consistency_weight) > 0.0 and len(leaf_feats) > 1:
            states_sched = [model.encode_leaf(x) for x in leaf_feats]
            sched_preds = []
            for sched in VALID_SCHEDULES:
                root_state_sched, _ = model._merge_states(
                    states_sched,
                    schedule=sched,
                    collect_merge_states=False,
                )
                sched_preds.append(model.predict_norm_from_state(root_state_sched))
            pred_stack = torch.stack(sched_preds, dim=0)
            consistency_raw = torch.mean((pred_stack - torch.mean(pred_stack)) ** 2)
            raw_consistency_term = float(consistency_raw.detach().cpu())
        else:
            raw_consistency_term = 0.0

        optimization_root_term = float(root_weight) * raw_root_term
        optimization_leaf_term = float(leaf_weight) * raw_leaf_term
        optimization_c2_term = float(c2_weight) * raw_c2_term
        optimization_merge_term = float(c3_weight) * raw_merge_term
        optimization_consistency_term = float(schedule_consistency_weight) * raw_consistency_term

        optimization_total_terms.append(
            float(
                optimization_root_term
                + optimization_leaf_term
                + optimization_c2_term
                + optimization_merge_term
                + optimization_consistency_term
            )
        )
        optimization_root_terms.append(float(optimization_root_term))
        optimization_leaf_terms.append(float(optimization_leaf_term))
        optimization_c2_terms.append(float(optimization_c2_term))
        optimization_merge_terms.append(float(optimization_merge_term))
        optimization_consistency_terms.append(float(optimization_consistency_term))
        raw_total_terms.append(
            float(
                raw_root_term + raw_leaf_term + raw_c2_term + raw_merge_term + raw_consistency_term
            )
        )
        raw_root_terms.append(float(raw_root_term))
        raw_leaf_terms.append(float(raw_leaf_term))
        raw_c2_terms.append(float(raw_c2_term))
        raw_merge_terms.append(float(raw_merge_term))
        raw_consistency_terms.append(float(raw_consistency_term))

    optimization_total_arr = np.asarray(optimization_total_terms, dtype=np.float64)
    optimization_root_arr = np.asarray(optimization_root_terms, dtype=np.float64)
    optimization_leaf_arr = np.asarray(optimization_leaf_terms, dtype=np.float64)
    optimization_c2_arr = np.asarray(optimization_c2_terms, dtype=np.float64)
    optimization_merge_arr = np.asarray(optimization_merge_terms, dtype=np.float64)
    optimization_consistency_arr = np.asarray(optimization_consistency_terms, dtype=np.float64)
    raw_total_arr = np.asarray(raw_total_terms, dtype=np.float64)
    raw_root_arr = np.asarray(raw_root_terms, dtype=np.float64)
    raw_leaf_arr = np.asarray(raw_leaf_terms, dtype=np.float64)
    raw_c2_arr = np.asarray(raw_c2_terms, dtype=np.float64)
    raw_merge_arr = np.asarray(raw_merge_terms, dtype=np.float64)
    raw_consistency_arr = np.asarray(raw_consistency_terms, dtype=np.float64)
    return ObjectiveMetrics(
        optimization_total_loss=float(np.mean(optimization_total_arr)),
        optimization_root_loss=float(np.mean(optimization_root_arr)),
        optimization_leaf_loss=float(np.mean(optimization_leaf_arr)),
        optimization_c2_loss=float(np.mean(optimization_c2_arr)),
        optimization_merge_loss=float(np.mean(optimization_merge_arr)),
        optimization_schedule_consistency_loss=float(np.mean(optimization_consistency_arr)),
        raw_total_loss=float(np.mean(raw_total_arr)),
        raw_root_loss=float(np.mean(raw_root_arr)),
        raw_leaf_loss=float(np.mean(raw_leaf_arr)),
        raw_c2_loss=float(np.mean(raw_c2_arr)),
        raw_merge_loss=float(np.mean(raw_merge_arr)),
        raw_schedule_consistency_loss=float(np.mean(raw_consistency_arr)),
        n_docs=int(len(docs)),
    )



# ---------------------------------------------------------------------------
# Classical baselines (extracted to markov_baselines.py for maintainability)
# ---------------------------------------------------------------------------
from treepo._research.ctreepo.sim.core.markov_baselines import (  # noqa: E402
    _rf_doc_features,
    _eval_rf_root_baseline,
    _fit_doc_level_baseline,
    _fit_doc_level_ridge_baseline,
    _fit_doc_token_ngram_ridge_baseline,
    _fit_doc_sequence_ctreepo_baseline,
    _fit_doc_sequence_baseline,
    _fit_doc_transformer_baseline,
    _fit_sampled_leaf_pool_ridge_baseline,
    _fit_leaf_ridge_tree_baseline,
    _balanced_additive_merge_counts,
    _eval_leaf_local_additive_predictor,
    _fit_leaf_knn_tree_baseline,
    _fit_leaf_endpoint_table_tree_baseline,
    _fit_leaf_dt_tree_baseline,
    _fit_leaf_rf_tree_baseline,
    _fit_sampled_leaf_pool_rf_baseline,
)



def _clip_norm_target(t: float) -> float:
    v = float(t)
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def _override_state_with_oracle_count(
    model: AdditiveCountSketch,
    state: torch.Tensor,
    *,
    target_count: float,
    override_mode: GuidanceOverrideModeName,
) -> torch.Tensor:
    target_scale = float(getattr(model, "target_scale", 1.0))
    target_norm = _clip_norm_target(float(target_count) / float(max(1e-12, target_scale)))

    if isinstance(model, AdditiveCountSketch):
        _count, first, last = model._split_state(state)
        guided_count = torch.full_like(_count, float(target_norm))
        return torch.cat([guided_count.unsqueeze(-1), first, last], dim=-1)

    raise TypeError(f"unsupported model type for guidance override: {type(model)!r}")


def _merge_balanced_with_guidance(
    model: AdditiveCountSketch,
    states: Sequence[torch.Tensor],
    *,
    merge_truth_counts_balanced: Sequence[float],
    guided_internal_indices: set[int],
    guidance_override_mode: GuidanceOverrideModeName,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
    if len(states) == 0:
        raise ValueError("need at least one state")
    if len(states) == 1:
        return states[0], [], []

    merge_states: List[torch.Tensor] = []
    merge_pred_counts: List[float] = []
    cur = list(states)
    merge_idx = 0
    while len(cur) > 1:
        nxt: List[torch.Tensor] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                i += 1
                continue

            left = cur[i]
            right = cur[i + 1]
            if isinstance(model, AdditiveCountSketch):
                left_c, left_first, left_last = model._split_state(left)
                right_c, right_first, right_last = model._split_state(right)
                if bool(model.use_endpoints):
                    same = torch.sum(left_last * right_first, dim=-1)
                    join = 1.0 - same
                    join_term = join / float(model.target_scale)
                else:
                    join_term = torch.zeros_like(left_c)
                merged_c = left_c + right_c + join_term
                merged = torch.cat([merged_c.unsqueeze(-1), left_first, right_last], dim=-1)
            else:
                raise TypeError(f"unsupported model type: {type(model)!r}")

            if merge_idx in guided_internal_indices and merge_idx < len(
                merge_truth_counts_balanced
            ):
                merged = _override_state_with_oracle_count(
                    model,
                    merged,
                    target_count=float(merge_truth_counts_balanced[merge_idx]),
                    override_mode=str(guidance_override_mode),
                )
                merge_pred_count = float(merge_truth_counts_balanced[merge_idx])
            else:
                merge_pred_count = float(model.predict_count_from_state(merged).detach().cpu())
            merge_states.append(merged)
            merge_pred_counts.append(float(merge_pred_count))
            nxt.append(merged)
            merge_idx += 1
            i += 2
        cur = nxt
    return cur[0], merge_states, merge_pred_counts


def _sample_guided_internal_indices(
    n_internal: int,
    *,
    q: float,
    include_root: bool,
    rng: random.Random,
) -> set[int]:
    n = int(max(0, n_internal))
    if n <= 0:
        return set()
    qq = float(max(0.0, min(1.0, float(q))))
    if qq <= 0.0:
        return set()

    if include_root:
        pool = list(range(n))
    else:
        pool = list(range(max(0, n - 1)))
    if not pool:
        return set()
    k = int(math.ceil(qq * float(n)))
    k = int(max(0, min(int(len(pool)), k)))
    if k <= 0:
        return set()
    if k >= len(pool):
        return set(pool)
    return set(int(i) for i in rng.sample(pool, k=k))


@torch.no_grad()
def _eval_guided_model_curve(
    model: AdditiveCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    tau: float,
    guidance_qs: Sequence[float],
    guidance_trials: int,
    guidance_include_root: bool,
    guidance_override_mode: GuidanceOverrideModeName,
    guidance_seed: int,
) -> Dict[str, object]:
    qs = [float(max(0.0, min(1.0, q))) for q in guidance_qs]
    qs = sorted({float(q) for q in qs})
    trials = int(max(0, guidance_trials))
    if len(docs) == 0 or trials <= 0 or not qs:
        return {
            "present": False,
            "include_root": bool(guidance_include_root),
            "trials": int(trials),
            "points": [],
        }

    model.eval()
    points: List[Dict[str, float | int]] = []
    for q_idx, q in enumerate(qs):
        root_abs: List[float] = []
        leaf_abs: List[float] = []
        merge_abs: List[float] = []
        guided_nodes: List[float] = []
        effective_qs: List[float] = []
        for trial in range(trials):
            rng = random.Random(int(guidance_seed) + 7919 * int(q_idx + 1) + 97 * int(trial + 1))
            for doc in docs:
                leaf_feats = _to_device(doc.leaf_features, device=device)
                states = [model.encode_leaf(x) for x in leaf_feats]
                n_internal = int(max(0, len(states) - 1))
                guided_idx = _sample_guided_internal_indices(
                    n_internal,
                    q=float(q),
                    include_root=bool(guidance_include_root),
                    rng=rng,
                )
                root_state, merge_states, merge_pred_counts = _merge_balanced_with_guidance(
                    model,
                    states,
                    merge_truth_counts_balanced=doc.merge_counts_balanced,
                    guided_internal_indices=guided_idx,
                    guidance_override_mode=str(guidance_override_mode),
                )
                root_idx = int(n_internal - 1)
                if (
                    root_idx >= 0
                    and root_idx in guided_idx
                    and root_idx < len(doc.merge_counts_balanced)
                ):
                    pred_root = float(doc.merge_counts_balanced[root_idx])
                else:
                    pred_root = float(model.predict_count_from_state(root_state).detach().cpu())
                root_abs.append(abs(pred_root - float(doc.root_count)))

                for st, truth in zip(states, doc.leaf_counts):
                    pred_leaf = float(model.predict_count_from_state(st).detach().cpu())
                    leaf_abs.append(abs(pred_leaf - float(truth)))

                for idx, pred_st in enumerate(merge_states):
                    if idx >= len(doc.merge_counts_balanced):
                        break
                    if idx < len(merge_pred_counts):
                        pred_merge = float(merge_pred_counts[idx])
                    else:
                        pred_merge = float(model.predict_count_from_state(pred_st).detach().cpu())
                    merge_abs.append(abs(pred_merge - float(doc.merge_counts_balanced[idx])))

                guided_nodes.append(float(len(guided_idx)))
                effective_qs.append(float(len(guided_idx)) / float(max(1, n_internal)))

        root_arr = np.asarray(root_abs, dtype=np.float64)
        leaf_arr = np.asarray(leaf_abs, dtype=np.float64)
        merge_arr = np.asarray(merge_abs, dtype=np.float64)
        guided_arr = np.asarray(guided_nodes, dtype=np.float64)
        effq_arr = np.asarray(effective_qs, dtype=np.float64)
        tau_v = float(tau)
        points.append(
            {
                "q": float(q),
                "root_mae": float(np.mean(root_arr)) if root_arr.size else 0.0,
                "root_median_abs_error": float(np.median(root_arr)) if root_arr.size else 0.0,
                "root_p95_abs_error": (
                    float(np.percentile(root_arr, 95.0)) if root_arr.size else 0.0
                ),
                "leaf_mae": float(np.mean(leaf_arr)) if leaf_arr.size else 0.0,
                "leaf_violation_rate": (
                    float(np.mean((leaf_arr > tau_v).astype(np.float64))) if leaf_arr.size else 0.0
                ),
                "merge_mae": float(np.mean(merge_arr)) if merge_arr.size else 0.0,
                "merge_violation_rate": (
                    float(np.mean((merge_arr > tau_v).astype(np.float64)))
                    if merge_arr.size
                    else 0.0
                ),
                "guided_internal_nodes_mean": (
                    float(np.mean(guided_arr)) if guided_arr.size else 0.0
                ),
                "guided_internal_nodes_p95": (
                    float(np.percentile(guided_arr, 95.0)) if guided_arr.size else 0.0
                ),
                "effective_q_mean": float(np.mean(effq_arr)) if effq_arr.size else 0.0,
                "n_eval_docs": int(len(docs) * trials),
            }
        )

    return {
        "present": True,
        "include_root": bool(guidance_include_root),
        "trials": int(trials),
        "points": points,
    }


def _audit_estimator_diagnostics(
    values: Sequence[float],
    preds: Sequence[float],
    inclusion_probs: Sequence[float],
    *,
    trials: int,
    seed: int,
) -> EstimatorDiagnostics:
    y = np.asarray(values, dtype=np.float64)
    m = np.asarray(preds, dtype=np.float64)
    pi = np.asarray(inclusion_probs, dtype=np.float64)
    if y.size == 0:
        return EstimatorDiagnostics(
            true_mean=0.0,
            naive_bias=0.0,
            ipw_bias=0.0,
            dsl_bias=0.0,
            ipw_var=0.0,
            dsl_var=0.0,
        )
    if y.shape != m.shape or y.shape != pi.shape:
        raise ValueError("values, preds, inclusion_probs must have the same shape")
    if np.any(pi <= 0.0) or np.any(pi > 1.0):
        raise ValueError("inclusion_probs must lie in (0,1]")

    rng = np.random.default_rng(int(seed))
    N = int(y.size)
    true_mean = float(np.mean(y))
    naive: List[float] = []
    ipw: List[float] = []
    dsl: List[float] = []

    for _ in range(int(max(1, trials))):
        inc = rng.random(N) < pi
        if not np.any(inc):
            continue
        y_s = y[inc]
        pi_s = pi[inc]
        m_s = m[inc]

        naive.append(float(np.mean(y_s)))
        ipw.append(float(np.sum(y_s / pi_s) / float(N)))
        dsl.append(float(np.mean(m) + np.sum((y_s - m_s) / pi_s) / float(N)))

    def _bias(xs: Sequence[float]) -> float:
        return float(np.mean(np.asarray(xs, dtype=np.float64)) - true_mean) if xs else 0.0

    def _var(xs: Sequence[float]) -> float:
        arr = np.asarray(xs, dtype=np.float64)
        return float(np.var(arr)) if arr.size else 0.0

    return EstimatorDiagnostics(
        true_mean=float(true_mean),
        naive_bias=float(_bias(naive)),
        ipw_bias=float(_bias(ipw)),
        dsl_bias=float(_bias(dsl)),
        ipw_var=float(_var(ipw)),
        dsl_var=float(_var(dsl)),
    )


def run_markov_changepoint_ops_count_experiment(
    config: OPSCountConfig,
    *,
    data_bundle: Optional[MarkovOPSDataBundle] = None,
) -> OPSCountSummary:
    _generator_profile_key(str(config.generator_profile))
    if int(config.n_regimes) < 1:
        raise ValueError("n_regimes must be >= 1")
    if config.min_distinct_regimes_per_doc is not None and int(config.min_distinct_regimes_per_doc) < 1:
        raise ValueError("min_distinct_regimes_per_doc must be >= 1 when set")
    if config.max_distinct_regimes_per_doc is not None and int(config.max_distinct_regimes_per_doc) < 1:
        raise ValueError("max_distinct_regimes_per_doc must be >= 1 when set")
    if (
        config.min_distinct_regimes_per_doc is not None
        and config.max_distinct_regimes_per_doc is not None
        and int(config.min_distinct_regimes_per_doc) > int(config.max_distinct_regimes_per_doc)
    ):
        raise ValueError(
            "min_distinct_regimes_per_doc must be <= max_distinct_regimes_per_doc"
        )
    if (
        config.min_distinct_regimes_per_doc is not None
        and int(config.min_distinct_regimes_per_doc) > int(config.min_segments)
    ):
        raise ValueError(
            "min_distinct_regimes_per_doc must be <= min_segments for every document to remain feasible"
        )
    if (
        config.max_distinct_regimes_per_doc is not None
        and int(config.max_distinct_regimes_per_doc) > int(config.max_segments)
    ):
        raise ValueError(
            "max_distinct_regimes_per_doc must be <= max_segments for every document to remain feasible"
        )
    if (
        config.max_distinct_regimes_per_doc is not None
        and int(config.max_distinct_regimes_per_doc) > int(config.n_regimes)
    ):
        raise ValueError("max_distinct_regimes_per_doc must be <= n_regimes")
    if (
        config.min_distinct_regimes_per_doc is not None
        and int(config.min_segments) > 1
        and int(config.min_distinct_regimes_per_doc) < 2
    ):
        raise ValueError(
            "min_distinct_regimes_per_doc must be >= 2 when documents can contain multiple segments"
        )
    if int(config.fixed_leaf_tokens) <= 0:
        raise ValueError("fixed_leaf_tokens must be positive")
    if int(config.train_docs) < 0 or int(config.val_docs) < 0 or int(config.test_docs) < 0:
        raise ValueError("train_docs/val_docs/test_docs must be non-negative")
    if str(config.audit_policy) not in VALID_AUDIT_POLICIES:
        raise ValueError(
            f"audit_policy={config.audit_policy!r} unsupported; expected one of {VALID_AUDIT_POLICIES}"
        )
    if float(config.audit_fraction) < 0.0:
        raise ValueError("audit_fraction must be non-negative")
    if float(config.audit_scale) <= 0.0:
        raise ValueError("audit_scale must be positive")
    if int(config.audit_fixed_nodes) < 0:
        raise ValueError("audit_fixed_nodes must be non-negative")
    if str(config.c3_audit_strategy) not in VALID_C3_AUDIT_STRATEGIES:
        raise ValueError(
            "c3_audit_strategy="
            f"{config.c3_audit_strategy!r} unsupported; expected one of {VALID_C3_AUDIT_STRATEGIES}"
        )
    if float(config.leaf_query_rate) < 0.0 or float(config.leaf_query_rate) > 1.0:
        raise ValueError("leaf_query_rate must lie in [0,1]")
    if float(config.leaf_weight) < 0.0:
        raise ValueError("leaf_weight must be non-negative")
    if float(config.c2_weight) < 0.0:
        raise ValueError("c2_weight must be non-negative")
    if float(config.c3_weight) < 0.0:
        raise ValueError("c3_weight must be non-negative")
    if config.local_law_weight is not None and float(config.local_law_weight) < 0.0:
        raise ValueError("local_law_weight must be non-negative")
    if config.local_law_weight is not None and float(config.local_law_weight) > 1.0:
        raise ValueError(
            "local_law_weight must lie in [0,1] under the normalized lambda parameterization"
        )
    if config.task_objective_weight is not None and float(config.task_objective_weight) < 0.0:
        raise ValueError("task_objective_weight must be non-negative")
    if float(config.c1_relative_weight) < 0.0:
        raise ValueError("c1_relative_weight must be non-negative")
    if float(config.c2_relative_weight) < 0.0:
        raise ValueError("c2_relative_weight must be non-negative")
    if float(config.c3_relative_weight) < 0.0:
        raise ValueError("c3_relative_weight must be non-negative")
    if config.local_law_weight is not None:
        if config.task_objective_weight is not None:
            raise ValueError("local_law_weight is mutually exclusive with task_objective_weight")
        if float(config.leaf_weight) > 0.0 or float(config.c2_weight) > 0.0 or float(config.c3_weight) > 0.0:
            raise ValueError("local_law_weight is mutually exclusive with explicit law weights")
        if (
            float(config.local_law_weight) > 0.0
            and (
                not math.isclose(float(config.c1_relative_weight), 1.0)
                or not math.isclose(float(config.c2_relative_weight), 1.0)
                or not math.isclose(float(config.c3_relative_weight), 1.0)
            )
        ):
            raise ValueError("lambda mode uses equal active-law weights")
    if float(config.root_weight) < 0.0:
        raise ValueError("root_weight must be non-negative")
    if float(config.schedule_consistency_weight) < 0.0:
        raise ValueError("schedule_consistency_weight must be non-negative")
    if float(config.doc_sequence_train_fraction) < 0.0 or float(config.doc_sequence_train_fraction) > 1.0:
        raise ValueError("doc_sequence_train_fraction must lie in [0,1]")
    if str(config.law_package or "").strip():
        package = str(config.law_package).strip().lower()
        if package not in VALID_LAW_PACKAGES:
            raise ValueError(
                f"law_package={package!r} unsupported; expected one of {VALID_LAW_PACKAGES}"
            )
    if str(config.exact_family or "").strip():
        exact_family = str(config.exact_family).strip()
        if exact_family not in VALID_EXACT_FAMILIES:
            raise ValueError(
                f"exact_family={exact_family!r} unsupported; expected one of {VALID_EXACT_FAMILIES}"
            )
    if str(config.model_family) not in VALID_MODEL_FAMILIES:
        raise ValueError(
            f"model_family={config.model_family!r} unsupported; expected one of {VALID_MODEL_FAMILIES}"
        )
    if str(config.doc_sequence_objective) not in VALID_DOC_SEQUENCE_OBJECTIVES:
        raise ValueError(
            "doc_sequence_objective="
            f"{config.doc_sequence_objective!r} unsupported; expected one of "
            f"{VALID_DOC_SEQUENCE_OBJECTIVES}"
        )
    if str(config.doc_sequence_fno_pooling) not in VALID_DOC_SEQUENCE_FNO_POOLING:
        raise ValueError(
            "doc_sequence_fno_pooling="
            f"{config.doc_sequence_fno_pooling!r} unsupported; expected one of "
            f"{VALID_DOC_SEQUENCE_FNO_POOLING}"
        )
    try:
        normalized_local_law_objective_mode = normalize_local_law_objective_mode(
            str(config.local_law_objective_mode)
        )
    except ValueError as exc:
        raise ValueError(
            "local_law_objective_mode="
            f"{config.local_law_objective_mode!r} unsupported; expected one of "
            f"{VALID_LOCAL_LAW_OBJECTIVE_MODES}"
        ) from exc
    if (
        str(config.tree_document_loss_normalization_mode)
        not in VALID_TREE_DOCUMENT_LOSS_NORMALIZATION_MODES
    ):
        raise ValueError(
            "tree_document_loss_normalization_mode="
            f"{config.tree_document_loss_normalization_mode!r} unsupported; expected one of "
            f"{VALID_TREE_DOCUMENT_LOSS_NORMALIZATION_MODES}"
        )
    if str(config.tree_root_supervision_kind) not in VALID_TREE_ROOT_SUPERVISION_KINDS:
        raise ValueError(
            "tree_root_supervision_kind="
            f"{config.tree_root_supervision_kind!r} unsupported; expected one of "
            f"{VALID_TREE_ROOT_SUPERVISION_KINDS}"
        )
    if str(config.tree_checkpoint_metric) not in VALID_TREE_CHECKPOINT_METRICS:
        raise ValueError(
            "tree_checkpoint_metric="
            f"{config.tree_checkpoint_metric!r} unsupported; expected one of "
            f"{VALID_TREE_CHECKPOINT_METRICS}"
        )
    if str(config.tree_stage1_checkpoint_metric) not in VALID_TREE_CHECKPOINT_METRICS:
        raise ValueError(
            "tree_stage1_checkpoint_metric="
            f"{config.tree_stage1_checkpoint_metric!r} unsupported; expected one of "
            f"{VALID_TREE_CHECKPOINT_METRICS}"
        )
    if str(config.tree_stage1_eval_mode) not in VALID_TREE_STAGE1_EVAL_MODES:
        raise ValueError(
            "tree_stage1_eval_mode="
            f"{config.tree_stage1_eval_mode!r} unsupported; expected one of "
            f"{VALID_TREE_STAGE1_EVAL_MODES}"
        )
    if str(config.tree_batch_pack_mode) not in VALID_TREE_BATCH_PACK_MODES:
        raise ValueError(
            "tree_batch_pack_mode="
            f"{config.tree_batch_pack_mode!r} unsupported; expected one of "
            f"{VALID_TREE_BATCH_PACK_MODES}"
        )
    if float(config.tree_join_bit_weight) < 0.0:
        raise ValueError("tree_join_bit_weight must be non-negative")
    if str(config.tree_training_schedule) not in VALID_TREE_TRAINING_SCHEDULES:
        raise ValueError(
            "tree_training_schedule="
            f"{config.tree_training_schedule!r} unsupported; expected one of "
            f"{VALID_TREE_TRAINING_SCHEDULES}"
        )
    if int(config.tree_stage1_epochs) < 0:
        raise ValueError("tree_stage1_epochs must be non-negative")
    if int(config.tree_stage2_epochs) < 0:
        raise ValueError("tree_stage2_epochs must be non-negative")
    if int(config.tree_stage1_screen_doc_limit) < 0:
        raise ValueError("tree_stage1_screen_doc_limit must be non-negative")
    if int(config.tree_stage1_final_exact_doc_limit) < 0:
        raise ValueError("tree_stage1_final_exact_doc_limit must be non-negative")
    if int(getattr(config, "exact_metric_selection_doc_limit", 0)) < 0:
        raise ValueError("exact_metric_selection_doc_limit must be non-negative")
    if int(getattr(config, "exact_metric_selection_interval", 1)) <= 0:
        raise ValueError("exact_metric_selection_interval must be positive")
    if int(getattr(config, "exact_metric_final_doc_limit", 0)) < 0:
        raise ValueError("exact_metric_final_doc_limit must be non-negative")
    if int(getattr(config, "tree_exact_eval_max_docs", 0)) < 0:
        raise ValueError("tree_exact_eval_max_docs must be non-negative")
    if int(getattr(config, "tree_posttrain_train_doc_limit", 0)) < 0:
        raise ValueError("tree_posttrain_train_doc_limit must be non-negative")
    if str(getattr(config, "posttrain_diagnostics_mode", "")) not in VALID_POSTTRAIN_DIAGNOSTICS_MODES:
        raise ValueError(
            "posttrain_diagnostics_mode="
            f"{getattr(config, 'posttrain_diagnostics_mode', '')!r} unsupported; expected one of "
            f"{VALID_POSTTRAIN_DIAGNOSTICS_MODES}"
        )
    if int(config.tree_batch_token_budget) < 0:
        raise ValueError("tree_batch_token_budget must be non-negative")
    if int(config.tree_batch_node_budget) < 0:
        raise ValueError("tree_batch_node_budget must be non-negative")
    if float(config.tree_batch_structural_pad_limit) < 0.0:
        raise ValueError("tree_batch_structural_pad_limit must be non-negative")
    if int(config.tree_batch_auto_queue_min_docs) < 0:
        raise ValueError("tree_batch_auto_queue_min_docs must be non-negative")
    if not 0.0 <= float(config.tree_batch_auto_queue_min_fill_ratio) <= 1.0:
        raise ValueError(
            "tree_batch_auto_queue_min_fill_ratio must be between 0 and 1"
        )
    if int(config.tree_eval_workers_per_mig) < 0:
        raise ValueError("tree_eval_workers_per_mig must be non-negative")
    if str(config.gpu_runtime_data_mode) not in VALID_GPU_RUNTIME_DATA_MODES:
        raise ValueError(
            "gpu_runtime_data_mode="
            f"{config.gpu_runtime_data_mode!r} unsupported; expected one of "
            f"{tuple(sorted(VALID_GPU_RUNTIME_DATA_MODES))}"
        )
    if str(config.gpu_runtime_bucket_mode) not in VALID_GPU_RUNTIME_BUCKET_MODES:
        raise ValueError(
            "gpu_runtime_bucket_mode="
            f"{config.gpu_runtime_bucket_mode!r} unsupported; expected one of "
            f"{tuple(sorted(VALID_GPU_RUNTIME_BUCKET_MODES))}"
        )
    if int(config.gpu_runtime_workers_per_mig) <= 0:
        raise ValueError("gpu_runtime_workers_per_mig must be positive")
    if int(config.gpu_runtime_capacity_workers_per_mig) <= 0:
        raise ValueError("gpu_runtime_capacity_workers_per_mig must be positive")
    if str(getattr(config, "tree_stage1_artifact_dir", "")).strip():
        Path(str(config.tree_stage1_artifact_dir)).expanduser()
    if str(getattr(config, "tree_stage1_artifact_root", "")).strip():
        Path(str(config.tree_stage1_artifact_root)).expanduser()
    if str(getattr(config, "prepared_data_root", "")).strip():
        Path(str(config.prepared_data_root)).expanduser()
    if float(getattr(config, "tree_stage1_root_weight", 0.0)) < 0.0:
        raise ValueError("tree_stage1_root_weight must be non-negative")
    if str(config.tree_task_head_mode) not in VALID_TREE_TASK_HEAD_MODES:
        raise ValueError(
            "tree_task_head_mode="
            f"{config.tree_task_head_mode!r} unsupported; expected one of "
            f"{VALID_TREE_TASK_HEAD_MODES}"
        )
    if str(config.tree_theorem_surface_mode) not in VALID_TREE_THEOREM_SURFACE_MODES:
        raise ValueError(
            "tree_theorem_surface_mode="
            f"{config.tree_theorem_surface_mode!r} unsupported; expected one of "
            f"{VALID_TREE_THEOREM_SURFACE_MODES}"
        )
    if str(config.tree_phi_alignment_loss) not in VALID_TREE_PHI_ALIGNMENT_LOSSES:
        raise ValueError(
            "tree_phi_alignment_loss="
            f"{config.tree_phi_alignment_loss!r} unsupported; expected one of "
            f"{VALID_TREE_PHI_ALIGNMENT_LOSSES}"
        )
    if str(getattr(config, "tree_c2_mode", "reconstruction")).strip().lower() not in {
        "reconstruction",
        "fiber",
    }:
        raise ValueError("tree_c2_mode must be 'reconstruction' or 'fiber'")
    adapter_name = str(
        getattr(config, "theorem_feature_adapter", DEFAULT_THEOREM_FEATURE_ADAPTER)
    ).strip() or DEFAULT_THEOREM_FEATURE_ADAPTER
    if adapter_name not in valid_theorem_feature_adapters():
        raise ValueError(
            "theorem_feature_adapter="
            f"{getattr(config, 'theorem_feature_adapter', '')!r} unsupported; expected one of "
            f"{valid_theorem_feature_adapters()}"
        )
    if str(config.tree_theorem_count_head_mode) not in VALID_TREE_THEOREM_COUNT_HEAD_MODES:
        raise ValueError(
            "tree_theorem_count_head_mode="
            f"{config.tree_theorem_count_head_mode!r} unsupported; expected one of "
            f"{VALID_TREE_THEOREM_COUNT_HEAD_MODES}"
        )
    if float(config.tree_theorem_count_ordinal_weight) < 0.0:
        raise ValueError("tree_theorem_count_ordinal_weight must be non-negative")
    if float(config.tree_theorem_count_scalar_aux_weight) < 0.0:
        raise ValueError("tree_theorem_count_scalar_aux_weight must be non-negative")
    if int(config.tree_theorem_feature_dim) <= 0:
        raise ValueError("tree_theorem_feature_dim must be positive")
    if int(config.tree_theorem_feature_hidden_dim) <= 0:
        raise ValueError("tree_theorem_feature_hidden_dim must be positive")
    if int(getattr(config, "tree_theorem_score_dim", 0)) < 0:
        raise ValueError("tree_theorem_score_dim must be non-negative")
    if int(getattr(config, "tree_theorem_fiber_dim", 0)) < 0:
        raise ValueError("tree_theorem_fiber_dim must be non-negative")
    if int(getattr(config, "tree_theorem_aux_dim", 0)) < 0:
        raise ValueError("tree_theorem_aux_dim must be non-negative")
    if (
        int(getattr(config, "tree_theorem_score_dim", 0))
        + int(getattr(config, "tree_theorem_fiber_dim", 0))
        + int(getattr(config, "tree_theorem_aux_dim", 0))
    ) not in {0, int(config.tree_theorem_feature_dim)}:
        raise ValueError(
            "tree_theorem_score_dim + tree_theorem_fiber_dim + tree_theorem_aux_dim "
            "must be zero/unset or equal tree_theorem_feature_dim"
        )
    if float(config.tree_phi_compose_weight) < 0.0:
        raise ValueError("tree_phi_compose_weight must be non-negative")
    if float(config.tree_phi_contrastive_weight) < 0.0:
        raise ValueError("tree_phi_contrastive_weight must be non-negative")
    if (
        getattr(config, "theorem_pair_same_threshold", None) is not None
        and not np.isfinite(float(config.theorem_pair_same_threshold))
    ):
        raise ValueError("theorem_pair_same_threshold must be finite when set")
    if (
        getattr(config, "theorem_pair_diff_threshold", None) is not None
        and not np.isfinite(float(config.theorem_pair_diff_threshold))
    ):
        raise ValueError("theorem_pair_diff_threshold must be finite when set")
    if str(config.tree_summary_spec_root_mode) not in VALID_TREE_SUMMARY_SPEC_ROOT_MODES:
        raise ValueError(
            "tree_summary_spec_root_mode="
            f"{config.tree_summary_spec_root_mode!r} unsupported; expected one of "
            f"{VALID_TREE_SUMMARY_SPEC_ROOT_MODES}"
        )
    if str(config.aligned_sketch_surface) not in VALID_ALIGNED_SKETCH_SURFACES:
        raise ValueError(
            "aligned_sketch_surface="
            f"{config.aligned_sketch_surface!r} unsupported; expected one of "
            f"{VALID_ALIGNED_SKETCH_SURFACES}"
        )
    if str(config.summary_spec_name) not in VALID_SUMMARY_SPEC_NAMES:
        raise ValueError(
            "summary_spec_name="
            f"{config.summary_spec_name!r} unsupported; expected one of "
            f"{VALID_SUMMARY_SPEC_NAMES}"
        )
    if int(config.slot_count) < 0:
        raise ValueError("slot_count must be non-negative")
    if str(config.leaf_supervision_kind) not in VALID_LEAF_SUPERVISION_KINDS:
        raise ValueError(
            "leaf_supervision_kind="
            f"{config.leaf_supervision_kind!r} unsupported; expected one of "
            f"{VALID_LEAF_SUPERVISION_KINDS}"
        )
    if float(config.leaf_label_rate) < 0.0 or float(config.leaf_label_rate) > 1.0:
        raise ValueError("leaf_label_rate must lie in [0,1]")
    theorem_dims = (
        int(config.tree_theorem_count_dim),
        int(config.tree_theorem_first_dim),
        int(config.tree_theorem_last_dim),
    )
    if str(config.summary_spec_name).strip():
        if int(config.slot_count) <= 0:
            raise ValueError("slot_count must be positive when summary_spec_name is set")
        if (
            not any(int(dim) > 0 for dim in theorem_dims)
            and int(config.state_dim) % int(config.slot_count) != 0
        ):
            raise ValueError(
                "state_dim must be divisible by slot_count when summary_spec_name is set"
            )
    if str(config.summary_spec_name).strip() and int(config.slot_count) < 4:
        raise ValueError("slot_count must be at least 4 when summary_spec_name is set")
    if any(dim < 0 for dim in theorem_dims):
        raise ValueError("tree theorem slot dims must be non-negative")
    if any(dim > 0 for dim in theorem_dims):
        if str(config.summary_spec_name).strip() != "markov_count_sketch":
            raise ValueError(
                "tree theorem slot dims require summary_spec_name='markov_count_sketch'"
            )
        if any(dim <= 0 for dim in theorem_dims):
            raise ValueError(
                "tree_theorem_count_dim, tree_theorem_first_dim, and "
                "tree_theorem_last_dim must all be positive when any are set"
            )
        if int(config.slot_count) != 4:
            raise ValueError("explicit theorem slot dims require slot_count == 4")
        if sum(theorem_dims) > int(config.state_dim):
            raise ValueError(
                "sum of explicit theorem slot dims must not exceed state_dim"
            )
    if str(config.tree_local_weighting_mode) not in VALID_TREE_LOCAL_WEIGHTING_MODES:
        raise ValueError(
            "tree_local_weighting_mode="
            f"{config.tree_local_weighting_mode!r} unsupported; expected one of "
            f"{VALID_TREE_LOCAL_WEIGHTING_MODES}"
        )
    if str(config.tree_supervision_source) not in VALID_TREE_SUPERVISION_SOURCES:
        raise ValueError(
            "tree_supervision_source="
            f"{config.tree_supervision_source!r} unsupported; expected one of "
            f"{VALID_TREE_SUPERVISION_SOURCES}"
        )
    if str(config.tree_exact_collapse_mode) not in VALID_TREE_EXACT_COLLAPSE_MODES:
        raise ValueError(
            "tree_exact_collapse_mode="
            f"{config.tree_exact_collapse_mode!r} unsupported; expected one of "
            f"{VALID_TREE_EXACT_COLLAPSE_MODES}"
        )
    if str(config.comparison_mode) not in VALID_COMPARISON_MODES:
        raise ValueError(
            "comparison_mode="
            f"{config.comparison_mode!r} unsupported; expected one of "
            f"{VALID_COMPARISON_MODES}"
        )
    if str(config.internal_supervision_kind) not in VALID_INTERNAL_SUPERVISION_KINDS:
        raise ValueError(
            "internal_supervision_kind="
            f"{config.internal_supervision_kind!r} unsupported; expected one of "
            f"{VALID_INTERNAL_SUPERVISION_KINDS}"
        )
    if float(config.internal_label_rate) < 0.0 or float(config.internal_label_rate) > 1.0:
        raise ValueError("internal_label_rate must lie in [0,1]")
    if (
        str(config.internal_supervision_kind) == "none"
        and float(config.internal_label_rate) > 0.0
    ):
        raise ValueError(
            "internal_label_rate must be zero when internal_supervision_kind='none'"
        )
    if (
        str(config.internal_supervision_kind) != "none"
        and float(config.internal_label_rate) <= 0.0
    ):
        raise ValueError(
            "internal_label_rate must be positive when internal supervision is enabled"
        )
    if int(config.budget_total_calls) < 0:
        raise ValueError("budget_total_calls must be non-negative")
    if float(config.budget_total_calls_per_doc) < 0.0:
        raise ValueError("budget_total_calls_per_doc must be non-negative")
    mass_target_per_doc = float(getattr(config, "mass_target_per_doc", float("nan")))
    if math.isfinite(mass_target_per_doc) and (
        mass_target_per_doc < 0.0 or mass_target_per_doc > 1.0
    ):
        raise ValueError("mass_target_per_doc must lie in [0,1] when provided")
    if float(config.full_doc_budget_share) < 0.0 or float(config.full_doc_budget_share) > 1.0:
        raise ValueError("full_doc_budget_share must lie in [0,1]")
    if str(config.doc_consumption_mode) not in VALID_BUDGETED_DOC_CONSUMPTION_MODES:
        raise ValueError(
            "doc_consumption_mode="
            f"{config.doc_consumption_mode!r} unsupported; expected one of "
            f"{VALID_BUDGETED_DOC_CONSUMPTION_MODES}"
        )
    if str(config.local_split_mode) not in VALID_BUDGETED_LOCAL_SPLIT_MODES:
        raise ValueError(
            "local_split_mode="
            f"{config.local_split_mode!r} unsupported; expected one of "
            f"{VALID_BUDGETED_LOCAL_SPLIT_MODES}"
        )
    if str(config.local_allocation_policy) not in VALID_BUDGETED_ALLOCATION_POLICIES:
        raise ValueError(
            "local_allocation_policy="
            f"{config.local_allocation_policy!r} unsupported; expected one of "
            f"{VALID_BUDGETED_ALLOCATION_POLICIES}"
        )
    for field_name in (
        "tree_leaf_fno_width",
        "tree_leaf_fno_n_modes",
        "tree_leaf_fno_n_layers",
    ):
        raw_value = getattr(config, field_name)
        if raw_value is None:
            continue
        if int(raw_value) <= 0:
            raise ValueError(f"{field_name} must be positive when provided")
    if str(config.doc_transformer_head_family) not in VALID_DOC_TRANSFORMER_HEAD_FAMILIES:
        raise ValueError(
            "doc_transformer_head_family="
            f"{config.doc_transformer_head_family!r} unsupported; expected one of "
            f"{VALID_DOC_TRANSFORMER_HEAD_FAMILIES}"
        )
    if int(config.doc_transformer_layers) < 0:
        raise ValueError("doc_transformer_layers must be non-negative")
    if int(config.eval_guidance_trials) < 0:
        raise ValueError("eval_guidance_trials must be non-negative")
    for q in tuple(config.eval_guidance_qs):
        qf = float(q)
        if qf < 0.0 or qf > 1.0:
            raise ValueError("eval_guidance_qs must contain values in [0,1]")
    if str(config.guidance_override_mode) not in VALID_GUIDANCE_OVERRIDE_MODES:
        raise ValueError(
            "guidance_override_mode="
            f"{config.guidance_override_mode!r} unsupported; expected one of {VALID_GUIDANCE_OVERRIDE_MODES}"
        )
    if bool(config.include_rf_root_baseline):
        if not bool(config.include_root_query):
            raise ValueError("include_rf_root_baseline requires include_root_query=true")
    if bool(config.include_rf_root_baseline) or bool(config.include_leaf_rf_tree_baseline):
        if int(config.rf_n_estimators) <= 0:
            raise ValueError("rf_n_estimators must be positive")
    if (
        bool(config.include_rf_root_baseline)
        or bool(config.include_leaf_dt_tree_baseline)
        or bool(config.include_leaf_rf_tree_baseline)
    ):
        if int(config.rf_max_depth) <= 0:
            raise ValueError("rf_max_depth must be positive")
        if int(config.rf_min_samples_leaf) <= 0:
            raise ValueError("rf_min_samples_leaf must be positive")
    if (
        bool(config.include_doc_level_ridge_baseline)
        or bool(config.include_leaf_ridge_tree_baseline)
        or bool(config.include_leaf_knn_tree_baseline)
        or bool(config.include_sampled_leaf_pool_ridge_baseline)
    ) and float(config.doc_level_ridge_alpha) < 0.0:
        raise ValueError("doc_level_ridge_alpha must be non-negative")
    if bool(config.include_leaf_knn_tree_baseline) and int(config.leaf_knn_neighbors) <= 0:
        raise ValueError("leaf_knn_neighbors must be positive")
    sampled_leaf_pool_leaf_counts = _sampled_leaf_budget_values(config)
    if (
        bool(config.include_sampled_leaf_pool_ridge_baseline)
        or bool(config.include_sampled_leaf_pool_rf_baseline)
    ) and not sampled_leaf_pool_leaf_counts:
        raise ValueError(
            "sampled_leaf_pool_leaf_counts must include at least one positive budget when "
            "sampled leaf pool baselines are enabled"
        )

    seeds = _resolve_runtime_seeds(config)
    _set_global_seed(int(seeds["effective_model_seed"]))
    if int(config.torch_threads) > 0:
        torch.set_num_threads(int(config.torch_threads))

    if config.use_cuda and torch.cuda.is_available():
        if config.cuda_device is not None:
            idx = int(config.cuda_device)
            if idx < 0 or idx >= int(torch.cuda.device_count()):
                raise ValueError(f"cuda_device={idx} out of range")
            torch.cuda.set_device(idx)
            device = torch.device(f"cuda:{idx}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    objective = _build_objective_summary(config)
    _validate_unified_fno_local_law_objective(config, objective)
    normalized_tree_supervision_source = str(
        getattr(config, "tree_supervision_source", "rate") or "rate"
    ).strip().lower()
    normalized_tree_local_weighting_mode = str(
        getattr(config, "tree_local_weighting_mode", "fixed_k_hajek")
        or "fixed_k_hajek"
    ).strip().lower()
    if normalized_tree_supervision_source != "rate":
        raise ValueError(
            "run_markov_changepoint_ops_count_experiment only supports "
            "tree_supervision_source='rate'. Use "
            "scripts/run_markov_optimization_tradeoff_pipeline.py "
            "for authoritative manifest-parity runs."
        )
    if normalized_tree_local_weighting_mode != "fixed_k_hajek":
        raise ValueError(
            "run_markov_changepoint_ops_count_experiment only supports "
            "tree_local_weighting_mode='fixed_k_hajek'. Use "
            "scripts/run_markov_optimization_tradeoff_pipeline.py "
            "for authoritative span-mass IPW parity runs."
        )
    if data_bundle is None:
        data_bundle = build_markov_changepoint_ops_count_data_bundle(config)
        data_bundle_source = "generated"
    else:
        data_bundle_source = "provided"
    if int(config.train_docs) > len(data_bundle.train_docs):
        raise ValueError("provided data_bundle does not contain enough training documents")
    if int(config.val_docs) > len(data_bundle.val_docs):
        raise ValueError("provided data_bundle does not contain enough validation documents")
    if int(config.test_docs) > len(data_bundle.test_docs):
        raise ValueError("provided data_bundle does not contain enough test documents")
    docs_train = tuple(data_bundle.train_docs[: int(config.train_docs)])
    docs_val = tuple(data_bundle.val_docs[: int(config.val_docs)])
    docs_test = tuple(data_bundle.test_docs[: int(config.test_docs)])
    train_condition_ids = _condition_ids_for_split(data_bundle, "train", len(docs_train))
    val_condition_ids = _condition_ids_for_split(data_bundle, "val", len(docs_val))
    test_condition_ids = _condition_ids_for_split(data_bundle, "test", len(docs_test))
    train_corpus_signature = _markov_corpus_signature(docs_train)
    val_corpus_signature = _markov_corpus_signature(docs_val)
    test_corpus_signature = _markov_corpus_signature(docs_test)
    full_sequence_input_signatures = _shared_full_sequence_input_signatures(
        train_docs=docs_train,
        val_docs=docs_val,
        test_docs=docs_test,
        pad_id=int(config.vocab_size),
    )
    train_target_diagnostics = _root_count_diagnostics(
        docs_train,
        condition_ids=train_condition_ids,
    )
    val_target_diagnostics = _root_count_diagnostics(
        docs_val,
        condition_ids=val_condition_ids,
    )
    test_target_diagnostics = _root_count_diagnostics(
        docs_test,
        condition_ids=test_condition_ids,
    )
    hazard_panel_metadata = dict(getattr(data_bundle, "metadata", {}) or {})
    config_payload = {
        **asdict(config),
        **seeds,
        "train_corpus_signature": str(train_corpus_signature),
        "val_corpus_signature": str(val_corpus_signature),
        "test_corpus_signature": str(test_corpus_signature),
        "data_bundle_source": str(data_bundle_source),
        "data_bundle_metadata": hazard_panel_metadata,
        "hazard_panel_id": str(hazard_panel_metadata.get("hazard_panel_id", "")),
        "full_sequence_input_backend": "shared_token_sequence_arrays",
        "full_sequence_input_signatures": dict(full_sequence_input_signatures),
        "train_target_diagnostics": train_target_diagnostics,
        "val_target_diagnostics": val_target_diagnostics,
        "test_target_diagnostics": test_target_diagnostics,
        "degenerate_root_target_detected": bool(
            train_target_diagnostics["is_constant"]
            or val_target_diagnostics["is_constant"]
            or test_target_diagnostics["is_constant"]
        ),
    }

    # Deterministic baselines (no training).
    exact = _eval_exact_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
    )
    leaf_bucket = _eval_leaf_bucket_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
    )
    undersupported = _eval_count_only_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
    )
    flip_r1 = _eval_flip_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
        rounds=1,
    )
    flip_r2 = _eval_flip_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
        rounds=2,
    )
    exact_families: Dict[str, SketchMetrics] = {
        "exact": exact,
        "leaf_bucket": leaf_bucket,
        "count_only": undersupported,
        "flip_R2": flip_r2,
    }

    # Learned sketch.
    train_prepped = _prepare_count_docs(
        docs_train,
        leaf_tokens=int(config.fixed_leaf_tokens),
        n_regimes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        feature_mode=str(config.feature_mode),
    )
    val_prepped = _prepare_count_docs(
        docs_val,
        leaf_tokens=int(config.fixed_leaf_tokens),
        n_regimes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        feature_mode=str(config.feature_mode),
    )
    test_prepped = _prepare_count_docs(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        n_regimes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        feature_mode=str(config.feature_mode),
    )
    target_scale = float(max(1, int(config.max_segments) - 1))
    if str(config.exact_family or "").strip():
        selected_stress_family = exact_families[str(config.exact_family).strip()]
        geom = _training_geometry(
            train_prepped,
            policy=str(config.audit_policy),
            fixed_nodes=int(config.audit_fixed_nodes),
            fraction=float(config.audit_fraction),
            scale=float(config.audit_scale),
            leaf_query_rate=float(config.leaf_query_rate),
            include_root_query=bool(config.include_root_query),
        )
        metrics: Dict[str, object] = {
            "stress_family": {
                **asdict(selected_stress_family),
                **_sketch_metric_alias_payload(selected_stress_family),
                "stress_family_name": str(config.exact_family).strip(),
                **_metrics_with_split_prefix(
                    selected_stress_family, prefix="test", target_scale=target_scale
                ),
                "test_theorem_bundle_score_n": float(
                    markov_law_bundle_score(
                        c1=float(selected_stress_family.leaf_mae) / float(target_scale),
                        c2=float(selected_stress_family.c2_count_drift_r1_mae)
                        / float(target_scale),
                        c3=float(selected_stress_family.merge_mae) / float(target_scale),
                    )
                ),
            },
            "exact": {**asdict(exact), **_sketch_metric_alias_payload(exact)},
            "leaf_bucket": {
                **asdict(leaf_bucket),
                **_sketch_metric_alias_payload(leaf_bucket),
            },
            "undersupported": {
                **asdict(undersupported),
                **_sketch_metric_alias_payload(undersupported),
            },
            "flip_R1": {**asdict(flip_r1), **_sketch_metric_alias_payload(flip_r1)},
            "flip_R2": {**asdict(flip_r2), **_sketch_metric_alias_payload(flip_r2)},
        }
        current_role = (
            PolicyRole.ORACLE_G.value
            if str(config.exact_family).strip() == "exact"
            else PolicyRole.COUNTEREXAMPLE_G.value
        )
        local_law_learnability, g_artifacts = _build_markov_local_law_learnability(
            config=config,
            seeds=seeds,
            target_scale=float(target_scale),
            objective_summary=objective,
            geom=geom,
            exact=exact,
            leaf_bucket=leaf_bucket,
            undersupported=undersupported,
            flip_r2=flip_r2,
            current_name=str(config.exact_family).strip(),
            current_role=str(current_role),
            current_train=None,
            current_val=None,
            current_test=selected_stress_family,
            current_selection_metric_name="configured_exact_family",
            current_selection_metric=float(
                markov_law_bundle_score(
                    c1=float(selected_stress_family.leaf_mae) / float(target_scale),
                    c2=float(selected_stress_family.c2_count_drift_r1_mae)
                    / float(target_scale),
                    c3=float(selected_stress_family.merge_mae) / float(target_scale),
                )
            ),
            model=None,
        )
        return OPSCountSummary(
            config=config_payload,
            training_geometry=asdict(geom),
            objective=objective,
            metrics=metrics,
            estimator_diagnostics={
                **asdict(
                    EstimatorDiagnostics(
                        true_mean=0.0,
                        naive_bias=0.0,
                        ipw_bias=0.0,
                        dsl_bias=0.0,
                        ipw_var=0.0,
                        dsl_var=0.0,
                    )
                ),
                "selection_demo_base_rate": 0.0,
                "selection_demo_pi_min": 0.0,
                "selection_demo_n_units": 0.0,
            },
            local_law_learnability=local_law_learnability,
            g_artifacts=g_artifacts,
        )
    _is_fno = str(config.model_family) in ("fno", "neural")
    if _is_fno:
        from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import HAS_NEURAL_OPERATOR
        if not HAS_NEURAL_OPERATOR:
            raise ImportError(
                f"model_family={config.model_family!r} requires the neuraloperator package. "
                "Install with: uv add neuraloperator  "
                "(or use model_family='additive' to avoid this dependency)"
            )
    doc_sequence_view_train = None
    doc_sequence_view_val = None
    doc_sequence_view_test = None
    doc_sequence_train_docs_used = 0
    if train_prepped:
        feature_dim = int(train_prepped[0].leaf_features[0].numel())
        if _is_fno:
            from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (
                FNOCountSketch,
                _class_setup,
                _apply_root_only_fraction_to_fno_docs,
                _eval_fno_doc_sequence_view,
                _eval_fno_model,
                _eval_fno_full_tree_ipw_metrics,
                _eval_fno_objective_terms,
                _gpu_runtime_config_from_ops_config,
                _FNOCountDoc,
                _prepare_fno_count_docs,
                _root_only_view_fno_docs,
            )
            fno_tree_train_regular = _prepare_fno_count_docs(
                docs_train, leaf_tokens=int(config.fixed_leaf_tokens)
            )
            fno_tree_val_regular = _prepare_fno_count_docs(
                docs_val, leaf_tokens=int(config.fixed_leaf_tokens)
            )
            fno_tree_test_regular = _prepare_fno_count_docs(
                docs_test, leaf_tokens=int(config.fixed_leaf_tokens)
            )
            fno_tree_train = _apply_root_only_fraction_to_fno_docs(
                fno_tree_train_regular,
                root_only_fraction=float(config.root_only_train_fraction),
                seed=int(seeds["effective_model_seed"]) + 41_003,
            )
            fno_tree_val = _apply_root_only_fraction_to_fno_docs(
                fno_tree_val_regular,
                root_only_fraction=float(config.root_only_train_fraction),
                seed=int(seeds["effective_model_seed"]) + 42_003,
            )
            fno_tree_test = fno_tree_test_regular
            fno_tree_train_root_only = _root_only_view_fno_docs(fno_tree_train_regular)
            fno_tree_val_root_only = _root_only_view_fno_docs(fno_tree_val_regular)
            fno_tree_test_root_only = _root_only_view_fno_docs(fno_tree_test_regular)
            _fno_root_train = np.asarray(
                [float(doc.root_count) for doc in fno_tree_train_regular],
                dtype=np.float64,
            )
            _fno_root_val = np.asarray(
                [float(doc.root_count) for doc in fno_tree_val_regular],
                dtype=np.float64,
            )
            _fno_root_test = np.asarray(
                [float(doc.root_count) for doc in fno_tree_test_regular],
                dtype=np.float64,
            )
            _unused_target_max, fno_doc_sequence_class_values, fno_doc_sequence_class_index, _unused_class_arr = _class_setup(
                _fno_root_train,
                _fno_root_val,
                _fno_root_test,
            )
            from treepo._research.ctreepo.sim.core.fno_arch_config import resolve_fno_arch
            _fno_arch = resolve_fno_arch(config)
            tree_leaf_fno_width = _fno_arch.width
            tree_leaf_fno_n_modes = _fno_arch.n_modes
            tree_leaf_fno_n_layers = _fno_arch.n_layers
            def _resolve_oracle_metric_from_config(cfg: OPSCountConfig):
                _om_name = str(getattr(cfg, "oracle_metric_name", "")).strip()
                if not _om_name:
                    return None
                from treepo._research.ctreepo.sim.core.oracle_metric import resolve_oracle_metric
                import treepo._research.ctreepo.sim.core.markov_oracle_metric  # noqa: F401 — register
                return resolve_oracle_metric(_om_name)

            model = FNOCountSketch(
                vocab_size=int(config.vocab_size),
                leaf_tokens=int(config.fixed_leaf_tokens),
                state_dim=int(config.state_dim),
                hidden_dim=int(config.hidden_dim),
                target_scale=float(target_scale),
                n_regimes=int(config.n_regimes),
                doc_sequence_class_values=fno_doc_sequence_class_values,
                fno_width=int(tree_leaf_fno_width),
                fno_n_modes=int(tree_leaf_fno_n_modes),
                fno_n_layers=int(tree_leaf_fno_n_layers),
                root_supervision_kind=str(config.tree_root_supervision_kind),
                root_count_class_values=fno_doc_sequence_class_values,
                endpoint_loss_scale=float(config.endpoint_loss_scale),
                aligned_sketch_surface=str(config.aligned_sketch_surface),
                summary_spec_name=str(config.summary_spec_name),
                slot_count=int(config.slot_count),
                join_bit_weight=float(config.tree_join_bit_weight),
                task_head_mode=str(config.tree_task_head_mode),
                theorem_surface_mode=str(config.tree_theorem_surface_mode),
                theorem_count_head_mode=str(config.tree_theorem_count_head_mode),
                theorem_count_ordinal_weight=float(
                    config.tree_theorem_count_ordinal_weight
                ),
                theorem_count_scalar_aux_weight=float(
                    config.tree_theorem_count_scalar_aux_weight
                ),
                theorem_count_threshold_balance=bool(
                    config.tree_theorem_count_threshold_balance
                ),
                theorem_feature_dim=int(config.tree_theorem_feature_dim),
                theorem_feature_hidden_dim=int(
                    config.tree_theorem_feature_hidden_dim
                ),
                merge_hidden_dim=int(getattr(config, "tree_merge_hidden_dim", 0)),
                theorem_score_dim=int(getattr(config, "tree_theorem_score_dim", 0)),
                theorem_fiber_dim=int(getattr(config, "tree_theorem_fiber_dim", 0)),
                theorem_aux_dim=int(getattr(config, "tree_theorem_aux_dim", 0)),
                score_merge_mode="gated_affine",
                phi_alignment_loss=str(config.tree_phi_alignment_loss),
                c2_mode=str(getattr(config, "tree_c2_mode", "reconstruction")),
                theorem_feature_adapter=str(
                    getattr(config, "theorem_feature_adapter", DEFAULT_THEOREM_FEATURE_ADAPTER)
                ),
                theorem_pair_same_threshold=getattr(
                    config, "theorem_pair_same_threshold", None
                ),
                theorem_pair_diff_threshold=getattr(
                    config, "theorem_pair_diff_threshold", None
                ),
                summary_spec_root_mode=str(config.tree_summary_spec_root_mode),
                theorem_count_dim=int(config.tree_theorem_count_dim),
                theorem_first_dim=int(config.tree_theorem_first_dim),
                theorem_last_dim=int(config.tree_theorem_last_dim),
                oracle_metric=_resolve_oracle_metric_from_config(config),
                oracle_same_threshold=float(getattr(config, "oracle_same_threshold", 0.0)),
                oracle_diff_threshold=float(getattr(config, "oracle_diff_threshold", 0.0)),
                tree_model_version=str(getattr(config, "tree_model_version", "legacy")),
            ).to(device=device)
            from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (
                train_fno_tree_local_law,
            )

            _fno_result = train_fno_tree_local_law(
                model=model,
                train_docs=fno_tree_train,
                val_docs=fno_tree_val,
                device=device,
                n_epochs=int(config.n_epochs),
                batch_size=int(config.batch_size),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay),
                leaf_sample_rate=float(config.ipw_leaf_sample_rate),
                internal_sample_rate=float(config.ipw_internal_sample_rate),
                root_objective_share=float(objective["root_share"]),
                local_law_objective_share=float(objective["local_law_weight"]),
                local_law_objective_mode=str(normalized_local_law_objective_mode),
                depth_discount_gamma=float(getattr(config, "depth_discount_gamma", 1.0)),
                use_residual_decomposition=bool(config.use_residual_decomposition),
                doc_sequence_train_fraction=float(config.doc_sequence_train_fraction),
                doc_sequence_objective=str(config.doc_sequence_objective),
                doc_sequence_class_index=fno_doc_sequence_class_index,
                grad_clip_norm=float(config.grad_clip_norm),
                seed=int(seeds["effective_model_seed"]),
            )
            _fno_loss_curve = tuple(float(x) for x in _fno_result["loss_curve"])
            train_loss_final = TrainFitDiagnostics(
                train_loss_final=float(_fno_result["train"]["root_mae"]),
                train_loss_curve=_fno_loss_curve,
                epochs_completed=int(config.n_epochs),
                selection_metric_curve=_fno_loss_curve,
                selection_mode="best_val_mae" if fno_tree_val else "final_epoch_no_validation",
                selection_split="val" if fno_tree_val else "config",
                selection_metric_name=(
                    str(_fno_result.get("selection_metric_name", "val_root_mae"))
                    if fno_tree_val
                    else "train_loss_final"
                ),
                selection_metric_value=float(_fno_result["best_val_mae"]),
                best_epoch=int(_fno_result["best_epoch"]),
            )
            doc_sequence_train_docs_used = int(_fno_result.get("doc_sequence_train_docs_used", 0))
        elif str(config.model_family) == "additive":
            model = AdditiveCountSketch(
                feature_dim=int(feature_dim),
                hidden_dim=int(config.hidden_dim),
                target_scale=float(target_scale),
                n_regimes=int(config.n_regimes),
                use_endpoints=str(config.feature_mode) == "full",
                c2_learned_resummary=bool(config.c2_learned_resummary),
            ).to(device=device)
            train_loss_final = _fit_additive_leaf_encoder_closed_form(
                model,
                train_prepped,
                device=device,
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                c3_audit_strategy=str(config.c3_audit_strategy),
                c3_include_root=bool(config.c3_include_root),
                leaf_query_rate=float(config.leaf_query_rate),
                seed=int(seeds["effective_model_seed"]),
            )
        else:
            raise ValueError(
                f"unsupported model_family={config.model_family!r}; "
                f"expected one of {VALID_MODEL_FAMILIES}"
            )
        # --- Evaluation dispatch ---
        _obj_kw = dict(
            device=device,
            leaf_weight=float(objective["local_law_c1_weight"]),
            c2_weight=float(objective["local_law_c2_weight"]),
            c3_weight=float(objective["local_law_c3_weight"]),
            root_weight=float(objective["root_share"]),
            schedule_consistency_weight=float(objective["proxy_schedule_consistency_weight"]),
            include_root_query=bool(config.include_root_query),
        )
        if _is_fno:
            learned_train = _eval_fno_model(
                model,
                fno_tree_train_regular,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=train_condition_ids,
            )
            learned_val = _eval_fno_model(
                model,
                fno_tree_val_regular,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=val_condition_ids,
            )
            learned = _eval_fno_model(
                model,
                fno_tree_test_regular,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=test_condition_ids,
            )
            root_only_view_train = _eval_fno_model(
                model,
                fno_tree_train_root_only,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=train_condition_ids,
            )
            root_only_view_val = _eval_fno_model(
                model,
                fno_tree_val_root_only,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=val_condition_ids,
            )
            root_only_view_test = _eval_fno_model(
                model,
                fno_tree_test_root_only,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=test_condition_ids,
            )
            doc_sequence_view_train = _eval_fno_doc_sequence_view(
                model,
                fno_tree_train_regular,
                device=device,
                tau=float(config.violation_tau),
            )
            doc_sequence_view_val = _eval_fno_doc_sequence_view(
                model,
                fno_tree_val_regular,
                device=device,
                tau=float(config.violation_tau),
            )
            doc_sequence_view_test = _eval_fno_doc_sequence_view(
                model,
                fno_tree_test_regular,
                device=device,
                tau=float(config.violation_tau),
            )
            train_weighted_objective = _eval_fno_objective_terms(
                model, fno_tree_train_regular, **_obj_kw
            )
            test_weighted_objective = _eval_fno_objective_terms(
                model, fno_tree_test_regular, **_obj_kw
            )
            val_weighted_objective = _eval_fno_objective_terms(
                model, fno_tree_val_regular, **_obj_kw
            )
            # DPO gap not yet supported for FNO models.
            dpo_gap_result = {
                "dpo_preference_gap": float("nan"),
                "dpo_mean_distortion": float("nan"),
                "dpo_n_pairs": 0,
                "dpo_beta": 1.0,
                "dpo_lipschitz_bound": float("nan"),
            }
        else:
            learned_train = _eval_learned_model(
                model,
                train_prepped,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=train_condition_ids,
            )
            learned_val = _eval_learned_model(
                model,
                val_prepped,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=val_condition_ids,
            )
            learned = _eval_learned_model(
                model,
                test_prepped,
                device=device,
                tau=float(config.violation_tau),
                condition_ids=test_condition_ids,
            )
            root_only_view_train = None
            root_only_view_val = None
            root_only_view_test = None
            doc_sequence_view_train = None
            doc_sequence_view_val = None
            doc_sequence_view_test = None
            train_weighted_objective = _eval_objective_terms(model, train_prepped, **_obj_kw)
            test_weighted_objective = _eval_objective_terms(model, test_prepped, **_obj_kw)
            val_weighted_objective = _eval_objective_terms(model, val_prepped, **_obj_kw)
            # DPO preference gap (Lean: dpo_gap_bounded in OPT/PreferenceBounds.lean).
            dpo_gap_result = _compute_dpo_preference_gap(
                model, test_prepped, device=device,
            )
        # --- Objective estimators ---
        if _is_fno:
            def _fno_encode_leaf(doc):
                return [
                    model.encode_leaf_tokens(doc.leaf_token_ids[j], device=device)
                    for j in range(len(doc.leaf_token_ids))
                ]
            _est_kw = dict(
                device=device,
                objective_summary=objective,
                config=config,
                leaf_query_rate=float(config.leaf_query_rate),
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                c3_audit_strategy=str(config.c3_audit_strategy),
                c3_include_root=bool(config.c3_include_root),
                encode_leaf_fn=_fno_encode_leaf,
            )
            train_objective_estimators = _markov_objective_estimator_payload(
                model, fno_tree_train_regular, exact_objective=train_weighted_objective,
                seed=int(seeds["effective_model_seed"]) + 1_003, **_est_kw,
            )
            val_objective_estimators = _markov_objective_estimator_payload(
                model, fno_tree_val_regular, exact_objective=val_weighted_objective,
                seed=int(seeds["effective_model_seed"]) + 2_003, **_est_kw,
            )
            test_objective_estimators = _markov_objective_estimator_payload(
                model, fno_tree_test_regular, exact_objective=test_weighted_objective,
                seed=int(seeds["effective_model_seed"]) + 3_003, **_est_kw,
            )
        else:
            train_objective_estimators = _markov_objective_estimator_payload(
                model,
                train_prepped,
                device=device,
                objective_summary=objective,
                config=config,
                exact_objective=train_weighted_objective,
                leaf_query_rate=float(config.leaf_query_rate),
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                c3_audit_strategy=str(config.c3_audit_strategy),
                c3_include_root=bool(config.c3_include_root),
                seed=int(seeds["effective_model_seed"]) + 1_003,
            )
            val_objective_estimators = _markov_objective_estimator_payload(
                model,
                val_prepped,
                device=device,
                objective_summary=objective,
                config=config,
                exact_objective=val_weighted_objective,
                leaf_query_rate=float(config.leaf_query_rate),
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                c3_audit_strategy=str(config.c3_audit_strategy),
                c3_include_root=bool(config.c3_include_root),
                seed=int(seeds["effective_model_seed"]) + 2_003,
            )
            test_objective_estimators = _markov_objective_estimator_payload(
                model,
                test_prepped,
                device=device,
                objective_summary=objective,
                config=config,
                exact_objective=test_weighted_objective,
                leaf_query_rate=float(config.leaf_query_rate),
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                c3_audit_strategy=str(config.c3_audit_strategy),
                c3_include_root=bool(config.c3_include_root),
                seed=int(seeds["effective_model_seed"]) + 3_003,
            )
    else:
        train_loss_final = TrainFitDiagnostics(
            train_loss_final=float("nan"),
            train_loss_curve=tuple(),
            epochs_completed=0,
            selection_metric_curve=tuple(),
            selection_mode="not_trained",
            selection_split="config",
            selection_metric_name="not_trained",
            selection_metric_value=float("nan"),
            best_epoch=0,
        )
        learned_train = _zero_sketch_metrics(n_docs=int(len(train_prepped)))
        learned_val = _zero_sketch_metrics(n_docs=int(len(val_prepped)))
        learned = _zero_sketch_metrics(n_docs=int(len(test_prepped)))
        train_weighted_objective = ObjectiveMetrics(
            optimization_total_loss=float("nan"),
            optimization_root_loss=float("nan"),
            optimization_leaf_loss=float("nan"),
            optimization_c2_loss=float("nan"),
            optimization_merge_loss=float("nan"),
            optimization_schedule_consistency_loss=float("nan"),
            raw_total_loss=float("nan"),
            raw_root_loss=float("nan"),
            raw_leaf_loss=float("nan"),
            raw_c2_loss=float("nan"),
            raw_merge_loss=float("nan"),
            raw_schedule_consistency_loss=float("nan"),
            n_docs=int(len(train_prepped)),
        )
        val_weighted_objective = ObjectiveMetrics(
            optimization_total_loss=float("nan"),
            optimization_root_loss=float("nan"),
            optimization_leaf_loss=float("nan"),
            optimization_c2_loss=float("nan"),
            optimization_merge_loss=float("nan"),
            optimization_schedule_consistency_loss=float("nan"),
            raw_total_loss=float("nan"),
            raw_root_loss=float("nan"),
            raw_leaf_loss=float("nan"),
            raw_c2_loss=float("nan"),
            raw_merge_loss=float("nan"),
            raw_schedule_consistency_loss=float("nan"),
            n_docs=int(len(val_prepped)),
        )
        train_objective_estimators = {}
        val_objective_estimators = {}
        test_objective_estimators = {}
        root_only_view_train = None
        root_only_view_val = None
        root_only_view_test = None
        test_weighted_objective = ObjectiveMetrics(
            optimization_total_loss=float("nan"),
            optimization_root_loss=float("nan"),
            optimization_leaf_loss=float("nan"),
            optimization_c2_loss=float("nan"),
            optimization_merge_loss=float("nan"),
            optimization_schedule_consistency_loss=float("nan"),
            raw_total_loss=float("nan"),
            raw_root_loss=float("nan"),
            raw_leaf_loss=float("nan"),
            raw_c2_loss=float("nan"),
            raw_merge_loss=float("nan"),
            raw_schedule_consistency_loss=float("nan"),
            n_docs=int(len(test_prepped)),
        )
        dpo_gap_result = {
            "dpo_preference_gap": float("nan"),
            "dpo_mean_distortion": float("nan"),
            "dpo_n_pairs": 0,
            "dpo_beta": 1.0,
            "dpo_lipschitz_bound": float("nan"),
        }

    rf_root: Optional[SketchMetrics] = None
    rf_root_val: Optional[SketchMetrics] = None
    if bool(config.include_rf_root_baseline):
        rf_root_val = _eval_rf_root_baseline(
            train_prepped,
            val_prepped,
            seed=int(seeds["effective_model_seed"]),
            n_estimators=int(config.rf_n_estimators),
            max_depth=int(config.rf_max_depth),
            min_samples_leaf=int(config.rf_min_samples_leaf),
        )
        rf_root = _eval_rf_root_baseline(
            train_prepped,
            test_prepped,
            seed=int(seeds["effective_model_seed"]),
            n_estimators=int(config.rf_n_estimators),
            max_depth=int(config.rf_max_depth),
            min_samples_leaf=int(config.rf_min_samples_leaf),
        )

    doc_level_train: Optional[SketchMetrics] = None
    doc_level_val: Optional[SketchMetrics] = None
    doc_level: Optional[SketchMetrics] = None
    doc_level_fit: Optional[TrainFitDiagnostics] = None
    doc_sequence_train: Optional[SketchMetrics] = None
    doc_sequence_val: Optional[SketchMetrics] = None
    doc_sequence: Optional[SketchMetrics] = None
    doc_sequence_fit: Optional[TrainFitDiagnostics] = None
    doc_transformer_train: Optional[SketchMetrics] = None
    doc_transformer_val: Optional[SketchMetrics] = None
    doc_transformer: Optional[SketchMetrics] = None
    doc_transformer_fit: Optional[TrainFitDiagnostics] = None
    fno_train: Optional[SketchMetrics] = None
    fno_val: Optional[SketchMetrics] = None
    fno_test: Optional[SketchMetrics] = None
    fno_fit: Optional[TrainFitDiagnostics] = None
    deeponet_train: Optional[SketchMetrics] = None
    deeponet_val: Optional[SketchMetrics] = None
    deeponet_test: Optional[SketchMetrics] = None
    deeponet_fit: Optional[TrainFitDiagnostics] = None
    mlp_bigram_train: Optional[SketchMetrics] = None
    mlp_bigram_val: Optional[SketchMetrics] = None
    mlp_bigram_test: Optional[SketchMetrics] = None
    mlp_bigram_fit: Optional[TrainFitDiagnostics] = None
    cnn1d_train: Optional[SketchMetrics] = None
    cnn1d_val: Optional[SketchMetrics] = None
    cnn1d_test: Optional[SketchMetrics] = None
    cnn1d_fit: Optional[TrainFitDiagnostics] = None
    doc_level_ridge_train: Optional[SketchMetrics] = None
    doc_level_ridge_val: Optional[SketchMetrics] = None
    doc_level_ridge: Optional[SketchMetrics] = None
    doc_level_ridge_fit: Optional[TrainFitDiagnostics] = None
    doc_level_ridge_breakdown_train: Dict[str, SketchMetrics] = {}
    doc_level_ridge_breakdown_val: Dict[str, SketchMetrics] = {}
    doc_level_ridge_breakdown_test: Dict[str, SketchMetrics] = {}
    doc_level_ridge_breakdown_fit: Dict[str, TrainFitDiagnostics] = {}
    leaf_ridge_tree_train: Optional[SketchMetrics] = None
    leaf_ridge_tree_val: Optional[SketchMetrics] = None
    leaf_ridge_tree: Optional[SketchMetrics] = None
    leaf_ridge_tree_fit: Optional[TrainFitDiagnostics] = None
    leaf_endpoint_table_tree_train: Optional[SketchMetrics] = None
    leaf_endpoint_table_tree_val: Optional[SketchMetrics] = None
    leaf_endpoint_table_tree: Optional[SketchMetrics] = None
    leaf_endpoint_table_tree_fit: Optional[TrainFitDiagnostics] = None
    leaf_dt_tree_train: Optional[SketchMetrics] = None
    leaf_dt_tree_val: Optional[SketchMetrics] = None
    leaf_dt_tree: Optional[SketchMetrics] = None
    leaf_dt_tree_fit: Optional[TrainFitDiagnostics] = None
    leaf_knn_tree_train: Optional[SketchMetrics] = None
    leaf_knn_tree_val: Optional[SketchMetrics] = None
    leaf_knn_tree: Optional[SketchMetrics] = None
    leaf_knn_tree_fit: Optional[TrainFitDiagnostics] = None
    leaf_rf_tree_train: Optional[SketchMetrics] = None
    leaf_rf_tree_val: Optional[SketchMetrics] = None
    leaf_rf_tree: Optional[SketchMetrics] = None
    leaf_rf_tree_fit: Optional[TrainFitDiagnostics] = None
    doc_level_supervision_artifact: Optional[str] = None
    doc_level_supervision_rows: int = 0
    if bool(config.include_doc_level_baseline) or bool(config.include_doc_level_ridge_baseline):
        train_doc_level = _prepare_doc_level_count_docs(
            docs_train,
            n_regimes=int(config.n_regimes),
            vocab_size=int(config.vocab_size),
            feature_mode=str(config.feature_mode),
        )
        val_doc_level = _prepare_doc_level_count_docs(
            docs_val,
            n_regimes=int(config.n_regimes),
            vocab_size=int(config.vocab_size),
            feature_mode=str(config.feature_mode),
        )
        test_doc_level = _prepare_doc_level_count_docs(
            docs_test,
            n_regimes=int(config.n_regimes),
            vocab_size=int(config.vocab_size),
            feature_mode=str(config.feature_mode),
        )
        doc_level_train_supervision = _doc_level_supervision_dataset(
            train_doc_level,
            split="train",
            target_scale=float(target_scale),
        )
        doc_level_val_supervision = _doc_level_supervision_dataset(
            val_doc_level,
            split="val",
            target_scale=float(target_scale),
        )
        doc_level_test_supervision = _doc_level_supervision_dataset(
            test_doc_level,
            split="test",
            target_scale=float(target_scale),
        )
        doc_level_supervision_rows = int(len(doc_level_train_supervision.response_judgments))
        artifact_dir = Path(str(config.artifact_dir)) if str(config.artifact_dir).strip() else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            doc_level_supervision_path = artifact_dir / "doc_level_supervision.json"
            doc_level_train_supervision.save(doc_level_supervision_path)
            doc_level_supervision_artifact = str(doc_level_supervision_path)

    if bool(config.include_doc_level_baseline):
        (
            doc_level_train,
            doc_level_val,
            doc_level,
            doc_level_fit,
        ) = _fit_doc_level_baseline(
            config=config,
            seeds=seeds,
            target_scale=float(target_scale),
            device=device,
            train_docs=train_doc_level,
            val_docs=val_doc_level,
            test_docs=test_doc_level,
            train_supervision=doc_level_train_supervision,
            val_supervision=doc_level_val_supervision,
            test_supervision=doc_level_test_supervision,
        )
    if bool(config.include_doc_level_ridge_baseline):
        (
            doc_level_ridge_train,
            doc_level_ridge_val,
            doc_level_ridge,
            doc_level_ridge_fit,
        ) = _fit_doc_level_ridge_baseline(
            train_docs=train_doc_level,
            val_docs=val_doc_level,
            test_docs=test_doc_level,
            train_supervision=doc_level_train_supervision,
            ridge_alpha=float(config.doc_level_ridge_alpha),
            tau=float(config.violation_tau),
        )
        for order in _normalized_ngram_orders(config.doc_level_ridge_breakdown_orders):
            label = _ngram_order_label(int(order))
            (
                doc_level_ridge_breakdown_train[label],
                doc_level_ridge_breakdown_val[label],
                doc_level_ridge_breakdown_test[label],
                doc_level_ridge_breakdown_fit[label],
            ) = _fit_doc_token_ngram_ridge_baseline(
                train_docs=docs_train,
                val_docs=docs_val,
                test_docs=docs_test,
                vocab_size=int(config.vocab_size),
                orders=(int(order),),
                ridge_alpha=float(config.doc_level_ridge_alpha),
                tau=float(config.violation_tau),
            )
    if bool(config.include_doc_sequence_baseline):
        (
            doc_sequence_train,
            doc_sequence_val,
            doc_sequence,
            doc_sequence_fit,
        ) = _fit_doc_sequence_baseline(
            config=config,
            seeds=seeds,
            device=device,
            train_docs=docs_train,
            val_docs=docs_val,
            test_docs=docs_test,
        )
    if bool(config.include_doc_transformer_baseline):
        (
            doc_transformer_train,
            doc_transformer_val,
            doc_transformer,
            doc_transformer_fit,
        ) = _fit_doc_transformer_baseline(
            config=config,
            seeds=seeds,
            device=device,
            train_docs=docs_train,
            val_docs=docs_val,
            test_docs=docs_test,
        )
    if bool(config.include_fno_baseline):
        from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import _fit_fno_baseline
        (fno_train, fno_val, fno_test, fno_fit) = _fit_fno_baseline(
            config=config, seeds=seeds, device=device,
            train_docs=docs_train, val_docs=docs_val, test_docs=docs_test,
        )
    if bool(config.include_deeponet_baseline):
        from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import _fit_deeponet_baseline
        (deeponet_train, deeponet_val, deeponet_test, deeponet_fit) = _fit_deeponet_baseline(
            config=config, seeds=seeds, device=device,
            train_docs=docs_train, val_docs=docs_val, test_docs=docs_test,
        )
    if bool(config.include_mlp_bigram_baseline):
        from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import _fit_mlp_bigram_baseline
        (mlp_bigram_train, mlp_bigram_val, mlp_bigram_test, mlp_bigram_fit) = _fit_mlp_bigram_baseline(
            config=config, seeds=seeds, device=device,
            train_docs=docs_train, val_docs=docs_val, test_docs=docs_test,
        )
    if bool(config.include_cnn1d_baseline):
        from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import _fit_cnn1d_baseline
        (cnn1d_train, cnn1d_val, cnn1d_test, cnn1d_fit) = _fit_cnn1d_baseline(
            config=config, seeds=seeds, device=device,
            train_docs=docs_train, val_docs=docs_val, test_docs=docs_test,
        )
    leaf_local_supervision_seed = int(seeds["effective_model_seed"]) + 61_003
    leaf_ridge_tree_supervision_artifact: Optional[str] = None
    leaf_ridge_tree_supervision_rows: int = 0
    if bool(config.include_leaf_ridge_tree_baseline):
        leaf_ridge_tree_supervision = _leaf_ridge_tree_supervision_dataset(
            train_prepped,
            split="train",
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            leaf_query_rate=float(config.leaf_query_rate),
            seed=int(leaf_local_supervision_seed),
        )
        leaf_ridge_tree_supervision_rows = int(
            len(leaf_ridge_tree_supervision.response_judgments)
        )
        artifact_dir = Path(str(config.artifact_dir)) if str(config.artifact_dir).strip() else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            supervision_path = artifact_dir / "leaf_ridge_tree_supervision.json"
            leaf_ridge_tree_supervision.save(supervision_path)
            leaf_ridge_tree_supervision_artifact = str(supervision_path)
        (
            leaf_ridge_tree_train,
            leaf_ridge_tree_val,
            leaf_ridge_tree,
            leaf_ridge_tree_fit,
        ) = _fit_leaf_ridge_tree_baseline(
            train_docs=train_prepped,
            val_docs=val_prepped,
            test_docs=test_prepped,
            train_supervision=leaf_ridge_tree_supervision,
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            ridge_alpha=float(config.doc_level_ridge_alpha),
            device=device,
            tau=float(config.violation_tau),
        )
    leaf_endpoint_table_tree_supervision_artifact: Optional[str] = None
    leaf_endpoint_table_tree_supervision_rows: int = 0
    if bool(config.include_leaf_endpoint_table_tree_baseline):
        leaf_endpoint_table_tree_supervision = _leaf_endpoint_table_supervision_dataset(
            train_prepped,
            split="train",
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            leaf_query_rate=float(config.leaf_query_rate),
            seed=int(leaf_local_supervision_seed),
        )
        leaf_endpoint_table_tree_supervision_rows = int(
            len(leaf_endpoint_table_tree_supervision.response_judgments)
        )
        artifact_dir = Path(str(config.artifact_dir)) if str(config.artifact_dir).strip() else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            supervision_path = artifact_dir / "leaf_endpoint_table_tree_supervision.json"
            leaf_endpoint_table_tree_supervision.save(supervision_path)
            leaf_endpoint_table_tree_supervision_artifact = str(supervision_path)
        (
            leaf_endpoint_table_tree_train,
            leaf_endpoint_table_tree_val,
            leaf_endpoint_table_tree,
            leaf_endpoint_table_tree_fit,
        ) = _fit_leaf_endpoint_table_tree_baseline(
            train_docs=train_prepped,
            val_docs=val_prepped,
            test_docs=test_prepped,
            train_supervision=leaf_endpoint_table_tree_supervision,
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            tau=float(config.violation_tau),
        )
    leaf_dt_tree_supervision_artifact: Optional[str] = None
    leaf_dt_tree_supervision_rows: int = 0
    if bool(config.include_leaf_dt_tree_baseline):
        leaf_dt_tree_supervision = _leaf_ridge_tree_supervision_dataset(
            train_prepped,
            split="train",
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            leaf_query_rate=float(config.leaf_query_rate),
            seed=int(leaf_local_supervision_seed),
        )
        leaf_dt_tree_supervision_rows = int(len(leaf_dt_tree_supervision.response_judgments))
        artifact_dir = Path(str(config.artifact_dir)) if str(config.artifact_dir).strip() else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            supervision_path = artifact_dir / "leaf_dt_tree_supervision.json"
            leaf_dt_tree_supervision.save(supervision_path)
            leaf_dt_tree_supervision_artifact = str(supervision_path)
        (
            leaf_dt_tree_train,
            leaf_dt_tree_val,
            leaf_dt_tree,
            leaf_dt_tree_fit,
        ) = _fit_leaf_dt_tree_baseline(
            train_docs=train_prepped,
            val_docs=val_prepped,
            test_docs=test_prepped,
            train_supervision=leaf_dt_tree_supervision,
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            seed=int(seeds["effective_model_seed"]) + 71_003,
            max_depth=int(config.rf_max_depth),
            min_samples_leaf=int(config.rf_min_samples_leaf),
            tau=float(config.violation_tau),
        )
    leaf_knn_tree_supervision_artifact: Optional[str] = None
    leaf_knn_tree_supervision_rows: int = 0
    if bool(config.include_leaf_knn_tree_baseline):
        leaf_knn_tree_supervision = _leaf_ridge_tree_supervision_dataset(
            train_prepped,
            split="train",
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            leaf_query_rate=float(config.leaf_query_rate),
            seed=int(leaf_local_supervision_seed),
        )
        leaf_knn_tree_supervision_rows = int(len(leaf_knn_tree_supervision.response_judgments))
        artifact_dir = Path(str(config.artifact_dir)) if str(config.artifact_dir).strip() else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            supervision_path = artifact_dir / "leaf_knn_tree_supervision.json"
            leaf_knn_tree_supervision.save(supervision_path)
            leaf_knn_tree_supervision_artifact = str(supervision_path)
        (
            leaf_knn_tree_train,
            leaf_knn_tree_val,
            leaf_knn_tree,
            leaf_knn_tree_fit,
        ) = _fit_leaf_knn_tree_baseline(
            train_docs=train_prepped,
            val_docs=val_prepped,
            test_docs=test_prepped,
            train_supervision=leaf_knn_tree_supervision,
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            n_neighbors=int(config.leaf_knn_neighbors),
            tau=float(config.violation_tau),
        )
    leaf_rf_tree_supervision_artifact: Optional[str] = None
    leaf_rf_tree_supervision_rows: int = 0
    if bool(config.include_leaf_rf_tree_baseline):
        leaf_rf_tree_supervision = _leaf_ridge_tree_supervision_dataset(
            train_prepped,
            split="train",
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            leaf_query_rate=float(config.leaf_query_rate),
            seed=int(leaf_local_supervision_seed),
        )
        leaf_rf_tree_supervision_rows = int(len(leaf_rf_tree_supervision.response_judgments))
        artifact_dir = Path(str(config.artifact_dir)) if str(config.artifact_dir).strip() else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            supervision_path = artifact_dir / "leaf_rf_tree_supervision.json"
            leaf_rf_tree_supervision.save(supervision_path)
            leaf_rf_tree_supervision_artifact = str(supervision_path)
        (
            leaf_rf_tree_train,
            leaf_rf_tree_val,
            leaf_rf_tree,
            leaf_rf_tree_fit,
        ) = _fit_leaf_rf_tree_baseline(
            train_docs=train_prepped,
            val_docs=val_prepped,
            test_docs=test_prepped,
            train_supervision=leaf_rf_tree_supervision,
            target_scale=float(target_scale),
            n_regimes=int(config.n_regimes),
            use_endpoints=bool(str(config.feature_mode) == "full"),
            seed=int(seeds["effective_model_seed"]) + 79_003,
            n_estimators=int(config.rf_n_estimators),
            max_depth=int(config.rf_max_depth),
            min_samples_leaf=int(config.rf_min_samples_leaf),
            tau=float(config.violation_tau),
        )

    sampled_leaf_pool_sweep: Optional[Dict[str, object]] = None
    if bool(config.include_sampled_leaf_pool_ridge_baseline) or bool(
        config.include_sampled_leaf_pool_rf_baseline
    ):
        sweep_points: List[Dict[str, object]] = []
        artifact_dir = Path(str(config.artifact_dir)) if str(config.artifact_dir).strip() else None
        for leaf_budget in sampled_leaf_pool_leaf_counts:
            sample_seed_base = int(seeds["effective_model_seed"]) + int(
                config.sampled_leaf_pool_seed_offset
            ) + 997 * int(leaf_budget)
            train_sampled_docs = _prepare_sampled_leaf_pool_docs(
                train_prepped,
                leaf_budget=int(leaf_budget),
                seed=int(sample_seed_base) + 11,
            )
            val_sampled_docs = _prepare_sampled_leaf_pool_docs(
                val_prepped,
                leaf_budget=int(leaf_budget),
                seed=int(sample_seed_base) + 29,
            )
            test_sampled_docs = _prepare_sampled_leaf_pool_docs(
                test_prepped,
                leaf_budget=int(leaf_budget),
                seed=int(sample_seed_base) + 47,
            )
            sampled_supervision = _sampled_leaf_pool_supervision_dataset(
                train_sampled_docs,
                split="train",
                target_scale=float(target_scale),
                leaf_budget=int(leaf_budget),
            )
            supervision_artifact_path: Optional[str] = None
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                supervision_path = artifact_dir / (
                    f"sampled_leaf_pool_budget_{int(leaf_budget)}_supervision.json"
                )
                sampled_supervision.save(supervision_path)
                supervision_artifact_path = str(supervision_path)

            point: Dict[str, object] = {
                "leaf_budget": int(leaf_budget),
                "pooling_strategy": "uniform_random_without_replacement",
                "train_observation": _sampled_leaf_pool_observation_summary(train_sampled_docs),
                "val_observation": _sampled_leaf_pool_observation_summary(val_sampled_docs),
                "test_observation": _sampled_leaf_pool_observation_summary(test_sampled_docs),
                "supervision_rows": int(len(sampled_supervision.response_judgments)),
            }
            if bool(config.include_sampled_leaf_pool_ridge_baseline):
                (
                    ridge_train,
                    ridge_val,
                    ridge_test,
                    ridge_fit,
                ) = _fit_sampled_leaf_pool_ridge_baseline(
                    train_docs=train_sampled_docs,
                    val_docs=val_sampled_docs,
                    test_docs=test_sampled_docs,
                    train_supervision=sampled_supervision,
                    ridge_alpha=float(config.doc_level_ridge_alpha),
                    tau=float(config.violation_tau),
                )
                point["ridge_train"] = asdict(ridge_train)
                point["ridge_val"] = asdict(ridge_val)
                point["ridge"] = asdict(ridge_test)
                point["ridge_training"] = {
                    "train_loss_final": float(ridge_fit.train_loss_final),
                    "train_loss_curve": [float(x) for x in ridge_fit.train_loss_curve],
                    "epochs_completed": int(ridge_fit.epochs_completed),
                    "selection_metric_curve": [
                        float(x) for x in ridge_fit.selection_metric_curve
                    ],
                    "selection_mode": str(ridge_fit.selection_mode),
                    "selection_split": str(ridge_fit.selection_split),
                    "selection_metric_name": str(ridge_fit.selection_metric_name),
                    "selection_metric_value": float(ridge_fit.selection_metric_value),
                    "best_epoch": int(ridge_fit.best_epoch),
                    "baseline_family": "sampled_leaf_pool_ridge",
                    "input_view": "sampled_leaf_pool_uniform",
                    "uses_tree_merges": False,
                    "supervision_artifact_path": supervision_artifact_path,
                    "sample_leaf_budget": int(leaf_budget),
                    "ridge_alpha": float(config.doc_level_ridge_alpha),
                    **supervision_training_contract(
                        representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                        target_kind=TARGET_SCALAR,
                        optimizer_family=OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
                        optimizer_backend="closed_form_ridge",
                        selection_mode=str(ridge_fit.selection_mode),
                        selection_split=str(ridge_fit.selection_split),
                        selection_metric_name=str(ridge_fit.selection_metric_name),
                        selection_metric_value=float(ridge_fit.selection_metric_value),
                        best_epoch=int(ridge_fit.best_epoch),
                        n_train_rows=int(len(sampled_supervision.response_judgments)),
                    ),
                }
            if bool(config.include_sampled_leaf_pool_rf_baseline):
                rf_train_pool, rf_val_pool, rf_test_pool = _fit_sampled_leaf_pool_rf_baseline(
                    train_sampled_docs,
                    val_sampled_docs,
                    test_sampled_docs,
                    seed=int(sample_seed_base) + 101,
                    n_estimators=int(config.rf_n_estimators),
                    max_depth=int(config.rf_max_depth),
                    min_samples_leaf=int(config.rf_min_samples_leaf),
                    tau=float(config.violation_tau),
                )
                point["rf_train"] = asdict(rf_train_pool)
                point["rf_val"] = asdict(rf_val_pool)
                point["rf"] = asdict(rf_test_pool)
                point["rf_training"] = {
                    "baseline_family": "sampled_leaf_pool_rf",
                    "input_view": "sampled_leaf_pool_uniform",
                    "uses_tree_merges": False,
                    "supervision_artifact_path": supervision_artifact_path,
                    "sample_leaf_budget": int(leaf_budget),
                    "rf_n_estimators": int(config.rf_n_estimators),
                    "rf_max_depth": int(config.rf_max_depth),
                    "rf_min_samples_leaf": int(config.rf_min_samples_leaf),
                    **supervision_training_contract(
                        representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                        target_kind=TARGET_SCALAR,
                        optimizer_family=OPTIMIZER_FAMILY_TREE_ENSEMBLE,
                        optimizer_backend="random_forest",
                        n_train_rows=int(len(sampled_supervision.response_judgments)),
                    ),
                }
            sweep_points.append(point)

        sampled_leaf_pool_sweep = {
            "present": bool(sweep_points),
            "input_view": "sampled_leaf_pool_uniform",
            "pooling_strategy": "uniform_random_without_replacement",
            "points": sweep_points,
        }

    geom = _training_geometry(
        train_prepped,
        policy=str(config.audit_policy),
        fixed_nodes=int(config.audit_fixed_nodes),
        fraction=float(config.audit_fraction),
        scale=float(config.audit_scale),
        leaf_query_rate=float(config.leaf_query_rate),
        include_root_query=bool(config.include_root_query),
    )

    # Selection-bias demo: estimate mean merge discrepancy under risk-biased sampling.
    #
    # We treat the learned model's per-merge absolute error on a fixed test set as a population.
    # We then sample merge nodes with non-uniform inclusion probabilities proportional to a
    # simple "risk score" (span size), and compare:
    #   - naive mean of sampled errors,
    #   - IPW mean,
    #   - DSL/AIPW mean using a crude learned proxy.
    #
    # This is intentionally simple: the point is to show bias under adaptive sampling and
    # how IPW/DSL correct it, while the magnitude of the population itself improves with
    # more training docs / more oracle labels.
    internal_label_rate = (
        float(geom.mean_internal_labels) / float(geom.mean_internal_nodes)
        if float(geom.mean_internal_nodes) > 0
        else 0.0
    )
    base = float(min(1.0, max(0.02, internal_label_rate)))
    pi_min = float(min(0.02, 0.10 * base))

    diag_errs: List[float] = []
    diag_scores: List[float] = []
    diag_preds: List[float] = []
    if train_prepped:
        assert "model" in locals()
        model.eval()
        with torch.no_grad():
            _diag_docs = fno_tree_test[:100] if _is_fno else test_prepped[:min(100, len(test_prepped))]
            for _diag_idx, _diag_doc in enumerate(_diag_docs):
                if _is_fno:
                    states = [
                        model.encode_leaf_tokens(_diag_doc.leaf_token_ids[j], device=device)
                        for j in range(len(_diag_doc.leaf_token_ids))
                    ]
                else:
                    leaf_feats = _to_device(_diag_doc.leaf_features, device=device)
                    states = [model.encode_leaf(x) for x in leaf_feats]
                _root, merge_states = model._merge_states(
                    states,
                    schedule="balanced",
                    collect_merge_states=True,
                )

                # Track merge span sizes (in leaves) in the same order as `merge_states`.
                cur_sizes = [1 for _ in range(len(states))]
                merge_sizes: List[int] = []
                while len(cur_sizes) > 1:
                    nxt_sizes: List[int] = []
                    i = 0
                    while i < len(cur_sizes):
                        if i + 1 >= len(cur_sizes):
                            nxt_sizes.append(int(cur_sizes[i]))
                            i += 1
                            continue
                        parent = int(cur_sizes[i]) + int(cur_sizes[i + 1])
                        merge_sizes.append(int(parent))
                        nxt_sizes.append(int(parent))
                        i += 2
                    cur_sizes = nxt_sizes

                _diag_merge_counts = _diag_doc.merge_counts_balanced
                for idx, st in enumerate(merge_states):
                    if idx >= len(_diag_merge_counts) or idx >= len(merge_sizes):
                        break
                    pred = float(model.predict_count_from_state(st).detach().cpu().numpy())
                    truth = float(_diag_merge_counts[idx])
                    diag_errs.append(abs(pred - truth))
                    diag_scores.append(float(merge_sizes[idx]))
                    diag_preds.append(float(pred))

    errs = np.asarray(diag_errs, dtype=np.float64)
    scores = np.asarray(diag_scores, dtype=np.float64)
    preds = np.asarray(diag_preds, dtype=np.float64)
    if errs.size > 0:
        scaled = base * (scores / float(np.mean(scores)))
        pi = np.clip(scaled, pi_min, 1.0)
        pred_mu = float(np.mean(preds))
        pred_err_proxy = np.abs(preds - pred_mu)
        diagnostics = _audit_estimator_diagnostics(
            errs.tolist(),
            pred_err_proxy.tolist(),
            pi.tolist(),
            trials=400,
            seed=int(seeds["effective_model_seed"]) + 991,
        )
    else:
        pi = np.full((0,), 1.0, dtype=np.float64)
        diagnostics = EstimatorDiagnostics(
            true_mean=0.0,
            naive_bias=0.0,
            ipw_bias=0.0,
            dsl_bias=0.0,
            ipw_var=0.0,
            dsl_var=0.0,
        )
    guided_eval_curve: Dict[str, object] = {
        "present": False,
        "include_root": bool(config.eval_guidance_include_root),
        "trials": int(config.eval_guidance_trials),
        "points": [],
    }
    # Guidance eval requires feature-based leaf encoding (encode_leaf), so it is
    # not available for FNO models which use token-based encoding (encode_leaf_tokens).
    if (
        train_prepped
        and int(config.eval_guidance_trials) > 0
        and tuple(config.eval_guidance_qs)
        and not _is_fno
    ):
        assert "model" in locals()
        guided_eval_curve = _eval_guided_model_curve(
            model,
            test_prepped,
            device=device,
            tau=float(config.violation_tau),
            guidance_qs=tuple(config.eval_guidance_qs),
            guidance_trials=int(config.eval_guidance_trials),
            guidance_include_root=bool(config.eval_guidance_include_root),
            guidance_override_mode=str(config.guidance_override_mode),
            guidance_seed=int(seeds["effective_model_seed"])
            + int(config.eval_guidance_seed_offset),
        )
        points = list(guided_eval_curve.get("points") or [])
        q0_pts = [p for p in points if abs(float(p.get("q", float("nan"))) - 0.0) <= 1e-12]
        if q0_pts:
            q0_root = float(q0_pts[0].get("root_mae", float("nan")))
            if math.isfinite(q0_root) and abs(float(q0_root) - float(learned.root_mae)) > 1e-12:
                raise ValueError(
                    "guided_eval_curve invariant failed: q=0 root_mae must match learned root_mae "
                    f"(got {q0_root} vs {learned.root_mae})"
                )
        if bool(config.eval_guidance_include_root):
            q1_pts = [p for p in points if abs(float(p.get("q", float("nan"))) - 1.0) <= 1e-12]
            if q1_pts:
                q1_root = float(q1_pts[0].get("root_mae", float("nan")))
                if (not math.isfinite(q1_root)) or q1_root > 1e-12:
                    raise ValueError(
                        "guided_eval_curve invariant failed: q=1 root_mae must be <=1e-12 "
                        f"when eval_guidance_include_root is true (got {q1_root})"
                    )

    full_tree_ipw_train: Dict[str, Any] | None = None
    full_tree_ipw_val: Dict[str, Any] | None = None
    full_tree_ipw_test: Dict[str, Any] | None = None
    if (
        _is_fno
        and str(normalized_local_law_objective_mode) == LOCAL_LAW_OBJECTIVE_SAMPLED_IPW
    ):
        full_tree_ipw_train = _eval_fno_full_tree_ipw_metrics(
            model,
            fno_tree_train_regular,
            device=device,
            leaf_sample_rate=float(config.ipw_leaf_sample_rate),
            internal_sample_rate=float(config.ipw_internal_sample_rate),
            use_residual_decomposition=bool(config.use_residual_decomposition),
            seed=int(seeds["effective_model_seed"]) + 11_003,
        )
        full_tree_ipw_val = _eval_fno_full_tree_ipw_metrics(
            model,
            fno_tree_val_regular,
            device=device,
            leaf_sample_rate=float(config.ipw_leaf_sample_rate),
            internal_sample_rate=float(config.ipw_internal_sample_rate),
            use_residual_decomposition=bool(config.use_residual_decomposition),
            seed=int(seeds["effective_model_seed"]) + 12_003,
        )
        full_tree_ipw_test = _eval_fno_full_tree_ipw_metrics(
            model,
            fno_tree_test_regular,
            device=device,
            leaf_sample_rate=float(config.ipw_leaf_sample_rate),
            internal_sample_rate=float(config.ipw_internal_sample_rate),
            use_residual_decomposition=bool(config.use_residual_decomposition),
            seed=int(seeds["effective_model_seed"]) + 13_003,
        )

    learned_payload = {
        **asdict(learned),
        **_sketch_metric_alias_payload(learned),
        "train_loss_final": float(train_loss_final.train_loss_final),
        "train_loss_curve": [float(x) for x in train_loss_final.train_loss_curve],
        "epochs_completed": int(train_loss_final.epochs_completed),
        "training_selection_metric_curve": [
            float(x) for x in train_loss_final.selection_metric_curve
        ],
        "training_selection_mode": str(train_loss_final.selection_mode),
        "training_selection_split": str(train_loss_final.selection_split),
        "training_selection_metric_name": str(train_loss_final.selection_metric_name),
        "training_selection_metric_value": float(train_loss_final.selection_metric_value),
        "training_selection_best_epoch": int(train_loss_final.best_epoch),
        "train_root_mae": float(learned_train.root_mae),
        "train_leaf_mae": float(learned_train.leaf_mae),
        "train_c2_count_drift_r1_mae": float(learned_train.c2_count_drift_r1_mae),
        "train_c2_count_drift_r2_mae": float(learned_train.c2_count_drift_r2_mae),
        "train_c2_count_drift_r4_mae": float(learned_train.c2_count_drift_r4_mae),
        "train_c2_root_count_drift_r1_mae": float(
            learned_train.c2_root_count_drift_r1_mae
        ),
        "train_c2_root_count_drift_r2_mae": float(
            learned_train.c2_root_count_drift_r2_mae
        ),
        "train_c2_root_count_drift_r4_mae": float(
            learned_train.c2_root_count_drift_r4_mae
        ),
        "train_c2_idempotence_mae": float(learned_train.c2_idempotence_mae),
        "train_c2_r1_mae": float(learned_train.c2_r1_mae),
        "train_c2_r2_mae": float(learned_train.c2_r2_mae),
        "train_c2_r4_mae": float(learned_train.c2_r4_mae),
        "train_merge_mae": float(learned_train.merge_mae),
        "train_schedule_spread_mean": float(learned_train.schedule_spread_mean),
        "train_resummary_root_drift_r1": float(learned_train.resummary_root_drift_r1),
        "train_resummary_root_drift_r2": float(learned_train.resummary_root_drift_r2),
        "train_resummary_root_drift_r4": float(learned_train.resummary_root_drift_r4),
        **_objective_with_split_prefix(train_weighted_objective, prefix="train"),
        **_objective_estimator_with_split_prefix(train_objective_estimators, prefix="train"),
        "train_theorem_score": float(
            markov_theorem_score(
                leaf=float(learned_train.leaf_mae),
                merge=float(learned_train.merge_mae),
                spread=float(learned_train.schedule_spread_mean),
            )
        ),
        "train_theorem_bundle_score_n": float(
            markov_law_bundle_score(
                c1=float(learned_train.leaf_mae) / float(target_scale),
                c2=float(learned_train.c2_count_drift_r1_mae) / float(target_scale),
                c3=float(learned_train.merge_mae) / float(target_scale),
            )
        ),
        **_metrics_with_split_prefix(learned_val, prefix="val", target_scale=target_scale),
        **_metrics_with_split_prefix(learned, prefix="test", target_scale=target_scale),
        **_objective_with_split_prefix(val_weighted_objective, prefix="val"),
        **_objective_with_split_prefix(test_weighted_objective, prefix="test"),
        **_objective_estimator_with_split_prefix(val_objective_estimators, prefix="val"),
        **_objective_estimator_with_split_prefix(test_objective_estimators, prefix="test"),
        "val_theorem_score": float(
            markov_theorem_score(
                leaf=float(learned_val.leaf_mae),
                merge=float(learned_val.merge_mae),
                spread=float(learned_val.schedule_spread_mean),
            )
        ),
        "test_theorem_score": float(
            markov_theorem_score(
                leaf=float(learned.leaf_mae),
                merge=float(learned.merge_mae),
                spread=float(learned.schedule_spread_mean),
            )
        ),
        "val_theorem_bundle_score_n": float(
            markov_law_bundle_score(
                c1=float(learned_val.leaf_mae) / float(target_scale),
                c2=float(learned_val.c2_count_drift_r1_mae) / float(target_scale),
                c3=float(learned_val.merge_mae) / float(target_scale),
            )
        ),
        "test_theorem_bundle_score_n": float(
            markov_law_bundle_score(
                c1=float(learned.leaf_mae) / float(target_scale),
                c2=float(learned.c2_count_drift_r1_mae) / float(target_scale),
                c3=float(learned.merge_mae) / float(target_scale),
            )
        ),
        "generalization_gap_optimization_objective_full_labels": float(
            test_weighted_objective.optimization_total_loss
            - train_weighted_objective.optimization_total_loss
        ),
        "generalization_gap_unweighted_objective_full_labels": float(
            test_weighted_objective.raw_total_loss - train_weighted_objective.raw_total_loss
        ),
        "generalization_gap_objective_full_labels": float(
            test_weighted_objective.optimization_total_loss
            - train_weighted_objective.optimization_total_loss
        ),
        "generalization_gap_root_mae": float(learned.root_mae - learned_train.root_mae),
        "generalization_gap_leaf_mae": float(learned.leaf_mae - learned_train.leaf_mae),
        "generalization_gap_c2_count_drift_r1_mae": float(
            learned.c2_count_drift_r1_mae - learned_train.c2_count_drift_r1_mae
        ),
        "generalization_gap_c2_idempotence_mae": float(
            learned.c2_idempotence_mae - learned_train.c2_idempotence_mae
        ),
        "generalization_gap_merge_mae": float(learned.merge_mae - learned_train.merge_mae),
        "generalization_gap_schedule_spread_mean": float(
            learned.schedule_spread_mean - learned_train.schedule_spread_mean
        ),
        "generalization_gap_resummary_root_drift_r2": float(
            learned.resummary_root_drift_r2 - learned_train.resummary_root_drift_r2
        ),
        "gap_to_exact_root_mae": float(learned.root_mae - exact.root_mae),
        "gap_to_exact_leaf_mae": float(learned.leaf_mae - exact.leaf_mae),
        "gap_to_exact_c2_count_drift_r1_mae": float(
            learned.c2_count_drift_r1_mae - exact.c2_count_drift_r1_mae
        ),
        "gap_to_exact_c2_idempotence_mae": float(
            learned.c2_idempotence_mae - exact.c2_idempotence_mae
        ),
        "gap_to_exact_merge_mae": float(learned.merge_mae - exact.merge_mae),
        "gap_to_exact_schedule_spread_mean": float(
            learned.schedule_spread_mean - exact.schedule_spread_mean
        ),
        "gap_to_undersupported_root_mae": float(learned.root_mae - undersupported.root_mae),
        "gap_to_undersupported_leaf_mae": float(learned.leaf_mae - undersupported.leaf_mae),
        "gap_to_undersupported_c2_count_drift_r1_mae": float(
            learned.c2_count_drift_r1_mae - undersupported.c2_count_drift_r1_mae
        ),
        "gap_to_undersupported_c2_idempotence_mae": float(
            learned.c2_idempotence_mae - undersupported.c2_idempotence_mae
        ),
        "gap_to_undersupported_merge_mae": float(learned.merge_mae - undersupported.merge_mae),
        "gap_to_undersupported_schedule_spread_mean": float(
            learned.schedule_spread_mean - undersupported.schedule_spread_mean
        ),
    }
    if full_tree_ipw_train is not None and full_tree_ipw_val is not None and full_tree_ipw_test is not None:
        learned_payload.update(
            {
                **_full_tree_ipw_with_split_prefix(full_tree_ipw_train, prefix="train"),
                **_full_tree_ipw_with_split_prefix(full_tree_ipw_val, prefix="val"),
                **_full_tree_ipw_with_split_prefix(full_tree_ipw_test, prefix="test"),
                "train_full_tree_ipw": dict(full_tree_ipw_train),
                "val_full_tree_ipw": dict(full_tree_ipw_val),
                "test_full_tree_ipw": dict(full_tree_ipw_test),
                "full_tree_ipw_enabled": True,
                "full_tree_ipw_leaf_sample_rate": float(config.ipw_leaf_sample_rate),
                "full_tree_ipw_internal_sample_rate": float(config.ipw_internal_sample_rate),
                "full_tree_ipw_local_law_objective_mode": str(
                    normalized_local_law_objective_mode
                ),
                "full_tree_ipw_use_residual_decomposition": bool(
                    config.use_residual_decomposition
                ),
            }
        )
    # DPO preference gap (Lean: dpo_gap_bounded in OPT/PreferenceBounds.lean).
    learned_payload["dpo_preference_gap"] = dpo_gap_result
    learned_payload["doc_sequence_train_fraction"] = float(config.doc_sequence_train_fraction)
    learned_payload["doc_sequence_train_docs_used"] = int(doc_sequence_train_docs_used)
    learned_payload["local_law_objective_mode"] = str(normalized_local_law_objective_mode)
    learned_payload["tree_root_supervision_kind"] = str(
        config.tree_root_supervision_kind
    )
    learned_payload["tree_exact_collapse_mode"] = str(
        config.tree_exact_collapse_mode
    )
    learned_payload["tree_model_version"] = str(
        getattr(config, "tree_model_version", "")
    )
    if _is_fno:
        learned_payload.update(
            {
                "tree_runtime_merge_kind": str(
                    getattr(model, "runtime_merge_kind", "")
                ),
                "tree_exact_projected_merge_is_runtime_merge": bool(
                    getattr(
                        model,
                        "exact_projected_merge_is_runtime_merge",
                        False,
                    )
                ),
                "uses_unified_g_learned_merge": bool(
                    getattr(model, "uses_unified_g_learned_merge", False)
                ),
            }
        )
    from treepo._research.ctreepo.sim.core.fno_arch_config import resolve_fno_arch
    _fno_arch_payload = resolve_fno_arch(config)
    learned_payload["tree_leaf_fno_width"] = _fno_arch_payload.width
    learned_payload["tree_leaf_fno_n_modes"] = _fno_arch_payload.n_modes
    learned_payload["tree_leaf_fno_n_layers"] = _fno_arch_payload.n_layers
    if root_only_view_train is not None and root_only_view_val is not None and root_only_view_test is not None:
        learned_payload.update(
            {
                "train_root_only_view_root_mae": float(root_only_view_train.root_mae),
                "train_root_only_view_leaf_mae": float(root_only_view_train.leaf_mae),
                "train_root_only_view_merge_mae": float(root_only_view_train.merge_mae),
                "val_root_only_view_root_mae": float(root_only_view_val.root_mae),
                "val_root_only_view_leaf_mae": float(root_only_view_val.leaf_mae),
                "val_root_only_view_merge_mae": float(root_only_view_val.merge_mae),
                "test_root_only_view_root_mae": float(root_only_view_test.root_mae),
                "test_root_only_view_leaf_mae": float(root_only_view_test.leaf_mae),
                "test_root_only_view_merge_mae": float(root_only_view_test.merge_mae),
                "root_only_train_fraction": float(config.root_only_train_fraction),
            }
        )
    if (
        doc_sequence_view_train is not None
        and doc_sequence_view_val is not None
        and doc_sequence_view_test is not None
    ):
        learned_payload.update(
            {
                "train_doc_sequence_view_root_mae": float(doc_sequence_view_train.root_mae),
                "train_doc_sequence_view_leaf_mae": float(doc_sequence_view_train.leaf_mae),
                "train_doc_sequence_view_merge_mae": float(doc_sequence_view_train.merge_mae),
                "val_doc_sequence_view_root_mae": float(doc_sequence_view_val.root_mae),
                "val_doc_sequence_view_leaf_mae": float(doc_sequence_view_val.leaf_mae),
                "val_doc_sequence_view_merge_mae": float(doc_sequence_view_val.merge_mae),
                "test_doc_sequence_view_root_mae": float(doc_sequence_view_test.root_mae),
                "test_doc_sequence_view_leaf_mae": float(doc_sequence_view_test.leaf_mae),
                "test_doc_sequence_view_merge_mae": float(doc_sequence_view_test.merge_mae),
            }
        )
    metrics: Dict[str, object] = {
        "exact": asdict(exact),
        "leaf_bucket": asdict(leaf_bucket),
        "undersupported": asdict(undersupported),
        "flip_R1": asdict(flip_r1),
        "flip_R2": asdict(flip_r2),
        "learned_train": asdict(learned_train),
        "learned_val": asdict(learned_val),
        "learned_test": learned_payload,
        "learned": learned_payload,
        "guided_eval_curve": guided_eval_curve,
    }
    if root_only_view_train is not None and root_only_view_val is not None and root_only_view_test is not None:
        metrics["learned_root_only_view_train"] = asdict(root_only_view_train)
        metrics["learned_root_only_view_val"] = asdict(root_only_view_val)
        metrics["learned_root_only_view_test"] = asdict(root_only_view_test)
    if (
        doc_sequence_view_train is not None
        and doc_sequence_view_val is not None
        and doc_sequence_view_test is not None
    ):
        metrics["learned_doc_sequence_view_train"] = asdict(doc_sequence_view_train)
        metrics["learned_doc_sequence_view_val"] = asdict(doc_sequence_view_val)
        metrics["learned_doc_sequence_view_test"] = asdict(doc_sequence_view_test)
    if rf_root is not None:
        rf_root_dict = asdict(rf_root)
        rf_root_dict["_note"] = (
            "rf_root uses mean/std of tree-leaf features (regime one-hots, "
            "transition matrices, span lengths), NOT the token sequence. "
            "Not comparable to token-sequence baselines like doc_level_ridge."
        )
        metrics["rf_root"] = rf_root_dict
    if rf_root_val is not None:
        metrics["rf_root_val"] = asdict(rf_root_val)
    if doc_level is not None:
        metrics["doc_level"] = asdict(doc_level)
    if doc_level_val is not None:
        metrics["doc_level_val"] = asdict(doc_level_val)
    if doc_level_train is not None:
        metrics["doc_level_train"] = asdict(doc_level_train)
    if doc_level_fit is not None:
        metrics["doc_level_training"] = {
            "train_loss_final": float(doc_level_fit.train_loss_final),
            "train_loss_curve": [float(x) for x in doc_level_fit.train_loss_curve],
            "epochs_completed": int(doc_level_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in doc_level_fit.selection_metric_curve
            ],
            "selection_mode": str(doc_level_fit.selection_mode),
            "selection_split": str(doc_level_fit.selection_split),
            "selection_metric_name": str(doc_level_fit.selection_metric_name),
            "selection_metric_value": float(doc_level_fit.selection_metric_value),
            "best_epoch": int(doc_level_fit.best_epoch),
            "baseline_family": str(config.model_family),
            "input_view": "single_full_document_leaf",
            "uses_tree_merges": False,
            "supervision_artifact_path": doc_level_supervision_artifact,
            "train_docs": int(config.train_docs),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family=OPTIMIZER_FAMILY_GRADIENT_DENSE,
                optimizer_backend="torch_mlp",
                selection_mode=str(doc_level_fit.selection_mode),
                selection_split=str(doc_level_fit.selection_split),
                selection_metric_name=str(doc_level_fit.selection_metric_name),
                selection_metric_value=float(doc_level_fit.selection_metric_value),
                best_epoch=int(doc_level_fit.best_epoch),
                n_train_rows=int(doc_level_supervision_rows),
            ),
        }
    if doc_sequence is not None:
        metrics["doc_sequence"] = asdict(doc_sequence)
    if doc_sequence_val is not None:
        metrics["doc_sequence_val"] = asdict(doc_sequence_val)
    if doc_sequence_train is not None:
        metrics["doc_sequence_train"] = asdict(doc_sequence_train)
    if doc_sequence_fit is not None:
        metrics["doc_sequence_training"] = {
            "train_loss_final": float(doc_sequence_fit.train_loss_final),
            "train_loss_curve": [float(x) for x in doc_sequence_fit.train_loss_curve],
            "epochs_completed": int(doc_sequence_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in doc_sequence_fit.selection_metric_curve
            ],
            "selection_mode": str(doc_sequence_fit.selection_mode),
            "selection_split": str(doc_sequence_fit.selection_split),
            "selection_metric_name": str(doc_sequence_fit.selection_metric_name),
            "selection_metric_value": float(doc_sequence_fit.selection_metric_value),
            "best_epoch": int(doc_sequence_fit.best_epoch),
            "train_exact_match_rate": float(doc_sequence_fit.train_exact_match_rate),
            "val_exact_match_rate": float(doc_sequence_fit.val_exact_match_rate),
            "test_exact_match_rate": float(doc_sequence_fit.test_exact_match_rate),
            "baseline_family": "official_neuraloperator_fno",
            "input_view": "full_document_token_sequence",
            "uses_tree_merges": False,
            "train_docs": int(config.train_docs),
            "operator_backend": "official_neuraloperator_package",
            "token_embedding_backend": "learned_token_embedding",
            "readout_mode": "count_support_classification",
            "root_summary_auxiliary_heads": [],
            "root_label_only_supervision": True,
            "doc_sequence_objective_requested": str(config.doc_sequence_objective),
            "doc_sequence_objective_effective": "count_ce_only",
            "fno_n_layers": 4,
            "sequence_input_backend": str(config_payload["full_sequence_input_backend"]),
            "sequence_input_signatures": dict(config_payload["full_sequence_input_signatures"]),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family=OPTIMIZER_FAMILY_GRADIENT_DENSE,
                optimizer_backend="official_neuraloperator_fno_count_classifier",
                selection_mode=str(doc_sequence_fit.selection_mode),
                selection_split=str(doc_sequence_fit.selection_split),
                selection_metric_name=str(doc_sequence_fit.selection_metric_name),
                selection_metric_value=float(doc_sequence_fit.selection_metric_value),
                best_epoch=int(doc_sequence_fit.best_epoch),
                n_train_rows=int(config.train_docs),
            ),
        }
    if doc_transformer is not None:
        metrics["doc_transformer"] = asdict(doc_transformer)
    if doc_transformer_val is not None:
        metrics["doc_transformer_val"] = asdict(doc_transformer_val)
    if doc_transformer_train is not None:
        metrics["doc_transformer_train"] = asdict(doc_transformer_train)
    if doc_transformer_fit is not None:
        metrics["doc_transformer_training"] = {
            "train_loss_final": float(doc_transformer_fit.train_loss_final),
            "train_loss_curve": [float(x) for x in doc_transformer_fit.train_loss_curve],
            "epochs_completed": int(doc_transformer_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in doc_transformer_fit.selection_metric_curve
            ],
            "selection_mode": str(doc_transformer_fit.selection_mode),
            "selection_split": str(doc_transformer_fit.selection_split),
            "selection_metric_name": str(doc_transformer_fit.selection_metric_name),
            "selection_metric_value": float(doc_transformer_fit.selection_metric_value),
            "best_epoch": int(doc_transformer_fit.best_epoch),
            "train_exact_match_rate": float(doc_transformer_fit.train_exact_match_rate),
            "val_exact_match_rate": float(doc_transformer_fit.val_exact_match_rate),
            "test_exact_match_rate": float(doc_transformer_fit.test_exact_match_rate),
            "baseline_family": "full_sequence_boundary_transformer",
            "input_view": "full_document_token_sequence",
            "uses_tree_merges": False,
            "train_docs": int(config.train_docs),
            "token_embedding_backend": "learned_token_embedding",
            "position_embedding_backend": "learned_position_embedding",
            "readout_mode": (
                "pooled_count_support_classification"
                if str(config.doc_transformer_head_family) == "pooled_count_classifier"
                else "summed_boundary_probabilities_with_count_classification"
            ),
            "root_summary_auxiliary_heads": ["count_class"]
            if str(config.doc_transformer_head_family) == "boundary_sum_count_hybrid"
            else [],
            "root_label_only_supervision": True,
            "doc_transformer_head_family": str(config.doc_transformer_head_family),
            "doc_transformer_layers": int(config.doc_transformer_layers)
            if int(config.doc_transformer_layers) > 0
            else (4 if int(max(64, min(512, max(int(config.state_dim), int(config.hidden_dim) // 2)))) >= 128 else 3),
            "sequence_input_backend": str(config_payload["full_sequence_input_backend"]),
            "sequence_input_signatures": dict(config_payload["full_sequence_input_signatures"]),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family=OPTIMIZER_FAMILY_GRADIENT_DENSE,
                optimizer_backend=(
                    "full_sequence_boundary_transformer_count_classifier"
                    if str(config.doc_transformer_head_family) == "pooled_count_classifier"
                    else "full_sequence_boundary_transformer_regression_count_classifier"
                ),
                selection_mode=str(doc_transformer_fit.selection_mode),
                selection_split=str(doc_transformer_fit.selection_split),
                selection_metric_name=str(doc_transformer_fit.selection_metric_name),
                selection_metric_value=float(doc_transformer_fit.selection_metric_value),
                best_epoch=int(doc_transformer_fit.best_epoch),
                n_train_rows=int(config.train_docs),
            ),
        }
    # Neural operator comparison baselines (FNO, DeepONet, MLP bigram, CNN1D).
    for _bl_label, _bl_test, _bl_val, _bl_train, _bl_fit in [
        ("fno", fno_test, fno_val, fno_train, fno_fit),
        ("deeponet", deeponet_test, deeponet_val, deeponet_train, deeponet_fit),
        ("mlp_bigram", mlp_bigram_test, mlp_bigram_val, mlp_bigram_train, mlp_bigram_fit),
        ("cnn1d", cnn1d_test, cnn1d_val, cnn1d_train, cnn1d_fit),
    ]:
        if _bl_test is not None:
            metrics[_bl_label] = asdict(_bl_test)
        if _bl_val is not None:
            metrics[f"{_bl_label}_val"] = asdict(_bl_val)
        if _bl_train is not None:
            metrics[f"{_bl_label}_train"] = asdict(_bl_train)
        if _bl_fit is not None:
            metrics[f"{_bl_label}_training"] = {
                "train_loss_final": float(_bl_fit.train_loss_final),
                "train_loss_curve": [float(x) for x in _bl_fit.train_loss_curve],
                "epochs_completed": int(_bl_fit.epochs_completed),
                "selection_metric_curve": [float(x) for x in _bl_fit.selection_metric_curve],
                "selection_mode": str(_bl_fit.selection_mode),
                "selection_split": str(_bl_fit.selection_split),
                "selection_metric_name": str(_bl_fit.selection_metric_name),
                "selection_metric_value": float(_bl_fit.selection_metric_value),
                "best_epoch": int(_bl_fit.best_epoch),
                "train_exact_match_rate": float(_bl_fit.train_exact_match_rate),
                "val_exact_match_rate": float(_bl_fit.val_exact_match_rate),
                "test_exact_match_rate": float(_bl_fit.test_exact_match_rate),
                "baseline_family": _bl_label,
                "input_view": "full_document_token_sequence",
                "uses_tree_merges": False,
                "root_label_only_supervision": True,
                "train_docs": int(config.train_docs),
            }
    if doc_level_ridge is not None:
        metrics["doc_level_ridge"] = asdict(doc_level_ridge)
    if doc_level_ridge_val is not None:
        metrics["doc_level_ridge_val"] = asdict(doc_level_ridge_val)
    if doc_level_ridge_train is not None:
        metrics["doc_level_ridge_train"] = asdict(doc_level_ridge_train)
    if doc_level_ridge_fit is not None:
        metrics["doc_level_ridge_training"] = {
            "train_loss_final": float(doc_level_ridge_fit.train_loss_final),
            "train_loss_curve": [float(x) for x in doc_level_ridge_fit.train_loss_curve],
            "epochs_completed": int(doc_level_ridge_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in doc_level_ridge_fit.selection_metric_curve
            ],
            "selection_mode": str(doc_level_ridge_fit.selection_mode),
            "selection_split": str(doc_level_ridge_fit.selection_split),
            "selection_metric_name": str(doc_level_ridge_fit.selection_metric_name),
            "selection_metric_value": float(doc_level_ridge_fit.selection_metric_value),
            "best_epoch": int(doc_level_ridge_fit.best_epoch),
            "baseline_family": "ridge",
            "input_view": "single_full_document_leaf",
            "uses_tree_merges": False,
            "supervision_artifact_path": doc_level_supervision_artifact,
            "train_docs": int(config.train_docs),
            "ridge_alpha": float(config.doc_level_ridge_alpha),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family=OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
                optimizer_backend="closed_form_ridge",
                selection_mode=str(doc_level_ridge_fit.selection_mode),
                selection_split=str(doc_level_ridge_fit.selection_split),
                selection_metric_name=str(doc_level_ridge_fit.selection_metric_name),
                selection_metric_value=float(doc_level_ridge_fit.selection_metric_value),
                best_epoch=int(doc_level_ridge_fit.best_epoch),
                n_train_rows=int(doc_level_supervision_rows),
            ),
        }
    for label, payload in sorted(doc_level_ridge_breakdown_test.items()):
        metrics[f"doc_level_ridge_{label}"] = asdict(payload)
    for label, payload in sorted(doc_level_ridge_breakdown_val.items()):
        metrics[f"doc_level_ridge_{label}_val"] = asdict(payload)
    for label, payload in sorted(doc_level_ridge_breakdown_train.items()):
        metrics[f"doc_level_ridge_{label}_train"] = asdict(payload)
    for label, fit_payload in sorted(doc_level_ridge_breakdown_fit.items()):
        metrics[f"doc_level_ridge_{label}_training"] = {
            "train_loss_final": float(fit_payload.train_loss_final),
            "train_loss_curve": [float(x) for x in fit_payload.train_loss_curve],
            "epochs_completed": int(fit_payload.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in fit_payload.selection_metric_curve
            ],
            "selection_mode": str(fit_payload.selection_mode),
            "selection_split": str(fit_payload.selection_split),
            "selection_metric_name": str(fit_payload.selection_metric_name),
            "selection_metric_value": float(fit_payload.selection_metric_value),
            "best_epoch": int(fit_payload.best_epoch),
            "baseline_family": "ridge",
            "input_view": f"full_document_token_{label}_counts",
            "uses_tree_merges": False,
            "train_docs": int(config.train_docs),
            "ridge_alpha": float(config.doc_level_ridge_alpha),
            "ngram_orders": [1 if label == 'unigram' else 2 if label == 'bigram' else 3 if label == 'trigram' else label],
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family=OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
                optimizer_backend="closed_form_ridge",
                selection_mode=str(fit_payload.selection_mode),
                selection_split=str(fit_payload.selection_split),
                selection_metric_name=str(fit_payload.selection_metric_name),
                selection_metric_value=float(fit_payload.selection_metric_value),
                best_epoch=int(fit_payload.best_epoch),
                n_train_rows=int(config.train_docs),
            ),
        }
    if leaf_ridge_tree is not None:
        metrics["leaf_ridge_tree"] = asdict(leaf_ridge_tree)
    if leaf_ridge_tree_val is not None:
        metrics["leaf_ridge_tree_val"] = asdict(leaf_ridge_tree_val)
    if leaf_ridge_tree_train is not None:
        metrics["leaf_ridge_tree_train"] = asdict(leaf_ridge_tree_train)
    if leaf_ridge_tree_fit is not None:
        metrics["leaf_ridge_tree_training"] = {
            "train_loss_final": float(leaf_ridge_tree_fit.train_loss_final),
            "train_loss_curve": [float(x) for x in leaf_ridge_tree_fit.train_loss_curve],
            "epochs_completed": int(leaf_ridge_tree_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in leaf_ridge_tree_fit.selection_metric_curve
            ],
            "selection_mode": str(leaf_ridge_tree_fit.selection_mode),
            "selection_split": str(leaf_ridge_tree_fit.selection_split),
            "selection_metric_name": str(leaf_ridge_tree_fit.selection_metric_name),
            "selection_metric_value": float(leaf_ridge_tree_fit.selection_metric_value),
            "best_epoch": int(leaf_ridge_tree_fit.best_epoch),
            "baseline_family": "leaf_ridge_tree",
            "input_view": "sampled_leaf_core_features",
            "uses_tree_merges": True,
            "supervision_artifact_path": leaf_ridge_tree_supervision_artifact,
            "train_docs": int(config.train_docs),
            "leaf_query_rate": float(config.leaf_query_rate),
            "ridge_alpha": float(config.doc_level_ridge_alpha),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family=OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
                optimizer_backend="closed_form_ridge",
                selection_mode=str(leaf_ridge_tree_fit.selection_mode),
                selection_split=str(leaf_ridge_tree_fit.selection_split),
                selection_metric_name=str(leaf_ridge_tree_fit.selection_metric_name),
                selection_metric_value=float(leaf_ridge_tree_fit.selection_metric_value),
                best_epoch=int(leaf_ridge_tree_fit.best_epoch),
                n_train_rows=int(leaf_ridge_tree_supervision_rows),
            ),
        }
    if leaf_endpoint_table_tree is not None:
        metrics["leaf_endpoint_table_tree"] = asdict(leaf_endpoint_table_tree)
    if leaf_endpoint_table_tree_val is not None:
        metrics["leaf_endpoint_table_tree_val"] = asdict(leaf_endpoint_table_tree_val)
    if leaf_endpoint_table_tree_train is not None:
        metrics["leaf_endpoint_table_tree_train"] = asdict(leaf_endpoint_table_tree_train)
    if leaf_endpoint_table_tree_fit is not None:
        metrics["leaf_endpoint_table_tree_training"] = {
            "train_loss_final": float(leaf_endpoint_table_tree_fit.train_loss_final),
            "train_loss_curve": [
                float(x) for x in leaf_endpoint_table_tree_fit.train_loss_curve
            ],
            "epochs_completed": int(leaf_endpoint_table_tree_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in leaf_endpoint_table_tree_fit.selection_metric_curve
            ],
            "selection_mode": str(leaf_endpoint_table_tree_fit.selection_mode),
            "selection_split": str(leaf_endpoint_table_tree_fit.selection_split),
            "selection_metric_name": str(leaf_endpoint_table_tree_fit.selection_metric_name),
            "selection_metric_value": float(
                leaf_endpoint_table_tree_fit.selection_metric_value
            ),
            "best_epoch": int(leaf_endpoint_table_tree_fit.best_epoch),
            "baseline_family": "leaf_endpoint_table_tree",
            "input_view": "sampled_leaf_endpoints_length",
            "uses_tree_merges": True,
            "supervision_artifact_path": leaf_endpoint_table_tree_supervision_artifact,
            "train_docs": int(config.train_docs),
            "leaf_query_rate": float(config.leaf_query_rate),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family="piecewise_local_regression",
                optimizer_backend="endpoint_length_group_mean",
                selection_mode=str(leaf_endpoint_table_tree_fit.selection_mode),
                selection_split=str(leaf_endpoint_table_tree_fit.selection_split),
                selection_metric_name=str(leaf_endpoint_table_tree_fit.selection_metric_name),
                selection_metric_value=float(
                    leaf_endpoint_table_tree_fit.selection_metric_value
                ),
                best_epoch=int(leaf_endpoint_table_tree_fit.best_epoch),
                n_train_rows=int(leaf_endpoint_table_tree_supervision_rows),
            ),
        }
    if leaf_dt_tree is not None:
        metrics["leaf_dt_tree"] = asdict(leaf_dt_tree)
    if leaf_dt_tree_val is not None:
        metrics["leaf_dt_tree_val"] = asdict(leaf_dt_tree_val)
    if leaf_dt_tree_train is not None:
        metrics["leaf_dt_tree_train"] = asdict(leaf_dt_tree_train)
    if leaf_dt_tree_fit is not None:
        metrics["leaf_dt_tree_training"] = {
            "train_loss_final": float(leaf_dt_tree_fit.train_loss_final),
            "train_loss_curve": [float(x) for x in leaf_dt_tree_fit.train_loss_curve],
            "epochs_completed": int(leaf_dt_tree_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in leaf_dt_tree_fit.selection_metric_curve
            ],
            "selection_mode": str(leaf_dt_tree_fit.selection_mode),
            "selection_split": str(leaf_dt_tree_fit.selection_split),
            "selection_metric_name": str(leaf_dt_tree_fit.selection_metric_name),
            "selection_metric_value": float(leaf_dt_tree_fit.selection_metric_value),
            "best_epoch": int(leaf_dt_tree_fit.best_epoch),
            "baseline_family": "leaf_dt_tree",
            "input_view": "sampled_leaf_core_features",
            "uses_tree_merges": True,
            "supervision_artifact_path": leaf_dt_tree_supervision_artifact,
            "train_docs": int(config.train_docs),
            "leaf_query_rate": float(config.leaf_query_rate),
            "tree_max_depth": int(config.rf_max_depth),
            "tree_min_samples_leaf": int(config.rf_min_samples_leaf),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family="tree_regression",
                optimizer_backend="decision_tree_regressor",
                selection_mode=str(leaf_dt_tree_fit.selection_mode),
                selection_split=str(leaf_dt_tree_fit.selection_split),
                selection_metric_name=str(leaf_dt_tree_fit.selection_metric_name),
                selection_metric_value=float(leaf_dt_tree_fit.selection_metric_value),
                best_epoch=int(leaf_dt_tree_fit.best_epoch),
                n_train_rows=int(leaf_dt_tree_supervision_rows),
            ),
        }
    if leaf_knn_tree is not None:
        metrics["leaf_knn_tree"] = asdict(leaf_knn_tree)
    if leaf_knn_tree_val is not None:
        metrics["leaf_knn_tree_val"] = asdict(leaf_knn_tree_val)
    if leaf_knn_tree_train is not None:
        metrics["leaf_knn_tree_train"] = asdict(leaf_knn_tree_train)
    if leaf_knn_tree_fit is not None:
        metrics["leaf_knn_tree_training"] = {
            "train_loss_final": float(leaf_knn_tree_fit.train_loss_final),
            "train_loss_curve": [float(x) for x in leaf_knn_tree_fit.train_loss_curve],
            "epochs_completed": int(leaf_knn_tree_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in leaf_knn_tree_fit.selection_metric_curve
            ],
            "selection_mode": str(leaf_knn_tree_fit.selection_mode),
            "selection_split": str(leaf_knn_tree_fit.selection_split),
            "selection_metric_name": str(leaf_knn_tree_fit.selection_metric_name),
            "selection_metric_value": float(leaf_knn_tree_fit.selection_metric_value),
            "best_epoch": int(leaf_knn_tree_fit.best_epoch),
            "baseline_family": "leaf_knn_tree",
            "input_view": "sampled_leaf_core_features",
            "uses_tree_merges": True,
            "supervision_artifact_path": leaf_knn_tree_supervision_artifact,
            "train_docs": int(config.train_docs),
            "leaf_query_rate": float(config.leaf_query_rate),
            "knn_neighbors": int(config.leaf_knn_neighbors),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family="instance_based_local_regression",
                optimizer_backend="distance_weighted_knn",
                selection_mode=str(leaf_knn_tree_fit.selection_mode),
                selection_split=str(leaf_knn_tree_fit.selection_split),
                selection_metric_name=str(leaf_knn_tree_fit.selection_metric_name),
                selection_metric_value=float(leaf_knn_tree_fit.selection_metric_value),
                best_epoch=int(leaf_knn_tree_fit.best_epoch),
                n_train_rows=int(leaf_knn_tree_supervision_rows),
            ),
        }
    if leaf_rf_tree is not None:
        metrics["leaf_rf_tree"] = asdict(leaf_rf_tree)
    if leaf_rf_tree_val is not None:
        metrics["leaf_rf_tree_val"] = asdict(leaf_rf_tree_val)
    if leaf_rf_tree_train is not None:
        metrics["leaf_rf_tree_train"] = asdict(leaf_rf_tree_train)
    if leaf_rf_tree_fit is not None:
        metrics["leaf_rf_tree_training"] = {
            "train_loss_final": float(leaf_rf_tree_fit.train_loss_final),
            "train_loss_curve": [float(x) for x in leaf_rf_tree_fit.train_loss_curve],
            "epochs_completed": int(leaf_rf_tree_fit.epochs_completed),
            "selection_metric_curve": [
                float(x) for x in leaf_rf_tree_fit.selection_metric_curve
            ],
            "selection_mode": str(leaf_rf_tree_fit.selection_mode),
            "selection_split": str(leaf_rf_tree_fit.selection_split),
            "selection_metric_name": str(leaf_rf_tree_fit.selection_metric_name),
            "selection_metric_value": float(leaf_rf_tree_fit.selection_metric_value),
            "best_epoch": int(leaf_rf_tree_fit.best_epoch),
            "baseline_family": "leaf_rf_tree",
            "input_view": "sampled_leaf_core_features",
            "uses_tree_merges": True,
            "supervision_artifact_path": leaf_rf_tree_supervision_artifact,
            "train_docs": int(config.train_docs),
            "leaf_query_rate": float(config.leaf_query_rate),
            "rf_n_estimators": int(config.rf_n_estimators),
            "rf_max_depth": int(config.rf_max_depth),
            "rf_min_samples_leaf": int(config.rf_min_samples_leaf),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family=OPTIMIZER_FAMILY_TREE_ENSEMBLE,
                optimizer_backend="random_forest_regressor",
                selection_mode=str(leaf_rf_tree_fit.selection_mode),
                selection_split=str(leaf_rf_tree_fit.selection_split),
                selection_metric_name=str(leaf_rf_tree_fit.selection_metric_name),
                selection_metric_value=float(leaf_rf_tree_fit.selection_metric_value),
                best_epoch=int(leaf_rf_tree_fit.best_epoch),
                n_train_rows=int(leaf_rf_tree_supervision_rows),
            ),
        }
    if sampled_leaf_pool_sweep is not None:
        metrics["sampled_leaf_pool_budget_sweep"] = sampled_leaf_pool_sweep
        points = list(sampled_leaf_pool_sweep.get("points", []) or [])
        if len(points) == 1:
            point = dict(points[0] or {})
            if isinstance(point.get("ridge"), dict):
                metrics["sampled_leaf_pool_ridge"] = dict(point["ridge"])
            if isinstance(point.get("ridge_train"), dict):
                metrics["sampled_leaf_pool_ridge_train"] = dict(point["ridge_train"])
            if isinstance(point.get("ridge_val"), dict):
                metrics["sampled_leaf_pool_ridge_val"] = dict(point["ridge_val"])
            if isinstance(point.get("ridge_training"), dict):
                metrics["sampled_leaf_pool_ridge_training"] = dict(point["ridge_training"])
            if isinstance(point.get("rf"), dict):
                metrics["sampled_leaf_pool_rf"] = dict(point["rf"])
            if isinstance(point.get("rf_train"), dict):
                metrics["sampled_leaf_pool_rf_train"] = dict(point["rf_train"])
            if isinstance(point.get("rf_val"), dict):
                metrics["sampled_leaf_pool_rf_val"] = dict(point["rf_val"])
            if isinstance(point.get("rf_training"), dict):
                metrics["sampled_leaf_pool_rf_training"] = dict(point["rf_training"])

    # Keep the old undersupported comparison as a diagnostic only.
    # Canonical cross-DGP law-stress now pairs learned packages with the matched
    # `root_only` baseline across runs in the unified reporting layer.
    from treepo._research.ctreepo.sim.core.law_stress_common import classify_law_stress as _classify_law_stress

    metrics["diagnostic_law_stress_vs_undersupported"] = _classify_law_stress(
        baseline_c1=float(undersupported.leaf_mae),
        baseline_c2=float(undersupported.c2_count_drift_r1_mae),
        baseline_c3=float(undersupported.merge_mae),
        baseline_spread=float(undersupported.schedule_spread_mean),
        baseline_root_mae=float(undersupported.root_mae),
        selected_c1=float(learned.leaf_mae),
        selected_c2=float(learned.c2_count_drift_r1_mae),
        selected_c3=float(learned.merge_mae),
        selected_spread=float(learned.schedule_spread_mean),
        selected_root_mae=float(learned.root_mae),
    ).to_dict()

    # Also expose a compact IPW mean check using TreeSample to tie to the repo's IPW tooling.
    # (We estimate the mean merge-violation rate at threshold tau for the learned-sketch merge error population.)
    tau = float(config.violation_tau)
    if errs.size > 0:
        violations = (errs > tau).astype(np.int64)
        samples: List[TreeSample] = []
        for idx, (v, p) in enumerate(zip(violations.tolist(), pi.tolist())):
            if random.random() < float(p):
                samples.append(
                    TreeSample(
                        doc_id="pop",
                        node_id=str(idx),
                        node_type=NodeType.MERGE,
                        violation=int(v),
                        sampling=SamplingMetadata(
                            unit_propensity=float(p),
                            unit_kind=ObservationUnitKind.MERGE,
                        ),
                    )
                )
        ipw_violation_rate = float(
            horvitz_thompson_mean(samples, lambda s: float(s.violation), float(len(violations)))
            if violations.size
            else 0.0
        )
        metrics["ipw_violation_rate_demo"] = {
            "population_source": "learned_merge_error",
            "tau": float(tau),
            "population": int(len(violations)),
            "sampled": int(len(samples)),
            "ipw_mean_violation": float(ipw_violation_rate),
            "true_mean_violation": float(np.mean(violations.astype(np.float64))),
        }
        # Certificate envelope (Lean: treepo_gap_with_calibration_estimation_clipping).
        # With exact oracle: b_cal = 0.
        if samples:
            cert = compute_certificate(
                samples,
                value_fn=lambda s: float(s.violation),
                delta=0.05,
                w_max=20.0,
                b_cal=0.0,
                value_min=0.0,
                value_max=1.0,
            )
            metrics["certificate_envelope"] = {
                "gap_clip": float(cert.gap_clip),
                "b_cal": float(cert.b_cal),
                "b_est": float(cert.b_est),
                "b_clip": float(cert.b_clip),
                "total_margin": float(cert.total_margin),
                "lean_theorem": str(cert.lean_theorem),
            }

    current_name = str(config.law_package).strip() or (
        "root_only" if float(objective.get("local_law_weight", 0.0)) <= 1e-12 else "learned_g"
    )
    current_role = (
        PolicyRole.BASELINE_G.value
        if str(current_name) == "root_only"
        else PolicyRole.LEARNED_G.value
    )
    if int(config.val_docs) > 0:
        current_selection_metric_name = str(
            learned_payload.get(
                "val_objective_selection_metric_name",
                "configured_objective",
            )
        )
        current_selection_metric_value = float(
            learned_payload.get(
                "val_objective_selection_metric_value",
                learned_payload.get("val_objective_full_labels", float("nan")),
            )
        )
    else:
        current_selection_metric_name = str(train_loss_final.selection_metric_name)
        current_selection_metric_value = float(train_loss_final.selection_metric_value)
    local_law_learnability, g_artifacts = _build_markov_local_law_learnability(
        config=config,
        seeds=seeds,
        target_scale=float(target_scale),
        objective_summary=objective,
        geom=geom,
        exact=exact,
        leaf_bucket=leaf_bucket,
        undersupported=undersupported,
        flip_r2=flip_r2,
        current_name=str(current_name),
        current_role=str(current_role),
        current_train=learned_train,
        current_val=learned_val,
        current_test=learned,
        current_selection_metric_name=str(current_selection_metric_name),
        current_selection_metric=float(current_selection_metric_value),
        current_train_payload=learned_payload,
        current_val_payload=learned_payload,
        current_test_payload=learned_payload,
        model=(model if train_prepped else None),
    )

    return OPSCountSummary(
        config=config_payload,
        training_geometry=asdict(geom),
        objective=objective,
        metrics=metrics,
        estimator_diagnostics={
            **asdict(diagnostics),
            "selection_demo_base_rate": float(base),
            "selection_demo_pi_min": float(pi_min),
            "selection_demo_n_units": float(errs.size),
        },
        local_law_learnability=local_law_learnability,
        g_artifacts=g_artifacts,
    )


__all__ = [
    "OPSCountConfig",
    "OPSCountSummary",
    "MarkovOPSDataBundle",
    "VALID_AUDIT_POLICIES",
    "VALID_C3_AUDIT_STRATEGIES",
    "VALID_DOC_SEQUENCE_OBJECTIVES",
    "VALID_DOC_TRANSFORMER_HEAD_FAMILIES",
    "VALID_EXACT_FAMILIES",
    "VALID_GENERATOR_PROFILES",
    "VALID_LAW_PACKAGES",
    "VALID_LOCAL_LAW_OBJECTIVE_MODES",
    "VALID_SCHEDULES",
    "audit_sample_count",
    "build_markov_changepoint_ops_count_data_bundle",
    "leaf_sample_count",
    "_eval_leaf_bucket_family",
    "run_markov_changepoint_ops_count_experiment",
]
