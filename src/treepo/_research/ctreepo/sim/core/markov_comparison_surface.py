from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any, Dict, Mapping, Sequence

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig
from treepo._research.ctreepo.sim.core.markov_observed_token_policy import (
    resolve_markov_observed_token_policy,
)
from treepo._research.ctreepo.sim.util import safe_float, safe_int


ComparisonModeName = str
VALID_COMPARISON_MODES: tuple[ComparisonModeName, ...] = (
    "legacy",
    "comparable",
    "exact_collapse",
)

MARKOV_COMPARISON_TREE_FAMILIES = frozenset(
    {"tree_neural", "tree_neural_c2", "tree_neural_c2c3"}
)
MARKOV_COMPARISON_OFFICIAL_FNO_FAMILIES = frozenset(
    {"official_fno", "official_fno_sumlen"}
)
FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS = 128

MARKOV_BENCHMARK_LOCKED_FIELDS: tuple[str, ...] = (
    "n_regimes",
    "vocab_size",
    "generator_profile",
    "min_tokens",
    "max_tokens",
    "min_segments",
    "max_segments",
    "min_seg_len",
    "max_seg_len",
    "min_distinct_regimes_per_doc",
    "max_distinct_regimes_per_doc",
    "fixed_leaf_tokens",
)

MARKOV_SHARED_COMPARISON_FIELDS: tuple[str, ...] = (
    "state_dim",
    "hidden_dim",
    "n_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "tree_leaf_fno_width",
    "tree_leaf_fno_n_modes",
    "tree_leaf_fno_n_layers",
    "tree_root_supervision_kind",
    "budget_total_calls_per_doc",
    "full_doc_budget_share",
    "doc_consumption_mode",
    "local_split_mode",
)

MARKOV_COMPARABLE_SURFACE_FIELDS: tuple[str, ...] = (
    *MARKOV_BENCHMARK_LOCKED_FIELDS,
    *MARKOV_SHARED_COMPARISON_FIELDS,
)

_DEFAULT_OPS_CONFIG = OPSCountConfig()


@dataclass(frozen=True)
class MarkovComparableSurface:
    comparison_mode: str
    n_regimes: int
    vocab_size: int
    generator_profile: str
    min_tokens: int
    max_tokens: int
    min_segments: int
    max_segments: int
    min_seg_len: int
    max_seg_len: int
    min_distinct_regimes_per_doc: int | None
    max_distinct_regimes_per_doc: int | None
    fixed_leaf_tokens: int
    state_dim: int
    hidden_dim: int
    n_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    tree_leaf_fno_width: int
    tree_leaf_fno_n_modes: int
    tree_leaf_fno_n_layers: int
    tree_root_supervision_kind: str
    budget_total_calls_per_doc: float
    full_doc_budget_share: float
    doc_consumption_mode: str
    local_split_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mapping_from_config(config: Mapping[str, Any] | OPSCountConfig | None) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return {}


def _normalized_families(families: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(
        str(family).strip()
        for family in list(families or ())
        if str(family).strip()
    )


def normalize_markov_comparison_mode(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "legacy"
    if text not in VALID_COMPARISON_MODES:
        raise ValueError(
            f"comparison_mode={value!r} unsupported; expected one of {VALID_COMPARISON_MODES}"
        )
    return text


def infer_markov_comparison_mode(
    *,
    requested_mode: str | None,
    baseline_families: Sequence[str] | None = None,
    tree_exact_collapse_mode: str | None = None,
) -> str:
    explicit = str(requested_mode or "").strip().lower()
    if explicit:
        return normalize_markov_comparison_mode(explicit)
    if str(tree_exact_collapse_mode or "").strip():
        return "exact_collapse"
    families = set(_normalized_families(baseline_families))
    if (
        families & MARKOV_COMPARISON_TREE_FAMILIES
        and families & MARKOV_COMPARISON_OFFICIAL_FNO_FAMILIES
    ):
        return "comparable"
    return "legacy"


_safe_int = safe_int
_safe_float = safe_float


def _resolve_positive_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    fallback: int,
) -> int:
    value = mapping.get(key, None)
    if value in {"", None}:
        return int(fallback)
    parsed = _safe_int(value, fallback)
    if parsed <= 0:
        return int(fallback)
    return int(parsed)


def resolve_markov_benchmark_locked_fields(
    *,
    benchmark: Any,
    config: Mapping[str, Any] | OPSCountConfig | None = None,
    comparison_mode: str | None = None,
) -> Dict[str, Any]:
    mapping = _mapping_from_config(config)
    mode = normalize_markov_comparison_mode(comparison_mode)
    policy = resolve_markov_observed_token_policy(
        profile_name=str(getattr(benchmark, "observed_token_profile")),
    )
    requested_fixed_leaf_tokens = _safe_int(mapping.get("fixed_leaf_tokens", 0), 0)
    preserve_requested_leaf_tokens = bool(
        mapping.get("preserve_requested_leaf_tokens", False)
        or mapping.get("official_fno_preserve_requested_leaf_tokens", False)
    )
    if (
        mode in {"comparable", "exact_collapse"}
        and int(requested_fixed_leaf_tokens) > 0
    ):
        preserve_requested_leaf_tokens = True
    locked_fields: Dict[str, Any] = {
        "n_regimes": int(policy.n_regimes),
        "vocab_size": int(policy.vocab_size),
        "generator_profile": str(policy.generator_profile),
        "min_tokens": int(policy.min_tokens),
        "max_tokens": int(policy.max_tokens),
        "min_segments": int(policy.min_segments),
        "max_segments": int(policy.max_segments),
        "min_seg_len": int(policy.min_seg_len),
        "max_seg_len": int(policy.max_seg_len),
        "fixed_leaf_tokens": int(policy.fixed_leaf_tokens),
        "min_distinct_regimes_per_doc": getattr(
            policy, "min_distinct_regimes_per_doc", None
        ),
        "max_distinct_regimes_per_doc": getattr(
            policy, "max_distinct_regimes_per_doc", None
        ),
    }
    benchmark_overrides = dict(getattr(benchmark, "config_overrides", {}) or {})
    if preserve_requested_leaf_tokens:
        benchmark_overrides.pop("fixed_leaf_tokens", None)
    for key in tuple(locked_fields):
        if key in benchmark_overrides and benchmark_overrides[key] is not None:
            locked_fields[key] = benchmark_overrides[key]
    if preserve_requested_leaf_tokens and int(requested_fixed_leaf_tokens) > 0:
        locked_fields["fixed_leaf_tokens"] = int(requested_fixed_leaf_tokens)
    return locked_fields


def resolve_markov_comparable_surface(
    *,
    benchmark: Any,
    config: Mapping[str, Any] | OPSCountConfig | None = None,
    comparison_mode: str | None = None,
) -> MarkovComparableSurface:
    mapping = _mapping_from_config(config)
    mode = normalize_markov_comparison_mode(comparison_mode)
    locked = resolve_markov_benchmark_locked_fields(
        benchmark=benchmark,
        config=mapping,
        comparison_mode=mode,
    )
    from treepo._research.ctreepo.sim.core.fno_arch_config import resolve_fno_arch_from_mapping
    _fno_arch_from_mapping = resolve_fno_arch_from_mapping(
        mapping,
        fallback_width=int(_DEFAULT_OPS_CONFIG.fno_width),
        fallback_n_modes=int(_DEFAULT_OPS_CONFIG.fno_n_modes),
        fallback_n_layers=int(_DEFAULT_OPS_CONFIG.fno_n_layers),
    )
    return MarkovComparableSurface(
        comparison_mode=str(mode),
        n_regimes=int(locked["n_regimes"]),
        vocab_size=int(locked["vocab_size"]),
        generator_profile=str(locked["generator_profile"]),
        min_tokens=int(locked["min_tokens"]),
        max_tokens=int(locked["max_tokens"]),
        min_segments=int(locked["min_segments"]),
        max_segments=int(locked["max_segments"]),
        min_seg_len=int(locked["min_seg_len"]),
        max_seg_len=int(locked["max_seg_len"]),
        min_distinct_regimes_per_doc=(
            None
            if locked["min_distinct_regimes_per_doc"] is None
            else int(locked["min_distinct_regimes_per_doc"])
        ),
        max_distinct_regimes_per_doc=(
            None
            if locked["max_distinct_regimes_per_doc"] is None
            else int(locked["max_distinct_regimes_per_doc"])
        ),
        fixed_leaf_tokens=int(locked["fixed_leaf_tokens"]),
        state_dim=_resolve_positive_int(
            mapping,
            "state_dim",
            fallback=int(getattr(benchmark, "official_state_dim", _DEFAULT_OPS_CONFIG.state_dim)),
        ),
        hidden_dim=_resolve_positive_int(
            mapping,
            "hidden_dim",
            fallback=int(
                getattr(benchmark, "official_hidden_dim", _DEFAULT_OPS_CONFIG.hidden_dim)
            ),
        ),
        n_epochs=_resolve_positive_int(
            mapping,
            "n_epochs",
            fallback=int(getattr(benchmark, "official_epochs", _DEFAULT_OPS_CONFIG.n_epochs)),
        ),
        batch_size=_resolve_positive_int(
            mapping,
            "batch_size",
            fallback=int(
                getattr(benchmark, "official_batch_size", _DEFAULT_OPS_CONFIG.batch_size)
            ),
        ),
        lr=float(
            mapping.get(
                "lr",
                getattr(benchmark, "official_lr", _DEFAULT_OPS_CONFIG.lr),
            )
        ),
        weight_decay=float(
            mapping.get(
                "weight_decay",
                getattr(
                    benchmark,
                    "official_weight_decay",
                    _DEFAULT_OPS_CONFIG.weight_decay,
                ),
            )
        ),
        tree_leaf_fno_width=_fno_arch_from_mapping.width,
        tree_leaf_fno_n_modes=_fno_arch_from_mapping.n_modes,
        tree_leaf_fno_n_layers=_fno_arch_from_mapping.n_layers,
        tree_root_supervision_kind=str(
            mapping.get(
                "tree_root_supervision_kind",
                _DEFAULT_OPS_CONFIG.tree_root_supervision_kind,
            )
            or _DEFAULT_OPS_CONFIG.tree_root_supervision_kind
        ),
        budget_total_calls_per_doc=float(
            mapping.get(
                "budget_total_calls_per_doc",
                _DEFAULT_OPS_CONFIG.budget_total_calls_per_doc,
            )
        ),
        full_doc_budget_share=float(
            mapping.get(
                "full_doc_budget_share",
                _DEFAULT_OPS_CONFIG.full_doc_budget_share,
            )
        ),
        doc_consumption_mode=str(
            mapping.get("doc_consumption_mode", _DEFAULT_OPS_CONFIG.doc_consumption_mode)
            or _DEFAULT_OPS_CONFIG.doc_consumption_mode
        ),
        local_split_mode=str(
            mapping.get("local_split_mode", _DEFAULT_OPS_CONFIG.local_split_mode)
            or _DEFAULT_OPS_CONFIG.local_split_mode
        ),
    )


def comparable_surface_snapshot_from_mapping(
    config: Mapping[str, Any] | OPSCountConfig | None,
) -> Dict[str, Any]:
    mapping = _mapping_from_config(config)
    from treepo._research.ctreepo.sim.core.fno_arch_config import resolve_fno_arch_from_mapping as _resolve_snap
    _snap_fno_arch = _resolve_snap(mapping)
    snapshot: Dict[str, Any] = {
        "comparison_mode": normalize_markov_comparison_mode(
            mapping.get("comparison_mode", "legacy")
        ),
        "n_regimes": _safe_int(mapping.get("n_regimes", 0), 0),
        "vocab_size": _safe_int(mapping.get("vocab_size", 0), 0),
        "generator_profile": str(mapping.get("generator_profile", "") or ""),
        "min_tokens": _safe_int(mapping.get("min_tokens", 0), 0),
        "max_tokens": _safe_int(mapping.get("max_tokens", 0), 0),
        "min_segments": _safe_int(mapping.get("min_segments", 0), 0),
        "max_segments": _safe_int(mapping.get("max_segments", 0), 0),
        "min_seg_len": _safe_int(mapping.get("min_seg_len", 0), 0),
        "max_seg_len": _safe_int(mapping.get("max_seg_len", 0), 0),
        "min_distinct_regimes_per_doc": (
            None
            if mapping.get("min_distinct_regimes_per_doc", None) is None
            else _safe_int(mapping.get("min_distinct_regimes_per_doc", 0), 0)
        ),
        "max_distinct_regimes_per_doc": (
            None
            if mapping.get("max_distinct_regimes_per_doc", None) is None
            else _safe_int(mapping.get("max_distinct_regimes_per_doc", 0), 0)
        ),
        "fixed_leaf_tokens": _safe_int(mapping.get("fixed_leaf_tokens", 0), 0),
        "state_dim": _safe_int(mapping.get("state_dim", 0), 0),
        "hidden_dim": _safe_int(mapping.get("hidden_dim", 0), 0),
        "n_epochs": _safe_int(mapping.get("n_epochs", 0), 0),
        "batch_size": _safe_int(mapping.get("batch_size", 0), 0),
        "lr": _safe_float(mapping.get("lr", 0.0), 0.0),
        "weight_decay": _safe_float(mapping.get("weight_decay", 0.0), 0.0),
        "tree_leaf_fno_width": _snap_fno_arch.width,
        "tree_leaf_fno_n_modes": _snap_fno_arch.n_modes,
        "tree_leaf_fno_n_layers": _snap_fno_arch.n_layers,
        "tree_root_supervision_kind": str(
            mapping.get("tree_root_supervision_kind", "") or ""
        ),
        "budget_total_calls_per_doc": _safe_float(
            mapping.get("budget_total_calls_per_doc", 0.0),
            0.0,
        ),
        "full_doc_budget_share": _safe_float(
            mapping.get("full_doc_budget_share", 1.0),
            1.0,
        ),
        "doc_consumption_mode": str(mapping.get("doc_consumption_mode", "") or ""),
        "local_split_mode": str(mapping.get("local_split_mode", "") or ""),
    }
    return snapshot


def comparison_surface_diff(
    *,
    expected_surface: Mapping[str, Any] | MarkovComparableSurface,
    actual_config: Mapping[str, Any] | OPSCountConfig | None,
) -> Dict[str, Any]:
    expected_mapping = (
        expected_surface.to_dict()
        if isinstance(expected_surface, MarkovComparableSurface)
        else dict(expected_surface or {})
    )
    actual_snapshot = comparable_surface_snapshot_from_mapping(actual_config)
    diff: Dict[str, Any] = {}
    for field_name in MARKOV_COMPARABLE_SURFACE_FIELDS:
        expected = expected_mapping.get(field_name)
        actual = actual_snapshot.get(field_name)
        if isinstance(expected, float):
            actual_value = _safe_float(actual, float("nan"))
            if not math.isfinite(actual_value) or abs(actual_value - float(expected)) > 1e-12:
                diff[field_name] = {"expected": expected, "actual": actual}
            continue
        if expected != actual:
            diff[field_name] = {"expected": expected, "actual": actual}
    return diff


def apply_comparable_surface_to_mapping(
    *,
    benchmark: Any,
    config: Mapping[str, Any] | OPSCountConfig | None,
    surface: Mapping[str, Any] | MarkovComparableSurface,
) -> Dict[str, Any]:
    """Apply a comparable surface to a config, ensuring both paths agree.

    Overwrites benchmark-locked fields and shared comparison fields from the
    surface. FNO architecture is written to both canonical (fno_*) and legacy
    (tree_leaf_fno_*) keys from a single resolved source.
    """
    mapping = _mapping_from_config(config)
    surface_mapping = (
        surface.to_dict() if isinstance(surface, MarkovComparableSurface) else dict(surface)
    )
    resolved = dict(mapping)
    # Apply all comparable surface fields in one pass.
    for field_name in MARKOV_COMPARABLE_SURFACE_FIELDS:
        if field_name in surface_mapping:
            resolved[field_name] = surface_mapping[field_name]
    # Ensure canonical fno_* keys are also set (for resolve_fno_arch).
    from treepo._research.ctreepo.sim.core.fno_arch_config import resolve_fno_arch_from_mapping
    _surface_fno = resolve_fno_arch_from_mapping(surface_mapping)
    resolved["fno_width"] = _surface_fno.width
    resolved["fno_n_modes"] = _surface_fno.n_modes
    resolved["fno_n_layers"] = _surface_fno.n_layers
    resolved["tree_leaf_fno_width"] = _surface_fno.width
    resolved["tree_leaf_fno_n_modes"] = _surface_fno.n_modes
    resolved["tree_leaf_fno_n_layers"] = _surface_fno.n_layers
    resolved["comparison_mode"] = normalize_markov_comparison_mode(
        surface_mapping.get("comparison_mode", "legacy")
    )
    benchmark_locked_defaults = resolve_markov_benchmark_locked_fields(
        benchmark=benchmark,
        config={},
        comparison_mode="legacy",
    )
    geometry_overridden = int(resolved.get("fixed_leaf_tokens", 0)) != int(
        benchmark_locked_defaults.get("fixed_leaf_tokens", 0)
    )
    resolved["preserve_requested_leaf_tokens"] = bool(
        geometry_overridden or resolved.get("preserve_requested_leaf_tokens", False)
    )
    resolved["official_fno_preserve_requested_leaf_tokens"] = bool(
        geometry_overridden
        or resolved.get("official_fno_preserve_requested_leaf_tokens", False)
    )
    return resolved


__all__ = [
    "FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS",
    "MARKOV_BENCHMARK_LOCKED_FIELDS",
    "MARKOV_COMPARABLE_SURFACE_FIELDS",
    "MARKOV_COMPARISON_OFFICIAL_FNO_FAMILIES",
    "MARKOV_COMPARISON_TREE_FAMILIES",
    "MARKOV_SHARED_COMPARISON_FIELDS",
    "MarkovComparableSurface",
    "VALID_COMPARISON_MODES",
    "apply_comparable_surface_to_mapping",
    "comparable_surface_snapshot_from_mapping",
    "comparison_surface_diff",
    "infer_markov_comparison_mode",
    "normalize_markov_comparison_mode",
    "resolve_markov_benchmark_locked_fields",
    "resolve_markov_comparable_surface",
]
