from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    LOCAL_LAW_ESTIMATOR_CORRECTED,
    LOCAL_LAW_ESTIMATOR_NONE,
    LOCAL_LAW_ESTIMATOR_ORACLE_EXACT,
    canonical_law_component_weights,
    canonical_law_id,
    normalize_objective_spec,
)


def _finite_nonnegative(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0
    return float(max(0.0, out)) if math.isfinite(out) else 0.0


def _finite_probability(value: object, *, name: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}") from exc
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")
    return float(out)


@dataclass(frozen=True)
class ResolvedObjectiveWeights:
    """Canonical root/local-law objective weights.

    ``local_law_weight`` is the nominal λ. In lambda mode it is supplied
    directly and split equally over active laws. In explicit mode it is implied
    by normalizing the root and law weights.
    """

    input_mode: str
    weighting_scheme: str
    root_share: float
    local_law_shares: Dict[str, float]

    @property
    def local_law_weight(self) -> float:
        return float(sum(float(v) for v in dict(self.local_law_shares).values()))

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "objective_input_mode": str(self.input_mode),
            "weighting_scheme": str(self.weighting_scheme),
            "root_share": float(self.root_share),
            "local_law_weight": float(self.local_law_weight),
            "local_law_shares": {
                str(k): float(v) for k, v in dict(self.local_law_shares).items()
            },
        }


def resolve_root_local_objective_weights(
    *,
    local_law_weight: Optional[float],
    active_laws: Sequence[str],
    explicit_root_weight: Optional[float] = None,
    explicit_law_weights: Optional[Mapping[str, float]] = None,
    objective_context: str = "objective",
) -> ResolvedObjectiveWeights:
    """Resolve the single theorem-facing root/local objective.

    Exactly one of the two modes is active:

    * lambda mode, when ``local_law_weight`` is supplied; and
    * explicit normalized-weight mode, when it is not.

    Callers must reject user-supplied hybrids before passing explicit weights
    alongside ``local_law_weight``. This helper performs the shared math and the
    edge-case checks used by Markov, LDA, DSPy, and sketch/HLL wrappers.
    """

    laws = tuple(canonical_law_id(str(law), allow_aliases=True) for law in active_laws if str(law))
    if local_law_weight is not None:
        if explicit_root_weight is not None or dict(explicit_law_weights or {}):
            raise ValueError(
                f"{objective_context}: local_law_weight is mutually exclusive "
                "with explicit root/law weights"
            )
        lam = _finite_probability(local_law_weight, name="local_law_weight")
        if lam > 0.0 and not laws:
            raise ValueError(f"{objective_context}: local_law_weight > 0 requires at least one active local law")
        share = float(lam / float(len(laws))) if laws and lam > 0.0 else 0.0
        return ResolvedObjectiveWeights(
            input_mode="lambda",
            weighting_scheme="normalized_lambda_tradeoff",
            root_share=float(1.0 - lam),
            local_law_shares={law: float(share) for law in laws},
        )

    root_raw = _finite_nonnegative(1.0 if explicit_root_weight is None else explicit_root_weight)
    law_raw = canonical_law_component_weights(dict(explicit_law_weights or {}), allow_aliases=True)
    law_raw = {str(name): _finite_nonnegative(value) for name, value in law_raw.items()}
    total = float(root_raw + sum(float(v) for v in law_raw.values()))
    if total <= 0.0:
        return ResolvedObjectiveWeights(
            input_mode="explicit_weights",
            weighting_scheme="normalized_explicit_weights",
            root_share=1.0,
            local_law_shares={str(name): 0.0 for name in law_raw.keys()},
        )
    return ResolvedObjectiveWeights(
        input_mode="explicit_weights",
        weighting_scheme="normalized_explicit_weights",
        root_share=float(root_raw / total),
        local_law_shares={
            str(name): float(value / total) for name, value in law_raw.items()
        },
    )


@dataclass(frozen=True, init=False)
class CompositeObjectiveSpec:
    name: str
    root_metric_name: str
    root_share: float
    local_law_component_weights: Dict[str, float] = field(default_factory=dict)
    auxiliary_diagnostic_weights: Dict[str, float] = field(default_factory=dict)
    weighting_scheme: str = "single_lambda_root_local"
    root_share_source: str = ""
    selection_metric_name: str = "configured_objective"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        name: str,
        selection_metric_name: str = "configured_objective",
        root_metric_name: Optional[str] = None,
        root_share: Optional[float] = None,
        local_law_component_weights: Optional[Mapping[str, float]] = None,
        auxiliary_diagnostic_weights: Optional[Mapping[str, float]] = None,
        weighting_scheme: str = "single_lambda_root_local",
        root_share_source: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        # Deprecated private aliases accepted only to keep older call sites
        # working while they are moved to the public contract vocabulary.
        task_name: Optional[str] = None,
        task_weight: Optional[float] = None,
        local_law_weights: Optional[Mapping[str, float]] = None,
        proxy_weights: Optional[Mapping[str, float]] = None,
        task_weight_source: Optional[str] = None,
    ) -> None:
        used_legacy_aliases = any(
            value is not None
            for value in (
                task_name,
                task_weight,
                local_law_weights,
                proxy_weights,
                task_weight_source,
            )
        )
        resolved_root_metric_name = str(root_metric_name or task_name or "root_loss")
        raw_local_weights = (
            dict(local_law_component_weights)
            if local_law_component_weights is not None
            else dict(local_law_weights or {})
        )
        local_weights = canonical_law_component_weights(
            raw_local_weights,
            allow_aliases=bool(used_legacy_aliases),
        )
        auxiliary_weights = {
            str(k): float(v)
            for k, v in dict(auxiliary_diagnostic_weights or proxy_weights or {}).items()
        }
        if root_share is not None and task_weight is not None:
            raise ValueError("root_share is mutually exclusive with task_weight")
        if root_share is None and task_weight is not None:
            root_raw = _finite_nonnegative(task_weight)
            total = float(root_raw + sum(_finite_nonnegative(v) for v in local_weights.values()))
            resolved_root_share = float(root_raw / total) if total > 0.0 else 1.0
            if total > 0.0:
                local_weights = {
                    str(k): float(_finite_nonnegative(v) / total)
                    for k, v in local_weights.items()
                }
        else:
            resolved_root_share = _finite_probability(
                1.0 if root_share is None else root_share,
                name="root_share",
            )
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "root_metric_name", resolved_root_metric_name)
        object.__setattr__(self, "root_share", float(resolved_root_share))
        object.__setattr__(
            self,
            "local_law_component_weights",
            {str(k): float(v) for k, v in local_weights.items()},
        )
        object.__setattr__(
            self,
            "auxiliary_diagnostic_weights",
            {str(k): float(v) for k, v in auxiliary_weights.items()},
        )
        object.__setattr__(self, "weighting_scheme", str(weighting_scheme))
        object.__setattr__(self, "root_share_source", str(root_share_source or task_weight_source or ""))
        object.__setattr__(self, "selection_metric_name", str(selection_metric_name))
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def task_name(self) -> str:
        return str(self.root_metric_name)

    @property
    def task_weight(self) -> float:
        return float(self.root_share)

    @property
    def local_law_weights(self) -> Dict[str, float]:
        return dict(self.local_law_component_weights)

    @property
    def proxy_weights(self) -> Dict[str, float]:
        return dict(self.auxiliary_diagnostic_weights)

    @property
    def task_weight_source(self) -> str:
        return str(self.root_share_source)

    def total_weight_without_proxy(self) -> float:
        return float(
            _finite_nonnegative(self.root_share)
            + sum(_finite_nonnegative(v) for v in self.local_law_component_weights.values())
        )

    def normalized_task_share(self) -> float:
        return float(self.root_share)

    def normalized_local_law_weights(self) -> Dict[str, float]:
        return {
            str(name): float(_finite_nonnegative(value))
            for name, value in dict(self.local_law_component_weights).items()
        }

    def local_law_weight(self) -> float:
        return float(sum(float(v) for v in self.normalized_local_law_weights().values()))

    def normalized_proxy_weights(self) -> Dict[str, float]:
        return {
            str(name): _finite_nonnegative(value)
            for name, value in dict(self.auxiliary_diagnostic_weights).items()
        }

    def to_objective_spec(self) -> Dict[str, Any]:
        local_law_weights = self.normalized_local_law_weights()
        return normalize_objective_spec(
            {
                "objective_family": str(self.name or "configured_objective"),
                "local_law_estimator": str(
                    self.metadata.get(
                        "local_law_estimator",
                        LOCAL_LAW_ESTIMATOR_CORRECTED
                        if local_law_weights
                        else LOCAL_LAW_ESTIMATOR_NONE,
                    )
                ),
                "root_share": float(self.normalized_task_share()),
                "local_law_weight": float(sum(float(v) for v in local_law_weights.values())),
                "local_law_component_weights": local_law_weights,
                "weighting_scheme": str(self.weighting_scheme),
                "root_share_source": str(self.root_share_source),
                "selection_metric_name": str(self.selection_metric_name),
                "metadata": {
                    "root_metric_name": str(self.root_metric_name),
                    "weighting_scheme": str(self.weighting_scheme),
                    "root_share_source": str(self.root_share_source),
                    "selection_metric_name": str(self.selection_metric_name),
                    **dict(self.metadata or {}),
                },
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        local_law_weight_total = float(sum(float(v) for v in dict(self.local_law_component_weights).values()))
        proxy_weight_total = float(sum(float(v) for v in dict(self.auxiliary_diagnostic_weights).values()))
        total_weight_without_proxy = float(self.total_weight_without_proxy())
        normalized_task_share = float(self.normalized_task_share())
        normalized_local_law_weights = self.normalized_local_law_weights()
        normalized_local_law_share = float(sum(float(v) for v in normalized_local_law_weights.values()))
        payload = {
            "name": str(self.name),
            "root_metric_name": str(self.root_metric_name),
            "weighting_scheme": str(self.weighting_scheme),
            "root_share_source": str(self.root_share_source),
            "selection_metric_name": str(self.selection_metric_name),
            "metadata": dict(self.metadata or {}),
        }
        payload["root_share"] = normalized_task_share
        payload["local_law_component_weights"] = normalized_local_law_weights
        payload["auxiliary_diagnostic_weights"] = {
            str(k): float(v) for k, v in dict(self.auxiliary_diagnostic_weights).items()
        }
        payload["local_law_weight_total"] = local_law_weight_total
        payload["auxiliary_diagnostic_weight_total"] = proxy_weight_total
        payload["total_weight_without_proxy"] = total_weight_without_proxy
        payload["local_law_weight"] = normalized_local_law_share
        payload["objective_input_mode"] = str(
            dict(self.metadata or {}).get("objective_input_mode", self.weighting_scheme)
        )
        payload["objective_spec"] = self.to_objective_spec()
        return payload


@dataclass(frozen=True)
class CompositeObjectiveEvaluation:
    total: float
    task_raw: float
    task_term: float
    local_law_raw: Dict[str, float] = field(default_factory=dict)
    local_law_terms: Dict[str, float] = field(default_factory=dict)
    proxy_raw: Dict[str, float] = field(default_factory=dict)
    proxy_terms: Dict[str, float] = field(default_factory=dict)
    root_share: Optional[float] = None
    local_law_weight: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "total": float(self.total),
            "task_raw": float(self.task_raw),
            "task_term": float(self.task_term),
            "local_law_raw": {str(k): float(v) for k, v in dict(self.local_law_raw).items()},
            "local_law_terms": {str(k): float(v) for k, v in dict(self.local_law_terms).items()},
            "proxy_raw": {str(k): float(v) for k, v in dict(self.proxy_raw).items()},
            "proxy_terms": {str(k): float(v) for k, v in dict(self.proxy_terms).items()},
            "local_law_raw_total": float(sum(float(v) for v in self.local_law_raw.values())),
            "local_law_term_total": float(sum(float(v) for v in self.local_law_terms.values())),
            "proxy_raw_total": float(sum(float(v) for v in self.proxy_raw.values())),
            "proxy_term_total": float(sum(float(v) for v in self.proxy_terms.values())),
        }
        if self.root_share is not None:
            payload["root_share"] = float(self.root_share)
        if self.local_law_weight is not None:
            payload["local_law_weight"] = float(self.local_law_weight)
        return payload

    def to_flat_dict(self, *, prefix: str) -> Dict[str, float]:
        payload = {
            str(prefix): float(self.total),
            f"{prefix}_task_raw": float(self.task_raw),
            f"{prefix}_task_term": float(self.task_term),
            f"{prefix}_local_law_raw_total": float(
                sum(float(v) for v in self.local_law_raw.values())
            ),
            f"{prefix}_local_law_term_total": float(
                sum(float(v) for v in self.local_law_terms.values())
            ),
            f"{prefix}_proxy_raw_total": float(sum(float(v) for v in self.proxy_raw.values())),
            f"{prefix}_proxy_term_total": float(sum(float(v) for v in self.proxy_terms.values())),
        }
        for name, value in self.local_law_raw.items():
            payload[f"{prefix}_{name}_raw"] = float(value)
        for name, value in self.local_law_terms.items():
            payload[f"{prefix}_{name}_term"] = float(value)
        for name, value in self.proxy_raw.items():
            payload[f"{prefix}_{name}_raw"] = float(value)
        for name, value in self.proxy_terms.items():
            payload[f"{prefix}_{name}_term"] = float(value)
        if self.root_share is not None:
            payload[f"{prefix}_root_share"] = float(self.root_share)
        if self.local_law_weight is not None:
            payload[f"{prefix}_local_law_weight"] = float(self.local_law_weight)
        return payload


OBJECTIVE_ESTIMATOR_KEYS = ("exact", "ht", "hajek", "eb_lo", "eb_hi")


def objective_estimator_alias(base_name: str, estimator: str) -> str:
    name = str(base_name)
    est = str(estimator)
    return name if est == "exact" else f"{name}_{est}"


def _safe_estimator_value(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def scalarize_objective_estimates(
    spec: CompositeObjectiveSpec,
    *,
    task_estimates: Mapping[str, float],
    local_law_estimates: Mapping[str, Mapping[str, float]],
    proxy_estimates: Optional[Mapping[str, Mapping[str, float]]] = None,
    selection_preference: str = "hajek",
) -> Dict[str, Any]:
    base_name = str(spec.name or spec.selection_metric_name or "configured_objective")
    proxy_source = dict(proxy_estimates or {})
    root_share = float(spec.normalized_task_share())
    local_law_shares = spec.normalized_local_law_weights()
    local_law_weight = float(sum(float(v) for v in local_law_shares.values()))

    term_breakdown: Dict[str, Dict[str, Dict[str, float]]] = {
        "task": {},
        "local_law": {},
        "proxy": {},
    }
    totals: Dict[str, Dict[str, float]] = {}

    for estimator in OBJECTIVE_ESTIMATOR_KEYS:
        task_raw = _safe_estimator_value(task_estimates.get(estimator))
        task_term = (
            root_share * float(task_raw) if math.isfinite(task_raw) else float("nan")
        )
        estimator_local_raw: Dict[str, float] = {}
        estimator_local_terms: Dict[str, float] = {}
        for name, weight in local_law_shares.items():
            raw_map = dict(local_law_estimates.get(str(name), {}) or {})
            raw_value = _safe_estimator_value(raw_map.get(estimator))
            estimator_local_raw[str(name)] = float(raw_value)
            estimator_local_terms[str(name)] = (
                float(weight) * float(raw_value) if math.isfinite(raw_value) else float("nan")
            )
        estimator_proxy_raw: Dict[str, float] = {}
        estimator_proxy_terms: Dict[str, float] = {}
        for name, weight in dict(spec.proxy_weights).items():
            raw_map = dict(proxy_source.get(str(name), {}) or {})
            raw_value = _safe_estimator_value(raw_map.get(estimator))
            estimator_proxy_raw[str(name)] = float(raw_value)
            estimator_proxy_terms[str(name)] = (
                float(weight) * float(raw_value) if math.isfinite(raw_value) else float("nan")
            )
        local_term_total = (
            sum(float(v) for v in estimator_local_terms.values())
            if all(math.isfinite(float(v)) for v in estimator_local_terms.values())
            else float("nan")
        )
        proxy_term_total = (
            sum(float(v) for v in estimator_proxy_terms.values())
            if all(math.isfinite(float(v)) for v in estimator_proxy_terms.values())
            else float("nan")
        )
        if math.isfinite(task_term) and math.isfinite(local_term_total):
            total = float(task_term + local_term_total)
        else:
            total = float("nan")
        local_law_objective_value = (
            float(local_term_total / local_law_weight)
            if local_law_weight > 0.0 and math.isfinite(local_term_total)
            else 0.0
        )
        totals[str(estimator)] = {
            "full_objective_value": float(total),
            "task_objective_value": float(task_raw),
            "task_objective_term": float(task_term),
            "local_law_objective_value": float(local_law_objective_value),
            "local_law_objective_term": float(
                sum(float(v) for v in estimator_local_terms.values())
            )
            if all(math.isfinite(float(v)) for v in estimator_local_terms.values())
            else float("nan"),
            "proxy_objective_value": float(sum(float(v) for v in estimator_proxy_raw.values()))
            if all(math.isfinite(float(v)) for v in estimator_proxy_raw.values())
            else float("nan"),
            "proxy_objective_term": float(sum(float(v) for v in estimator_proxy_terms.values()))
            if all(math.isfinite(float(v)) for v in estimator_proxy_terms.values())
            else float("nan"),
        }
        term_breakdown["task"][str(estimator)] = {
            "raw": float(task_raw),
            "term": float(task_term),
        }
        for name in dict(spec.local_law_weights).keys():
            term_breakdown["local_law"].setdefault(str(name), {})[str(estimator)] = {
                "raw": float(estimator_local_raw[str(name)]),
                "term": float(estimator_local_terms[str(name)]),
            }
        for name in dict(spec.proxy_weights).keys():
            term_breakdown["proxy"].setdefault(str(name), {})[str(estimator)] = {
                "raw": float(estimator_proxy_raw[str(name)]),
                "term": float(estimator_proxy_terms[str(name)]),
            }

    available_estimators = [
        str(estimator)
        for estimator in OBJECTIVE_ESTIMATOR_KEYS
        if math.isfinite(_safe_estimator_value(totals[str(estimator)]["full_objective_value"]))
    ]
    preferred = str(selection_preference)
    if preferred not in available_estimators:
        preferred = "exact" if "exact" in available_estimators else (
            available_estimators[0] if available_estimators else "exact"
        )
    selection_metric_name = objective_estimator_alias(base_name, preferred)
    selection_metric_value = _safe_estimator_value(
        totals.get(preferred, {}).get("full_objective_value")
    )

    payload: Dict[str, Any] = {
        "objective_name": base_name,
        "selection_metric_name": str(selection_metric_name),
        "selection_estimator": str(preferred),
        "selection_metric_value": float(selection_metric_value),
        "available_estimators": [str(x) for x in available_estimators],
        "estimator_components": term_breakdown,
        "root_share": float(root_share),
        "local_law_weight": float(local_law_weight),
        "local_law_shares": {str(k): float(v) for k, v in local_law_shares.items()},
    }
    exact = totals.get("exact", {})
    payload["full_objective_value"] = float(
        _safe_estimator_value(exact.get("full_objective_value"))
    )
    payload["task_objective_value"] = float(
        _safe_estimator_value(exact.get("task_objective_value"))
    )
    payload["task_objective_term"] = float(
        _safe_estimator_value(exact.get("task_objective_term"))
    )
    payload["regular_objective_value"] = float(
        _safe_estimator_value(exact.get("task_objective_value"))
    )
    payload["regular_objective_term"] = float(
        _safe_estimator_value(exact.get("task_objective_term"))
    )
    payload["local_law_objective_value"] = float(
        _safe_estimator_value(exact.get("local_law_objective_value"))
    )
    payload["local_law_objective_term"] = float(
        _safe_estimator_value(exact.get("local_law_objective_term"))
    )
    payload["proxy_objective_value"] = float(
        _safe_estimator_value(exact.get("proxy_objective_value"))
    )
    payload["proxy_objective_term"] = float(
        _safe_estimator_value(exact.get("proxy_objective_term"))
    )
    for estimator, metrics in totals.items():
        alias = objective_estimator_alias(base_name, estimator)
        payload[str(alias)] = float(_safe_estimator_value(metrics.get("full_objective_value")))
        payload[f"{alias}_task_objective_value"] = float(
            _safe_estimator_value(metrics.get("task_objective_value"))
        )
        payload[f"{alias}_task_objective_term"] = float(
            _safe_estimator_value(metrics.get("task_objective_term"))
        )
        payload[f"{alias}_local_law_objective_value"] = float(
            _safe_estimator_value(metrics.get("local_law_objective_value"))
        )
        payload[f"{alias}_local_law_objective_term"] = float(
            _safe_estimator_value(metrics.get("local_law_objective_term"))
        )
        payload[f"{alias}_proxy_objective_value"] = float(
            _safe_estimator_value(metrics.get("proxy_objective_value"))
        )
        payload[f"{alias}_proxy_objective_term"] = float(
            _safe_estimator_value(metrics.get("proxy_objective_term"))
        )
    eb_lo = _safe_estimator_value(totals.get("eb_lo", {}).get("full_objective_value"))
    eb_hi = _safe_estimator_value(totals.get("eb_hi", {}).get("full_objective_value"))
    payload[f"{base_name}_eb_width"] = (
        float(max(0.0, eb_hi - eb_lo))
        if math.isfinite(eb_lo) and math.isfinite(eb_hi)
        else float("nan")
    )
    payload[f"{base_name}_selection_value"] = float(selection_metric_value)
    return payload


def evaluate_composite_objective(
    spec: CompositeObjectiveSpec,
    *,
    task_value: float,
    local_law_values: Mapping[str, float],
    proxy_values: Optional[Mapping[str, float]] = None,
) -> CompositeObjectiveEvaluation:
    """Evaluate the single-lambda root/local objective.

    ``task_weight`` and ``local_law_weights`` are normalized internally, so the
    theorem-facing objective is always ``root_share * L_root + Σ law_share_i *
    L_i``. Proxy terms are returned as diagnostics and are not included in
    ``total``.
    """

    task_raw = float(task_value)
    root_share = float(spec.normalized_task_share())
    local_law_shares = spec.normalized_local_law_weights()
    local_law_weight = float(sum(float(v) for v in local_law_shares.values()))
    task_term = root_share * task_raw

    local_law_raw = {
        str(name): float(local_law_values.get(name, 0.0))
        for name in dict(spec.local_law_weights).keys()
    }
    local_law_terms = {
        str(name): float(local_law_shares.get(name, 0.0)) * float(local_law_raw[str(name)])
        for name in local_law_shares.keys()
    }

    proxy_source = dict(proxy_values or {})
    proxy_raw = {
        str(name): float(proxy_source.get(name, 0.0)) for name in dict(spec.proxy_weights).keys()
    }
    proxy_terms = {
        str(name): float(spec.proxy_weights.get(name, 0.0)) * float(proxy_raw[str(name)])
        for name in dict(spec.proxy_weights).keys()
    }

    local_law_term_total = sum(float(v) for v in local_law_terms.values())
    total = float(task_term + local_law_term_total)
    return CompositeObjectiveEvaluation(
        total=total,
        task_raw=task_raw,
        task_term=task_term,
        local_law_raw=local_law_raw,
        local_law_terms=local_law_terms,
        proxy_raw=proxy_raw,
        proxy_terms=proxy_terms,
        root_share=float(root_share),
        local_law_weight=float(local_law_weight),
    )


def evaluate_composite_objective_from_metrics(
    spec: CompositeObjectiveSpec,
    *,
    metrics: Mapping[str, object],
    task_metric_name: Optional[str] = None,
    local_law_metric_names: Optional[Mapping[str, str]] = None,
    proxy_metric_names: Optional[Mapping[str, str]] = None,
) -> CompositeObjectiveEvaluation:
    metadata = dict(spec.metadata)
    resolved_task_metric_name = str(
        task_metric_name
        or metadata.get("root_metric_name")
        or metadata.get("task_metric_name")
        or spec.root_metric_name
    )
    resolved_local_law_metric_names = dict(
        metadata.get("local_law_metric_names", {})
        if isinstance(metadata.get("local_law_metric_names"), Mapping)
        else {}
    )
    if local_law_metric_names is not None:
        resolved_local_law_metric_names.update(
            {str(name): str(metric_name) for name, metric_name in local_law_metric_names.items()}
        )
    resolved_proxy_metric_names = dict(
        metadata.get("proxy_metric_names", {})
        if isinstance(metadata.get("proxy_metric_names"), Mapping)
        else {}
    )
    if proxy_metric_names is not None:
        resolved_proxy_metric_names.update(
            {str(name): str(metric_name) for name, metric_name in proxy_metric_names.items()}
        )

    return evaluate_composite_objective(
        spec,
        task_value=float(metrics.get(resolved_task_metric_name, 0.0)),
        local_law_values={
            str(name): float(
                metrics.get(resolved_local_law_metric_names.get(str(name), str(name)), 0.0)
            )
            for name in dict(spec.local_law_weights).keys()
        },
        proxy_values={
            str(name): float(
                metrics.get(resolved_proxy_metric_names.get(str(name), str(name)), 0.0)
            )
            for name in dict(spec.proxy_weights).keys()
        },
    )
