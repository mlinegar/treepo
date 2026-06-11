"""Single, centralized registry of canonical methods.

**Axis-factored, not Cartesian.** Three public methods total — one per
orthogonal public axis. Adding a new oracle, fixture, or family does *not*
add a method; the new thing is reachable through the same call site by
passing its name as config.

| Method | Axis | What's variable | What's fixed |
|---|---|---|---|
| ``"fit"``    | spec / spec-kwargs | the full :class:`CTreePOLearningSpec`     | nothing |
| ``"oracle"`` | ``oracle_name``    | which registered oracle to score with     | family=oracle, eval-only |
| ``"audit"``  | ``rows``           | the local-law audit rows                  | post-hoc; no fit() call |

Adding the Markov oracle? Just pass ``{"oracle_name": "markov_changepoint_count"}``
and supply ``eval_data`` (no markov auto-fixture in v1).

Mirrors the established :func:`treepo.bench.runner.run_single` pattern
(flat tuple of method names, ``allowed_config_keys`` discovery, single
``run`` entry) but factored along true axes rather than enumerated
experiment pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from treepo._research.ctreepo.contracts import CTreePOFitResult, CTreePOLearningSpec
from treepo._research.ctreepo.oracles import get_oracle, list_oracles
from treepo._research.methods.hll_config import HllSketchConfig
from treepo.methods.learning import fit
from treepo.local_law import (
    LocalLawAuditRow,
    compute_influence_weighted_overlap,
    compute_influence_weighted_overlap_by_law_kind,
    local_law_objective_summary,
    local_law_objective_summary_by_law_kind,
)


_Handler = Callable[[Mapping[str, Any]], Any]

_REGISTRY: dict[str, tuple[_Handler, frozenset[str]]] = {}


def register_method(
    name: str,
    handler: _Handler,
    *,
    allowed_config_keys: frozenset[str] | set[str] | tuple[str, ...],
) -> None:
    key = _normalize(name)
    if not key:
        raise ValueError("method name must be non-empty")
    if key in _REGISTRY:
        raise ValueError(f"method {name!r} already registered")
    _REGISTRY[key] = (handler, frozenset(str(k) for k in allowed_config_keys))


def list_methods() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def allowed_config_keys(method: str) -> set[str]:
    return set(_lookup(method)[1])


def method_info(method: str) -> dict[str, Any]:
    key = _normalize(method)
    _, allowed = _lookup(method)
    return {"name": key, "allowed_config_keys": sorted(allowed)}


def run(method: str, config: Mapping[str, Any] | None = None) -> Any:
    key = _normalize(method)
    handler, allowed = _lookup(method)
    payload = dict(config or {})
    unknown = sorted(k for k in payload if k not in allowed)
    if unknown:
        raise ValueError(
            f"unknown config keys for method {key!r}: {unknown}; allowed: {sorted(allowed)}"
        )
    return handler(payload)


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _lookup(method: str) -> tuple[_Handler, frozenset[str]]:
    key = _normalize(method)
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown method {method!r}; available: {', '.join(list_methods())}"
        )
    return _REGISTRY[key]


# --------------------------------------------------------------------------- #
# Method 1: fit — universal entry. Either spec or high-level kwargs.
# --------------------------------------------------------------------------- #


_FIT_KWARGS: tuple[str, ...] = (
    "space_kind", "family", "schedule", "initial_artifacts",
    "train_data", "eval_data", "backend_config", "axis",
)


def _method_fit(config: Mapping[str, Any]) -> CTreePOFitResult:
    """Accept either ``config['spec']`` (a CTreePOLearningSpec or mapping)
    or the high-level kwargs that build one (``family``, ``train_data``,
    ``eval_data``, ``backend_config``, ``axis``, ...).
    """
    spec = config.get("spec")
    if spec is not None:
        if isinstance(spec, CTreePOLearningSpec):
            return fit(spec)
        if isinstance(spec, Mapping):
            return fit(CTreePOLearningSpec.from_mapping(spec))
        raise TypeError(
            f"fit config['spec'] must be CTreePOLearningSpec or mapping; got {type(spec).__name__}"
        )
    if "family" not in config:
        raise ValueError(
            "fit requires either config['spec'] or config['family'] (with optional "
            f"{list(_FIT_KWARGS)})"
        )
    built = CTreePOLearningSpec(
        space_kind=str(config.get("space_kind", "fit")),
        family=str(config["family"]),
        schedule=str(config.get("schedule", "fg")),
        initial_artifacts=dict(config.get("initial_artifacts") or {"f": None, "g": None}),
        train_data=list(config.get("train_data") or []),
        eval_data=list(config.get("eval_data") or []),
        backend_config=dict(config.get("backend_config") or {}),
        axis=dict(config.get("axis") or {"max_iterations": 0, "axis_value": 0}),
    )
    return fit(built)


# --------------------------------------------------------------------------- #
# Method 2: oracle — any registered oracle, fixture by domain.
# --------------------------------------------------------------------------- #


def _make_oracle_fixture_classical_sketch(config: Mapping[str, Any]) -> list:
    from treepo.methods.fixtures import make_hll_token_trees

    defaults = HllSketchConfig()
    return make_hll_token_trees(
        n_trees=int(config.get("n_trees", defaults.n_trees)),
        leaves_per_tree=int(config.get("leaves_per_tree", defaults.leaves_per_tree)),
        leaf_token_count=int(config.get("leaf_token_count", defaults.leaf_token_count)),
        vocabulary_size=int(config.get("vocabulary_size", defaults.vocabulary_size)),
        seed=int(config.get("seed", defaults.seed)),
    )


def _make_oracle_fixture_markov(config: Mapping[str, Any]) -> list:
    """Auto-build a Markov change-point corpus for the ``markov_changepoint_count`` oracle.

    Knobs match the upstream ``MarkovChangepointConfig`` field defaults;
    callers can override individual values through the oracle config dict.
    """
    from treepo.methods.fixtures import make_markov_changepoint_trees

    return make_markov_changepoint_trees(
        n_regimes=int(config.get("n_regimes", 4)),
        vocab_size=int(config.get("vocab_size", 96)),
        min_tokens=int(config.get("min_tokens", 96)),
        max_tokens=int(config.get("max_tokens", 96)),
        min_segments=int(config.get("min_segments", 2)),
        max_segments=int(config.get("max_segments", 5)),
        min_seg_len=int(config.get("min_seg_len", 8)),
        max_seg_len=int(config.get("max_seg_len", 32)),
        train_docs=int(config.get("train_docs", 120)),
        test_docs=int(config.get("test_docs", 60)),
        sinkhorn_iters=int(config.get("sinkhorn_iters", 30)),
        transition_log_std=float(config.get("transition_log_std", 1.25)),
        seed=int(config.get("seed", 0)),
        split=str(config.get("split", "test")),
    )


_ORACLE_DOMAIN_FIXTURES: dict[str, Callable[[Mapping[str, Any]], list]] = {
    "classical_sketch": _make_oracle_fixture_classical_sketch,
    "markov": _make_oracle_fixture_markov,
}
_RESEARCH_ONLY_ORACLES = {"leaf_local_mixture_target"}


def _method_oracle(config: Mapping[str, Any]) -> CTreePOFitResult:
    """Score eval trees against any registered oracle.

    - ``oracle_name`` (required) — must be in
      :func:`src.ctreepo.oracles.list_oracles`. The oracle's ``domain``
      drives auto-fixture selection.
    - ``eval_data`` (optional) — caller-supplied trees override the
      auto-fixture. Required for domains without a v1 fixture builder
      (e.g. ``"markov"``).
    - Domain-specific fixture knobs (``seed``, ``n_trees``, ``split``,
      etc.) — passed through to the auto-fixture builder.
    """
    name = config.get("oracle_name")
    if not name:
        available = ", ".join(list_oracles())
        raise ValueError(f"oracle requires config['oracle_name']; registered: {available}")
    spec_oracle = get_oracle(str(name))
    domain = str(spec_oracle.domain)

    eval_data = config.get("eval_data")
    if eval_data is None:
        builder = _ORACLE_DOMAIN_FIXTURES.get(domain)
        if builder is None:
            raise ValueError(
                f"oracle {name!r} has domain {domain!r} with no auto-fixture in v1; "
                f"pass config['eval_data'] explicitly. Available domains with fixtures: "
                f"{sorted(_ORACLE_DOMAIN_FIXTURES)}"
            )
        eval_data = builder(config)

    backend_config: dict[str, Any] = {"oracle_name": str(name)}
    if config.get("output_dir") is not None:
        backend_config["output_dir"] = str(config["output_dir"])
    spec = CTreePOLearningSpec(
        space_kind=f"oracle:{name}",
        family="oracle",
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=list(eval_data),
        backend_config=backend_config,
        axis={"max_iterations": 0, "axis_value": 0},
    )
    return fit(spec)


# --------------------------------------------------------------------------- #
# Method 3: audit — post-hoc local-law audit on supplied rows.
# --------------------------------------------------------------------------- #


def _method_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    rows_payload = config.get("rows")
    if not rows_payload:
        raise ValueError(
            "audit requires config['rows'] (sequence of LocalLawAuditRow or mappings)"
        )
    rows: list[LocalLawAuditRow] = []
    for item in rows_payload:
        if isinstance(item, LocalLawAuditRow):
            rows.append(item)
        elif isinstance(item, Mapping):
            rows.append(LocalLawAuditRow(**dict(item)))
        else:
            raise TypeError(
                f"audit rows must be LocalLawAuditRow or mappings; got {type(item).__name__}"
            )
    mode = str(config.get("objective_mode") or "corrected_local_law")
    gamma = float(config.get("gamma_depth", 1.0))
    summary = local_law_objective_summary(rows, objective_mode=mode, gamma_depth=gamma)
    overlap = compute_influence_weighted_overlap(rows)
    by_kind_summaries = local_law_objective_summary_by_law_kind(
        rows, objective_mode=mode, gamma_depth=gamma
    )
    by_kind_overlaps = compute_influence_weighted_overlap_by_law_kind(rows)
    payload = {
        "status": "success",
        "method": "audit",
        "local_law_objective": summary.to_dict(),
        "influence_weighted_overlap": overlap.to_dict(),
        # f-vs-f* decomposition: C1 (leaf preservation), C2 (idempotence),
        # C3 (merge preservation) each get their own objective + overlap.
        "by_law_kind": {
            kind: {
                "local_law_objective": by_kind_summaries[kind].to_dict(),
                "influence_weighted_overlap": by_kind_overlaps[kind].to_dict(),
            }
            for kind in sorted(by_kind_summaries)
        },
        "n_rows": len(rows),
    }
    output_dir = config.get("output_dir")
    if output_dir:
        out = Path(str(output_dir))
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


# --------------------------------------------------------------------------- #
# Discovery helpers — what's available within each axis.
# --------------------------------------------------------------------------- #


def list_oracle_domains_with_fixtures() -> tuple[str, ...]:
    """Oracle domains that have a v1 auto-fixture in
    :data:`_ORACLE_DOMAIN_FIXTURES`. Oracles with other domains still
    work — the caller must supply ``eval_data`` explicitly.
    """
    return tuple(sorted(_ORACLE_DOMAIN_FIXTURES))


def list_registered_oracles() -> tuple[str, ...]:
    return tuple(name for name in list_oracles() if name not in _RESEARCH_ONLY_ORACLES)


# --------------------------------------------------------------------------- #
# Registry — four lines, four axes.
# --------------------------------------------------------------------------- #


register_method(
    "fit",
    _method_fit,
    allowed_config_keys={"spec", *_FIT_KWARGS},
)
register_method(
    "oracle",
    _method_oracle,
    allowed_config_keys={
        "oracle_name", "eval_data", "output_dir",
        # Common fixture knobs across domains.
        "seed", "split",
        # HLL/classical_sketch knobs.
        "n_trees", "leaves_per_tree", "leaf_token_count", "vocabulary_size",
        # Markov change-point fixture knobs.
        "n_regimes", "vocab_size", "min_tokens", "max_tokens",
        "min_segments", "max_segments", "min_seg_len", "max_seg_len",
        "train_docs", "test_docs", "sinkhorn_iters", "transition_log_std",
    },
)
register_method(
    "audit",
    _method_audit,
    allowed_config_keys={"rows", "objective_mode", "gamma_depth", "output_dir"},
)
__all__ = [
    "allowed_config_keys",
    "list_methods",
    "list_oracle_domains_with_fixtures",
    "list_registered_oracles",
    "method_info",
    "register_method",
    "run",
]
