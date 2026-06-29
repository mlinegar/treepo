"""Single, centralized registry of canonical methods.

**Axis-factored, not Cartesian.** Public methods are one per orthogonal axis.
Adding a new oracle, fixture, or family does *not* add a method; the new thing
is reachable through the same call site by passing its name as config.

| Method | Axis | What's variable | What's fixed |
|---|---|---|---|
| ``"fit"``    | spec / spec-kwargs | the full :class:`CTreePOLearningSpec` | nothing |
| ``"oracle"`` | ``oracle_name``    | lightweight built-in oracle scoring   | family=oracle, eval-only |
| ``"audit"``  | ``rows``           | the local-law audit rows                  | post-hoc; no fit() call |

Research oracles/families can register themselves from a downstream package.

Mirrors the established :func:`treepo.bench.runner.run_single` pattern
(flat tuple of method names, ``allowed_config_keys`` discovery, single
``run`` entry) but factored along true axes rather than enumerated
experiment pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from treepo.methods.contracts import CTreePOFitResult, CTreePOLearningSpec
from treepo.methods.learning import fit
from treepo.methods.oracles import list_oracles, oracle_domain
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
    "family", "estimator", "g_estimator", "train_data", "eval_data", "backend_config", "axis",
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
    estimator = config.get("estimator")
    if estimator is None:
        estimator = config.get("g_estimator")
    family = config.get("family")
    if family is None and estimator is not None:
        from treepo.methods.estimators import resolve_estimator

        family = resolve_estimator(estimator, dict(config.get("backend_config") or {})).family
    if family is None:
        raise ValueError(
            "fit requires config['spec'], config['family'], config['estimator'], "
            "or compatibility config['g_estimator'] "
            f"(with optional {list(_FIT_KWARGS)})"
        )
    built = CTreePOLearningSpec(
        space_kind="fit",
        family=str(family),
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=list(config.get("train_data") or []),
        eval_data=list(config.get("eval_data") or []),
        backend_config=dict(config.get("backend_config") or {}),
        axis=dict(config.get("axis") or {"max_iterations": 0, "axis_value": 0}),
        estimator=config.get("estimator"),
        g_estimator=config.get("g_estimator"),
    )
    return fit(built)


# --------------------------------------------------------------------------- #
# Method 2: oracle — any registered oracle, fixture by domain.
# --------------------------------------------------------------------------- #


def _make_oracle_fixture_classical_sketch(config: Mapping[str, Any]) -> list:
    from treepo.methods.fixtures import make_hll_token_trees

    return make_hll_token_trees(
        n_trees=int(config.get("n_trees", 6)),
        leaves_per_tree=int(config.get("leaves_per_tree", 4)),
        leaf_token_count=int(config.get("leaf_token_count", 24)),
        vocabulary_size=int(config.get("vocabulary_size", 200)),
        seed=int(config.get("seed", 0)),
        split=str(config.get("split", "test")),
    )


def _make_oracle_fixture_markov(config: Mapping[str, Any]) -> list:
    from treepo.methods.fixtures import make_markov_changepoint_trees

    return make_markov_changepoint_trees(
        n_trees=int(config.get("n_trees", 8)),
        n_states=int(config.get("n_states", 4)),
        doc_tokens=int(config.get("doc_tokens", 128)),
        leaf_token_count=int(config.get("leaf_token_count", 16)),
        transition_prob=float(config.get("transition_prob", 0.15)),
        vocabulary_size=int(config.get("vocabulary_size", 256)),
        seed=int(config.get("seed", 0)),
        split=str(config.get("split", "test")),
    )


_ORACLE_DOMAIN_FIXTURES: dict[str, Callable[[Mapping[str, Any]], list]] = {
    "classical_sketch": _make_oracle_fixture_classical_sketch,
    "markov": _make_oracle_fixture_markov,
}


def _method_oracle(config: Mapping[str, Any]) -> CTreePOFitResult:
    """Score eval trees against any registered oracle.

    - ``oracle_name`` (required) — must be in
      :func:`list_registered_oracles`. The oracle's ``domain``
      drives auto-fixture selection.
    - ``eval_data`` (optional) — caller-supplied trees override the
      auto-fixture.
    - Domain-specific fixture knobs (``seed``, ``n_trees``, ``split``,
      etc.) — passed through to the auto-fixture builder.
    """
    name = config.get("oracle_name")
    if not name:
        available = ", ".join(list_oracles())
        raise ValueError(f"oracle requires config['oracle_name']; registered: {available}")
    domain = oracle_domain(str(name))

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
    return list_oracles()


# --------------------------------------------------------------------------- #
# Registry — one entry per public method axis.
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
        # Markov knobs.
        "n_states", "doc_tokens", "transition_prob",
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
