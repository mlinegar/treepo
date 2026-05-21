"""Single, centralized registry of canonical methods.

**Axis-factored, not Cartesian.** Four methods total — one per orthogonal
axis. Adding a new oracle, sketch adapter, fixture, or family does *not*
add a method; the new thing is reachable through the same call site by
passing its name as config.

| Method | Axis | What's variable | What's fixed |
|---|---|---|---|
| ``"fit"``    | spec / spec-kwargs | the full :class:`CTreePOLearningSpec`     | nothing |
| ``"oracle"`` | ``oracle_name``    | which registered oracle to score with     | family=oracle, eval-only |
| ``"sketch"`` | ``sketch_kind``    | which classical sketch adapter to use     | family=sketch, eval-only, token-tree fixture |
| ``"audit"``  | ``rows``           | the local-law audit rows                  | post-hoc; no fit() call |

Adding HLL+precision=14? Just pass ``{"sketch_kind": "hll", "precision": 14}``.
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
from treepo.cld.learning import fit
from treepo.cld.local_law import (
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
    from treepo.cld.fixtures import make_hll_token_trees

    return make_hll_token_trees(
        n_trees=int(config.get("n_trees", 8)),
        leaves_per_tree=int(config.get("leaves_per_tree", 4)),
        leaf_token_count=int(config.get("leaf_token_count", 12)),
        vocabulary_size=int(config.get("vocabulary_size", 32)),
        seed=int(config.get("seed", 0)),
    )


def _make_oracle_fixture_lda(config: Mapping[str, Any]) -> list:
    from treepo.cld.fixtures import make_leaf_local_mixture_trees

    trees, _cfg = make_leaf_local_mixture_trees(
        seed=int(config.get("seed", 0)),
        split=str(config.get("split", "test")),
    )
    return trees


def _make_oracle_fixture_markov(config: Mapping[str, Any]) -> list:
    """Auto-build a Markov change-point corpus for the ``markov_changepoint_count`` oracle.

    Knobs match the upstream ``MarkovChangepointConfig`` field defaults;
    callers can override individual values through the oracle config dict.
    """
    from treepo.cld.fixtures import make_markov_changepoint_trees

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
    "lda": _make_oracle_fixture_lda,
    "markov": _make_oracle_fixture_markov,
}


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
# Method 3: sketch — any classical sketch adapter, native fixture.
# --------------------------------------------------------------------------- #


def _factory_hll(config: Mapping[str, Any]):
    from treepo.sketches.adapters import make_hll_adapter

    return make_hll_adapter(
        backend=str(config.get("backend", "native")),
        precision=int(config.get("precision", 12)),
        hash_bits=int(config.get("hash_bits", 64)),
    )


def _factory_count_min(config: Mapping[str, Any]):
    from treepo.sketches.adapters import make_count_min_adapter

    return make_count_min_adapter(
        num_hashes=int(config.get("num_hashes", 5)),
        num_buckets=int(config.get("num_buckets", 256)),
    )


def _factory_theta(config: Mapping[str, Any]):
    from treepo.sketches.adapters import make_theta_adapter

    return make_theta_adapter(lg_k=int(config.get("lg_k", 12)))


def _factory_cpc(config: Mapping[str, Any]):
    from treepo.sketches.adapters import make_cpc_adapter

    return make_cpc_adapter(lg_k=int(config.get("lg_k", 10)))


def _factory_kll_floats(config: Mapping[str, Any]):
    from treepo.sketches.adapters import make_kll_floats_adapter

    return make_kll_floats_adapter(k=int(config.get("k", 200)))


# Map sketch_kind → adapter factory. New sketches added by appending one line.
_SKETCH_FACTORIES: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "hll": _factory_hll,
    "count_min": _factory_count_min,
    "theta": _factory_theta,
    "cpc": _factory_cpc,
    "kll_floats": _factory_kll_floats,
}


def _method_sketch(config: Mapping[str, Any]) -> CTreePOFitResult:
    """Run any classical sketch on token trees.

    - ``sketch_kind`` (required) — key in :data:`_SKETCH_FACTORIES`.
    - sketch-config (e.g. ``precision``, ``num_hashes``, ``lg_k``,
      ``k``) — forwarded to the adapter factory.
    - ``eval_data`` (optional) — caller-supplied trees override the
      auto token-tree fixture.
    """
    kind = config.get("sketch_kind")
    if not kind:
        available = ", ".join(sorted(_SKETCH_FACTORIES))
        raise ValueError(f"sketch requires config['sketch_kind']; available: {available}")
    factory = _SKETCH_FACTORIES.get(str(kind))
    if factory is None:
        available = ", ".join(sorted(_SKETCH_FACTORIES))
        raise ValueError(f"unknown sketch_kind {kind!r}; available: {available}")
    adapter = factory(config)

    eval_data = config.get("eval_data")
    if eval_data is None:
        from treepo.cld.fixtures import make_hll_token_trees

        eval_data = make_hll_token_trees(
            n_trees=int(config.get("n_trees", 8)),
            leaves_per_tree=int(config.get("leaves_per_tree", 4)),
            leaf_token_count=int(config.get("leaf_token_count", 24)),
            vocabulary_size=int(config.get("vocabulary_size", 200)),
            seed=int(config.get("seed", 0)),
        )

    backend_config: dict[str, Any] = {
        "sketch_adapter": adapter,
        "sketch_schedule": str(config.get("schedule", "balanced")),
    }
    if config.get("output_dir") is not None:
        backend_config["output_dir"] = str(config["output_dir"])
    spec = CTreePOLearningSpec(
        space_kind=f"sketch:{kind}",
        family="sketch",
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=list(eval_data),
        backend_config=backend_config,
        axis={"max_iterations": 0, "axis_value": 0},
    )
    return fit(spec)


# --------------------------------------------------------------------------- #
# Method 4: audit — post-hoc local-law audit on supplied rows.
# --------------------------------------------------------------------------- #


def _method_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the existing ``scripts/probe_clean_unified_no.py`` verbatim.

    Subprocess-dispatches the paper's standalone Markov-FNO probe with
    user-supplied CLI args, captures the ``summary.json`` it writes, and
    returns the parsed payload alongside the subprocess stdout/stderr.
    No reimplementation of the probe — it stays paper code.

    Required: ``output_root`` (where the probe writes its outputs).
    All other keys map 1:1 to the probe's argparse flags (kebab-case is
    auto-derived from snake_case).
    """
    import subprocess
    import sys as _sys

    output_root = config.get("output_root")
    if not output_root:
        raise ValueError("probe requires config['output_root']")
    out_dir = Path(str(output_root))
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "probe_clean_unified_no.py"
    if not script.exists():
        raise FileNotFoundError(f"probe script missing: {script}")

    cmd: list[str] = [_sys.executable, str(script), "--output-root", str(out_dir)]
    skip_keys = {"output_root", "timeout"}
    for k, v in config.items():
        if k in skip_keys:
            continue
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    timeout = float(config.get("timeout", 600.0))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "method": "probe",
        "command": cmd,
        "returncode": int(result.returncode),
        "summary": summary,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }


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


def list_sketch_kinds() -> tuple[str, ...]:
    """Sketch kinds registered for the ``"sketch"`` method."""
    return tuple(sorted(_SKETCH_FACTORIES))


def list_registered_oracles() -> tuple[str, ...]:
    return tuple(list_oracles())


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
    "sketch",
    _method_sketch,
    allowed_config_keys={
        "sketch_kind", "eval_data", "output_dir", "schedule",
        # Token-tree fixture knobs.
        "seed", "n_trees", "leaves_per_tree", "leaf_token_count", "vocabulary_size",
        # Adapter factory knobs (union — handler ignores keys not relevant to its kind).
        "backend", "precision", "hash_bits",
        "num_hashes", "num_buckets",
        "lg_k", "k",
    },
)
register_method(
    "audit",
    _method_audit,
    allowed_config_keys={"rows", "objective_mode", "gamma_depth", "output_dir"},
)
register_method(
    "probe",
    _method_probe,
    # Required: output_root. Every snake_case key here is forwarded as a
    # ``--snake-case`` CLI flag to scripts/probe_clean_unified_no.py.
    # Auto-generated from the probe's argparse surface — keep in sync with:
    #   grep 'parser.add_argument(' scripts/probe_clean_unified_no.py
    # (a drift test verifies this mirrors the argparse keyspace).
    allowed_config_keys={
        # Required + dispatch
        "output_root", "timeout",
        # Data / corpus
        "benchmark", "load_data_bundle", "doc_tokens", "expected_boundaries",
        "leaf_tokens", "train_docs", "eval_docs",
        # Architecture
        "channels", "g_n_modes", "g_n_layers",
        "scorer_n_modes", "scorer_n_layers", "leaf_pool",
        # Training-loop
        "epochs", "batch_size", "lr", "optimizer", "weight_decay",
        "lr_schedule", "grad_clip", "seed", "device", "gpu",
        # Objective
        "training_objective", "root_only", "enable_contextual_sufficiency",
        # Contextual-sufficiency knobs
        "context_samples_per_doc", "contextual_loss_weight",
        "contextual_dependence_objective", "contextual_response_regressor",
        "response_signature_contexts", "response_signature_slices",
        "infomax_loss_weight",
        # Markov local-laws knobs
        "markov_law_weight", "markov_law_readout", "markov_law_count_weight",
        "markov_law_edge_weight", "markov_law_idempotence_weight",
        "markov_law_leaf_weight", "markov_law_merge_weight",
        # Markov witness-supervision ablation knobs
        "markov_witness_weight", "markov_witness_readout",
        "markov_witness_count_weight", "markov_witness_edge_weight",
        "markov_witness_epochs", "markov_witness_lr",
        "run_markov_witness_supervision_ablation",
        # Boundary-supervision ablation knobs
        "run_boundary_supervision_ablation",
        "boundary_supervision_epochs", "boundary_supervision_lr",
        "boundary_supervision_weight",
        # Diagnostic baselines
        "diagnostic_baselines", "diagnostic_baseline_epochs",
        "diagnostic_baseline_batch_size", "diagnostic_baseline_lr",
        # Misc
        "exact_witness_n_regimes", "palette_ridge_alpha",
        "require_exact_contract_zero",
    },
)


__all__ = [
    "allowed_config_keys",
    "list_methods",
    "list_oracle_domains_with_fixtures",
    "list_registered_oracles",
    "list_sketch_kinds",
    "method_info",
    "register_method",
    "run",
]
