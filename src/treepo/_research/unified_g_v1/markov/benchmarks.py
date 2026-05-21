from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from treepo._research.unified_g_v1.core.contracts import MarkovScope
from treepo._research.unified_g_v1.core.manifest import write_json

from treepo._research.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    FullDocDiagnosticBenchmarkSpec,
    resolve_full_doc_diagnostic_benchmark,
)
from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    build_markov_changepoint_ops_count_data_bundle,
)


DOC_TOKENS = 128


@dataclass(frozen=True)
class ScopeBenchmark:
    scope: MarkovScope
    benchmark: FullDocDiagnosticBenchmarkSpec
    scope_label: str
    scope_subtitle: str
    figure_prefix: str

    def figure_filename(self, *, train_docs: int, with_leaf_mass_eq: bool) -> str:
        suffix = (
            f"{self.figure_prefix}_with_leaf_mass_equivalent_train{int(train_docs)}.png"
            if with_leaf_mass_eq
            else f"{self.figure_prefix}_train{int(train_docs)}.png"
        )
        return suffix


_SCOPE_META: dict[MarkovScope, dict[str, str]] = {
    MarkovScope.RECOVERABLE_V5_T128: {
        "benchmark_name": "recoverable_v5_t128",
        "scope_label": "Counting Topic Changes (Simple Case)",
        "scope_subtitle": (
            "128-token sticky recoverable benchmark with 4 hidden regimes and "
            "about 5 expected regime changes per document."
        ),
        "figure_prefix": "recoverable_root_only_leaf_size_fixed",
    },
    MarkovScope.R12_P079: {
        "benchmark_name": "structural_core_v2::r12_p079",
        "scope_label": "Counting Topic Changes (Structural Case)",
        "scope_subtitle": (
            "128-token structural benchmark with 12 hidden regimes and about "
            "10 expected regime changes per document."
        ),
        "figure_prefix": "structural_root_only_leaf_size_fixed",
    },
}


def resolve_scope_benchmark(scope: MarkovScope | str) -> ScopeBenchmark:
    resolved_scope = scope if isinstance(scope, MarkovScope) else MarkovScope.parse(str(scope))
    meta = dict(_SCOPE_META[resolved_scope])
    benchmark = resolve_full_doc_diagnostic_benchmark(str(meta["benchmark_name"]))
    return ScopeBenchmark(
        scope=resolved_scope,
        benchmark=benchmark,
        scope_label=str(meta["scope_label"]),
        scope_subtitle=str(meta["scope_subtitle"]),
        figure_prefix=str(meta["figure_prefix"]),
    )


def _slice_bundle_train_docs(
    bundle: MarkovOPSDataBundle,
    *,
    train_docs: int,
) -> MarkovOPSDataBundle:
    if int(train_docs) > int(len(bundle.train_docs)):
        raise ValueError(
            f"train_docs={int(train_docs)} exceeds available bundle size "
            f"{len(bundle.train_docs)}"
        )
    selected_train = tuple(bundle.train_docs[: int(train_docs)])
    signature = hashlib.sha256()
    for doc in selected_train:
        payload = {
            "tokens": [int(x) for x in doc.tokens],
            "token_regimes": [int(x) for x in doc.token_regimes],
            "transition_regimes": [int(x) for x in doc.transition_regimes],
            "true_boundaries": [int(x) for x in doc.true_boundaries],
        }
        signature.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        signature.update(b"\n")
    return MarkovOPSDataBundle(
        train_docs=selected_train,
        val_docs=tuple(bundle.val_docs),
        test_docs=tuple(bundle.test_docs),
        train_corpus_signature=signature.hexdigest(),
        val_corpus_signature=str(bundle.val_corpus_signature),
        test_corpus_signature=str(bundle.test_corpus_signature),
    )


def _preferred_existing_bundle(
    benchmark: FullDocDiagnosticBenchmarkSpec,
    *,
    train_docs: int,
) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    canonical = str(getattr(benchmark, "canonical_bundle_path", "") or "").strip()
    expanded = str(getattr(benchmark, "expanded_bundle_path", "") or "").strip()
    canonical_capacity = int(getattr(benchmark, "canonical_train_docs_capacity", 0) or 0)
    expanded_capacity = int(getattr(benchmark, "expanded_train_docs_capacity", 0) or 0)
    if canonical:
        path = Path(canonical).expanduser()
        if path.exists() and canonical_capacity >= int(train_docs):
            candidates.append((canonical_capacity, path))
    if expanded:
        path = Path(expanded).expanduser()
        if path.exists() and expanded_capacity >= int(train_docs):
            candidates.append((expanded_capacity, path))
    if not candidates:
        for raw in (expanded, canonical):
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.exists():
                return path
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _bundle_generation_config(
    benchmark: FullDocDiagnosticBenchmarkSpec,
    *,
    train_docs: int,
) -> OPSCountConfig:
    overrides = dict(getattr(benchmark, "config_overrides", {}) or {})
    hazard_switch_prob = float(
        overrides.get(
            "hazard_switch_prob",
            getattr(benchmark, "hazard_switch_prob", float("nan")),
        )
    )
    config_kwargs = {
        "generator_profile": str(overrides.get("generator_profile", "piecewise_markov")),
        "n_regimes": int(overrides.get("n_regimes", 4)),
        "vocab_size": int(overrides.get("vocab_size", 16)),
        "min_tokens": int(overrides.get("min_tokens", DOC_TOKENS)),
        "max_tokens": int(overrides.get("max_tokens", DOC_TOKENS)),
        "min_segments": int(overrides.get("min_segments", 2)),
        "max_segments": int(overrides.get("max_segments", 6)),
        "min_seg_len": int(overrides.get("min_seg_len", 8)),
        "max_seg_len": int(overrides.get("max_seg_len", 32)),
        "train_docs": int(train_docs),
        "val_docs": int(overrides.get("val_docs", 128)),
        "test_docs": int(overrides.get("test_docs", 256)),
        "fixed_leaf_tokens": DOC_TOKENS,
        "use_cuda": False,
        "torch_threads": 1,
        "seed": 0,
        "data_seed": 0,
        "model_seed": 0,
    }
    if math.isfinite(hazard_switch_prob):
        config_kwargs["hazard_switch_prob"] = float(hazard_switch_prob)
    for field_name in ("min_distinct_regimes_per_doc", "max_distinct_regimes_per_doc"):
        if field_name in overrides and overrides[field_name] is not None:
            config_kwargs[field_name] = int(overrides[field_name])
    return OPSCountConfig(**config_kwargs)


def materialize_scope_bundle(
    scope: MarkovScope | str,
    *,
    train_docs: int,
    output_root: str | Path | None = None,
) -> tuple[ScopeBenchmark, MarkovOPSDataBundle, str]:
    scope_benchmark = resolve_scope_benchmark(scope)
    benchmark = scope_benchmark.benchmark
    existing_path = _preferred_existing_bundle(benchmark, train_docs=int(train_docs))
    if existing_path is not None:
        loaded = MarkovOPSDataBundle.load(existing_path)
        sliced = _slice_bundle_train_docs(loaded, train_docs=int(train_docs))
        return (
            scope_benchmark,
            sliced,
            f"{existing_path}::train_prefix_{int(train_docs)}",
        )

    generated_root = (
        Path(output_root).expanduser()
        if output_root is not None
        else Path.cwd() / "outputs" / "unified_g_v1_generated_bundles"
    )
    generated_root.mkdir(parents=True, exist_ok=True)
    generated_path = generated_root / f"{benchmark.name}_train{int(train_docs)}.json"
    if generated_path.exists():
        loaded = MarkovOPSDataBundle.load(generated_path)
        sliced = _slice_bundle_train_docs(loaded, train_docs=int(train_docs))
        return scope_benchmark, sliced, str(generated_path)

    generation_cfg = _bundle_generation_config(benchmark, train_docs=int(train_docs))
    generated = build_markov_changepoint_ops_count_data_bundle(generation_cfg)
    generated.save(generated_path)
    write_json(
        generated_path.with_suffix(".manifest.json"),
        {
            "scope": scope_benchmark.scope.value,
            "benchmark_name": benchmark.name,
            "generated_bundle_path": str(generated_path),
            "train_docs": int(train_docs),
            "generator_config": generation_cfg.__dict__,
        },
    )
    sliced = _slice_bundle_train_docs(generated, train_docs=int(train_docs))
    return scope_benchmark, sliced, str(generated_path)
