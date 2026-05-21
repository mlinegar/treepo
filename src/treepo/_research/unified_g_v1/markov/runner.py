from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from treepo._research.unified_g_v1.core.contracts import (
    MarkovRunSpec,
    MarkovScope,
    Profile,
    SupervisionPolicy,
)
from treepo._research.unified_g_v1.core.manifest import now_iso, write_json
from treepo._research.unified_g_v1.markov.benchmarks import DOC_TOKENS, ScopeBenchmark, materialize_scope_bundle
from treepo._research.unified_g_v1.markov.program import MarkovUnifiedGBinding, resolve_markov_unified_g_binding

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    OPSCountSummary,
    run_markov_changepoint_ops_count_experiment,
)


@dataclass(frozen=True)
class MarkovRunRecord:
    spec: MarkovRunSpec
    scope_label: str
    benchmark_name: str
    benchmark_cell_id: str
    bundle_source: str
    run_dir: Path
    summary_path: Path
    config: Mapping[str, Any]
    extracted_metrics: Mapping[str, Any]
    training_geometry: Mapping[str, Any]
    objective: Mapping[str, Any]
    metrics: Mapping[str, Any]
    estimator_diagnostics: Mapping[str, Any]
    local_law_learnability: Mapping[str, Any]
    g_artifacts: Mapping[str, Any]
    program_contract: Mapping[str, Any]

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "run_key": self.spec.run_key,
            "scope": self.spec.scope.value,
            "scope_label": self.scope_label,
            "benchmark_name": self.benchmark_name,
            "benchmark_cell_id": self.benchmark_cell_id,
            "bundle_source": self.bundle_source,
            "summary_path": str(self.summary_path),
            "run_dir": str(self.run_dir),
            "spec": _spec_payload(self.spec),
            "extracted_metrics": dict(self.extracted_metrics),
            "program_contract": dict(self.program_contract),
        }


def _safe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return float(out)


def _spec_payload(spec: MarkovRunSpec) -> dict[str, Any]:
    return {
        "scope": spec.scope.value,
        "train_docs": int(spec.train_docs),
        "root_share": int(spec.root_share),
        "leaf_tokens": int(spec.leaf_tokens),
        "supervision_policy": spec.supervision_policy.value,
        "profile": spec.profile.value,
        "seed": int(spec.seed),
        "comparator_policy": spec.comparator_policy.value,
        "run_key": spec.run_key,
    }


def build_ops_config(
    spec: MarkovRunSpec,
    *,
    scope_benchmark: ScopeBenchmark,
    use_cuda: bool,
    cuda_device: int | None = None,
    torch_threads: int = 0,
    binding: MarkovUnifiedGBinding | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> OPSCountConfig:
    benchmark = scope_benchmark.benchmark
    benchmark_overrides = dict(getattr(benchmark, "config_overrides", {}) or {})
    resolved_binding = binding or resolve_markov_unified_g_binding(spec)
    config_kwargs: dict[str, Any] = {
        "generator_profile": str(benchmark_overrides.get("generator_profile", "piecewise_markov")),
        "n_regimes": int(benchmark_overrides.get("n_regimes", 4)),
        "vocab_size": int(benchmark_overrides.get("vocab_size", 16)),
        "min_tokens": int(benchmark_overrides.get("min_tokens", DOC_TOKENS)),
        "max_tokens": int(benchmark_overrides.get("max_tokens", DOC_TOKENS)),
        "min_segments": int(benchmark_overrides.get("min_segments", 2)),
        "max_segments": int(benchmark_overrides.get("max_segments", 6)),
        "min_seg_len": int(benchmark_overrides.get("min_seg_len", 8)),
        "max_seg_len": int(benchmark_overrides.get("max_seg_len", 32)),
        "fixed_leaf_tokens": int(spec.leaf_tokens),
        "train_docs": int(spec.train_docs),
        "val_docs": int(benchmark_overrides.get("val_docs", 128)),
        "test_docs": int(benchmark_overrides.get("test_docs", 256)),
        "model_family": "neural",
        "feature_mode": "token_full",
        "seed": int(spec.seed),
        "data_seed": 0,
        "model_seed": int(spec.seed),
        "use_cuda": bool(use_cuda),
        "cuda_device": int(cuda_device) if cuda_device is not None else None,
        "torch_threads": int(torch_threads),
        "preserve_requested_leaf_tokens": True,
        "official_fno_preserve_requested_leaf_tokens": True,
        "comparison_mode": "legacy",
    }
    if benchmark_overrides.get("hazard_switch_prob") is not None:
        config_kwargs["hazard_switch_prob"] = float(benchmark_overrides["hazard_switch_prob"])
    for field_name in ("min_distinct_regimes_per_doc", "max_distinct_regimes_per_doc"):
        if benchmark_overrides.get(field_name) is not None:
            config_kwargs[field_name] = int(benchmark_overrides[field_name])
    config = OPSCountConfig(**config_kwargs)
    config = resolved_binding.apply(config)
    if config_overrides:
        config = OPSCountConfig(**{**asdict(config), **dict(config_overrides)})
    return config


def _extract_metrics(summary: OPSCountSummary) -> dict[str, Any]:
    learned = dict(summary.metrics.get("learned", {}) or {})
    learned_root_only_view = dict(summary.metrics.get("learned_root_only_view_test", {}) or {})
    official_fno = dict(summary.metrics.get("fno", {}) or {})
    return {
        "learned_root_mae": _safe_float(learned.get("root_mae")),
        "learned_leaf_mae": _safe_float(learned.get("leaf_mae")),
        "learned_merge_mae": _safe_float(learned.get("merge_mae")),
        "learned_root_only_view_root_mae": _safe_float(learned_root_only_view.get("root_mae")),
        "learned_root_only_view_leaf_mae": _safe_float(learned_root_only_view.get("leaf_mae")),
        "official_fno_root_mae": _safe_float(official_fno.get("root_mae")),
        "official_fno_leaf_mae": _safe_float(official_fno.get("leaf_mae")),
        "exact_root_mae": _safe_float((summary.metrics.get("exact", {}) or {}).get("root_mae")),
    }


def _write_run_payload(
    *,
    run_dir: Path,
    spec: MarkovRunSpec,
    scope_benchmark: ScopeBenchmark,
    bundle_source: str,
    config: OPSCountConfig,
    summary: OPSCountSummary,
    binding: MarkovUnifiedGBinding,
) -> dict[str, Any]:
    extracted_metrics = _extract_metrics(summary)
    payload = {
        "generated_at": now_iso(),
        "run_key": spec.run_key,
        "scope": scope_benchmark.scope.value,
        "scope_label": scope_benchmark.scope_label,
        "benchmark": {
            "name": scope_benchmark.benchmark.name,
            "cell_id": str(getattr(scope_benchmark.benchmark, "cell_id", "") or ""),
            "description": str(getattr(scope_benchmark.benchmark, "description", "") or ""),
            "observed_token_profile": str(
                getattr(scope_benchmark.benchmark, "observed_token_profile", "") or ""
            ),
        },
        "bundle_source": str(bundle_source),
        "spec": _spec_payload(spec),
        "config": asdict(config),
        "extracted_metrics": dict(extracted_metrics),
        "training_geometry": dict(summary.training_geometry),
        "objective": dict(summary.objective),
        "metrics": dict(summary.metrics),
        "estimator_diagnostics": dict(summary.estimator_diagnostics),
        "local_law_learnability": dict(summary.local_law_learnability),
        "g_artifacts": dict(summary.g_artifacts),
        "program_contract": binding.contract.to_dict(),
    }
    write_json(run_dir / "config.json", payload["config"])
    write_json(run_dir / "summary.json", payload)
    return payload


def _load_saved_run(summary_path: Path) -> MarkovRunRecord:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    spec_payload = dict(payload.get("spec") or {})
    spec = MarkovRunSpec(
        scope=MarkovScope.parse(str(spec_payload.get("scope", ""))),
        train_docs=int(spec_payload.get("train_docs", 0)),
        root_share=int(spec_payload.get("root_share", 0)),
        leaf_tokens=int(spec_payload.get("leaf_tokens", 0)),
        supervision_policy=SupervisionPolicy.parse(
            str(spec_payload.get("supervision_policy", ""))
        ),
        profile=Profile.parse(str(spec_payload.get("profile", ""))),
        seed=int(spec_payload.get("seed", 0)),
    )
    return MarkovRunRecord(
        spec=spec,
        scope_label=str(payload.get("scope_label", "") or ""),
        benchmark_name=str(((payload.get("benchmark") or {}).get("name", "")) or ""),
        benchmark_cell_id=str(((payload.get("benchmark") or {}).get("cell_id", "")) or ""),
        bundle_source=str(payload.get("bundle_source", "") or ""),
        run_dir=summary_path.parent,
        summary_path=summary_path,
        config=dict(payload.get("config") or {}),
        extracted_metrics=dict(payload.get("extracted_metrics") or {}),
        training_geometry=dict(payload.get("training_geometry") or {}),
        objective=dict(payload.get("objective") or {}),
        metrics=dict(payload.get("metrics") or {}),
        estimator_diagnostics=dict(payload.get("estimator_diagnostics") or {}),
        local_law_learnability=dict(payload.get("local_law_learnability") or {}),
        g_artifacts=dict(payload.get("g_artifacts") or {}),
        program_contract=dict(payload.get("program_contract") or {}),
    )


def run_markov_spec(
    spec: MarkovRunSpec,
    *,
    output_root: str | Path,
    use_cuda: bool = True,
    cuda_device: int | None = None,
    torch_threads: int = 0,
    reuse_existing: bool = True,
    config_overrides: Mapping[str, Any] | None = None,
) -> MarkovRunRecord:
    output_root = Path(output_root).expanduser()
    run_dir = output_root / "runs" / spec.run_key
    summary_path = run_dir / "summary.json"
    if reuse_existing and summary_path.exists():
        return _load_saved_run(summary_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    scope_benchmark, data_bundle, bundle_source = materialize_scope_bundle(
        spec.scope,
        train_docs=int(spec.train_docs),
        output_root=output_root / "_generated_bundles",
    )
    binding = resolve_markov_unified_g_binding(spec)
    config = build_ops_config(
        spec,
        scope_benchmark=scope_benchmark,
        use_cuda=bool(use_cuda),
        cuda_device=cuda_device,
        torch_threads=int(torch_threads),
        binding=binding,
        config_overrides=config_overrides,
    )
    summary = run_markov_changepoint_ops_count_experiment(config, data_bundle=data_bundle)
    _write_run_payload(
        run_dir=run_dir,
        spec=spec,
        scope_benchmark=scope_benchmark,
        bundle_source=bundle_source,
        config=config,
        summary=summary,
        binding=binding,
    )
    return _load_saved_run(run_dir / "summary.json")


def _default_report_specs(
    *,
    train_docs: int,
    root_shares: Sequence[int],
    leaf_tokens: Sequence[int],
    scopes: Sequence[MarkovScope],
    seed: int,
    include_duplicate_local_label_one_leaf: bool,
) -> list[MarkovRunSpec]:
    specs: list[MarkovRunSpec] = []
    for scope in scopes:
        for root_share in root_shares:
            for leaf in leaf_tokens:
                specs.append(
                    MarkovRunSpec(
                        scope=scope,
                        train_docs=int(train_docs),
                        root_share=int(root_share),
                        leaf_tokens=int(leaf),
                        supervision_policy=SupervisionPolicy.ROOT_ONLY,
                        profile=Profile.ROOT_ONLY,
                        seed=int(seed),
                    )
                )
                specs.append(
                    MarkovRunSpec(
                        scope=scope,
                        train_docs=int(train_docs),
                        root_share=int(root_share),
                        leaf_tokens=int(leaf),
                        supervision_policy=SupervisionPolicy.LEAF_MASS_EQ,
                        profile=Profile.STANDARD,
                        seed=int(seed),
                    )
                )
            specs.append(
                MarkovRunSpec(
                    scope=scope,
                    train_docs=int(train_docs),
                    root_share=int(root_share),
                    leaf_tokens=128,
                    supervision_policy=SupervisionPolicy.ROOT_ONLY,
                    profile=Profile.FNO_CANARY,
                    seed=int(seed),
                )
            )
            if include_duplicate_local_label_one_leaf:
                specs.append(
                    MarkovRunSpec(
                        scope=scope,
                        train_docs=int(train_docs),
                        root_share=int(root_share),
                        leaf_tokens=128,
                        supervision_policy=SupervisionPolicy.ROOT_ONLY,
                        profile=Profile.DUPLICATE_LOCAL_LABEL_ONE_LEAF,
                        seed=int(seed),
                    )
                )
    return specs


def run_fixed_report_suite(
    *,
    output_root: str | Path,
    train_docs: int = 10240,
    scopes: Sequence[MarkovScope] = (
        MarkovScope.RECOVERABLE_V5_T128,
        MarkovScope.R12_P079,
    ),
    root_shares: Sequence[int] = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10),
    leaf_tokens: Sequence[int] = (128, 64, 32, 16, 8),
    seed: int = 0,
    include_duplicate_local_label_one_leaf: bool = True,
    use_cuda: bool = True,
    cuda_device: int | None = None,
    torch_threads: int = 0,
    reuse_existing: bool = True,
    config_overrides: Mapping[str, Any] | None = None,
) -> list[MarkovRunRecord]:
    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    from treepo._research.unified_g_v1.training import TrainerConfig, fit

    results = [
        fit(
            trainer_config=TrainerConfig(
                run_spec=spec,
                use_cuda=bool(use_cuda),
                cuda_device=cuda_device,
                torch_threads=int(torch_threads),
                reuse_existing=bool(reuse_existing),
                config_overrides=config_overrides,
            ),
            output_dir=output_root,
        ).summary["record"]
        for spec in _default_report_specs(
            train_docs=int(train_docs),
            root_shares=root_shares,
            leaf_tokens=leaf_tokens,
            scopes=scopes,
            seed=int(seed),
            include_duplicate_local_label_one_leaf=bool(
                include_duplicate_local_label_one_leaf
            ),
        )
    ]
    write_json(
        output_root / "run_manifest.json",
        {
            "generated_at": now_iso(),
            "output_root": str(output_root),
            "train_doc_count": int(train_docs),
            "root_shares": [int(value) for value in root_shares],
            "leaf_tokens": [int(value) for value in leaf_tokens],
            "scopes": [scope.value for scope in scopes],
            "run_count": len(results),
            "runs": [record.to_manifest_entry() for record in results],
        },
    )
    return results
