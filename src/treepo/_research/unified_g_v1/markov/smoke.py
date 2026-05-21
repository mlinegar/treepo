from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo._research.unified_g_v1.core.contracts import MarkovRunSpec, MarkovScope, Profile, SupervisionPolicy
from treepo._research.unified_g_v1.core.manifest import now_iso, write_json
from treepo._research.unified_g_v1.markov.runner import MarkovRunRecord, run_markov_spec
from treepo._research.unified_g_v1.training import TrainerConfig, fit


def default_markov_smoke_specs(
    *,
    train_docs: int = 1024,
    seed: int = 0,
    scope: MarkovScope = MarkovScope.RECOVERABLE_V5_T128,
) -> list[MarkovRunSpec]:
    return [
        MarkovRunSpec(
            scope=scope,
            train_docs=int(train_docs),
            root_share=100,
            leaf_tokens=128,
            supervision_policy=SupervisionPolicy.ROOT_ONLY,
            profile=Profile.ROOT_ONLY,
            seed=int(seed),
        ),
        MarkovRunSpec(
            scope=scope,
            train_docs=int(train_docs),
            root_share=50,
            leaf_tokens=32,
            supervision_policy=SupervisionPolicy.LEAF_MASS_EQ,
            profile=Profile.STANDARD,
            seed=int(seed),
        ),
        MarkovRunSpec(
            scope=scope,
            train_docs=int(train_docs),
            root_share=100,
            leaf_tokens=128,
            supervision_policy=SupervisionPolicy.ROOT_ONLY,
            profile=Profile.FNO_CANARY,
            seed=int(seed),
        ),
    ]


def smoke_config_overrides(spec: MarkovRunSpec) -> dict[str, Any]:
    base = {
        "val_docs": 16,
        "test_docs": 16,
        "batch_size": 32,
        "hidden_dim": 64,
        "state_dim": 32,
        "tree_leaf_fno_width": 32,
        "tree_leaf_fno_n_layers": 1,
        "tree_leaf_fno_n_modes": 4,
        "tree_theorem_feature_hidden_dim": 64,
        "tree_theorem_feature_dim": 16,
        "tree_theorem_score_dim": 1,
        "tree_theorem_fiber_dim": 15,
        "tree_theorem_aux_dim": 0,
        "tree_theorem_count_dim": 4,
        "tree_theorem_first_dim": 4,
        "tree_theorem_last_dim": 4,
        "tree_stage1_eval_mode": "end_only",
    }
    if spec.profile == Profile.STANDARD:
        return {
            **base,
            "n_epochs": 2,
            "tree_stage1_epochs": 1,
            "tree_stage2_epochs": 1,
        }
    return {
        **base,
        "n_epochs": 1,
        "tree_stage1_epochs": 0,
        "tree_stage2_epochs": 0,
    }


def run_markov_smoke_suite(
    *,
    output_root: str | Path,
    train_docs: int = 1024,
    seed: int = 0,
    scope: MarkovScope = MarkovScope.RECOVERABLE_V5_T128,
    use_cuda: bool = False,
    cuda_device: int | None = None,
    torch_threads: int = 1,
    reuse_existing: bool = True,
    specs: Sequence[MarkovRunSpec] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_specs = (
        default_markov_smoke_specs(train_docs=train_docs, seed=seed, scope=scope)
        if specs is None
        else list(specs)
    )
    records: list[MarkovRunRecord] = []
    runs_payload: list[dict[str, Any]] = []
    for spec in resolved_specs:
        overrides = smoke_config_overrides(spec)
        fit_result = fit(
            trainer_config=TrainerConfig(
                run_spec=spec,
                use_cuda=bool(use_cuda),
                cuda_device=cuda_device,
                torch_threads=int(torch_threads),
                reuse_existing=bool(reuse_existing),
                config_overrides=overrides,
            ),
            output_dir=output_root,
        )
        record: MarkovRunRecord = fit_result.summary["record"]
        records.append(record)
        runs_payload.append(
            {
                **record.to_manifest_entry(),
                "smoke_config_overrides": dict(overrides),
            }
        )
    payload = {
        "generated_at": now_iso(),
        "output_root": str(output_root),
        "train_docs": int(train_docs),
        "seed": int(seed),
        "scope": scope.value,
        "run_count": len(records),
        "runs": runs_payload,
    }
    write_json(output_root / "smoke_manifest.json", payload)
    return payload
