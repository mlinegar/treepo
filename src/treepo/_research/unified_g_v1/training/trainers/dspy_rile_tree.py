"""Manifesto RILE tree-structured DSPy bootstrap trainer.

This is the live DSPy-GEPA path for the tree reward defined in
`realdoc.rile_tree`. It trains the `RileTreeProgram` end-to-end against a
`ManifestoRileTreeOracle`, so GEPA reflects on leaf summaries, merge prompts,
and the scorer together instead of optimizing only a flat excerpt scorer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from treepo._research.unified_g_v1.training.trainers import register_trainer


def dspy_rile_tree_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    import dspy  # lazy; only needed here

    from treepo._research.unified_g_v1.eval.law_stress import (
        DEFAULT_PRIMARY_GAIN_THRESHOLD,
        gain_frac as _gain_frac,
    )
    from treepo._research.unified_g_v1.realdoc.rile_tree_program import RileTreeProgram
    from treepo._research.unified_g_v1.training.fit import FitResult
    from treepo._research.unified_g_v1.training.oracles.manifesto_rile_tree import tree_scaffold_from_example
    from treepo._research.unified_g_v1.training.trainers.rile_tree_feedback import (
        dspy_gepa_metric_from_rollout,
        tree_rile_reward_for_rollout,
    )

    if cfg.oracle is None:
        raise ValueError("dspy_rile_tree_trainer requires cfg.oracle")
    if not cfg.model_name:
        raise ValueError("dspy_rile_tree_trainer requires cfg.model_name (served-model-name on the vLLM)")

    extra = dict(cfg.extra or {})
    api_base = str(extra.get("api_base", "http://localhost:8000/v1")).rstrip("/")
    api_key = str(extra.get("api_key", "EMPTY"))
    temperature = float(extra.get("temperature", 0.0))
    max_tokens = int(extra.get("max_tokens", 1024))
    max_bootstrapped = int(extra.get("max_bootstrapped_demos", 4))
    max_labeled = int(extra.get("max_labeled_demos", 0))
    max_train_examples = int(extra.get("max_train_examples", 0))
    n_val_cap = int(extra.get("n_val", 0))
    optimizer_name = str(extra.get("optimizer", "gepa")).lower()
    gepa_auto = str(extra.get("gepa_auto", "medium"))
    gepa_num_threads = int(extra.get("gepa_num_threads", 16))
    gepa_max_metric_calls = int(extra.get("gepa_max_metric_calls", 0))
    gepa_minibatch_size = int(extra.get("gepa_reflection_minibatch_size", 3))
    gepa_valset_cap = int(extra.get("gepa_valset_cap", 64))
    gepa_resume_log_dir = extra.get("gepa_resume_log_dir") or None
    reflection_api_base = str(extra.get("reflection_api_base", "") or api_base).rstrip("/")
    reflection_model_name = str(extra.get("reflection_model_name", "") or cfg.model_name)
    reflection_max_tokens = int(extra.get("reflection_max_tokens", 16384))
    local_law_weight = float(extra.get("local_law_weight", 0.3))
    c1_relative_weight = float(extra.get("c1_relative_weight", 1.0))
    c2_relative_weight = float(extra.get("c2_relative_weight", 1.0))
    c3_relative_weight = float(extra.get("c3_relative_weight", 1.0))

    lm = dspy.LM(
        model=f"openai/{cfg.model_name}",
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)

    reflection_lm = dspy.LM(
        model=f"openai/{reflection_model_name}",
        api_base=reflection_api_base,
        api_key=api_key,
        temperature=0.7,
        max_tokens=reflection_max_tokens,
    )

    program = RileTreeProgram()

    def _to_examples(raw_items):
        built = []
        for ex in raw_items:
            scaffold = tree_scaffold_from_example(ex)
            built.append(
                dspy.Example(
                    scaffold=scaffold,
                    doc_id=str(ex.extra.get("doc_id", scaffold.doc_id)),
                    root_rile=float(ex.target),
                ).with_inputs("scaffold")
            )
        return built

    trainset = _to_examples(cfg.oracle.train_examples())
    valset = _to_examples(cfg.oracle.val_examples())
    if max_train_examples > 0:
        trainset = trainset[:max_train_examples]
    if n_val_cap > 0:
        valset = valset[:n_val_cap]

    if trainset and valset:
        train_mean = sum(float(ex.root_rile) for ex in trainset) / float(len(trainset))
        baseline_val_mae = sum(
            abs(train_mean - float(ex.root_rile)) for ex in valset
        ) / float(len(valset))
    else:
        baseline_val_mae = 0.0

    def _reward(rollout):
        return tree_rile_reward_for_rollout(
            rollout,
            local_law_weight=local_law_weight,
            c1_relative_weight=c1_relative_weight,
            c2_relative_weight=c2_relative_weight,
            c3_relative_weight=c3_relative_weight,
        )

    def _bootstrap_metric(example, prediction, trace=None) -> float:
        del example, trace
        return float(_reward(prediction).score)

    def _evaluate(examples) -> dict[str, Any]:
        root_errors: list[float] = []
        rewards: list[float] = []
        leaf_maes: list[float] = []
        merge_maes: list[float] = []
        commutativity_maes: list[float] = []
        predictions: list[dict[str, Any]] = []
        strict_leaf_docs = 0
        strict_merge_docs = 0
        commutativity_docs = 0
        for ex in examples:
            rollout = program(scaffold=ex.scaffold)
            reward = _reward(rollout)
            root_errors.append(float(reward.root_mae))
            rewards.append(float(reward.score))
            if reward.n_leaves_strict:
                leaf_maes.append(float(reward.leaf_mae))
                strict_leaf_docs += 1
            if reward.n_merges_strict:
                merge_maes.append(float(reward.merge_mae))
                strict_merge_docs += 1
            if reward.n_commutativity_samples:
                commutativity_maes.append(float(reward.commutativity_mae))
                commutativity_docs += 1
            predictions.append(
                {
                    "doc_id": str(ex.doc_id),
                    "root_prediction": float(rollout.root_prediction),
                    "root_target": float(ex.root_rile),
                    "root_abs_error": float(reward.root_mae),
                    "reward": float(reward.score),
                    "strict_leaf_count": int(reward.n_leaves_strict),
                    "strict_merge_count": int(reward.n_merges_strict),
                    "commutativity_count": int(reward.n_commutativity_samples),
                }
            )
        root_mae = float(sum(root_errors) / len(root_errors)) if root_errors else float("nan")
        reward_mean = float(sum(rewards) / len(rewards)) if rewards else float("nan")
        out = {
            "count": int(len(examples)),
            "root_mae": root_mae,
            "reward_mean": reward_mean,
            "leaf_mae_mean": (
                float(sum(leaf_maes) / len(leaf_maes)) if leaf_maes else float("nan")
            ),
            "merge_mae_mean": (
                float(sum(merge_maes) / len(merge_maes)) if merge_maes else float("nan")
            ),
            "commutativity_mae_mean": (
                float(sum(commutativity_maes) / len(commutativity_maes))
                if commutativity_maes
                else float("nan")
            ),
            "docs_with_strict_leaves": int(strict_leaf_docs),
            "docs_with_strict_merges": int(strict_merge_docs),
            "docs_with_commutativity": int(commutativity_docs),
            "predictions": predictions,
        }
        if baseline_val_mae > 0.0 and root_errors:
            gf = _gain_frac(model_mae=root_mae, baseline_mae=baseline_val_mae)
            out["baseline_val_mae"] = float(baseline_val_mae)
            out["val_mae_gain_frac"] = float(gf)
            out["val_mae_pass"] = 1.0 if gf >= DEFAULT_PRIMARY_GAIN_THRESHOLD else 0.0
        return out

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_val = _evaluate(valset) if valset else {"count": 0, "root_mae": None}
    (output_dir / "baseline_val_eval.json").write_text(
        json.dumps(baseline_val, indent=2),
        encoding="utf-8",
    )

    if optimizer_name == "gepa":
        def _gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            del gold, trace, pred_name, pred_trace
            return dspy_gepa_metric_from_rollout(
                pred,
                local_law_weight=local_law_weight,
                c1_relative_weight=c1_relative_weight,
                c2_relative_weight=c2_relative_weight,
                c3_relative_weight=c3_relative_weight,
            )

        gepa_log_dir = output_dir / "gepa_logs"
        gepa_log_dir.mkdir(parents=True, exist_ok=True)
        if gepa_resume_log_dir:
            import shutil

            src = Path(str(gepa_resume_log_dir)) / "gepa_state.bin"
            if src.exists():
                shutil.copy2(src, gepa_log_dir / "gepa_state.bin")
        gepa_kwargs: dict[str, Any] = dict(
            metric=_gepa_metric,
            reflection_lm=reflection_lm,
            num_threads=gepa_num_threads,
            reflection_minibatch_size=gepa_minibatch_size,
            candidate_selection_strategy="pareto",
            use_merge=True,
            max_merge_invocations=5,
            track_stats=True,
            add_format_failure_as_feedback=True,
            log_dir=str(gepa_log_dir),
            seed=int(cfg.seed),
        )
        if gepa_max_metric_calls > 0:
            gepa_kwargs["max_metric_calls"] = int(gepa_max_metric_calls)
        else:
            gepa_kwargs["auto"] = gepa_auto
        teleprompter = dspy.teleprompt.GEPA(**gepa_kwargs)
        compiled_program = teleprompter.compile(
            program,
            trainset=trainset,
            valset=list(valset[:gepa_valset_cap]) if gepa_valset_cap > 0 else valset,
        )
    else:
        teleprompter = dspy.teleprompt.BootstrapFewShot(
            metric=_bootstrap_metric,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )
        compiled_program = teleprompter.compile(program, trainset=trainset)

    program = compiled_program
    final_val = _evaluate(valset) if valset else {"count": 0, "root_mae": None}
    (output_dir / "final_val_eval.json").write_text(
        json.dumps(final_val, indent=2),
        encoding="utf-8",
    )

    compiled_path = output_dir / "compiled_module.json"
    try:
        compiled_program.save(str(compiled_path))
    except Exception:
        compiled_path.write_text(
            json.dumps({"note": "compiled module could not be serialized"}),
            encoding="utf-8",
        )

    metrics: dict[str, float] = {
        "n_train": float(len(trainset)),
        "n_val": float(len(valset)),
        "max_bootstrapped_demos": float(max_bootstrapped),
        "baseline_val_mae": float(baseline_val_mae),
        "zero_shot_val_root_mae": (
            float(baseline_val["root_mae"]) if baseline_val.get("root_mae") is not None else float("nan")
        ),
    }
    if final_val.get("root_mae") is not None:
        final_root_mae = float(final_val["root_mae"])
        metrics["val_root_mae"] = final_root_mae
        metrics["root_mae"] = final_root_mae
        if final_val.get("reward_mean") is not None:
            metrics["val_reward_mean"] = float(final_val["reward_mean"])
        if baseline_val_mae > 0.0:
            gf = _gain_frac(model_mae=final_root_mae, baseline_mae=baseline_val_mae)
            metrics["val_mae_gain_frac"] = float(gf)
            metrics["val_mae_pass"] = 1.0 if gf >= DEFAULT_PRIMARY_GAIN_THRESHOLD else 0.0

    return FitResult(
        backend="dspy_rile_tree",
        summary={
            "backend": "dspy_rile_tree",
            "api_base": api_base,
            "model_name": cfg.model_name,
            "n_train": len(trainset),
            "n_val": len(valset),
            "baseline_val": baseline_val,
            "final_val": final_val,
            "compiled_path": str(compiled_path),
            "local_law_weight": float(local_law_weight),
            "c1_relative_weight": float(c1_relative_weight),
            "c2_relative_weight": float(c2_relative_weight),
            "c3_relative_weight": float(c3_relative_weight),
        },
        status="completed",
        metrics=metrics,
        artifacts={
            "compiled_path": str(compiled_path),
            "baseline_val_eval": str(output_dir / "baseline_val_eval.json"),
            "final_val_eval": str(output_dir / "final_val_eval.json"),
        },
    )


register_trainer("dspy_rile_tree", dspy_rile_tree_trainer)
