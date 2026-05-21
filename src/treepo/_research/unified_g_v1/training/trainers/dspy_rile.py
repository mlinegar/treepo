"""Manifesto RILE DSPy bootstrap trainer.

Given a `text_pairs_v1` oracle and a vLLM OpenAI-compatible endpoint, this
trainer bootstraps a DSPy predictor that reads a manifesto excerpt and
returns a RILE score. The metric is mean-absolute-error against the
doc-level RILE target. Queries go to the running vLLM — no local GPU
needed for training the LLM side; training is purely prompt optimization.

Expected TrainerConfig fields:
  * `oracle`                  — produces TreeExample with leaves[0]=prompt, extra["target_raw"]=float.
  * `model_name`              — the served-model-name the vLLM advertises.
  * `extra["api_base"]`       — e.g. "http://localhost:8000/v1"
  * `extra["api_key"]`        — defaults to "EMPTY"
  * `extra["temperature"]`    — defaults to 0.0
  * `extra["max_tokens"]`     — defaults to 1024 (matched to leaf_tokens input;
                                 must leave headroom for reasoning + score)
  * `extra["max_bootstrapped_demos"]` — defaults to 4
  * `extra["max_train_examples"]`     — defaults to 0 (no cap)
  * `extra["n_val"]`          — defaults to 0 (evaluate on all val)
  * `extra["optimizer"]`      — "gepa" (default) or "bootstrap"
  * `extra["gepa_auto"]`      — "light" (default) | "medium" | "heavy"
  * `extra["gepa_num_threads"]` — default 8
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from treepo._research.unified_g_v1.training.trainers import register_trainer


_RILE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_rile(text: str) -> float | None:
    m = _RILE_RE.search(str(text or ""))
    if not m:
        return None
    try:
        value = float(m.group(0))
    except ValueError:
        return None
    return max(-100.0, min(100.0, value))


def dspy_rile_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    import dspy  # lazy; only needed here

    from treepo._research.unified_g_v1.training.fit import FitResult

    if cfg.oracle is None:
        raise ValueError("dspy_rile_trainer requires cfg.oracle")
    if not cfg.model_name:
        raise ValueError("dspy_rile_trainer requires cfg.model_name (served-model-name on the vLLM)")

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
    # GEPA reflection prompts stack N traces + feedback. Keep them compact:
    # minibatch=3 and a modest reflection_max_tokens fit comfortably in an
    # 8k-context student (gemma-4-E2B-it via vLLM --max-model-len 8192).
    gepa_minibatch_size = int(extra.get("gepa_reflection_minibatch_size", 3))
    gepa_valset_cap = int(extra.get("gepa_valset_cap", 64))
    # Resume: by default GEPA auto-loads `gepa_state.bin` if one exists in
    # `output_dir/gepa_logs`. Pass `gepa_resume_log_dir` to load state from a
    # different prior run's log directory instead.
    gepa_resume_log_dir = extra.get("gepa_resume_log_dir") or None
    # Optional separate reflection LM; defaults to the student.
    reflection_api_base = str(extra.get("reflection_api_base", "") or api_base).rstrip("/")
    reflection_model_name = str(extra.get("reflection_model_name", "") or cfg.model_name)
    # Reflection output can be verbose — the reflection LM writes a revised
    # instruction/prompt for the student. Default high; what we pay for is
    # keeping the INPUT we send (minibatch traces) compact.
    reflection_max_tokens = int(extra.get("reflection_max_tokens", 16384))

    # Configure DSPy to call the vLLM OpenAI-compatible endpoint.
    lm = dspy.LM(
        model=f"openai/{cfg.model_name}",
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)

    # Reflection LM (may be a separate, larger-context model).
    reflection_lm = dspy.LM(
        model=f"openai/{reflection_model_name}",
        api_base=reflection_api_base,
        api_key=api_key,
        temperature=0.7,  # GEPA-recommended: higher temp for diverse proposals
        max_tokens=reflection_max_tokens,
    )

    class RilePredictor(dspy.Signature):
        """Predict the manifesto RILE (Right-Left) score.

        RILE is the sum of right-coded quasi-sentence shares minus left-coded
        shares. Right-coded themes: free market, military, traditional morality,
        law & order, nationalism. Left-coded: welfare state, anti-imperialism,
        labor rights, peace, environmental protection.

        Keep the reasoning under 150 words so the answer field is always emitted.
        """

        manifesto_excerpt: str = dspy.InputField(desc="A ~1024-token excerpt of a party manifesto.")
        rile_score: str = dspy.OutputField(
            desc="A single numeric RILE score in [-100, +100]. Just the number, e.g. '-12.5' or '7.0'."
        )

    # ChainOfThought so GEPA can reflect on the model's reasoning trace — that
    # reasoning is what the reflection LM edits to improve the program.
    program = dspy.ChainOfThought(RilePredictor)

    # Build DSPy Examples from the oracle. target_raw holds the numeric RILE.
    def _to_examples(raw_items):
        built = []
        for ex in raw_items:
            prompt = ex.leaves[0] if ex.leaves else ""
            target = ex.extra.get("target_raw")
            if target is None:
                continue
            built.append(
                dspy.Example(
                    manifesto_excerpt=str(prompt),
                    rile_score=float(target),
                ).with_inputs("manifesto_excerpt")
            )
        return built

    trainset = _to_examples(cfg.oracle.train_examples())
    valset = _to_examples(cfg.oracle.val_examples())
    if max_train_examples > 0:
        trainset = trainset[:max_train_examples]
    if n_val_cap > 0:
        valset = valset[:n_val_cap]

    # Predict-train-mean baseline MAE on val — law-stress-style denominator.
    from treepo._research.unified_g_v1.eval.law_stress import (
        DEFAULT_PRIMARY_GAIN_THRESHOLD,
        gain_frac as _gain_frac,
    )

    def _extract_rile(ex) -> float:
        return float(ex.rile_score)

    if trainset and valset:
        _train_mean = sum(_extract_rile(ex) for ex in trainset) / float(len(trainset))
        baseline_val_mae = sum(
            abs(_train_mean - _extract_rile(ex)) for ex in valset
        ) / float(len(valset))
    else:
        baseline_val_mae = 0.0

    # Metric: -MAE so higher-is-better for DSPy's optimizer; report raw MAE too.
    def _mae_metric(example, prediction, trace=None) -> float:
        del trace
        predicted = _parse_rile(getattr(prediction, "rile_score", prediction))
        if predicted is None:
            return 0.0  # unparsable → worst score in [0, 1]
        err = abs(float(predicted) - float(example.rile_score))
        # Map error -> reward in [0, 1]. An error of 0 is 1.0; >=200 is 0.
        return max(0.0, 1.0 - err / 200.0)

    def _evaluate(examples) -> dict[str, Any]:
        abs_errors: list[float] = []
        unparsable = 0
        predictions: list[dict[str, Any]] = []
        for ex in examples:
            pred = program(manifesto_excerpt=ex.manifesto_excerpt)
            predicted = _parse_rile(getattr(pred, "rile_score", pred))
            if predicted is None:
                unparsable += 1
                continue
            err = abs(float(predicted) - float(ex.rile_score))
            abs_errors.append(err)
            predictions.append(
                {
                    "predicted_rile": float(predicted),
                    "target_rile": float(ex.rile_score),
                    "abs_error": err,
                }
            )
        n = max(1, len(abs_errors))
        mae_raw = float(sum(abs_errors) / n) if abs_errors else float("nan")
        out = {
            "count": int(len(examples)),
            "parsed": int(len(abs_errors)),
            "unparsable": int(unparsable),
            "mae_raw": mae_raw,
            "predictions": predictions,
        }
        # Law-stress gain_frac on the same (-∞, 1.0] scale as C1/C2/C3.
        if baseline_val_mae > 0.0 and abs_errors:
            gf = _gain_frac(model_mae=mae_raw, baseline_mae=baseline_val_mae)
            out["baseline_val_mae"] = float(baseline_val_mae)
            out["val_mae_gain_frac"] = float(gf)
            out["val_mae_pass"] = 1.0 if gf >= DEFAULT_PRIMARY_GAIN_THRESHOLD else 0.0
        return out

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Baseline: zero-shot evaluation before any bootstrapping -----------
    baseline_val = _evaluate(valset) if valset else {"count": 0, "mae_raw": None}
    (output_dir / "baseline_val_eval.json").write_text(
        json.dumps(baseline_val, indent=2), encoding="utf-8"
    )

    # ---- Optimize ---------------------------------------------------------
    if optimizer_name == "gepa":
        # GEPA's metric signature: (gold, pred, trace, pred_name, pred_trace).
        # Must return a ScoreWithFeedback (score + textual feedback the
        # reflection LM reads) — plain dicts confuse dspy's parallelizer.
        from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

        def _truncate(text: str, n: int) -> str:
            text = str(text or "")
            return text if len(text) <= n else text[: n - 3] + "..."

        def _gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            del trace, pred_name, pred_trace
            predicted_raw = getattr(pred, "rile_score", pred)
            predicted = _parse_rile(predicted_raw)
            target = float(gold.rile_score)
            reasoning = _truncate(getattr(pred, "reasoning", "") or "", 800)
            if predicted is None:
                feedback = (
                    "Output failed to parse as a RILE score. Target was "
                    f"{target:.1f}. Return a single numeric value in [-100, +100] "
                    "in the `rile_score` field. Your reasoning was:\n"
                    f"{reasoning or '(empty)'}"
                )
                return ScoreWithFeedback(score=0.0, feedback=feedback)
            err = abs(float(predicted) - target)
            score = max(0.0, 1.0 - err / 200.0)
            direction = "more positive (rightward)" if predicted < target else "more negative (leftward)"
            magnitude_hint = (
                "The error is severe (>40 RILE units); re-examine signals of "
                "economic policy, social programs, law and order, nationalism, "
                "and environmentalism."
                if err > 40.0
                else "The error is moderate; refine the weighting of conflicting cues."
                if err > 15.0
                else "The error is small; the direction may be approximately right."
            )
            feedback = (
                f"Predicted RILE {predicted:.1f} vs target {target:.1f} (abs error {err:.1f}).\n"
                f"The true RILE is {direction}. {magnitude_hint}\n"
                f"Recall: RILE = (sum of right-coded quasi-sentence shares) - (sum of left-coded).\n"
                f"Right-coded themes: free market, military, traditional morality, law & order, national way of life.\n"
                f"Left-coded themes: welfare state, anti-imperialism, labor, peace, environmental protection.\n"
                f"Your reasoning was:\n{reasoning or '(empty)'}"
            )
            return ScoreWithFeedback(score=float(score), feedback=feedback)

        # If user pointed us at a prior gepa_logs dir, copy its state into
        # our fresh log_dir so GEPA's initialize_gepa_state auto-resumes.
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
            # Cap GEPA's internal val sweep for speed; our own _evaluate() below
            # still runs on the full valset to report the final MAE.
            valset=list(valset[:gepa_valset_cap]) if gepa_valset_cap > 0 else valset,
        )
    else:
        teleprompter = dspy.teleprompt.BootstrapFewShot(
            metric=_mae_metric,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )
        compiled_program = teleprompter.compile(program, trainset=trainset)

    # Swap in the compiled program for evaluation.
    program = compiled_program

    # ---- Evaluate the bootstrapped program on val -------------------------
    final_val = _evaluate(valset) if valset else {"count": 0, "mae_raw": None}
    (output_dir / "final_val_eval.json").write_text(
        json.dumps(final_val, indent=2), encoding="utf-8"
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
        # Predict-train-mean baseline on val (law-stress denominator).
        "baseline_val_mae": float(baseline_val_mae),
        # Zero-shot model performance before optimization, for reference.
        "zero_shot_val_mae_raw": (
            float(baseline_val["mae_raw"]) if baseline_val.get("mae_raw") is not None else float("nan")
        ),
    }
    if final_val.get("mae_raw") is not None:
        final_mae = float(final_val["mae_raw"])
        metrics["val_mae"] = final_mae
        metrics["val_mae_raw"] = final_mae
        metrics["mae_raw"] = final_mae
        if baseline_val_mae > 0.0:
            gf = _gain_frac(model_mae=final_mae, baseline_mae=baseline_val_mae)
            metrics["val_mae_gain_frac"] = float(gf)
            metrics["val_mae_pass"] = 1.0 if gf >= DEFAULT_PRIMARY_GAIN_THRESHOLD else 0.0

    return FitResult(
        backend="dspy_rile",
        summary={
            "backend": "dspy_rile",
            "api_base": api_base,
            "model_name": cfg.model_name,
            "n_train": len(trainset),
            "n_val": len(valset),
            "baseline_val": baseline_val,
            "final_val": final_val,
            "compiled_path": str(compiled_path),
        },
        status="completed",
        metrics=metrics,
        artifacts={
            "compiled_path": str(compiled_path),
            "baseline_val_eval": str(output_dir / "baseline_val_eval.json"),
            "final_val_eval": str(output_dir / "final_val_eval.json"),
        },
    )


register_trainer("dspy_rile", dspy_rile_trainer)
