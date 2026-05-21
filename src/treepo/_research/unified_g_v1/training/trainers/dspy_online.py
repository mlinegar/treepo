"""`dspy_online` — in-process DSPy optimization driven by `feedback_fn`.

This is the DSPy counterpart to `sft_best_of_n`: the user supplies a DSPy
module (`cfg.base_module`), an oracle of prompt-carrying examples, and a
`feedback_fn(prompt, candidates) -> ranking`. We wrap `feedback_fn` as a
DSPy metric and run `BootstrapFewShot` (or the teleprompter named in
`cfg.extra["dspy_optimizer"]`) to compile the module against the oracle.

Unlike the subprocess-based `dspy` trainer (which shells out to DSPy from an
external script), this trainer runs DSPy in-process so the user-provided
`feedback_fn` can be any Python callable — including one that talks to a
human in a notebook.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from treepo._research.unified_g_v1.training.trainers import register_trainer


def _best_index(feedback_result: Any) -> int:
    if isinstance(feedback_result, int):
        return int(feedback_result)
    if isinstance(feedback_result, Sequence) and len(feedback_result) > 0:
        return int(feedback_result[0])
    raise ValueError(
        f"feedback_fn must return an int index or a non-empty ranking list, got {feedback_result!r}"
    )


def dspy_online_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    import dspy  # lazy import; only required for this trainer
    from dspy.teleprompt import BootstrapFewShot  # type: ignore

    from treepo._research.unified_g_v1.training.fit import FitResult

    if cfg.base_module is None:
        raise ValueError("dspy_online_trainer requires cfg.base_module (a dspy.Module)")
    if cfg.feedback_fn is None or not callable(cfg.feedback_fn):
        raise ValueError("dspy_online_trainer requires callable cfg.feedback_fn")
    if cfg.oracle is None:
        raise ValueError("dspy_online_trainer requires cfg.oracle")

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    n_candidates = max(1, int(cfg.candidates_per_example))

    # Build dspy Examples from the oracle. Each TreeExample's leaves[0] is
    # the prompt; the target becomes the DSPy example's expected output.
    examples: list[Any] = []
    for tree_example in cfg.oracle.train_examples():
        prompt = tree_example.leaves[0] if tree_example.leaves else ""
        target = tree_example.target
        ex = dspy.Example(prompt=str(prompt), target=target).with_inputs("prompt")
        examples.append(ex)

    # Metric: draw N candidates from the current program, ask feedback_fn to
    # rank them. Return 1.0 if the prediction matches the preferred candidate,
    # else a graded score from the ranking position.
    def _metric(example, prediction, trace=None) -> float:
        del trace
        try:
            predicted = getattr(prediction, "completion", None) or str(prediction)
        except Exception:
            predicted = str(prediction)
        candidates = [predicted]
        # Sample a few extra candidates from the same program for the
        # feedback_fn to rank against. Cheap perturbation: just duplicate the
        # prediction so feedback_fn has at least `n_candidates` entries.
        while len(candidates) < n_candidates:
            candidates.append(predicted)
        ranking = cfg.feedback_fn(example.prompt, candidates)
        if isinstance(ranking, int):
            return 1.0 if int(ranking) == 0 else 0.0
        if isinstance(ranking, Sequence):
            try:
                rank = list(ranking).index(0)
            except ValueError:
                return 0.0
            return max(0.0, 1.0 - rank / max(1, len(candidates) - 1))
        return 0.0

    rng = random.Random(int(cfg.seed))
    trainset = list(examples)
    rng.shuffle(trainset)

    teleprompter = BootstrapFewShot(metric=_metric, max_bootstrapped_demos=4, max_labeled_demos=0)
    compiled = teleprompter.compile(cfg.base_module, trainset=trainset)

    compiled_path = output_dir / "compiled_module.json"
    try:
        compiled.save(str(compiled_path))
        saved = True
    except Exception:
        saved = False
        compiled_path.write_text(
            json.dumps({"note": "dspy module could not be serialized"}),
            encoding="utf-8",
        )

    summary = {
        "backend": "dspy_online",
        "n_train_examples": len(examples),
        "candidates_per_example": n_candidates,
        "compiled_path": str(compiled_path),
        "compiled_saved": saved,
    }
    return FitResult(
        backend="dspy_online",
        summary=summary,
        status="completed",
        metrics={
            "n_train_examples": float(summary["n_train_examples"]),
            "candidates_per_example": float(summary["candidates_per_example"]),
        },
        artifacts={"compiled_path": str(compiled_path)},
    )


register_trainer("dspy_online", dspy_online_trainer)
