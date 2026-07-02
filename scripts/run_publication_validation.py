#!/usr/bin/env python3
"""Run publication-style validation for package examples.

This runner is intentionally small and source-tree local. It exercises the
current public examples with larger synthetic workloads, confirms the learned
``f``/``g`` structure is present in neural-operator artifacts, and writes a
human-readable report plus machine-readable manifest under ``outputs/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    output_dir: Path
    gpu: str | None = None
    kind: str = "method"
    timeout_seconds: int | None = None


@dataclass
class JobResult:
    name: str
    kind: str
    gpu: str | None
    returncode: int
    seconds: float
    output_dir: str
    log_path: str
    command: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else _default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "configs").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)

    if bool(args.detach) and not bool(args.summarize_only):
        return _launch_detached(args, output_root)

    if bool(args.summarize_only):
        report = _build_report(output_root, _load_existing_job_results(output_root))
        _write_json(output_root / "manifest.json", report)
        _write_markdown(output_root / "publication_validation.md", report)
        return 0 if all(item["ok"] for item in report["jobs"]) and report["checks"]["ok"] else 1

    _write_generated_configs(output_root)
    jobs_by_lane = _jobs(output_root, quick=bool(args.quick))
    manifest: dict[str, Any] = {
        "started_at": _now(),
        "output_root": str(output_root),
        "quick": bool(args.quick),
        "lanes": {lane: [job.name for job in jobs] for lane, jobs in jobs_by_lane.items()},
    }
    _write_json(output_root / "manifest.started.json", manifest)

    all_results: list[JobResult] = []
    lock = threading.Lock()

    def run_lane(lane: str, lane_jobs: Sequence[Job]) -> None:
        for job in lane_jobs:
            result = _run_job(job, output_root)
            with lock:
                all_results.append(result)
                _write_json(output_root / "manifest.live.json", _manifest_payload(output_root, all_results))
            if result.returncode != 0 and bool(args.stop_on_failure):
                break

    threads = [
        threading.Thread(target=run_lane, args=(lane, lane_jobs), name=f"lane-{lane}")
        for lane, lane_jobs in jobs_by_lane.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    report = _build_report(output_root, all_results)
    _write_json(output_root / "manifest.json", report)
    _write_markdown(output_root / "publication_validation.md", report)
    return 0 if all(item["ok"] for item in report["jobs"]) and report["checks"]["ok"] else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", default=False)
    parser.add_argument("--stop-on-failure", action="store_true", default=False)
    parser.add_argument("--summarize-only", action="store_true", default=False)
    parser.add_argument("--detach", action="store_true", default=False)
    return parser.parse_args(list(argv) if argv is not None else None)


def _default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / f"publication_validation_{stamp}"


def _write_generated_configs(output_root: Path) -> None:
    configs = output_root / "configs"
    _write_text(
        configs / "lda_vector_1k_12topics.toml",
        """
operator_kinds = ["fno", "tfno", "uno", "conv1d"]
leaf_token_counts = [32, 64, 128, 256]
n_train = 1024
n_eval = 256
n_topics = 12
doc_tokens = 512
vocabulary_size = 512
doc_topic_concentration = 0.7
topic_word_concentration = 0.05
target_topic = 0
topic_seed = 0
seed = 101
max_iterations = 5
embedding_dim = 32
hidden_channels = 32
n_modes = 8
n_layers = 2
conv_kernel_size = 3
head_hidden_dim = 64
epochs_per_iteration = 5
batch_size = 64
learning_rate = 0.003
device = "cuda"
sklearn_max_iter = 75
""".strip() + "\n",
    )
    _write_text(
        configs / "lda_vector_2k_16topics.toml",
        """
operator_kinds = ["fno", "tfno", "uno", "conv1d"]
leaf_token_counts = [64, 128, 256]
n_train = 2048
n_eval = 512
n_topics = 16
doc_tokens = 768
vocabulary_size = 1024
doc_topic_concentration = 0.7
topic_word_concentration = 0.05
target_topic = 0
topic_seed = 7
seed = 202
max_iterations = 5
embedding_dim = 32
hidden_channels = 48
n_modes = 8
n_layers = 2
conv_kernel_size = 3
head_hidden_dim = 96
epochs_per_iteration = 5
batch_size = 64
learning_rate = 0.003
device = "cuda"
sklearn_max_iter = 75
""".strip() + "\n",
    )
    _write_text(
        configs / "markov_1k_8states_leaf32.toml",
        """
operator_kinds = ["fno", "tfno", "uno", "conv1d"]
n_train = 1024
n_eval = 256
n_states = 8
doc_tokens = 512
leaf_token_count = 32
transition_prob = 0.10
vocabulary_size = 1024
seed = 303
max_iterations = 5
embedding_dim = 32
hidden_channels = 32
n_modes = 8
n_layers = 2
conv_kernel_size = 3
head_hidden_dim = 64
epochs_per_iteration = 5
batch_size = 64
learning_rate = 0.003
device = "cuda"
""".strip() + "\n",
    )
    _write_text(
        configs / "markov_2k_8states_leaf64.toml",
        """
operator_kinds = ["fno", "tfno", "uno", "conv1d"]
n_train = 2048
n_eval = 512
n_states = 8
doc_tokens = 768
leaf_token_count = 64
transition_prob = 0.18
vocabulary_size = 1024
seed = 404
max_iterations = 5
embedding_dim = 32
hidden_channels = 48
n_modes = 8
n_layers = 2
conv_kernel_size = 3
head_hidden_dim = 96
epochs_per_iteration = 5
batch_size = 64
learning_rate = 0.003
device = "cuda"
""".strip() + "\n",
    )
    _write_text(
        configs / "markov_leaf_grid_1k_8states.toml",
        """
operator_kinds = ["fno", "tfno", "uno", "conv1d"]
leaf_token_counts = [32, 64, 128, 256]
n_train = 1024
n_eval = 256
n_states = 8
doc_tokens = 512
transition_prob = 0.12
vocabulary_size = 1024
seed = 505
max_iterations = 5
embedding_dim = 32
hidden_channels = 32
n_modes = 8
n_layers = 2
conv_kernel_size = 3
head_hidden_dim = 64
epochs_per_iteration = 5
batch_size = 64
learning_rate = 0.003
device = "cuda"
normalize_targets = true
numeric_transition_state_weight = 0.05
""".strip() + "\n",
    )
    _write_text(
        configs / "markov_leaf_grid_2k_8states.toml",
        """
operator_kinds = ["fno", "tfno", "uno", "conv1d"]
leaf_token_counts = [64, 128, 256]
n_train = 2048
n_eval = 512
n_states = 8
doc_tokens = 768
transition_prob = 0.18
vocabulary_size = 1024
seed = 606
max_iterations = 5
embedding_dim = 32
hidden_channels = 48
n_modes = 8
n_layers = 2
conv_kernel_size = 3
head_hidden_dim = 96
epochs_per_iteration = 5
batch_size = 64
learning_rate = 0.003
device = "cuda"
normalize_targets = true
numeric_transition_state_weight = 0.05
""".strip() + "\n",
    )
    _write_text(
        configs / "lda_quick.toml",
        """
operator_kinds = ["fno", "conv1d"]
leaf_token_counts = [32, 64]
n_train = 64
n_eval = 16
n_topics = 6
doc_tokens = 128
vocabulary_size = 128
doc_topic_concentration = 0.7
topic_word_concentration = 0.05
target_topic = 0
topic_seed = 0
seed = 11
max_iterations = 3
embedding_dim = 16
hidden_channels = 8
n_modes = 4
n_layers = 1
conv_kernel_size = 3
head_hidden_dim = 16
epochs_per_iteration = 1
batch_size = 16
learning_rate = 0.01
device = "cuda"
sklearn_max_iter = 20
""".strip() + "\n",
    )
    _write_text(
        configs / "markov_quick.toml",
        """
operator_kinds = ["fno", "conv1d"]
n_train = 64
n_eval = 16
n_states = 5
doc_tokens = 128
leaf_token_count = 32
transition_prob = 0.12
vocabulary_size = 256
seed = 12
max_iterations = 3
embedding_dim = 16
hidden_channels = 8
n_modes = 4
n_layers = 1
conv_kernel_size = 3
head_hidden_dim = 16
epochs_per_iteration = 1
batch_size = 16
learning_rate = 0.01
device = "cuda"
""".strip() + "\n",
    )
    _write_text(
        configs / "markov_leaf_grid_quick.toml",
        """
operator_kinds = ["fno", "conv1d"]
leaf_token_counts = [32, 64]
n_train = 64
n_eval = 16
n_states = 5
doc_tokens = 128
transition_prob = 0.12
vocabulary_size = 256
seed = 13
max_iterations = 3
embedding_dim = 16
hidden_channels = 8
n_modes = 4
n_layers = 1
conv_kernel_size = 3
head_hidden_dim = 16
epochs_per_iteration = 1
batch_size = 16
learning_rate = 0.01
device = "cuda"
normalize_targets = true
numeric_transition_state_weight = 0.05
""".strip() + "\n",
    )


def _jobs(output_root: Path, *, quick: bool) -> dict[str, list[Job]]:
    configs = output_root / "configs"
    lanes: dict[str, list[Job]] = {
        "cpu": [
            *_method_smoke_jobs(output_root),
            Job("pytest_full", (PYTHON, "-m", "pytest", "tests", "-q"), output_root / "pytest", kind="check"),
            Job("release_check", (PYTHON, "-m", "treepo.release"), output_root / "release", kind="check"),
            *_benchmark_jobs(output_root),
        ],
        "gpu1": [],
        "gpu2": [],
        "gpu3": [],
    }
    if quick:
        lanes["gpu1"].append(_lda_job("lda_quick", configs / "lda_quick.toml", output_root / "methods" / "lda_quick", gpu="1"))
        lanes["gpu2"].append(_markov_job("markov_quick", configs / "markov_quick.toml", output_root / "methods" / "markov_quick", gpu="2"))
        lanes["gpu3"].append(_markov_leaf_grid_job("markov_leaf_grid_quick", configs / "markov_leaf_grid_quick.toml", output_root / "methods" / "markov_leaf_grid_quick", gpu="3"))
        return lanes
    lanes["gpu1"].extend([
        _lda_job("lda_vector_1k_12topics", configs / "lda_vector_1k_12topics.toml", output_root / "methods" / "lda_vector_1k_12topics", gpu="1"),
        _markov_job("markov_2k_8states_leaf64", configs / "markov_2k_8states_leaf64.toml", output_root / "methods" / "markov_2k_8states_leaf64", gpu="1"),
    ])
    lanes["gpu2"].append(
        _lda_job("lda_vector_2k_16topics", configs / "lda_vector_2k_16topics.toml", output_root / "methods" / "lda_vector_2k_16topics", gpu="2")
    )
    lanes["gpu2"].append(
        _markov_leaf_grid_job("markov_leaf_grid_1k_8states", configs / "markov_leaf_grid_1k_8states.toml", output_root / "methods" / "markov_leaf_grid_1k_8states", gpu="2")
    )
    lanes["gpu3"].extend([
        _markov_job("markov_1k_8states_leaf32", configs / "markov_1k_8states_leaf32.toml", output_root / "methods" / "markov_1k_8states_leaf32", gpu="3"),
        _markov_leaf_grid_job("markov_leaf_grid_2k_8states", configs / "markov_leaf_grid_2k_8states.toml", output_root / "methods" / "markov_leaf_grid_2k_8states", gpu="3"),
    ])
    return lanes


def _method_smoke_jobs(output_root: Path) -> list[Job]:
    root = output_root / "method_smoke"
    return [
        _method_example_job("method_hll_sketch", "run_hll_sketch.py", root / "hll_sketch"),
        _method_example_job("method_fno_markov", "run_fno_markov.py", root / "fno_markov"),
        _method_example_job("method_manifesto_dspy", "run_manifesto_replications.py", root / "manifesto_dspy"),
        _method_example_job(
            "method_manifesto_prompted_llm",
            "run_manifesto_replications.py",
            root / "manifesto_prompted_llm",
            extra=("--estimator", "prompted_llm"),
        ),
        _method_example_job("method_neural_operator_lda", "run_neural_operator_lda.py", root / "neural_operator_lda"),
        _method_example_job("method_neural_operator_markov_compare", "run_neural_operator_markov_compare.py", root / "neural_operator_markov_compare"),
        _method_example_job("method_neural_operator_lda_leaf_grid", "run_neural_operator_lda_leaf_grid.py", root / "neural_operator_lda_leaf_grid"),
        _method_example_job("method_neural_operator_markov_leaf_grid", "run_neural_operator_markov_leaf_grid.py", root / "neural_operator_markov_leaf_grid"),
    ]


def _method_example_job(name: str, script: str, out: Path, *, extra: Sequence[str] = ()) -> Job:
    return Job(
        name=name,
        command=(PYTHON, f"examples/methods/{script}", "--output-dir", str(out), *tuple(extra)),
        output_dir=out,
        kind="method_smoke",
    )


def _lda_job(name: str, config: Path, out: Path, *, gpu: str) -> Job:
    return Job(
        name=name,
        command=(PYTHON, "examples/methods/run_neural_operator_lda_leaf_grid.py", "--config", str(config), "--output-dir", str(out)),
        output_dir=out,
        gpu=gpu,
        kind="lda",
    )


def _markov_job(name: str, config: Path, out: Path, *, gpu: str) -> Job:
    return Job(
        name=name,
        command=(PYTHON, "examples/methods/run_neural_operator_markov_compare.py", "--config", str(config), "--output-dir", str(out)),
        output_dir=out,
        gpu=gpu,
        kind="markov",
    )


def _markov_leaf_grid_job(name: str, config: Path, out: Path, *, gpu: str) -> Job:
    return Job(
        name=name,
        command=(PYTHON, "examples/methods/run_neural_operator_markov_leaf_grid.py", "--config", str(config), "--output-dir", str(out)),
        output_dir=out,
        gpu=gpu,
        kind="markov_leaf_grid",
    )


def _benchmark_jobs(output_root: Path) -> list[Job]:
    specs = [
        ("bench_classical_sketches", "classical-sketches", "examples/bench/classical_sketches.yaml"),
        ("bench_markov", "markov", "examples/bench/markov.yaml"),
    ]
    jobs = []
    for name, experiment, config in specs:
        out = output_root / "bench" / name
        jobs.append(
            Job(
                name=name,
                command=(PYTHON, "-m", "treepo.bench.cli", "run", experiment, "--config", config, "--json-out", str(out / "summary.json"), "--csv-out", str(out / "summary.csv")),
                output_dir=out,
                kind="bench",
            )
        )
    return jobs



def _launch_detached(args: argparse.Namespace, output_root: Path) -> int:
    command = [PYTHON, str(Path(__file__).resolve()), "--output-dir", str(output_root)]
    if bool(args.quick):
        command.append("--quick")
    if bool(args.stop_on_failure):
        command.append("--stop-on-failure")
    log_path = output_root / "logs" / "launcher.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    _write_json(
        output_root / "launcher.json",
        {
            "pid": int(proc.pid),
            "started_at": _now(),
            "output_root": str(output_root),
            "log_path": str(log_path),
            "command": command,
        },
    )
    print(f"launched pid={proc.pid} output_root={output_root} log={log_path}")
    return 0


def _run_job(job: Job, output_root: Path) -> JobResult:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / f"{job.name}.log"
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    if job.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at={_now()}\n")
        log.write(f"cwd={ROOT}\n")
        log.write(f"gpu={job.gpu or ''}\n")
        log.write("command=" + " ".join(job.command) + "\n\n")
        log.flush()
        proc = subprocess.run(
            list(job.command),
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=job.timeout_seconds,
        )
        log.write(f"\nfinished_at={_now()}\nreturncode={proc.returncode}\n")
    return JobResult(
        name=job.name,
        kind=job.kind,
        gpu=job.gpu,
        returncode=int(proc.returncode),
        seconds=float(time.time() - started),
        output_dir=str(job.output_dir),
        log_path=str(log_path),
        command=list(job.command),
    )


def _load_existing_job_results(output_root: Path) -> list[JobResult]:
    for filename in ("manifest.live.json", "manifest.json"):
        path = output_root / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        loaded = []
        for item in list(payload.get("jobs") or []):
            try:
                loaded.append(
                    JobResult(
                        name=str(item.get("name") or ""),
                        kind=str(item.get("kind") or ""),
                        gpu=(None if item.get("gpu") is None else str(item.get("gpu"))),
                        returncode=int(item.get("returncode") or 0),
                        seconds=float(item.get("seconds") or 0.0),
                        output_dir=str(item.get("output_dir") or ""),
                        log_path=str(item.get("log_path") or ""),
                        command=[str(x) for x in list(item.get("command") or [])],
                    )
                )
            except Exception:
                continue
        if loaded:
            return loaded
    return []


def _build_report(output_root: Path, results: Sequence[JobResult]) -> dict[str, Any]:
    jobs = [
        {
            "name": item.name,
            "kind": item.kind,
            "gpu": item.gpu,
            "ok": item.ok,
            "returncode": item.returncode,
            "seconds": item.seconds,
            "output_dir": item.output_dir,
            "log_path": item.log_path,
            "command": item.command,
        }
        for item in sorted(results, key=lambda r: r.name)
    ]
    methods = _collect_method_results(output_root)
    checks = _publication_checks(jobs, methods)
    return {
        "started_at": _read_started_at(output_root),
        "finished_at": _now(),
        "output_root": str(output_root),
        "jobs": jobs,
        "methods": methods,
        "checks": checks,
    }


def _manifest_payload(output_root: Path, results: Sequence[JobResult]) -> dict[str, Any]:
    return {
        "updated_at": _now(),
        "output_root": str(output_root),
        "jobs": [item.__dict__ | {"ok": item.ok} for item in results],
    }


def _collect_method_results(output_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    methods_root = output_root / "methods"
    if not methods_root.exists():
        return out
    for result_path in sorted(methods_root.glob("*/neural_operator_lda_leaf_grid.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in list(payload.get("rows") or [])]
        best_scalar = _min_row(rows, "internal_f_mae")
        best_vector = _min_row(rows, "mean_topic_vector_mae")
        out.append({
            "name": result_path.parent.name,
            "kind": "lda",
            "path": str(result_path),
            "n_rows": len(rows),
            "config": payload.get("config"),
            "sklearn_target_mae": _float(payload.get("sklearn_baseline", {}).get("target_mae")),
            "sklearn_mean_mae": _float(payload.get("sklearn_baseline", {}).get("mean_mae")),
            "average_guess_target_mae": _float(payload.get("average_guess_baseline", {}).get("target_mae")),
            "average_guess_mean_mae": _float(payload.get("average_guess_baseline", {}).get("mean_mae")),
            "best_scalar": best_scalar,
            "best_vector": best_vector,
            "g_checks": _g_checks_for_rows(rows),
            "vector_metric_checks": _vector_metric_checks(rows),
        })
    for result_path in sorted(methods_root.glob("*/neural_operator_markov_compare.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows = []
        for kind, result in dict(payload.get("results") or {}).items():
            metrics = dict(result.get("metrics") or {})
            rows.append({
                "operator_kind": str(kind),
                "internal_f_mae": _float(metrics.get("internal_f_mae")),
                "internal_f_pearson": _float(metrics.get("internal_f_pearson")),
                "n": _float(metrics.get("n")),
                "f_kind": (((result.get("artifacts") or {}).get("f") or {}).get("kind")),
                "g_kind": (((result.get("artifacts") or {}).get("g") or {}).get("kind")),
                "g_trained": (((result.get("artifacts") or {}).get("g") or {}).get("trained")),
            })
        out.append({
            "name": result_path.parent.name,
            "kind": "markov",
            "path": str(result_path),
            "n_rows": len(rows),
            "config": payload.get("config"),
            "best_scalar": _min_row(rows, "internal_f_mae"),
            "rows": rows,
            "g_checks": {"all_trained_g": all(row.get("g_trained") == "g" for row in rows)},
        })
    for result_path in sorted(methods_root.glob("*/neural_operator_markov_leaf_grid.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in list(payload.get("rows") or [])]
        out.append({
            "name": result_path.parent.name,
            "kind": "markov_leaf_grid",
            "path": str(result_path),
            "n_rows": len(rows),
            "config": payload.get("config"),
            "average_guess_mae": _float(payload.get("average_guess_baseline", {}).get("mae")),
            "best_scalar": _min_row(rows, "internal_f_mae"),
            "rows": rows,
            "g_checks": {"all_trained_g": all(row.get("g_trained") == "g" for row in rows)},
        })
    return out


def _publication_checks(jobs: Sequence[Mapping[str, Any]], methods: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    if any(not bool(job.get("ok")) for job in jobs):
        failures.append("one_or_more_jobs_failed")
    if not any(job.get("kind") == "method_smoke" for job in jobs):
        failures.append("missing_method_smoke_jobs")
    if not any(item.get("kind") == "lda" for item in methods):
        failures.append("missing_lda_method_results")
    if not any(item.get("kind") == "markov" for item in methods):
        failures.append("missing_markov_method_results")
    if not any(item.get("kind") == "markov_leaf_grid" for item in methods):
        failures.append("missing_markov_leaf_grid_method_results")
    for item in methods:
        checks = dict(item.get("g_checks") or {})
        if checks.get("all_trained_g") is False:
            failures.append(f"{item.get('name')}:missing_trained_g")
        if item.get("kind") == "lda":
            vector_checks = dict(item.get("vector_metric_checks") or {})
            if not bool(vector_checks.get("all_rows_have_mean_topic_vector_mae")):
                failures.append(f"{item.get('name')}:missing_vector_mae")
            best = dict(item.get("best_vector") or {})
            avg = _float(item.get("average_guess_mean_mae"))
            val = _float(best.get("mean_topic_vector_mae"))
            if avg is not None and val is not None and val > avg:
                failures.append(f"{item.get('name')}:best_vector_worse_than_average_guess")
    return {"ok": not failures, "failures": failures}


def _g_checks_for_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trained = []
    for row in rows:
        manifest_path = row.get("manifest_path")
        if not manifest_path:
            trained.append(False)
            continue
        try:
            manifest = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
        except Exception:
            trained.append(False)
            continue
        # The manifest records final metrics. The result JSON carries artifacts;
        # prediction records for iter_02 prove train_g ran in the canonical ladder.
        pred_dir = Path(str(manifest_path)).parent / "prediction_records"
        trained.append((pred_dir / "iter_02_post_eval.jsonl").exists())
    return {"all_trained_g": all(trained), "n_checked": len(trained)}


def _vector_metric_checks(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [_float(row.get("mean_topic_vector_mae")) for row in rows]
    return {
        "all_rows_have_mean_topic_vector_mae": bool(values) and all(value is not None for value in values),
        "n_checked": len(values),
    }


def _min_row(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [dict(row) for row in rows if _float(row.get(key)) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key]))


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Treepo Publication Validation",
        "",
        f"Started: `{report.get('started_at')}`",
        f"Finished: `{report.get('finished_at')}`",
        f"Output root: `{report.get('output_root')}`",
        "",
        "## Job Status",
        "",
        "| Job | Kind | GPU | Status | Seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for job in report.get("jobs", []):
        lines.append(
            f"| {job['name']} | {job['kind']} | {job.get('gpu') or ''} | "
            f"{'ok' if job.get('ok') else 'FAILED'} | {float(job.get('seconds') or 0.0):.1f} |"
        )
    lines.extend(["", "## Method Results", ""])
    for item in report.get("methods", []):
        lines.append(f"### {item.get('name')}")
        lines.append("")
        if item.get("kind") == "lda":
            config = dict(item.get("config") or {})
            best_v = dict(item.get("best_vector") or {})
            best_s = dict(item.get("best_scalar") or {})
            lines.extend([
                f"- Task: LDA topic proportions with `{config.get('n_train')}` train docs, `{config.get('n_eval')}` eval docs, `{config.get('n_topics')}` topics.",
                f"- Best target-topic MAE: `{best_s.get('internal_f_mae')}` at leaf `{best_s.get('leaf_token_count')}` / operator `{best_s.get('operator_kind')}`.",
                f"- Best mean topic-vector MAE: `{best_v.get('mean_topic_vector_mae')}` at leaf `{best_v.get('leaf_token_count')}` / operator `{best_v.get('operator_kind')}`.",
                f"- Average-guess mean MAE: `{item.get('average_guess_mean_mae')}`; sklearn mean MAE: `{item.get('sklearn_mean_mae')}`.",
                f"- Trained g check: `{item.get('g_checks')}`; vector metric check: `{item.get('vector_metric_checks')}`.",
                "",
            ])
        elif item.get("kind") == "markov":
            config = dict(item.get("config") or {})
            best = dict(item.get("best_scalar") or {})
            lines.extend([
                f"- Task: Markov changepoint count with `{config.get('n_train')}` train docs, `{config.get('n_eval')}` eval docs, `{config.get('n_states')}` states.",
                f"- Best MAE: `{best.get('internal_f_mae')}` / Pearson `{best.get('internal_f_pearson')}` with operator `{best.get('operator_kind')}`.",
                f"- Trained g check: `{item.get('g_checks')}`.",
                "",
            ])
        elif item.get("kind") == "markov_leaf_grid":
            config = dict(item.get("config") or {})
            best = dict(item.get("best_scalar") or {})
            lines.extend([
                f"- Task: Markov leaf-size grid with `{config.get('n_train')}` train docs, `{config.get('n_eval')}` eval docs, `{config.get('n_states')}` states.",
                f"- Best MAE: `{best.get('internal_f_mae')}` / Pearson `{best.get('internal_f_pearson')}` at leaf `{best.get('leaf_token_count')}` / operator `{best.get('operator_kind')}`.",
                f"- Average-guess MAE: `{item.get('average_guess_mae')}`.",
                f"- Trained g check: `{item.get('g_checks')}`.",
                "",
            ])
    checks = dict(report.get("checks") or {})
    lines.extend(["## Checks", "", f"Overall: `{'ok' if checks.get('ok') else 'FAILED'}`", ""])
    for failure in checks.get("failures") or []:
        lines.append(f"- {failure}")
    _write_text(path, "\n".join(lines) + "\n")


def _read_started_at(output_root: Path) -> str | None:
    path = output_root / "manifest.started.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("started_at")
    except Exception:
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
