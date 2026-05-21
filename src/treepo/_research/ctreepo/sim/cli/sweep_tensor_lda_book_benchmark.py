#!/usr/bin/env python3
"""Build (and optionally execute) xargs-friendly sweeps for the Tensor-LDA "books" benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

from treepo._research.ctreepo.sim.core.tensor_lda_book_benchmark import TensorLDABookBenchmarkConfig
from treepo._research.ctreepo.sim.manifest import RunSpec, write_manifest_jsonl
from treepo._research.ctreepo.sim.runner import run_commands


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if x:
            out.append(x)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _cmd_prefix(python_bin: str) -> str:
    return f"{python_bin} -m src.ctreepo.cli sim run tensor-lda-books"


def _iter_runs(
    *,
    python_bin: str,
    output_root: Path,
    seeds: Iterable[int],
    n_books_train_grid: Iterable[int],
    n_books_test: int,
    calibration_rates: Iterable[float],
    eval_leaf_rates: Iterable[float],
    eval_internal_rates: Iterable[float],
    eval_internal_query_design: str,
    calibration_policy: str,
    # LDA DGP
    n_topics: int,
    vocab_size: int,
    chapters_per_book: int,
    tokens_per_chapter: int,
    alpha_topic: float,
    beta_word: float,
    chapter_concentration: float,
    # Proxy
    anchor_words_per_topic: int,
    proxy_temperature: float,
    proxy_noise_std: float,
    # Calibration knobs
    calibration_ridge: float,
    calibration_pi_min: float,
    # Thresholds / audit
    c1_threshold: float,
    c3_threshold: float,
    selection_audit_trials: int,
    selection_audit_sample_rate: float,
    selection_audit_pi_min: float,
    skip_existing: bool,
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    prefix = _cmd_prefix(str(python_bin))

    for btr in n_books_train_grid:
        for cal in calibration_rates:
            for el in eval_leaf_rates:
                for ei in eval_internal_rates:
                    for seed in seeds:
                        sub = (
                            f"train_{int(btr)}"
                            f"/cal_{_fmt_float(float(cal))}"
                            f"/eval_leaf_{_fmt_float(float(el))}"
                            f"/eval_internal_{_fmt_float(float(ei))}"
                        )
                        base = output_root / sub / f"seed_{int(seed)}"
                        out_json = base.with_suffix(".json")
                        out_csv = base.with_suffix(".csv")
                        if skip_existing and out_json.exists() and out_csv.exists():
                            continue

                        cmd = (
                            f"{prefix} "
                            f"--n-topics {int(n_topics)} --vocab-size {int(vocab_size)} "
                            f"--chapters-per-book {int(chapters_per_book)} "
                            f"--tokens-per-chapter {int(tokens_per_chapter)} "
                            f"--alpha-topic {float(alpha_topic)} --beta-word {float(beta_word)} "
                            f"--chapter-concentration {float(chapter_concentration)} "
                            f"--n-books-train {int(btr)} --n-books-test {int(n_books_test)} "
                            f"--anchor-words-per-topic {int(anchor_words_per_topic)} "
                            f"--proxy-temperature {float(proxy_temperature)} "
                            f"--proxy-noise-std {float(proxy_noise_std)} "
                            f"--calibration-leaf-query-rate {float(cal)} "
                            f"--calibration-policy {str(calibration_policy)} "
                            f"--calibration-ridge {float(calibration_ridge)} "
                            f"--calibration-pi-min {float(calibration_pi_min)} "
                            f"--eval-leaf-query-rate {float(el)} "
                            f"--eval-internal-query-rate {float(ei)} "
                            f"--eval-internal-query-design {str(eval_internal_query_design)} "
                            f"--c1-threshold {float(c1_threshold)} --c3-threshold {float(c3_threshold)} "
                            f"--selection-audit-trials {int(selection_audit_trials)} "
                            f"--selection-audit-sample-rate {float(selection_audit_sample_rate)} "
                            f"--selection-audit-pi-min {float(selection_audit_pi_min)} "
                            f"--seed {int(seed)} "
                            f"--json-summary {out_json} --csv-summary {out_csv}"
                        )

                        cfg = TensorLDABookBenchmarkConfig(
                            n_topics=int(n_topics),
                            vocab_size=int(vocab_size),
                            chapters_per_book=int(chapters_per_book),
                            tokens_per_chapter=int(tokens_per_chapter),
                            alpha_topic=float(alpha_topic),
                            beta_word=float(beta_word),
                            chapter_concentration=float(chapter_concentration),
                            n_books_train=int(btr),
                            n_books_test=int(n_books_test),
                            anchor_words_per_topic=int(anchor_words_per_topic),
                            proxy_temperature=float(proxy_temperature),
                            proxy_noise_std=float(proxy_noise_std),
                            calibration_leaf_query_rate=float(cal),
                            calibration_policy=str(calibration_policy),
                            calibration_ridge=float(calibration_ridge),
                            calibration_pi_min=float(calibration_pi_min),
                            eval_leaf_query_rate=float(el),
                            eval_internal_query_rate=float(ei),
                            eval_internal_query_design=str(eval_internal_query_design),
                            c1_threshold=float(c1_threshold),
                            c3_threshold=float(c3_threshold),
                            selection_audit_trials=int(selection_audit_trials),
                            selection_audit_sample_rate=float(selection_audit_sample_rate),
                            selection_audit_pi_min=float(selection_audit_pi_min),
                            seed=int(seed),
                        )

                        runs.append(
                            RunSpec.create(
                                family="tensor-lda-books",
                                config=asdict(cfg),
                                outputs={"json_summary": str(out_json), "csv_summary": str(out_csv)},
                                command=cmd,
                                requires=[],
                            )
                        )

    return runs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Tensor-LDA books sweep command list.")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--out-cmds", type=str, default="logs/tensor_lda_book_benchmark_cmds.txt")
    p.add_argument("--out-manifest", type=str, default="")
    p.add_argument("--output-root", type=str, default="outputs/tensor_lda_book_weight_benchmark")

    p.add_argument("--seeds", type=str, default="0 1 2 3 4 5 6 7")
    p.add_argument("--n-books-train-grid", type=str, default="64 128 256 512")
    p.add_argument("--n-books-test", type=int, default=256)
    p.add_argument("--calibration-rates", type=str, default="0 0.05 0.1 0.25 0.5")
    p.add_argument("--eval-leaf-rates", type=str, default="0")
    p.add_argument("--eval-internal-rates", type=str, default="0 0.05 0.1 0.25 0.5 1.0")
    p.add_argument("--eval-internal-query-design", choices=["none", "uniform", "risk"], default="none")
    p.add_argument("--calibration-policy", choices=["uniform", "entropy"], default="uniform")

    # LDA DGP.
    p.add_argument("--n-topics", type=int, default=5)
    p.add_argument("--vocab-size", type=int, default=600)
    p.add_argument("--chapters-per-book", type=int, default=16)
    p.add_argument("--tokens-per-chapter", type=int, default=128)
    p.add_argument("--alpha-topic", type=float, default=0.20)
    p.add_argument("--beta-word", type=float, default=0.10)
    p.add_argument("--chapter-concentration", type=float, default=40.0)

    # Proxy.
    p.add_argument("--anchor-words-per-topic", type=int, default=20)
    p.add_argument("--proxy-temperature", type=float, default=0.50)
    p.add_argument("--proxy-noise-std", type=float, default=0.05)

    # Calibration.
    p.add_argument("--calibration-ridge", type=float, default=1e-4)
    p.add_argument("--calibration-pi-min", type=float, default=0.01)

    # Thresholds / audit.
    p.add_argument("--c1-threshold", type=float, default=0.20)
    p.add_argument("--c3-threshold", type=float, default=0.20)
    p.add_argument("--selection-audit-trials", type=int, default=0)
    p.add_argument("--selection-audit-sample-rate", type=float, default=0.10)
    p.add_argument("--selection-audit-pi-min", type=float, default=0.01)

    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--log-dir", type=str, default="")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    out_cmds = Path(args.out_cmds)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    out_manifest = (
        Path(args.out_manifest)
        if str(args.out_manifest).strip()
        else out_cmds.with_name(out_cmds.stem + "_manifest.jsonl")
    )

    runs = _iter_runs(
        python_bin=str(args.python_bin),
        output_root=Path(args.output_root),
        seeds=_parse_ints(args.seeds),
        n_books_train_grid=_parse_ints(args.n_books_train_grid),
        n_books_test=int(args.n_books_test),
        calibration_rates=_parse_floats(args.calibration_rates),
        eval_leaf_rates=_parse_floats(args.eval_leaf_rates),
        eval_internal_rates=_parse_floats(args.eval_internal_rates),
        eval_internal_query_design=str(args.eval_internal_query_design),
        calibration_policy=str(args.calibration_policy),
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        chapters_per_book=int(args.chapters_per_book),
        tokens_per_chapter=int(args.tokens_per_chapter),
        alpha_topic=float(args.alpha_topic),
        beta_word=float(args.beta_word),
        chapter_concentration=float(args.chapter_concentration),
        anchor_words_per_topic=int(args.anchor_words_per_topic),
        proxy_temperature=float(args.proxy_temperature),
        proxy_noise_std=float(args.proxy_noise_std),
        calibration_ridge=float(args.calibration_ridge),
        calibration_pi_min=float(args.calibration_pi_min),
        c1_threshold=float(args.c1_threshold),
        c3_threshold=float(args.c3_threshold),
        selection_audit_trials=int(args.selection_audit_trials),
        selection_audit_sample_rate=float(args.selection_audit_sample_rate),
        selection_audit_pi_min=float(args.selection_audit_pi_min),
        skip_existing=bool(args.skip_existing),
    )

    cmds = [r.command for r in runs]
    out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    write_manifest_jsonl(out_manifest, runs)

    print(
        json.dumps(
            {
                "out_cmds": str(out_cmds),
                "out_manifest": str(out_manifest),
                "n_commands": int(len(cmds)),
            },
            indent=2,
        )
    )

    if not bool(args.execute) or not cmds:
        return 0

    log_dir = Path(args.log_dir) if str(args.log_dir).strip() else (out_cmds.parent / f"{out_cmds.stem}_logs")
    try:
        results = run_commands(
            cmds,
            jobs=int(args.jobs),
            log_dir=log_dir,
            fail_fast=bool(args.fail_fast),
        )
    except KeyboardInterrupt:
        return 130

    n_fail = sum(1 for r in results if int(r.returncode) != 0)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

