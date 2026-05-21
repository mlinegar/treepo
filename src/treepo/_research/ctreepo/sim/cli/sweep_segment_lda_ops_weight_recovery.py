#!/usr/bin/env python3
"""Build (and optionally execute) xargs-friendly sweeps for Segment-LDA OPS weight recovery."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

from treepo._research.ctreepo.sim.core.segment_lda_ops_weight_recovery import (
    SegmentLDAOpsWeightRecoveryConfig,
    VALID_DEVICE_MODES,
)
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
    return f"{python_bin} -m src.ctreepo.cli sim run segment-lda-ops"


def _iter_runs(
    *,
    python_bin: str,
    train_docs: Iterable[int],
    test_docs: int,
    audit_fractions: Iterable[float],
    topic_phi_docs: Iterable[int],
    topic_phi_estimators: Iterable[str],
    topic_processes: Iterable[str],
    lambda_multipliers: Iterable[float],
    seeds: Iterable[int],
    output_root: Path,
    topic_source: str,
    feature_inference: str,
    n_topics: int,
    vocab_size: int,
    min_tokens: int,
    max_tokens: int,
    leaf_tokens: int,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    run_all_feature_modes: bool,
    skip_existing: bool,
) -> List[RunSpec]:
    runs: List[RunSpec] = []

    all_feature_flag = "--run-all-feature-modes" if bool(run_all_feature_modes) else "--no-run-all-feature-modes"
    prefix = _cmd_prefix(str(python_bin))

    for est in topic_phi_estimators:
        for proc in topic_processes:
            for td in train_docs:
                for af in audit_fractions:
                    for phi_docs in topic_phi_docs:
                        for lam in lambda_multipliers:
                            for seed in seeds:
                                sub = (
                                    f"phi_{est}"
                                    f"/proc_{proc}"
                                    f"/train_{int(td)}"
                                    f"/audit_{_fmt_float(af)}"
                                    f"/phi_docs_{int(phi_docs)}"
                                    f"/lam_{_fmt_float(lam)}"
                                )
                                base = output_root / sub / f"seed_{int(seed)}"
                                out_json = base.with_suffix(".json")
                                out_csv = base.with_suffix(".csv")
                                if skip_existing and out_json.exists() and out_csv.exists():
                                    continue

                                cmd = (
                                    f"{prefix} "
                                    f"--n-topics {int(n_topics)} --vocab-size {int(vocab_size)} "
                                    f"--min-tokens {int(min_tokens)} --max-tokens {int(max_tokens)} "
                                    f"--leaf-tokens {int(leaf_tokens)} "
                                    f"--topic-process {proc} "
                                    f"--lambda-multiplier {float(lam)} "
                                    f"--topic-source {topic_source} "
                                    f"--feature-inference {feature_inference} "
                                    f"--audit-policy fraction --audit-fraction {float(af)} "
                                    f"--topic-phi-estimator {est} --topic-phi-docs {int(phi_docs)} "
                                    f"--device {str(device)} "
                                    f"{all_feature_flag} "
                                    f"--train-docs {int(td)} --test-docs {int(test_docs)} "
                                    f"--torch-threads {int(torch_threads)} "
                                    f"--seed {int(seed)} "
                                    f"--json-summary {out_json} --csv-summary {out_csv}"
                                )
                                if cuda_device is not None:
                                    cmd += f" --cuda-device {int(cuda_device)}"

                                cfg = SegmentLDAOpsWeightRecoveryConfig(
                                    n_topics=int(n_topics),
                                    vocab_size=int(vocab_size),
                                    min_tokens=int(min_tokens),
                                    max_tokens=int(max_tokens),
                                    leaf_tokens=int(leaf_tokens),
                                    topic_process=str(proc),
                                    lambda_multiplier=float(lam),
                                    topic_source=str(topic_source),
                                    feature_inference=str(feature_inference),
                                    audit_policy="fraction",
                                    audit_fraction=float(af),
                                    topic_phi_estimator=str(est),
                                    topic_phi_docs=int(phi_docs),
                                    device=str(device),
                                    cuda_device=int(cuda_device) if cuda_device is not None else None,
                                    torch_threads=int(torch_threads),
                                    run_all_feature_modes=bool(run_all_feature_modes),
                                    train_docs=int(td),
                                    test_docs=int(test_docs),
                                    seed=int(seed),
                                )

                                requires: List[str] = []
                                est_norm = str(est).strip().lower()
                                if est_norm == "sklearn_lda":
                                    requires.append("sklearn")
                                if est_norm.startswith("neural_"):
                                    requires.append("torch")

                                runs.append(
                                    RunSpec.create(
                                        family="segment-lda-ops",
                                        config=asdict(cfg),
                                        outputs={"json_summary": str(out_json), "csv_summary": str(out_csv)},
                                        command=cmd,
                                        requires=sorted(set(requires)),
                                    )
                                )

    return runs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Segment-LDA OPS weight-recovery sweep command list.")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--out-cmds", type=str, default="logs/segment_lda_ops_weight_recovery_cmds.txt")
    p.add_argument("--out-manifest", type=str, default="")
    p.add_argument("--output-root", type=str, default="outputs/segment_lda_ops_weight_recovery")

    p.add_argument("--train-docs", type=str, default="100 200 500 1000 2000")
    p.add_argument("--test-docs", type=int, default=2000)
    p.add_argument("--audit-fractions", type=str, default="0.05 0.1 0.2 0.5 1.0")
    p.add_argument("--topic-phi-docs", type=str, default="0")
    p.add_argument(
        "--topic-phi-estimators",
        type=str,
        default=(
            "true noisy_theory tensor_lda online_tensor_lda embedding_spectral "
            "neural_ctreepo neural_mergeable_sketch neural_hybrid neural_embedding_hybrid"
        ),
        help="Space-separated list.",
    )
    p.add_argument("--topic-processes", type=str, default="segments bag_of_words", help="Space-separated list.")
    p.add_argument("--lambda-multipliers", type=str, default="0 0.25 1.0", help="Space-separated list.")
    p.add_argument("--seeds", type=str, default="0 1 2 3 4 5 6 7")

    p.add_argument("--topic-source", choices=["true", "infer"], default="infer")
    p.add_argument("--feature-inference", choices=["hard", "soft"], default="hard")

    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=384)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--leaf-tokens", type=int, default=16)
    p.add_argument("--device", type=str, choices=list(VALID_DEVICE_MODES), default="auto")
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=0)

    p.add_argument("--run-all-feature-modes", action=argparse.BooleanOptionalAction, default=False)
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
        train_docs=_parse_ints(args.train_docs),
        test_docs=int(args.test_docs),
        audit_fractions=_parse_floats(args.audit_fractions),
        topic_phi_docs=_parse_ints(args.topic_phi_docs),
        topic_phi_estimators=_parse_items(args.topic_phi_estimators),
        topic_processes=_parse_items(args.topic_processes),
        lambda_multipliers=_parse_floats(args.lambda_multipliers),
        seeds=_parse_ints(args.seeds),
        output_root=Path(args.output_root),
        topic_source=str(args.topic_source),
        feature_inference=str(args.feature_inference),
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        leaf_tokens=int(args.leaf_tokens),
        device=str(args.device),
        cuda_device=(int(args.cuda_device) if args.cuda_device is not None else None),
        torch_threads=int(args.torch_threads),
        run_all_feature_modes=bool(args.run_all_feature_modes),
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
