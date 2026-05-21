from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Dict, List, Sequence

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    build_markov_changepoint_ops_count_data_bundle,
)
from treepo._research.ctreepo.sim.manifest import RunSpec
from treepo._research.ctreepo.sim.suite.common import (
    SuiteGroupRuns,
    build_suite_meta,
    emit_grouped_suite_artifacts,
    read_suite_meta,
    resolve_grouped_suite_paths,
    run_manifest_queue_suite,
    select_known_items,
    utc_run_id,
    write_suite_meta,
)
from treepo._research.ctreepo.sim.suite.markov_observed_token_policy import (
    MarkovObservedTokenPolicy,
    resolve_markov_observed_token_policy,
)


_GROUPS: tuple[str, ...] = ("root_only", "local_labels")


def _resolve_policy_from_args(args: argparse.Namespace) -> MarkovObservedTokenPolicy:
    return resolve_markov_observed_token_policy(
        profile_name=str(args.profile),
        train_docs=(int(args.train_docs) if int(args.train_docs) > 0 else None),
        val_docs=(int(args.val_docs) if int(args.val_docs) > 0 else None),
        test_docs=(int(args.test_docs) if int(args.test_docs) > 0 else None),
        state_dim=(int(args.state_dim) if int(args.state_dim) > 0 else None),
        hidden_dim=(int(args.hidden_dim) if int(args.hidden_dim) > 0 else None),
        n_epochs=(int(args.n_epochs) if int(args.n_epochs) > 0 else None),
        batch_size=(int(args.batch_size) if int(args.batch_size) > 0 else None),
        lr=(float(args.lr) if float(args.lr) > 0.0 else None),
        doc_sequence_objective=(
            str(args.doc_sequence_objective).strip() if str(args.doc_sequence_objective).strip() else None
        ),
        doc_transformer_head_family=(
            str(args.doc_transformer_head_family).strip()
            if str(args.doc_transformer_head_family).strip()
            else None
        ),
        doc_transformer_layers=(
            int(args.doc_transformer_layers) if int(args.doc_transformer_layers) > 0 else None
        ),
        seed=(int(args.seed) if int(args.seed) >= 0 else None),
        device=(str(args.device) if str(args.device).strip() else None),
        torch_threads=(int(args.torch_threads) if int(args.torch_threads) > 0 else None),
    )


def _resources_for_policy(policy: MarkovObservedTokenPolicy) -> Dict[str, object]:
    device = str(policy.device).strip().lower()
    accelerator = "cpu"
    gpu_eligible = False
    gpu_preferred = False
    if device == "cuda":
        accelerator = "gpu"
        gpu_eligible = True
        gpu_preferred = True
    elif device == "auto":
        accelerator = "auto"
        gpu_eligible = True
    return {
        "accelerator": accelerator,
        "device_mode": device or "cpu",
        "gpu_eligible": bool(gpu_eligible),
        "gpu_preferred": bool(gpu_preferred),
        "cpu_threads": int(max(1, policy.torch_threads)),
        "torch_threads": int(policy.torch_threads),
    }


def _bundle_config(policy: MarkovObservedTokenPolicy) -> OPSCountConfig:
    return OPSCountConfig(
        generator_profile=str(policy.generator_profile),
        n_regimes=int(policy.n_regimes),
        vocab_size=int(policy.vocab_size),
        min_tokens=int(policy.min_tokens),
        max_tokens=int(policy.max_tokens),
        min_segments=int(policy.min_segments),
        max_segments=int(policy.max_segments),
        min_seg_len=int(policy.min_seg_len),
        max_seg_len=int(policy.max_seg_len),
        fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
        train_docs=int(policy.train_docs),
        val_docs=int(policy.val_docs),
        test_docs=int(policy.test_docs),
        feature_mode="token_full",
        state_dim=int(policy.state_dim),
        hidden_dim=int(policy.hidden_dim),
        n_epochs=int(policy.n_epochs),
        batch_size=int(policy.batch_size),
        lr=float(policy.lr),
        weight_decay=float(policy.weight_decay),
        seed=int(policy.seed),
        use_cuda=False,
        torch_threads=int(policy.torch_threads),
    )


def _ensure_bundle(
    *,
    output_root: Path,
    policy: MarkovObservedTokenPolicy,
    bundle_file: Path | None = None,
) -> tuple[Path, MarkovOPSDataBundle]:
    bundle_path = output_root / "markov_data" / "observed_token_bundle.json"
    if bundle_file is not None:
        source = bundle_file.resolve()
        bundle = MarkovOPSDataBundle.load(source)
        if bundle_path.resolve() != source:
            bundle.save(bundle_path)
        return bundle_path, bundle
    if bundle_path.exists():
        return bundle_path, MarkovOPSDataBundle.load(bundle_path)
    bundle = build_markov_changepoint_ops_count_data_bundle(_bundle_config(policy))
    bundle.save(bundle_path)
    return bundle_path, bundle


def _base_argv(
    *,
    python_bin: str,
    policy: MarkovObservedTokenPolicy,
    bundle_path: Path,
    json_path: Path,
    csv_path: Path,
) -> List[str]:
    return [
        str(python_bin),
        "-m",
        "src.ctreepo.cli",
        "sim",
        "run",
        "markov-ops-count",
        "--model-family",
        "neural",
        "--generator-profile",
        str(policy.generator_profile),
        "--feature-mode",
        "token_full",
        "--n-regimes",
        str(int(policy.n_regimes)),
        "--vocab-size",
        str(int(policy.vocab_size)),
        "--min-tokens",
        str(int(policy.min_tokens)),
        "--max-tokens",
        str(int(policy.max_tokens)),
        "--min-segments",
        str(int(policy.min_segments)),
        "--max-segments",
        str(int(policy.max_segments)),
        "--min-seg-len",
        str(int(policy.min_seg_len)),
        "--max-seg-len",
        str(int(policy.max_seg_len)),
        "--fixed-leaf-tokens",
        str(int(policy.fixed_leaf_tokens)),
        "--train-docs",
        str(int(policy.train_docs)),
        "--val-docs",
        str(int(policy.val_docs)),
        "--test-docs",
        str(int(policy.test_docs)),
        "--state-dim",
        str(int(policy.state_dim)),
        "--hidden-dim",
        str(int(policy.hidden_dim)),
        "--n-epochs",
        str(int(policy.n_epochs)),
        "--batch-size",
        str(int(policy.batch_size)),
        "--lr",
        str(float(policy.lr)),
        "--weight-decay",
        str(float(policy.weight_decay)),
        "--seed",
        str(int(policy.seed)),
        "--device",
        str(policy.device),
        "--torch-threads",
        str(int(policy.torch_threads)),
        "--rf-n-estimators",
        str(int(policy.rf_n_estimators)),
        "--rf-max-depth",
        str(int(policy.rf_max_depth)),
        "--rf-min-samples-leaf",
        str(int(policy.rf_min_samples_leaf)),
        "--leaf-knn-neighbors",
        str(int(policy.leaf_knn_neighbors)),
        "--doc-level-ridge-alpha",
        str(float(policy.doc_level_ridge_alpha)),
        "--doc-level-ridge-breakdown-orders",
        ",".join(str(int(x)) for x in tuple(policy.doc_level_ridge_breakdown_orders)),
        "--doc-sequence-objective",
        str(policy.doc_sequence_objective),
        "--doc-transformer-head-family",
        str(policy.doc_transformer_head_family),
        "--doc-transformer-layers",
        str(int(policy.doc_transformer_layers)),
        "--load-data-bundle",
        str(bundle_path),
        "--json-summary",
        str(json_path),
        "--csv-summary",
        str(csv_path),
    ]


def _run_spec(
    *,
    python_bin: str,
    output_root: Path,
    bundle_path: Path,
    policy: MarkovObservedTokenPolicy,
    group_key: str,
    skip_existing: bool,
) -> RunSpec | None:
    summary_root = output_root / "markov_changepoint_ops_count" / group_key
    json_path = summary_root / f"seed_{int(policy.seed)}.json"
    csv_path = summary_root / f"seed_{int(policy.seed)}.csv"
    if bool(skip_existing) and json_path.exists() and csv_path.exists():
        return None

    argv = _base_argv(
        python_bin=python_bin,
        policy=policy,
        bundle_path=bundle_path,
        json_path=json_path,
        csv_path=csv_path,
    )
    if group_key == "root_only":
        argv.extend(
            [
                "--audit-fraction",
                "0.0",
                "--leaf-query-rate",
                "0.0",
                "--include-doc-level-baseline",
                "--include-doc-sequence-baseline",
                "--include-doc-transformer-baseline",
                "--include-doc-level-ridge-baseline",
                "--include-rf-root-baseline",
            ]
        )
    elif group_key == "local_labels":
        argv.extend(
            [
                "--audit-fraction",
                "1.0",
                "--leaf-query-rate",
                "1.0",
                "--local-law-weight",
                str(float(policy.local_law_weight)),
                "--c1-relative-weight",
                str(float(policy.c1_relative_weight)),
                "--c2-relative-weight",
                str(float(policy.c2_relative_weight)),
                "--c3-relative-weight",
                str(float(policy.c3_relative_weight)),
                "--include-doc-level-baseline",
                "--include-doc-sequence-baseline",
                "--include-doc-transformer-baseline",
                "--include-doc-level-ridge-baseline",
                "--include-rf-root-baseline",
                "--include-leaf-ridge-tree-baseline",
                "--include-leaf-endpoint-table-tree-baseline",
                "--include-leaf-dt-tree-baseline",
                "--include-leaf-knn-tree-baseline",
                "--include-leaf-rf-tree-baseline",
            ]
        )
    else:
        raise ValueError(f"unknown group: {group_key}")

    command = " ".join(shlex.quote(part) for part in argv)
    return RunSpec.create(
        family="markov-ops-count",
        config={
            **policy.to_dict(),
            "feature_mode": "token_full",
            "comparison_mode": str(group_key),
            "suite_group": str(group_key),
            "data_bundle_file": str(bundle_path),
        },
        outputs={
            "json_summary": str(json_path),
            "csv_summary": str(csv_path),
        },
        command=command,
        resources=_resources_for_policy(policy),
    )


def _build_groups(
    *,
    python_bin: str,
    output_root: Path,
    policy: MarkovObservedTokenPolicy,
    skip_existing: bool,
    bundle_file: Path | None,
) -> tuple[List[SuiteGroupRuns], Path, MarkovOPSDataBundle]:
    bundle_path, bundle = _ensure_bundle(
        output_root=output_root,
        policy=policy,
        bundle_file=bundle_file,
    )
    groups: List[SuiteGroupRuns] = []
    for key in _GROUPS:
        run = _run_spec(
            python_bin=python_bin,
            output_root=output_root,
            bundle_path=bundle_path,
            policy=policy,
            group_key=key,
            skip_existing=skip_existing,
        )
        groups.append(
            SuiteGroupRuns(
                key=str(key),
                family="markov-ops-count",
                runs=[] if run is None else [run],
            )
        )
    return groups, bundle_path, bundle


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    requested_groups: Sequence[str],
    policy: MarkovObservedTokenPolicy,
    skip_existing: bool,
    bundle_file: Path | None = None,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    paths = resolve_grouped_suite_paths(output_root)
    selected_groups = select_known_items(
        requested=requested_groups,
        available=_GROUPS,
        item_name="markov observed-token groups",
    )
    groups, bundle_path, bundle = _build_groups(
        python_bin=python_bin,
        output_root=output_root,
        policy=policy,
        skip_existing=skip_existing,
        bundle_file=bundle_file,
    )
    filtered_groups = [group for group in groups if group.key in selected_groups]
    artifacts = emit_grouped_suite_artifacts(paths, filtered_groups)
    meta = build_suite_meta(
        suite_name="markov-observed-token",
        suite_role="diagnostic",
        run_id=str(run_id),
        profile=str(policy.profile),
        policy=policy.to_dict(),
        python_bin=str(python_bin),
        output_root=output_root,
        cmds_file=paths.suite_cmds,
        manifest_file=paths.suite_manifest,
        selected_groups=selected_groups,
        group_cmd_files=artifacts.group_cmd_files,
        group_manifest_files=artifacts.group_manifest_files,
        group_families=artifacts.group_families,
        extra={
            "skip_existing": bool(skip_existing),
            "counts_by_group": artifacts.counts_by_group,
            "n_commands_total": int(len(artifacts.all_cmds)),
            "data_bundle_file": str(bundle_path),
            "data_signatures": {
                "train": str(bundle.train_corpus_signature),
                "val": str(bundle.val_corpus_signature),
                "test": str(bundle.test_corpus_signature),
            },
            "source_bundle_file": str(bundle_file.resolve()) if bundle_file is not None else "",
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observed-token Markov comparison suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("build", "run"):
        subp = sub.add_parser(name)
        subp.add_argument("--run-id", type=str, default="")
        subp.add_argument("--python-bin", type=str, default="")
        subp.add_argument("--output-root", type=str, default="" if name == "build" else None, required=name == "run")
        subp.add_argument("--groups", type=str, default="")
        subp.add_argument("--profile", type=str, default="smoke")
        subp.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
        subp.add_argument("--train-docs", type=int, default=0)
        subp.add_argument("--val-docs", type=int, default=0)
        subp.add_argument("--test-docs", type=int, default=0)
        subp.add_argument("--state-dim", type=int, default=0)
        subp.add_argument("--hidden-dim", type=int, default=0)
        subp.add_argument("--n-epochs", type=int, default=0)
        subp.add_argument("--batch-size", type=int, default=0)
        subp.add_argument("--lr", type=float, default=0.0)
        subp.add_argument("--doc-sequence-objective", type=str, default="")
        subp.add_argument("--doc-transformer-head-family", type=str, default="")
        subp.add_argument("--doc-transformer-layers", type=int, default=0)
        subp.add_argument("--seed", type=int, default=-1)
        subp.add_argument("--device", type=str, default="")
        subp.add_argument("--torch-threads", type=int, default=0)
        subp.add_argument("--bundle-file", type=str, default="")
        if name == "run":
            subp.add_argument("--jobs", type=int, default=1)
            subp.add_argument("--gpu-tokens", type=str, default="auto")
            subp.add_argument("--log-dir", type=str, default="")
            subp.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
            subp.add_argument("--set-thread-env", action=argparse.BooleanOptionalAction, default=True)

    report = sub.add_parser("report")
    report.add_argument("--output-root", type=str, required=True)
    report.add_argument("--out-dir", type=str, default="")
    report.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        run_id = utc_run_id(args.run_id)
        output_root = (
            Path(args.output_root)
            if str(args.output_root).strip()
            else Path(f"outputs/markov_observed_token_{run_id}")
        )
        meta = build_suite(
            run_id=run_id,
            python_bin=str(args.python_bin).strip() or sys.executable,
            output_root=output_root,
            requested_groups=[x for x in str(args.groups).replace(",", " ").split() if x.strip()],
            policy=_resolve_policy_from_args(args),
            skip_existing=bool(args.skip_existing),
            bundle_file=Path(str(args.bundle_file).strip()).resolve()
            if str(args.bundle_file).strip()
            else None,
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    if args.cmd == "run":
        output_root = Path(args.output_root).resolve()
        paths = resolve_grouped_suite_paths(output_root)
        if bool(args.rebuild) or not paths.suite_meta.exists() or not paths.suite_manifest.exists():
            build_suite(
                run_id=utc_run_id(args.run_id or output_root.name),
                python_bin=str(args.python_bin).strip() or sys.executable,
                output_root=output_root,
                requested_groups=[x for x in str(args.groups).replace(",", " ").split() if x.strip()],
                policy=_resolve_policy_from_args(args),
                skip_existing=bool(args.skip_existing),
                bundle_file=Path(str(args.bundle_file).strip()).resolve()
                if str(args.bundle_file).strip()
                else None,
            )
        meta = read_suite_meta(paths.suite_meta)
        built_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
        selected_groups = select_known_items(
            requested=[x for x in str(args.groups).replace(",", " ").split() if x.strip()],
            available=built_groups,
            item_name="markov observed-token groups",
        )
        manifest_files = dict(meta.get("group_manifest_files", {}) or {})
        manifest_paths = [Path(str(manifest_files[key])) for key in selected_groups]
        payload = run_manifest_queue_suite(
            manifest_paths=manifest_paths,
            cpu_workers=int(args.jobs),
            gpu_tokens=str(args.gpu_tokens),
            log_dir=Path(args.log_dir).resolve() if str(args.log_dir).strip() else paths.queue_log_dir,
            set_thread_env=bool(args.set_thread_env),
        )
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "selected_groups": selected_groups,
                    **payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if int(payload["summary"].get("n_fail", 0)) == 0 else 1

    if args.cmd == "report":
        from treepo._research.ctreepo.sim.cli.report.markov_observed_token import main as _report_main

        report_argv: List[str] = ["--output-root", str(Path(args.output_root).resolve())]
        if str(args.out_dir).strip():
            report_argv.extend(["--out-dir", str(Path(args.out_dir).resolve())])
        report_argv.append("--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf")
        return int(_report_main(report_argv))

    raise ValueError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
