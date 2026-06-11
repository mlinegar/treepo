#!/usr/bin/env python3
# Vendored 2026-06-11 from ThinkingTrees/scripts/distill_ctreepo_students.py
# with src.* imports rewritten to treepo._research.* (TRL family's training
# subprocess; see treepo._research/ctreepo/trl_family.py train_f/train_g).
"""Export/train unified C-TreePO distillation targets from labeled trees."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from treepo._research.ctreepo.distillation import (
    DistillationContractConfig,
    DistillationTrainConfig,
    FEmbeddingConfig,
    FLMConfig,
    GLMConfig,
    ScoreTargetConfig,
    SummaryTargetConfig,
    build_f_embedding_examples,
    fit,
    load_labeled_trees,
    split_labeled_trees,
    write_jsonl_records,
)
from treepo._research.training.config_sections import (
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    TestConfig,
    TrainConfig,
    ValidationConfig,
)


LOGGER = logging.getLogger(__name__)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_example_jsonl(path: Path, examples: Iterable[Any]) -> Path:
    rows = []
    for ex in examples:
        try:
            rows.append(asdict(ex))
        except TypeError:
            rows.append(
                {
                    "doc_id": str(getattr(ex, "doc_id", "")),
                    "text": str(getattr(ex, "text", "")),
                    "target_score": float(getattr(ex, "target_score", 0.0)),
                    "truth_label_source": str(
                        getattr(ex, "truth_label_source", "unknown")
                    ),
                }
            )
    return write_jsonl_records(path, rows)


def _build_trl_config(args: argparse.Namespace):
    from treepo._research.training.trl_training import (
        TRLLoraConfig,
        TRLQuantizationConfig,
        TRLSequenceConfig,
        TRLTrainingConfig,
    )

    return TRLTrainingConfig(
        train=TrainConfig(
            epochs=int(args.trl_epochs),
            batch_size=int(args.trl_batch_size),
            gradient_accumulation_steps=int(args.trl_grad_accumulation_steps),
        ),
        optimizer=OptimizerConfig(learning_rate=float(args.trl_learning_rate)),
        runtime=RuntimeConfig(
            bf16=not bool(args.no_bf16),
            gradient_checkpointing=True,
        ),
        lora=TRLLoraConfig(use_lora=not bool(args.no_lora)),
        quantization=TRLQuantizationConfig(load_in_4bit=not bool(args.no_4bit)),
        sequence=TRLSequenceConfig(max_length=int(args.trl_max_length)),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build g-target SFT rows and f-target scorer artifacts "
            "from C-TreePO LabeledTree JSON/JSONL files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labeled-tree-artifacts",
        type=Path,
        nargs="+",
        required=True,
        help="LabeledTree JSON/JSONL file(s) or directories from Stage 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/ctreepo_distill/<timestamp>.",
    )
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--val-splits", nargs="+", default=["val"])
    parser.add_argument("--test-splits", nargs="+", default=["test"])
    parser.add_argument(
        "--disable-test-export",
        action="store_true",
        help="Do not export/evaluate held-out test split records.",
    )
    parser.add_argument(
        "--include-identity-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use node text as a fallback target when no teacher summary exists.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Write g/f dataset snapshots but do not train the f proxy.",
    )
    parser.add_argument(
        "--skip-g-export",
        action="store_true",
        help="Do not write g-target SFT JSONL files.",
    )
    parser.add_argument(
        "--skip-f-fit",
        action="store_true",
        help="Do not fit the f-target embedding proxy.",
    )
    parser.add_argument(
        "--export-f-lm-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write scalar-regression JSONL rows for a small-LM f target.",
    )
    parser.add_argument(
        "--run-g-sft",
        action="store_true",
        help="Run TRL SFT for the g target after exporting SFT rows.",
    )
    parser.add_argument("--g-model-name", type=str, default=None)
    parser.add_argument(
        "--run-f-lm-regression",
        action="store_true",
        help="Run TRL sequence-classification regression for the f target.",
    )
    parser.add_argument("--f-lm-model-name", type=str, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a previously-trained HF model directory to warmstart from. "
            "Overrides --g-model-name / --f-lm-model-name when set. Honors the "
            "'never reset between rungs' invariant: train_f at k>=2 (or train_g "
            "at k>=2) must resume from the prior iteration, never from the raw "
            "base model."
        ),
    )

    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=32)

    parser.add_argument("--f-method", choices=["ridge", "linear_sgd"], default="ridge")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--f-epochs", type=int, default=25)
    parser.add_argument("--f-learning-rate", type=float, default=5e-3)
    parser.add_argument("--f-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--f-model-id",
        type=str,
        default="ctreepo_f_embedding_proxy",
    )
    parser.add_argument("--target-min", type=float, default=-100.0)
    parser.add_argument("--target-max", type=float, default=100.0)
    parser.add_argument("--trl-learning-rate", type=float, default=1e-5)
    parser.add_argument("--trl-epochs", type=int, default=3)
    parser.add_argument("--trl-batch-size", type=int, default=2)
    parser.add_argument("--trl-grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--trl-max-length", type=int, default=2048)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "ctreepo_distill" / _now_stamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_trees = []
    for path in args.labeled_tree_artifacts:
        loaded = load_labeled_trees(path)
        LOGGER.info("Loaded %d labeled tree(s) from %s", len(loaded), path)
        labeled_trees.extend(loaded)
    if not labeled_trees:
        raise ValueError("No labeled trees loaded")

    train_trees, val_trees = split_labeled_trees(
        labeled_trees,
        train_splits=args.train_splits,
        val_splits=args.val_splits,
    )
    if not train_trees:
        raise ValueError("No training labeled trees selected")

    artifacts: Dict[str, Optional[str]] = {
        "g_train": None,
        "g_val": None,
        "g_test": None,
        "f_train_examples": None,
        "f_val_examples": None,
        "f_lm_train": None,
        "f_lm_val": None,
        "f_lm_test": None,
        "f_embedding_proxy": None,
        "g_model": None,
        "f_lm_model": None,
    }

    g_train_count = 0
    g_val_count = 0
    g_metadata: Dict[str, Any] = {"status": "skipped", "reason": "skip_g_export"}
    if not bool(args.skip_g_export):
        run_g_sft = bool(args.run_g_sft) and not bool(args.export_only)
        # --init-checkpoint overrides --g-model-name when set, so a k>=2
        # iteration resumes from the prior iterate's dir rather than the base.
        effective_g_model = args.init_checkpoint or args.g_model_name
        if run_g_sft and not effective_g_model:
            raise ValueError(
                "--run-g-sft requires either --g-model-name or --init-checkpoint"
            )
        g_result = fit(
            labeled_trees,
            DistillationTrainConfig(
                contract=DistillationContractConfig(
                    train_targets=("g",),
                    student_model_class="lm_sft",
                    supervision_source="labeled_tree_artifact",
                ),
                run=RunConfig(output_dir=output_dir),
                train=TrainConfig(train_splits=tuple(args.train_splits)),
                validation=ValidationConfig(val_splits=tuple(args.val_splits)),
                test=TestConfig(
                    test_splits=tuple(args.test_splits),
                    enabled=not bool(args.disable_test_export),
                ),
                summary_targets=SummaryTargetConfig(
                    include_identity_targets=bool(args.include_identity_targets),
                ),
                g_lm=GLMConfig(
                    run_trl_sft=run_g_sft,
                    model_name=effective_g_model,
                    trl_config=_build_trl_config(args) if run_g_sft else None,
                ),
            ),
        )
        artifacts["g_train"] = g_result.metadata.get("sft_train_path")
        artifacts["g_val"] = g_result.metadata.get("sft_val_path")
        artifacts["g_test"] = g_result.metadata.get("sft_test_path")
        artifacts["g_model"] = g_result.trained_artifact
        g_train_count = int(g_result.metadata.get("sft_train_records", 0))
        g_val_count = int(g_result.metadata.get("sft_val_records", 0))
        g_metadata = dict(g_result.metadata)

    f_train_examples = build_f_embedding_examples(
        train_trees,
        include_identity_targets=bool(args.include_identity_targets),
        target_min=float(args.target_min),
        target_max=float(args.target_max),
    )
    f_val_examples = build_f_embedding_examples(
        val_trees,
        include_identity_targets=bool(args.include_identity_targets),
        target_min=float(args.target_min),
        target_max=float(args.target_max),
    )
    artifacts["f_train_examples"] = str(
        _write_example_jsonl(output_dir / "f_embedding_train_examples.jsonl", f_train_examples)
    )
    artifacts["f_val_examples"] = str(
        _write_example_jsonl(output_dir / "f_embedding_val_examples.jsonl", f_val_examples)
    )

    f_lm_metadata: Dict[str, Any] = {
        "status": "skipped",
        "reason": "export_f_lm_records_disabled",
    }
    if bool(args.export_f_lm_records):
        run_f_lm = bool(args.run_f_lm_regression) and not bool(args.export_only)
        effective_f_model = args.init_checkpoint or args.f_lm_model_name
        if run_f_lm and not effective_f_model:
            raise ValueError(
                "--run-f-lm-regression requires --f-lm-model-name or --init-checkpoint"
            )
        f_lm_result = fit(
            labeled_trees,
            DistillationTrainConfig(
                contract=DistillationContractConfig(
                    train_targets=("f",),
                    student_model_class="lm_scalar_regression",
                    supervision_source="labeled_tree_artifact",
                ),
                run=RunConfig(output_dir=output_dir),
                train=TrainConfig(train_splits=tuple(args.train_splits)),
                validation=ValidationConfig(val_splits=tuple(args.val_splits)),
                test=TestConfig(
                    test_splits=tuple(args.test_splits),
                    enabled=not bool(args.disable_test_export),
                ),
                score_targets=ScoreTargetConfig(
                    include_identity_targets=bool(args.include_identity_targets),
                    target_min=float(args.target_min),
                    target_max=float(args.target_max),
                ),
                f_lm=FLMConfig(
                    run_trl_scalar_reward=run_f_lm,
                    model_name=effective_f_model,
                    trl_config=_build_trl_config(args) if run_f_lm else None,
                ),
            ),
        )
        artifacts["f_lm_train"] = f_lm_result.metadata.get("lm_regression_train_path")
        artifacts["f_lm_val"] = f_lm_result.metadata.get("lm_regression_val_path")
        artifacts["f_lm_test"] = f_lm_result.metadata.get("lm_regression_test_path")
        artifacts["f_lm_model"] = f_lm_result.trained_artifact
        f_lm_metadata = dict(f_lm_result.metadata)

    f_metadata: Dict[str, Any] = {
        "status": "skipped",
        "reason": "export_only" if bool(args.export_only) else "skip_f_fit",
    }
    if not bool(args.export_only) and not bool(args.skip_f_fit):
        from treepo._research.config.settings import get_embedding_model, get_embedding_url, load_settings
        from treepo._research.training.embedding_proxy import VLLMEmbeddingClient

        settings = load_settings()
        embedding_url = (args.embedding_url or get_embedding_url(settings)).rstrip("/")
        embedding_model = args.embedding_model or get_embedding_model(settings) or None
        client = VLLMEmbeddingClient(
            api_base=embedding_url,
            model=embedding_model,
            api_key=str(args.embedding_api_key),
            timeout_seconds=float(args.embedding_timeout_seconds),
            batch_size=int(args.embedding_batch_size),
        )
        result = fit(
            labeled_trees,
            DistillationTrainConfig(
                contract=DistillationContractConfig(
                    train_targets=("f",),
                    student_model_class="embedding_ridge_proxy",
                    supervision_source="labeled_tree_artifact",
                ),
                run=RunConfig(output_dir=output_dir),
                train=TrainConfig(train_splits=tuple(args.train_splits)),
                validation=ValidationConfig(val_splits=tuple(args.val_splits)),
                test=TestConfig(
                    test_splits=tuple(args.test_splits),
                    enabled=not bool(args.disable_test_export),
                ),
                score_targets=ScoreTargetConfig(
                    include_identity_targets=bool(args.include_identity_targets),
                    target_min=float(args.target_min),
                    target_max=float(args.target_max),
                ),
                f_embedding=FEmbeddingConfig(
                    method=str(args.f_method),
                    ridge_lambda=float(args.ridge_lambda),
                    epochs=int(args.f_epochs),
                    learning_rate=float(args.f_learning_rate),
                    weight_decay=float(args.f_weight_decay),
                    model_id=str(args.f_model_id),
                ),
            ),
            embedding_client=client,
        )
        artifacts["f_embedding_proxy"] = str(
            result.metadata.get("model_path") or output_dir / "f_embedding_proxy.json"
        )
        f_metadata = dict(result.metadata)

    manifest = {
        "created_at": _now_iso(),
        "status": "completed",
        "inputs": [str(path) for path in args.labeled_tree_artifacts],
        "config": {
            "train_splits": list(args.train_splits),
            "val_splits": list(args.val_splits),
            "test_splits": list(args.test_splits),
            "include_identity_targets": bool(args.include_identity_targets),
            "export_only": bool(args.export_only),
            "test_export_enabled": not bool(args.disable_test_export),
            "skip_g_export": bool(args.skip_g_export),
            "skip_f_fit": bool(args.skip_f_fit),
            "export_f_lm_records": bool(args.export_f_lm_records),
            "run_g_sft": bool(args.run_g_sft) and not bool(args.export_only),
            "run_f_lm_regression": bool(args.run_f_lm_regression) and not bool(args.export_only),
            "f_method": str(args.f_method),
            "target_min": float(args.target_min),
            "target_max": float(args.target_max),
        },
        "counts": {
            "labeled_trees": int(len(labeled_trees)),
            "train_trees": int(len(train_trees)),
            "val_trees": int(len(val_trees)),
            "g_train_records": int(g_train_count),
            "g_val_records": int(g_val_count),
            "g_test_records": int(g_metadata.get("sft_test_records", 0)),
            "f_train_examples": int(len(f_train_examples)),
            "f_val_examples": int(len(f_val_examples)),
            "f_lm_train_records": int(f_lm_metadata.get("lm_regression_train_records", 0)),
            "f_lm_val_records": int(f_lm_metadata.get("lm_regression_val_records", 0)),
            "f_lm_test_records": int(f_lm_metadata.get("lm_regression_test_records", 0)),
        },
        "artifacts": artifacts,
        "distillation_contracts": {
            "g": g_metadata.get("distillation_contract"),
            "f_embedding_proxy": f_metadata.get("distillation_contract"),
            "f_lm_regression": f_lm_metadata.get("distillation_contract"),
        },
        "g_target": g_metadata,
        "f_embedding_target": f_metadata,
        "f_lm_target": f_lm_metadata,
    }
    _write_json(output_dir / "distillation_manifest.json", manifest)
    LOGGER.info("Distillation artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
