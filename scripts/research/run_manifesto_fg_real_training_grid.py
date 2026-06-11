#!/usr/bin/env python3
# Vendored 2026-06-11 from ThinkingTrees/scripts/run_manifesto_fg_real_training_grid.py
# with src.* imports rewritten to treepo._research.* (paper teacher-metrics script;
# tests/methods/test_manifesto_paper_parity.py pins _fg_teacher_metrics against it).
"""Run real LM f/g distillation for the manifesto economic f/g grid.

This is the heavyweight companion to ``build_manifesto_fg_ladder.py``.  The
ladder script maps teacher artifacts and exports proxy datasets; this runner
launches the actual TRL-backed students through ``distill_ctreepo_students.py``
and writes one manifest per ``{init_mode, leaf_count, stage}``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from treepo._research.ctreepo.distillation import (  # noqa: E402
    build_f_embedding_examples,
    build_f_lm_regression_records,
    build_g_sft_records,
    load_labeled_trees,
    split_labeled_trees,
    write_labeled_trees_jsonl,
)
from treepo._research.tree.labeled import LabeledTree  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_STUDENT_MODEL = "/mnt/data/models/Qwen/Qwen3.5-4B"
DEFAULT_FG_GRID_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_teacher_fg_leaf_grid"
    / "economic_gemma4_aligned_l1_2_4_8_16"
)
DEFAULT_BENOIT_INIT_ARTIFACT = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_teacher_fg_leaf_grid"
    / "economic_gemma4_f_existing_summary_leaf1"
    / "leaf_001"
    / "labeled_trees.jsonl"
)
DEFAULT_FULL_DOC_INIT_ARTIFACT = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_teacher_fg_leaf_grid"
    / "economic_gemma4_f_only_leaf1"
    / "leaf_001"
    / "labeled_trees.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "manifesto_fg_real_training_grid"

STAGES = ("fg", "fgf", "fgfg")
INIT_MODES = ("fresh", "benoit", "full_doc")
G_BACKENDS = ("teacher", "trl_lm", "dspy_lm")
F_BACKENDS = ("teacher", "trl_lm", "dspy_lm", "embedding_ridge", "embedding_linear_sgd")
EMBEDDING_F_BACKENDS = ("embedding_ridge", "embedding_linear_sgd")
BACKEND_MATRICES = ("curated", "full", "smoke")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def _parse_csv(value: str, *, allowed: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    items = tuple(
        dict.fromkeys(
            part.strip()
            for part in str(value or "").replace(";", ",").split(",")
            if part.strip()
        )
    )
    allowed_set = set(allowed or ())
    if allowed_set:
        unknown = sorted(set(items) - allowed_set)
        if unknown:
            raise ValueError(f"unknown value(s): {unknown}; allowed={sorted(allowed_set)}")
    return items


def _parse_leaf_grid(value: str) -> Tuple[int, ...]:
    leaves = tuple(int(item) for item in _parse_csv(value))
    if not leaves or any(leaf <= 0 for leaf in leaves):
        raise ValueError(f"leaf grid must contain positive integers: {value!r}")
    return leaves


def _parse_embedding_methods(value: str) -> Tuple[str, ...]:
    aliases = {
        "ridge": "embedding_ridge",
        "embedding_ridge": "embedding_ridge",
        "linear": "embedding_linear_sgd",
        "linear_sgd": "embedding_linear_sgd",
        "embedding_linear_sgd": "embedding_linear_sgd",
    }
    methods: List[str] = []
    for item in _parse_csv(value):
        key = str(item).strip().lower().replace("-", "_")
        if key not in aliases:
            raise ValueError(
                f"unknown embedding method {item!r}; allowed={sorted(set(aliases))}"
            )
        methods.append(aliases[key])
    return tuple(dict.fromkeys(methods))


def _embedding_backend_to_method(backend: str) -> str:
    if backend == "embedding_ridge":
        return "ridge"
    if backend == "embedding_linear_sgd":
        return "linear_sgd"
    raise ValueError(f"not an embedding f backend: {backend!r}")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    denom = denom_x * denom_y
    if denom <= 0.0:
        return None
    return float(sum(x * y for x, y in zip(dx, dy)) / denom)


def _rankdata(values: Sequence[float]) -> List[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(ordered):
        end = pos + 1
        while end < len(ordered) and ordered[end][1] == ordered[pos][1]:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[ordered[idx][0]] = float(rank)
        pos = end
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _regression_metrics(rows: Sequence[Mapping[str, Any]], *, pred_key: str, truth_key: str) -> Dict[str, Any]:
    preds: List[float] = []
    truths: List[float] = []
    for row in rows:
        pred = _safe_float(row.get(pred_key))
        truth = _safe_float(row.get(truth_key))
        if pred is None or truth is None:
            continue
        preds.append(float(pred))
        truths.append(float(truth))
    errors = [p - y for p, y in zip(preds, truths)]
    abs_errors = [abs(value) for value in errors]
    sq_errors = [value * value for value in errors]
    mse = _mean(sq_errors)
    return {
        "n": int(len(preds)),
        "pearson_r": _pearson(preds, truths),
        "spearman_r": _spearman(preds, truths),
        "mae": _mean(abs_errors),
        "mse": mse,
        "rmse": math.sqrt(mse) if mse is not None else None,
        "mean_prediction": _mean(preds),
        "mean_truth": _mean(truths),
    }


def _parse_first_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return _safe_float(value)
    text = str(value or "")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return None
    return _safe_float(match.group(0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _denormalize(value: float, *, target_min: float, target_max: float) -> float:
    return float(target_min) + _clamp01(float(value)) * (float(target_max) - float(target_min))


def _root_node_ids(trees: Sequence[LabeledTree]) -> Dict[str, str]:
    roots: Dict[str, str] = {}
    for tree in trees:
        root_id = ""
        for level in reversed(tree.levels or []):
            if level:
                root_id = str(level[0])
                break
        if root_id:
            roots[str(tree.doc_id)] = root_id
    return roots


def _tree_lookup(trees: Sequence[LabeledTree]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    roots = _root_node_ids(trees)
    for tree in trees:
        tmeta = dict(tree.metadata or {})
        root_id = roots.get(str(tree.doc_id), "")
        for node in tree.nodes.values():
            nmeta = dict(node.metadata or {})
            lookup[(str(tree.doc_id), str(node.node_id))] = {
                "doc_id": str(tree.doc_id),
                "node_id": str(node.node_id),
                "split": str(tmeta.get("split") or nmeta.get("split") or ""),
                "level": int(node.level),
                "is_root": str(node.node_id) == str(root_id),
                "teacher_score_1_7": _safe_float(node.score),
                "expert_score_1_7": _safe_float(tmeta.get("expert_score_1_7")),
                "existing_teacher_score_1_7": _safe_float(
                    tmeta.get("teacher_score_1_7_existing_root", tmeta.get("teacher_score_1_7"))
                ),
                "has_teacher_summary": bool(
                    str(
                        nmeta.get("teacher_summary")
                        or nmeta.get("target_summary")
                        or nmeta.get("summary")
                        or ""
                    ).strip()
                ),
            }
    return lookup


def _select_trees(trees: Sequence[LabeledTree], splits: Sequence[str]) -> List[LabeledTree]:
    split_set = {str(split).lower() for split in splits}
    return [
        tree
        for tree in trees
        if str((tree.metadata or {}).get("split") or "").lower() in split_set
    ]


def _root_summary_for_tree(tree: LabeledTree) -> str:
    for level in reversed(tree.levels or []):
        if not level:
            continue
        node = tree.get_node(str(level[0]))
        if node is None:
            continue
        meta = dict(node.metadata or {})
        for key in ("teacher_summary", "target_summary", "generated_summary", "summary"):
            value = str(meta.get(key) or "").strip()
            if value:
                return value
        if str(node.text or "").strip():
            return str(node.text)
    return ""


def _summary_coverage(trees: Sequence[LabeledTree]) -> Dict[str, Any]:
    total_nodes = 0
    summarized_nodes = 0
    tree_counts: Dict[str, int] = {}
    for tree in trees:
        split = str((tree.metadata or {}).get("split") or "unknown")
        tree_counts[split] = int(tree_counts.get(split, 0) + 1)
        for node in tree.nodes.values():
            total_nodes += 1
            nmeta = dict(node.metadata or {})
            if str(nmeta.get("teacher_summary") or nmeta.get("target_summary") or "").strip():
                summarized_nodes += 1
    return {
        "tree_counts": tree_counts,
        "node_count": int(total_nodes),
        "nodes_with_summary_target": int(summarized_nodes),
        "summary_target_coverage": (
            float(summarized_nodes / total_nodes) if total_nodes else None
        ),
    }


def _scalar_model_predictions(
    *,
    model_path: Path,
    records_path: Path,
    output_path: Path,
    labeled_trees_path: Path,
    max_length: int,
    target_min: float,
    target_max: float,
) -> Dict[str, Any]:
    """Run f-LM scalar predictions for exported f records."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from treepo._research.training.trl_training import _concat_prompt_response  # type: ignore

    model_path = Path(model_path)
    records = _read_jsonl(records_path)
    trees = load_labeled_trees(labeled_trees_path)
    node_lookup = _tree_lookup(trees)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        _write_json(output_path.with_suffix(".summary.json"), {"status": "skipped", "reason": "no_records"})
        return {"status": "skipped", "reason": "no_records"}

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        num_labels=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    span = float(target_max) - float(target_min)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            metadata = dict(record.get("metadata") or {})
            doc_id = str(metadata.get("doc_id") or "")
            node_id = str(metadata.get("node_id") or "")
            text = _concat_prompt_response(
                str(record.get("prompt") or ""),
                str(record.get("response") or ""),
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pred_norm = float(output.logits.detach().view(-1)[0].cpu())
            pred_norm = max(0.0, min(1.0, pred_norm))
            pred_raw = float(target_min + pred_norm * span)
            info = node_lookup.get((doc_id, node_id), {})
            rows.append(
                {
                    "doc_id": doc_id,
                    "node_id": node_id,
                    "split": str(metadata.get("split") or info.get("split") or ""),
                    "level": metadata.get("level"),
                    "law_role": metadata.get("law_role"),
                    "is_root": bool(info.get("is_root")),
                    "prediction_normalized": pred_norm,
                    "prediction_1_7": pred_raw,
                    "target_normalized": _safe_float(record.get("score")),
                    "target_1_7": _safe_float(metadata.get("target_score_raw")),
                    "teacher_score_1_7": info.get("teacher_score_1_7"),
                    "expert_score_1_7": info.get("expert_score_1_7"),
                    "existing_teacher_score_1_7": info.get("existing_teacher_score_1_7"),
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    root_rows = [row for row in rows if row.get("is_root")]
    summary = {
        "status": "completed",
        "model_path": str(model_path),
        "records_path": str(records_path),
        "predictions_path": str(output_path),
        "rows": int(len(rows)),
        "root_rows": int(len(root_rows)),
        "node_vs_teacher": _regression_metrics(
            rows,
            pred_key="prediction_1_7",
            truth_key="target_1_7",
        ),
        "root_vs_teacher": _regression_metrics(
            root_rows,
            pred_key="prediction_1_7",
            truth_key="target_1_7",
        ),
        "root_vs_expert": _regression_metrics(
            root_rows,
            pred_key="prediction_1_7",
            truth_key="expert_score_1_7",
        ),
        "root_vs_existing_teacher": _regression_metrics(
            root_rows,
            pred_key="prediction_1_7",
            truth_key="existing_teacher_score_1_7",
        ),
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


@dataclass(frozen=True)
class RowSpec:
    init_mode: str
    leaf_count: Optional[int]
    stage: str
    g_backend: str
    f_backend: str
    labeled_trees: Path

    @property
    def key(self) -> str:
        leaf = "init" if self.leaf_count is None else f"leaf_{int(self.leaf_count):03d}"
        return (
            f"{self.init_mode}_{leaf}_{self.stage}"
            f"_g-{self.g_backend}_f-{self.f_backend}"
        )


def _run_command(command: Sequence[str], *, cwd: Path, log_path: Path, dry_run: bool) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": list(command),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
        return {**payload, "returncode": 0, "status": "dry_run"}
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        **payload,
        "returncode": int(completed.returncode),
        "status": "completed" if completed.returncode == 0 else "failed",
    }


def _base_distill_command(
    *,
    args: argparse.Namespace,
    labeled_trees: Path,
    output_dir: Path,
) -> List[str]:
    command = [
        str(args.python_bin),
        "scripts/distill_ctreepo_students.py",
        "--labeled-tree-artifacts",
        str(labeled_trees),
        "--output-dir",
        str(output_dir),
        "--target-min",
        str(float(args.target_min)),
        "--target-max",
        str(float(args.target_max)),
        "--trl-epochs",
        str(int(args.epochs)),
        "--trl-batch-size",
        str(int(args.batch_size)),
        "--trl-grad-accumulation-steps",
        str(int(args.grad_accumulation_steps)),
        "--trl-learning-rate",
        str(float(args.learning_rate)),
        "--trl-max-length",
        str(int(args.max_length)),
    ]
    if bool(args.no_lora):
        command.append("--no-lora")
    if bool(args.no_4bit):
        command.append("--no-4bit")
    if bool(args.no_bf16):
        command.append("--no-bf16")
    if bool(args.include_identity_targets):
        command.append("--include-identity-targets")
    if args.embedding_url:
        command.extend(["--embedding-url", str(args.embedding_url)])
    if args.embedding_model:
        command.extend(["--embedding-model", str(args.embedding_model)])
    command.extend(
        [
            "--embedding-api-key",
            str(args.embedding_api_key),
            "--embedding-timeout-seconds",
            str(float(args.embedding_timeout_seconds)),
            "--embedding-batch-size",
            str(int(args.embedding_batch_size)),
        ]
    )
    return command


def _trl_g_command(
    *,
    args: argparse.Namespace,
    labeled_trees: Path,
    output_dir: Path,
    g_model_name: str,
) -> List[str]:
    command = _base_distill_command(args=args, labeled_trees=labeled_trees, output_dir=output_dir)
    command.extend(
        [
            "--run-g-sft",
            "--g-model-name",
            str(g_model_name),
            "--skip-f-fit",
            "--no-export-f-lm-records",
        ]
    )
    return command


def _trl_f_command(
    *,
    args: argparse.Namespace,
    labeled_trees: Path,
    output_dir: Path,
    f_lm_model_name: str,
) -> List[str]:
    command = _base_distill_command(args=args, labeled_trees=labeled_trees, output_dir=output_dir)
    command.extend(
        [
            "--skip-g-export",
            "--run-f-lm-regression",
            "--f-lm-model-name",
            str(f_lm_model_name),
            "--skip-f-fit",
        ]
    )
    return command


def _embedding_f_command(
    *,
    args: argparse.Namespace,
    labeled_trees: Path,
    output_dir: Path,
    f_backend: str,
) -> List[str]:
    command = _base_distill_command(args=args, labeled_trees=labeled_trees, output_dir=output_dir)
    command.extend(
        [
            "--skip-g-export",
            "--no-export-f-lm-records",
            "--f-method",
            _embedding_backend_to_method(f_backend),
        ]
    )
    return command


def _dspy_model_name(args: argparse.Namespace) -> str:
    if args.dspy_model:
        model = str(args.dspy_model)
    else:
        model = ""
        try:
            import requests

            response = requests.get(
                f"{str(args.dspy_api_base).rstrip('/')}/models",
                headers={"Authorization": f"Bearer {args.dspy_api_key}"},
                timeout=float(args.dspy_timeout_seconds),
            )
            response.raise_for_status()
            rows = response.json().get("data", [])
            if rows:
                model = str(rows[0].get("id") or "")
        except Exception:
            model = ""
        if not model:
            model = str(args.student_model)
    return model if model.startswith("openai/") else f"openai/{model}"


def _configure_dspy_lm(args: argparse.Namespace) -> Any:
    import dspy
    from treepo._research.config.dspy_config import configure_dspy

    lm = dspy.LM(
        model=_dspy_model_name(args),
        api_base=str(args.dspy_api_base).rstrip("/"),
        api_key=str(args.dspy_api_key),
        temperature=float(args.dspy_temperature),
        max_tokens=int(args.dspy_max_tokens),
        cache=not bool(args.disable_dspy_cache),
        timeout=float(args.dspy_timeout_seconds),
        num_retries=int(args.dspy_num_retries),
    )
    configure_dspy(lm=lm)
    return lm


def _compile_with_dspy_optimizer(
    *,
    args: argparse.Namespace,
    program: Any,
    metric: Any,
    trainset: Sequence[Any],
    valset: Sequence[Any],
) -> Any:
    import dspy

    optimizer_name = str(args.dspy_optimizer).strip().lower()
    budget = str(args.dspy_budget).strip().lower()
    if optimizer_name == "gepa":
        optimizer = dspy.GEPA(
            metric=metric,
            reflection_lm=dspy.settings.lm,
            auto=budget,
            num_threads=int(args.dspy_num_threads),
            use_wandb=False,
            use_mlflow=False,
        )
    elif optimizer_name == "mipro":
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=budget,
            num_threads=int(args.dspy_num_threads),
        )
    elif optimizer_name in {"bootstrap_random_search", "bootstrap-random-search", "random_search"}:
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch

        optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=int(args.dspy_max_bootstrapped_demos),
            max_labeled_demos=int(args.dspy_max_labeled_demos),
            num_candidate_programs=int(args.dspy_num_candidate_programs),
            num_threads=int(args.dspy_num_threads),
        )
    elif optimizer_name in {"bootstrap", "bootstrap_fewshot"}:
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=int(args.dspy_max_bootstrapped_demos),
            max_labeled_demos=int(args.dspy_max_labeled_demos),
            max_rounds=int(args.dspy_max_rounds),
        )
    else:
        raise ValueError(f"unknown DSPy optimizer: {args.dspy_optimizer!r}")

    try:
        return optimizer.compile(program, trainset=list(trainset), valset=list(valset))
    except TypeError:
        return optimizer.compile(program, trainset=list(trainset))


def _dspy_program_path(output_dir: Path, name: str) -> Path:
    return output_dir / name / "program.json"


def _dspy_train_g(
    *,
    args: argparse.Namespace,
    labeled_trees: Path,
    output_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    trees = load_labeled_trees(labeled_trees)
    train_trees, val_trees = split_labeled_trees(trees, train_splits=("train",), val_splits=("val",))
    test_trees = _select_trees(trees, ("test",))
    train_records = build_g_sft_records(train_trees, include_identity_targets=bool(args.include_identity_targets))
    val_records = build_g_sft_records(val_trees, include_identity_targets=bool(args.include_identity_targets))
    test_records = build_g_sft_records(test_trees, include_identity_targets=bool(args.include_identity_targets))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "g_dspy_train.jsonl", train_records)
    _write_jsonl(output_dir / "g_dspy_val.jsonl", val_records)
    _write_jsonl(output_dir / "g_dspy_test.jsonl", test_records)
    program_path = _dspy_program_path(output_dir, "g_dspy")
    if dry_run:
        _write_json(program_path, {"status": "dry_run", "backend": "dspy_lm", "target": "g"})
        state_path = output_dir / "g_dspy_states" / "labeled_trees.jsonl"
        write_labeled_trees_jsonl(state_path, trees)
        return {
            "status": "dry_run",
            "artifact": str(program_path),
            "artifact_type": "dspy_program",
            "state_artifact": str(state_path),
            "state_artifact_type": "labeled_tree_g_states",
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
        }

    import dspy

    class CTreePOGSignature(dspy.Signature):
        """Generate a score-preserving C-TreePO node summary."""

        prompt: str = dspy.InputField()
        completion: str = dspy.OutputField(desc="score-preserving summary")

    trainset = [
        dspy.Example(prompt=str(row.get("prompt") or ""), completion=str(row.get("completion") or "")).with_inputs("prompt")
        for row in train_records
    ]
    valset = [
        dspy.Example(prompt=str(row.get("prompt") or ""), completion=str(row.get("completion") or "")).with_inputs("prompt")
        for row in val_records
    ]
    if not trainset:
        raise ValueError("DSPy g training requested but no g records are available")

    def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
        from difflib import SequenceMatcher

        expected = str(getattr(gold, "completion", "") or "")
        actual = str(getattr(pred, "completion", "") or "")
        if not expected or not actual:
            return 0.0
        return float(SequenceMatcher(None, expected, actual).ratio())

    lm = _configure_dspy_lm(args)
    with dspy.context(lm=lm):
        compiled = _compile_with_dspy_optimizer(
            args=args,
            program=dspy.Predict(CTreePOGSignature),
            metric=metric,
            trainset=trainset,
            valset=valset,
        )
    program_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(program_path))
    state_result = _materialize_dspy_g_states(
        args=args,
        program=compiled,
        labeled_trees=labeled_trees,
        output_path=output_dir / "g_dspy_states" / "labeled_trees.jsonl",
    )
    return {
        "status": "completed",
        "artifact": str(program_path),
        "artifact_type": "dspy_program",
        "state_artifact": state_result.get("artifact"),
        "state_artifact_type": state_result.get("artifact_type"),
        "state_materialization": state_result,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
        "dspy_optimizer": str(args.dspy_optimizer),
        "dspy_budget": str(args.dspy_budget),
    }


def _dspy_train_f(
    *,
    args: argparse.Namespace,
    labeled_trees: Path,
    output_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    trees = load_labeled_trees(labeled_trees)
    train_trees, val_trees = split_labeled_trees(trees, train_splits=("train",), val_splits=("val",))
    test_trees = _select_trees(trees, ("test",))
    train_records = build_f_lm_regression_records(
        train_trees,
        include_identity_targets=bool(args.include_identity_targets),
        target_min=float(args.target_min),
        target_max=float(args.target_max),
    )
    val_records = build_f_lm_regression_records(
        val_trees,
        include_identity_targets=bool(args.include_identity_targets),
        target_min=float(args.target_min),
        target_max=float(args.target_max),
    )
    test_records = build_f_lm_regression_records(
        test_trees,
        include_identity_targets=bool(args.include_identity_targets),
        target_min=float(args.target_min),
        target_max=float(args.target_max),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = _write_jsonl(output_dir / "f_dspy_train.jsonl", train_records)
    val_path = _write_jsonl(output_dir / "f_dspy_val.jsonl", val_records)
    test_path = _write_jsonl(output_dir / "f_dspy_test.jsonl", test_records)
    program_path = _dspy_program_path(output_dir, "f_dspy")
    if dry_run:
        _write_json(program_path, {"status": "dry_run", "backend": "dspy_lm", "target": "f"})
        return {
            "status": "dry_run",
            "artifact": str(program_path),
            "artifact_type": "dspy_program",
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
            "test_path": str(test_path),
        }

    import dspy

    class CTreePOFSignature(dspy.Signature):
        """Predict the normalized scalar score for a C-TreePO node summary."""

        prompt: str = dspy.InputField()
        response: str = dspy.InputField()
        score: str = dspy.OutputField(desc="normalized scalar score in [0, 1]")

    trainset = [
        dspy.Example(
            prompt=str(row.get("prompt") or ""),
            response=str(row.get("response") or ""),
            score=str(float(row.get("score", 0.5))),
        ).with_inputs("prompt", "response")
        for row in train_records
    ]
    valset = [
        dspy.Example(
            prompt=str(row.get("prompt") or ""),
            response=str(row.get("response") or ""),
            score=str(float(row.get("score", 0.5))),
        ).with_inputs("prompt", "response")
        for row in val_records
    ]
    if not trainset:
        raise ValueError("DSPy f training requested but no f records are available")

    def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
        target = _parse_first_float(getattr(gold, "score", None))
        value = _parse_first_float(getattr(pred, "score", None))
        if target is None or value is None:
            return 0.0
        return max(0.0, 1.0 - abs(_clamp01(value) - _clamp01(target)))

    lm = _configure_dspy_lm(args)
    with dspy.context(lm=lm):
        compiled = _compile_with_dspy_optimizer(
            args=args,
            program=dspy.Predict(CTreePOFSignature),
            metric=metric,
            trainset=trainset,
            valset=valset,
        )
    program_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(program_path))
    f_eval = _dspy_f_predictions(
        args=args,
        program=compiled,
        records_path=test_path,
        output_path=output_dir / "f_dspy_test_predictions.jsonl",
        labeled_trees_path=labeled_trees,
    )
    return {
        "status": "completed",
        "artifact": str(program_path),
        "artifact_type": "dspy_program",
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "evaluation": f_eval,
        "dspy_optimizer": str(args.dspy_optimizer),
        "dspy_budget": str(args.dspy_budget),
    }


def _set_node_generated_summary(node: Any, summary: str, *, backend: str) -> None:
    metadata = dict(node.metadata or {})
    original = str(metadata.get("teacher_summary") or metadata.get("target_summary") or "")
    if original and "teacher_summary_original" not in metadata:
        metadata["teacher_summary_original"] = original
    metadata["generated_summary"] = str(summary)
    metadata["teacher_summary"] = str(summary)
    metadata["target_summary"] = str(summary)
    metadata["g_backend"] = str(backend)
    node.metadata = metadata


def _materialize_dspy_g_states(
    *,
    args: argparse.Namespace,
    program: Any,
    labeled_trees: Path,
    output_path: Path,
) -> Dict[str, Any]:
    trees = [copy.deepcopy(tree) for tree in load_labeled_trees(labeled_trees)]
    updates = 0
    for tree in trees:
        records = build_g_sft_records([tree], include_identity_targets=bool(args.include_identity_targets))
        prompts = {
            str((row.get("metadata") or {}).get("node_id") or ""): str(row.get("prompt") or "")
            for row in records
            if str((row.get("metadata") or {}).get("node_id") or "")
        }
        for node in tree.nodes.values():
            prompt = prompts.get(str(node.node_id))
            if not prompt:
                continue
            pred = program(prompt=prompt)
            summary = str(getattr(pred, "completion", "") or "").strip()
            if not summary:
                continue
            _set_node_generated_summary(node, summary, backend="dspy_lm")
            updates += 1
    write_labeled_trees_jsonl(output_path, trees)
    return {
        "status": "completed",
        "artifact": str(output_path),
        "artifact_type": "labeled_tree_g_states",
        "backend": "dspy_lm",
        "updated_nodes": int(updates),
    }


def _materialize_trl_g_states(
    *,
    args: argparse.Namespace,
    model_path: Path,
    labeled_trees: Path,
    output_path: Path,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trees = [copy.deepcopy(tree) for tree in load_labeled_trees(labeled_trees)]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    updates = 0
    with torch.no_grad():
        for tree in trees:
            records = build_g_sft_records([tree], include_identity_targets=bool(args.include_identity_targets))
            prompts = {
                str((row.get("metadata") or {}).get("node_id") or ""): str(row.get("prompt") or "")
                for row in records
                if str((row.get("metadata") or {}).get("node_id") or "")
            }
            for node in tree.nodes.values():
                prompt = prompts.get(str(node.node_id))
                if not prompt:
                    continue
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_length))
                encoded = {key: value.to(device) for key, value in encoded.items()}
                generated = model.generate(
                    **encoded,
                    max_new_tokens=int(args.g_generate_max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                prompt_len = int(encoded["input_ids"].shape[-1])
                text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
                if not text:
                    continue
                _set_node_generated_summary(node, text, backend="trl_lm")
                updates += 1
    write_labeled_trees_jsonl(output_path, trees)
    return {
        "status": "completed",
        "artifact": str(output_path),
        "artifact_type": "labeled_tree_g_states",
        "backend": "trl_lm",
        "updated_nodes": int(updates),
    }


def _dspy_f_predictions(
    *,
    args: argparse.Namespace,
    program: Any,
    records_path: Path,
    output_path: Path,
    labeled_trees_path: Path,
) -> Dict[str, Any]:
    records = _read_jsonl(records_path)
    trees = load_labeled_trees(labeled_trees_path)
    node_lookup = _tree_lookup(trees)
    rows: List[Dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        metadata = dict(record.get("metadata") or {})
        doc_id = str(metadata.get("doc_id") or "")
        node_id = str(metadata.get("node_id") or "")
        pred = program(prompt=str(record.get("prompt") or ""), response=str(record.get("response") or ""))
        pred_norm = _parse_first_float(getattr(pred, "score", None))
        if pred_norm is None:
            continue
        pred_norm = _clamp01(pred_norm)
        info = node_lookup.get((doc_id, node_id), {})
        rows.append(
            {
                "doc_id": doc_id,
                "node_id": node_id,
                "split": str(metadata.get("split") or info.get("split") or ""),
                "level": metadata.get("level"),
                "law_role": metadata.get("law_role"),
                "is_root": bool(info.get("is_root")),
                "prediction_normalized": pred_norm,
                "prediction_1_7": _denormalize(
                    pred_norm,
                    target_min=float(args.target_min),
                    target_max=float(args.target_max),
                ),
                "target_normalized": _safe_float(record.get("score")),
                "target_1_7": _safe_float(metadata.get("target_score_raw")),
                "teacher_score_1_7": info.get("teacher_score_1_7"),
                "expert_score_1_7": info.get("expert_score_1_7"),
                "existing_teacher_score_1_7": info.get("existing_teacher_score_1_7"),
            }
        )
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    root_rows = [row for row in rows if row.get("is_root")]
    summary = {
        "status": "completed",
        "predictions_path": str(output_path),
        "rows": int(len(rows)),
        "root_rows": int(len(root_rows)),
        "node_vs_teacher": _regression_metrics(rows, pred_key="prediction_1_7", truth_key="target_1_7"),
        "root_vs_teacher": _regression_metrics(root_rows, pred_key="prediction_1_7", truth_key="target_1_7"),
        "root_vs_expert": _regression_metrics(root_rows, pred_key="prediction_1_7", truth_key="expert_score_1_7"),
        "root_vs_existing_teacher": _regression_metrics(
            root_rows,
            pred_key="prediction_1_7",
            truth_key="existing_teacher_score_1_7",
        ),
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _embedding_model_predictions(
    *,
    args: argparse.Namespace,
    model_path: Path,
    labeled_trees_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    from treepo._research.config.settings import get_embedding_model, get_embedding_url, load_settings
    from treepo._research.training.embedding_proxy import VLLMEmbeddingClient, load_embedding_proxy_model

    settings = load_settings()
    embedding_url = (args.embedding_url or get_embedding_url(settings) or "").rstrip("/")
    embedding_model = args.embedding_model or get_embedding_model(settings) or None
    client = VLLMEmbeddingClient(
        api_base=embedding_url,
        model=embedding_model,
        api_key=str(args.embedding_api_key),
        timeout_seconds=float(args.embedding_timeout_seconds),
        batch_size=int(args.embedding_batch_size),
    )
    model = load_embedding_proxy_model(model_path)
    trees = load_labeled_trees(labeled_trees_path)
    node_lookup = _tree_lookup(trees)
    rows: List[Dict[str, Any]] = []
    for tree in trees:
        if str((tree.metadata or {}).get("split") or "").lower() != "test":
            continue
        for node in tree.nodes.values():
            text = ""
            meta = dict(node.metadata or {})
            for key in ("teacher_summary", "target_summary", "generated_summary", "summary"):
                text = str(meta.get(key) or "").strip()
                if text:
                    break
            if not text:
                continue
            pred_norm = float(model.predict_from_embedding(client.embed_texts([text])[0]))
            info = node_lookup.get((str(tree.doc_id), str(node.node_id)), {})
            rows.append(
                {
                    "doc_id": str(tree.doc_id),
                    "node_id": str(node.node_id),
                    "split": "test",
                    "level": int(node.level),
                    "is_root": bool(info.get("is_root")),
                    "prediction_normalized": pred_norm,
                    "prediction_1_7": _denormalize(
                        pred_norm,
                        target_min=float(args.target_min),
                        target_max=float(args.target_max),
                    ),
                    "target_1_7": float(node.score),
                    "teacher_score_1_7": info.get("teacher_score_1_7"),
                    "expert_score_1_7": info.get("expert_score_1_7"),
                    "existing_teacher_score_1_7": info.get("existing_teacher_score_1_7"),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    root_rows = [row for row in rows if row.get("is_root")]
    summary = {
        "status": "completed",
        "model_path": str(model_path),
        "predictions_path": str(output_path),
        "rows": int(len(rows)),
        "root_rows": int(len(root_rows)),
        "node_vs_teacher": _regression_metrics(rows, pred_key="prediction_1_7", truth_key="target_1_7"),
        "root_vs_teacher": _regression_metrics(root_rows, pred_key="prediction_1_7", truth_key="target_1_7"),
        "root_vs_expert": _regression_metrics(root_rows, pred_key="prediction_1_7", truth_key="expert_score_1_7"),
        "root_vs_existing_teacher": _regression_metrics(
            root_rows,
            pred_key="prediction_1_7",
            truth_key="existing_teacher_score_1_7",
        ),
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _load_distillation_manifest(path: Path) -> Dict[str, Any]:
    manifest_path = Path(path) / "distillation_manifest.json"
    if not manifest_path.exists():
        return {"status": "missing", "path": str(manifest_path)}
    return _read_json(manifest_path)


def _fg_teacher_metrics(labeled_trees_path: Path) -> Dict[str, Any]:
    trees = load_labeled_trees(labeled_trees_path)
    lookup = _tree_lookup(trees)
    rows = list(lookup.values())
    root_rows = [row for row in rows if row.get("is_root")]
    return {
        **_summary_coverage(trees),
        "root_vs_expert": _regression_metrics(
            root_rows,
            pred_key="teacher_score_1_7",
            truth_key="expert_score_1_7",
        ),
        "root_vs_existing_teacher": _regression_metrics(
            root_rows,
            pred_key="teacher_score_1_7",
            truth_key="existing_teacher_score_1_7",
        ),
    }


def _row_manifest_payload(
    *,
    row: RowSpec,
    row_dir: Path,
    command_result: Optional[Mapping[str, Any]],
    distill_manifest: Optional[Mapping[str, Any]],
    f_eval: Optional[Mapping[str, Any]],
    teacher_metrics: Optional[Mapping[str, Any]],
    args: argparse.Namespace,
    g_model_name: Optional[str],
    f_lm_model_name: Optional[str],
    step_results: Optional[Sequence[Mapping[str, Any]]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_artifacts = dict(artifacts or {})
    return {
        "created_at": _now_iso(),
        "row_key": row.key,
        "dimension": str(args.dimension),
        "init_mode": row.init_mode,
        "leaf_count": row.leaf_count,
        "stage": row.stage,
        "g_backend": row.g_backend,
        "f_backend": row.f_backend,
        "labeled_trees": str(row.labeled_trees),
        "output_dir": str(row_dir),
        "backend": {
            "g_backend": row.g_backend,
            "f_backend": row.f_backend,
            "backend_matrix": str(args.backend_matrix),
        },
        "artifacts": normalized_artifacts,
        "student_models": {
            "base_student_model": str(args.student_model),
            "g_model_name": str(g_model_name) if g_model_name else None,
            "f_lm_model_name": str(f_lm_model_name) if f_lm_model_name else None,
            "dspy_model": str(args.dspy_model or ""),
        },
        "command_result": dict(command_result or {}),
        "step_results": [dict(item) for item in (step_results or [])],
        "distillation_manifest": dict(distill_manifest or {}),
        "teacher_metrics": dict(teacher_metrics or {}),
        "f_lm_evaluation": dict(f_eval or {}),
        "f_evaluation": dict(f_eval or {}),
    }


def _resolve_init_models(
    *,
    init_mode: str,
    init_manifests: Mapping[str, Mapping[str, Any]],
    student_model: str,
) -> Tuple[str, str]:
    if init_mode == "benoit":
        manifest = dict(init_manifests.get("benoit") or {})
        artifacts = dict(manifest.get("artifacts") or {})
        return (
            str(artifacts.get("g_model") or student_model),
            str(artifacts.get("f_lm_model") or student_model),
        )
    if init_mode == "full_doc":
        manifest = dict(init_manifests.get("full_doc") or {})
        artifacts = dict(manifest.get("artifacts") or {})
        return (str(student_model), str(artifacts.get("f_lm_model") or student_model))
    return (str(student_model), str(student_model))


def _init_command(
    *,
    args: argparse.Namespace,
    mode: str,
    artifact: Path,
    output_dir: Path,
) -> List[str]:
    command = [
        str(args.python_bin),
        "scripts/distill_ctreepo_students.py",
        "--labeled-tree-artifacts",
        str(artifact),
        "--output-dir",
        str(output_dir),
        "--target-min",
        str(float(args.target_min)),
        "--target-max",
        str(float(args.target_max)),
        "--trl-epochs",
        str(int(args.init_epochs)),
        "--trl-batch-size",
        str(int(args.batch_size)),
        "--trl-grad-accumulation-steps",
        str(int(args.grad_accumulation_steps)),
        "--trl-learning-rate",
        str(float(args.learning_rate)),
        "--trl-max-length",
        str(int(args.max_length)),
        "--skip-f-fit",
        "--run-f-lm-regression",
        "--f-lm-model-name",
        str(args.student_model),
    ]
    if mode == "benoit":
        command.extend(["--run-g-sft", "--g-model-name", str(args.student_model)])
    else:
        command.append("--skip-g-export")
    if bool(args.no_lora):
        command.append("--no-lora")
    if bool(args.no_4bit):
        command.append("--no-4bit")
    if bool(args.no_bf16):
        command.append("--no-bf16")
    if bool(args.include_identity_targets):
        command.append("--include-identity-targets")
    return command


def _matrix_pairs_for_stage(args: argparse.Namespace, stage: str) -> Tuple[Tuple[str, str], ...]:
    allowed_g = set(_parse_csv(args.g_backends, allowed=G_BACKENDS))
    allowed_f = set(_parse_csv(args.f_backends, allowed=F_BACKENDS))
    embedding_methods = _parse_embedding_methods(args.embedding_methods)
    matrix = str(args.backend_matrix).strip().lower()
    if matrix == "smoke":
        embedding_methods = tuple(method for method in embedding_methods if method == "embedding_ridge") or ("embedding_ridge",)

    pairs: List[Tuple[str, str]] = []
    if stage == "fg":
        pairs = [("teacher", "teacher")]
    elif matrix == "full":
        if stage == "fgf":
            pairs = [("teacher", f_backend) for f_backend in F_BACKENDS if f_backend != "teacher"]
        elif stage == "fgfg":
            pairs = [
                (g_backend, f_backend)
                for g_backend in G_BACKENDS
                if g_backend != "teacher"
                for f_backend in F_BACKENDS
                if f_backend != "teacher"
            ]
    else:
        if stage == "fgf":
            pairs = [
                ("teacher", "trl_lm"),
                ("teacher", "dspy_lm"),
                *[("teacher", method) for method in embedding_methods],
            ]
        elif stage == "fgfg":
            pairs = [
                ("trl_lm", "trl_lm"),
                ("dspy_lm", "dspy_lm"),
                *[("trl_lm", method) for method in embedding_methods],
                *[("dspy_lm", method) for method in embedding_methods],
            ]
    filtered = [
        (g_backend, f_backend)
        for g_backend, f_backend in pairs
        if g_backend in allowed_g and f_backend in allowed_f
    ]
    return tuple(dict.fromkeys(filtered))


def _requested_rows(args: argparse.Namespace) -> List[RowSpec]:
    init_modes = _parse_csv(args.init_modes, allowed=INIT_MODES)
    stages = _parse_csv(args.stages, allowed=STAGES)
    leaves = _parse_leaf_grid(args.leaf_grid)
    if bool(args.smoke):
        init_modes = ("fresh",)
        stages = tuple(stage for stage in stages if stage in {"fg", "fgf", "fgfg"})
        leaves = tuple(leaf for leaf in (1, 8) if leaf in set(leaves))
        if not leaves:
            leaves = (1, 8)
        if str(args.backend_matrix) == "curated":
            args.backend_matrix = "smoke"

    rows: List[RowSpec] = []
    for init_mode in init_modes:
        for leaf_count in leaves:
            labeled_trees = Path(args.fg_grid_dir) / f"leaf_{int(leaf_count):03d}" / "labeled_trees.jsonl"
            for stage in stages:
                for g_backend, f_backend in _matrix_pairs_for_stage(args, str(stage)):
                    rows.append(
                        RowSpec(
                            init_mode=str(init_mode),
                            leaf_count=int(leaf_count),
                            stage=str(stage),
                            g_backend=str(g_backend),
                            f_backend=str(f_backend),
                            labeled_trees=labeled_trees,
                        )
                    )
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real f/g LM distillation for the manifesto economic pilot grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dimension", default="economic")
    parser.add_argument("--fg-grid-dir", type=Path, default=DEFAULT_FG_GRID_DIR)
    parser.add_argument("--benoit-init-artifact", type=Path, default=DEFAULT_BENOIT_INIT_ARTIFACT)
    parser.add_argument("--full-doc-init-artifact", type=Path, default=DEFAULT_FULL_DOC_INIT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--python-bin", default=str(PROJECT_ROOT / "venv" / "bin" / "python"))
    parser.add_argument("--leaf-grid", default="1,2,4,8,16")
    parser.add_argument("--init-modes", default="fresh,benoit,full_doc")
    parser.add_argument("--stages", default="fg,fgf,fgfg")
    parser.add_argument("--g-backends", default="teacher,trl_lm,dspy_lm")
    parser.add_argument("--f-backends", default="teacher,trl_lm,dspy_lm,embedding_ridge,embedding_linear_sgd")
    parser.add_argument("--backend-matrix", choices=BACKEND_MATRICES, default="curated")
    parser.add_argument("--embedding-methods", default="embedding_ridge,embedding_linear_sgd")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-init-training", action="store_true")
    parser.add_argument("--evaluate-f", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-min", type=float, default=1.0)
    parser.add_argument("--target-max", type=float, default=7.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--init-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--g-generate-max-new-tokens", type=int, default=256)
    parser.add_argument("--dspy-api-base", default="http://localhost:8010/v1")
    parser.add_argument("--dspy-api-key", default="EMPTY")
    parser.add_argument("--dspy-model", default=None)
    parser.add_argument("--dspy-optimizer", choices=["bootstrap_random_search", "bootstrap", "gepa", "mipro"], default="bootstrap_random_search")
    parser.add_argument("--dspy-budget", default="light")
    parser.add_argument("--dspy-temperature", type=float, default=0.0)
    parser.add_argument("--dspy-max-tokens", type=int, default=512)
    parser.add_argument("--dspy-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--dspy-num-retries", type=int, default=1)
    parser.add_argument("--dspy-num-threads", type=int, default=128)
    parser.add_argument("--dspy-max-bootstrapped-demos", type=int, default=8)
    parser.add_argument("--dspy-max-labeled-demos", type=int, default=8)
    parser.add_argument("--dspy-num-candidate-programs", type=int, default=6)
    parser.add_argument("--dspy-max-rounds", type=int, default=1)
    parser.add_argument("--disable-dspy-cache", action="store_true")
    parser.add_argument("--embedding-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-api-key", default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--include-identity-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if bool(args.verbose) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR / f"{args.dimension}_{_now_stamp()}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _requested_rows(args)
    init_modes = sorted({row.init_mode for row in rows})
    init_manifests: Dict[str, Mapping[str, Any]] = {}

    if not bool(args.skip_init_training):
        for mode, artifact in (
            ("benoit", Path(args.benoit_init_artifact)),
            ("full_doc", Path(args.full_doc_init_artifact)),
        ):
            if mode not in init_modes:
                continue
            init_dir = output_dir / "initializers" / mode
            manifest_path = init_dir / "distillation_manifest.json"
            if bool(args.skip_existing) and manifest_path.exists():
                init_manifests[mode] = _read_json(manifest_path)
                continue
            command = _init_command(args=args, mode=mode, artifact=artifact, output_dir=init_dir)
            result = _run_command(
                command,
                cwd=PROJECT_ROOT,
                log_path=init_dir / "distill.log",
                dry_run=bool(args.dry_run),
            )
            if result.get("returncode") != 0:
                _write_json(init_dir / "initializer_manifest.json", {"command_result": result})
                raise SystemExit(f"initializer {mode} failed; see {init_dir / 'distill.log'}")
            init_manifest = _load_distillation_manifest(init_dir)
            if bool(args.dry_run):
                init_manifest = {
                    "status": "dry_run",
                    "artifacts": {
                        "g_model": str(args.student_model) if mode == "benoit" else None,
                        "f_lm_model": str(args.student_model),
                    },
                }
            init_manifests[mode] = init_manifest
            _write_json(init_dir / "initializer_manifest.json", {"command_result": result, "distillation_manifest": init_manifest})

    row_manifests: List[Dict[str, Any]] = []
    failures = 0
    for row in rows:
        row_dir = output_dir / "rows" / row.key
        row_manifest_path = row_dir / "row_manifest.json"
        if bool(args.skip_existing) and row_manifest_path.exists():
            row_manifests.append(_read_json(row_manifest_path))
            continue
        if not row.labeled_trees.exists():
            failures += 1
            payload = _row_manifest_payload(
                row=row,
                row_dir=row_dir,
                command_result={"status": "missing_labeled_trees"},
                distill_manifest=None,
                f_eval=None,
                teacher_metrics=None,
                args=args,
                g_model_name=None,
                f_lm_model_name=None,
                artifacts={"g_backend": row.g_backend, "f_backend": row.f_backend},
            )
            row_manifests.append(payload)
            _write_json(row_manifest_path, payload)
            continue

        teacher_metrics = _fg_teacher_metrics(row.labeled_trees)
        artifacts: Dict[str, Any] = {
            "g_backend": row.g_backend,
            "f_backend": row.f_backend,
            "g_artifact": None,
            "f_artifact": None,
            "g_state_artifact": str(row.labeled_trees),
            "g_artifact_type": "teacher_labeled_tree" if row.g_backend == "teacher" else None,
            "f_artifact_type": "teacher_scores" if row.f_backend == "teacher" else None,
        }
        if row.stage == "fg":
            payload = _row_manifest_payload(
                row=row,
                row_dir=row_dir,
                command_result={"status": "teacher_only"},
                distill_manifest=None,
                f_eval=None,
                teacher_metrics=teacher_metrics,
                args=args,
                g_model_name=None,
                f_lm_model_name=None,
                artifacts=artifacts,
            )
            row_manifests.append(payload)
            _write_json(row_manifest_path, payload)
            continue

        g_model_name, f_lm_model_name = _resolve_init_models(
            init_mode=row.init_mode,
            init_manifests=init_manifests,
            student_model=str(args.student_model),
        )
        step_results: List[Dict[str, Any]] = []
        distill_manifest: Dict[str, Any] = {}
        f_eval: Optional[Dict[str, Any]] = None
        state_trees = row.labeled_trees

        if row.g_backend == "trl_lm":
            g_dir = row_dir / "g_trl"
            command = _trl_g_command(
                args=args,
                labeled_trees=row.labeled_trees,
                output_dir=g_dir,
                g_model_name=g_model_name,
            )
            result = _run_command(
                command,
                cwd=PROJECT_ROOT,
                log_path=g_dir / "distill.log",
                dry_run=bool(args.dry_run),
            )
            step_results.append({"step": "train_g", "backend": "trl_lm", **result})
            if result.get("returncode") != 0:
                failures += 1
            g_manifest = _load_distillation_manifest(g_dir)
            if bool(args.dry_run):
                g_manifest = {
                    "status": "dry_run",
                    "artifacts": {
                        "g_model": str(g_model_name),
                        "g_test": None,
                    },
                }
            g_artifact = (g_manifest.get("artifacts") or {}).get("g_model")
            artifacts["g_artifact"] = g_artifact
            artifacts["g_artifact_type"] = "hf_trl_lm"
            state_trees = row_dir / "g_trl_states" / "labeled_trees.jsonl"
            artifacts["g_state_artifact"] = str(state_trees)
            if bool(args.dry_run):
                write_labeled_trees_jsonl(state_trees, load_labeled_trees(row.labeled_trees))
            if not bool(args.dry_run) and g_artifact and Path(str(g_artifact)).exists():
                state_result = _materialize_trl_g_states(
                    args=args,
                    model_path=Path(str(g_artifact)),
                    labeled_trees=row.labeled_trees,
                    output_path=state_trees,
                )
                step_results.append({"step": "materialize_g_states", **state_result})
            distill_manifest["g"] = g_manifest
        elif row.g_backend == "dspy_lm":
            g_result = _dspy_train_g(
                args=args,
                labeled_trees=row.labeled_trees,
                output_dir=row_dir / "g_dspy",
                dry_run=bool(args.dry_run),
            )
            step_results.append({"step": "train_g", "backend": "dspy_lm", **g_result})
            artifacts["g_artifact"] = g_result.get("artifact")
            artifacts["g_artifact_type"] = g_result.get("artifact_type")
            artifacts["g_state_artifact"] = g_result.get("state_artifact")
            state_trees = Path(str(g_result.get("state_artifact") or row.labeled_trees))
            distill_manifest["g"] = g_result

        f_input_trees = state_trees if Path(state_trees).exists() else row.labeled_trees
        if row.f_backend == "trl_lm":
            f_dir = row_dir / "f_trl"
            command = _trl_f_command(
                args=args,
                labeled_trees=Path(f_input_trees),
                output_dir=f_dir,
                f_lm_model_name=f_lm_model_name,
            )
            result = _run_command(
                command,
                cwd=PROJECT_ROOT,
                log_path=f_dir / "distill.log",
                dry_run=bool(args.dry_run),
            )
            step_results.append({"step": "train_f", "backend": "trl_lm", **result})
            if result.get("returncode") != 0:
                failures += 1
            f_manifest = _load_distillation_manifest(f_dir)
            if bool(args.dry_run):
                f_manifest = {
                    "status": "dry_run",
                    "artifacts": {
                        "f_lm_model": str(f_lm_model_name),
                        "f_lm_test": None,
                    },
                }
            f_artifacts = dict(f_manifest.get("artifacts") or {})
            artifacts["f_artifact"] = f_artifacts.get("f_lm_model")
            artifacts["f_artifact_type"] = "hf_trl_lm"
            distill_manifest["f"] = f_manifest
            f_model_path = f_artifacts.get("f_lm_model")
            f_test_path = f_artifacts.get("f_lm_test") or (
                (f_manifest.get("f_lm_target") or {}).get("lm_regression_test_path")
                if isinstance(f_manifest.get("f_lm_target"), Mapping)
                else None
            )
            if (
                bool(args.evaluate_f)
                and not bool(args.dry_run)
                and f_model_path
                and f_test_path
                and Path(str(f_model_path)).exists()
                and Path(str(f_test_path)).exists()
            ):
                f_eval = _scalar_model_predictions(
                    model_path=Path(str(f_model_path)),
                    records_path=Path(str(f_test_path)),
                    output_path=row_dir / "f_trl_test_predictions.jsonl",
                    labeled_trees_path=Path(f_input_trees),
                    max_length=int(args.max_length),
                    target_min=float(args.target_min),
                    target_max=float(args.target_max),
                )
        elif row.f_backend == "dspy_lm":
            f_result = _dspy_train_f(
                args=args,
                labeled_trees=Path(f_input_trees),
                output_dir=row_dir / "f_dspy",
                dry_run=bool(args.dry_run),
            )
            step_results.append({"step": "train_f", "backend": "dspy_lm", **f_result})
            artifacts["f_artifact"] = f_result.get("artifact")
            artifacts["f_artifact_type"] = f_result.get("artifact_type")
            distill_manifest["f"] = f_result
            f_eval = dict(f_result.get("evaluation") or {}) if f_result.get("evaluation") else None
        elif row.f_backend in EMBEDDING_F_BACKENDS:
            f_dir = row_dir / "f_embedding"
            command = _embedding_f_command(
                args=args,
                labeled_trees=Path(f_input_trees),
                output_dir=f_dir,
                f_backend=row.f_backend,
            )
            result = _run_command(
                command,
                cwd=PROJECT_ROOT,
                log_path=f_dir / "distill.log",
                dry_run=bool(args.dry_run),
            )
            step_results.append({"step": "train_f", "backend": row.f_backend, **result})
            if result.get("returncode") != 0:
                failures += 1
            f_manifest = _load_distillation_manifest(f_dir)
            if bool(args.dry_run):
                f_manifest = {
                    "status": "dry_run",
                    "artifacts": {
                        "f_embedding_proxy": str(f_dir / "f_embedding_proxy.json"),
                    },
                }
            f_artifacts = dict(f_manifest.get("artifacts") or {})
            artifacts["f_artifact"] = f_artifacts.get("f_embedding_proxy")
            artifacts["f_artifact_type"] = "embedding_proxy_json"
            distill_manifest["f"] = f_manifest
            if (
                bool(args.evaluate_f)
                and not bool(args.dry_run)
                and artifacts.get("f_artifact")
                and Path(str(artifacts["f_artifact"])).exists()
            ):
                f_eval = _embedding_model_predictions(
                    args=args,
                    model_path=Path(str(artifacts["f_artifact"])),
                    labeled_trees_path=Path(f_input_trees),
                    output_path=row_dir / "f_embedding_test_predictions.jsonl",
                )

        if bool(args.evaluate_f) and f_eval is None:
            f_eval = {
                "status": "skipped",
                "reason": "dry_run_or_missing_model_or_records",
                "f_backend": row.f_backend,
                "f_artifact": artifacts.get("f_artifact"),
            }

        failed_steps = [step for step in step_results if step.get("returncode", 0) not in (0, None)]
        if bool(args.dry_run) and step_results:
            row_status = "dry_run"
        elif failed_steps:
            row_status = "failed"
        elif step_results:
            row_status = "completed"
        else:
            row_status = "teacher_only"
        command_result = {
            "status": row_status,
            "step_count": int(len(step_results)),
            "commands": [
                step.get("command")
                for step in step_results
                if isinstance(step.get("command"), list)
            ],
        }

        payload = _row_manifest_payload(
            row=row,
            row_dir=row_dir,
            command_result=command_result,
            distill_manifest=distill_manifest,
            f_eval=f_eval,
            teacher_metrics=teacher_metrics,
            args=args,
            g_model_name=g_model_name,
            f_lm_model_name=f_lm_model_name,
            step_results=step_results,
            artifacts=artifacts,
        )
        row_manifests.append(payload)
        _write_json(row_manifest_path, payload)

    summary = {
        "created_at": _now_iso(),
        "status": "completed" if failures == 0 else "completed_with_failures",
        "dimension": str(args.dimension),
        "output_dir": str(output_dir),
        "config": {
            "fg_grid_dir": str(args.fg_grid_dir),
            "benoit_init_artifact": str(args.benoit_init_artifact),
            "full_doc_init_artifact": str(args.full_doc_init_artifact),
            "student_model": str(args.student_model),
            "leaf_grid": sorted({int(row.leaf_count) for row in rows if row.leaf_count is not None}),
            "init_modes": sorted({row.init_mode for row in rows}),
            "stages": sorted({row.stage for row in rows}),
            "g_backends": sorted({row.g_backend for row in rows}),
            "f_backends": sorted({row.f_backend for row in rows}),
            "backend_matrix": str(args.backend_matrix),
            "embedding_methods": list(_parse_embedding_methods(args.embedding_methods)),
            "dspy_optimizer": str(args.dspy_optimizer),
            "dspy_budget": str(args.dspy_budget),
            "smoke": bool(args.smoke),
            "dry_run": bool(args.dry_run),
            "target_min": float(args.target_min),
            "target_max": float(args.target_max),
        },
        "rows_total": int(len(rows)),
        "row_manifests": [str(output_dir / "rows" / row.key / "row_manifest.json") for row in rows],
        "failures": int(failures),
        "initializers": init_manifests,
    }
    _write_json(output_dir / "grid_manifest.json", summary)
    print(json.dumps({"manifest": str(output_dir / "grid_manifest.json"), "failures": failures}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
