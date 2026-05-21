#!/usr/bin/env python3
"""
Unified Model Training for OPS Framework.

Consolidates training workflows into a single entry point:
- ops-comparison: Train OPS comparison module from supervision data

Usage Examples:
    # Train OPS comparison module from supervision data
    python -m src.training.train_model --type ops-comparison \
        --supervision-data data/preferences/supervision.json \
        --output-dir models/comparison

    # Train OPS comparison module from supervision data
    python -m src.training.train_model --type ops-comparison \
        --supervision-data data/preferences/supervision.json \
        --output-dir models/comparison

    # Task-specific oracle training lives under the task package, e.g.:
    # python -m src.tasks.<task>.train_oracle --output-dir models/oracle
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from treepo._research.config.logging import setup_logging, get_logger

logger = get_logger(__name__)


# =============================================================================
# Common Utilities
# =============================================================================

def preference_metric(example, prediction, trace=None) -> float:
    """Metric for binary projection accuracy."""
    predicted = str(getattr(prediction, "preferred", "")).upper().strip()
    actual = str(getattr(example, "preferred", "")).upper().strip()

    if predicted == actual:
        return 1.0
    if predicted == "TIE" or actual == "TIE":
        return 0.5
    return 0.0


def print_banner(title: str, config: dict) -> None:
    """Print configuration banner."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    for key, value in config.items():
        print(f"  {key:20s} {value}")
    print("=" * 70)
    print()


# =============================================================================
# Type: OPS Comparison Training
# =============================================================================

def train_ops_comparison(args) -> None:
    """Train OPS comparison module from supervision judgments."""
    import dspy
    from treepo._research.config.dspy_config import configure_dspy
    from treepo._research.config.settings import load_settings
    from treepo._research.training.comparison import OPSComparisonModule
    from treepo._research.training.supervision import SupervisionDataset

    if not args.supervision_data or not args.supervision_data.exists():
        raise ValueError(f"Supervision data required: {args.supervision_data}")

    print_banner("OPS COMPARISON MODULE TRAINING", {
        "Supervision Data:": str(args.supervision_data),
        "Law Type:": args.law_type,
        "Model:": args.model,
        "Port:": str(args.port),
        "Budget:": args.budget,
        "Output Directory:": str(args.output_dir),
    })

    # Load and filter dataset
    logger.info("Loading supervision dataset...")
    supervision = SupervisionDataset.load(args.supervision_data)
    dataset = supervision.project_binary(projection="adjacent")

    if args.law_type != "all":
        pairs = [p for p in dataset.pairs if p.law_type == args.law_type]
        dataset = dataset.__class__(pairs)

    if args.max_pairs:
        dataset = dataset.__class__(dataset.pairs[:args.max_pairs])

    if len(dataset) == 0:
        raise ValueError("No binary supervision pairs available after filtering")

    weighted_dataset = dataset
    if hasattr(dataset, "resample_by_propensity"):
        weighted_dataset = dataset.resample_by_propensity(
            target_size=len(dataset),
            seed=args.seed,
        )

    logger.info(f"Training on {len(dataset)} projected pairs (law_type={args.law_type})")

    # Configure LM
    settings = load_settings(args.config)
    gen_cfg = settings.get("generation", {})
    judge_cfg = gen_cfg.get("comparison_judge", {})

    temperature = args.temperature or judge_cfg.get("temperature", 0.3)
    max_tokens = args.max_tokens or judge_cfg.get("max_tokens", 2048)

    lm = dspy.LM(
        model=args.model,
        api_base=f"http://localhost:{args.port}/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    configure_dspy(lm=lm)

    # Split and convert to examples
    train_set, val_set = weighted_dataset.split(train_ratio=args.train_ratio, shuffle=True)
    train_examples = train_set.to_dspy_examples()
    val_examples = val_set.to_dspy_examples()

    logger.info(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    # Create and optimize module
    comparison_module = OPSComparisonModule(use_cot=not args.no_cot)
    optimizer = dspy.GEPA(
        metric=preference_metric,
        auto=args.budget,
        num_threads=args.num_threads,
        use_wandb=False,
        use_mlflow=False,
    )

    compile_kwargs = {
        "student": comparison_module,
        "trainset": train_examples,
    }
    if val_examples:
        compile_kwargs["valset"] = val_examples

    logger.info("Starting GEPA optimization...")
    trained_module = optimizer.compile(**compile_kwargs)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.output_dir / f"ops_comparison_{timestamp}.json"
    trained_module.save(str(model_path))

    stats = {
        "created_at": datetime.now().isoformat(),
        "type": "ops-comparison",
        "model_path": str(model_path),
        "num_supervision_judgments": len(supervision),
        "num_pairs": len(dataset),
        "effective_pairs_after_resample": len(weighted_dataset),
        "law_type": args.law_type,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "config": {
            "model": args.model,
            "port": args.port,
            "budget": args.budget,
            "num_threads": args.num_threads,
            "use_cot": not args.no_cot,
            "train_ratio": args.train_ratio,
        },
    }

    stats_path = args.output_dir / f"ops_comparison_{timestamp}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model saved to:    {model_path}")
    print(f"  Stats saved to:    {stats_path}")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified model training for OPS framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Type selection
    parser.add_argument(
        "--type", type=str, required=True,
        choices=["ops-comparison"],
        help="Training type"
    )

    # Common options
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--model", type=str, default="openai/qwen-30b-thinking")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    # OPS comparison options
    parser.add_argument(
        "--supervision-data",
        "--preference-data",
        dest="supervision_data",
        type=Path,
        default=None,
        help="Path to supervision JSON file",
    )
    parser.add_argument("--law-type", type=str, default="all",
                       choices=["all", "sufficiency", "idempotence", "merge"])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--budget", type=str, default="heavy",
                       choices=["light", "medium", "heavy", "superheavy"])
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    setup_logging(verbose=args.verbose)

    if args.type == "ops-comparison":
        train_ops_comparison(args)
    else:
        logger.error(f"Unknown type: {args.type}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
