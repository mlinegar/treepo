#!/usr/bin/env python3
"""
Manifesto RILE Oracle Training.

Train a RILE oracle classifier from manifesto results or raw samples.
This script is task-specific and lives under src/tasks/manifesto.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

from treepo._research.config.dspy_config import configure_dspy
from treepo._research.config.logging import setup_logging, get_logger
from treepo._research.config.settings import load_settings
from treepo._research.tasks.manifesto import (
    ManifestoDataset,
    ManifestoPipeline,
    create_training_examples,
    rile_metric,
)
from treepo._research.tasks.manifesto.constants import RILE_MIN, RILE_MAX

logger = get_logger(__name__)


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


def train_rile_oracle(args) -> None:
    """Train RILE oracle classifier from manifesto results."""
    print_banner("RILE ORACLE CLASSIFIER TRAINING", {
        "Results Dir:": str(args.results_dir) if args.results_dir else "Process new",
        "Samples:": str(args.samples),
        "Port:": str(args.port),
        "Bin Size:": str(args.bin_size),
        "Output Directory:": str(args.output_dir),
    })

    # Configure DSPy
    settings = load_settings(args.config)
    gen_cfg = settings.get("generation", {})
    summarizer_cfg = gen_cfg.get("summarizer", {})

    lm = dspy.LM(
        "openai/default",
        api_base=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
        temperature=summarizer_cfg.get("temperature", 0.3),
        max_tokens=summarizer_cfg.get("max_tokens", 8192),
    )
    configure_dspy(lm=lm)
    logger.info(f"DSPy configured with vLLM on port {args.port}")

    # Get training data
    if args.results_dir and args.results_dir.exists():
        # Load existing results
        logger.info(f"Loading results from {args.results_dir}...")
        result_files = list(args.results_dir.glob("**/results.json"))
        if not result_files:
            raise FileNotFoundError(f"No results files found in {args.results_dir}")

        with open(result_files[0]) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} results")

        # Create training examples from results
        training_examples = []
        for r in results:
            if r.get('estimated_score') is not None:
                raw_score = r.get('reference_score', 0.0)
                normalized_score = (raw_score - RILE_MIN) / (RILE_MAX - RILE_MIN)
                normalized_score = max(0.0, min(1.0, normalized_score))
                training_examples.append(dspy.Example(
                    text=r.get('text', ''),  # Use full text - truncation corrupts training
                    score=normalized_score,
                ).with_inputs('text'))
    else:
        # Process new manifestos
        logger.info("Processing new manifestos...")
        dataset = ManifestoDataset(
            countries=[51, 41],
            min_year=1990,
            require_text=True,
        )

        sample_ids = dataset.get_all_ids()[:args.samples]
        samples = [dataset.get_sample(sid) for sid in sample_ids if dataset.get_sample(sid)]

        training_examples = create_training_examples(samples)
        logger.info(f"Created {len(training_examples)} training examples")

    if len(training_examples) < 4:
        raise ValueError(f"Need at least 4 training examples, got {len(training_examples)}")

    # Limit training examples
    if args.max_examples and len(training_examples) > args.max_examples:
        training_examples = training_examples[:args.max_examples]

    logger.info(f"Using {len(training_examples)} training examples")

    # Create and train pipeline
    pipeline = ManifestoPipeline(chunk_size=2000)

    optimizer = BootstrapFewShot(
        metric=rile_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    logger.info("Starting BootstrapFewShot optimization...")
    trained_pipeline = optimizer.compile(pipeline, trainset=training_examples)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.output_dir / f"rile_oracle_{timestamp}.json"
    trained_pipeline.save(str(model_path))

    stats = {
        "created_at": datetime.now().isoformat(),
        "type": "rile-oracle",
        "model_path": str(model_path),
        "num_examples": len(training_examples),
        "bin_size": args.bin_size,
        "config": {
            "port": args.port,
            "samples": args.samples,
            "max_examples": args.max_examples,
        },
    }

    stats_path = args.output_dir / f"rile_oracle_{timestamp}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model saved to:    {model_path}")
    print(f"  Stats saved to:    {stats_path}")
    print("=" * 70)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="RILE oracle training for manifesto task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--bin-size", type=float, default=10.0)
    parser.add_argument("--max-examples", type=int, default=50)

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    setup_logging(verbose=args.verbose)

    train_rile_oracle(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
