#!/usr/bin/env python3
"""
Unified Supervision Collection for OPS Training.

Consolidates three collection workflows into a single, task-agnostic script:
1. GenRM judge on direct documents
2. GenRM judge on labeled trees
3. Oracle scorer on direct documents

Usage Examples:
    # GenRM on direct documents
    python -m src.training.collect_preferences \\
        --task document_analysis \\
        --judge-type genrm \\
        --source-type direct \\
        --output-dir data/preferences

    # Oracle on direct documents
    python -m src.training.collect_preferences \\
        --task document_analysis \\
        --judge-type oracle \\
        --source-type direct \\
        --output-dir data/preferences

    # GenRM on labeled trees
    python -m src.training.collect_preferences \\
        --task document_analysis \\
        --judge-type genrm \\
        --source-type labeled \\
        --labels-dir data/labels \\
        --law-type merge \\
        --output-dir data/preferences

    # Full options
    python -m src.training.collect_preferences \\
        --task document_analysis \\
        --judge-type genrm \\
        --source-type direct \\
        --law-type sufficiency \\
        --summarizer-port 8000 \\
        --judge-port 8001 \\
        --k-candidates 4 \\
        --temperatures 0.3 0.5 0.7 0.9 \\
        --max-documents 100 \\
        --output-dir data/preferences \\
        --seed 42 \\
        --verbose
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from treepo._research.config.logging import setup_logging, get_logger
from treepo._research.config.settings import DEFAULT_TASK

logger = get_logger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="Unified supervision collection for OPS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GenRM on direct documents
  python -m src.training.collect_preferences --task document_analysis --judge-type genrm

  # Oracle on direct documents
  python -m src.training.collect_preferences --task document_analysis --judge-type oracle

  # GenRM on labeled trees
  python -m src.training.collect_preferences --source-type labeled --labels-dir data/labels
        """,
    )

    # === Task Configuration ===
    task_group = parser.add_argument_group("Task Configuration")
    task_group.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help=f"Task name from TaskRegistry (default: {DEFAULT_TASK})",
    )
    task_group.add_argument(
        "--law-type",
        type=str,
        default="sufficiency",
        choices=["sufficiency", "idempotence", "merge", "all"],
        help="OPS law type (default: sufficiency)",
    )

    # === Judge Configuration ===
    judge_group = parser.add_argument_group("Judge Configuration")
    judge_group.add_argument(
        "--judge-type",
        type=str,
        default="genrm",
        choices=["genrm", "oracle", "dspy"],
        help="Type of supervision backend for comparative judgments (default: genrm)",
    )
    judge_group.add_argument(
        "--judge-port",
        type=int,
        default=None,
        help="Port for judge/oracle server (default: from config)",
    )
    judge_group.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for judge (default: auto-detect)",
    )
    judge_group.add_argument(
        "--tie-margin",
        type=float,
        default=None,
        help="Tie margin for oracle judge (default: 5.0)",
    )

    # === Data Source Configuration ===
    source_group = parser.add_argument_group("Data Source Configuration")
    source_group.add_argument(
        "--source-type",
        type=str,
        default="direct",
        choices=["direct", "labeled", "synthetic"],
        help="Type of data source (default: direct)",
    )
    source_group.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum documents to process (default: all)",
    )
    source_group.add_argument(
        "--train-only",
        action="store_true",
        help="Only use training split",
    )
    source_group.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory with labeled trees (for --source-type labeled)",
    )
    source_group.add_argument(
        "--max-trees",
        type=int,
        default=None,
        help="Maximum trees to process (for labeled source)",
    )
    source_group.add_argument(
        "--max-nodes-per-tree",
        type=int,
        default=None,
        help="Maximum nodes per tree (for labeled source)",
    )
    source_group.add_argument(
        "--synthetic-data",
        type=Path,
        default=None,
        help="Path to synthetic data file (for --source-type synthetic)",
    )

    # === Generation Configuration ===
    gen_group = parser.add_argument_group("Generation Configuration")
    gen_group.add_argument(
        "--summarizer-port",
        type=int,
        default=None,
        help="Port for summarizer model (default: from config)",
    )
    gen_group.add_argument(
        "--summarizer-model",
        type=str,
        default=None,
        help="Model name for summarizer (default: from config)",
    )
    gen_group.add_argument(
        "--k-candidates",
        "-k",
        type=int,
        default=None,
        help=(
            "Number of candidate summaries per input (default: 4). "
            "Listwise-capable judges consume these in one judgment; "
            "pairwise-only backends fall back internally."
        ),
    )
    gen_group.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Temperatures for diverse generation (default: from config)",
    )

    # === Output Configuration ===
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preferences"),
        help="Output directory (default: data/preferences)",
    )
    output_group.add_argument(
        "--no-dpo",
        action="store_true",
        help="Skip DPO format export",
    )
    output_group.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: auto-generated)",
    )

    # === Misc ===
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser


def print_banner(config) -> None:
    """Print configuration banner."""
    print()
    print("=" * 70)
    print("  UNIFIED SUPERVISION COLLECTION")
    print("=" * 70)
    print(f"  Task:             {config.task_name}")
    print(f"  Judge Type:       {config.judge.judge_type.value}")
    print(f"  Data Source:      {config.data_source.source_type.value}")
    print(f"  Law Type:         {config.law_type}")
    print(f"  K Candidates:     {config.generation.k_candidates}")
    print(f"  Temperatures:     {config.generation.temperatures}")
    print(f"  Summarizer:       localhost:{config.server.summarizer_port}")
    print(f"  Judge:            localhost:{config.server.judge_port}")
    print(f"  Output:           {config.output_dir}")
    print("=" * 70)
    print()


def print_summary(stats: dict, output_file: Path) -> None:
    """Print collection summary."""
    pairs = dict(stats.get("pairs", {}) or {})
    preferences = dict(stats.get("preferences", {}) or {})
    comparative = dict(stats.get("comparative", {}) or {})
    print()
    print("=" * 70)
    print("  COLLECTION COMPLETE")
    print("=" * 70)
    print(f"  Binary pairs:      {pairs.get('collected', 0)}")
    if comparative:
        print(f"  Comparative recs:  {comparative.get('records_collected', 0)}")
    print(f"  Prefer A:          {preferences.get('prefer_a', 0)}")
    print(f"  Prefer B:          {preferences.get('prefer_b', 0)}")
    print(f"  Ties:              {preferences.get('ties', 0)}")
    print(f"  Avg confidence:    {preferences.get('avg_confidence', 0):.2f}")
    print(f"  Output file:       {output_file}")
    print("=" * 70)


def create_data_source(config, task):
    """Create appropriate data source based on configuration."""
    from .data_sources import (
        DirectDocumentSource,
        LabeledTreeSource,
        SyntheticDataSource,
    )
    from .preference_config import DataSourceType

    ds = config.data_source

    if ds.source_type == DataSourceType.DIRECT:
        splits = ["train"] if ds.train_only else ["train", "val"]
        return DirectDocumentSource(
            task=task,
            max_documents=ds.max_documents,
            splits=splits,
        )

    elif ds.source_type == DataSourceType.LABELED:
        if ds.labels_dir is None:
            raise ValueError("--labels-dir required for labeled source")
        return LabeledTreeSource(
            labels_dir=ds.labels_dir,
            law_type=config.law_type,
            max_trees=ds.max_trees,
            max_nodes_per_tree=ds.max_nodes_per_tree,
        )

    elif ds.source_type == DataSourceType.SYNTHETIC:
        if ds.synthetic_data_path is None:
            raise ValueError("--synthetic-data required for synthetic source")
        rubric = task.create_rubric()
        return SyntheticDataSource(
            data_path=ds.synthetic_data_path,
            rubric=rubric,
        )

    else:
        raise ValueError(f"Unknown source type: {ds.source_type}")


def create_collector(config, task, summarizer, memory=None):
    """Create an appropriate supervision collector based on judge type."""
    from .preference_config import JudgeType
    from treepo._research.training.supervision import GenerationConfig, PreferenceCollector
    from treepo._research.training.judges import GenRMJudge, LargeJudgeListwiseModule

    gen = config.generation
    judge_settings = config.judge
    server = config.server

    # Build generation configs
    generation_configs = [
        GenerationConfig(temperature=temp, prompt_variant=f"temp_{temp}")
        for temp in gen.temperatures[: gen.k_candidates]
    ]

    if judge_settings.judge_type == JudgeType.GENRM:
        genrm_judge = GenRMJudge(
            base_url=server.judge_url,
            model_name=server.judge_model or "nvidia/Qwen3-Nemotron-235B-A22B-GenRM",
            temperature=judge_settings.judge_temperature,
            top_p=judge_settings.judge_top_p,
            max_tokens=judge_settings.judge_max_tokens,
        )
        return PreferenceCollector(
            summarizer=summarizer,
            strategy="genrm",
            genrm_judge=genrm_judge,
            k=gen.k_candidates,
            generation_configs=generation_configs,
            memory=memory,
        )

    elif judge_settings.judge_type == JudgeType.ORACLE:
        # Get oracle predictor from task
        oracle_predict = task.create_oracle_scorer()

        return PreferenceCollector(
            summarizer=summarizer,
            strategy="oracle",
            oracle_predict=oracle_predict,
            k=gen.k_candidates,
            generation_configs=generation_configs,
            tie_margin=judge_settings.tie_margin,
            memory=memory,
        )

    elif judge_settings.judge_type == JudgeType.DSPY:
        return PreferenceCollector(
            summarizer=summarizer,
            judge=LargeJudgeListwiseModule(use_cot=True),
            strategy="judge",
            k=gen.k_candidates,
            generation_configs=generation_configs,
            comparison_mode="listwise",
            memory=memory,
        )

    else:
        raise ValueError(f"Unknown judge type: {judge_settings.judge_type}")


def save_results(collector, config) -> Path:
    """Save the primary supervision artifact and any explicitly requested projections."""
    from treepo._research.training.supervision import save_supervision_artifact_bundle

    supervision_dataset = collector.get_supervision_dataset()
    stats = collector.get_statistics()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = config.output_prefix or f"{config.task_name}_{config.judge.judge_type.value}"
    supervision_file = config.output_dir / f"{prefix}_supervision_{timestamp}.json"
    dpo_file = config.output_dir / f"{prefix}_dpo_{timestamp}.json" if (
        config.save_dpo_format and config.law_type == "sufficiency"
    ) else None
    if config.save_dpo_format and config.law_type == "sufficiency":
        logger.info("Saving DPO projection for sufficiency supervision")

    # Save statistics
    stats["config"] = config.to_dict()
    stats_file = config.output_dir / f"{prefix}_stats_{timestamp}.json"
    save_supervision_artifact_bundle(
        supervision_dataset,
        supervision_path=supervision_file,
        dpo_path=dpo_file,
        stats_path=stats_file,
        stats=stats,
        law_type="sufficiency" if config.law_type == "sufficiency" else None,
    )
    logger.info("Saved primary supervision artifact to %s", supervision_file)
    logger.info("Saved statistics to %s", stats_file)

    return supervision_file


def main(args: Optional[argparse.Namespace] = None) -> int:
    """
    Main entry point for unified supervision collection.

    Args:
        args: Optional pre-parsed arguments. If None, parses from sys.argv.

    Returns:
        Exit code (0 for success)
    """
    # Parse arguments
    if args is None:
        parser = create_argument_parser()
        args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Import dependencies (delayed to speed up --help)
    import dspy
    from treepo._research.config.dspy_config import configure_dspy
    from treepo._research.config.settings import load_settings
    from treepo._research.tasks import get_task

    from .preference_config import PreferenceCollectionConfig

    # Load settings and build config
    settings = load_settings(args.config)
    config = PreferenceCollectionConfig.from_cli_and_settings(args, settings)

    # Validate config
    if config.data_source.source_type.value == "labeled":
        if config.data_source.labels_dir is None:
            logger.error("--labels-dir is required for labeled source")
            return 1

    if config.data_source.source_type.value == "synthetic":
        if config.data_source.synthetic_data_path is None:
            logger.error("--synthetic-data is required for synthetic source")
            return 1

    # Set random seed
    random.seed(config.seed)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Print banner
    print_banner(config)

    # Get task from registry
    logger.info(f"Loading task '{config.task_name}'...")
    task = get_task(config.task_name)
    rubric = task.create_rubric()

    # Configure summarizer LM
    logger.info(f"Configuring summarizer on port {config.server.summarizer_port}...")
    summarizer_lm = dspy.LM(
        model=config.server.summarizer_model,
        api_base=config.server.summarizer_url,
        api_key="not-needed",
        temperature=config.generation.summarizer_temperature,
        max_tokens=config.generation.summarizer_max_tokens,
    )
    configure_dspy(lm=summarizer_lm)

    # Create summarizer module from task
    summarizer = task.create_summarizer()

    # Create data source and collector
    logger.info("Creating data source...")
    data_source = create_data_source(config, task)

    logger.info("Creating supervision collector...")
    collector = create_collector(config, task, summarizer)

    # Collect supervision
    logger.info(f"Starting supervision collection from {data_source.source_name}...")
    print()

    example_count = 0
    for i, example in enumerate(data_source.get_examples()):
        logger.info(f"[{i+1}] Processing {example.example_id}...")

        try:
            if getattr(collector, "comparison_mode", "pairwise") == "listwise":
                record = collector.collect_comparative_for_example(
                    example_id=example.example_id,
                    original_text=example.text,
                    rubric=example.rubric,
                    reference_score=example.reference_score or 0.0,
                    law_type=config.law_type,
                )
                logger.info(
                    "  Generated 1 comparative record with %d candidates",
                    len(record.candidates),
                )
            else:
                pairs = collector.collect_pairs_for_example(
                    example_id=example.example_id,
                    original_text=example.text,
                    rubric=example.rubric,
                    reference_score=example.reference_score or 0.0,
                    law_type=config.law_type,
                )
                logger.info(f"  Generated {len(pairs)} binary projection pairs")
            example_count += 1

        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

        # Progress update every 10 examples
        if (i + 1) % 10 == 0:
            stats = collector.get_statistics()
            logger.info(
                "Progress: %s pairs, %s comparative records from %s examples",
                stats.get("pairs", {}).get("collected", 0),
                stats.get("comparative", {}).get("records_collected", 0),
                example_count,
            )

    # Save results
    logger.info("Saving results...")
    output_file = save_results(collector, config)

    # Print summary
    stats = collector.get_statistics()
    print_summary(stats, output_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
