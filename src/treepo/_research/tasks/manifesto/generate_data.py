#!/usr/bin/env python3
"""
Unified Data Generation for OPS Training.

Consolidates three data generation workflows into a single entry point:
- dpo: Generate DPO training data using a trained comparison module
- labeled: Generate oracle labeled trees for documents
- synthetic: Generate synthetic training data using a large oracle model

Usage Examples:
    # Generate DPO data
    python -m src.tasks.manifesto.generate_data --type dpo \
        --comparison-module models/comparison.json \
        --output-dir data/dpo

    # Generate labeled trees
    python -m src.tasks.manifesto.generate_data --type labeled \
        --oracle-port 8001 \
        --max-documents 10 \
        --output-dir data/labels

    # Generate synthetic data
    python -m src.tasks.manifesto.generate_data --type synthetic \
        --oracle-port 8001 \
        --output-dir data/synthetic
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from treepo._research.config.logging import setup_logging, get_logger
from treepo._research.tasks.manifesto import RILE_SCALE

logger = get_logger(__name__)


# =============================================================================
# Common Utilities
# =============================================================================

RILE_RUBRIC = """Preserve the political positioning (left-right stance) of the content.

Key information to preserve:
- Left-wing indicators: social welfare, equality, international cooperation, environmental protection
- Right-wing indicators: traditional values, free enterprise, national strength, law and order
- Overall political stance and intensity
- Key policy positions and their framing

The RILE score ranges from 0.0 (far left) to 1.0 (far right)."""


def normalize_score(value: Optional[float], scale) -> Optional[float]:
    """Normalize a raw score to 0-1."""
    if value is None:
        return None
    normalized = scale.normalize(float(value))
    return max(0.0, min(1.0, normalized))


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


def print_summary(title: str, stats: dict) -> None:
    """Print generation summary."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:20s} {value:.2f}")
        else:
            print(f"  {key:20s} {value}")
    print("=" * 70)


# =============================================================================
# Type: DPO Data Generation
# =============================================================================

def generate_dpo_data(args) -> None:
    """Generate DPO training data using a trained comparison module."""
    import dspy
    from treepo._research.config.dspy_config import configure_dspy
    from treepo._research.config.settings import load_settings
    from treepo._research.training.comparison import OPSComparisonModule
    from treepo._research.training.supervision import BinaryComparison, BinaryProjectionDataset
    from treepo._research.tasks.manifesto import LeafSummarizer, ManifestoDataset

    if not args.comparison_module or not args.comparison_module.exists():
        raise ValueError(f"Comparison module required for DPO generation: {args.comparison_module}")

    print_banner("DPO DATA GENERATION", {
        "Comparison Module:": str(args.comparison_module),
        "Summarizer Port:": args.summarizer_port,
        "Judge Port:": args.judge_port,
        "K Candidates:": args.k_candidates,
        "Law Type:": args.law_type,
        "Output Directory:": str(args.output_dir),
    })

    settings = load_settings(args.config)
    gen_cfg = settings.get("generation", {})
    summarizer_cfg = gen_cfg.get("summarizer", {})
    judge_cfg = gen_cfg.get("comparison_judge", {})

    temperatures = args.temperatures or summarizer_cfg.get("candidate_temperatures", [0.3, 0.5, 0.7, 0.9])

    # Configure LMs
    logger.info("Configuring LMs...")
    summarizer_lm = dspy.LM(
        model=args.summarizer_model,
        api_base=f"http://localhost:{args.summarizer_port}/v1",
        api_key="not-needed",
        temperature=summarizer_cfg.get("temperature", 0.5),
        max_tokens=summarizer_cfg.get("max_tokens", 2048),
    )
    judge_lm = dspy.LM(
        model=args.judge_model,
        api_base=f"http://localhost:{args.judge_port}/v1",
        api_key="not-needed",
        temperature=judge_cfg.get("temperature", 0.3),
        max_tokens=judge_cfg.get("max_tokens", 2048),
    )

    summarizer = LeafSummarizer(use_cot=True)
    judge = OPSComparisonModule(use_cot=True)
    judge.load(str(args.comparison_module))

    # Load data
    logger.info("Loading manifesto data...")
    loader = ManifestoDataset()
    train_samples, val_samples, _ = loader.get_temporal_split()
    samples = train_samples if args.train_only else train_samples + val_samples
    if args.max_documents:
        samples = samples[:args.max_documents]

    logger.info(f"Processing {len(samples)} documents")

    dataset = BinaryProjectionDataset()
    pair_counter = 0

    for i, sample in enumerate(samples):
        doc_id = sample.get("id", f"doc_{i}")
        doc_text = sample.get("text", "") or sample.get("content", "")
        reference_score = normalize_score(sample.get("rile", 0.0), RILE_SCALE)

        if not doc_text:
            continue

        logger.info(f"[{i+1}/{len(samples)}] Processing {doc_id}...")

        # Generate candidates
        configure_dspy(lm=summarizer_lm)
        candidates = []
        for temp in temperatures[:args.k_candidates]:
            try:
                summarizer_lm.kwargs["temperature"] = temp
                result = summarizer(content=doc_text, rubric=RILE_RUBRIC)  # Use full text
                candidates.append(getattr(result, "summary", str(result)))
            except Exception as e:
                logger.warning(f"Candidate generation failed: {e}")

        if len(candidates) < 2:
            continue

        # Compare pairs
        configure_dspy(lm=judge_lm)
        for a_idx in range(len(candidates)):
            for b_idx in range(a_idx + 1, len(candidates)):
                summary_a, summary_b = candidates[a_idx], candidates[b_idx]

                swapped = random.random() < 0.5
                if swapped:
                    summary_a, summary_b = summary_b, summary_a

                result = judge(
                    law_type=args.law_type,
                    rubric=RILE_RUBRIC,
                    original_text=doc_text,  # Use full text
                    summary_a=summary_a,
                    summary_b=summary_b,
                    reference_score=reference_score,
                )

                preferred = str(getattr(result, "preferred", "tie"))
                if swapped and preferred != "tie":
                    preferred = "B" if preferred == "A" else "A"

                pair_counter += 1
                from treepo._research.core.preference_supervision import preference_supervision_metadata

                dataset.add_comparison(BinaryComparison(
                    pair_id=f"dpo_{pair_counter:06d}",
                    source_example_id=doc_id,
                    original_text=doc_text,  # Use full text
                    rubric=RILE_RUBRIC,
                    reference_score=reference_score,
                    law_type=args.law_type,
                    preference_supervision=preference_supervision_metadata(
                        application_name="manifesto_generate_data",
                        law_type=args.law_type,
                    ),
                    summary_a=summary_a if not swapped else summary_b,
                    summary_b=summary_b if not swapped else summary_a,
                    preferred=preferred,
                    reasoning=str(getattr(result, "reasoning", "")),
                    confidence=float(getattr(result, "confidence", 0.5)),
                    judge_model=str(args.comparison_module),
                    generation_config_a={"temperature": temperatures[a_idx]},
                    generation_config_b={"temperature": temperatures[b_idx]},
                ))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset.save(args.output_dir / f"preferences_{timestamp}.json")

    dpo_data = dataset.to_dpo_format(law_type="sufficiency")
    with open(args.output_dir / f"dpo_data_{timestamp}.json", "w") as f:
        json.dump(dpo_data, f, indent=2)

    print_summary("DPO GENERATION COMPLETE", {
        "Total pairs:": len(dataset),
        "Documents processed:": len(samples),
    })


# =============================================================================
# Type: Labeled Tree Generation
# =============================================================================

def generate_labeled_trees(args) -> None:
    """Generate oracle labeled trees for documents."""
    import dspy
    from treepo._research.config.dspy_config import configure_dspy
    from treepo._research.config.settings import load_settings
    from treepo._research.tasks.manifesto import ManifestoDataset, create_rile_oracle
    from treepo._research.pipelines.batched import chunk_text
    from treepo._research.training.tree import (
        LabeledNode,
        LabeledTree,
        LabeledDataset,
    )

    print_banner("LABELED TREE GENERATION", {
        "Oracle Port:": args.oracle_port,
        "Oracle Model:": args.oracle_model,
        "Chunk Size:": args.chunk_size,
        "Output Directory:": str(args.output_dir),
    })

    settings = load_settings(args.config)
    gen_cfg = settings.get("generation", {})
    label_cfg = gen_cfg.get("labeled", {})

    # Configure oracle
    logger.info(f"Configuring oracle on port {args.oracle_port}...")
    oracle_lm = dspy.LM(
        model=args.oracle_model,
        api_base=f"http://localhost:{args.oracle_port}/v1",
        api_key="not-needed",
        temperature=label_cfg.get("temperature", 0.3),
        max_tokens=label_cfg.get("max_tokens", 2048),
    )
    configure_dspy(lm=oracle_lm)

    rile_oracle = create_rile_oracle()

    def oracle_predict(text: str) -> dict:
        try:
            result = rile_oracle.value_extractor(text)
            return {
                "score": float(result),
                "reasoning": getattr(result, 'reasoning', ""),
                "left_indicators": getattr(result, 'left_indicators', ""),
                "right_indicators": getattr(result, 'right_indicators', ""),
            }
        except Exception as e:
            return {"score": 0.0, "reasoning": f"Error: {e}", "left_indicators": "", "right_indicators": ""}

    # Load data
    logger.info("Loading manifesto data...")
    dataset = ManifestoDataset()
    train_ids, val_ids, test_ids = dataset.create_temporal_split()
    sample_ids = train_ids if args.train_only else train_ids + val_ids
    if args.max_documents:
        sample_ids = sample_ids[:args.max_documents]

    samples = list(dataset.get_split_samples(sample_ids))
    logger.info(f"Processing {len(samples)} documents")

    label_dataset = LabeledDataset()

    for i, sample in enumerate(samples):
        doc_id = sample.manifesto_id
        logger.info(f"[{i+1}/{len(samples)}] Processing {doc_id}...")

        if not sample.text:
            continue

        # Build tree structure
        chunks = chunk_text(sample.text, args.chunk_size)
        levels = [chunks]
        current = chunks
        while len(current) > 1:
            next_level = []
            for j in range(0, len(current), 2):
                if j + 1 < len(current):
                    next_level.append(f"{current[j]}\n\n{current[j+1]}")
                else:
                    next_level.append(current[j])
            levels.append(next_level)
            current = next_level

        # Create tree and score all nodes
        tree = LabeledTree(
            doc_id=doc_id,
            document_text=sample.text,
            document_score=normalize_score(sample.rile, RILE_SCALE),
            label_source=args.oracle_model,
        )

        for level_idx, level_texts in enumerate(levels):
            for node_idx, text in enumerate(level_texts):
                result = oracle_predict(text)
                left_child = right_child = None
                if level_idx > 0:
                    base = node_idx * 2
                    left_child = f"{doc_id}_L{level_idx-1}_N{base}"
                    if base + 1 < len(levels[level_idx - 1]):
                        right_child = f"{doc_id}_L{level_idx-1}_N{base+1}"

                tree.add_node(LabeledNode(
                    node_id=f"{doc_id}_L{level_idx}_N{node_idx}",
                    doc_id=doc_id,
                    level=level_idx,
                    text=text,
                    score=result["score"],
                    reasoning=result["reasoning"],
                    confidence=1.0,
                    left_child_id=left_child,
                    right_child_id=right_child,
                    metadata={
                        "left_indicators": result["left_indicators"],
                        "right_indicators": result["right_indicators"],
                    },
                ))

        label_dataset.add_tree(tree)
        tree.save(args.output_dir / f"{doc_id}_labels.json")

    label_dataset.save(args.output_dir)

    stats = label_dataset.get_statistics()
    print_summary("LABELED TREE GENERATION COMPLETE", {
        "Documents:": len(label_dataset),
        "Total chunks:": stats['total_chunks'],
        "Total leaves:": stats['total_leaves'],
        "Total merge nodes:": stats['total_merge_nodes'],
    })


# =============================================================================
# Type: Synthetic Data Generation
# =============================================================================

def generate_synthetic_data(args) -> None:
    """Generate synthetic training data using a large oracle model."""
    import dspy
    from treepo._research.config.dspy_config import configure_dspy
    from treepo._research.config.settings import load_settings
    from treepo._research.training.synthetic import (
        SyntheticDataGenerator,
        ChallengeGenerator,
        ReferenceGenerator,
        SyntheticDataset,
    )
    from treepo._research.tasks.manifesto import ManifestoDataset

    print_banner("SYNTHETIC DATA GENERATION", {
        "Oracle Model:": args.oracle_model,
        "Oracle Port:": args.oracle_port,
        "Reasoning Mode:": args.reasoning_mode,
        "Min Quality:": args.min_quality_score,
        "Output Directory:": str(args.output_dir),
    })

    settings = load_settings(args.config)
    gen_cfg = settings.get("generation", {})
    synth_cfg = gen_cfg.get("synthetic_data", {})

    temperature = args.temperature or synth_cfg.get("temperature", 0.6)
    top_p = args.top_p or synth_cfg.get("top_p", 0.95)

    # Configure oracle
    system_prompt = "detailed thinking on" if args.reasoning_mode == "on" else "detailed thinking off"
    oracle_lm = dspy.LM(
        model=f"openai/nvidia/{args.oracle_model}",
        api_base=f"http://localhost:{args.oracle_port}/v1",
        api_key="not-needed",
        temperature=temperature,
        top_p=top_p,
        max_tokens=synth_cfg.get("max_tokens", 4096),
        system_prompt=system_prompt,
    )
    configure_dspy(lm=oracle_lm)

    # Load data
    logger.info("Loading manifesto data...")
    loader = ManifestoDataset()
    train_samples, val_samples, _ = loader.get_temporal_split()
    samples = train_samples + val_samples
    if args.max_documents:
        samples = samples[:args.max_documents]

    logger.info(f"Processing {len(samples)} documents")

    # Create generators
    generator = SyntheticDataGenerator(
        challenge_generator=ChallengeGenerator(use_cot=True),
        reference_generator=ReferenceGenerator(use_cot=True),
        min_quality_score=args.min_quality_score,
        target_compression=args.target_compression,
        oracle_model_name=args.oracle_model,
    )

    successful = []
    failed = 0

    for i, sample in enumerate(samples):
        doc_text = sample.get('text', '') or sample.get('content', '')
        if not doc_text:
            failed += 1
            continue

        logger.info(f"[{i+1}/{len(samples)}] Processing document...")

        try:
            example = generator.generate_example(
                document=doc_text,
                rubric=RILE_RUBRIC,
                max_retries=args.max_retries,
            )
            if example:
                successful.append(example)
                logger.info(f"  Score: {example.preservation_score:.1f}, Compression: {example.compression_ratio:.2f}")
            else:
                failed += 1
        except Exception as e:
            failed += 1
            logger.error(f"  Error: {e}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = SyntheticDataset(successful)
    dataset.save(args.output_dir / f"synthetic_data_{timestamp}.json")

    # Save training formats
    dspy_examples = dataset.to_dspy_examples()
    with open(args.output_dir / f"dspy_examples_{timestamp}.json", 'w') as f:
        json.dump([e.toDict() for e in dspy_examples], f, indent=2)

    sft_data = dataset.to_sft_format()
    with open(args.output_dir / f"sft_data_{timestamp}.json", 'w') as f:
        json.dump(sft_data, f, indent=2)

    stats = generator.get_statistics()
    print_summary("SYNTHETIC GENERATION COMPLETE", {
        "Documents processed:": len(samples),
        "Successful:": len(successful),
        "Failed:": failed,
        "Avg quality:": stats.get('avg_quality_score', 0),
    })


# =============================================================================
# Main Entry Point
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified data generation for OPS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Type selection
    parser.add_argument(
        "--type", type=str, required=True,
        choices=["dpo", "labeled", "synthetic"],
        help="Generation type"
    )

    # Common options
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    # DPO options
    parser.add_argument("--comparison-module", type=Path, default=None)
    parser.add_argument("--summarizer-port", type=int, default=8000)
    parser.add_argument("--judge-port", type=int, default=8000)
    parser.add_argument("--summarizer-model", type=str, default="openai/qwen-30b-thinking")
    parser.add_argument("--judge-model", type=str, default="openai/qwen-30b-thinking")
    parser.add_argument("--k-candidates", type=int, default=4)
    parser.add_argument("--temperatures", type=float, nargs="+", default=None)
    parser.add_argument("--law-type", type=str, default="sufficiency",
                       choices=["sufficiency", "idempotence", "merge"])

    # Labeled / Synthetic options
    parser.add_argument("--oracle-port", type=int, default=8001)
    parser.add_argument("--oracle-model", type=str, default="openai/qwen-30b-thinking")
    parser.add_argument("--chunk-size", type=int, default=4000)

    # Synthetic-specific options
    parser.add_argument("--reasoning-mode", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--min-quality-score", type=float, default=70.0)
    parser.add_argument("--target-compression", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    setup_logging(verbose=args.verbose)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "dpo":
        generate_dpo_data(args)
    elif args.type == "labeled":
        generate_labeled_trees(args)
    elif args.type == "synthetic":
        generate_synthetic_data(args)
    else:
        logger.error(f"Unknown type: {args.type}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
