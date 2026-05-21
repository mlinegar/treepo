#!/usr/bin/env python3
"""Build xargs-friendly bundle commands for fixed-world learned LDA recovery sweeps."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _parse_int_grid(s: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.replace(",", " ").split() if x.strip())
    if not vals:
        raise ValueError("expected a non-empty integer grid")
    return vals


def _parse_float_grid(s: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in s.replace(",", " ").split() if x.strip())
    if not vals:
        raise ValueError("expected a non-empty float grid")
    return vals


def _bool_flag(flag: str, value: bool) -> str:
    return f"--{flag}" if bool(value) else f"--no-{flag}"


def _output_base(
    *,
    output_root: Path,
    doc_topic_concentration: float,
    quadratic_utility_weight: float,
    leaf_tokens: int,
    train_docs: int,
    state_dim: int,
    seed: int,
) -> Path:
    return (
        output_root
        / f"dtc_{doc_topic_concentration:g}"
        / f"qweight_{quadratic_utility_weight:g}"
        / f"leaf_{leaf_tokens}"
        / f"train_{train_docs}"
        / f"state_{state_dim}"
        / f"seed_{seed}"
    )


def _bundle_is_complete(
    *,
    output_root: Path,
    doc_topic_concentration: float,
    quadratic_utility_weights: Sequence[float],
    leaf_tokens_grid: Sequence[int],
    train_docs_grid: Sequence[int],
    state_dims: Sequence[int],
    seed: int,
) -> bool:
    for lambda_multiplier, leaf_tokens, train_docs, state_dim in product(
        quadratic_utility_weights,
        leaf_tokens_grid,
        train_docs_grid,
        state_dims,
    ):
        base = _output_base(
            output_root=output_root,
            doc_topic_concentration=float(doc_topic_concentration),
            quadratic_utility_weight=float(lambda_multiplier),
            leaf_tokens=int(leaf_tokens),
            train_docs=int(train_docs),
            state_dim=int(state_dim),
            seed=int(seed),
        )
        if not (base.with_suffix(".json").exists() and base.with_suffix(".csv").exists()):
            return False
    return True


def _iter_commands(args: argparse.Namespace) -> Iterable[str]:
    script = "scripts/run_lda_tree_recovery_learned_world_batch.py"
    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    world_cache_dir = Path(str(args.world_cache_dir))
    prepared_cache_dir = Path(str(args.prepared_cache_dir))
    world_cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_int_grid(str(args.seeds))
    train_docs_grid = _parse_int_grid(str(args.train_docs_grid))
    leaf_tokens_grid = _parse_int_grid(str(args.leaf_tokens_grid))
    state_dims = _parse_int_grid(str(args.state_dims))
    doc_topic_concs = _parse_float_grid(str(args.doc_topic_concentrations))
    quadratic_utility_weights = _parse_float_grid(str(args.quadratic_utility_weights))
    max_train_docs_capacity = int(args.max_train_docs_capacity) if int(args.max_train_docs_capacity) > 0 else max(train_docs_grid)

    for doc_topic_concentration, seed in product(doc_topic_concs, seeds):
        if bool(args.skip_existing) and _bundle_is_complete(
            output_root=output_root,
            doc_topic_concentration=float(doc_topic_concentration),
            quadratic_utility_weights=quadratic_utility_weights,
            leaf_tokens_grid=leaf_tokens_grid,
            train_docs_grid=train_docs_grid,
            state_dims=state_dims,
            seed=int(seed),
        ):
            continue

        parts: List[str] = [
            "venv/bin/python",
            script,
            f"--output-root {output_root}",
            f"--world-cache-dir {world_cache_dir}",
            f"--prepared-cache-dir {prepared_cache_dir}",
            f"--doc-topic-concentration {float(doc_topic_concentration)}",
            f"--seed {int(seed)}",
            f"--leaf-tokens-grid \"{' '.join(str(x) for x in leaf_tokens_grid)}\"",
            f"--train-docs-grid \"{' '.join(str(x) for x in train_docs_grid)}\"",
            f"--state-dims \"{' '.join(str(x) for x in state_dims)}\"",
            f"--quadratic-weight-grid \"{' '.join(str(x) for x in quadratic_utility_weights)}\"",
            _bool_flag("skip-existing", bool(args.skip_existing)),
            f"--n-topics {int(args.n_topics)}",
            f"--vocab-size {int(args.vocab_size)}",
            f"--min-tokens {int(args.min_tokens)}",
            f"--max-tokens {int(args.max_tokens)}",
            f"--topic-concentration {float(args.topic_concentration)}",
            f"--emission-mode {str(args.emission_mode)}",
            f"--anchor-words-per-topic {int(args.anchor_words_per_topic)}",
            f"--anchor-multiplier {float(args.anchor_multiplier)}",
            f"--relevant-topics {int(args.relevant_topics)}",
            f"--theta-scale {float(args.theta_scale)}",
            _bool_flag("zero-diagonal", bool(args.zero_diagonal)),
            f"--max-train-docs-capacity {int(max_train_docs_capacity)}",
            f"--test-docs {int(args.test_docs)}",
            f"--inference-prior-mass {float(args.inference_prior_mass)}",
            f"--inference-max-iter {int(args.inference_max_iter)}",
            f"--inference-tol {float(args.inference_tol)}",
            f"--full-hidden-dim {int(args.full_hidden_dim)}",
            f"--full-n-layers {int(args.full_n_layers)}",
            _bool_flag("supervise-all-balanced-nodes", bool(args.supervise_all_balanced_nodes)),
            f"--n-epochs {int(args.n_epochs)}",
            f"--batch-size {int(args.batch_size)}",
            f"--lr {float(args.lr)}",
            f"--weight-decay {float(args.weight_decay)}",
            f"--device {str(args.device)}",
            f"--torch-threads {int(args.torch_threads)}",
        ]
        if args.cuda_device is not None:
            parts.append(f"--cuda-device {int(args.cuda_device)}")
        yield " ".join(parts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bundle command lists for learned LDA tree-recovery sweeps.")
    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=384)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--doc-topic-concentrations", type=str, default="0.6")
    p.add_argument("--topic-concentration", type=float, default=0.2)
    p.add_argument("--emission-mode", type=str, default="anchored")
    p.add_argument("--anchor-words-per-topic", type=int, default=20)
    p.add_argument("--anchor-multiplier", type=float, default=25.0)
    p.add_argument("--relevant-topics", type=int, default=2)
    p.add_argument("--theta-scale", type=float, default=1.0)
    p.add_argument("--zero-diagonal", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--quadratic-utility-weights",
        "--lambda-multipliers",
        dest="quadratic_utility_weights",
        type=str,
        default="1.0",
    )
    p.add_argument("--leaf-tokens-grid", type=str, default="384 16")
    p.add_argument("--train-docs-grid", type=str, default="128 256 512")
    p.add_argument("--max-train-docs-capacity", type=int, default=0)
    p.add_argument("--test-docs", type=int, default=256)
    p.add_argument("--inference-prior-mass", type=float, default=0.25)
    p.add_argument("--inference-max-iter", type=int, default=200)
    p.add_argument("--inference-tol", type=float, default=1e-9)
    p.add_argument("--full-hidden-dim", type=int, default=128)
    p.add_argument("--full-n-layers", type=int, default=2)
    p.add_argument("--state-dims", type=str, default="16 32 64 128 256 512")
    p.add_argument("--supervise-all-balanced-nodes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--seeds", type=str, default="0 1 2 3 4")
    p.add_argument("--out-cmds", type=str, default="logs/lda_tree_recovery_learned_world_batch_cmds.txt")
    p.add_argument("--output-root", type=str, default="outputs/lda_tree_recovery_learned/results")
    p.add_argument("--world-cache-dir", type=str, default="outputs/lda_tree_recovery_learned/cache/world_cache")
    p.add_argument("--prepared-cache-dir", type=str, default="outputs/lda_tree_recovery_learned/cache/prepared_cache")
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip bundle commands whose constituent JSON and CSV outputs already exist.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cmds = list(_iter_commands(args))
    out_path = Path(str(args.out_cmds))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote {len(cmds)} commands to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
