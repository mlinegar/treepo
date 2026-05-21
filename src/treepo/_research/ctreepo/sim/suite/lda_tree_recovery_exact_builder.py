#!/usr/bin/env python3
"""Build xargs-friendly command lists for the exact bag-of-words LDA tree-recovery family."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


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


def _iter_commands(
    *,
    python_bin: str,
    leaf_tokens: Iterable[int],
    test_docs: int,
    doc_topic_concentrations: Iterable[float],
    quadratic_utility_weights: Iterable[float],
    seeds: Iterable[int],
    output_root: Path,
    n_topics: int,
    vocab_size: int,
    min_tokens: int,
    max_tokens: int,
    train_docs: int,
    skip_existing: bool,
) -> List[str]:
    cmds: List[str] = []
    script = "scripts/run_lda_tree_recovery_simulation.py"
    for leaf in leaf_tokens:
        for alpha in doc_topic_concentrations:
            for quad_weight in quadratic_utility_weights:
                for seed in seeds:
                    sub = (
                        f"leaf_{int(leaf)}"
                        f"/alpha_{_fmt_float(alpha)}"
                        f"/qweight_{_fmt_float(quad_weight)}"
                    )
                    base = output_root / sub / f"seed_{int(seed)}"
                    out_json = base.with_suffix(".json")
                    out_csv = base.with_suffix(".csv")
                    if skip_existing and out_json.exists() and out_csv.exists():
                        continue
                    cmd = (
                        f"{python_bin} -u {script} "
                        f"--n-topics {int(n_topics)} --vocab-size {int(vocab_size)} "
                        f"--min-tokens {int(min_tokens)} --max-tokens {int(max_tokens)} "
                        f"--doc-topic-concentration {float(alpha)} "
                        f"--quadratic-utility-weight {float(quad_weight)} "
                        f"--leaf-tokens {int(leaf)} "
                        f"--train-docs {int(train_docs)} --test-docs {int(test_docs)} "
                        f"--seed {int(seed)} "
                        f"--json-summary {out_json} --csv-summary {out_csv}"
                    )
                    cmds.append(cmd)
    return cmds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build command lists for bag-of-words LDA tree-recovery sweeps.")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--out-cmds", type=str, default="logs/lda_tree_recovery_cmds.txt")
    p.add_argument("--output-root", type=str, default="outputs/lda_tree_recovery")

    p.add_argument("--leaf-tokens", type=str, default="384 192 96 48 24 16")
    p.add_argument("--doc-topic-concentrations", type=str, default="0.3 0.6 1.2")
    p.add_argument(
        "--quadratic-utility-weights",
        "--lambda-multipliers",
        dest="quadratic_utility_weights",
        type=str,
        default="0 1 2",
    )
    p.add_argument("--seeds", type=str, default="0 1 2 3 4")

    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=384)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--train-docs", type=int, default=0)
    p.add_argument("--test-docs", type=int, default=1024)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_cmds = Path(args.out_cmds)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    cmds = _iter_commands(
        python_bin=str(args.python_bin),
        leaf_tokens=_parse_ints(args.leaf_tokens),
        test_docs=int(args.test_docs),
        doc_topic_concentrations=_parse_floats(args.doc_topic_concentrations),
        quadratic_utility_weights=_parse_floats(args.quadratic_utility_weights),
        seeds=_parse_ints(args.seeds),
        output_root=Path(args.output_root),
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        train_docs=int(args.train_docs),
        skip_existing=bool(args.skip_existing),
    )
    out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote_cmds | {out_cmds} | n_commands={len(cmds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
