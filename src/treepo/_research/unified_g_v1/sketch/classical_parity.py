"""Classical-HLL parity task — runs flat and tree-reduced HLL through `fit()`.

This module is the empirical companion to Proposition 1 of the C-TreePO paper.
The task wires a classical HyperLogLog implementation (native or Apache
DataSketches) behind a `SketchAdapter` into the unified `fit()` framework as
a zero-optimization "trainer". The same `TrainerConfig` shape is used for
flat-reference and TreePO-reduced runs; the `n_leaves` field is the only
difference.

Everything flows through `fit()` — there is no separate benchmark
orchestration. This keeps classical-reference and learned-merge paths in
isomorphic correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import random
import time
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from treepo.bench.sketches import make_hll_adapter, treepo_reduce

from treepo._research.unified_g_v1.training.tree_task import TreeExample, TrainerConfig


ScheduleName = Literal["balanced", "left_to_right", "right_to_left"]
BackendName = Literal["native", "datasketches"]
OracleKind = Literal["analytic", "hll_reference"]

# A target function maps a sequence of leaf-token ids to a scalar f*(span).
# Passed to the oracle so the *same* f* defines per-leaf (C1), per-merge (C3),
# and root targets — ensuring local laws reference a single oracle throughout.
TargetFn = Callable[[Sequence[int]], float]


# ---------------------------------------------------------------------------
# Configuration + synthetic document generation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassicalHLLParityConfig:
    precision: int = 11
    n_leaves: int | None = 4
    leaf_size: int | None = None
    schedule: ScheduleName = "balanced"
    backend: BackendName = "datasketches"
    n_val: int = 64
    seed: int = 0
    universe_size: int = 50_000
    min_tokens: int = 512
    max_tokens: int = 2048
    zipf_alphas: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    # Oracle function selector. "analytic" (default) sets f*(x) = |set(x)|,
    # the true distinct cardinality. "hll_reference" sets f*(x) = the classical
    # HLL's own estimate on the flat token list — this is the scoring head of
    # the sketch itself. Using "hll_reference" tests whether a learned merge
    # operator g can recover the classical HLL's behavior; classical parity
    # against analytic truth becomes classical parity against HLL-as-oracle.
    oracle_kind: OracleKind = "analytic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": int(self.precision),
            "n_leaves": int(self.n_leaves) if self.n_leaves is not None else None,
            "leaf_size": int(self.leaf_size) if self.leaf_size is not None else None,
            "schedule": str(self.schedule),
            "backend": str(self.backend),
            "n_val": int(self.n_val),
            "seed": int(self.seed),
            "universe_size": int(self.universe_size),
            "min_tokens": int(self.min_tokens),
            "max_tokens": int(self.max_tokens),
            "zipf_alphas": list(self.zipf_alphas),
            "oracle_kind": str(self.oracle_kind),
        }


def _zipf_weights(universe: int, alpha: float) -> list[float]:
    weights = [1.0 / (float(k) ** float(alpha)) for k in range(1, int(universe) + 1)]
    total = sum(weights)
    return [w / total for w in weights]


def _sample_tokens(rng: random.Random, cdf: list[float], n: int) -> list[int]:
    import bisect
    out: list[int] = []
    for _ in range(n):
        u = rng.random()
        idx = bisect.bisect_left(cdf, u)
        out.append(int(min(idx, len(cdf) - 1)))
    return out


def _generate_documents(
    cfg: ClassicalHLLParityConfig,
) -> list[tuple[tuple[tuple[int, ...], ...], float, list[int]]]:
    """Generate synthetic documents partitioned into leaves.

    Returns a list of `(per_leaf_tokens, true_cardinality, flat_tokens)` triples.
    """
    if cfg.n_val <= 0:
        return []
    if cfg.n_leaves is None and cfg.leaf_size is None:
        raise ValueError("either n_leaves or leaf_size must be set")
    if cfg.n_leaves is not None and int(cfg.n_leaves) <= 0:
        raise ValueError("n_leaves must be positive")
    if cfg.leaf_size is not None and int(cfg.leaf_size) <= 0:
        raise ValueError("leaf_size must be positive")
    if cfg.max_tokens < cfg.min_tokens:
        raise ValueError("max_tokens must be >= min_tokens")

    rng = random.Random(int(cfg.seed))
    alpha_choices = list(cfg.zipf_alphas) or [1.0]
    cdf_cache: dict[float, list[float]] = {}
    for alpha in alpha_choices:
        ws = _zipf_weights(cfg.universe_size, float(alpha))
        cdf: list[float] = []
        running = 0.0
        for w in ws:
            running += w
            cdf.append(running)
        cdf_cache[float(alpha)] = cdf

    out: list[tuple[tuple[tuple[int, ...], ...], float, list[int]]] = []
    for _ in range(int(cfg.n_val)):
        alpha = float(rng.choice(alpha_choices))
        n_tok = int(rng.randint(int(cfg.min_tokens), int(cfg.max_tokens)))
        # Pad so every fixed-count leaf is non-empty.
        if cfg.n_leaves is not None:
            n_tok = max(n_tok, int(cfg.n_leaves))
        tokens = _sample_tokens(rng, cdf_cache[alpha], n_tok)
        leaves: list[tuple[int, ...]] = []
        if cfg.leaf_size is not None:
            leaf_size = max(1, int(cfg.leaf_size))
            for start in range(0, len(tokens), leaf_size):
                leaves.append(tuple(int(t) for t in tokens[start : start + leaf_size]))
        else:
            n_leaves = int(cfg.n_leaves or 1)
            per_leaf_size = max(1, (len(tokens) + n_leaves - 1) // n_leaves)
            for i in range(n_leaves):
                chunk = tokens[i * per_leaf_size : (i + 1) * per_leaf_size]
                if len(chunk) == 0:
                    chunk = [int(rng.randrange(cfg.universe_size))]
                leaves.append(tuple(int(t) for t in chunk))
        # Re-flatten so `flat_tokens` exactly equals `concat(leaves)`.
        flat = [t for leaf in leaves for t in leaf]
        truth = float(len(set(flat)))
        out.append((tuple(leaves), truth, flat))
    return out


# ---------------------------------------------------------------------------
# Oracle.
# ---------------------------------------------------------------------------


def _cumulative_targets(
    leaves: Sequence[tuple[int, ...]],
    *,
    target_fn: TargetFn,
) -> list[float]:
    """Left-to-right cumulative f*(concat(leaves[:i+1])) for each internal node."""
    if len(leaves) <= 1:
        return []
    out: list[float] = []
    running: list[int] = list(leaves[0])
    for leaf in leaves[1:]:
        running.extend(leaf)
        out.append(float(target_fn(running)))
    return out


def _analytic_target_fn(tokens: Sequence[int]) -> float:
    return float(len(set(tokens)))


def _hll_reference_target_fn(
    *,
    backend: BackendName,
    precision: int,
) -> TargetFn:
    """Return a target function that *is* the classical HLL scoring head.

    f*(x) := adapter.query(adapter.encode(x)). Using this as the oracle tests
    whether TreePO's merge (or a learned g) can match the flat classical HLL
    reference at every node — this is what the user's "pass the HLL scoring
    head as f" request translates to.
    """
    adapter = make_hll_adapter(backend=backend, precision=precision)

    def _fn(tokens: Sequence[int]) -> float:
        return float(adapter.query(adapter.encode(list(tokens))))

    return _fn


def resolve_target_fn(cfg: ClassicalHLLParityConfig) -> TargetFn:
    """Return the `f*` callable implied by `cfg.oracle_kind`."""
    if cfg.oracle_kind == "analytic":
        return _analytic_target_fn
    if cfg.oracle_kind == "hll_reference":
        return _hll_reference_target_fn(backend=cfg.backend, precision=cfg.precision)
    raise ValueError(f"unsupported oracle_kind: {cfg.oracle_kind!r}")


@dataclass
class ClassicalHLLParityOracle:
    """Oracle for the classical-HLL parity task.

    `target_fn` is the oracle function f* — applied to the flat-token list at
    every node of the tree (root, each leaf, each internal merge) to produce
    per-node targets. Override `target_fn` at construction time to supply a
    custom oracle; by default it follows `config.oracle_kind`:

    - `"analytic"` (default): f*(x) = |set(x)|, the true distinct cardinality.
    - `"hll_reference"`: f*(x) = classical HLL's own estimate on the flat
      span — i.e. the "scoring head" of the classical sketch. This is the
      target signal a learned g must match if we want g to reproduce the
      classical HLL's behavior.
    """

    config: ClassicalHLLParityConfig
    target_fn: TargetFn | None = None

    def __post_init__(self) -> None:
        if self.target_fn is None:
            self.target_fn = resolve_target_fn(self.config)

    def _to_tree_example(
        self,
        leaves: tuple[tuple[int, ...], ...],
        _analytic_truth: float,
        flat_tokens: list[int],
    ) -> TreeExample:
        target_fn: TargetFn = self.target_fn  # type: ignore[assignment]
        leaf_cards = [float(target_fn(list(leaf))) for leaf in leaves]
        cum_cards = _cumulative_targets(leaves, target_fn=target_fn)
        root_target = float(target_fn(flat_tokens))
        analytic_root = float(len(set(flat_tokens)))
        extra = {
            "flat_tokens": flat_tokens,
            "leaf_cardinalities": leaf_cards,
            "cumulative_cardinalities": cum_cards,
            "analytic_root_cardinality": analytic_root,
            "oracle_kind": str(self.config.oracle_kind),
        }
        return TreeExample(leaves=leaves, target=root_target, extra=extra)

    def train_examples(self) -> Sequence[TreeExample]:
        return []  # classical baseline has no training set

    def val_examples(self) -> Sequence[TreeExample]:
        raw = _generate_documents(self.config)
        return [self._to_tree_example(*item) for item in raw]

    def metadata(self) -> Mapping[str, Any]:
        return {
            "oracle": "classical_hll_parity",
            "space_kind": "numeric_sequence",
            **self.config.as_dict(),
        }


# ---------------------------------------------------------------------------
# Trainer callable — zero optimization, computes metrics only.
# ---------------------------------------------------------------------------


def _hll_rse(precision: int) -> float:
    return 1.04 / math.sqrt(float(1 << int(precision)))


def _relative_error(pred: float, truth: float) -> float:
    return (float(pred) - float(truth)) / max(1.0, float(truth))


def _run_adapter_over_examples(
    items: Sequence[TreeExample],
    *,
    cfg: ClassicalHLLParityConfig,
) -> dict[str, Any]:
    adapter = make_hll_adapter(backend=cfg.backend, precision=cfg.precision)

    root_abs_errs: list[float] = []
    root_sq_errs: list[float] = []
    root_rel_errs: list[float] = []
    leaf_abs_errs: list[float] = []
    merge_abs_errs: list[float] = []

    flat_vs_tree_abs: list[float] = []
    flat_vs_tree_rel: list[float] = []
    state_bytes_equal_count = 0
    state_equal_count = 0
    tree_wall: list[float] = []
    flat_wall: list[float] = []
    memory_bytes_samples: list[float] = []

    for ex in items:
        leaves = list(ex.leaves)
        truth = float(ex.target)
        flat_tokens = ex.extra["flat_tokens"]

        t0 = time.perf_counter()
        flat_state = adapter.encode(flat_tokens)
        flat_wall.append(time.perf_counter() - t0)
        flat_est = float(adapter.query(flat_state))

        t0 = time.perf_counter()
        tree_state = treepo_reduce(leaves, adapter, schedule=cfg.schedule)
        tree_wall.append(time.perf_counter() - t0)
        tree_est = float(adapter.query(tree_state))

        err = tree_est - truth
        root_abs_errs.append(abs(err))
        root_sq_errs.append(err * err)
        root_rel_errs.append(_relative_error(tree_est, truth))
        flat_vs_tree_abs.append(abs(tree_est - flat_est))
        flat_vs_tree_rel.append(abs(tree_est - flat_est) / max(1.0, abs(flat_est)))
        if adapter.serialize(tree_state) == adapter.serialize(flat_state):
            state_bytes_equal_count += 1
        if adapter.state_equal(tree_state, flat_state):
            state_equal_count += 1
        memory_bytes_samples.append(float(adapter.memory_bytes(tree_state)))

        # C1 — per-leaf cardinality estimate vs analytic truth.
        leaf_targets = list(ex.extra["leaf_cardinalities"])
        for leaf_tokens, target_card in zip(leaves, leaf_targets):
            leaf_state = adapter.encode(leaf_tokens)
            leaf_est = float(adapter.query(leaf_state))
            leaf_abs_errs.append(abs(leaf_est - float(target_card)))

        # C3 — cumulative merges along the left-to-right path.
        cum_targets = list(ex.extra["cumulative_cardinalities"])
        if cum_targets:
            running = adapter.encode(leaves[0])
            for leaf_tokens, target_card in zip(leaves[1:], cum_targets):
                running = adapter.merge(running, adapter.encode(leaf_tokens))
                cum_est = float(adapter.query(running))
                merge_abs_errs.append(abs(cum_est - float(target_card)))

    n = max(1, len(items))
    rmse = math.sqrt(sum(root_sq_errs) / n) if root_sq_errs else 0.0
    return {
        "count": int(len(items)),
        "val_mae": float(sum(root_abs_errs) / n) if root_abs_errs else 0.0,
        "root_mae": float(sum(root_abs_errs) / n) if root_abs_errs else 0.0,
        "root_rmse": float(rmse),
        "root_rel_mae": float(sum(abs(e) for e in root_rel_errs) / n) if root_rel_errs else 0.0,
        "c1_mae": float(sum(leaf_abs_errs) / max(1, len(leaf_abs_errs))) if leaf_abs_errs else 0.0,
        "c3_mae": float(sum(merge_abs_errs) / max(1, len(merge_abs_errs))) if merge_abs_errs else 0.0,
        "flat_vs_tree_abs_mean": float(sum(flat_vs_tree_abs) / n),
        "flat_vs_tree_abs_max": float(max(flat_vs_tree_abs)) if flat_vs_tree_abs else 0.0,
        "flat_vs_tree_rel_mean": float(sum(flat_vs_tree_rel) / n),
        "state_bytes_equal_rate": float(state_bytes_equal_count) / n,
        "state_equal_rate": float(state_equal_count) / n,
        "tree_wall_ms_mean": float(1000.0 * sum(tree_wall) / n),
        "flat_wall_ms_mean": float(1000.0 * sum(flat_wall) / n),
        "memory_bytes_mean": float(sum(memory_bytes_samples) / n),
        "hll_rse_theory": float(_hll_rse(cfg.precision)),
    }


def run_classical_sketch_baseline(
    cfg: TrainerConfig,
    output_dir: str | Path,
    dataset: Any = None,
):
    """`TrainerConfig.trainer` callable for the classical-parity path.

    Reads `cfg.oracle` (a `ClassicalHLLParityOracle`) and
    `cfg.extra["parity_config"]` (a `ClassicalHLLParityConfig`), runs the
    adapter through the val set via `treepo_reduce`, and returns a `FitResult`.

    No optimizer, no gradients — this is the empirical companion to
    Proposition 1. The comparison lives in the metrics schema, which matches
    `MergeableSketchObjective`'s so overlays are a CSV join.

    The oracle's `target_fn` (= its f*) is threaded through: the same function
    defines per-leaf (C1), per-merge (C3), and root targets, so when
    `oracle_kind="hll_reference"` the entire evaluation is against the HLL
    scoring head and a future learned g can pick up the same signal.
    """
    from treepo._research.unified_g_v1.training.fit import FitResult

    oracle = cfg.oracle
    if oracle is None:
        raise ValueError("classical_sketch_baseline requires cfg.oracle")
    parity_cfg_raw = dict(cfg.extra or {}).get("parity_config")
    if parity_cfg_raw is None:
        raise ValueError(
            "classical_sketch_baseline requires cfg.extra['parity_config'] "
            "(a ClassicalHLLParityConfig)"
        )
    if isinstance(parity_cfg_raw, ClassicalHLLParityConfig):
        parity_cfg = parity_cfg_raw
    else:
        parity_cfg = ClassicalHLLParityConfig(**dict(parity_cfg_raw))

    items = list(oracle.val_examples())
    t0 = time.perf_counter()
    metrics = _run_adapter_over_examples(items, cfg=parity_cfg)
    metrics["total_wall_seconds"] = float(time.perf_counter() - t0)
    metrics["oracle_kind"] = parity_cfg.oracle_kind

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import json
        (out_dir / "metrics.json").write_text(
            json.dumps({"config": parity_cfg.as_dict(), "metrics": metrics}, indent=2, sort_keys=True)
        )
    except Exception:
        pass

    return FitResult(
        backend="classical_sketch_baseline",
        summary={"config": parity_cfg.as_dict(), "metrics": dict(metrics)},
        status="completed",
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        artifacts={"metrics_json": str(out_dir / "metrics.json")},
        history=[{"epoch": 0, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}],
    )


# ---------------------------------------------------------------------------
# Preset constructor.
# ---------------------------------------------------------------------------


def classical_hll_parity_task(
    *,
    precision: int = 11,
    n_leaves: int | None = 4,
    leaf_size: int | None = None,
    schedule: ScheduleName = "balanced",
    backend: BackendName = "datasketches",
    n_val: int = 64,
    seed: int = 0,
    universe_size: int = 50_000,
    min_tokens: int = 512,
    max_tokens: int = 2048,
    zipf_alphas: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
    oracle_kind: OracleKind = "analytic",
    target_fn: TargetFn | None = None,
) -> TrainerConfig:
    """Build a `TrainerConfig` for the classical-HLL parity path.

    `fit(trainer_config=classical_hll_parity_task(n_leaves=1, ...))` runs the
    flat-reference HLL. `n_leaves=L` with the same other knobs runs the
    TreePO tree reduction. Both paths return a `FitResult` with identical
    metrics schema so summaries compare row-for-row.

    `oracle_kind="hll_reference"` swaps the oracle function f* from analytic
    cardinality to the classical HLL's own scoring head. This is the knob the
    user asked for — "pass the oracle function as f" — so that a learned g
    can later be trained to match the classical sketch's behavior at every
    node. Pass `target_fn` directly for a fully custom oracle.
    """
    cfg = ClassicalHLLParityConfig(
        precision=int(precision),
        n_leaves=int(n_leaves) if n_leaves is not None else None,
        leaf_size=int(leaf_size) if leaf_size is not None else None,
        schedule=schedule,
        backend=backend,
        n_val=int(n_val),
        seed=int(seed),
        universe_size=int(universe_size),
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
        zipf_alphas=tuple(float(a) for a in zipf_alphas),
        oracle_kind=oracle_kind,
    )
    oracle = ClassicalHLLParityOracle(config=cfg, target_fn=target_fn)
    return TrainerConfig(
        oracle=oracle,
        trainer=run_classical_sketch_baseline,
        n_epochs=0,
        seed=int(seed),
        best_metric_key="val_mae",
        extra={"parity_config": cfg},
    )


__all__ = [
    "BackendName",
    "ClassicalHLLParityConfig",
    "ClassicalHLLParityOracle",
    "OracleKind",
    "ScheduleName",
    "TargetFn",
    "classical_hll_parity_task",
    "generate_documents",
    "resolve_target_fn",
    "run_classical_sketch_baseline",
]


# Expose `_generate_documents` under a public name so the learned-g companion
# module can reuse the same synthetic-document generator without duplicating
# the Zipfian sampler.
generate_documents = _generate_documents
