"""Generic learned scalar sketch tasks.

This module covers the broad case where the readout target is any scalar
oracle (exact functions or official sketch query functions). It exposes two
training primitives and one sequencing wrapper that *iteratively learn* an
underlying readout ``f`` and merge state ``g``.

Per-component primitives (single training stage each)
-----------------------------------------------------

* ``learned_scalar_sketch_task(variant="f", ...)`` — train ``f_head``; ``g``
  is held fixed (identity-merge passthrough by default, or loaded from
  ``init_g_from`` and frozen).
* ``learned_scalar_sketch_task(variant="g", ...)`` — train ``g``; ``f_head``
  is held fixed (target-scaled sigmoid of ``state[0]`` by default, or
  loaded from ``init_f_from`` and frozen).

Sequencing wrapper (multi-stage iterative learning)
---------------------------------------------------

* ``learned_sketch_sequence_task(variant="<letters>", ...)`` accepts any
  non-empty string in ``{f, g}+`` and runs one stage per letter, in order.
  The grammar is the schedule: ``"f"`` learns ``f`` once; ``"fg"`` learns
  ``f``, then refines ``g`` given the trained ``f``; ``"fgf"`` further
  refines ``f`` against the new ``g``; and so on for ``"fgfgf"`` etc.

At each stage the trained component's checkpoint becomes the next stage's
``init_<comp>_from``, so the underlying ``f`` and ``g`` models accumulate
improvements across letters. After the run, each component's *most recent*
checkpoint is the **final f** / **final g** of that schedule; both are
recorded in the result's ``artifacts`` and ``summary`` so callers can load
them for inference via :func:`load_final_sketch_models`.

The HLL-specific learned parity task in :mod:`learned_hll_parity` is the
register-space-constrained sibling of this module; it keeps a separate model
because of the classical-HLL-estimator readout constraint.

Joint single-call training (one optimizer over both components in a single
``fit()``) is not used; "iteratively learn ``f`` and ``g``" is always
expressed as a multi-letter sequence.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from treepo.sketches import (
    make_count_min_adapter,
    make_cpc_adapter,
    make_frequent_strings_adapter,
    make_hll_adapter,
    make_kll_floats_adapter,
    make_quantiles_floats_adapter,
    make_req_floats_adapter,
    make_tdigest_double_adapter,
    make_theta_adapter,
    make_tuple_accumulator_adapter,
    make_varopt_strings_adapter,
)
from treepo.sketches.adapters.datasketches_cardinality import (
    theta_a_not_b_estimate,
    theta_intersection_estimate,
    theta_union_estimate,
)

from treepo._research.unified_g_v1.dimension_guards import promote_dim
from treepo._research.unified_g_v1.sketch.classical_parity import (
    BackendName,
    ClassicalHLLParityConfig,
    ScheduleName,
    generate_documents,
)
from treepo._research.unified_g_v1.sketch.sampled_supervision import (
    attach_persistent_uniform_node_scores,
    persistent_uniform_node_mask,
    sampled_axis_mse,
    sampled_batch_mse,
    sampled_tree_node_mse,
)
from treepo._research.unified_g_v1.training.component_ladder import (
    ComponentLadderStageContext,
    ComponentLadderStageOutput,
    run_component_ladder,
)
from treepo._research.unified_g_v1.training.tree_task import TrainerConfig, TreeExample

ScalarTargetKind = str

ScalarTargetFn = Callable[[Sequence[int]], float]


def _leaf_width_floor(
    *,
    max_tokens: int,
    n_leaves: int | None,
    leaf_size: int,
) -> int:
    """Return the 2x no-compression width for the largest leaf in a cell."""

    if n_leaves is not None:
        max_leaf_tokens = int(
            math.ceil(float(max(1, int(max_tokens))) / float(max(1, int(n_leaves))))
        )
    else:
        max_leaf_tokens = min(max(1, int(max_tokens)), max(1, int(leaf_size)))
    return 2 * int(max_leaf_tokens)


@dataclass(frozen=True)
class LearnedScalarSketchConfig:
    target_kind: ScalarTargetKind = "exact_distinct"
    precision: int = 8
    n_leaves: int | None = 4
    leaf_size: int = 64
    schedule: ScheduleName = "balanced"
    backend: BackendName = "native"
    n_train: int = 64
    n_val: int = 16
    seed: int = 0
    universe_size: int = 1_000
    min_tokens: int = 64
    max_tokens: int = 256
    zipf_alphas: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    focus_token: int = 0
    cms_num_hashes: int = 5
    cms_num_buckets: int = 256
    frequent_lg_max_map_size: int = 8
    theta_lg_k: int | None = None
    quantile_query: float = 0.5
    kll_k: int = 128
    quantiles_k: int = 128
    req_k: int = 12
    tdigest_k: int = 100
    tuple_lg_k: int = 12
    varopt_k: int = 64
    input_vocab_size: int | None = None
    leaf_feature_mode: str = "count_vector"

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["zipf_alphas"] = list(self.zipf_alphas)
        return out


def _build_target_fn(cfg: LearnedScalarSketchConfig) -> ScalarTargetFn:
    target_kind = str(cfg.target_kind)
    q = float(cfg.quantile_query)
    if target_kind == "exact_distinct":
        return lambda tokens: float(len(set(int(t) for t in tokens)))
    if target_kind == "exact_frequency":
        focus = int(cfg.focus_token)
        return lambda tokens: float(sum(1 for t in tokens if int(t) == focus))
    if target_kind == "exact_total_weight":
        return lambda tokens: float(len(tokens))
    if target_kind == "exact_quantile":
        return lambda tokens: _quantile_value(tokens, q)
    if target_kind == "exact_set_union":
        return lambda tokens: float(len(_tagged_sets(tokens, int(cfg.universe_size))[0] | _tagged_sets(tokens, int(cfg.universe_size))[1]))
    if target_kind == "exact_set_intersection":
        return lambda tokens: float(len(_tagged_sets(tokens, int(cfg.universe_size))[0] & _tagged_sets(tokens, int(cfg.universe_size))[1]))
    if target_kind == "exact_set_a_not_b":
        return lambda tokens: float(len(_tagged_sets(tokens, int(cfg.universe_size))[0] - _tagged_sets(tokens, int(cfg.universe_size))[1]))
    if target_kind == "hll_reference":
        adapter = make_hll_adapter(backend=cfg.backend, precision=int(cfg.precision))

        def _hll(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([int(t) for t in tokens]), None))

        return _hll
    if target_kind == "cpc_reference":
        adapter = make_cpc_adapter(lg_k=int(cfg.precision))

        def _cpc(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([int(t) for t in tokens]), None))

        return _cpc
    if target_kind == "theta_reference":
        lg_k = int(cfg.theta_lg_k if cfg.theta_lg_k is not None else cfg.precision)
        adapter = make_theta_adapter(lg_k=lg_k)

        def _theta(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([int(t) for t in tokens]), None))

        return _theta
    if target_kind == "count_min_reference":
        adapter = make_count_min_adapter(
            num_hashes=int(cfg.cms_num_hashes),
            num_buckets=int(cfg.cms_num_buckets),
        )
        focus = int(cfg.focus_token)

        def _cms(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([int(t) for t in tokens]), focus))

        return _cms
    if target_kind == "frequent_strings_reference":
        adapter = make_frequent_strings_adapter(lg_max_map_size=int(cfg.frequent_lg_max_map_size))
        focus = str(int(cfg.focus_token))

        def _freq(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([str(int(t)) for t in tokens]), focus))

        return _freq
    if target_kind == "kll_reference":
        adapter = make_kll_floats_adapter(k=int(cfg.kll_k))

        def _kll(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([float(int(t)) for t in tokens]), q))

        return _kll
    if target_kind == "quantiles_reference":
        adapter = make_quantiles_floats_adapter(k=int(cfg.quantiles_k))

        def _quantiles(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([float(int(t)) for t in tokens]), q))

        return _quantiles
    if target_kind == "req_reference":
        adapter = make_req_floats_adapter(k=int(cfg.req_k), high_rank_accuracy=True)

        def _req(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([float(int(t)) for t in tokens]), q))

        return _req
    if target_kind == "tdigest_reference":
        adapter = make_tdigest_double_adapter(k=int(cfg.tdigest_k))

        def _tdigest(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([float(int(t)) for t in tokens]), q))

        return _tdigest
    if target_kind in {"theta_union_reference", "theta_intersection_reference", "theta_a_not_b_reference"}:
        lg_k = int(cfg.theta_lg_k if cfg.theta_lg_k is not None else cfg.precision)
        adapter = make_theta_adapter(lg_k=lg_k)

        def _theta_set(tokens: Sequence[int]) -> float:
            a_set, b_set = _tagged_sets(tokens, int(cfg.universe_size))
            a_state = adapter.encode(sorted(a_set))
            b_state = adapter.encode(sorted(b_set))
            if target_kind == "theta_union_reference":
                return float(theta_union_estimate(a_state, b_state, lg_k=lg_k))
            if target_kind == "theta_intersection_reference":
                return float(theta_intersection_estimate(a_state, b_state))
            return float(theta_a_not_b_estimate(a_state, b_state))

        return _theta_set
    if target_kind == "tuple_summary_sum_reference":
        adapter = make_tuple_accumulator_adapter(lg_k=int(cfg.tuple_lg_k))

        def _tuple_sum(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([(str(int(t)), 1) for t in tokens]), "summary_sum"))

        return _tuple_sum
    if target_kind == "varopt_total_weight_reference":
        adapter = make_varopt_strings_adapter(k=int(cfg.varopt_k))

        def _varopt(tokens: Sequence[int]) -> float:
            return float(adapter.query(adapter.encode([str(int(t)) for t in tokens]), None))

        return _varopt
    raise ValueError(f"unsupported scalar target_kind: {cfg.target_kind!r}")


def _quantile_value(tokens: Sequence[int], q: float) -> float:
    if not tokens:
        return 0.0
    return float(np.quantile(np.asarray([float(int(t)) for t in tokens], dtype=np.float64), float(q)))


def _rank_at_value(tokens: Sequence[int], value: float) -> float:
    if not tokens:
        return 0.0
    arr = np.sort(np.asarray([float(int(t)) for t in tokens], dtype=np.float64))
    return float(np.searchsorted(arr, float(value), side="right")) / float(len(arr))


def _tagged_sets(tokens: Sequence[int], universe_size: int) -> tuple[set[int], set[int]]:
    a_set: set[int] = set()
    b_set: set[int] = set()
    split = int(universe_size)
    for raw in tokens:
        tok = int(raw)
        if tok < split:
            a_set.add(tok)
        else:
            b_set.add(tok - split)
    return a_set, b_set


def target_family_query(target_kind: str, *, quantile_query: float = 0.5) -> tuple[str, str]:
    if target_kind in {
        "exact_distinct",
        "exact_distinct_union_state_space",
        "hll_reference",
        "hll_register_space",
        "cpc_reference",
        "theta_reference",
    }:
        return "distinct", "cardinality"
    if target_kind in {
        "exact_frequency",
        "exact_frequency_state_space",
        "count_min_reference",
        "count_min_state_space",
        "frequent_strings_reference",
    }:
        return "frequency", "focus_frequency"
    if target_kind in {"exact_quantile", "kll_reference", "quantiles_reference", "req_reference", "tdigest_reference"}:
        return "quantile", f"rank_at_q{float(quantile_query):g}"
    if target_kind in {"exact_set_union", "theta_union_reference"}:
        return "set", "union"
    if target_kind in {"exact_set_intersection", "theta_intersection_reference"}:
        return "set", "intersection"
    if target_kind in {"exact_set_a_not_b", "theta_a_not_b_reference"}:
        return "set", "a_not_b"
    if target_kind in {
        "exact_total_weight",
        "exact_total_weight_state_space",
        "varopt_total_weight_reference",
    }:
        return "sampling", "total_weight"
    if target_kind == "tuple_summary_sum_reference":
        return "sampling", "accumulator_summary_sum"
    return "learned", target_kind


def _is_set_target(target_kind: str) -> bool:
    return target_kind in {
        "exact_set_union",
        "exact_set_intersection",
        "exact_set_a_not_b",
        "theta_union_reference",
        "theta_intersection_reference",
        "theta_a_not_b_reference",
    }


def _input_vocab_size(cfg: LearnedScalarSketchConfig) -> int:
    if cfg.input_vocab_size is not None:
        return int(cfg.input_vocab_size)
    if _is_set_target(str(cfg.target_kind)):
        return 2 * int(cfg.universe_size)
    return int(cfg.universe_size)


def _cumulative_targets(
    leaves: Sequence[tuple[int, ...]],
    *,
    target_fn: ScalarTargetFn,
) -> list[float]:
    if len(leaves) <= 1:
        return []
    out: list[float] = []
    running: list[int] = list(leaves[0])
    for leaf in leaves[1:]:
        running.extend(int(t) for t in leaf)
        out.append(float(target_fn(running)))
    return out


def _partition_flat(tokens: Sequence[int], n_leaves: int) -> tuple[tuple[int, ...], ...]:
    items = list(int(t) for t in tokens)
    if not items:
        return ((),)
    n = max(1, min(int(n_leaves), len(items)))
    q, r = divmod(len(items), n)
    out: list[tuple[int, ...]] = []
    start = 0
    for idx in range(n):
        step = q + (1 if idx < r else 0)
        out.append(tuple(items[start : start + step]))
        start += step
    return tuple(out)


def _partition_by_leaf_size(tokens: Sequence[int], leaf_size: int) -> tuple[tuple[int, ...], ...]:
    items = list(int(t) for t in tokens)
    if not items:
        return ((),)
    size = max(1, int(leaf_size))
    return tuple(tuple(items[start : start + size]) for start in range(0, len(items), size))


def _partition_for_config(tokens: Sequence[int], cfg: LearnedScalarSketchConfig) -> tuple[tuple[int, ...], ...]:
    if cfg.n_leaves is not None:
        return _partition_flat(tokens, int(cfg.n_leaves))
    return _partition_by_leaf_size(tokens, int(cfg.leaf_size))


class LearnedScalarSketchOracle:
    """Train/val oracle for scalar sketch readouts on synthetic token streams."""

    def __init__(self, *, config: LearnedScalarSketchConfig) -> None:
        self.config = config
        self.target_fn = _build_target_fn(config)
        self._cached: list[TreeExample] | None = None

    def _to_tree_example(
        self,
        leaves: tuple[tuple[int, ...], ...],
        _analytic_truth: float,
        flat_tokens: list[int],
    ) -> TreeExample:
        leaf_values = [float(self.target_fn(list(leaf))) for leaf in leaves]
        cumulative = _cumulative_targets(leaves, target_fn=self.target_fn)
        root_target = float(self.target_fn(flat_tokens))
        extra = {
            "flat_tokens": list(flat_tokens),
            "leaf_values": leaf_values,
            "cumulative_values": cumulative,
            # Back-compat keys let existing HLL objectives inspect these
            # examples without an adapter layer.
            "leaf_cardinalities": leaf_values,
            "cumulative_cardinalities": cumulative,
            "target_kind": str(self.config.target_kind),
        }
        if str(self.config.leaf_feature_mode) == "count_vector":
            vocab = int(_input_vocab_size(self.config))
            extra["leaf_token_ids"] = tuple(
                np.asarray([int(t) % vocab for t in leaf], dtype=np.int64)
                for leaf in leaves
            )
        return TreeExample(leaves=leaves, target=root_target, extra=extra)

    def _all_examples(self) -> list[TreeExample]:
        if self._cached is None:
            if _is_set_target(str(self.config.target_kind)):
                self._cached = self._set_examples()
            else:
                data_cfg = ClassicalHLLParityConfig(
                    precision=int(self.config.precision),
                    n_leaves=int(self.config.n_leaves or 1),
                    schedule=self.config.schedule,
                    backend=self.config.backend,
                    n_val=int(self.config.n_train + self.config.n_val),
                    seed=int(self.config.seed),
                    universe_size=int(self.config.universe_size),
                    min_tokens=int(self.config.min_tokens),
                    max_tokens=int(self.config.max_tokens),
                    zipf_alphas=tuple(float(a) for a in self.config.zipf_alphas),
                    oracle_kind="analytic",
                )
                raw = generate_documents(data_cfg)
                if self.config.n_leaves is None:
                    self._cached = [
                        self._to_tree_example(
                            _partition_for_config(flat_tokens, self.config),
                            analytic_truth,
                            flat_tokens,
                        )
                        for _leaves, analytic_truth, flat_tokens in raw
                    ]
                else:
                    self._cached = [self._to_tree_example(*item) for item in raw]
            self._cached = attach_persistent_uniform_node_scores(
                self._cached,
                seed=int(self.config.seed),
            )
        return self._cached

    def _set_examples(self) -> list[TreeExample]:
        total = int(self.config.n_train + self.config.n_val)
        base = dict(
            precision=int(self.config.precision),
            n_leaves=1,
            schedule=self.config.schedule,
            backend=self.config.backend,
            n_val=total,
            universe_size=int(self.config.universe_size),
            min_tokens=max(1, int(self.config.min_tokens) // 2),
            max_tokens=max(1, int(self.config.max_tokens) // 2),
            zipf_alphas=tuple(float(a) for a in self.config.zipf_alphas),
            oracle_kind="analytic",
        )
        a_cfg = ClassicalHLLParityConfig(seed=int(self.config.seed), **base)
        b_cfg = ClassicalHLLParityConfig(seed=int(self.config.seed) + 7919, **base)
        out: list[TreeExample] = []
        for (_a_leaves, _a_truth, a_flat), (_b_leaves, _b_truth, b_flat) in zip(
            generate_documents(a_cfg),
            generate_documents(b_cfg),
        ):
            tagged = [int(t) for t in a_flat] + [int(t) + int(self.config.universe_size) for t in b_flat]
            leaves = _partition_for_config(tagged, self.config)
            out.append(self._to_tree_example(leaves, 0.0, tagged))
        return out

    def train_examples(self) -> Sequence[TreeExample]:
        return self._all_examples()[: int(self.config.n_train)]

    def val_examples(self) -> Sequence[TreeExample]:
        start = int(self.config.n_train)
        return self._all_examples()[start : start + int(self.config.n_val)]

    def metadata(self) -> Mapping[str, Any]:
        return {
            "oracle": "learned_scalar_sketch",
            "space_kind": "numeric_sequence",
            **self.config.as_dict(),
        }

    def max_target(self) -> float:
        vals: list[float] = []
        for item in self._all_examples():
            vals.append(float(item.target))
            vals.extend(float(x) for x in item.extra.get("leaf_values", ()))
            vals.extend(float(x) for x in item.extra.get("cumulative_values", ()))
        return max(vals) if vals else 1.0


class LearnedScalarSketchMergeModel(nn.Module):
    """Latent merge state plus scalar readout for f/g iterative learning.

    The architecture is uniform across variants: ``f_head``, ``g``, and
    ``merge_adapter`` always exist at the same sizes (``state_dim`` defaults
    to ``2*summary_dim`` per the FNO head invariant). The variant selects
    *which* component is trainable in this stage; the held-fixed component is
    initialized deterministically from the configured seed and frozen
    (``requires_grad=False``). When ``init_f_from`` / ``init_g_from`` is
    supplied, the corresponding submodule's weights are loaded from the
    checkpoint (warm-started if it's the trained component, frozen if it's
    the held-fixed one). ``leaf_adapter`` is a shared-interface module, so
    staged schedules load it from the most recent stage checkpoint rather
    than tying it permanently to either the f or g side.

    There is intentionally no "identity merge" or "fixed sigmoid readout"
    code path: those would be subtly different forward shapes only used in
    edge cases and would break checkpoint chaining. A held-fixed component
    is just an ordinary nn.Module whose weights aren't being updated this
    stage.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int = 32,
        summary_dim: int = 64,
        state_dim: int | None = None,
        hidden_dim: int = 128,
        target_scale: float = 1.0,
        merge_schedule: str = "left_to_right",
        variant: str = "fg",
        leaf_feature_mode: str = "count_vector",
        leaf_feature_scale: float = 1.0,
        init_f_from: str | Path | None = None,
        init_g_from: str | Path | None = None,
        init_leaf_adapter_from: str | Path | None = None,
        learn_readout: bool | None = None,
    ) -> None:
        super().__init__()
        if learn_readout is not None:
            if bool(learn_readout):
                raise ValueError(
                    "LearnedScalarSketchMergeModel: `learn_readout=True` is removed. "
                    "Use the sequenced training API: "
                    "learned_sketch_sequence_task(variant='fg', ...) or 'gf', or longer "
                    "sequences in {f,g}+. Each stage is one single-component training."
                )
            warnings.warn(
                "LearnedScalarSketchMergeModel: `learn_readout=False` is deprecated; "
                "use variant='g'.",
                DeprecationWarning,
                stacklevel=2,
            )
            variant = "g"
        if variant not in ("f", "g"):
            raise ValueError(
                f"learned_scalar_sketch: variant must be one of {{'f','g'}}; got {variant!r}. "
                f"For multi-stage variants (fg, gf, fgf, ...) use learned_sketch_sequence_task."
            )
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.summary_dim = int(summary_dim)
        self.hidden_dim = int(hidden_dim)
        self.target_scale = float(max(1.0, target_scale))
        self.merge_schedule = str(merge_schedule)
        self.variant = str(variant)
        self.leaf_feature_mode = str(leaf_feature_mode or "count_vector")
        if self.leaf_feature_mode not in {"count_vector", "embedding_mean"}:
            raise ValueError("leaf_feature_mode must be 'count_vector' or 'embedding_mean'")
        self.leaf_feature_scale = float(max(1.0, float(leaf_feature_scale)))
        self.freeze_g = variant == "f"
        self.freeze_f_head = variant == "g"

        self.state_dim = int(state_dim or 2 * self.summary_dim)
        if self.state_dim < 2 * self.summary_dim:
            raise ValueError(
                f"learned_scalar_sketch: state_dim={self.state_dim} violates the "
                f"FNO head 2x invariant (summary_dim={self.summary_dim})"
            )

        self.embedding = None
        leaf_input_dim = self.embedding_dim
        if self.leaf_feature_mode == "embedding_mean":
            self.embedding = nn.Embedding(
                self.vocab_size + 1,
                self.embedding_dim,
                padding_idx=self.vocab_size,
            )
        else:
            leaf_input_dim = self.vocab_size
        self.leaf_adapter = nn.Sequential(
            nn.Linear(int(leaf_input_dim), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.summary_dim),
        )
        # f_head is built BEFORE g for variant="fg" so f's parameters claim the
        # first deterministic random draws under the configured seed. Always
        # built so checkpoints have the slot.
        self.f_head = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.g = nn.Sequential(
            nn.Linear(self.summary_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.merge_adapter = nn.Sequential(
            nn.Linear(2 * self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.summary_dim),
        )

        if init_f_from is not None:
            self._load_submodule_from(Path(init_f_from), prefix="f_head.")
        if init_g_from is not None:
            self._load_submodule_from(Path(init_g_from), prefix="g.")
            self._load_submodule_from(Path(init_g_from), prefix="merge_adapter.", required=False)
        leaf_adapter_source = init_leaf_adapter_from
        if leaf_adapter_source is None:
            leaf_adapter_source = init_g_from if init_g_from is not None else init_f_from
        if leaf_adapter_source is not None:
            self._load_submodule_from(Path(leaf_adapter_source), prefix="leaf_adapter.", required=False)

        if self.freeze_g:
            for p in self.g.parameters():
                p.requires_grad = False
            for p in self.merge_adapter.parameters():
                p.requires_grad = False
        if self.freeze_f_head:
            for p in self.f_head.parameters():
                p.requires_grad = False

    def _load_submodule_from(self, ckpt_path: Path, *, prefix: str, required: bool = True) -> None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        own = self.state_dict()
        loaded = 0
        for key, value in state.items():
            if not key.startswith(prefix):
                continue
            if key not in own:
                continue
            own[key].copy_(value)
            loaded += 1
        if required and loaded == 0:
            raise ValueError(
                f"learned_scalar_sketch: no parameters with prefix {prefix!r} "
                f"found in checkpoint {ckpt_path}"
            )

    def _pad_token_ids(self, leaves: Sequence[Sequence[int]]) -> torch.Tensor:
        max_tokens = max(1, max(len(leaf) for leaf in leaves))
        pad = self.vocab_size
        device = next(self.parameters()).device
        out = torch.full((len(leaves), max_tokens), pad, dtype=torch.long, device=device)
        for i, leaf in enumerate(leaves):
            if leaf:
                out[i, : len(leaf)] = torch.tensor(
                    [int(t) % self.vocab_size for t in leaf],
                    dtype=torch.long,
                    device=device,
                )
        return out

    def _leaf_count_features(self, leaves: Sequence[Sequence[int]]) -> torch.Tensor:
        device = next(self.parameters()).device
        out = np.zeros((len(leaves), self.vocab_size), dtype=np.float32)
        for i, leaf in enumerate(leaves):
            if not leaf:
                continue
            ids = np.asarray([int(t) % self.vocab_size for t in leaf], dtype=np.int64)
            np.add.at(out[i], ids, 1.0)
        return torch.as_tensor(out, dtype=torch.float32, device=device) / float(self.leaf_feature_scale)

    def _encode_count_vector_token_rows(self, token_rows: Sequence[Any]) -> torch.Tensor:
        """Apply the count-vector leaf adapter without materializing dense counts.

        For count features, the first linear layer in ``leaf_adapter`` can be
        applied as a summed embedding lookup over its transposed weight matrix:
        ``Linear(counts / scale) = sum(W[:, token]) / scale + bias``. We then
        run the remaining nonlinear adapter layers normally. This keeps the
        synthetic token/count representation exact while avoiding dense
        ``[leaves, vocab]`` matrices.
        """
        device = next(self.parameters()).device
        lengths = [int(len(row)) for row in token_rows]
        if lengths:
            offsets_np = np.empty(len(lengths) + 1, dtype=np.int64)
            offsets_np[0] = 0
            np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets_np[1:])
        else:
            offsets_np = np.asarray([0], dtype=np.int64)
        if int(offsets_np[-1]) > 0:
            flat_np = np.concatenate(
                [np.asarray(row, dtype=np.int64) for row in token_rows if len(row) > 0]
            )
        else:
            flat_np = np.asarray([], dtype=np.int64)
        first = self.leaf_adapter[0]
        if not isinstance(first, nn.Linear):
            raise RuntimeError("count-vector fast path expects a linear first adapter layer")
        weight = first.weight.t().contiguous()
        if flat_np.size:
            token_ids = torch.as_tensor(flat_np, dtype=torch.long, device=device)
            offset_tensor = torch.as_tensor(offsets_np, dtype=torch.long, device=device)
            summed = F.embedding_bag(
                token_ids,
                weight,
                offset_tensor,
                mode="sum",
                include_last_offset=True,
            )
        else:
            summed = torch.zeros(
                (len(token_rows), first.out_features),
                dtype=weight.dtype,
                device=device,
            )
        hidden = summed / float(self.leaf_feature_scale)
        if first.bias is not None:
            hidden = hidden + first.bias
        summary = self.leaf_adapter[1:](hidden)
        return self.g(summary)

    def _encode_count_vector_leaves(self, leaves: Sequence[Sequence[int]]) -> torch.Tensor:
        token_rows = [
            np.asarray([int(t) % self.vocab_size for t in leaf], dtype=np.int64)
            for leaf in leaves
        ]
        return self._encode_count_vector_token_rows(token_rows)

    def _encode_leaves(self, leaves: Sequence[Sequence[int]]) -> torch.Tensor:
        if self.leaf_feature_mode == "count_vector":
            return self._encode_count_vector_leaves(leaves)
        token_ids = self._pad_token_ids(leaves)
        mask = (token_ids != self.vocab_size).float().unsqueeze(-1)
        if self.embedding is None:
            raise RuntimeError("embedding_mean mode requires an embedding module")
        embeds = self.embedding(token_ids)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.g(self.leaf_adapter(pooled))

    def _is_rectangular_batch(self, batch: Sequence[TreeExample]) -> bool:
        if not batch:
            return False
        width = len(batch[0].leaves)
        return all(len(item.leaves) == width for item in batch)

    def _encode_leaf_grid(self, batch: Sequence[TreeExample]) -> torch.Tensor:
        """Encode a rectangular batch as ``[batch, leaves, state_dim]``."""
        if not self._is_rectangular_batch(batch):
            raise ValueError("leaf grid requires equal leaf count across batch")
        batch_size = int(len(batch))
        n_leaves = int(len(batch[0].leaves))
        if self.leaf_feature_mode == "count_vector":
            cached_rows: list[Any] = []
            all_cached = True
            for item in batch:
                cached = item.extra.get("leaf_token_ids") if hasattr(item, "extra") else None
                if cached is None:
                    all_cached = False
                    break
                cached_rows.extend(cached)
            if all_cached:
                return self._encode_count_vector_token_rows(cached_rows).reshape(
                    batch_size,
                    n_leaves,
                    self.state_dim,
                )
        flat_leaves = [leaf for item in batch for leaf in item.leaves]
        return self._encode_leaves(flat_leaves).reshape(batch_size, n_leaves, self.state_dim)

    def _predict_scalar(self, state: torch.Tensor) -> torch.Tensor:
        logit = self.f_head(state).reshape(-1)
        return torch.sigmoid(logit) * self.target_scale

    def _merge_pair_batch(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([left, right], dim=-1)
        summary = self.merge_adapter(pair)
        return self.g(summary)

    def _merge_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self._merge_pair_batch(left.unsqueeze(0), right.unsqueeze(0)).squeeze(0)

    def _merge_states_batch(self, states: torch.Tensor, schedule: str) -> torch.Tensor:
        """Merge ``[batch, leaves, state_dim]`` states across the leaf axis."""
        if states.shape[1] <= 1:
            return states[:, 0, :]
        sched = str(schedule)
        if sched == "right_to_left":
            state = states[:, -1, :]
            for idx in range(states.shape[1] - 2, -1, -1):
                state = self._merge_pair_batch(states[:, idx, :], state)
            return state
        if sched == "balanced":
            current = states
            while current.shape[1] > 1:
                pair_count = current.shape[1] // 2
                left = current[:, : 2 * pair_count : 2, :]
                right = current[:, 1 : 2 * pair_count : 2, :]
                merged = self._merge_pair_batch(
                    left.reshape(-1, self.state_dim),
                    right.reshape(-1, self.state_dim),
                ).reshape(current.shape[0], pair_count, self.state_dim)
                if current.shape[1] % 2:
                    current = torch.cat([merged, current[:, -1:, :]], dim=1)
                else:
                    current = merged
            return current[:, 0, :]
        state = states[:, 0, :]
        for idx in range(1, states.shape[1]):
            state = self._merge_pair_batch(state, states[:, idx, :])
        return state

    def _merge_states(self, states: torch.Tensor, schedule: str) -> torch.Tensor:
        if states.shape[0] <= 1:
            return states[0]
        sched = str(schedule)
        if sched == "right_to_left":
            state = states[-1]
            for idx in range(states.shape[0] - 2, -1, -1):
                state = self._merge_pair(states[idx], state)
            return state
        if sched == "balanced":
            current = [states[idx] for idx in range(states.shape[0])]
            while len(current) > 1:
                nxt: list[torch.Tensor] = []
                for idx in range(0, len(current), 2):
                    if idx + 1 >= len(current):
                        nxt.append(current[idx])
                    else:
                        nxt.append(self._merge_pair(current[idx], current[idx + 1]))
                current = nxt
            return current[0]
        state = states[0]
        for idx in range(1, states.shape[0]):
            state = self._merge_pair(state, states[idx])
        return state

    def predict_scalars(self, batch: Sequence[TreeExample], *, schedule: str = "balanced") -> torch.Tensor:
        if self._is_rectangular_batch(batch):
            leaf_states = self._encode_leaf_grid(batch)
            roots = self._merge_states_batch(leaf_states, schedule)
            return self._predict_scalar(roots)
        roots: list[torch.Tensor] = []
        for item in batch:
            leaf_states = self._encode_leaves(list(item.leaves))
            roots.append(self._merge_states(leaf_states, schedule).unsqueeze(0))
        return self._predict_scalar(torch.cat(roots, dim=0))

    def forward_tree(
        self,
        batch: Sequence[TreeExample],
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        device = next(self.parameters()).device
        if self._is_rectangular_batch(batch):
            leaf_states = self._encode_leaf_grid(batch)
            batch_size, n_leaves, _state_dim = leaf_states.shape
            leaf_scalars = self._predict_scalar(
                leaf_states.reshape(batch_size * n_leaves, self.state_dim)
            ).reshape(batch_size, n_leaves)
            if n_leaves > 1:
                state = leaf_states[:, 0, :]
                merge_scalars: list[torch.Tensor] = []
                for idx in range(1, n_leaves):
                    state = self._merge_pair_batch(state, leaf_states[:, idx, :])
                    merge_scalars.append(self._predict_scalar(state).unsqueeze(1))
                merge_tensor = torch.cat(merge_scalars, dim=1)
            else:
                merge_tensor = torch.zeros(
                    batch_size,
                    0,
                    device=device,
                    dtype=leaf_states.dtype,
                )
            root_state = self._merge_states_batch(leaf_states, self.merge_schedule)
            root_scalar = self._predict_scalar(root_state)
            return root_state, root_scalar, {
                "leaf_scalars": leaf_scalars,
                "merge_scalars": merge_tensor,
            }

        roots: list[torch.Tensor] = []
        leaf_scalars_list: list[torch.Tensor] = []
        merge_scalars_list: list[torch.Tensor] = []
        for item in batch:
            leaf_states = self._encode_leaves(list(item.leaves))
            leaf_scalars = self._predict_scalar(leaf_states)
            state = leaf_states[0]
            merge_scalars: list[torch.Tensor] = []
            for idx in range(1, leaf_states.shape[0]):
                state = self._merge_pair(state, leaf_states[idx])
                merge_scalars.append(self._predict_scalar(state.unsqueeze(0)).squeeze(0))
            roots.append(self._merge_states(leaf_states, self.merge_schedule).unsqueeze(0))
            leaf_scalars_list.append(leaf_scalars.unsqueeze(0))
            if merge_scalars:
                merge_scalars_list.append(torch.stack(merge_scalars).unsqueeze(0))
            else:
                merge_scalars_list.append(
                    torch.zeros(1, 0, device=device, dtype=leaf_states.dtype)
                )
        root_state = torch.cat(roots, dim=0)
        root_scalar = self._predict_scalar(root_state)
        return root_state, root_scalar, {
            "leaf_scalars": torch.cat(leaf_scalars_list, dim=0),
            "merge_scalars": torch.cat(merge_scalars_list, dim=0),
        }


def _leaf_count_batches(
    items: Sequence[TreeExample],
    batch_size: int,
) -> list[list[TreeExample]]:
    """Return minibatches with equal observed leaf counts.

    Leaf-size mode can produce a small number of distinct leaf counts because
    documents have varying lengths. Local-law tensors have a leaf axis, so each
    minibatch must be rectangular even though the full validation set need not
    be. This preserves large minibatches instead of degrading to batch size 1.
    """
    size = max(1, int(batch_size))
    buckets: dict[int, list[TreeExample]] = {}
    order: list[int] = []
    for item in items:
        key = int(len(item.leaves))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)
    out: list[list[TreeExample]] = []
    for key in order:
        bucket = buckets[key]
        for start in range(0, len(bucket), size):
            out.append(list(bucket[start : start + size]))
    return out


@dataclass
class LearnedScalarSketchObjective:
    local_law_weight: float = 0.3
    c1_relative_weight: float = 1.0
    c2_relative_weight: float = 1.0
    c3_relative_weight: float = 1.0
    root_query_rate: float = 1.0
    leaf_query_rate: float = 1.0
    internal_query_rate: float = 1.0
    supervision_sampling_policy: str = "separate_axes"

    def _stack_1d(self, items: Sequence[TreeExample], key: str) -> torch.Tensor:
        return torch.tensor([list(item.extra[key]) for item in items], dtype=torch.float32)

    def _stack_targets(self, items: Sequence[TreeExample]) -> torch.Tensor:
        return torch.tensor([float(item.target) for item in items], dtype=torch.float32)

    def compute_loss(
        self,
        *,
        root_state: torch.Tensor,
        prediction: torch.Tensor,
        batch: Sequence[TreeExample],
        forward_aux: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, int, Mapping[str, Any]]:
        del root_state
        device = prediction.device
        targets = self._stack_targets(batch).to(device)
        root_loss = sampled_batch_mse(
            prediction,
            targets,
            rate=float(self.root_query_rate),
        )
        if forward_aux is None:
            return root_loss, int(len(batch)), {
                "root_loss": float(root_loss.detach()),
                "root_query_rate": float(self.root_query_rate),
            }

        leaf_targets = self._stack_1d(batch, "leaf_values").to(device)
        c1_loss = sampled_axis_mse(
            forward_aux["leaf_scalars"],
            leaf_targets,
            rate=float(self.leaf_query_rate),
        )
        merge_scalars = forward_aux["merge_scalars"]
        if merge_scalars.numel() > 0:
            merge_targets = self._stack_1d(batch, "cumulative_values").to(device)
            c3_loss = sampled_axis_mse(
                merge_scalars,
                merge_targets,
                rate=float(self.internal_query_rate),
            )
        else:
            c3_loss = torch.zeros((), dtype=root_loss.dtype, device=device)
        c2_loss = torch.zeros((), dtype=root_loss.dtype, device=device)

        if str(self.supervision_sampling_policy) == "uniform_all_nodes":
            node_rate = (
                float(self.root_query_rate)
                if abs(float(self.root_query_rate) - float(self.leaf_query_rate)) <= 1e-12
                and abs(float(self.root_query_rate) - float(self.internal_query_rate)) <= 1e-12
                else None
            )
            if node_rate is None:
                raise ValueError(
                    "uniform_all_nodes requires equal root, leaf, and internal query rates"
                )
            # ``cumulative_values`` includes the final root merge. The uniform
            # node pool adds root explicitly, so keep only non-root internals.
            internal_pred = merge_scalars[:, :-1] if merge_scalars.ndim >= 2 else merge_scalars[:0]
            internal_target = (
                merge_targets[:, :-1]
                if merge_scalars.numel() > 0 and merge_targets.ndim >= 2
                else None
            )
            node_width = 1 + int(leaf_targets.shape[1]) + int(internal_pred.shape[1])
            node_mask = persistent_uniform_node_mask(
                list(batch),
                width=int(node_width),
                rate=float(node_rate),
                device=device,
            )
            total = sampled_tree_node_mse(
                root_pred=prediction,
                root_target=targets,
                leaf_pred=forward_aux["leaf_scalars"],
                leaf_target=leaf_targets,
                internal_pred=internal_pred,
                internal_target=internal_target,
                rate=float(node_rate),
                node_mask=node_mask,
                node_propensity=float(node_rate),
            )
            return total, int(len(batch)), {
                "root_loss": float(root_loss.detach()),
                "c1_loss": float(c1_loss.detach()),
                "c2_loss": 0.0,
                "c3_loss": float(c3_loss.detach()),
                "local_block": float(total.detach()),
                "uniform_node_loss": float(total.detach()),
                "local_law_weight": 1.0,
                "root_query_rate": float(self.root_query_rate),
                "leaf_query_rate": float(self.leaf_query_rate),
                "internal_query_rate": float(self.internal_query_rate),
                "node_query_rate": float(node_rate),
                "supervision_sampling_policy": str(self.supervision_sampling_policy),
            }

        rho = {
            "c1": max(0.0, float(self.c1_relative_weight)),
            "c2": max(0.0, float(self.c2_relative_weight)),
            "c3": max(0.0, float(self.c3_relative_weight)),
        }
        rho_total = rho["c1"] + rho["c2"] + rho["c3"]
        lam = max(0.0, min(1.0, float(self.local_law_weight)))
        has_root_supervision = float(self.root_query_rate) > 0.0
        has_local_supervision = (
            float(self.leaf_query_rate) > 0.0 or float(self.internal_query_rate) > 0.0
        )
        if not has_root_supervision and has_local_supervision:
            lam = 1.0
        elif has_root_supervision and not has_local_supervision:
            lam = 0.0
        if rho_total <= 0.0 or not has_local_supervision:
            total = root_loss
            local = torch.zeros((), dtype=root_loss.dtype, device=device)
        else:
            local = (
                rho["c1"] * c1_loss + rho["c2"] * c2_loss + rho["c3"] * c3_loss
            ) / rho_total
            total = (1.0 - lam) * root_loss + lam * local
        return total, int(len(batch)), {
            "root_loss": float(root_loss.detach()),
            "c1_loss": float(c1_loss.detach()),
            "c2_loss": 0.0,
            "c3_loss": float(c3_loss.detach()),
            "local_block": float(local.detach()),
            "local_law_weight": float(lam),
            "root_query_rate": float(self.root_query_rate),
            "leaf_query_rate": float(self.leaf_query_rate),
            "internal_query_rate": float(self.internal_query_rate),
        }

    def evaluate(
        self,
        *,
        model: nn.Module,
        items: Sequence[TreeExample],
        batch_size: int,
    ) -> Mapping[str, Any]:
        model.eval()
        preds_chunks: list[torch.Tensor] = []
        targets_chunks: list[torch.Tensor] = []
        c1_chunks: list[torch.Tensor] = []
        c3_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in _leaf_count_batches(list(items), int(batch_size)):
                _root_state, prediction, forward_aux = model.forward_tree(batch)
                preds_chunks.append(prediction.detach().cpu())
                targets_chunks.append(self._stack_targets(batch))
                leaf_targets = self._stack_1d(batch, "leaf_values")
                c1_chunks.append(
                    (forward_aux["leaf_scalars"].detach().cpu() - leaf_targets)
                    .abs()
                    .mean(dim=-1)
                )
                if forward_aux["merge_scalars"].numel() > 0:
                    merge_targets = self._stack_1d(batch, "cumulative_values")
                    c3_chunks.append(
                        (forward_aux["merge_scalars"].detach().cpu() - merge_targets)
                        .abs()
                        .mean(dim=-1)
                    )
        preds = torch.cat(preds_chunks) if preds_chunks else torch.zeros(0)
        targets = torch.cat(targets_chunks) if targets_chunks else torch.zeros(0)
        if preds.numel() == 0:
            return {"count": 0, "mae_raw": 0.0, "mae_normalized": 0.0}
        mae = float((preds - targets).abs().mean())
        rmse = float(torch.sqrt(((preds - targets) ** 2).mean()))
        scale = float(targets.abs().clamp_min(1.0).mean())
        out: dict[str, Any] = {
            "count": int(len(items)),
            "mae_raw": mae,
            "mae_normalized": mae / max(1.0, scale),
            "root_mae": mae,
            "root_rmse": rmse,
            "root_rel_mae": mae / max(1.0, scale),
        }
        if c1_chunks:
            out["c1_mae"] = float(torch.cat(c1_chunks).mean())
            out["val_leaf_loss"] = out["c1_mae"]
        if c3_chunks:
            out["c3_mae"] = float(torch.cat(c3_chunks).mean())
            out["val_c3_loss"] = out["c3_mae"]
        return out


def learned_scalar_sketch_task(
    *,
    target_kind: ScalarTargetKind = "exact_distinct",
    precision: int = 8,
    n_leaves: int | None = 4,
    leaf_size: int = 64,
    schedule: ScheduleName = "balanced",
    backend: BackendName = "native",
    n_train: int = 64,
    n_val: int = 16,
    seed: int = 0,
    universe_size: int = 1_000,
    min_tokens: int = 64,
    max_tokens: int = 256,
    zipf_alphas: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
    focus_token: int = 0,
    cms_num_hashes: int = 5,
    cms_num_buckets: int = 256,
    frequent_lg_max_map_size: int = 8,
    theta_lg_k: int | None = None,
    quantile_query: float = 0.5,
    kll_k: int = 128,
    quantiles_k: int = 128,
    req_k: int = 12,
    tdigest_k: int = 100,
    tuple_lg_k: int = 12,
    varopt_k: int = 64,
    input_vocab_size: int | None = None,
    leaf_feature_mode: str = "count_vector",
    embedding_dim: int | None = None,
    summary_dim: int | None = None,
    state_dim: int | None = None,
    hidden_dim: int | None = None,
    n_epochs: int = 10,
    train_batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
    root_query_rate: float = 1.0,
    leaf_query_rate: float = 1.0,
    internal_query_rate: float = 1.0,
    supervision_sampling_policy: str = "separate_axes",
    variant: str | None = None,
    learn_readout: bool | None = None,
    init_f_from: str | Path | None = None,
    init_g_from: str | Path | None = None,
    init_leaf_adapter_from: str | Path | None = None,
    use_cuda: bool = False,
    cuda_device: int | None = None,
    best_metric_key: str = "mae_raw",
    eval_every_n_epochs: int = 1,
    evaluate_train_on_eval: bool = True,
) -> TrainerConfig:
    """Build a single-stage CPU-first learned scalar sketch task.

    ``variant`` selects which component is trained: ``"f"`` or ``"g"``. Each
    call returns a ``TrainerConfig`` for one stage; sequenced training (e.g.
    ``"fg"``, ``"gf"``, ``"fgf"``) is provided by
    :func:`learned_sketch_sequence_task`, which composes single-stage configs
    and threads checkpoints between stages.

    The legacy ``learn_readout`` boolean is accepted only as ``False`` (maps
    to ``variant="g"``) with a ``DeprecationWarning``. ``learn_readout=True``
    is removed and raises ``ValueError`` pointing at
    :func:`learned_sketch_sequence_task`.
    """
    if variant is None and learn_readout is not None:
        if bool(learn_readout):
            raise ValueError(
                "learned_scalar_sketch_task: `learn_readout=True` is removed. "
                "Use learned_sketch_sequence_task(variant='fg', ...) (or 'gf', "
                "or longer sequences in {f,g}+) for joint-style training."
            )
        warnings.warn(
            "learned_scalar_sketch_task: `learn_readout=False` is deprecated; "
            "use variant='g'.",
            DeprecationWarning,
            stacklevel=2,
        )
        variant = "g"
    if variant is None:
        variant = "g"
    if variant not in ("f", "g"):
        raise ValueError(
            f"learned_scalar_sketch_task: variant must be one of {{'f','g'}}; got {variant!r}. "
            f"For multi-stage variants use learned_sketch_sequence_task."
        )
    leaf_width_floor = _leaf_width_floor(
        max_tokens=int(max_tokens),
        n_leaves=int(n_leaves) if n_leaves is not None else None,
        leaf_size=int(leaf_size),
    )
    max_leaf_tokens = max(1, int(leaf_width_floor) // 2)
    context = "learned_scalar_sketch"
    resolved_embedding_dim = promote_dim(
        name="embedding_dim",
        requested=embedding_dim,
        default=int(leaf_width_floor),
        minimum=int(leaf_width_floor),
        context=context,
        reason="embedding width must be at least 2x the maximum leaf-token count",
    )
    resolved_summary_dim = promote_dim(
        name="summary_dim",
        requested=summary_dim,
        default=int(max_leaf_tokens),
        minimum=int(max_leaf_tokens),
        context=context,
        reason="leaf summary width must cover the maximum leaf-token count",
    )
    resolved_state_dim = promote_dim(
        name="state_dim",
        requested=state_dim,
        default=2 * int(resolved_summary_dim),
        minimum=2 * int(resolved_summary_dim),
        context=context,
        reason="state width must satisfy the 2x FNO head invariant relative to summary_dim",
    )
    if str(leaf_feature_mode) == "embedding_mean" and int(resolved_embedding_dim) < int(leaf_width_floor):
        raise ValueError(
            f"embedding_dim={resolved_embedding_dim} is below the 2x leaf-token "
            f"floor for embedding_mean mode; use at least {leaf_width_floor}"
        )
    hidden_floor = max(
        128,
        int(resolved_state_dim),
        2 * int(resolved_state_dim),  # merge input is (left_state, right_state)
        int(resolved_summary_dim),
    )
    resolved_hidden_dim = promote_dim(
        name="hidden_dim",
        requested=hidden_dim,
        default=int(hidden_floor),
        minimum=int(hidden_floor),
        context=context,
        reason="f hidden width must cover state_dim and g hidden width must cover 2*state_dim",
    )
    cfg = LearnedScalarSketchConfig(
        target_kind=target_kind,
        precision=int(precision),
        n_leaves=int(n_leaves) if n_leaves is not None else None,
        leaf_size=int(leaf_size),
        schedule=schedule,
        backend=backend,
        n_train=int(n_train),
        n_val=int(n_val),
        seed=int(seed),
        universe_size=int(universe_size),
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
        zipf_alphas=tuple(float(a) for a in zipf_alphas),
        focus_token=int(focus_token),
        cms_num_hashes=int(cms_num_hashes),
        cms_num_buckets=int(cms_num_buckets),
        frequent_lg_max_map_size=int(frequent_lg_max_map_size),
        theta_lg_k=theta_lg_k,
        quantile_query=float(quantile_query),
        kll_k=int(kll_k),
        quantiles_k=int(quantiles_k),
        req_k=int(req_k),
        tdigest_k=int(tdigest_k),
        tuple_lg_k=int(tuple_lg_k),
        varopt_k=int(varopt_k),
        input_vocab_size=input_vocab_size,
        leaf_feature_mode=str(leaf_feature_mode),
    )
    oracle = LearnedScalarSketchOracle(config=cfg)
    target_scale = max(1.0, 1.2 * float(oracle.max_target()))
    model = LearnedScalarSketchMergeModel(
        vocab_size=_input_vocab_size(cfg),
        embedding_dim=int(resolved_embedding_dim),
        summary_dim=int(resolved_summary_dim),
        state_dim=int(resolved_state_dim),
        hidden_dim=int(resolved_hidden_dim),
        target_scale=target_scale,
        merge_schedule=str(schedule),
        variant=str(variant),
        leaf_feature_mode=str(leaf_feature_mode),
        leaf_feature_scale=float(max_leaf_tokens),
        init_f_from=init_f_from,
        init_g_from=init_g_from,
        init_leaf_adapter_from=init_leaf_adapter_from,
    )
    objective = LearnedScalarSketchObjective(
        local_law_weight=float(local_law_weight),
        c1_relative_weight=float(c1_relative_weight),
        c2_relative_weight=float(c2_relative_weight),
        c3_relative_weight=float(c3_relative_weight),
        root_query_rate=float(root_query_rate),
        leaf_query_rate=float(leaf_query_rate),
        internal_query_rate=float(internal_query_rate),
        supervision_sampling_policy=str(supervision_sampling_policy),
    )
    return TrainerConfig(
        oracle=oracle,
        model=model,
        objective=objective,
        n_epochs=int(n_epochs),
        train_batch_size=int(train_batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        seed=int(seed),
        best_metric_key=str(best_metric_key),
        use_cuda=bool(use_cuda),
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        extra={
            "method": f"learned_{variant}",
            "variant": str(variant),
            "learn_readout": variant != "g" or init_f_from is not None,
            "readout_kind": "frozen_mlp" if model.freeze_f_head else "learned_mlp",
            "leaf_query_rate": float(leaf_query_rate),
            "internal_query_rate": float(internal_query_rate),
            "merge_kind": "frozen_mlp" if model.freeze_g else "learned_mlp",
            "init_f_from": str(init_f_from) if init_f_from is not None else None,
            "init_g_from": str(init_g_from) if init_g_from is not None else None,
            "init_leaf_adapter_from": (
                str(init_leaf_adapter_from)
                if init_leaf_adapter_from is not None
                else None
            ),
            "target_kind": str(target_kind),
            "scalar_sketch_config": cfg,
            "target_scale": float(target_scale),
            "embedding_dim": int(resolved_embedding_dim),
            "summary_dim": int(resolved_summary_dim),
            "state_dim": int(model.state_dim),
            "hidden_dim": int(resolved_hidden_dim),
            "leaf_width_floor": int(leaf_width_floor),
            "leaf_feature_mode": str(leaf_feature_mode),
            "leaf_feature_dim": int(_input_vocab_size(cfg)),
            "leaf_feature_scale": float(max_leaf_tokens),
            "projection_kind": "mergeable_projection",
            "g_input_dim": int(2 * model.state_dim),
            "batch_key": "leaf_count" if n_leaves is None else "",
            "eval_every_n_epochs": int(eval_every_n_epochs),
            "evaluate_train_on_eval": bool(evaluate_train_on_eval),
        },
    )


# --------------------------------------------------------------------------- #
# Sequenced per-component training (Pass 4)                                    #
# --------------------------------------------------------------------------- #
#
# `sequenced_learned_sketch_trainer` is a `cfg.trainer` callable that iterates a
# `{f,g}+` variant string. For each letter, it builds a single-stage
# TrainerConfig (via `learned_scalar_sketch_task`) and runs it through `fit()`
# recursively. State chaining: each component's most-recent checkpoint is
# threaded forward as the next stage's `init_*_from`. This naturally generalises
# from `f` / `g` (single stage) through `fg` / `gf` (two stages) up to arbitrary
# `{f,g}+` strings (`fgf`, `fgfgf`, ...).
#
# Joint single-call training (Pass 3 `variant="fg"`) is removed.


def learned_variant_codename(variant: str) -> str:
    """Return the user-facing codename for a staged learned-sketch variant."""
    variant = str(variant)
    if variant in {"f", "g"}:
        return variant
    return "joint"


def learned_method_name(variant: str) -> str:
    return f"learned_{learned_variant_codename(variant)}"


def sequenced_learned_sketch_trainer(
    cfg: TrainerConfig,
    output_dir: "Path | str",
    dataset: Any | None = None,
):
    """`cfg.trainer` callable that runs a `{f,g}+` variant string as N stages.

    Reads ``cfg.extra["sequence_spec"]`` for ``variant``, component inits, a
    shared leaf-adapter init, and ``per_stage_kwargs``. For each letter in ``variant``,
    builds a per-stage ``TrainerConfig`` via :func:`learned_scalar_sketch_task`
    and dispatches it through the centralised :func:`fit`. Per-stage
    checkpoints chain forward.
    """
    from treepo._research.unified_g_v1.training.fit import FitResult, fit  # late import; avoid cycle

    spec = dict(cfg.extra.get("sequence_spec", {}))
    variant = str(spec.get("variant") or "fg")
    if not variant or any(c not in ("f", "g") for c in variant):
        raise ValueError(
            f"sequenced_learned_sketch_trainer: variant must be a non-empty string "
            f"in {{f,g}}+, got {variant!r}"
        )
    per_stage_kwargs = dict(spec.get("per_stage_kwargs") or {})
    initial_components = {
        "f": spec.get("init_f_from"),
        "g": spec.get("init_g_from"),
    }
    initial_shared = {
        "leaf_adapter": (
            spec.get("init_leaf_adapter_from")
            or spec.get("init_g_from")
            or spec.get("init_f_from")
        )
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _train_stage(context: ComponentLadderStageContext) -> ComponentLadderStageOutput:
        comp = str(context.component)
        stage_cfg = learned_scalar_sketch_task(
            variant=comp,
            init_f_from=context.component_artifacts.get("f"),
            init_g_from=context.component_artifacts.get("g"),
            init_leaf_adapter_from=context.shared_artifacts.get("leaf_adapter"),
            **per_stage_kwargs,
        )
        result = fit(trainer_config=stage_cfg, output_dir=context.stage_dir)
        ckpt = context.stage_dir / "best_model.pt"
        return ComponentLadderStageOutput(
            component_artifact=ckpt,
            shared_artifacts={"leaf_adapter": ckpt},
            result=result,
        )

    ladder = run_component_ladder(
        schedule=variant,
        output_dir=output_dir,
        train_stage=_train_stage,
        initial_component_artifacts=initial_components,
        initial_shared_artifacts=initial_shared,
        allowed_components=frozenset({"f", "g"}),
    )

    stage_results = [record.result for record in ladder.stages]
    last = stage_results[-1]
    # The "final f" and "final g" are each component's most-recent checkpoint
    # (the one written by the latest stage that trained it). For a variant
    # like `fgf`, final_f comes from stage 2 and final_g from stage 1; for
    # `f` alone, final_g is whatever caller-supplied init_g_from was, or None
    # (no g was ever trained, identity merge default applies at inference).
    final_f_ckpt = (
        str(ladder.component_artifacts.get("f"))
        if ladder.component_artifacts.get("f") is not None
        else None
    )
    final_g_ckpt = (
        str(ladder.component_artifacts.get("g"))
        if ladder.component_artifacts.get("g") is not None
        else None
    )
    final_leaf_adapter_ckpt = (
        str(ladder.shared_artifacts.get("leaf_adapter"))
        if ladder.shared_artifacts.get("leaf_adapter") is not None
        else None
    )
    return FitResult(
        backend="learned_sketch_sequence",
        status=last.status,
        metrics=dict(last.metrics),
        artifacts={
            "final_checkpoint": str(ladder.final_artifact or ""),
            "final_f_checkpoint": final_f_ckpt or "",
            "final_g_checkpoint": final_g_ckpt or "",
            "final_leaf_adapter_checkpoint": final_leaf_adapter_ckpt or "",
        },
        history=last.history,
        summary={
            "variant": variant,
            "stage_components": list(ladder.schedule),
            "codename": learned_variant_codename(variant),
            "method": learned_method_name(variant),
            "final_f_checkpoint": final_f_ckpt,
            "final_g_checkpoint": final_g_ckpt,
            "final_leaf_adapter_checkpoint": final_leaf_adapter_ckpt,
            "stages": [
                {
                    "component": record.component,
                    "stage_dir": str(record.stage_dir),
                    "metrics": dict(record.result.metrics),
                    "history": list(record.result.history),
                }
                for record in ladder.stages
            ],
        },
    )


def learned_sketch_sequence_task(
    *,
    variant: str = "fg",
    init_f_from: str | Path | None = None,
    init_g_from: str | Path | None = None,
    init_leaf_adapter_from: str | Path | None = None,
    **per_stage_kwargs: Any,
) -> TrainerConfig:
    """Build a sequenced TrainerConfig that trains stages in `variant` order.

    ``variant`` is any non-empty string in ``{f, g}+``; e.g. ``"f"``, ``"g"``,
    ``"fg"``, ``"gf"``, ``"fgf"``, ``"fgfgf"``. Each letter is one stage of
    single-component training. Stage checkpoints chain forward: a component's
    most-recent checkpoint becomes the next stage's ``init_<comp>_from`` (so
    repeated stages of the same component warm-start from the prior stage).

    All other kwargs (``target_kind``, ``n_epochs``, ``n_train``, etc.) are
    forwarded verbatim to each per-stage :func:`learned_scalar_sketch_task`
    call.
    """
    if not variant or any(c not in ("f", "g") for c in variant):
        raise ValueError(
            f"learned_sketch_sequence_task: variant must be a non-empty string "
            f"in {{f,g}}+, got {variant!r}"
        )
    if "target_kind" not in per_stage_kwargs:
        raise ValueError(
            "learned_sketch_sequence_task: target_kind is required (forwarded to "
            "every per-stage learned_scalar_sketch_task call)"
        )
    return TrainerConfig(
        trainer=sequenced_learned_sketch_trainer,
        seed=int(per_stage_kwargs.get("seed", 0)),
        n_epochs=int(per_stage_kwargs.get("n_epochs", 10)),
        train_batch_size=int(per_stage_kwargs.get("train_batch_size", 16)),
        learning_rate=float(per_stage_kwargs.get("learning_rate", 1e-3)),
        weight_decay=float(per_stage_kwargs.get("weight_decay", 1e-5)),
        best_metric_key=str(per_stage_kwargs.get("best_metric_key", "mae_raw")),
        use_cuda=bool(per_stage_kwargs.get("use_cuda", False)),
        cuda_device=(
            int(per_stage_kwargs["cuda_device"])
            if per_stage_kwargs.get("cuda_device") is not None
            else None
        ),
        extra={
            "method": learned_method_name(variant),
            "codename": learned_variant_codename(variant),
            "variant": str(variant),
            "stage_components": list(variant),
            "sequence_spec": {
                "variant": str(variant),
                "init_f_from": str(init_f_from) if init_f_from is not None else None,
                "init_g_from": str(init_g_from) if init_g_from is not None else None,
                "init_leaf_adapter_from": (
                    str(init_leaf_adapter_from)
                    if init_leaf_adapter_from is not None
                    else None
                ),
                "per_stage_kwargs": dict(per_stage_kwargs),
            },
        },
    )


def load_final_sketch_models(
    run_dir: "str | Path",
    *,
    target_kind: str,
    **per_stage_kwargs: Any,
) -> TrainerConfig:
    """Build a single-stage TrainerConfig that loads the *final* f and g models
    written by a sequenced run at ``run_dir``.

    Scans ``run_dir`` for ``stage_<i>_<comp>/best_model.pt`` checkpoints and
    picks the most recent stage per component, plus the most recent stage
    overall for the shared ``leaf_adapter``. Returns a ``TrainerConfig`` whose
    ``model`` has the latest f_head, g weights, and shared interface loaded;
    the variant is set to the trailing letter of the recovered sequence so
    freeze flags reflect what was last trained. The returned config is suitable for
    inference (``cfg.model.predict_scalars(...)``) — call ``fit()`` on it
    only if you want to *continue* training from this point.

    ``target_kind`` and the other per-stage kwargs are forwarded verbatim to
    :func:`learned_scalar_sketch_task` so the model has the right oracle and
    architectural sizing. Pass the same values the original run used (or load
    them from ``run_dir/.../scalar_sketch_config.json`` if you persisted them
    there).
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"sequenced-run directory not found: {run_dir}")
    stage_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("stage_")
    )
    if not stage_dirs:
        raise ValueError(
            f"no stage_<i>_<comp>/ subdirectories under {run_dir}; "
            f"is this a sequenced-run output directory?"
        )

    # Recover the variant string and per-component latest checkpoints.
    variant_chars: list[str] = []
    init_f_from: Path | None = None
    init_g_from: Path | None = None
    init_leaf_adapter_from: Path | None = None
    for stage_dir in stage_dirs:
        # Directory name is "stage_<i>_<comp>".
        try:
            _, _idx, comp = stage_dir.name.split("_", 2)
        except ValueError:
            continue
        if comp not in ("f", "g"):
            continue
        variant_chars.append(comp)
        ckpt = stage_dir / "best_model.pt"
        if not ckpt.exists():
            continue
        init_leaf_adapter_from = ckpt
        if comp == "f":
            init_f_from = ckpt
        else:
            init_g_from = ckpt

    if not variant_chars:
        raise ValueError(f"no recognised stage subdirs under {run_dir}")
    variant = "".join(variant_chars)
    last_letter = variant[-1]

    return learned_scalar_sketch_task(
        target_kind=target_kind,
        variant=last_letter,
        init_f_from=init_f_from,
        init_g_from=init_g_from,
        init_leaf_adapter_from=init_leaf_adapter_from,
        **per_stage_kwargs,
    )


__all__ = [
    "LearnedScalarSketchConfig",
    "LearnedScalarSketchMergeModel",
    "LearnedScalarSketchObjective",
    "LearnedScalarSketchOracle",
    "ScalarTargetKind",
    "learned_scalar_sketch_task",
    "learned_method_name",
    "learned_sketch_sequence_task",
    "learned_variant_codename",
    "load_final_sketch_models",
    "sequenced_learned_sketch_trainer",
]
