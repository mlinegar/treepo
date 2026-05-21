"""Learned overlays for the broad classical-sketch grid."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from queue import Empty
from typing import Any, Iterable

import numpy as np
import torch
from treepo.bench.classical_sketches import (
    ClassicalSketchComparisonConfig,
    _metric_row,
    _row_axes,
)
from treepo.common import VALID_SCHEDULES

from treepo._research.unified_g_v1.sketch.learned_additive_state import learned_additive_state_task
from treepo._research.unified_g_v1.sketch.learned_scalar_sketch import (
    _rank_at_value,
    learned_scalar_sketch_task,
    learned_sketch_sequence_task,
    learned_variant_codename,
    target_family_query,
)
from treepo._research.unified_g_v1.training.fit import fit


@dataclass(frozen=True)
class LearnedGridTarget:
    target_kind: str
    sketch: str
    variant: str = "fg"
    quantile_query: float = 0.5
    codename: str = "joint"
    run_slug: str | None = None
    readout_arch: str = "structured"


@dataclass(frozen=True)
class _LadderPrefix:
    """Final artifacts from a completed schedule prefix for one target cell."""

    variant: str
    run_slug: str
    f_checkpoint: str | None
    g_checkpoint: str | None
    leaf_adapter_checkpoint: str | None


def _ordered_labels(values: Iterable[str]) -> list[str]:
    if isinstance(values, str):
        values = values.split(",")
    out: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if text and text not in out:
            out.append(text)
    return out or ["all"]


def _norm_targets(values: Iterable[str]) -> set[str]:
    return set(_ordered_labels(values)) or {"all"}


def _learned_variants(config: ClassicalSketchComparisonConfig) -> tuple[str, ...]:
    raw = _ordered_labels(getattr(config, "learned_variants", ("fg",)))
    if "all" in raw:
        return ("f", "g", "fg", "gf")
    aliases = {
        "f": "f",
        "f_only": "f",
        "f-only": "f",
        "learned_f": "f",
        "g": "g",
        "g_only": "g",
        "g-only": "g",
        "learned_g": "g",
        "fg": "fg",
        "g+f": "fg",
        "f+g": "fg",
        "learned_fg": "fg",
        "joint": "fg",
        "learned_joint": "fg",
        "gf": "gf",
        "learned_gf": "gf",
        "g+fg": "gfg",
        "g_then_fg": "gfg",
        "g-then-fg": "gfg",
        "learned_gfg": "gfg",
    }
    out: list[str] = []
    for value in raw:
        mapped = aliases.get(value)
        if mapped is None:
            # Allow longer raw {f,g}+ strings (e.g. "fgf", "gfg") as canonical
            # without an alias entry.
            if value and all(c in ("f", "g") for c in value):
                mapped = value
        if mapped is not None and mapped not in out:
            out.append(mapped)
    return tuple(out or ["fg"])


def _learned_readout_archs(config: ClassicalSketchComparisonConfig) -> tuple[str, ...]:
    raw = _ordered_labels(getattr(config, "learned_readout_archs", ("structured",)))
    aliases = {
        "structured": "structured",
        "exact": "structured",
        "oracle": "structured",
        "fixed": "structured",
        "mlp": "mlp",
        "learned": "mlp",
        "learned_mlp": "mlp",
    }
    out: list[str] = []
    for value in raw:
        mapped = aliases.get(value)
        if mapped is None:
            raise ValueError(
                f"unknown learned readout arch {value!r}; expected structured or mlp"
            )
        if mapped not in out:
            out.append(mapped)
    return tuple(out or ["structured"])


_ORACLE_STATE_TARGETS = frozenset(
    {
        "hll_register_space",
        "exact_distinct_union_state_space",
        "exact_frequency_state_space",
        "count_min_state_space",
        "exact_total_weight_state_space",
    }
)


def learned_grid_targets(config: ClassicalSketchComparisonConfig) -> list[LearnedGridTarget]:
    requested = _norm_targets(config.learned_targets)
    variants = _learned_variants(config)
    readout_archs = _learned_readout_archs(config)
    families = {str(x).strip().lower() for x in tuple(config.include_families)}

    def want(name: str, family: str) -> bool:
        return "all" in requested or family in requested or name in requested

    def variants_for_kind(kind: str) -> tuple[str, ...]:
        if kind in _ORACLE_STATE_TARGETS:
            # Exact state-level recovery lanes: supplied leaf state + fixed f*
            # + learned g, supplied state + exact merge + learned f, and
            # arbitrary alternating f/g schedules for diagnostics.
            return tuple(v for v in variants if v and all(c in {"f", "g"} for c in v))
        # Generic non-HLL rows are learned mergeable projections. They do not
        # have a supplied exact readout/state pair, so standalone learned-g is
        # not an oracle-f lane and should not be plotted as one.
        return tuple(v for v in variants if "f" in v and "g" in v)

    def add(kind: str, family: str, *, quantile_query: float = 0.5, name: str | None = None) -> None:
        if not want(kind, family) and (name is None or name not in requested):
            return
        label = name or kind
        for variant in variants_for_kind(kind):
            codename = learned_variant_codename(variant)
            prefix = f"learned_{codename}"
            archs = (
                readout_archs
                if kind in _ORACLE_STATE_TARGETS and "f" in str(variant)
                else ("structured",)
            )
            for readout_arch in archs:
                arch_suffix = "" if readout_arch == "structured" else f"_{readout_arch}_readout"
                sketch = f"{prefix}_{label}{arch_suffix}"
                run_slug = (
                    f"{prefix}_{variant}_{label}{arch_suffix}"
                    if codename == "joint"
                    else f"{prefix}_{label}{arch_suffix}"
                )
                out.append(
                    LearnedGridTarget(
                        kind,
                        sketch,
                        variant=variant,
                        quantile_query=float(quantile_query),
                        codename=codename,
                        run_slug=run_slug,
                        readout_arch=str(readout_arch),
                    )
                )

    out: list[LearnedGridTarget] = []
    if "distinct" in families:
        for kind in (
            "exact_distinct",
            "exact_distinct_union_state_space",
            "hll_reference",
            "hll_register_space",
            "cpc_reference",
            "theta_reference",
        ):
            add(kind, "distinct")
    if "frequency" in families:
        for kind in (
            "exact_frequency",
            "exact_frequency_state_space",
            "count_min_reference",
            "count_min_state_space",
            "frequent_strings_reference",
        ):
            add(kind, "frequency")
    if "quantile" in families:
        for q in tuple(float(x) for x in config.quantile_queries):
            for kind in (
                "exact_quantile",
                "kll_reference",
                "quantiles_reference",
                "req_reference",
                "tdigest_reference",
            ):
                name = f"{kind}_q{q:g}"
                add(kind, "quantile", quantile_query=q, name=name)
    if "set" in families:
        for kind in (
            "exact_set_union",
            "theta_union_reference",
            "exact_set_intersection",
            "theta_intersection_reference",
            "exact_set_a_not_b",
            "theta_a_not_b_reference",
        ):
            add(kind, "set")
    if "sampling" in families:
        for kind in (
            "exact_total_weight",
            "exact_total_weight_state_space",
            "tuple_summary_sum_reference",
            "varopt_total_weight_reference",
        ):
            add(kind, "sampling")
    return out


def _iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _effective_learned_batch_size(config: ClassicalSketchComparisonConfig) -> int:
    base = max(1, int(getattr(config, "learned_batch_size", 1024)))
    if config.n_leaves is not None:
        return base
    reference = max(1, int(getattr(config, "learned_batch_reference_leaf_size", 128)))
    leaf_size = max(1, int(config.leaf_size))
    max_batch = max(1, int(getattr(config, "learned_max_batch_size", base)))
    effective = int(float(base) * float(reference) / float(leaf_size))
    return max(1, min(int(max_batch), int(effective)))


def _hll_oracle_state_batch_size(config: ClassicalSketchComparisonConfig) -> int:
    requested = max(1, int(_effective_learned_batch_size(config)))
    m = int(1 << int(config.distinct_lg_k))
    hidden = int(config.learned_hidden_dim or 128)
    if config.n_leaves is not None:
        max_merges = max(1, int(config.n_leaves) - 1)
    else:
        max_merges = max(1, int(np.ceil(float(config.max_tokens) / float(max(1, config.leaf_size)))) - 1)
    # Backprop through the per-register merge MLP retains roughly
    # B * merges * m * hidden activations. Keep that under about 1 GiB so the
    # run fits beside resident vLLM workers with only a few GiB free.
    target_bytes = 1.0 * 1024.0 * 1024.0 * 1024.0
    denom = max(1.0, float(max_merges) * float(m) * float(hidden) * 4.0)
    cap = max(1, int(target_bytes // denom))
    return max(1, min(requested, cap))


def _learned_batch_metadata(
    config: ClassicalSketchComparisonConfig,
    *,
    actual_batch_size: int,
    hll_memory_capped: bool = False,
) -> dict[str, object]:
    base = max(1, int(getattr(config, "learned_batch_size", 1024)))
    reference = max(1, int(getattr(config, "learned_batch_reference_leaf_size", 128)))
    effective_uncapped = int(_effective_learned_batch_size(config))
    return {
        "learned_batch_size": int(actual_batch_size),
        "learned_effective_batch_size": int(actual_batch_size),
        "learned_effective_batch_size_uncapped": int(effective_uncapped),
        "learned_batch_size_base": int(base),
        "learned_batch_reference_leaf_size": int(reference),
        "learned_max_batch_size": int(getattr(config, "learned_max_batch_size", 8192)),
        "learned_leaf_token_batch_budget": int(base * reference),
        "learned_hll_batch_memory_capped": bool(hll_memory_capped),
        "learned_target_jobs": str(getattr(config, "learned_target_jobs", 1)),
        "learned_gpu_ids": str(getattr(config, "learned_gpu_ids", "")),
        "learned_use_cuda": bool(getattr(config, "learned_use_cuda", False)),
        "learned_root_query_rate": float(getattr(config, "learned_root_query_rate", 1.0)),
        "learned_leaf_query_rate": float(getattr(config, "learned_leaf_query_rate", 1.0)),
        "learned_internal_query_rate": float(getattr(config, "learned_internal_query_rate", 1.0)),
        "learned_supervision_sampling_policy": str(
            getattr(config, "learned_supervision_sampling_policy", "separate_axes")
        ),
        "learned_supervision_mask_design": (
            "persistent_bernoulli"
            if str(getattr(config, "learned_supervision_sampling_policy", "separate_axes"))
            == "uniform_all_nodes"
            else "legacy_axis_resampled"
        ),
        "learned_local_label_rate": (
            float(getattr(config, "learned_leaf_query_rate", 1.0))
            if abs(
                float(getattr(config, "learned_leaf_query_rate", 1.0))
                - float(getattr(config, "learned_internal_query_rate", 1.0))
            )
            <= 1e-12
            else float("nan")
        ),
        "learned_cuda_device": (
            ""
            if getattr(config, "learned_cuda_device", None) is None
            else int(getattr(config, "learned_cuda_device"))
        ),
    }


def _attach_tree_bundle_contract(
    row: dict[str, object],
    *,
    config: ClassicalSketchComparisonConfig,
    target: LearnedGridTarget,
    state_contract: str,
    f_init: str,
    g_init: str,
    schedule: str,
    final_f_checkpoint: str | None,
    final_g_checkpoint: str | None,
    final_leaf_adapter_checkpoint: str | None,
) -> dict[str, object]:
    try:
        from treepo._research.ctreepo.contracts import (
            fg_lineage_metadata,
            sketch_tree_bundle_metadata,
            tree_bundle_manifest_digest,
        )
    except Exception:
        return row

    summary_dim = int(row.get("learned_summary_dim") or 0) or None
    state_dim = int(row.get("learned_state_dim") or 0) or None
    leaf_policy = {
        "leaf_axis": "n_leaves" if config.n_leaves is not None else "leaf_size",
        "leaf_size": int(config.leaf_size),
        "n_leaves": int(config.n_leaves) if config.n_leaves is not None else None,
        "min_tokens": int(config.min_tokens),
        "max_tokens": int(config.max_tokens),
    }
    f_lineage = {
        "init": str(f_init),
        "trained": "f" in str(schedule),
        "final_checkpoint": str(final_f_checkpoint or ""),
    }
    g_lineage = {
        "init": str(g_init),
        "trained": "g" in str(schedule),
        "final_checkpoint": str(final_g_checkpoint or ""),
        "leaf_adapter_checkpoint": str(final_leaf_adapter_checkpoint or ""),
    }
    bundle = sketch_tree_bundle_metadata(
        family=str(row.get("family", "")),
        query=str(row.get("query", "")),
        sketch=str(row.get("sketch", target.sketch)),
        source_kind="raw_input",
        state_contract=str(state_contract),
        summary_dim=summary_dim,
        state_dim=state_dim,
        f_init=str(f_init),
        g_init=str(g_init),
        schedule=str(schedule),
        f_lineage=f_lineage,
        g_lineage=g_lineage,
        leaf_policy=leaf_policy,
        metadata={
            "learned_target_kind": str(target.target_kind),
            "learned_variant": str(target.variant),
            "projection_kind": str(row.get("projection_kind", "")),
        },
    )
    row.update(
        {
            "f_init": str(f_init),
            "g_init": str(g_init),
            "fg_schedule": str(schedule),
            "reducer_contract": "bottom_up",
            "tree_bundle_manifest": bundle["tree_bundle_manifest"],
            "tree_bundle_manifest_digest": tree_bundle_manifest_digest(
                bundle["tree_bundle_manifest"]
            ),
            "fg_lineage": fg_lineage_metadata(
                f_init=str(f_init),
                g_init=str(g_init),
                schedule=str(schedule),
                f_lineage=f_lineage,
                g_lineage=g_lineage,
                tree_bundle=bundle,
            ),
        }
    )
    return row


def _parse_learned_gpu_ids(value: Any) -> tuple[int, ...]:
    text = str(value if value is not None else "auto").strip().lower()
    if text in {"", "auto", "visible"}:
        try:
            if torch.cuda.is_available():
                return tuple(range(int(torch.cuda.device_count())))
        except Exception:
            return ()
        return ()
    if text in {"cpu", "none", "off", "false", "no"}:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    return tuple(int(part) for part in str(value).replace(",", " ").split() if part.strip())


def _resolve_learned_target_jobs(
    config: ClassicalSketchComparisonConfig,
    *,
    gpu_ids: tuple[int, ...],
) -> int:
    raw = getattr(config, "learned_target_jobs", 1)
    text = str(raw).strip().lower()
    if text == "auto":
        if not gpu_ids:
            return 1
        per_gpu_raw = os.environ.get("TREEPO_LEARNED_TARGET_JOBS_PER_GPU", "4")
        try:
            per_gpu = max(1, int(per_gpu_raw))
        except ValueError:
            per_gpu = 4
        return max(1, len(gpu_ids) * per_gpu)
    return max(1, int(raw))


def _row_axes_from_examples(
    config: ClassicalSketchComparisonConfig,
    items: list[Any],
) -> dict[str, object]:
    axes = _row_axes(config)
    counts = [int(len(getattr(item, "leaves", ()))) for item in items]
    tokens_per_leaf: list[float] = []
    for item in items:
        leaves = list(getattr(item, "leaves", ()))
        if not leaves:
            continue
        flat_tokens = item.extra.get("flat_tokens") if hasattr(item, "extra") else None
        if flat_tokens is None:
            total = sum(len(leaf) for leaf in leaves)
        else:
            total = len(flat_tokens)
        tokens_per_leaf.append(float(total) / float(max(1, len(leaves))))
    if counts:
        axes.update(
            {
                "leaf_count_min": int(min(counts)),
                "leaf_count_mean": float(np.mean(np.asarray(counts, dtype=np.float64))),
                "leaf_count_max": int(max(counts)),
            }
        )
    if tokens_per_leaf:
        arr = np.asarray(tokens_per_leaf, dtype=np.float64)
        axes.update(
            {
                "tokens_per_leaf_min": float(np.min(arr)),
                "tokens_per_leaf_mean": float(np.mean(arr)),
                "tokens_per_leaf_max": float(np.max(arr)),
            }
        )
    return axes


def _predict_by_schedule(
    task_cfg: Any,
    *,
    batch_size: int,
    target: LearnedGridTarget,
    items: list[Any] | None = None,
) -> dict[str, list[float]]:
    model = task_cfg.model
    items = list(task_cfg.oracle.val_examples()) if items is None else items
    out: dict[str, list[float]] = {str(s): [] for s in VALID_SCHEDULES}
    model.eval()
    with torch.no_grad():
        for schedule in VALID_SCHEDULES:
            preds: list[float] = []
            for batch in _iter_batches(items, batch_size):
                vals = model.predict_scalars(batch, schedule=str(schedule)).detach().cpu().tolist()
                preds.extend(float(v) for v in vals)
            if str(target_family_query(target.target_kind, quantile_query=target.quantile_query)[0]) == "quantile":
                preds = [
                    _rank_at_value(item.extra.get("flat_tokens", ()), pred)
                    for item, pred in zip(items, preds)
                ]
            out[str(schedule)] = preds
    return out


def _truth(task_cfg: Any, *, target: LearnedGridTarget, items: list[Any] | None = None) -> list[float]:
    items = list(task_cfg.oracle.val_examples()) if items is None else items
    family, _query = target_family_query(target.target_kind, quantile_query=target.quantile_query)
    if family == "quantile":
        return [float(target.quantile_query) for _ in items]
    return [float(item.target) for item in items]


def _model_memory_bytes(model: torch.nn.Module) -> float:
    return float(sum(int(p.numel()) * int(p.element_size()) for p in model.parameters()))


def _stage_kwargs_for_target(
    config: ClassicalSketchComparisonConfig,
    target: LearnedGridTarget,
) -> dict[str, Any]:
    """Per-stage kwargs forwarded to every `learned_scalar_sketch_task` call."""
    return dict(
        target_kind=target.target_kind,
        precision=int(config.distinct_lg_k),
        n_leaves=int(config.n_leaves) if config.n_leaves is not None else None,
        leaf_size=int(config.leaf_size),
        schedule="balanced",
        backend="datasketches" if target.target_kind == "hll_reference" else "native",
        n_train=int(config.learned_n_train),
        n_val=int(config.learned_n_val),
        seed=int(config.seed),
        universe_size=int(config.universe_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        focus_token=0,
        cms_num_hashes=int(config.cms_num_hashes),
        cms_num_buckets=int(config.cms_num_buckets),
        frequent_lg_max_map_size=int(config.frequent_lg_max_map_size),
        theta_lg_k=int(config.theta_lg_k),
        quantile_query=float(target.quantile_query),
        kll_k=int(config.kll_k),
        quantiles_k=int(config.quantiles_k),
        req_k=int(config.req_k),
        tdigest_k=int(config.tdigest_k),
        tuple_lg_k=int(config.tuple_lg_k),
        varopt_k=int(config.varopt_k),
        embedding_dim=(
            None
            if config.learned_embedding_dim is None
            else int(config.learned_embedding_dim)
        ),
        summary_dim=(
            None
            if config.learned_summary_dim is None
            else int(config.learned_summary_dim)
        ),
        state_dim=config.learned_state_dim,
        leaf_feature_mode=str(config.learned_leaf_feature_mode),
        use_cuda=bool(config.learned_use_cuda),
        cuda_device=config.learned_cuda_device,
        hidden_dim=(
            None if config.learned_hidden_dim is None else int(config.learned_hidden_dim)
        ),
        n_epochs=int(config.learned_n_epochs),
        train_batch_size=_effective_learned_batch_size(config),
        learning_rate=float(config.learned_learning_rate),
        root_query_rate=float(getattr(config, "learned_root_query_rate", 1.0)),
        leaf_query_rate=float(getattr(config, "learned_leaf_query_rate", 1.0)),
        internal_query_rate=float(getattr(config, "learned_internal_query_rate", 1.0)),
        supervision_sampling_policy=str(
            getattr(config, "learned_supervision_sampling_policy", "separate_axes")
        ),
        eval_every_n_epochs=int(config.learned_eval_every_n_epochs),
        evaluate_train_on_eval=bool(config.learned_evaluate_train_on_eval),
    )


def _build_predict_task(
    stage_kwargs: dict[str, Any],
    *,
    variant: str,
    run_dir: Path | None = None,
    f_checkpoint: str | Path | None = None,
    g_checkpoint: str | Path | None = None,
    leaf_adapter_checkpoint: str | Path | None = None,
) -> Any:
    """Build a single-stage TrainerConfig matching the *final* state of a sequenced
    run. Each component's most recent stage checkpoint is loaded; the variant is
    set to the last letter (so freeze flags reflect what was last trained).
    Used only to obtain a properly-initialised model for inference.
    """
    letters = list(variant)
    last_letter = letters[-1]

    init_f_from = Path(f_checkpoint) if f_checkpoint is not None else None
    init_g_from = Path(g_checkpoint) if g_checkpoint is not None else None
    init_leaf_adapter_from = (
        Path(leaf_adapter_checkpoint) if leaf_adapter_checkpoint is not None else None
    )
    if run_dir is not None and (
        init_f_from is None or init_g_from is None or init_leaf_adapter_from is None
    ):
        for i in range(len(letters) - 1, -1, -1):
            comp = letters[i]
            ckpt = Path(run_dir) / f"stage_{i}_{comp}" / "best_model.pt"
            if init_leaf_adapter_from is None:
                init_leaf_adapter_from = ckpt
            if comp == "f" and init_f_from is None:
                init_f_from = ckpt
            elif comp == "g" and init_g_from is None:
                init_g_from = ckpt
            if init_f_from is not None and init_g_from is not None:
                break

    return learned_scalar_sketch_task(
        variant=last_letter,
        init_f_from=init_f_from,
        init_g_from=init_g_from,
        init_leaf_adapter_from=init_leaf_adapter_from,
        **stage_kwargs,
    )


def _resolve_prefix_suffix(
    target: LearnedGridTarget,
    prefix: _LadderPrefix | None,
) -> tuple[str, str, str]:
    requested_variant = str(target.variant)
    prefix_variant = str(prefix.variant) if prefix is not None else ""
    if prefix is not None and not requested_variant.startswith(prefix_variant):
        raise ValueError(
            f"prefix variant {prefix_variant!r} is not a prefix of requested "
            f"variant {requested_variant!r}"
        )
    suffix_variant = (
        requested_variant[len(prefix_variant) :]
        if prefix is not None
        else requested_variant
    )
    if not suffix_variant:
        raise ValueError(f"empty suffix for requested variant {requested_variant!r}")
    return requested_variant, prefix_variant, suffix_variant


def _last_history_metric(result: Any, name: str, default: float = float("nan")) -> float:
    last_history = result.history[-1] if getattr(result, "history", None) else {}
    for source in (last_history, result.metrics):
        try:
            value = float(source.get(name, default))
        except (TypeError, ValueError):
            continue
        if not np.isnan(value):
            return value
    return float(default)


def _exact_state_projection_kind(
    *,
    base: str,
    variant: str,
    readout_arch: str,
) -> str:
    if "f" not in str(variant):
        return base
    if str(readout_arch) == "structured":
        return f"{base}_structured_readout"
    return f"{base}_mlp_readout_diagnostic"


def _exact_state_common_row(
    *,
    config: ClassicalSketchComparisonConfig,
    target: LearnedGridTarget,
    family: str,
    query: str,
    result: Any,
    cfg: Any,
    started_at: float,
    requested_variant: str,
    prefix_variant: str,
    suffix_variant: str,
    final_checkpoint: str,
    final_f_checkpoint: str,
    final_g_checkpoint: str,
    projection_kind: str,
    hll_parity_method: str = "",
    hll_memory_capped: bool = False,
) -> dict[str, object]:
    val_items = list(cfg.oracle.val_examples())
    state_dim_value = int(cfg.extra.get("state_dim", 0))
    row: dict[str, object] = {
        "family": family,
        "sketch": target.sketch,
        "query": query,
        "implementation_status": "learned_empirical",
        "formal_status": "learned_empirical",
        "relative_rmse": _last_history_metric(result, "root_rel_mae"),
        "schedule_spread_mean": 0.0,
        "official_floor_rel_rmse": 0.0
        if str(target.target_kind) in _ORACLE_STATE_TARGETS
        else float("nan"),
        "distance_to_official_floor": (
            _last_history_metric(result, "root_rel_mae")
            if str(target.target_kind) in _ORACLE_STATE_TARGETS
            else float("nan")
        ),
        "bound_coverage_2sigma": float("nan"),
        "theoretical_error": float("nan"),
        "memory_bytes_mean": float(_model_memory_bytes(cfg.model)),
        "val_mae_raw": _last_history_metric(result, "val_mae_raw"),
        "c1_mae": _last_history_metric(result, "c1_mae"),
        "c3_mae": _last_history_metric(result, "c3_mae"),
        "merge_state_mae": _last_history_metric(result, "merge_state_mae"),
    }
    row.update(_row_axes_from_examples(config, val_items))
    readout_arch = str(cfg.extra.get("readout_arch", target.readout_arch or "structured"))
    row.update(
        {
            "learned_target_kind": str(target.target_kind),
            "learned_variant": str(target.variant),
            "learned_codename": str(target.codename),
            "learned_run_slug": str(target.run_slug or target.sketch),
            "learn_readout": "f" in str(target.variant),
            "learned_backend": str(result.backend),
            "learned_status": str(result.status),
            "learned_wall_seconds": float(time.perf_counter() - started_at),
            "learned_eval_every_n_epochs": int(config.learned_eval_every_n_epochs),
            "learned_evaluate_train_on_eval": bool(config.learned_evaluate_train_on_eval),
            "learned_n_train": int(config.learned_n_train),
            "learned_n_val": int(config.learned_n_val),
            "learned_best_metric_value": float(result.metrics.get("best_metric_value", float("nan"))),
            "learned_stage_components": list(requested_variant),
            "learned_trained_stage_components": list(suffix_variant),
            "learned_reused_prefix": bool(prefix_variant),
            "learned_prefix_variant": str(prefix_variant),
            "learned_prefix_run_slug": "",
            "learned_suffix_variant": str(suffix_variant),
            "final_checkpoint": str(final_checkpoint),
            "final_f_checkpoint": str(final_f_checkpoint),
            "final_g_checkpoint": str(final_g_checkpoint),
            "final_leaf_adapter_checkpoint": "",
            "hll_parity_method": str(hll_parity_method),
            "learned_embedding_dim": int(cfg.extra.get("embedding_dim", 0)),
            "learned_summary_dim": int(cfg.extra.get("summary_dim", 0)),
            "learned_state_dim": state_dim_value,
            "learned_hidden_dim": int(cfg.extra.get("hidden_dim", 0)),
            "learned_leaf_width_floor": int(cfg.extra.get("leaf_width_floor", 0)),
            "projection_kind": str(projection_kind),
            "readout_arch": readout_arch,
            "learned_readout_arch": readout_arch,
            "exact_state_mode": "structured_exact" if readout_arch == "structured" else "mlp_readout_stress",
            "state_space_kind": str(cfg.extra.get("state_space_kind", "")),
            "merge_kind": str(cfg.extra.get("merge_kind", "")),
            "readout_kind": str(cfg.extra.get("readout_kind", "")),
            "g_input_dim": int(cfg.extra.get("g_input_dim", 2 * state_dim_value)),
            "leaf_feature_dim": int(cfg.extra.get("leaf_feature_dim", state_dim_value)),
            "output_dim": 1,
        }
    )
    row.update(
        _learned_batch_metadata(
            config,
            actual_batch_size=int(cfg.train_batch_size),
            hll_memory_capped=bool(hll_memory_capped),
        )
    )
    return _attach_tree_bundle_contract(
        row,
        config=config,
        target=target,
        state_contract="oracle_state",
        f_init="official_oracle",
        g_init="oracle_state",
        schedule=str(requested_variant),
        final_f_checkpoint=str(final_f_checkpoint),
        final_g_checkpoint=str(final_g_checkpoint),
        final_leaf_adapter_checkpoint=None,
    )


def _run_target_hll_register_space(
    config: ClassicalSketchComparisonConfig,
    target: LearnedGridTarget,
    *,
    output_dir: Path,
    prefix: _LadderPrefix | None = None,
) -> tuple[dict[str, object], _LadderPrefix | None] | None:
    """Dispatch a broad-suite exact-state HLL cell through an f/g ladder."""
    from treepo._research.unified_g_v1.sketch.learned_hll_parity import learned_hll_parity_task

    variant = str(target.variant)
    if not variant or any(ch not in {"f", "g"} for ch in variant):
        return None

    family, query = target_family_query(target.target_kind, quantile_query=target.quantile_query)
    run_dir = output_dir / (target.run_slug or target.sketch)
    t0 = time.perf_counter()
    requested_variant, prefix_variant, suffix_variant = _resolve_prefix_suffix(target, prefix)
    current_checkpoint = (
        str(prefix.leaf_adapter_checkpoint or prefix.g_checkpoint or prefix.f_checkpoint)
        if prefix is not None
        else ""
    )
    latest_f = str(prefix.f_checkpoint or "") if prefix is not None else ""
    latest_g = str(prefix.g_checkpoint or "") if prefix is not None else ""
    result = None
    cfg = None
    hll_memory_capped = False
    last_method = ""
    for offset, component in enumerate(suffix_variant):
        method = "learned_f_oracle_state" if component == "f" else "learned_g_oracle_state"
        use_learned_readout = (
            True
            if component == "f"
            else str(target.readout_arch or "structured") == "mlp" and bool(latest_f)
        )
        stage_index = len(prefix_variant) + offset
        stage_batch = (
            _hll_oracle_state_batch_size(config)
            if component == "g"
            else _effective_learned_batch_size(config)
        )
        hll_memory_capped = hll_memory_capped or (
            component == "g" and int(stage_batch) < int(_effective_learned_batch_size(config))
        )
        cfg = learned_hll_parity_task(
            method=method,
            precision=int(config.distinct_lg_k),
            n_leaves=int(config.n_leaves) if config.n_leaves is not None else None,
            leaf_size=int(config.leaf_size) if config.n_leaves is None else None,
            n_train=int(config.learned_n_train),
            n_val=int(config.learned_n_val),
            seed=int(config.seed),
            universe_size=int(config.universe_size),
            min_tokens=int(config.min_tokens),
            max_tokens=int(config.max_tokens),
            embedding_dim=(
                None
                if config.learned_embedding_dim is None
                else int(config.learned_embedding_dim)
            ),
            summary_dim=(
                None
                if config.learned_summary_dim is None
                else int(config.learned_summary_dim)
            ),
            state_dim=config.learned_state_dim,
            hidden_dim=(
                None if config.learned_hidden_dim is None else int(config.learned_hidden_dim)
            ),
            readout_arch=str(target.readout_arch or "structured"),
            use_learned_readout=use_learned_readout,
            n_epochs=int(config.learned_n_epochs),
            train_batch_size=int(stage_batch),
            learning_rate=float(config.learned_learning_rate),
            use_cuda=bool(config.learned_use_cuda),
            cuda_device=config.learned_cuda_device,
            eval_every_n_epochs=int(config.learned_eval_every_n_epochs),
            evaluate_train_on_eval=bool(config.learned_evaluate_train_on_eval),
            root_query_rate=float(getattr(config, "learned_root_query_rate", 1.0)),
            leaf_query_rate=float(getattr(config, "learned_leaf_query_rate", 1.0)),
            internal_query_rate=float(getattr(config, "learned_internal_query_rate", 1.0)),
            supervision_sampling_policy=str(
                getattr(config, "learned_supervision_sampling_policy", "separate_axes")
            ),
            init_from=current_checkpoint or None,
        )
        stage_dir = run_dir / f"stage_{stage_index}_{component}"
        result = fit(trainer_config=cfg, output_dir=stage_dir)
        current_checkpoint = str(stage_dir / "best_model.pt")
        if component == "f" or latest_f:
            latest_f = current_checkpoint
        if component == "g" or latest_g:
            latest_g = current_checkpoint
        last_method = method
    if result is None or cfg is None:
        return None
    final_checkpoint = current_checkpoint
    projection_kind = _exact_state_projection_kind(
        base="hll_oracle_state",
        variant=requested_variant,
        readout_arch=str(target.readout_arch or "structured"),
    )
    row = _exact_state_common_row(
        config=config,
        target=target,
        family=family,
        query=query,
        result=result,
        cfg=cfg,
        started_at=t0,
        requested_variant=requested_variant,
        prefix_variant=prefix_variant,
        suffix_variant=suffix_variant,
        final_checkpoint=final_checkpoint,
        final_f_checkpoint=latest_f,
        final_g_checkpoint=latest_g,
        projection_kind=projection_kind,
        hll_parity_method=last_method,
        hll_memory_capped=hll_memory_capped,
    )
    if prefix is not None:
        row["learned_prefix_run_slug"] = str(prefix.run_slug)
    state = _LadderPrefix(
        variant=requested_variant,
        run_slug=str(target.run_slug or target.sketch),
        f_checkpoint=latest_f or None,
        g_checkpoint=latest_g or None,
        leaf_adapter_checkpoint=None,
    )
    return row, state


def _run_target_additive_oracle_state(
    config: ClassicalSketchComparisonConfig,
    target: LearnedGridTarget,
    *,
    output_dir: Path,
    prefix: _LadderPrefix | None = None,
) -> tuple[dict[str, object], _LadderPrefix | None] | None:
    """Run an additive exact-state target through an f/g ladder."""
    variant = str(target.variant)
    if not variant or any(ch not in {"f", "g"} for ch in variant):
        return None

    family, query = target_family_query(target.target_kind, quantile_query=target.quantile_query)
    run_dir = output_dir / (target.run_slug or target.sketch)
    t0 = time.perf_counter()
    common_kwargs = dict(
        target_kind=target.target_kind,  # type: ignore[arg-type]
        precision=int(config.distinct_lg_k),
        n_leaves=int(config.n_leaves) if config.n_leaves is not None else None,
        leaf_size=int(config.leaf_size) if config.n_leaves is None else None,
        schedule="balanced",
        backend="native",
        n_train=int(config.learned_n_train),
        n_val=int(config.learned_n_val),
        seed=int(config.seed),
        universe_size=int(config.universe_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        focus_token=0,
        cms_num_hashes=int(config.cms_num_hashes),
        cms_num_buckets=int(config.cms_num_buckets),
        n_epochs=int(config.learned_n_epochs),
        train_batch_size=_effective_learned_batch_size(config),
        learning_rate=float(config.learned_learning_rate),
        use_cuda=bool(config.learned_use_cuda),
        cuda_device=config.learned_cuda_device,
        eval_every_n_epochs=int(config.learned_eval_every_n_epochs),
        evaluate_train_on_eval=bool(config.learned_evaluate_train_on_eval),
        root_query_rate=float(getattr(config, "learned_root_query_rate", 1.0)),
        leaf_query_rate=float(getattr(config, "learned_leaf_query_rate", 1.0)),
        internal_query_rate=float(getattr(config, "learned_internal_query_rate", 1.0)),
        supervision_sampling_policy=str(
            getattr(config, "learned_supervision_sampling_policy", "separate_axes")
        ),
    )
    requested_variant, prefix_variant, suffix_variant = _resolve_prefix_suffix(target, prefix)
    current_checkpoint = (
        str(prefix.leaf_adapter_checkpoint or prefix.g_checkpoint or prefix.f_checkpoint)
        if prefix is not None
        else ""
    )
    latest_f = str(prefix.f_checkpoint or "") if prefix is not None else ""
    latest_g = str(prefix.g_checkpoint or "") if prefix is not None else ""
    result = None
    cfg = None
    for offset, component in enumerate(suffix_variant):
        use_learned_readout = (
            True
            if component == "f"
            else str(target.readout_arch or "structured") == "mlp" and bool(latest_f)
        )
        cfg = learned_additive_state_task(
            **common_kwargs,
            variant=component,  # type: ignore[arg-type]
            readout_arch=str(target.readout_arch or "structured"),
            readout_hidden_dim=config.learned_hidden_dim,
            use_learned_readout=use_learned_readout,
            init_from=current_checkpoint or None,
        )
        stage_index = len(prefix_variant) + offset
        stage_dir = run_dir / f"stage_{stage_index}_{component}"
        result = fit(trainer_config=cfg, output_dir=stage_dir)
        current_checkpoint = str(stage_dir / "best_model.pt")
        if component == "f" or latest_f:
            latest_f = current_checkpoint
        if component == "g" or latest_g:
            latest_g = current_checkpoint
    if result is None or cfg is None:
        return None
    final_checkpoint = current_checkpoint
    projection_kind = _exact_state_projection_kind(
        base=str(cfg.extra.get("projection_kind", "oracle_state")),
        variant=requested_variant,
        readout_arch=str(target.readout_arch or "structured"),
    )
    row = _exact_state_common_row(
        config=config,
        target=target,
        family=family,
        query=query,
        result=result,
        cfg=cfg,
        started_at=t0,
        requested_variant=requested_variant,
        prefix_variant=prefix_variant,
        suffix_variant=suffix_variant,
        final_checkpoint=final_checkpoint,
        final_f_checkpoint=latest_f,
        final_g_checkpoint=latest_g,
        projection_kind=projection_kind,
        hll_parity_method="",
        hll_memory_capped=False,
    )
    if prefix is not None:
        row["learned_prefix_run_slug"] = str(prefix.run_slug)
    state = _LadderPrefix(
        variant=requested_variant,
        run_slug=str(target.run_slug or target.sketch),
        f_checkpoint=latest_f or None,
        g_checkpoint=latest_g or None,
        leaf_adapter_checkpoint=None,
    )
    return row, state


def _run_target(
    config: ClassicalSketchComparisonConfig,
    target: LearnedGridTarget,
    *,
    output_dir: Path,
    prefix: _LadderPrefix | None = None,
) -> tuple[dict[str, object], _LadderPrefix | None]:
    if str(target.target_kind) == "hll_register_space":
        # Dispatch register-space HLL cells to the dedicated parity task.
        # Returns None if the variant is incompatible (e.g. standalone `f`).
        result = _run_target_hll_register_space(
            config, target, output_dir=output_dir, prefix=prefix
        )
        if result is not None:
            return result
        # Fall through with a placeholder so the grid records an entry.
        family, query = target_family_query(target.target_kind, quantile_query=target.quantile_query)
        return {
            "family": family,
            "sketch": target.sketch,
            "query": query,
            "implementation_status": "skipped",
            "formal_status": "skipped",
            **_row_axes(config),
            "learned_target_kind": str(target.target_kind),
            "learned_variant": str(target.variant),
            "learned_codename": str(target.codename),
            "learned_run_slug": str(target.run_slug or target.sketch),
            "learned_skip_reason": "hll_register_space has no analog for variant",
        }, None

    if str(target.target_kind) in _ORACLE_STATE_TARGETS:
        result = _run_target_additive_oracle_state(
            config, target, output_dir=output_dir, prefix=prefix
        )
        if result is not None:
            return result
        family, query = target_family_query(target.target_kind, quantile_query=target.quantile_query)
        return {
            "family": family,
            "sketch": target.sketch,
            "query": query,
            "implementation_status": "skipped",
            "formal_status": "skipped",
            **_row_axes(config),
            "learned_target_kind": str(target.target_kind),
            "learned_variant": str(target.variant),
            "learned_codename": str(target.codename),
            "learned_run_slug": str(target.run_slug or target.sketch),
            "learned_skip_reason": "oracle-state lane only supports variant=f or variant=g",
        }, None

    family, query = target_family_query(target.target_kind, quantile_query=target.quantile_query)
    stage_kwargs = _stage_kwargs_for_target(config, target)
    run_dir = output_dir / (target.run_slug or target.sketch)
    requested_variant = str(target.variant)
    prefix_variant = str(prefix.variant) if prefix is not None else ""
    if prefix is not None and not requested_variant.startswith(prefix_variant):
        raise ValueError(
            f"prefix variant {prefix_variant!r} is not a prefix of requested "
            f"variant {requested_variant!r}"
        )
    suffix_variant = requested_variant[len(prefix_variant) :] if prefix is not None else requested_variant
    if not suffix_variant:
        raise ValueError(f"empty suffix for requested variant {requested_variant!r}")

    seq_cfg = learned_sketch_sequence_task(
        variant=suffix_variant,
        init_f_from=prefix.f_checkpoint if prefix is not None else None,
        init_g_from=prefix.g_checkpoint if prefix is not None else None,
        init_leaf_adapter_from=(
            prefix.leaf_adapter_checkpoint if prefix is not None else None
        ),
        **stage_kwargs,
    )
    t0 = time.perf_counter()
    result = fit(trainer_config=seq_cfg, output_dir=run_dir)
    final_f_checkpoint = str(result.summary.get("final_f_checkpoint") or "")
    final_g_checkpoint = str(result.summary.get("final_g_checkpoint") or "")
    final_leaf_adapter_checkpoint = str(
        result.summary.get("final_leaf_adapter_checkpoint") or ""
    )

    # Build a predict task that reflects the *final* trained state, with the
    # right freeze flags and dimensions for the trailing component.
    predict_cfg = _build_predict_task(
        stage_kwargs,
        variant=requested_variant,
        f_checkpoint=final_f_checkpoint or None,
        g_checkpoint=final_g_checkpoint or None,
        leaf_adapter_checkpoint=final_leaf_adapter_checkpoint or None,
    )
    val_items = list(predict_cfg.oracle.val_examples())
    pred_by_schedule = _predict_by_schedule(
        predict_cfg,
        batch_size=_effective_learned_batch_size(config),
        target=target,
        items=val_items,
    )
    truth = _truth(predict_cfg, target=target, items=val_items)
    mem = [_model_memory_bytes(predict_cfg.model) for _ in truth]
    row = _metric_row(
        family=family,
        sketch=target.sketch,
        query=query,
        implementation_status="learned_empirical",
        formal_status="learned_empirical",
        truth=truth,
        pred_by_schedule=pred_by_schedule,
        memory_bytes=mem,
        theoretical_error=None,
    )
    row.update(_row_axes_from_examples(config, val_items))
    row.update(
        {
            "learned_target_kind": str(target.target_kind),
            "learned_variant": str(target.variant),
            "learned_codename": str(target.codename),
            "learned_run_slug": str(target.run_slug or target.sketch),
            "learn_readout": "f" in str(target.variant),
            "learned_backend": str(result.backend),
            "learned_status": str(result.status),
            "learned_wall_seconds": float(time.perf_counter() - t0),
            "learned_eval_every_n_epochs": int(config.learned_eval_every_n_epochs),
            "learned_evaluate_train_on_eval": bool(config.learned_evaluate_train_on_eval),
            "learned_n_train": int(config.learned_n_train),
            "learned_n_val": int(config.learned_n_val),
            "learned_best_metric_value": float(result.metrics.get("best_metric_value", np.nan)),
            "learned_stage_components": list(requested_variant),
            "learned_trained_stage_components": list(
                result.summary.get("stage_components", [])
            ),
            "learned_reused_prefix": prefix is not None,
            "learned_prefix_variant": prefix_variant,
            "learned_prefix_run_slug": "" if prefix is None else str(prefix.run_slug),
            "learned_suffix_variant": suffix_variant,
            "final_f_checkpoint": final_f_checkpoint,
            "final_g_checkpoint": final_g_checkpoint,
            "final_leaf_adapter_checkpoint": final_leaf_adapter_checkpoint,
            "learned_embedding_dim": int(predict_cfg.extra.get("embedding_dim", 0)),
            "learned_summary_dim": int(predict_cfg.extra.get("summary_dim", 0)),
            "learned_state_dim": int(predict_cfg.extra.get("state_dim", 0)),
            "learned_hidden_dim": int(predict_cfg.extra.get("hidden_dim", 0)),
            "learned_leaf_width_floor": int(predict_cfg.extra.get("leaf_width_floor", 0)),
            "projection_kind": str(predict_cfg.extra.get("projection_kind", "mergeable_projection")),
            "state_space_kind": str(
                predict_cfg.extra.get("state_space_kind", "learned_projection_state")
            ),
            "merge_kind": str(
                predict_cfg.extra.get("merge_kind", "learned_projection_merge")
            ),
            "readout_kind": str(
                predict_cfg.extra.get("readout_kind", "learned_scalar_readout")
            ),
            "leaf_feature_mode": str(predict_cfg.extra.get("leaf_feature_mode", "")),
            "leaf_feature_dim": int(predict_cfg.extra.get("leaf_feature_dim", 0)),
            "g_input_dim": int(predict_cfg.extra.get("g_input_dim", 0)),
            "output_dim": 1,
        }
    )
    row.update(
        _learned_batch_metadata(
            config,
            actual_batch_size=int(stage_kwargs.get("train_batch_size", 0)),
        )
    )
    row = _attach_tree_bundle_contract(
        row,
        config=config,
        target=target,
        state_contract="bottom_up_g",
        f_init="official_oracle",
        g_init="raw_concat",
        schedule=str(requested_variant),
        final_f_checkpoint=final_f_checkpoint or None,
        final_g_checkpoint=final_g_checkpoint or None,
        final_leaf_adapter_checkpoint=final_leaf_adapter_checkpoint or None,
    )
    state = _LadderPrefix(
        variant=requested_variant,
        run_slug=str(target.run_slug or target.sketch),
        f_checkpoint=final_f_checkpoint or None,
        g_checkpoint=final_g_checkpoint or None,
        leaf_adapter_checkpoint=final_leaf_adapter_checkpoint or None,
    )
    return row, state


def _prefix_key(target: LearnedGridTarget) -> tuple[str, float, str]:
    return (
        str(target.target_kind),
        float(target.quantile_query),
        str(target.readout_arch or "structured"),
    )


def _chain_key(target: LearnedGridTarget) -> tuple[str, float]:
    return (str(target.target_kind), float(target.quantile_query))


def _longest_prefix(
    completed: dict[tuple[str, float, str], list[_LadderPrefix]],
    target: LearnedGridTarget,
) -> _LadderPrefix | None:
    variant = str(target.variant)
    key = _prefix_key(target)
    structured_key = (key[0], key[1], "structured")
    prefix_pool = list(completed.get(key, []))
    if str(target.readout_arch) != "structured":
        # A standalone exact-state g checkpoint is independent of readout
        # architecture. Let mlp readout stress rows reuse the structured g
        # anchor instead of retraining the identical first stage.
        prefix_pool.extend(
            state for state in completed.get(structured_key, []) if state.variant == "g"
        )
    if variant == "gfg":
        g_candidates = [
            state
            for state in prefix_pool
            if state.variant == "g"
        ]
        if g_candidates:
            return g_candidates[-1]
    candidates = [
        state
        for state in prefix_pool
        if variant.startswith(state.variant) and state.variant != variant
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda state: len(state.variant))


def _group_indexed_targets(
    targets: list[LearnedGridTarget],
) -> list[tuple[tuple[str, float], list[tuple[int, LearnedGridTarget]]]]:
    groups: dict[tuple[str, float], list[tuple[int, LearnedGridTarget]]] = {}
    order: list[tuple[str, float]] = []
    for index, target in enumerate(targets):
        key = _chain_key(target)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((index, target))
    return [(key, groups[key]) for key in order]


def _run_target_chain(
    config: ClassicalSketchComparisonConfig,
    *,
    output_dir: Path,
    indexed_targets: list[tuple[int, LearnedGridTarget]],
    chain_index: int,
    chain_count: int,
) -> list[tuple[int, dict[str, object]]]:
    completed_prefixes: dict[tuple[str, float, str], list[_LadderPrefix]] = {}
    rows: list[tuple[int, dict[str, object]]] = []
    for local_index, (target_index, target) in enumerate(indexed_targets, start=1):
        prefix = _longest_prefix(completed_prefixes, target)
        print(
            "[learned_sketch_grid] "
            f"chain={chain_index + 1}/{chain_count} "
            f"target_index={target_index + 1} local={local_index}/{len(indexed_targets)} "
            f"target={target.target_kind} variant={target.variant} "
            f"slug={target.run_slug or target.sketch}"
            + (
                ""
                if prefix is None
                else f" prefix={prefix.variant}:{prefix.run_slug}"
            )
            + f" cuda={config.learned_use_cuda}:{config.learned_cuda_device}",
            flush=True,
        )
        row, state = _run_target(config, target, output_dir=output_dir, prefix=prefix)
        rows.append((target_index, row))
        if state is not None:
            completed_prefixes.setdefault(_prefix_key(target), []).append(state)
        print(
            "[learned_sketch_grid] "
            f"chain={chain_index + 1}/{chain_count} "
            f"done slug={target.run_slug or target.sketch}",
            flush=True,
        )
    return rows


def _run_target_chain_worker(
    config_payload: dict[str, Any],
    output_dir: str,
    chain_specs: list[tuple[int, list[tuple[int, LearnedGridTarget]]]],
    chain_count: int,
    cuda_device: int | None,
) -> list[tuple[int, dict[str, object]]]:
    config = ClassicalSketchComparisonConfig(**dict(config_payload))
    if cuda_device is None:
        config = replace(config, learned_use_cuda=False, learned_cuda_device=None)
    else:
        config = replace(config, learned_use_cuda=True, learned_cuda_device=int(cuda_device))
    rows: list[tuple[int, dict[str, object]]] = []
    for chain_index, indexed_targets in chain_specs:
        rows.extend(
            _run_target_chain(
                config,
                output_dir=Path(output_dir),
                indexed_targets=indexed_targets,
                chain_index=int(chain_index),
                chain_count=int(chain_count),
            )
        )
    return rows


def _run_target_chain_queue_worker(
    config_payload: dict[str, Any],
    output_dir: str,
    chain_queue: Any,
    chain_count: int,
    cuda_device: int | None,
) -> list[tuple[int, dict[str, object]]]:
    config = ClassicalSketchComparisonConfig(**dict(config_payload))
    if cuda_device is None:
        config = replace(config, learned_use_cuda=False, learned_cuda_device=None)
    else:
        config = replace(config, learned_use_cuda=True, learned_cuda_device=int(cuda_device))
    rows: list[tuple[int, dict[str, object]]] = []
    while True:
        try:
            chain_index, indexed_targets = chain_queue.get_nowait()
        except Empty:
            break
        rows.extend(
            _run_target_chain(
                config,
                output_dir=Path(output_dir),
                indexed_targets=indexed_targets,
                chain_index=int(chain_index),
                chain_count=int(chain_count),
            )
        )
    return rows


def run_learned_sketch_grid(
    config: ClassicalSketchComparisonConfig,
    *,
    output_dir: str | Path,
) -> list[dict[str, object]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = learned_grid_targets(config)
    gpu_ids = (
        _parse_learned_gpu_ids(getattr(config, "learned_gpu_ids", "auto"))
        if bool(getattr(config, "learned_use_cuda", False))
        else ()
    )
    target_jobs = _resolve_learned_target_jobs(config, gpu_ids=gpu_ids)
    target_jobs = min(max(1, int(target_jobs)), max(1, len(_group_indexed_targets(targets))))
    resolved_gpu_label = ",".join(str(x) for x in gpu_ids) if gpu_ids else "cpu"
    config = replace(
        config,
        learned_target_jobs=int(target_jobs),
        learned_gpu_ids=resolved_gpu_label,
        learned_use_cuda=bool(gpu_ids),
    )
    effective_batch = int(_effective_learned_batch_size(config))
    print(
        "[learned_sketch_grid] "
        f"seed={config.seed} capacity={config.capacity_label} "
        f"leaf_axis={'leaf_size' if config.n_leaves is None else 'n_leaves'} "
        f"leaf_size={config.leaf_size} n_leaves={config.n_leaves} "
        f"targets={len(targets)} batch={effective_batch} "
        f"base_batch={config.learned_batch_size} "
        f"ref_leaf={config.learned_batch_reference_leaf_size} "
        f"target_jobs={target_jobs} gpu_ids={resolved_gpu_label} "
        f"train={config.learned_n_train} val={config.learned_n_val} "
        f"root_rate={getattr(config, 'learned_root_query_rate', 1.0)} "
        f"leaf_rate={getattr(config, 'learned_leaf_query_rate', 1.0)} "
        f"internal_rate={getattr(config, 'learned_internal_query_rate', 1.0)} "
        f"sampling_policy={getattr(config, 'learned_supervision_sampling_policy', 'separate_axes')} "
        f"cuda={config.learned_use_cuda}",
        flush=True,
    )
    chains = _group_indexed_targets(targets)
    indexed_rows: list[tuple[int, dict[str, object]]] = []
    if target_jobs <= 1:
        indexed_rows = _run_target_chain(
            config,
            output_dir=out_dir,
            indexed_targets=[item for _key, chain in chains for item in chain],
            chain_index=0,
            chain_count=1,
        )
    else:
        # Use spawn for both CUDA and CPU workers. Forking after PyTorch or
        # API-client imports have started helper threads can deadlock in the
        # test runner, and CUDA also requires a non-fork start method.
        ctx = mp.get_context("spawn")
        with ctx.Manager() as manager, ProcessPoolExecutor(max_workers=target_jobs, mp_context=ctx) as pool:
            chain_queue = manager.Queue()
            config_payload = asdict(config)
            for chain_index, (_key, chain) in enumerate(chains):
                chain_queue.put((chain_index, chain))
            futures = []
            for worker_index in range(target_jobs):
                cuda_device = (
                    int(gpu_ids[worker_index % len(gpu_ids)])
                    if gpu_ids
                    else None
                )
                futures.append(
                    pool.submit(
                        _run_target_chain_queue_worker,
                        config_payload,
                        str(out_dir),
                        chain_queue,
                        len(chains),
                        cuda_device,
                    )
                )
            for future in as_completed(futures):
                indexed_rows.extend(future.result())
    indexed_rows.sort(key=lambda item: item[0])
    return [row for _index, row in indexed_rows]


__all__ = [
    "LearnedGridTarget",
    "learned_grid_targets",
    "run_learned_sketch_grid",
]
