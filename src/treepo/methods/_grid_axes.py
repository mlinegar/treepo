"""First-class grid axes for ``treepo.fit``: doc-gold-n, local-label-mix, seed.

Phase 3 of ``docs/treepo_fit_grid_upgrade_plan_2026_07_10.md`` promotes three
supervision-grid knobs that ThinkingTrees expresses across three separate
scripts today into validated spec fields with pinned, deterministic selection
and persisted provenance:

* ``doc_gold_n`` — how many documents contribute gold document-level labels.
  Selection is drawn from a single per-seed permutation and taken as a prefix,
  so cells at increasing ``n`` are nested (``n=25 ⊂ n=50 ⊂ n=100``) and every
  cell at the same ``(seed, n)`` trains on the identical subset. The selected
  ids are persisted so the split is pinned, never resampled.
* ``local_label_mix`` — one enum (``none`` / ``llm_distilled`` /
  ``gold_fraction``) unifying node-level supervision. ``gold_fraction`` keeps
  gold node labels on a deterministic ``p``-fraction of nodes (pinned like the
  document split); ``llm_distilled`` routes to a cached-teacher predictor.
* ``seed`` — one seed per ``fit()`` call; the grid expander crosses a
  ``seeds`` list into one fully specified cell per combination.

The per-node loss machinery that consumes ``gold_fraction`` / ``llm_distilled``
node labels lands in Phases 1-2; this module establishes the validated axis
surface, the pinned selection, and the provenance now.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping, Sequence

LOCAL_LABEL_MIXES: tuple[str, ...] = ("none", "llm_distilled", "gold_fraction")
DEFAULT_LOCAL_LABEL_MIX = "none"

# backend_config keys that can carry the cached-teacher node predictor used by
# the ``llm_distilled`` mix.
NODE_PREDICTOR_KEYS: tuple[str, ...] = ("node_oracle_predictor", "predict_fn")


@dataclass(frozen=True)
class GridAxes:
    """Validated view of the three first-class grid axes for one ``fit`` cell."""

    doc_gold_n: int | None = None
    local_label_mix: str = DEFAULT_LOCAL_LABEL_MIX
    gold_fraction_p: float = 1.0
    distilled_labels_path: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.doc_gold_n is not None and int(self.doc_gold_n) < 0:
            raise ValueError(
                f"doc_gold_n must be a non-negative int or None, got {self.doc_gold_n!r}"
            )
        if self.local_label_mix not in LOCAL_LABEL_MIXES:
            raise ValueError(
                "local_label_mix must be one of "
                f"{LOCAL_LABEL_MIXES}, got {self.local_label_mix!r}"
            )
        p = float(self.gold_fraction_p)
        if p < 0.0 or p > 1.0:
            raise ValueError(f"gold_fraction_p must be in [0, 1], got {self.gold_fraction_p!r}")

    @classmethod
    def from_spec(cls, spec: Any) -> "GridAxes":
        doc_gold_n = getattr(spec, "doc_gold_n", None)
        return cls(
            doc_gold_n=None if doc_gold_n is None else int(doc_gold_n),
            local_label_mix=str(getattr(spec, "local_label_mix", None) or DEFAULT_LOCAL_LABEL_MIX),
            gold_fraction_p=float(
                getattr(spec, "gold_fraction_p", 1.0)
                if getattr(spec, "gold_fraction_p", None) is not None
                else 1.0
            ),
            distilled_labels_path=(
                None
                if getattr(spec, "distilled_labels_path", None) is None
                else str(spec.distilled_labels_path)
            ),
            seed=int(getattr(spec, "seed", 0) or 0),
        )


def tree_doc_id(tree: Any, index: int) -> str:
    """Stable document id for an arbitrary train trace/tree."""
    for attr in ("doc_id", "tree_id", "id"):
        value = getattr(tree, attr, None)
        if value is not None:
            return str(value)
    if isinstance(tree, Mapping):
        for key in ("doc_id", "tree_id", "id"):
            if tree.get(key) is not None:
                return str(tree[key])
    return f"index_{int(index)}"


def tree_node_units(tree: Any, index: int) -> list[str]:
    """Stable per-node unit ids (``doc_id::node``) for one tree's leaves."""
    doc_id = tree_doc_id(tree, index)
    leaves = getattr(tree, "leaves", None)
    if leaves is None and isinstance(tree, Mapping):
        leaves = tree.get("leaves")
    units: list[str] = []
    for leaf_idx, leaf in enumerate(list(leaves or ())):
        node_id = (
            getattr(leaf, "qid", None)
            or getattr(leaf, "node_id", None)
            or getattr(leaf, "id", None)
        )
        if node_id is None and isinstance(leaf, Mapping):
            node_id = leaf.get("qid") or leaf.get("node_id") or leaf.get("id")
        if node_id is None:
            node_id = f"leaf_{leaf_idx}"
        units.append(f"{doc_id}::{node_id}")
    return units


def select_gold_docs(
    traces: Sequence[Any],
    *,
    doc_gold_n: int | None,
    seed: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Pin a nested prefix of documents whose gold labels are used.

    A single permutation is drawn per ``seed`` and taken as a prefix, so
    ``doc_gold_n=25 ⊂ 50 ⊂ 100`` for a fixed seed. ``doc_gold_n=None`` keeps the
    full population (today's default). Returns the selected traces and a
    provenance record carrying the pinned selected doc ids.
    """
    population = list(traces or ())
    doc_ids = [tree_doc_id(tree, idx) for idx, tree in enumerate(population)]
    n_total = len(population)

    if doc_gold_n is None or int(doc_gold_n) >= n_total:
        selected = list(population)
        selected_ids = list(doc_ids)
        prefix_nested = False
    else:
        k = max(0, int(doc_gold_n))
        order = list(range(n_total))
        random.Random(int(seed)).shuffle(order)
        prefix = order[:k]
        selected = [population[i] for i in prefix]
        selected_ids = [doc_ids[i] for i in prefix]
        prefix_nested = True

    provenance = {
        "doc_gold_n": None if doc_gold_n is None else int(doc_gold_n),
        "seed": int(seed),
        "population_size": n_total,
        "selected_count": len(selected_ids),
        "selected_doc_ids": selected_ids,
        "nested_prefix": prefix_nested,
    }
    return selected, provenance


def _resolve_node_predictor(backend_config: Mapping[str, Any]) -> Callable[..., Any] | None:
    for key in NODE_PREDICTOR_KEYS:
        candidate = backend_config.get(key)
        if callable(candidate):
            return candidate
    return None


def resolve_label_mix(
    traces: Sequence[Any],
    *,
    axes: GridAxes,
    backend_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the node-label mix and pin its selection/source.

    * ``none`` — root-only supervision; no node labels.
    * ``gold_fraction`` — keep gold node labels on a deterministic ``p``-fraction
      of nodes (pinned per seed like the document split).
    * ``llm_distilled`` — resolve a cached-teacher node source; error with a
      clear message when none is configured (the cached-jsonl loader itself is
      Phase 2).
    """
    mix = axes.local_label_mix
    provenance: dict[str, Any] = {"mix": mix, "seed": int(axes.seed)}

    if mix == "none":
        provenance["node_labels"] = "root_only"
        return provenance

    if mix == "gold_fraction":
        units: list[str] = []
        for idx, tree in enumerate(list(traces or ())):
            units.extend(tree_node_units(tree, idx))
        n_nodes = len(units)
        p = float(axes.gold_fraction_p)
        if p >= 1.0:
            k = n_nodes
        elif p <= 0.0:
            k = 0
        else:
            k = max(1, int(round(p * n_nodes)))
        order = list(range(n_nodes))
        random.Random(int(axes.seed)).shuffle(order)
        chosen = sorted(order[:k])
        provenance.update(
            {
                "gold_fraction_p": p,
                "population_node_count": n_nodes,
                "selected_node_count": len(chosen),
                "selected_node_units": [units[i] for i in chosen],
                "state_source": "f_states",  # gold-leak guard (project memory)
            }
        )
        return provenance

    # mix == "llm_distilled"
    predictor = _resolve_node_predictor(backend_config)
    labels_path = axes.distilled_labels_path
    if predictor is None and not labels_path:
        raise ValueError(
            "local_label_mix='llm_distilled' needs a cached-teacher node source: "
            "set spec.distilled_labels_path to a teacher_node_rows.jsonl path, or "
            "provide a callable in backend_config['node_oracle_predictor'] "
            "(or backend_config['predict_fn']). The cached-jsonl loader is Phase 2 "
            "of the fit-grid upgrade plan; wire one of these to route distilled "
            "node labels through fit()."
        )
    if labels_path:
        provenance.update({"labels_source": "cached_jsonl", "distilled_labels_path": str(labels_path)})
    else:
        provenance.update(
            {
                "labels_source": "node_oracle_predictor",
                "predictor": getattr(predictor, "__name__", type(predictor).__name__),
            }
        )
    return provenance


def apply_grid_axes(
    traces: Sequence[Any],
    *,
    axes: GridAxes,
    backend_config: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Apply the doc-gold and label-mix axes; return traces + full provenance."""
    selected, doc_provenance = select_gold_docs(
        traces, doc_gold_n=axes.doc_gold_n, seed=axes.seed
    )
    label_provenance = resolve_label_mix(
        selected, axes=axes, backend_config=backend_config
    )
    provenance = {
        "seed": int(axes.seed),
        "doc_gold": doc_provenance,
        "local_label_mix": label_provenance,
    }
    return selected, provenance


def expand_grid_cells(
    *,
    seeds: Sequence[int] = (0,),
    doc_gold_ns: Sequence[int | None] = (None,),
    local_label_mixes: Sequence[str] = (DEFAULT_LOCAL_LABEL_MIX,),
    leaf_unit_counts: Sequence[int] = (1,),
    gold_fraction_p: float = 1.0,
    base: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Cartesian expansion of the grid axes into fully specified cells.

    One cell per ``(seed, doc_gold_n, local_label_mix, leaf_unit_count)``
    combination. Every mix name is validated up front so a bad axis fails at
    expansion time rather than mid-run. ``gold_fraction_p`` is attached only to
    ``gold_fraction`` cells.
    """
    mixes = tuple(str(m) for m in (local_label_mixes or (DEFAULT_LOCAL_LABEL_MIX,)))
    for mix in mixes:
        if mix not in LOCAL_LABEL_MIXES:
            raise ValueError(
                f"local_label_mix must be one of {LOCAL_LABEL_MIXES}, got {mix!r}"
            )
    base_cell = dict(base or {})
    cells: list[dict[str, Any]] = []
    for seed, doc_gold_n, mix, leaf_count in product(
        tuple(seeds or (0,)),
        tuple(doc_gold_ns or (None,)),
        mixes,
        tuple(leaf_unit_counts or (1,)),
    ):
        cell = dict(base_cell)
        cell.update(
            {
                "seed": int(seed),
                "doc_gold_n": None if doc_gold_n is None else int(doc_gold_n),
                "local_label_mix": str(mix),
                "leaf_unit_count": max(1, int(leaf_count or 1)),
            }
        )
        if mix == "gold_fraction":
            cell["gold_fraction_p"] = float(gold_fraction_p)
        cells.append(cell)
    return tuple(cells)


__all__ = [
    "DEFAULT_LOCAL_LABEL_MIX",
    "GridAxes",
    "LOCAL_LABEL_MIXES",
    "NODE_PREDICTOR_KEYS",
    "apply_grid_axes",
    "expand_grid_cells",
    "resolve_label_mix",
    "select_gold_docs",
    "tree_doc_id",
    "tree_node_units",
]
