"""FamilyRuntime wrapper for any ``treepo.bench.sketches.protocol.SketchAdapter``.

Classical sketches (HLL, Count-Min, Theta, KLL, ...) are fixed by
construction: f = ``adapter.query(root_state, None)``, merge =
``adapter.merge``, leaf encoding = ``adapter.encode``. Training is a
no-op. This wrapper exposes that pipeline through the
:class:`FamilyRuntime` contract so it dispatches through
:func:`treepo.methods.fit` just like learned families.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

from treepo.bench.sketches.protocol import SketchAdapter
from treepo.bench.sketches.tree_reducer import treepo_reduce


class ClassicalSketchFamilyRuntime:
    """Wrap a :class:`SketchAdapter` as a :class:`FamilyRuntime`.

    ``leaf_items_fn(tree)`` must return one iterable of items per leaf.
    The default implementation looks for ``tree.leaves`` (each leaf
    exposing ``.tokens`` or ``.items``) and falls back to top-level
    ``tree.tokens`` / ``tree.items`` for single-leaf trees.

    Schedules pass through to :func:`treepo_reduce` and must be one of
    ``"balanced"`` / ``"left_to_right"`` / ``"right_to_left"``.
    """

    def __init__(
        self,
        *,
        adapter: SketchAdapter,
        schedule: str = "balanced",
        leaf_items_fn: Optional[Callable[[Any], Sequence[Iterable[Any]]]] = None,
    ) -> None:
        self._adapter = adapter
        self._schedule = schedule
        self._leaf_items_fn = leaf_items_fn or _default_leaf_items
        self._name = f"sketch:{getattr(adapter, 'name', adapter.__class__.__name__)}"

    @property
    def name(self) -> str:
        return self._name

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        return f_init if f_init is not None else self._name

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        return g_init if g_init is not None else self._name

    def score_roots_with_f(self, *, f, g, trees) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for tree in trees:
            leaves_list = [list(items) for items in self._leaf_items_fn(tree)]
            if not leaves_list:
                out.append(None)
                continue
            root_state = treepo_reduce(leaves_list, self._adapter, schedule=self._schedule)
            try:
                out.append(float(self._adapter.query(root_state, None)))
            except (TypeError, ValueError):
                out.append(None)
        return out

    def validate_artifact(self, *, kind, artifact) -> None:
        return None


def _default_leaf_items(tree: Any) -> Sequence[Iterable[Any]]:
    """Default leaf-items extractor. Tries ``tree.leaves[i].tokens|.items``,
    then top-level ``tree.tokens|.items``. Raises if neither is present —
    callers with custom tree shapes pass their own ``leaf_items_fn``.
    """
    leaves = getattr(tree, "leaves", None)
    if leaves is not None:
        return [
            getattr(leaf, "tokens", None) or getattr(leaf, "items", None) or leaf
            for leaf in leaves
        ]
    items = getattr(tree, "tokens", None) or getattr(tree, "items", None)
    if items is not None:
        return [items]
    raise AttributeError(
        "ClassicalSketchFamilyRuntime default leaf_items_fn expects "
        "tree.leaves with .tokens/.items or top-level tree.tokens/.items; "
        "supply backend_config['leaf_items_fn'] for custom tree shapes."
    )


__all__ = ["ClassicalSketchFamilyRuntime"]
