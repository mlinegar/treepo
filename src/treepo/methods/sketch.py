"""Classical sketch family for the unified methods loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ClassicalSketchFamilyConfig:
    sketch: str = "hll"
    backend: str = "datasketches"
    precision: int = 14
    hash_bits: int = 64
    schedule: str = "balanced"


class ClassicalSketchFamily:
    """Wrap a mergeable sketch as a minimal ``FamilyRuntime``."""

    name = "classical_sketch"

    def __init__(self, config: ClassicalSketchFamilyConfig) -> None:
        self.config = config

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        return {
            "kind": "treepo_classical_sketch_f",
            "trained": "f",
            "iteration": int(iteration),
            "config": asdict(self.config),
        }

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        return {
            "kind": "treepo_classical_sketch_g",
            "trained": "g",
            "iteration": int(iteration),
            "config": asdict(self.config),
        }

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[float | None]:
        adapter = _make_adapter(self.config)
        from treepo.bench.sketches.tree_reducer import treepo_reduce

        out: list[float | None] = []
        for tree in trees:
            leaf_items = _leaf_items(tree)
            if not leaf_items:
                out.append(None)
                continue
            root_state = treepo_reduce(leaf_items, adapter, schedule=self.config.schedule)
            out.append(float(adapter.query(root_state, None)))
        return out

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if artifact is None:
            return
        if not isinstance(artifact, Mapping):
            raise TypeError(f"classical_sketch {kind} artifact must be a mapping")


def build_classical_sketch_family(backend_config: Mapping[str, Any]) -> ClassicalSketchFamily:
    return ClassicalSketchFamily(
        ClassicalSketchFamilyConfig(
            sketch=str(backend_config.get("sketch", "hll")),
            backend=str(backend_config.get("backend", "datasketches")),
            precision=int(backend_config.get("precision", 14)),
            hash_bits=int(backend_config.get("hash_bits", 64)),
            schedule=str(backend_config.get("schedule", "balanced")),
        )
    )


def _make_adapter(config: ClassicalSketchFamilyConfig) -> Any:
    if config.sketch != "hll":
        raise ValueError("classical_sketch currently supports sketch='hll'")
    from treepo.bench.sketches.adapters import make_hll_adapter

    return make_hll_adapter(
        backend=config.backend,  # type: ignore[arg-type]
        precision=int(config.precision),
        hash_bits=int(config.hash_bits),
    )


def _leaf_items(tree: Any) -> list[list[Any]]:
    leaves = getattr(tree, "leaves", None)
    if not leaves and callable(getattr(tree, "get_leaves", None)):
        leaves = tree.get_leaves()
    groups: list[list[Any]] = []
    for leaf in list(leaves or []):
        tokens = getattr(leaf, "tokens", None)
        if tokens is None and isinstance(leaf, Mapping):
            tokens = leaf.get("tokens")
        if tokens is not None:
            groups.append(list(tokens))
    if groups:
        return groups
    tokens = getattr(tree, "tokens", None)
    return [list(tokens)] if tokens is not None else []


__all__ = [
    "ClassicalSketchFamily",
    "ClassicalSketchFamilyConfig",
    "build_classical_sketch_family",
]
