"""Classical sketch family for the unified methods loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.local_law import LawKind, LocalLawAuditRow
from treepo.statistic import StatisticInfo
from treepo.tree import tree_leaves, tree_row_id


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

    def as_statistic(self, *, f: Any = None, g: Any = None) -> Any:
        del f, g
        return ClassicalSketchStatistic(config=self.config, adapter=_make_adapter(self.config))


class ClassicalSketchStatistic:
    """ComposableStatistic wrapper around a ``SketchAdapter``."""

    def __init__(self, *, config: ClassicalSketchFamilyConfig, adapter: Any) -> None:
        self.config = config
        self.adapter = adapter
        self.info = StatisticInfo(
            name=f"classical_sketch:{config.sketch}",
            state_kind=str(getattr(adapter, "name", config.sketch)),
            exact=True,
            supports_local_laws=True,
            metadata={
                "config": asdict(config),
                "adapter_config": dict(getattr(adapter, "config", {}) or {}),
                "is_associative": bool(getattr(adapter, "is_associative", False)),
                "is_commutative": bool(getattr(adapter, "is_commutative", False)),
                "is_idempotent": bool(getattr(adapter, "is_idempotent", False)),
                "is_byte_deterministic": bool(getattr(adapter, "is_byte_deterministic", False)),
            },
        )

    def encode_leaf(self, leaf: Any) -> Any:
        return self.adapter.encode(_leaf_tokens(leaf))

    def merge(self, left: Any, right: Any) -> Any:
        return self.adapter.merge(left, right)

    def readout(self, state: Any, query: Any = None) -> Any:
        return self.adapter.query(state, query)

    def state_equal(self, left: Any, right: Any) -> bool:
        return bool(self.adapter.state_equal(left, right))

    def serialize(self, state: Any) -> bytes:
        return self.adapter.serialize(state)

    def memory_bytes(self, state: Any) -> float:
        return float(self.adapter.memory_bytes(state))

    def encode_tree(self, tree: Any, *, schedule: str | None = None) -> Any:
        from treepo.bench.sketches.tree_reducer import treepo_reduce

        return treepo_reduce(_leaf_items(tree), self.adapter, schedule=str(schedule or self.config.schedule))

    def local_law_rows(
        self,
        units: Sequence[Any],
        *,
        query: Any = None,
        oracle: Any = None,
    ) -> Sequence[LocalLawAuditRow]:
        from treepo.bench.sketches.tree_reducer import fold_states

        rows: list[LocalLawAuditRow] = []
        for idx, unit in enumerate(list(units or ())):
            leaf_items = _leaf_items(unit)
            if not leaf_items:
                continue
            leaf_states = [self.adapter.encode(items) for items in leaf_items]
            root = fold_states(leaf_states, self.adapter, schedule=self.config.schedule)
            row_prefix = _unit_row_prefix(unit, idx)
            if bool(getattr(self.adapter, "is_associative", False)):
                left_root = fold_states(leaf_states, self.adapter, schedule="left_to_right")
                rows.append(
                    _state_law_row(
                        row_id=f"{row_prefix}:schedule:left_to_right",
                        law_kind=LawKind.C3_MERGE,
                        loss=0.0 if self.adapter.state_equal(root, left_root) else 1.0,
                        metadata={
                            "statistic": self.info.name,
                            "check": "schedule_invariance",
                            "schedule": "left_to_right",
                            "law_facet": "c3b_compositionality",
                        },
                    )
                )
            if bool(getattr(self.adapter, "is_commutative", False)):
                right_root = fold_states(leaf_states, self.adapter, schedule="right_to_left")
                rows.append(
                    _state_law_row(
                        row_id=f"{row_prefix}:schedule:right_to_left",
                        law_kind=LawKind.C3_MERGE,
                        loss=0.0 if self.adapter.state_equal(root, right_root) else 1.0,
                        metadata={
                            "statistic": self.info.name,
                            "check": "schedule_invariance",
                            "schedule": "right_to_left",
                            "law_facet": "c3b_compositionality",
                        },
                    )
                )
            if bool(getattr(self.adapter, "is_idempotent", False)):
                rows.append(
                    _state_law_row(
                        row_id=f"{row_prefix}:idempotence",
                        law_kind=LawKind.C2_IDEMPOTENCE,
                        loss=0.0 if self.adapter.state_equal(root, self.adapter.merge(root, root)) else 1.0,
                        metadata={
                            "statistic": self.info.name,
                            "check": "idempotence",
                            "law_facet": "c2_idempotence",
                        },
                    )
                )
            target = _oracle_value(oracle, unit, idx)
            if target is not None:
                try:
                    prediction = float(self.adapter.query(root, query))
                    truth = float(target)
                except (TypeError, ValueError):
                    prediction = truth = None
                if prediction is not None and truth is not None:
                    loss = float((prediction - truth) ** 2)
                    rows.append(
                        _state_law_row(
                            row_id=f"{row_prefix}:readout",
                            law_kind=LawKind.C1_LEAF,
                            loss=loss,
                            metadata={
                                "statistic": self.info.name,
                                "check": "readout_oracle_agreement",
                                # Exact merges make root-readout agreement a
                                # certificate of leaf sufficiency (C1).
                                "law_facet": "c1_sufficiency_via_exact_merges",
                                "prediction": prediction,
                                "target": truth,
                            },
                        )
                    )
        return tuple(rows)


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
    # Canonical leaf extraction: also handles TreeRecord's callable leaves().
    leaves = tree_leaves(tree)
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


def _leaf_tokens(leaf: Any) -> list[Any]:
    tokens = getattr(leaf, "tokens", None)
    if tokens is None and isinstance(leaf, Mapping):
        tokens = leaf.get("tokens")
    return list(tokens or [])


def _unit_row_prefix(unit: Any, idx: int) -> str:
    return tree_row_id(unit, idx, fallback_prefix="unit")


def _state_law_row(
    *,
    row_id: str,
    law_kind: LawKind,
    loss: float,
    metadata: Mapping[str, Any],
) -> LocalLawAuditRow:
    value = float(loss)
    return LocalLawAuditRow(
        row_id=row_id,
        law_kind=law_kind,
        proxy_loss=value,
        oracle_loss=value,
        observed=True,
        propensity=1.0,
        metadata=dict(metadata or {}),
    )


def _oracle_value(oracle: Any, unit: Any, idx: int) -> Any:
    if oracle is None:
        return None
    if callable(oracle):
        try:
            return oracle(unit)
        except TypeError:
            return oracle(unit, idx)
    if isinstance(oracle, Mapping):
        metadata = getattr(unit, "metadata", None)
        keys = [idx, str(idx)]
        if isinstance(metadata, Mapping):
            keys.extend(
                value
                for value in (
                    metadata.get("tree_id"),
                    metadata.get("doc_id"),
                    metadata.get("unit_id"),
                )
                if value is not None
            )
        for key in keys:
            if key in oracle:
                return oracle[key]
    return None


__all__ = [
    "ClassicalSketchFamily",
    "ClassicalSketchFamilyConfig",
    "ClassicalSketchStatistic",
    "build_classical_sketch_family",
]
