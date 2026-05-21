from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from treepo.manifest import RoleTuple


@dataclass(frozen=True)
class ThreeLayerHonestyConfig:
    enabled: bool = False
    split_seed: int = 23
    chunk_train_fraction: float = 0.5
    summarizer_train_fraction: float = 0.5
    oracle_train_fraction: float = 0.5
    train_role: str = "train"
    eval_role: str = "eval"


@dataclass(frozen=True)
class HonestChunkingPolicy:
    enabled: bool = False
    boundary_fraction: float = 0.5
    split_seed: int = 17
    boundary_role: str = "boundary"
    evaluation_role: str = "evaluation"


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def stable_unit_interval(sample_id: str, *, seed: int, salt: str = "") -> float:
    payload = f"{int(seed)}:{salt}:{sample_id}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def assign_three_layer_split(sample_id: str, layer: str, cfg: ThreeLayerHonestyConfig) -> str:
    if not cfg.enabled:
        return "all"
    fraction_by_layer = {
        "chunk": cfg.chunk_train_fraction,
        "chunker": cfg.chunk_train_fraction,
        "summarizer": cfg.summarizer_train_fraction,
        "g": cfg.summarizer_train_fraction,
        "oracle": cfg.oracle_train_fraction,
    }
    train_fraction = _clamp_unit(fraction_by_layer.get(str(layer), 0.5))
    u = stable_unit_interval(sample_id, seed=cfg.split_seed, salt=f"three_layer:{layer}")
    return cfg.train_role if u < train_fraction else cfg.eval_role


def assign_three_layer_roles(sample_id: str, cfg: ThreeLayerHonestyConfig) -> dict[str, str]:
    if not cfg.enabled:
        return {"chunk": "all", "summarizer": "all", "oracle": "all"}
    return {
        "chunk": assign_three_layer_split(sample_id, "chunk", cfg),
        "summarizer": assign_three_layer_split(sample_id, "summarizer", cfg),
        "oracle": assign_three_layer_split(sample_id, "oracle", cfg),
    }


def role_tuple_for_unit(sample_id: str, cfg: ThreeLayerHonestyConfig) -> RoleTuple:
    roles = assign_three_layer_roles(sample_id, cfg)
    return RoleTuple(
        chunker=roles["chunk"],
        g=roles["summarizer"],
        oracle=roles["oracle"],
    )


def assign_honest_split(sample_id: str, policy: HonestChunkingPolicy | None = None) -> str:
    if policy is None or not policy.enabled:
        return "all"
    draw = stable_unit_interval(sample_id, seed=policy.split_seed)
    return policy.boundary_role if draw < _clamp_unit(policy.boundary_fraction) else policy.evaluation_role


def _extract_unit_id(item: Any, fallback: str) -> str:
    for name in ("top_level_unit_id", "doc_id", "source_doc_id", "example_id", "id"):
        value = getattr(item, name, None)
        if value is not None:
            return str(value)
    if isinstance(item, Mapping):
        for name in ("top_level_unit_id", "doc_id", "source_doc_id", "example_id", "id"):
            if item.get(name) is not None:
                return str(item[name])
    return str(fallback)


def filter_items_by_three_layer_role(
    items: Sequence[Any],
    cfg: ThreeLayerHonestyConfig,
    *,
    layer: str,
    role: str,
) -> list[Any]:
    if not cfg.enabled:
        return list(items)
    out: list[Any] = []
    for idx, item in enumerate(items):
        unit_id = _extract_unit_id(item, fallback=f"{layer}_{idx}")
        if assign_three_layer_split(unit_id, layer, cfg) == role:
            out.append(item)
    return out


__all__ = [
    "HonestChunkingPolicy",
    "ThreeLayerHonestyConfig",
    "assign_honest_split",
    "assign_three_layer_roles",
    "assign_three_layer_split",
    "filter_items_by_three_layer_role",
    "role_tuple_for_unit",
    "stable_unit_interval",
]
