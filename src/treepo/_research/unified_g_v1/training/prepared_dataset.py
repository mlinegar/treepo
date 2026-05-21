from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASET_MANIFEST_VERSION = 1
MANIFEST_FILENAME = "dataset_manifest.json"

_SPACE_KINDS = frozenset({
    "embedding_sequence",
    "text",
    "token_id_sequence",
    "preference_pairs",
})

_PAYLOAD_SCHEMAS = frozenset({
    "embedding_tree_v1",
    "text_pairs_v1",
    "preference_pairs_v1",
    "token_id_sequence_v1",
})


@dataclass(frozen=True)
class PreparedDataset:
    """Lightweight wrapper over a prepared-dataset manifest on disk.

    The on-disk layout is:

        prepared_dataset_root/
          dataset_manifest.json
          splits/split_ids.json       (optional)
          shards/<split>-NNNN.{pt,jsonl}

    Training backends read shard paths through `shard_paths(split)` and are
    free to interpret them according to `payload_schema`.
    """

    root: Path
    manifest: Mapping[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> "PreparedDataset":
        root_path = Path(root).expanduser()
        manifest_path = root_path / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing {MANIFEST_FILENAME} at {root_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(root=root_path, manifest=data)

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def space_kind(self) -> str:
        return str(self.manifest.get("space_kind", ""))

    @property
    def payload_schema(self) -> str:
        return str(self.manifest.get("payload_schema", ""))

    @property
    def split_ids_path(self) -> Path | None:
        rel = self.manifest.get("split_ids_path")
        if not rel:
            return None
        return (self.root / str(rel)).resolve()

    def shard_paths(self, split: str) -> list[Path]:
        shards = self.manifest.get("shards") or {}
        entries = shards.get(split) or []
        return [(self.root / str(entry)).resolve() for entry in entries]

    def has_split(self, split: str) -> bool:
        return bool(self.shard_paths(split))

    def feature_dim(self) -> int | None:
        value = self.manifest.get("feature_dim")
        return None if value is None else int(value)

    def stats(self) -> Mapping[str, Any]:
        return dict(self.manifest.get("stats") or {})

    def provenance(self) -> Mapping[str, Any]:
        return dict(self.manifest.get("provenance") or {})


def write_dataset_manifest(
    root: str | Path,
    *,
    space_kind: str,
    payload_schema: str,
    shards: Mapping[str, Sequence[str]],
    split_ids_path: str | None = None,
    feature_dim: int | None = None,
    tokenizer_or_adapter_id: str | None = None,
    stats: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Emit a canonical prepared-dataset manifest at <root>/dataset_manifest.json."""
    if space_kind not in _SPACE_KINDS:
        raise ValueError(f"unknown space_kind={space_kind!r}")
    if payload_schema not in _PAYLOAD_SCHEMAS:
        raise ValueError(f"unknown payload_schema={payload_schema!r}")
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": DATASET_MANIFEST_VERSION,
        "space_kind": str(space_kind),
        "payload_schema": str(payload_schema),
        "shards": {str(k): [str(p) for p in v] for k, v in shards.items()},
    }
    if split_ids_path is not None:
        manifest["split_ids_path"] = str(split_ids_path)
    if feature_dim is not None:
        manifest["feature_dim"] = int(feature_dim)
    if tokenizer_or_adapter_id is not None:
        manifest["tokenizer_or_adapter_id"] = str(tokenizer_or_adapter_id)
    if stats is not None:
        manifest["stats"] = dict(stats)
    if provenance is not None:
        manifest["provenance"] = dict(provenance)
    manifest_path = root_path / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
