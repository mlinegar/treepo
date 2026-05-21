from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DocumentSplitIds:
    train_doc_ids: tuple[str, ...]
    val_doc_ids: tuple[str, ...]
    test_doc_ids: tuple[str, ...] = tuple()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train_doc_ids),
            "val": list(self.val_doc_ids),
            "test": list(self.test_doc_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DocumentSplitIds":
        def _coerce(key: str) -> tuple[str, ...]:
            raw = payload.get(key, []) or []
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"split ids field {key!r} must be a JSON list")
            out: list[str] = []
            for item in raw:
                if item is None:
                    continue
                rendered = str(item).strip()
                if rendered:
                    out.append(rendered)
            return tuple(out)

        return cls(
            train_doc_ids=_coerce("train"),
            val_doc_ids=_coerce("val"),
            test_doc_ids=_coerce("test"),
        )


def load_phase1_split_ids(phase1_data_path: str | Path) -> DocumentSplitIds:
    payload = pickle.loads(Path(phase1_data_path).expanduser().read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("phase1_data.pkl must contain a dict payload")

    def _collect(key: str) -> tuple[str, ...]:
        out: list[str] = []
        for item in list(payload.get(key) or []):
            doc_id = str(getattr(item, "doc_id", "") or "").strip()
            if doc_id:
                out.append(doc_id)
        return tuple(out)

    return DocumentSplitIds(
        train_doc_ids=_collect("train_results"),
        val_doc_ids=_collect("val_results"),
        test_doc_ids=_collect("test_results"),
    )


def load_split_ids_json(split_ids_path: str | Path) -> DocumentSplitIds:
    payload = json.loads(Path(split_ids_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split ids JSON must contain an object")
    return DocumentSplitIds.from_dict(payload)


def resolve_document_split_ids(
    *,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
) -> tuple[DocumentSplitIds, str]:
    if phase1_data_path and split_ids_path:
        raise ValueError("provide either phase1_data_path or split_ids_path, not both")
    if phase1_data_path:
        resolved = Path(phase1_data_path).expanduser()
        return load_phase1_split_ids(resolved), str(resolved)
    if split_ids_path:
        resolved = Path(split_ids_path).expanduser()
        return load_split_ids_json(resolved), str(resolved)
    raise ValueError("one of phase1_data_path or split_ids_path is required")


def write_split_ids_json(
    split_ids: DocumentSplitIds,
    path: str | Path,
) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split_ids.to_dict(), indent=2), encoding="utf-8")
    return output_path
