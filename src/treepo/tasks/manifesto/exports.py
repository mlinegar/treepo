"""Manifesto preference export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from treepo.methods.preference import export_preference_records, summarize_preference_views
from treepo.tasks.manifesto.common import slug
from treepo.tasks.manifesto.documents import ManifestoReplicationTree
from treepo.tasks.manifesto.preferences import make_manifesto_preferences


def export_manifesto_reward_views(
    *,
    output_dir: Path,
    trees: Sequence[ManifestoReplicationTree],
    preference_scopes: Sequence[str],
    preference_modes: Sequence[str],
    sample_size: int | None = None,
    sample_rate: float | None = None,
    seed: int = 0,
    export_formats: Sequence[str] = ("general", "supervised", "dpo", "reward", "grpo"),
) -> list[dict[str, Any]]:
    """Write scoped Manifesto preference exports for reward/optimizer examples."""

    rows: list[dict[str, Any]] = []
    for scope in tuple(str(v) for v in preference_scopes):
        for mode in tuple(str(v) for v in preference_modes):
            cell_dir = Path(output_dir) / f"{slug(scope)}_{slug(mode)}"
            preferences = make_manifesto_preferences(
                trees,
                mode=mode,
                scope=scope,
                sample_size=sample_size,
                sample_rate=sample_rate,
                seed=int(seed),
            )
            artifacts = export_preference_records(
                preferences,
                cell_dir / "preference",
                formats=tuple(str(v) for v in export_formats),
            )
            rows.append(
                {
                    "scope": scope,
                    "mode": mode,
                    "output_dir": str(cell_dir),
                    "counts": dict(artifacts.get("counts") or {}),
                    "files": dict(artifacts.get("files") or {}),
                    "optimizer_views": summarize_preference_views(preferences),
                    "summary": dict(artifacts.get("summary") or {}),
                }
            )
    return rows


__all__ = ["export_manifesto_reward_views"]
