"""Statistic and local-law payload helpers for methods runtimes."""

from __future__ import annotations

from typing import Any, Sequence

from treepo.local_law import local_law_objective_summary
from treepo.methods.contracts import FamilyRuntime
from treepo.statistic import family_statistic


def statistic_payload(
    *,
    family: FamilyRuntime,
    f_artifact: Any,
    g_artifact: Any,
    eval_trees: Sequence[Any],
) -> dict[str, Any]:
    statistic = family_statistic(family, f=f_artifact, g=g_artifact)
    if statistic is None:
        return {}
    payload: dict[str, Any] = {"info": statistic.info.to_dict()}
    try:
        rows = list(statistic.local_law_rows(list(eval_trees or ())))
    except Exception as exc:  # pragma: no cover - defensive metadata path
        payload["local_law_error"] = f"{type(exc).__name__}: {exc}"
        return payload
    if rows:
        payload["local_law_summary"] = local_law_objective_summary(rows).to_dict()
        payload["local_law_row_count"] = int(len(rows))
    return payload


__all__ = ["statistic_payload"]
