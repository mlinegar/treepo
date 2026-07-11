"""Provider-neutral prompt and oracle helpers for Manifesto/RILE examples."""

from __future__ import annotations

from typing import Any, Mapping


def manifesto_oracle_predict_fn(*, tree: Any, **kwargs: Any) -> Mapping[str, float | None]:
    del kwargs
    meta = getattr(tree, "metadata", None) or {}
    value = meta.get("teacher_score_native")
    if value is None:
        # The law audit scores leaves and composed states too; an oracle keyed
        # on the document label honestly abstains on units without one.
        value = getattr(tree, "score", None)
    return {"score": None if value is None else float(value)}


def manifesto_prompt_template() -> str:
    return (
        "Estimate the document-level RILE score. Return only one number.\n\n"
        "Document:\n{text}\n\nSupervised examples:\n{supervised_examples}\n\nScore:"
    )


__all__ = ["manifesto_oracle_predict_fn", "manifesto_prompt_template"]
