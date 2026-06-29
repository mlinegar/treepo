"""Small built-in oracle family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


_ORACLE_DOMAINS = {
    "hll_exact": "classical_sketch",
    "markov_changepoint_count": "markov",
}


def list_oracles() -> tuple[str, ...]:
    return tuple(sorted(_ORACLE_DOMAINS))


def oracle_domain(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _ORACLE_DOMAINS:
        raise KeyError(f"unknown oracle {name!r}; available: {', '.join(list_oracles())}")
    return _ORACLE_DOMAINS[key]


class OracleFamilyRuntime:
    """FamilyRuntime wrapper for lightweight built-in oracles."""

    def __init__(self, oracle_name: str) -> None:
        key = str(oracle_name).strip().lower()
        if key not in _ORACLE_DOMAINS:
            raise KeyError(f"unknown oracle {oracle_name!r}; available: {', '.join(list_oracles())}")
        self.oracle_name = key
        self.name = f"oracle:{key}"

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> str:
        return f"oracle:{self.oracle_name}"

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        return g_init

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[float | None]:
        if self.oracle_name == "hll_exact":
            return [_hll_exact(tree) for tree in trees]
        if self.oracle_name == "markov_changepoint_count":
            return [_markov_changepoint_count(tree) for tree in trees]
        raise KeyError(self.oracle_name)

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        return None


def _markov_changepoint_count(tree: Any) -> float | None:
    regimes = list(getattr(tree, "regimes", None) or [])
    if not regimes:
        for leaf in list(getattr(tree, "leaves", None) or []):
            leaf_regimes = getattr(leaf, "regimes", None)
            if leaf_regimes is not None:
                regimes.extend(list(leaf_regimes))
    if regimes:
        return float(
            sum(1 for left, right in zip(regimes, regimes[1:]) if int(left) != int(right))
        )
    metadata = getattr(tree, "metadata", None) or {}
    value = metadata.get("teacher_score_native", metadata.get("teacher_score_1_7"))
    return float(value) if value is not None else None


def _hll_exact(tree: Any) -> float | None:
    tokens: list[Any] = []
    direct = getattr(tree, "tokens", None)
    if direct is not None:
        tokens.extend(list(direct))
    for leaf in list(getattr(tree, "leaves", None) or []):
        leaf_tokens = getattr(leaf, "tokens", None)
        if leaf_tokens is not None:
            tokens.extend(list(leaf_tokens))
    if tokens:
        return float(len(set(tokens)))
    metadata = getattr(tree, "metadata", None) or {}
    value = metadata.get("teacher_score_native", metadata.get("teacher_score_1_7"))
    return float(value) if value is not None else None


__all__ = ["OracleFamilyRuntime", "list_oracles", "oracle_domain"]
