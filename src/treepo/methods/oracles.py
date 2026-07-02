"""Small built-in oracle family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _make_oracle_fixture_classical_sketch(config: Mapping[str, Any]) -> list[Any]:
    from treepo.methods.fixtures import make_hll_item_trees

    return make_hll_item_trees(
        n_trees=int(config.get("n_trees", 6)),
        leaves_per_tree=int(config.get("leaves_per_tree", 4)),
        leaf_unit_count=int(config.get("leaf_unit_count", 24)),
        doc_unit_kind=str(config.get("doc_unit_kind", "item")),
        vocabulary_size=int(config.get("vocabulary_size", 200)),
        seed=int(config.get("seed", 0)),
        split=str(config.get("split", "test")),
    )


def _make_oracle_fixture_markov(config: Mapping[str, Any]) -> list[Any]:
    from treepo.methods.fixtures import make_markov_changepoint_trees

    return make_markov_changepoint_trees(
        n_trees=int(config.get("n_trees", 8)),
        n_states=int(config.get("n_states", 4)),
        doc_tokens=int(config.get("doc_tokens", 128)),
        leaf_unit_count=int(config.get("leaf_unit_count", 16)),
        doc_unit_kind=str(config.get("doc_unit_kind", "token")),
        transition_prob=float(config.get("transition_prob", 0.15)),
        vocabulary_size=int(config.get("vocabulary_size", 256)),
        seed=int(config.get("seed", 0)),
        split=str(config.get("split", "test")),
    )


_ORACLE_DOMAIN_FIXTURES = {
    "classical_sketch": _make_oracle_fixture_classical_sketch,
    "markov": _make_oracle_fixture_markov,
}


ORACLE_CONFIG_KEYS = frozenset(
    {
        "oracle_name",
        "eval_data",
        "output_dir",
        "seed",
        "split",
        "n_trees",
        "leaves_per_tree",
        "doc_unit_kind",
        "leaf_unit_count",
        "vocabulary_size",
        "n_states",
        "doc_tokens",
        "transition_prob",
    }
)


def score_oracle(config: Mapping[str, Any] | None = None) -> Any:
    payload = dict(config or {})
    unknown = sorted(str(key) for key in payload if str(key) not in ORACLE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown oracle config keys: {unknown}; allowed: {sorted(ORACLE_CONFIG_KEYS)}")
    name = payload.get("oracle_name")
    if not name:
        available = ", ".join(list_oracles())
        raise ValueError(f"oracle requires config['oracle_name']; registered: {available}")
    domain = oracle_domain(str(name))
    eval_data = payload.get("eval_data")
    if eval_data is None:
        builder = _ORACLE_DOMAIN_FIXTURES.get(domain)
        if builder is None:
            raise ValueError(
                f"oracle {name!r} has domain {domain!r} with no auto-fixture; "
                f"pass config['eval_data'] explicitly. Available domains with fixtures: "
                f"{sorted(_ORACLE_DOMAIN_FIXTURES)}"
            )
        eval_data = builder(payload)
    backend_config: dict[str, Any] = {"oracle_name": str(name)}
    if payload.get("output_dir") is not None:
        backend_config["output_dir"] = str(payload["output_dir"])
    from treepo.methods.contracts import CTreePOLearningSpec
    from treepo.methods.learning import fit

    return fit(
        CTreePOLearningSpec(
            space_kind=f"oracle:{name}",
            family="oracle",
            schedule="fg",
            initial_artifacts={"f": None, "g": None},
            train_data=[],
            eval_data=list(eval_data),
            backend_config=backend_config,
            axis={"max_iterations": 0, "axis_value": 0},
        )
    )


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

    def as_statistic(self, *, f: Any = None, g: Any = None) -> None:
        del f, g
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


__all__ = ["ORACLE_CONFIG_KEYS", "OracleFamilyRuntime", "list_oracles", "oracle_domain", "score_oracle"]
