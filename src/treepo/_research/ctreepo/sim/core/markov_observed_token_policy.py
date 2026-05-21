from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


_CONFIG_DIR = Path(__file__).resolve().parents[4] / "config" / "markov" / "observed_token_profiles"


@dataclass(frozen=True)
class MarkovObservedTokenPolicy:
    profile: str = "smoke"
    generator_profile: str = "piecewise_palette"
    n_regimes: int = 4
    vocab_size: int = 16
    min_tokens: int = 96
    max_tokens: int = 96
    min_segments: int = 2
    max_segments: int = 6
    min_seg_len: int = 8
    max_seg_len: int = 24
    fixed_leaf_tokens: int = 16
    train_docs: int = 64
    val_docs: int = 16
    test_docs: int = 32
    state_dim: int = 16
    hidden_dim: int = 64
    n_epochs: int = 4
    batch_size: int = 8
    lr: float = 5e-4
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    torch_threads: int = 1
    local_law_weight: float = 0.5
    c1_relative_weight: float = 1.0
    c2_relative_weight: float = 0.0
    c3_relative_weight: float = 1.0
    rf_n_estimators: int = 100
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 2
    leaf_knn_neighbors: int = 32
    doc_level_ridge_alpha: float = 1.0
    doc_level_ridge_breakdown_orders: Tuple[int, ...] = (1, 2, 3)
    doc_sequence_objective: str = "count_ce_only"
    doc_transformer_head_family: str = "boundary_sum_count_hybrid"
    doc_transformer_layers: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_profile_from_config(profile_key: str) -> Dict[str, Any]:
    """Load an observed-token profile from a TOML config file.

    Searches ``config/markov/observed_token_profiles/<profile_key>.toml``
    and returns the ``[observed_token_policy]`` table.
    """
    config_path = _CONFIG_DIR / f"{profile_key}.toml"
    if not config_path.exists():
        raise ValueError(
            f"unknown observed-token profile: {profile_key!r} "
            f"(no config file at {config_path})"
        )
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    section = raw.get("observed_token_policy", {})
    if not section:
        raise ValueError(
            f"config file {config_path} missing [observed_token_policy] table"
        )
    return dict(section)


def _coerce_policy_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce config values to the types expected by MarkovObservedTokenPolicy."""
    field_types = {f.name: f.type for f in fields(MarkovObservedTokenPolicy)}
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in field_types:
            continue
        expected = field_types[key]
        if expected == "int":
            result[key] = int(value)
        elif expected == "float":
            result[key] = float(value)
        elif expected == "str":
            result[key] = str(value)
        elif "Tuple" in expected:
            result[key] = tuple(int(x) for x in value)
        else:
            result[key] = value
    return result


def resolve_markov_observed_token_policy(
    *,
    profile_name: str = "smoke",
    train_docs: int | None = None,
    val_docs: int | None = None,
    test_docs: int | None = None,
    state_dim: int | None = None,
    hidden_dim: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    doc_sequence_objective: str | None = None,
    doc_transformer_head_family: str | None = None,
    doc_transformer_layers: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    torch_threads: int | None = None,
) -> MarkovObservedTokenPolicy:
    profile_key = str(profile_name or "smoke").strip().lower() or "smoke"
    raw_config = _load_profile_from_config(profile_key)
    coerced = _coerce_policy_fields(raw_config)
    defaults = MarkovObservedTokenPolicy(**coerced)

    # Apply caller overrides on top of config-file defaults.
    return MarkovObservedTokenPolicy(
        profile=str(defaults.profile),
        generator_profile=str(defaults.generator_profile),
        n_regimes=int(defaults.n_regimes),
        vocab_size=int(defaults.vocab_size),
        min_tokens=int(defaults.min_tokens),
        max_tokens=int(defaults.max_tokens),
        min_segments=int(defaults.min_segments),
        max_segments=int(defaults.max_segments),
        min_seg_len=int(defaults.min_seg_len),
        max_seg_len=int(defaults.max_seg_len),
        fixed_leaf_tokens=int(defaults.fixed_leaf_tokens),
        train_docs=int(defaults.train_docs if train_docs is None else train_docs),
        val_docs=int(defaults.val_docs if val_docs is None else val_docs),
        test_docs=int(defaults.test_docs if test_docs is None else test_docs),
        state_dim=int(defaults.state_dim if state_dim is None else state_dim),
        hidden_dim=int(defaults.hidden_dim if hidden_dim is None else hidden_dim),
        n_epochs=int(defaults.n_epochs if n_epochs is None else n_epochs),
        batch_size=int(defaults.batch_size if batch_size is None else batch_size),
        lr=float(defaults.lr if lr is None else lr),
        weight_decay=float(defaults.weight_decay),
        doc_sequence_objective=str(
            defaults.doc_sequence_objective
            if doc_sequence_objective is None
            else doc_sequence_objective
        ),
        doc_transformer_head_family=str(
            defaults.doc_transformer_head_family
            if doc_transformer_head_family is None
            else doc_transformer_head_family
        ),
        doc_transformer_layers=int(
            defaults.doc_transformer_layers
            if doc_transformer_layers is None
            else doc_transformer_layers
        ),
        seed=int(defaults.seed if seed is None else seed),
        device=str(defaults.device if device is None else device),
        torch_threads=int(defaults.torch_threads if torch_threads is None else torch_threads),
        local_law_weight=float(defaults.local_law_weight),
        c1_relative_weight=float(defaults.c1_relative_weight),
        c2_relative_weight=float(defaults.c2_relative_weight),
        c3_relative_weight=float(defaults.c3_relative_weight),
        rf_n_estimators=int(defaults.rf_n_estimators),
        rf_max_depth=int(defaults.rf_max_depth),
        rf_min_samples_leaf=int(defaults.rf_min_samples_leaf),
        leaf_knn_neighbors=int(defaults.leaf_knn_neighbors),
        doc_level_ridge_alpha=float(defaults.doc_level_ridge_alpha),
        doc_level_ridge_breakdown_orders=tuple(int(x) for x in defaults.doc_level_ridge_breakdown_orders),
    )


__all__ = [
    "MarkovObservedTokenPolicy",
    "resolve_markov_observed_token_policy",
]
