from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from treepo._research.ctreepo.contracts import (
    LEGACY_RUN_AXIS_PUBLIC_FIELDS,
    RunAxisSpec,
    migrate_legacy_run_axis_mapping,
)


FULL_DOC_CONFIG_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("tree_local_law_weight", "local_law_weight"),
    ("tree_task_objective_weight", "root_share"),
    ("tree_c1_relative_weight", "c1_relative_weight"),
    ("tree_c2_relative_weight", "c2_relative_weight"),
    ("tree_c3_relative_weight", "c3_relative_weight"),
)

LEGACY_PUBLIC_OBJECTIVE_CONFIG_FIELDS = frozenset(
    {
        "task_objective_weight",
        "tree_local_law_weight",
        "tree_task_objective_weight",
        "lambda_local",
        "selected_lambda_local",
        "law_package",
        "law_package_names",
        "leaf_weight",
        "c1_weight",
        "c2_weight",
        "c3_weight",
        "root_weight",
        "local_law_weights",
        "proxy_weights",
    }
)

LEGACY_PUBLIC_RUN_AXIS_CONFIG_FIELDS = LEGACY_RUN_AXIS_PUBLIC_FIELDS


def mapping_from_config_like(config_like: Any) -> Dict[str, Any]:
    if config_like is None:
        return {}
    if isinstance(config_like, Mapping):
        return dict(config_like)
    if is_dataclass(config_like):
        return asdict(config_like)
    if hasattr(config_like, "__dict__"):
        return dict(vars(config_like))
    return dict(config_like)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_value(item)
            for key, item in dict(value).items()
        }
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set):
        return [_json_safe_value(item) for item in sorted(value, key=repr)]
    return value


def canonicalize_full_doc_config_mapping(
    config_like: Any,
    *,
    include_tree_aliases: bool = False,
    include_runtime_aliases: bool = True,
    allow_private_tree_aliases: bool | None = None,
) -> Dict[str, Any]:
    if allow_private_tree_aliases is None:
        allow_private_tree_aliases = bool(is_dataclass(config_like))
    mapping = mapping_from_config_like(config_like)
    normalized = dict(mapping)
    legacy_run_axis_fields = sorted(
        key
        for key in LEGACY_PUBLIC_RUN_AXIS_CONFIG_FIELDS
        if key in normalized and normalized.get(key) not in {"", None}
    )
    if legacy_run_axis_fields and not allow_private_tree_aliases:
        raise ValueError(
            "legacy public run-axis config fields are not supported: "
            + ", ".join(legacy_run_axis_fields)
            + ". Use method_id, law_set_id, method_runs, and "
            "reference_method_runs."
        )
    legacy_fields = sorted(
        key
        for key in LEGACY_PUBLIC_OBJECTIVE_CONFIG_FIELDS
        if key in normalized and normalized.get(key) not in {"", None}
    )
    if legacy_fields and not allow_private_tree_aliases:
        raise ValueError(
            "legacy public objective config fields are not supported: "
            + ", ".join(legacy_fields)
            + ". Use local_law_weight and root_share."
        )
    normalized.pop("task_objective_weight", None)
    for tree_key, runtime_key in FULL_DOC_CONFIG_ALIAS_PAIRS:
        tree_value = normalized.get(tree_key)
        runtime_value = normalized.get(runtime_key)
        if (
            tree_value not in {"", None}
            and runtime_value not in {"", None}
            and tree_value != runtime_value
        ):
            raise ValueError(
                f"Config alias conflict: {tree_key}={tree_value!r} vs "
                f"{runtime_key}={runtime_value!r}. "
                f"Set only one, or ensure they match."
            )
        if runtime_value in {"", None} and tree_value not in {"", None}:
            normalized[runtime_key] = tree_value
        if tree_value in {"", None} and runtime_value not in {"", None}:
            normalized[tree_key] = runtime_value
    if (
        not allow_private_tree_aliases
        and normalized.get("local_law_weight") not in {"", None}
        and normalized.get("root_share") not in {"", None}
    ):
        raise ValueError(
            "local_law_weight is mutually exclusive with root_share"
        )
    if not include_tree_aliases:
        for tree_key, _runtime_key in FULL_DOC_CONFIG_ALIAS_PAIRS:
            normalized.pop(tree_key, None)
    if not include_runtime_aliases:
        for _tree_key, runtime_key in FULL_DOC_CONFIG_ALIAS_PAIRS:
            normalized.pop(runtime_key, None)
    return normalized


def public_run_axis_from_config_like(config_like: Any) -> Dict[str, Any]:
    """Parse one canonical public run-axis config without legacy fallback."""

    return RunAxisSpec.from_mapping(mapping_from_config_like(config_like)).to_dict()


def migrate_legacy_public_run_axis_config(config_like: Any) -> Dict[str, Any]:
    """Explicit migration helper for historical run-axis config fragments."""

    return migrate_legacy_run_axis_mapping(mapping_from_config_like(config_like))


def runtime_config_overrides_from_config_like(
    config_like: Any,
    *,
    allow_private_tree_aliases: bool | None = None,
) -> Dict[str, Any]:
    return canonicalize_full_doc_config_mapping(
        config_like,
        include_tree_aliases=False,
        include_runtime_aliases=True,
        allow_private_tree_aliases=(
            bool(is_dataclass(config_like))
            if allow_private_tree_aliases is None
            else bool(allow_private_tree_aliases)
        ),
    )


def tree_run_config_mapping_from_config_like(config_like: Any) -> Dict[str, Any]:
    return canonicalize_full_doc_config_mapping(
        config_like,
        include_tree_aliases=False,
        include_runtime_aliases=True,
        allow_private_tree_aliases=bool(is_dataclass(config_like)),
    )


def serialize_full_doc_runtime_config(
    config_like: Any,
    *,
    metadata: Any | None = None,
    allow_private_tree_aliases: bool | None = None,
) -> Dict[str, Any]:
    payload = runtime_config_overrides_from_config_like(
        config_like,
        allow_private_tree_aliases=allow_private_tree_aliases,
    )
    if metadata is not None:
        payload.update(mapping_from_config_like(metadata))
    return _json_safe_value(payload)


def serialize_tree_run_config(
    config_like: Any,
    *,
    metadata: Any | None = None,
) -> Dict[str, Any]:
    payload = tree_run_config_mapping_from_config_like(config_like)
    if metadata is not None:
        payload.update(mapping_from_config_like(metadata))
    return _json_safe_value(payload)


def write_tree_run_config_json(path: Path, config_like: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_tree_run_config(config_like)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
