"""Exact component targets for a Manifesto/RILE Semantic Forest.

The compact target estimates three non-header shares: left, other, and right.
The deliberately expanded target estimates all 56 CMP policy-code shares plus
one residual non-policy share. Both are exact factorizations of the same RILE
readout and both travel through one named vector ``f`` and one shared ``g``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from treepo.forest import OracleTargetSpec, oracle_vector_l1
from treepo.tree import TreeNode, TreeRecord

MANIFESTO_RILE_COMPONENT_SCHEMA_VERSION = "treepo.manifesto.rile_components.v1"
MANIFESTO_RILE_COMPONENT_SPACE_KIND = "manifesto_rile_component_state.v1"
MANIFESTO_RILE_NORMALIZED_SCHEMA_VERSION = "treepo.manifesto.rile_normalized.v1"
RILE_NORMALIZED_TARGET_NAME = "rile_normalized"
RILE_NORMALIZED_TARGET_KEY = "rile_normalized_vector"
RILE_NORMALIZED_ORACLE_ID = "manifesto_cmp_annotations:v1"

RILE_LEFT_CODES = (
    "103",
    "105",
    "106",
    "107",
    "202",
    "403",
    "404",
    "406",
    "412",
    "413",
    "504",
    "506",
    "701",
)
RILE_RIGHT_CODES = (
    "104",
    "201",
    "203",
    "305",
    "401",
    "402",
    "407",
    "414",
    "505",
    "601",
    "603",
    "605",
    "606",
)

CMP_POLICY_CODES = (
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "110",
    "201",
    "202",
    "203",
    "204",
    "301",
    "302",
    "303",
    "304",
    "305",
    "401",
    "402",
    "403",
    "404",
    "405",
    "406",
    "407",
    "408",
    "409",
    "410",
    "411",
    "412",
    "413",
    "414",
    "415",
    "416",
    "501",
    "502",
    "503",
    "504",
    "505",
    "506",
    "507",
    "601",
    "602",
    "603",
    "604",
    "605",
    "606",
    "607",
    "608",
    "701",
    "702",
    "703",
    "704",
    "705",
    "706",
)

RILE_POLARITY_GRANULARITY = "polarity"
RILE_CMP56_GRANULARITY = "cmp56"
RILE_POLARITY_TARGET_KEY = "rile_polarity_shares"
RILE_CMP56_TARGET_KEY = "rile_cmp56_shares"
RILE_POLARITY_TARGET_NAMES = (
    "rile_left_share",
    "rile_other_share",
    "rile_right_share",
)
RILE_CMP56_TARGET_NAMES = tuple(f"cmp_{code}_share" for code in CMP_POLICY_CODES) + (
    "cmp_residual_other_share",
)

_LEFT_SET = frozenset(RILE_LEFT_CODES)
_RIGHT_SET = frozenset(RILE_RIGHT_CODES)
_CMP_SET = frozenset(CMP_POLICY_CODES)


def normalize_cmp_code(value: Any) -> str | None:
    """Return one canonical three-digit CMP code, ``H``, or ``000``."""

    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        text = f"{int(value):03d}"
    else:
        text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in {"H", "HEADER", "HEADLINE"}:
        return "H"
    if upper in {"NAN", "NONE", "NULL"}:
        return None
    if "." in text:
        text = text.split(".", 1)[0]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if int(digits) == 0:
        return "000"
    return digits[:3] if len(digits) >= 3 else digits.zfill(3)


def manifesto_rile_component_target_names(
    granularity: str = RILE_CMP56_GRANULARITY,
) -> tuple[str, ...]:
    """Return the frozen target order for a compact or expanded RILE vector."""

    resolved = _granularity(granularity)
    return (
        RILE_POLARITY_TARGET_NAMES
        if resolved == RILE_POLARITY_GRANULARITY
        else RILE_CMP56_TARGET_NAMES
    )


def manifesto_rile_component_target_key(
    granularity: str = RILE_CMP56_GRANULARITY,
) -> str:
    """Return the metadata key consumed by the named-vector fit."""

    return (
        RILE_POLARITY_TARGET_KEY
        if _granularity(granularity) == RILE_POLARITY_GRANULARITY
        else RILE_CMP56_TARGET_KEY
    )


def manifesto_rile_component_output_schema(
    granularity: str = RILE_CMP56_GRANULARITY,
) -> dict[str, Any]:
    """Return a strict JSON Schema for a full-document component estimate."""

    resolved = _granularity(granularity)
    targets = manifesto_rile_component_targets(resolved)
    properties: dict[str, Any] = {}
    for target in targets:
        polarity = str(target.metadata["polarity"])
        code = target.metadata.get("cmp_code")
        description = (
            f"Estimated non-header share for CMP category {code} ({polarity} for RILE)."
            if code is not None
            else f"Estimated non-header {polarity} share."
        )
        properties[target.target_name] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": description,
        }
    return {
        "type": "object",
        "description": (
            "Dense full-document RILE component shares in the package's frozen "
            f"{resolved} target order."
        ),
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def manifesto_rile_components_from_codes(
    codes: Iterable[Any],
    *,
    granularity: str = RILE_CMP56_GRANULARITY,
) -> dict[str, float]:
    """Build exact component shares from one code per quasi-sentence."""

    counts: Counter[str] = Counter()
    for raw_code in codes:
        code = normalize_cmp_code(raw_code)
        if code is None:
            raise ValueError(f"CMP code {raw_code!r} cannot be normalized")
        counts[code] += 1
    return manifesto_rile_components_from_counts(
        counts,
        granularity=granularity,
    )


def manifesto_rile_components_from_counts(
    counts: Mapping[Any, Any],
    *,
    total_non_header_mass: float | None = None,
    granularity: str = RILE_CMP56_GRANULARITY,
) -> dict[str, float]:
    """Convert authoritative CMP counts into dense non-header shares.

    ``counts`` may be sparse, but an omitted category is an observed zero only
    when the mapping is authoritative for the span. If only the RILE-bearing
    counts are supplied, callers must also supply the exact non-header mass;
    the difference becomes residual ``other`` mass. Headers are excluded from
    every target coordinate and from the denominator.
    """

    resolved = _granularity(granularity)
    normalized: Counter[str] = Counter()
    for raw_code, raw_mass in dict(counts or {}).items():
        mass = _nonnegative(raw_mass, name=f"count[{raw_code!r}]")
        if mass == 0.0:
            continue
        code = normalize_cmp_code(raw_code)
        if code is None:
            raise ValueError(f"CMP code {raw_code!r} cannot be normalized")
        normalized[code] += mass

    observed_non_header = float(sum(mass for code, mass in normalized.items() if code != "H"))
    if total_non_header_mass is None:
        denominator = observed_non_header
    else:
        denominator = _nonnegative(
            total_non_header_mass,
            name="total_non_header_mass",
        )
        if observed_non_header > denominator + 1e-9:
            raise ValueError(
                "observed non-header CMP mass exceeds total_non_header_mass: "
                f"{observed_non_header} > {denominator}"
            )
    if denominator <= 0.0:
        raise ValueError("total non-header CMP mass must be positive")

    left = float(sum(normalized.get(code, 0.0) for code in _LEFT_SET))
    right = float(sum(normalized.get(code, 0.0) for code in _RIGHT_SET))
    if resolved == RILE_POLARITY_GRANULARITY:
        return {
            "rile_left_share": left / denominator,
            "rile_other_share": (denominator - left - right) / denominator,
            "rile_right_share": right / denominator,
        }

    policy_mass = float(sum(normalized.get(code, 0.0) for code in CMP_POLICY_CODES))
    values = {
        f"cmp_{code}_share": float(normalized.get(code, 0.0)) / denominator
        for code in CMP_POLICY_CODES
    }
    values["cmp_residual_other_share"] = (denominator - policy_mass) / denominator
    return values


def manifesto_rile_component_targets(
    granularity: str = RILE_CMP56_GRANULARITY,
    *,
    oracle_id: str = "manifesto_cmp_annotations:v1",
) -> tuple[OracleTargetSpec, ...]:
    """Return the ordered coordinates of one joint RILE component oracle."""

    resolved = _granularity(granularity)
    targets: list[OracleTargetSpec] = []
    for name in manifesto_rile_component_target_names(resolved):
        code = (
            name.removeprefix("cmp_").removesuffix("_share")
            if name.startswith("cmp_") and name != "cmp_residual_other_share"
            else None
        )
        polarity = (
            "left"
            if code in _LEFT_SET or name == "rile_left_share"
            else ("right" if code in _RIGHT_SET or name == "rile_right_share" else "other")
        )
        coefficient = -100.0 if polarity == "left" else 100.0 if polarity == "right" else 0.0
        targets.append(
            OracleTargetSpec(
                target_name=name,
                oracle_id=oracle_id,
                metadata={
                    "schema_version": MANIFESTO_RILE_COMPONENT_SCHEMA_VERSION,
                    "granularity": resolved,
                    "cmp_code": code,
                    "polarity": polarity,
                    "carrier": "share_of_non_header_qsentences",
                    "bounds": [0.0, 1.0],
                    "rile_readout_coefficient": coefficient,
                },
            )
        )
    return tuple(targets)


def manifesto_rile_component_readout(
    values: Mapping[str, Any] | Sequence[Any],
    *,
    granularity: str = RILE_CMP56_GRANULARITY,
) -> float:
    """Apply the frozen linear RILE readout without clipping predictions."""

    resolved = _granularity(granularity)
    names = manifesto_rile_component_target_names(resolved)
    vector = _ordered_values(values, names=names)
    if resolved == RILE_POLARITY_GRANULARITY:
        left, _other, right = vector
    else:
        by_name = dict(zip(names, vector))
        left = sum(by_name[f"cmp_{code}_share"] for code in _LEFT_SET)
        right = sum(by_name[f"cmp_{code}_share"] for code in _RIGHT_SET)
    return float(100.0 * (right - left))


def manifesto_rile_component_report(
    prediction: Mapping[str, Any] | Sequence[Any],
    target: Mapping[str, Any] | Sequence[Any],
    *,
    granularity: str = RILE_CMP56_GRANULARITY,
) -> dict[str, Any]:
    """Report component L1, derived RILE error, and the exact L1 envelope."""

    names = manifesto_rile_component_target_names(granularity)
    predicted = _ordered_values(prediction, names=names)
    observed = _ordered_values(target, names=names)
    component_l1 = oracle_vector_l1(predicted, observed)
    predicted_rile = manifesto_rile_component_readout(
        predicted,
        granularity=granularity,
    )
    target_rile = manifesto_rile_component_readout(
        observed,
        granularity=granularity,
    )
    raw_error = abs(predicted_rile - target_rile)
    normalized_error = raw_error / 200.0
    envelope = 0.5 * component_l1
    return {
        "component_l1": component_l1,
        "predicted_rile": predicted_rile,
        "target_rile": target_rile,
        "rile_absolute_error": raw_error,
        "rile_normalized_absolute_error": normalized_error,
        "rile_normalized_l1_upper_bound": envelope,
        "rile_l1_bound_holds": bool(normalized_error <= envelope + 1e-12),
        "prediction_sum": float(sum(predicted)),
        "target_sum": float(sum(observed)),
        "prediction_negative_mass": float(sum(-value for value in predicted if value < 0.0)),
    }


def _manifesto_rile_component_signature_instructions(
    granularity: str,
) -> tuple[str, str]:
    """Return task-owned f/shared-g instructions for one component space."""

    resolved = _granularity(granularity)
    names = manifesto_rile_component_target_names(resolved)
    required_keys = ", ".join(names)
    if resolved == RILE_POLARITY_GRANULARITY:
        target_description = (
            "three shares of non-header quasi-sentence mass: left RILE mass "
            "(rile_left_share), other/non-left-or-right mass "
            "(rile_other_share), and "
            "right RILE mass (rile_right_share)"
        )
        readout = "The derived RILE score is exactly 100 * (rile_right_share - rile_left_share)."
        preservation = (
            "Preserve evidence and denominator mass distinguishing left, right, "
            "and other/non-left-or-right quasi-sentences. The other bucket is "
            "not an ideological-neutral label."
        )
    else:
        left_codes = ", ".join(RILE_LEFT_CODES)
        right_codes = ", ".join(RILE_RIGHT_CODES)
        target_description = (
            "all 56 frozen CMP policy-code shares plus "
            "cmp_residual_other_share for non-header mass outside those CMP codes"
        )
        readout = (
            "The derived RILE score is exactly 100 times the sum of right CMP "
            f"shares ({right_codes}) minus the sum of left CMP shares "
            f"({left_codes})."
        )
        preservation = (
            "Preserve the CMP category evidence/count mass, the non-header "
            "denominator, and residual non-policy mass needed for every one of "
            "the 57 coordinates."
        )

    denominator = (
        "Every coordinate uses the same total number/mass of non-header "
        "quasi-sentences as its denominator; headers are excluded. The dense "
        "shares therefore sum to one for an authoritative observed span."
    )
    f_instructions = (
        "Estimate the Manifesto Project component factorization of RILE from "
        f"one task state. The target is {target_description}. {denominator} "
        f"{readout} Return exactly one strict JSON object with every declared "
        "named coordinate once, no additional coordinates, and no scalar or "
        f"positional substitute. Required keys in order: {required_keys}."
    )
    g_instructions = (
        "Implement the one shared Manifesto/RILE state operator g at leaf "
        "reduction, unary recompression, and binary merge calls. "
        f"{preservation} {denominator} The resulting state must support the same "
        f"joint f readout for {target_description}. Do not use separate g "
        "programs for leaf, recompression, or merge roles."
    )
    return f_instructions, g_instructions


def _manifesto_rile_normalized_signature_instructions() -> tuple[str, str]:
    """Return task-owned f/shared-g instructions for normalized singleton RILE."""

    formula = (
        "rile_normalized = (RILE + 100) / 200 = "
        "0.5 + 0.5 * (right - left) / nonheader"
    )
    denominator = (
        "Here left, right, and nonheader are authoritative quasi-sentence "
        "count/mass totals for the observed span. Headers are excluded from "
        "both the numerator and denominator. Other residual non-header mass "
        "is not an ideological-neutral label."
    )
    f_instructions = (
        "Estimate normalized Manifesto RILE from one task state. The sole "
        f"coordinate is {formula}. {denominator} Return exactly one strict "
        "JSON object with the named coordinate rile_normalized once, no "
        "additional coordinates, and no bare scalar or positional substitute."
    )
    g_instructions = (
        "Implement the one shared Manifesto/RILE state operator g at leaf "
        "reduction, unary recompression, and binary merge calls. Preserve the "
        "left, right, and other residual non-header evidence/count mass plus "
        f"the common non-header denominator needed for {formula}. {denominator} "
        "Do not use separate g programs for leaf, recompression, or merge roles."
    )
    return f_instructions, g_instructions


def manifesto_rile_normalized_fit_fragment(
    *,
    oracle_id: str = RILE_NORMALIZED_ORACLE_ID,
    analytic_leaf_rollup: bool = False,
    rollup_weight_key: str = "total_non_header_qsentences",
) -> dict[str, Any]:
    """Return the normalized singleton-RILE part of one ``treepo.fit`` spec.

    The public target remains a one-coordinate named vector, so K=1 uses the
    same strict vector I/O and sum-L1 path as the component K=3 and K=57
    objectives.
    """

    instructions = _manifesto_rile_normalized_signature_instructions()
    f_instructions, g_instructions = instructions
    backend_config: dict[str, Any] = {
        "target_vector_key": RILE_NORMALIZED_TARGET_KEY,
        "node_target_key": RILE_NORMALIZED_TARGET_KEY,
        "node_target_exclusive": True,
        "target_min": 0.0,
        "target_max": 1.0,
        "normalize_targets": False,
        "root_readout": "leaf_mean" if analytic_leaf_rollup else "root_state",
        "f_signature_instructions": f_instructions,
        "g_signature_instructions": g_instructions,
    }
    if analytic_leaf_rollup:
        backend_config["rollup_weight_key"] = str(rollup_weight_key)
    return {
        "space_kind": MANIFESTO_RILE_COMPONENT_SPACE_KIND,
        "oracle_targets": (
            OracleTargetSpec(
                target_name=RILE_NORMALIZED_TARGET_NAME,
                oracle_id=str(oracle_id),
                metadata={
                    "schema_version": MANIFESTO_RILE_NORMALIZED_SCHEMA_VERSION,
                    "bounds": [0.0, 1.0],
                    "carrier": "normalized_rile_from_non_header_qsentence_mass",
                    "formula": "(RILE+100)/200=0.5+0.5*(right-left)/nonheader",
                    "readout": "raw_rile=200*rile_normalized-100",
                },
            ),
        ),
        "backend_config": backend_config,
    }


def manifesto_rile_component_fit_fragment(
    granularity: str = RILE_CMP56_GRANULARITY,
    *,
    oracle_id: str = "manifesto_cmp_annotations:v1",
    analytic_leaf_rollup: bool = False,
    rollup_weight_key: str = "total_non_header_qsentences",
) -> dict[str, Any]:
    """Return the RILE-specific part of one ordinary ``treepo.fit`` spec."""

    resolved = _granularity(granularity)
    f_instructions, g_instructions = _manifesto_rile_component_signature_instructions(resolved)
    backend_config: dict[str, Any] = {
        "target_vector_key": manifesto_rile_component_target_key(resolved),
        "node_target_key": manifesto_rile_component_target_key(resolved),
        "node_target_exclusive": True,
        "target_min": 0.0,
        "target_max": 1.0,
        "normalize_targets": False,
        "root_readout": "leaf_mean" if analytic_leaf_rollup else "root_state",
        "f_signature_instructions": f_instructions,
        "g_signature_instructions": g_instructions,
    }
    if analytic_leaf_rollup:
        backend_config["rollup_weight_key"] = str(rollup_weight_key)
    return {
        "space_kind": MANIFESTO_RILE_COMPONENT_SPACE_KIND,
        "oracle_targets": manifesto_rile_component_targets(
            resolved,
            oracle_id=oracle_id,
        ),
        "backend_config": backend_config,
    }


def attach_manifesto_rile_components(
    records: Iterable[Any],
    *,
    granularity: str = RILE_CMP56_GRANULARITY,
    counts_key: str = "cmp_counts",
    total_non_header_key: str = "total_non_header_qsentences",
    component_key: str | None = None,
    law_state_key: str | None = None,
    require_all_nodes: bool = True,
    drop_scalar_root_label: bool = False,
) -> list[TreeRecord]:
    """Hydrate root/node component vectors from authoritative CMP metadata.

    Existing labels and metadata are preserved unless
    ``drop_scalar_root_label=True``. The returned fit fragment sets
    ``node_target_exclusive=True``, so retained scalar node labels cannot be
    consumed by the component objective.
    """

    resolved = _granularity(granularity)
    target_key = component_key or manifesto_rile_component_target_key(resolved)
    names = manifesto_rile_component_target_names(resolved)
    output: list[TreeRecord] = []
    for raw_record in records:
        record = TreeRecord.from_value(raw_record)
        nodes: list[TreeNode] = []
        for node in record.nodes:
            metadata = dict(node.metadata or {})
            raw_counts = metadata.get(counts_key)
            if not isinstance(raw_counts, Mapping):
                if require_all_nodes:
                    raise ValueError(
                        f"tree {record.tree_id!r} node {node.node_id!r} is missing "
                        f"authoritative {counts_key!r} metadata"
                    )
                nodes.append(node)
                continue
            raw_total = metadata.get(total_non_header_key)
            if raw_total is None and require_all_nodes:
                raise ValueError(
                    f"tree {record.tree_id!r} node {node.node_id!r} is missing "
                    f"{total_non_header_key!r}"
                )
            components = manifesto_rile_components_from_counts(
                raw_counts,
                total_non_header_mass=raw_total,
                granularity=resolved,
            )
            metadata[target_key] = components
            metadata["rile_component_schema_version"] = MANIFESTO_RILE_COMPONENT_SCHEMA_VERSION
            metadata["rile_component_granularity"] = resolved
            if raw_total is not None:
                metadata["rile_component_denominator_mass"] = float(raw_total)
            if law_state_key:
                denominator = (
                    float(raw_total)
                    if raw_total is not None
                    else float(
                        sum(
                            _nonnegative(value, name=f"count[{key!r}]")
                            for key, value in raw_counts.items()
                            if normalize_cmp_code(key) != "H"
                        )
                    )
                )
                metadata[law_state_key] = [float(components[name]) * denominator for name in names]
            nodes.append(replace(node, metadata=metadata))

        hydrated = replace(record, nodes=tuple(nodes))
        root = hydrated.root()
        if root is None:
            raise ValueError(f"tree {record.tree_id!r} has no root node")
        root_components = dict(root.metadata or {}).get(target_key)
        if not isinstance(root_components, Mapping):
            raise ValueError(
                f"tree {record.tree_id!r} root is missing hydrated target {target_key!r}"
            )
        metadata = dict(hydrated.metadata or {})
        metadata[target_key] = dict(root_components)
        metadata["rile_component_schema_version"] = MANIFESTO_RILE_COMPONENT_SCHEMA_VERSION
        metadata["rile_component_granularity"] = resolved
        metadata["rile_from_components"] = manifesto_rile_component_readout(
            root_components,
            granularity=resolved,
        )
        output.append(
            replace(
                hydrated,
                root_label=None if drop_scalar_root_label else hydrated.root_label,
                metadata=metadata,
            )
        )
    return output


def _granularity(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "compact": RILE_POLARITY_GRANULARITY,
        "polarity": RILE_POLARITY_GRANULARITY,
        "expanded": RILE_CMP56_GRANULARITY,
        "cmp": RILE_CMP56_GRANULARITY,
        "cmp56": RILE_CMP56_GRANULARITY,
    }
    if normalized not in aliases:
        raise ValueError(f"granularity must be 'polarity' or 'cmp56', got {value!r}")
    return aliases[normalized]


def _nonnegative(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _ordered_values(
    values: Mapping[str, Any] | Sequence[Any],
    *,
    names: tuple[str, ...],
) -> tuple[float, ...]:
    if isinstance(values, Mapping):
        by_name = {str(key): value for key, value in values.items()}
        expected = set(names)
        actual = set(by_name)
        if actual != expected:
            raise ValueError(
                "component mapping must exactly cover the declared targets; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        raw = tuple(by_name[name] for name in names)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw = tuple(values)
        if len(raw) != len(names):
            raise ValueError(f"component vector width {len(raw)} does not match {len(names)}")
    else:
        raise TypeError("components must be a named mapping or numeric sequence")
    output: list[float] = []
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("components must contain only finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError("components must contain only finite numbers")
        output.append(number)
    return tuple(output)


__all__ = [
    "CMP_POLICY_CODES",
    "MANIFESTO_RILE_COMPONENT_SCHEMA_VERSION",
    "MANIFESTO_RILE_COMPONENT_SPACE_KIND",
    "MANIFESTO_RILE_NORMALIZED_SCHEMA_VERSION",
    "RILE_CMP56_GRANULARITY",
    "RILE_CMP56_TARGET_KEY",
    "RILE_CMP56_TARGET_NAMES",
    "RILE_LEFT_CODES",
    "RILE_POLARITY_GRANULARITY",
    "RILE_POLARITY_TARGET_KEY",
    "RILE_POLARITY_TARGET_NAMES",
    "RILE_NORMALIZED_ORACLE_ID",
    "RILE_NORMALIZED_TARGET_KEY",
    "RILE_NORMALIZED_TARGET_NAME",
    "RILE_RIGHT_CODES",
    "attach_manifesto_rile_components",
    "manifesto_rile_component_fit_fragment",
    "manifesto_rile_normalized_fit_fragment",
    "manifesto_rile_component_output_schema",
    "manifesto_rile_component_readout",
    "manifesto_rile_component_report",
    "manifesto_rile_component_target_key",
    "manifesto_rile_component_target_names",
    "manifesto_rile_component_targets",
    "manifesto_rile_components_from_codes",
    "manifesto_rile_components_from_counts",
    "normalize_cmp_code",
]
