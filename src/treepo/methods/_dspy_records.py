"""DSPy training-record adapter for labeled :class:`~treepo.tree.TreeRecord`s.

The adapter is deliberately independent of DSPy itself.  It preserves the
package-owned binary topology and ``TaskState`` supervision, constructs the
node-wide f/g records used by the optimizer, applies sampling and role weights
once, and performs group-disjoint train/validation splitting.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from treepo.sampling import resolve_effective_propensity
from treepo.state import TaskState, state_from_value, state_to_dict
from treepo.tree import TreeNode, TreeRecord

ParseTarget = Callable[[Any], Any | None]
ReduceStates = Callable[[TreeRecord, "BinaryTopology"], Mapping[str, str]]


@dataclass(frozen=True)
class BinaryTopology:
    """One connected explicit binary/unary topology.

    Unary parents are the package representation of an odd-node carry.  A
    parent with more than two supplied children is rejected; the adapter never
    invents intermediate nodes or an arbitrary left fold.
    """

    root: TreeNode
    children_by_id: Mapping[str, tuple[TreeNode, ...]]

    def children(self, node: TreeNode) -> tuple[TreeNode, ...]:
        return tuple(self.children_by_id.get(str(node.node_id), ()))


def binary_topology(record: TreeRecord) -> BinaryTopology:
    """Validate and return the exact supplied binary/unary tree topology."""

    if not record.nodes:
        raise ValueError("binary topology requires at least one TreeRecord node")
    by_id = {str(node.node_id): node for node in record.nodes}
    if len(by_id) != len(record.nodes) or any(not node_id for node_id in by_id):
        raise ValueError(f"tree {record.tree_id!r} has missing or duplicate node ids")

    parent_children: dict[str, list[TreeNode]] = {}
    for node in record.nodes:
        if node.parent_id is None:
            continue
        parent_id = str(node.parent_id)
        if parent_id not in by_id:
            raise ValueError(
                f"tree {record.tree_id!r} node {node.node_id!r} names missing parent {parent_id!r}"
            )
        parent_children.setdefault(parent_id, []).append(node)

    children_by_id: dict[str, tuple[TreeNode, ...]] = {}
    named_as_child: set[str] = set()
    owner_by_child: dict[str, str] = {}
    for node in record.nodes:
        node_id = str(node.node_id)
        explicit_ids: list[str] = []
        for child_id in (node.left_child_id, node.right_child_id):
            if child_id is None:
                continue
            child_key = str(child_id)
            if child_key not in by_id:
                raise ValueError(
                    f"tree {record.tree_id!r} node {node_id!r} names missing child {child_key!r}"
                )
            if child_key in explicit_ids:
                raise ValueError(
                    f"tree {record.tree_id!r} node {node_id!r} duplicates child edge {child_key!r}"
                )
            explicit_ids.append(child_key)
        inferred = sorted(
            parent_children.get(node_id, ()),
            key=lambda child: (
                -1 if child.position is None else int(child.position),
                str(child.node_id),
            ),
        )
        inferred_ids = [str(child.node_id) for child in inferred]
        if len(inferred_ids) > 2:
            raise ValueError(
                f"tree {record.tree_id!r} node {node_id!r} has "
                f"{len(inferred_ids)} supplied children; DSPy requires explicit "
                "binary/unary topology and will not fabricate a fold"
            )
        if explicit_ids and set(explicit_ids) != set(inferred_ids or explicit_ids):
            raise ValueError(
                f"tree {record.tree_id!r} node {node_id!r} has inconsistent "
                "child-id and parent-id topology"
            )
        child_ids = explicit_ids or inferred_ids
        if len(child_ids) > 2:
            raise ValueError(f"tree {record.tree_id!r} node {node_id!r} is not binary/unary")
        children = tuple(by_id[child_id] for child_id in child_ids)
        for child in children:
            child_id = str(child.node_id)
            prior_owner = owner_by_child.get(child_id)
            if prior_owner is not None and prior_owner != node_id:
                raise ValueError(
                    f"tree {record.tree_id!r} child {child_id!r} is owned by "
                    f"multiple parents: {prior_owner!r} and {node_id!r}"
                )
            owner_by_child[child_id] = node_id
            child_parent = None if child.parent_id is None else str(child.parent_id)
            if child_parent is not None and child_parent != node_id:
                raise ValueError(
                    f"tree {record.tree_id!r} child {child.node_id!r} points to "
                    f"parent {child_parent!r}, not {node_id!r}"
                )
            named_as_child.add(str(child.node_id))
        children_by_id[node_id] = children

    roots = [
        node
        for node in record.nodes
        if str(node.node_id) not in named_as_child and node.parent_id is None
    ]
    declared_roots = [node for node in roots if str(node.unit_type) == "root"]
    if len(declared_roots) == 1:
        root = declared_roots[0]
    elif len(roots) == 1:
        root = roots[0]
    else:
        raise ValueError(
            f"tree {record.tree_id!r} must expose exactly one connected root; "
            f"found {[str(node.node_id) for node in roots]!r}"
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: TreeNode) -> None:
        node_id = str(node.node_id)
        if node_id in visiting:
            raise ValueError(f"tree {record.tree_id!r} contains a cycle at {node_id!r}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child in children_by_id[node_id]:
            visit(child)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(root)
    if visited != set(by_id):
        missing = sorted(set(by_id) - visited)
        raise ValueError(
            f"tree {record.tree_id!r} has nodes outside the root topology: {missing!r}"
        )
    return BinaryTopology(root=root, children_by_id=children_by_id)


def f_training_records(
    traces: Sequence[Any],
    *,
    config: Any,
    parse_target: ParseTarget,
    reduce_states: ReduceStates | None = None,
    allow_identity_g_input: bool = False,
) -> list[dict[str, Any]]:
    """Build one f record for every observed labeled node.

    ``TaskState.text`` is the reference g-state and ``TaskState.measures`` is
    the preferred f target.  When a current learned g is supplied,
    ``reduce_states`` may replace reference states with its node-wise outputs.
    Gold targets remain output fields only. ``allow_identity_g_input`` is the
    canonical singleton equation: when the runtime supplies ``g(x)=x``, the
    resulting state consumed by f is raw node text even though fabricating an
    identity target for a learned g remains disabled.
    """

    rows: list[dict[str, Any]] = []
    for trace_index, trace in enumerate(traces):
        record = TreeRecord.from_value(trace)
        if not record.nodes:
            metadata = metadata_of(trace)
            if metadata.get("observed") is False:
                continue
            target = target_from_trace(trace, config=config, parse_target=parse_target)
            is_f_preference = str(metadata.get("preference_target") or "").lower() == "f"
            if allow_identity_g_input and not is_f_preference:
                state = trace_text(trace).strip()
                f_state_source = "identity_raw_text"
            elif is_f_preference:
                state = f_trace_state(trace, metadata)
                f_state_source = "preference_prompt"
            else:
                state = f_trace_state(trace, metadata)
                f_state_source = "reference_state"
            if target is None or not state:
                continue
            supervision_role = explicit_supervision_role(metadata, default="root")
            weight = example_weight(config, metadata, supervision_role=supervision_role)
            if weight["effective_weight"] <= 0.0:
                continue
            rows.append(
                {
                    "state": state,
                    "prediction_json": target_json(target),
                    "target_json": target_json(target),
                    "role": supervision_role,
                    "supervision_role": supervision_role,
                    "row_id": row_id(trace, index=trace_index, role=supervision_role),
                    "group_id": group_id(trace, metadata, fallback=trace_index),
                    "source_split": split_name(metadata),
                    "f_state_source": f_state_source,
                    **weight,
                    "inputs": ("state",),
                }
            )
            continue

        topology = binary_topology(record)
        generated = dict(reduce_states(record, topology) if reduce_states else {})
        root_id = str(topology.root.node_id)
        for node_index, node in enumerate(record.nodes):
            metadata = merged_metadata(record, node)
            if metadata.get("observed") is False:
                continue
            children = topology.children(node)
            supervision_role = explicit_supervision_role(
                metadata,
                default=node_supervision_role(
                    node,
                    root_id=root_id,
                    children=children,
                ),
            )
            weight = example_weight(
                config,
                metadata,
                supervision_role=supervision_role,
            )
            if weight["effective_weight"] <= 0.0:
                continue
            target = (
                target_from_trace(trace, config=config, parse_target=parse_target)
                if str(node.node_id) == root_id
                else target_from_node(node, config=config, parse_target=parse_target)
            )
            if target is None:
                continue
            if allow_identity_g_input:
                raw_text = record.text or node.text if str(node.node_id) == root_id else node.text
                state = str(raw_text or "").strip()
                f_state_source = "identity_raw_text"
            else:
                generated_state = str(generated.get(str(node.node_id)) or "").strip()
                state = generated_state
                if not state:
                    state = reference_state(
                        node,
                        fallback=str(node.text or ""),
                        allow_fallback=bool(config.allow_identity_g_targets),
                    )
                f_state_source = "current_g_generated" if generated_state else "reference_state"
            if not state:
                continue
            rows.append(
                {
                    "state": state,
                    "prediction_json": target_json(target),
                    "target_json": target_json(target),
                    "role": supervision_role,
                    "supervision_role": supervision_role,
                    "row_id": (f"{record.tree_id}:{node.node_id}:f:{trace_index}:{node_index}"),
                    "group_id": group_id(record, metadata, fallback=trace_index),
                    "source_split": split_name(metadata),
                    "f_state_source": f_state_source,
                    **weight,
                    "inputs": ("state",),
                }
            )
    return rows


def g_training_records(
    traces: Sequence[Any],
    *,
    config: Any,
    parse_target: ParseTarget,
    generated_states_by_trace: Mapping[int, Mapping[str, str]] | None = None,
    scheduled_sampling_rate: float = 0.0,
    scheduled_sampling_seed: int = 0,
) -> list[dict[str, Any]]:
    """Build tagged leaf/merge records for one shared g program."""

    rate = float(scheduled_sampling_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("scheduled_sampling_rate must be in [0, 1]")
    generated_by_trace = generated_states_by_trace or {}
    rows: list[dict[str, Any]] = []
    for trace_index, trace in enumerate(traces):
        record = TreeRecord.from_value(trace)
        if not record.nodes:
            metadata = metadata_of(trace)
            if metadata.get("observed") is False:
                continue
            target = target_from_trace(trace, config=config, parse_target=parse_target)
            reference = trace_reference_state(
                trace,
                metadata,
                allow_identity=bool(config.allow_identity_g_targets),
            )
            if target is None or not reference:
                continue
            identity = not has_explicit_trace_reference(trace, metadata)
            call_role = explicit_g_call_role(metadata, default="leaf")
            default_supervision = "leaf" if call_role == "leaf" else "merge"
            supervision_role = explicit_supervision_role(metadata, default=default_supervision)
            weight = example_weight(config, metadata, supervision_role=supervision_role)
            if weight["effective_weight"] <= 0.0:
                continue
            rows.append(
                {
                    "prompt": trace_g_prompt(trace, metadata),
                    "state": reference,
                    "target_json": target_json(target),
                    "role": call_role,
                    "supervision_role": supervision_role,
                    "row_id": row_id(trace, index=trace_index, role=call_role),
                    "c3_eligible": call_role == "merge",
                    "group_id": group_id(trace, metadata, fallback=trace_index),
                    "source_split": split_name(metadata),
                    **weight,
                    "identity_g_target": identity,
                    "g_target_source": g_target_source(config, identity),
                    "scheduled_sampling_rate": rate,
                    "used_generated_children": False,
                    "inputs": ("prompt",),
                }
            )
            continue

        topology = binary_topology(record)
        root_id = str(topology.root.node_id)
        generated_states = generated_by_trace.get(trace_index, {})
        for node_index, node in enumerate(record.nodes):
            metadata = merged_metadata(record, node)
            if metadata.get("observed") is False:
                continue
            children = topology.children(node)
            call_role = (
                "leaf" if not children else "recompression" if len(children) == 1 else "merge"
            )
            supervision_role = node_supervision_role(
                node,
                root_id=root_id,
                children=children,
            )
            weight = example_weight(
                config,
                metadata,
                supervision_role=supervision_role,
            )
            if weight["effective_weight"] <= 0.0:
                continue
            target = (
                target_from_trace(trace, config=config, parse_target=parse_target)
                if str(node.node_id) == root_id
                else target_from_node(node, config=config, parse_target=parse_target)
            )
            if target is None:
                continue
            used_generated_children = False
            if call_role == "leaf":
                prompt = node_prompt(node, metadata) or leaf_prompt(str(node.text or ""))
                reference = reference_state(
                    node,
                    fallback=str(node.text or ""),
                    allow_fallback=bool(config.allow_identity_g_targets),
                )
            else:
                child_states = [
                    reference_state(
                        child,
                        fallback=str(child.text or ""),
                        allow_fallback=bool(config.allow_identity_g_targets),
                    )
                    for child in children
                ]
                sampled_child_states = list(child_states)
                for child_index, child in enumerate(children):
                    generated = str(generated_states.get(str(child.node_id)) or "").strip()
                    if generated and scheduled_sampling_selects_child(
                        rate=rate,
                        seed=scheduled_sampling_seed,
                        tree_id=str(record.tree_id),
                        child_id=str(child.node_id),
                    ):
                        sampled_child_states[child_index] = generated
                        used_generated_children = True
                explicit_prompt = node_prompt(node, metadata)
                if used_generated_children:
                    prompt = merge_prompt(
                        sampled_child_states[0] if sampled_child_states else "",
                        (sampled_child_states[1] if len(sampled_child_states) > 1 else None),
                    )
                else:
                    prompt = explicit_prompt or merge_prompt(
                        child_states[0] if child_states else "",
                        child_states[1] if len(child_states) > 1 else None,
                    )
                reference = reference_state(
                    node,
                    fallback="\n".join(value for value in child_states if value),
                    allow_fallback=bool(config.allow_identity_g_targets),
                )
            if not reference:
                continue
            identity = not has_explicit_node_reference(node)
            rows.append(
                {
                    "prompt": prompt,
                    "state": reference,
                    "target_json": target_json(target),
                    "role": call_role,
                    "supervision_role": supervision_role,
                    "row_id": (
                        f"{record.tree_id}:{node.node_id}:{call_role}:{trace_index}:{node_index}"
                    ),
                    "group_id": group_id(record, metadata, fallback=trace_index),
                    "source_split": split_name(metadata),
                    **weight,
                    "identity_g_target": identity,
                    "g_target_source": g_target_source(config, identity),
                    "c3_eligible": call_role == "merge",
                    "scheduled_sampling_rate": rate,
                    "used_generated_children": used_generated_children,
                    "inputs": ("prompt",),
                }
            )
    return rows


def split_examples(
    rows: Sequence[Any],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any], Mapping[str, Any]]:
    """Split whole source groups, excluding test rows and preventing leakage."""

    all_grouped: dict[str, list[Any]] = {}
    for row in rows:
        all_grouped.setdefault(str(getattr(row, "group_id", "")), []).append(row)
    all_group_splits: dict[str, set[str]] = {
        key: {split_of(row) for row in values if split_of(row)}
        for key, values in all_grouped.items()
    }
    conflicts = {key: sorted(values) for key, values in all_group_splits.items() if len(values) > 1}
    if conflicts:
        raise ValueError(
            f"DSPy source groups span multiple explicit splits; refusing leakage: {conflicts!r}"
        )
    test_groups = {key for key, values in all_group_splits.items() if values == {"test"}}
    eligible = [row for row in rows if str(getattr(row, "group_id", "")) not in test_groups]
    if not eligible:
        raise ValueError("DSPy optimizer examples are all marked split='test'; refusing leakage")
    grouped = {key: values for key, values in all_grouped.items() if key not in test_groups}
    group_splits = {
        key: values for key, values in all_group_splits.items() if key not in test_groups
    }

    val_labels = {"val", "validation", "dev"}
    explicit_val_groups = {
        key for key, values in group_splits.items() if values and next(iter(values)) in val_labels
    }
    has_explicit = bool(explicit_val_groups)
    if has_explicit:
        val_groups = explicit_val_groups
        train_groups = set(grouped) - val_groups
        strategy = "explicit_split_metadata_grouped"
    else:
        group_ids = sorted(grouped)
        random.Random(int(seed)).shuffle(group_ids)
        n_val = int(round(len(group_ids) * float(validation_fraction)))
        if validation_fraction > 0.0 and len(group_ids) > 1:
            n_val = max(1, min(len(group_ids) - 1, n_val))
        else:
            n_val = 0
        val_groups = set(group_ids[:n_val])
        train_groups = set(group_ids[n_val:])
        strategy = "seeded_group_split"

        moved_roles: list[str] = []
        roles = sorted(
            {str(getattr(row, "role", "")) for row in eligible if getattr(row, "role", None)}
        )
        for role in roles:
            if any(
                str(getattr(row, "role", "")) == role
                for key in train_groups
                for row in grouped[key]
            ):
                continue
            candidate = next(
                (
                    key
                    for key in sorted(val_groups)
                    if any(str(getattr(row, "role", "")) == role for row in grouped[key])
                ),
                None,
            )
            if candidate is not None:
                val_groups.remove(candidate)
                train_groups.add(candidate)
                moved_roles.append(role)
        if moved_roles:
            strategy += "_role_covered"

    trainset = [row for row in eligible if str(getattr(row, "group_id", "")) in train_groups]
    valset = [row for row in eligible if str(getattr(row, "group_id", "")) in val_groups]
    if not trainset:
        raise ValueError("DSPy optimizer split produced an empty trainset")
    if train_groups & val_groups:
        raise RuntimeError("DSPy internal error: train/validation groups overlap")
    split = {
        "strategy": strategy,
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "eligible_count": len(eligible),
        "excluded_test_count": len(rows) - len(eligible),
        "train_count": len(trainset),
        "validation_count": len(valset),
        "train_group_count": len(train_groups),
        "validation_group_count": len(val_groups),
        "groups_disjoint": True,
        "train_row_ids_sha256": rows_digest(trainset),
        "validation_row_ids_sha256": rows_digest(valset),
    }
    return trainset, valset, split


def example_weight(
    config: Any,
    metadata: Mapping[str, Any],
    *,
    supervision_role: str,
) -> dict[str, Any]:
    """Return one effective weight with IPW and role weight applied once."""

    resolved_propensity = resolve_effective_propensity(metadata)
    propensity = resolved_propensity.propensity
    logged_source = metadata.get("propensity_source")
    propensity_source = (
        str(logged_source).strip()
        if logged_source is not None and str(logged_source).strip()
        else resolved_propensity.source
    )
    min_propensity = float(config.min_propensity)
    if propensity < min_propensity:
        raise ValueError(
            f"DSPy example propensity {propensity!r} is below min_propensity={min_propensity!r}"
        )
    if "node_weight" in metadata or "weight" in metadata:
        raw = nonnegative_weight(metadata.get("node_weight", metadata.get("weight", 1.0)))
        base_effective = raw / propensity
        source = "raw_weight_over_propensity"
    elif metadata.get("sample_weight") is not None:
        # Legacy preference adapters retain an already IPW-corrected value.
        # Never divide that value by the propensity a second time.
        raw = nonnegative_weight(metadata.get("sample_weight"))
        base_effective = raw
        source = "precomputed_sample_weight"
    else:
        raw = 1.0
        base_effective = raw / propensity
        source = "unit_weight_over_propensity"

    role_weight = nonnegative_weight(getattr(config, f"{str(supervision_role)}_weight", 1.0))
    effective = float(base_effective) * role_weight
    cap = config.importance_weight_cap
    if cap is not None:
        effective = min(effective, float(cap))
    return {
        "raw_weight": float(raw),
        "propensity": float(propensity),
        "supervision_weight": float(role_weight),
        "effective_weight": float(effective),
        "propensity_source": propensity_source,
        "weight_source": (
            f"{source}_times_{supervision_role}_weight[propensity_source={propensity_source}]"
        ),
    }


def target_from_trace(trace: Any, *, config: Any, parse_target: ParseTarget) -> Any | None:
    metadata = metadata_of(trace)
    record = TreeRecord.from_value(trace)
    candidates: list[Any] = []
    for value in (
        getattr(trace, "value", None),
        getattr(trace, "oracle_target", None),
        metadata.get("oracle_target"),
    ):
        state = state_from_value(value)
        if isinstance(state, TaskState):
            candidates.append(target_mapping(state.measures, config=config))
    for key in (
        config.target_vector_key,
        config.target_key,
        "oracle_target",
        "target_vector",
        "target_scores",
        "topic_proportions",
        "teacher_score_native",
        "expert_score",
    ):
        if key:
            candidates.extend((metadata.get(str(key)), getattr(trace, str(key), None)))
    candidates.extend(
        (
            getattr(trace, "value", None),
            getattr(trace, "root_label", None),
            record.root_label,
        )
    )
    document_target = first_target(
        candidates,
        config=config,
        parse_target=parse_target,
    )
    if document_target is not None:
        return document_target
    root = record.root()
    if root is None:
        return None
    # Root-node routing is deliberately lazy. A valid document/root label is
    # authoritative and must not force an exclusive node-target lookup; when
    # the document label is absent, the ordinary node contract still applies.
    return target_from_node(root, config=config, parse_target=parse_target)


def target_from_node(node: TreeNode, *, config: Any, parse_target: ParseTarget) -> Any | None:
    return first_target(
        target_candidates_from_node(node, config=config),
        config=config,
        parse_target=parse_target,
    )


def target_candidates_from_node(node: TreeNode, *, config: Any) -> list[Any]:
    metadata = metadata_of(node)
    state = state_from_value(node.state)
    if bool(getattr(config, "node_target_exclusive", False)):
        key = str(getattr(config, "node_target_key", None) or "").strip()
        if not key:
            raise ValueError("node_target_exclusive=True requires node_target_key")
        if key == "state.measures":
            value = state.measures if isinstance(state, TaskState) else None
            return [target_mapping(value, config=config)]
        if key == "state.metadata.target_scores":
            value = state.metadata.get("target_scores") if isinstance(state, TaskState) else None
            return [target_mapping(value, config=config)]
        value = metadata.get(key)
        if key == "target_scores":
            value = target_mapping(value, config=config)
        return [value, getattr(node, key, None)]

    candidates: list[Any] = []
    if isinstance(state, TaskState):
        candidates.append(target_mapping(state.measures, config=config))
        candidates.append(target_mapping(state.metadata.get("target_scores"), config=config))
    for key in (
        config.node_target_key,
        config.target_vector_key,
        "oracle_target",
        "target_vector",
        "target_scores",
        "topic_proportions",
        "score",
        "oracle_score",
        "teacher_score_native",
    ):
        if key:
            value = metadata.get(str(key))
            if str(key) == "target_scores":
                value = target_mapping(value, config=config)
            candidates.extend((value, getattr(node, str(key), None)))
    candidates.append(node.label)
    return candidates


def first_target(
    candidates: Sequence[Any],
    *,
    config: Any,
    parse_target: ParseTarget,
) -> Any | None:
    for candidate in candidates:
        if candidate is None:
            continue
        parsed = parse_target(candidate)
        if parsed is not None:
            return parsed
    return None


def target_mapping(value: Any, *, config: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    names = tuple(str(name) for name in (config.target_names or ()))
    if names and all(name in value for name in names):
        return {name: value[name] for name in names}
    return value


def reference_state(node: TreeNode, *, fallback: str, allow_fallback: bool) -> str:
    state = state_from_value(node.state)
    if isinstance(state, TaskState) and str(state.text or "").strip():
        return str(state.text).strip()
    if isinstance(state, str) and state.strip():
        return state.strip()
    metadata = metadata_of(node)
    for key in (
        "summary",
        "teacher_summary",
        "target_summary",
        "reference_summary",
        "completion",
        "response",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if not allow_fallback:
        return ""
    return str(node.text or fallback or "").strip()


def has_explicit_node_reference(node: TreeNode) -> bool:
    state = state_from_value(node.state)
    if isinstance(state, TaskState):
        return bool(str(state.text or "").strip())
    if isinstance(state, str) and state.strip():
        return True
    return any(
        metadata_of(node).get(key) is not None and str(metadata_of(node).get(key)).strip()
        for key in (
            "summary",
            "teacher_summary",
            "target_summary",
            "reference_summary",
            "completion",
            "response",
        )
    )


def trace_reference_state(trace: Any, metadata: Mapping[str, Any], *, allow_identity: bool) -> str:
    for value in (getattr(trace, "value", None), getattr(trace, "oracle_target", None)):
        state = state_from_value(value)
        if isinstance(state, TaskState) and str(state.text or "").strip():
            return str(state.text).strip()
    for key in (
        "reference_state",
        "summary",
        "teacher_summary",
        "target_summary",
        "completion",
        "response",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            if key == "completion":
                decoded = state_from_value(getattr(trace, "value", None))
                if isinstance(decoded, TaskState) and decoded.text:
                    return str(decoded.text).strip()
            return str(value).strip()
    state = state_from_value(getattr(trace, "state", None))
    if isinstance(state, TaskState) and state.text:
        return str(state.text).strip()
    if isinstance(state, str) and state.strip():
        return state.strip()
    if str(metadata.get("preference_target") or "").lower() == "g" or allow_identity:
        return trace_text(trace).strip()
    return ""


def has_explicit_trace_reference(trace: Any, metadata: Mapping[str, Any]) -> bool:
    if str(metadata.get("preference_target") or "").lower() == "g":
        return True
    for value in (getattr(trace, "value", None), getattr(trace, "oracle_target", None)):
        state = state_from_value(value)
        if isinstance(state, TaskState) and str(state.text or "").strip():
            return True
    if any(
        metadata.get(key) is not None and str(metadata.get(key)).strip()
        for key in (
            "reference_state",
            "summary",
            "teacher_summary",
            "target_summary",
            "completion",
            "response",
        )
    ):
        return True
    state = state_from_value(getattr(trace, "state", None))
    return (isinstance(state, TaskState) and bool(state.text)) or (
        isinstance(state, str) and bool(state.strip())
    )


def f_trace_state(trace: Any, metadata: Mapping[str, Any]) -> str:
    # Preference f rows historically concatenate prompt and completion into
    # ``text``. The completion is supervision, never an f input.
    for value in (getattr(trace, "value", None), getattr(trace, "oracle_target", None)):
        state = state_from_value(value)
        if isinstance(state, TaskState) and str(state.text or "").strip():
            return str(state.text).strip()
    if str(metadata.get("preference_target") or "").lower() == "f":
        prompt = getattr(trace, "prompt", None) or metadata.get("prompt")
        return str(prompt or "").strip()
    return trace_text(trace).strip()


def trace_g_prompt(trace: Any, metadata: Mapping[str, Any]) -> str:
    prompt = getattr(trace, "prompt", None) or metadata.get("prompt")
    if prompt is not None and str(prompt).strip():
        return str(prompt).strip()
    return leaf_prompt(trace_text(trace))


def node_prompt(node: TreeNode, metadata: Mapping[str, Any]) -> str:
    value = metadata.get("prompt") or metadata.get("g_prompt")
    return str(value or "").strip()


def leaf_prompt(text: str) -> str:
    return (
        "[TREEPO_G_CALL=leaf]\n"
        "Apply the shared C-Tree state operator g to this leaf. Return only the state.\n"
        f"LEAF_TEXT:\n{text}"
    )


def merge_prompt(left: str, right: str | None) -> str:
    if right is None:
        return (
            "[TREEPO_G_CALL=recompression]\n"
            "Promote this odd carry with the same shared C-Tree state operator g. "
            "Do not fabricate a second child. Return only the parent state.\n"
            f"CHILD_STATE:\n{left}"
        )
    return (
        "[TREEPO_G_CALL=merge]\n"
        "Apply the same shared C-Tree state operator g to these two child states. "
        "Return only the merged state.\n"
        f"LEFT_STATE:\n{left}\n\nRIGHT_STATE:\n{right}"
    )


def scheduled_sampling_selects_child(
    *,
    rate: float,
    seed: int,
    tree_id: str,
    child_id: str,
) -> bool:
    """Return a process- and traversal-stable Bernoulli choice for one child."""

    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    payload = json.dumps(
        [int(seed), str(tree_id), str(child_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(1 << 64)
    return draw < float(rate)


def explicit_g_call_role(metadata: Mapping[str, Any], *, default: str) -> str:
    value = metadata.get("call_role", metadata.get("g_call_role", default))
    role = str(value or default).strip().lower()
    if role not in {"leaf", "recompression", "merge"}:
        raise ValueError(
            f"DSPy g call_role must be 'leaf', 'recompression', or 'merge'; got {value!r}"
        )
    return role


def explicit_supervision_role(metadata: Mapping[str, Any], *, default: str) -> str:
    value = metadata.get("supervision_role", default)
    role = str(value or default).strip().lower()
    if role not in {"root", "leaf", "merge"}:
        raise ValueError(f"DSPy supervision_role must be 'root', 'leaf', or 'merge'; got {value!r}")
    return role


def node_supervision_role(
    node: TreeNode,
    *,
    root_id: str,
    children: Sequence[TreeNode],
) -> str:
    if str(node.node_id) == str(root_id):
        return "root"
    return "leaf" if not children else "merge"


def metadata_of(value: Any) -> dict[str, Any]:
    metadata = (
        value.get("metadata") if isinstance(value, Mapping) else getattr(value, "metadata", None)
    )
    return dict(metadata or {}) if isinstance(metadata, Mapping) else {}


def merged_metadata(record: TreeRecord, node: TreeNode) -> dict[str, Any]:
    metadata = dict(record.metadata or {})
    metadata.update(dict(node.metadata or {}))
    return metadata


def trace_text(trace: Any) -> str:
    if isinstance(trace, Mapping):
        value = trace.get("text", trace.get("content", trace.get("document_text", "")))
    else:
        value = getattr(
            trace,
            "text",
            getattr(trace, "content", getattr(trace, "document_text", "")),
        )
    return str(value or metadata_of(trace).get("text") or "")


def group_id(value: Any, metadata: Mapping[str, Any], *, fallback: Any) -> str:
    for candidate in (
        metadata.get("group_id"),
        metadata.get("split_group"),
        metadata.get("source_doc_id"),
        getattr(value, "tree_id", None),
        getattr(value, "doc_id", None),
        metadata.get("doc_id"),
        metadata.get("tree_id"),
        metadata.get("preference_unit_id"),
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return f"row-group:{fallback}"


def row_id(trace: Any, *, index: int, role: str) -> str:
    metadata = metadata_of(trace)
    value = (
        metadata.get("preference_unit_id")
        or metadata.get("row_id")
        or metadata.get("tree_id")
        or metadata.get("doc_id")
        or getattr(trace, "tree_id", None)
        or getattr(trace, "doc_id", None)
        or index
    )
    return f"{value}:{role}:{index}"


def split_name(metadata: Mapping[str, Any]) -> str:
    return str(metadata.get("split", metadata.get("source_split", "")) or "").strip().lower()


def split_of(row: Any) -> str:
    return str(getattr(row, "source_split", "") or "").strip().lower()


def g_target_source(config: Any, identity: bool) -> str:
    if config.g_target_source:
        return str(config.g_target_source)
    return "identity_reference_text_opt_in" if identity else "explicit_reference_state"


def target_json(value: Any) -> str:
    return json.dumps(state_to_dict(value), separators=(",", ":"), sort_keys=True)


def rows_digest(rows: Sequence[Any]) -> str:
    payload = json.dumps(
        [str(getattr(row, "row_id", "")) for row in rows],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nonnegative_weight(value: Any) -> float:
    number = finite_number(value)
    if number is None or number < 0.0:
        raise ValueError(f"DSPy example weight must be finite and non-negative, got {value!r}")
    return float(number)


__all__ = [
    "BinaryTopology",
    "binary_topology",
    "f_training_records",
    "g_training_records",
    "leaf_prompt",
    "merge_prompt",
    "split_examples",
]
