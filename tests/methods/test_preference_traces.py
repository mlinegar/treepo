from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo import fit
from treepo.methods._preference_traces import preference_training_rows
from treepo.methods.dspy import build_dspy_family
from treepo.methods.preference import (
    Candidate,
    PreferenceDataset,
    PreferenceRecord,
    preference_units_from_trees,
)
from treepo.state import TaskState
from treepo.tree import TreeNode, TreeRecord


class _Program:
    def __init__(self, kind: str) -> None:
        self.kind = str(kind)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.kind == "f":
            return {"prediction_json": json.dumps({"score": 0.0})}
        return {"state": "compiled-state"}

    def save(self, path: str, save_program: bool = False) -> None:
        del save_program
        Path(path).write_text(json.dumps({"kind": self.kind}), encoding="utf-8")


class _Compiler:
    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []

    def compile(self, *, program, metric, trainset, valset, kind, **kwargs):
        del metric, kwargs
        self.calls.append(SimpleNamespace(kind=kind, trainset=list(trainset), valset=list(valset)))
        return program


def _load_program(*, path: Path, kind: str, **kwargs) -> _Program:
    del path, kwargs
    return _Program(kind)


def test_g_preference_trace_preserves_text_state_and_sampling_structure() -> None:
    dataset = PreferenceDataset.from_records(
        (
            PreferenceRecord(
                unit_id="tree:merge",
                unit_type="merge",
                target="g",
                context="  Merge left and right exactly.  ",
                weight=2.0,
                propensity=0.25,
                metadata={"call_role": "merge", "tag": "kept"},
                candidates=(
                    Candidate(
                        id="best",
                        value="  compressed state  ",
                        score=0.9,
                        preferred=True,
                    ),
                ),
            ),
        )
    )

    (row,) = preference_training_rows(dataset, target="g")

    assert row.prompt == "  Merge left and right exactly.  "
    assert row.completion == "  compressed state  "
    assert row.text == "Merge left and right exactly.\ncompressed state"
    assert row.value == "  compressed state  "
    assert row.oracle_target == "  compressed state  "
    assert row.document_score is None
    assert row.weight == 2.0
    assert row.propensity == 0.25
    assert row.sample_weight == 8.0
    assert row.call_role == "merge"
    assert row.metadata["prompt"] == row.prompt
    assert row.metadata["completion"] == row.completion
    assert row.metadata["oracle_target"] == row.oracle_target
    assert row.metadata["weight"] == 2.0
    assert row.metadata["propensity"] == 0.25
    assert row.metadata["sample_weight"] == 8.0
    assert row.metadata["call_role"] == "merge"
    assert row.metadata["tag"] == "kept"


def test_g_preference_trace_infers_leaf_role_and_preserves_task_state() -> None:
    state = TaskState(
        kind="manifesto_state",
        counts={"left": 2.0, "right": 1.0},
        text="state summary",
    )
    dataset = PreferenceDataset.from_records(
        (
            PreferenceRecord(
                unit_id="tree:leaf",
                unit_type="leaf",
                target="g",
                context={"prompt": "Encode this leaf."},
                candidates=(Candidate(id="gold", value=state, preferred=True),),
            ),
        )
    )

    (row,) = preference_training_rows(dataset, target="g")

    assert row.call_role == "leaf"
    assert row.metadata["call_role"] == "leaf"
    assert row.prompt == "Encode this leaf."
    assert row.oracle_target["kind"] == "manifesto_state"
    assert row.oracle_target["counts"] == {"left": 2.0, "right": 1.0}
    assert row.completion


def test_f_preference_trace_keeps_numeric_label_behavior() -> None:
    dataset = PreferenceDataset.from_records(
        (
            PreferenceRecord(
                unit_id="tree:root",
                unit_type="root",
                target="f",
                context="Read out the root.",
                weight=1.5,
                propensity=0.5,
                candidates=(Candidate(id="gold", value=2.5, score=0.7),),
            ),
        )
    )

    (row,) = preference_training_rows(dataset, target="f")

    assert row.document_score == 2.5
    assert row.oracle_target == 2.5
    assert row.sample_weight == 3.0
    assert row.propensity == 0.5
    assert row.call_role is None
    assert row.metadata["teacher_score_native"] == 2.5


def _state(text: str) -> TaskState:
    return TaskState(kind="fixture", text=text, measures={"score": 0.0})


def _g_tree(tree_id: str, *, child_count: int) -> TreeRecord:
    children = tuple(
        TreeNode(
            node_id=f"child-{index}",
            unit_type="leaf",
            text=f"raw child {index}",
            parent_id="root",
            state=_state(f"child-state-{tree_id}-{index}"),
            metadata={"split": "train"},
        )
        for index in range(child_count)
    )
    root = TreeNode(
        node_id="root",
        unit_type="root",
        text=f"raw root {tree_id}",
        state=_state(f"root-state-{tree_id}"),
        metadata={"split": "train"},
    )
    return TreeRecord(
        tree_id=tree_id,
        text=f"document {tree_id}",
        root_label={"score": 0.0},
        nodes=(*children, root),
        metadata={"split": "train"},
    )


def _malformed_g_tree(case: str) -> TreeRecord:
    child = TreeNode(node_id="child", unit_type="leaf", state=_state("child-state"))
    if case == "duplicate_child":
        nodes = (
            TreeNode(
                node_id="root",
                unit_type="root",
                left_child_id="child",
                right_child_id="child",
                state=_state("root-state"),
            ),
            child,
        )
    elif case == "shared_child":
        nodes = (
            TreeNode(
                node_id="root",
                unit_type="root",
                left_child_id="left",
                right_child_id="right",
                state=_state("root-state"),
            ),
            TreeNode(
                node_id="left",
                parent_id="root",
                left_child_id="child",
                state=_state("left-state"),
            ),
            TreeNode(
                node_id="right",
                parent_id="root",
                left_child_id="child",
                state=_state("right-state"),
            ),
            child,
        )
    elif case == "inconsistent_endpoints":
        nodes = (
            TreeNode(
                node_id="root",
                unit_type="root",
                left_child_id="child",
                state=_state("root-state"),
            ),
            TreeNode(node_id="other", state=_state("other-state")),
            TreeNode(
                node_id="child",
                parent_id="other",
                state=_state("child-state"),
            ),
        )
    elif case == "cycle":
        nodes = (
            TreeNode(
                node_id="a",
                unit_type="root",
                parent_id="b",
                left_child_id="b",
                state=_state("a-state"),
            ),
            TreeNode(
                node_id="b",
                parent_id="a",
                left_child_id="a",
                state=_state("b-state"),
            ),
        )
    elif case == "disconnected_root":
        nodes = (
            TreeNode(node_id="root", unit_type="root", state=_state("root-state")),
            TreeNode(node_id="orphan", unit_type="leaf", state=_state("orphan-state")),
        )
    elif case == "three_children":
        nodes = (
            TreeNode(node_id="root", unit_type="root", state=_state("root-state")),
            *(
                TreeNode(
                    node_id=f"child-{index}",
                    parent_id="root",
                    state=_state(f"child-{index}-state"),
                )
                for index in range(3)
            ),
        )
    else:  # pragma: no cover - test helper contract
        raise AssertionError(case)
    return TreeRecord(tree_id=f"malformed-{case}", nodes=nodes)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("duplicate_child", "duplicates child edge"),
        ("shared_child", "multiple parents"),
        ("inconsistent_endpoints", "points to parent"),
        ("cycle", "connected root"),
        ("disconnected_root", "outside the root topology"),
        ("three_children", "3 supplied children"),
    ],
)
def test_tree_preference_export_rejects_malformed_topology(
    case: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        preference_units_from_trees((_malformed_g_tree(case),), target="g")


def _dspy_config(compiler: _Compiler) -> dict[str, object]:
    return {
        "target_names": ("score",),
        "target_oracle_ids": ("fixture:score",),
        "target_dim": 1,
        "target_vector_key": "target_vector",
        "node_target_key": "state.measures",
        "target_min": -1.0,
        "target_max": 1.0,
        "optimizer": "bootstrap",
        "validation_fraction": 0.0,
        "allow_identity_g_targets": False,
        "audit_laws": False,
        "root_weight": 11.0,
        "leaf_weight": 3.0,
        "merge_weight": 7.0,
        "f_program": _Program("f"),
        "g_program": _Program("g"),
        "compiler": compiler,
        "program_loader": _load_program,
    }


def test_tree_g_preferences_preserve_unary_binary_topology_and_child_states() -> None:
    unary = _g_tree("unary", child_count=1)
    binary = _g_tree("binary", child_count=2)
    dataset = preference_units_from_trees((unary, binary), target="g")
    supervised = {str(row["unit_id"]): row for row in dataset.to_records("supervised")}
    assert supervised["unary:root"]["metadata"]["call_role"] == "recompression"
    assert supervised["binary:root"]["metadata"]["call_role"] == "merge"
    assert supervised["unary:root"]["metadata"]["supervision_role"] == "root"
    assert supervised["binary:root"]["metadata"]["supervision_role"] == "root"
    assert supervised["unary:child-0"]["metadata"]["parent_id"] == "root"

    traces = preference_training_rows(dataset, target="g")
    by_unit = {str(row.unit_id): row for row in traces}
    unary_root = by_unit["unary:root"]
    binary_root = by_unit["binary:root"]
    assert unary_root.call_role == "recompression"
    assert binary_root.call_role == "merge"
    assert unary_root.prompt.startswith("[TREEPO_G_CALL=recompression]")
    assert "child-state-unary-0" in unary_root.prompt
    assert "raw root unary" not in unary_root.prompt
    assert binary_root.prompt.startswith("[TREEPO_G_CALL=merge]")
    assert "child-state-binary-0" in binary_root.prompt
    assert "child-state-binary-1" in binary_root.prompt

    compiler = _Compiler()
    family = build_dspy_family(_dspy_config(compiler))
    examples = family._dspy_g_examples(traces)
    unary_example = next(row for row in examples if str(row.row_id).startswith("unary:root:"))
    binary_example = next(row for row in examples if str(row.row_id).startswith("binary:root:"))
    assert (unary_example.role, unary_example.supervision_role, unary_example.c3_eligible) == (
        "recompression",
        "root",
        False,
    )
    assert (binary_example.role, binary_example.supervision_role, binary_example.c3_eligible) == (
        "merge",
        "root",
        True,
    )


def test_treepo_fit_retains_preference_recompression_and_merge_domains(tmp_path: Path) -> None:
    unary = _g_tree("unary-fit", child_count=1)
    binary = _g_tree("binary-fit", child_count=2)
    compiler = _Compiler()
    backend = _dspy_config(compiler)
    backend["output_dir"] = str(tmp_path / "fit")
    preferences = preference_units_from_trees((unary, binary), target="g")
    result = fit(
        {
            "family": "dspy",
            "train_data": [unary, binary],
            "eval_data": [unary, binary],
            "preference_data": preferences,
            "oracle_targets": [{"target_name": "score", "oracle_id": "fixture:score"}],
            "backend_config": backend,
            "axis": {"max_iterations": 2, "axis_value": 2},
        }
    )

    assert result.status == "success"
    artifact = result.artifacts["g"]
    assert artifact["g_training_call_roles"] == ["leaf", "recompression", "merge"]
    assert artifact["recompression_training_row_count"] == 1
    assert artifact["merge_domain_training_row_count"] == 1
    assert result.summary["g_contract"]["merge_domain_training_observed"] is True
    g_call = next(call for call in compiler.calls if call.kind == "g")
    roots = [row for row in g_call.trainset if ":root:" in str(row.row_id)]
    assert {row.role for row in roots} == {"recompression", "merge"}
    assert {row.supervision_role for row in roots} == {"root"}
    assert {row.role: row.c3_eligible for row in roots} == {"recompression": False, "merge": True}
