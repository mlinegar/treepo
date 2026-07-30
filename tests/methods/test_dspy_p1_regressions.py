from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo.methods.dspy import DSPyFamilyConfig, build_dspy_family
from treepo.state import TaskState
from treepo.tree import TreeNode, TreeRecord

TARGET = {"score": 0.75}


class _Program:
    def __init__(self, kind: str) -> None:
        self.kind = str(kind)

    def __call__(self, **_kwargs):
        if self.kind == "f":
            return {"prediction_json": json.dumps(TARGET)}
        return {"state": "candidate-state"}

    def save(self, path: str, save_program: bool = False) -> None:
        Path(path).write_text(
            json.dumps({"kind": self.kind, "save_program": bool(save_program)}),
            encoding="utf-8",
        )


class _Compiler:
    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []

    def compile(self, *, program, trainset, valset, kind, **_kwargs):
        self.calls.append(
            SimpleNamespace(
                program=program,
                trainset=list(trainset),
                valset=list(valset),
                kind=str(kind),
            )
        )
        return program


def _load_program(*, path: Path, **_kwargs):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _Program(str(payload["kind"]))


def _config(**overrides):
    values = {
        "target_names": ("score",),
        "target_oracle_ids": ("fixture:score",),
        "target_dim": 1,
        "target_vector_key": "target_vector",
        "node_target_key": "target_vector",
        "optimizer": "bootstrap",
        "validation_fraction": 0.0,
        "allow_identity_g_targets": False,
        "audit_laws": False,
        "f_program": _Program("f"),
        "g_program": _Program("g"),
        "program_loader": _load_program,
    }
    values.update(overrides)
    return values


def _preference_trace(
    *,
    unit_id: str,
    split: str = "train",
    group_id: str | None = None,
    preference_target: str = "f",
    **metadata,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=f"candidate::{unit_id}",
        prompt=f"prompt::{unit_id}",
        metadata={
            "oracle_target": dict(TARGET),
            "preference_target": preference_target,
            "preference_unit_id": unit_id,
            "group_id": group_id or unit_id,
            "split": split,
            **metadata,
        },
    )


def _singleton_tree(
    *,
    tree_id: str = "singleton",
    node_metadata: dict[str, object] | None = None,
) -> TreeRecord:
    node = TreeNode(
        node_id="root",
        unit_type="root",
        text="raw root",
        state=TaskState(
            kind="fixture",
            text="gold state",
            measures=dict(TARGET),
        ),
        metadata={"split": "train", **dict(node_metadata or {})},
    )
    return TreeRecord(
        tree_id=tree_id,
        text="whole document",
        nodes=(node,),
        metadata={"split": "train"},
    )


@pytest.mark.parametrize(
    ("call_role", "c3_eligible"),
    [("merge", True), ("recompression", False)],
)
def test_nodeless_g_preference_preserves_call_role_and_uses_merge_weight(
    call_role: str,
    c3_eligible: bool,
) -> None:
    family = build_dspy_family(_config(root_weight=11.0, leaf_weight=3.0, merge_weight=7.0))
    trace = _preference_trace(
        unit_id=call_role,
        preference_target="g",
        call_role=call_role,
        weight=2.0,
        propensity=0.5,
    )

    rows = family._dspy_g_examples([trace])

    assert len(rows) == 1
    assert rows[0].role == call_role
    assert rows[0].supervision_role == "merge"
    assert rows[0].supervision_weight == pytest.approx(7.0)
    assert rows[0].effective_weight == pytest.approx((2.0 / 0.5) * 7.0)
    assert rows[0].c3_eligible is c3_eligible


def test_explicit_f_supervision_role_overrides_structural_default() -> None:
    family = build_dspy_family(_config(root_weight=11.0, leaf_weight=3.0, merge_weight=7.0))
    preference = _preference_trace(
        unit_id="preference",
        supervision_role="leaf",
        weight=2.0,
        propensity=0.5,
    )
    tree = _singleton_tree(
        node_metadata={
            "supervision_role": "leaf",
            "weight": 2.0,
            "propensity": 0.5,
        }
    )

    preference_row = family._dspy_f_examples([preference], g=None)[0]
    node_row = family._dspy_f_examples([tree], g=None)[0]

    assert preference_row.role == "leaf"
    assert node_row.role == "leaf"
    assert preference_row.supervision_role == "leaf"
    assert node_row.supervision_role == "leaf"
    assert preference_row.effective_weight == pytest.approx((2.0 / 0.5) * 3.0)
    assert node_row.effective_weight == pytest.approx((2.0 / 0.5) * 3.0)


def test_group_spanning_train_and_test_fails_before_compiler(tmp_path: Path) -> None:
    compiler = _Compiler()
    family = build_dspy_family(_config(compiler=compiler))
    traces = [
        _preference_trace(unit_id="train", group_id="shared", split="train"),
        _preference_trace(unit_id="test", group_id="shared", split="test"),
    ]

    with pytest.raises(ValueError, match="span multiple explicit splits"):
        family.train_f(
            f_init=None,
            g=None,
            traces=traces,
            output_dir=tmp_path,
            iteration=1,
        )

    assert compiler.calls == []


def test_whole_test_only_group_is_excluded_from_compiler(tmp_path: Path) -> None:
    compiler = _Compiler()
    family = build_dspy_family(_config(compiler=compiler))
    traces = [
        _preference_trace(unit_id="train", group_id="training-party"),
        _preference_trace(unit_id="test-a", group_id="held-party", split="test"),
        _preference_trace(unit_id="test-b", group_id="held-party", split="test"),
    ]

    artifact = family.train_f(
        f_init=None,
        g=None,
        traces=traces,
        output_dir=tmp_path,
        iteration=1,
    )

    assert len(compiler.calls) == 1
    assert {row.group_id for row in compiler.calls[0].trainset} == {"training-party"}
    assert compiler.calls[0].valset == []
    assert artifact["split"]["excluded_test_count"] == 2


def test_exclusive_unrelated_node_key_never_falls_back_to_taskstate_measures() -> None:
    family = build_dspy_family(
        _config(
            node_target_exclusive=True,
            node_target_key="declared_but_missing",
        )
    )

    assert family._dspy_f_examples([_singleton_tree()], g=None) == []
    assert family._dspy_g_examples([_singleton_tree()]) == []


def test_exclusive_state_measures_key_reads_the_declared_taskstate_field() -> None:
    family = build_dspy_family(
        _config(
            node_target_exclusive=True,
            node_target_key="state.measures",
        )
    )

    f_rows = family._dspy_f_examples([_singleton_tree()], g=None)
    g_rows = family._dspy_g_examples([_singleton_tree()])

    assert len(f_rows) == 1
    assert len(g_rows) == 1
    assert json.loads(f_rows[0].target_json) == TARGET
    assert json.loads(g_rows[0].target_json) == TARGET


def test_exclusive_without_key_is_deferred_until_node_supervision_is_consumed() -> None:
    family = build_dspy_family(
        _config(
            node_target_exclusive=True,
            node_target_key=None,
        )
    )
    inference_config = DSPyFamilyConfig(
        target_names=("score",),
        target_oracle_ids=("fixture:score",),
        node_target_exclusive=True,
        optimizer="none",
    )

    # Construction and root-only examples remain valid because no node target
    # lookup is requested.
    assert inference_config.node_target_key is None
    assert (
        len(
            family._dspy_f_examples(
                [_preference_trace(unit_id="root-only")],
                g=None,
            )
        )
        == 1
    )

    with pytest.raises(
        ValueError,
        match="node_target_exclusive=True requires node_target_key",
    ):
        family._dspy_f_examples([_singleton_tree()], g=None)


def test_identity_g_trains_f_on_raw_singleton_x_without_learned_g_target_opt_in(
    tmp_path: Path,
) -> None:
    compiler = _Compiler()
    family = build_dspy_family(
        _config(
            compiler=compiler,
            allow_identity_g_targets=False,
        )
    )
    raw_x = "raw singleton document X"
    tree = TreeRecord(
        tree_id="full-doc-direct",
        text=raw_x,
        root_label=dict(TARGET),
        nodes=(
            TreeNode(
                node_id="root",
                unit_type="root",
                text=raw_x,
                label=dict(TARGET),
                # Deliberately no teacher summary or TaskState text: identity
                # g means f consumes X itself.
                state=None,
                metadata={"split": "train"},
            ),
        ),
        metadata={"split": "train"},
    )
    identity_g = {
        "kind": "treepo_identity_g",
        "g_mode": "identity",
        "operator": "identity",
        "trainable": False,
    }

    rows = family._dspy_f_examples([tree], g=identity_g)
    assert len(rows) == 1
    assert rows[0].state == raw_x

    artifact = family.train_f(
        f_init=None,
        g=identity_g,
        traces=[tree],
        output_dir=tmp_path,
        iteration=1,
    )
    assert artifact["n_train"] == 1
    assert len(compiler.calls) == 1
    assert compiler.calls[0].trainset[0].state == raw_x


def test_identity_g_root_prefers_document_text_and_ignores_cached_summary() -> None:
    raw_x = "raw singleton document X"
    tree = TreeRecord(
        tree_id="identity-ignores-summary",
        text=raw_x,
        root_label=dict(TARGET),
        nodes=(
            TreeNode(
                node_id="root",
                unit_type="root",
                text="raw root-node span that is not the whole document",
                state=TaskState(
                    kind="fixture",
                    text="cached teacher summary that identity g must bypass",
                    measures=dict(TARGET),
                ),
                metadata={"split": "train"},
            ),
        ),
        metadata={"split": "train"},
    )
    family = build_dspy_family(_config())
    identity_g = {
        "kind": "treepo_identity_g",
        "g_mode": "identity",
        "operator": "identity",
        "trainable": False,
    }

    rows = family._dspy_f_examples([tree], g=identity_g)

    assert len(rows) == 1
    assert rows[0].state == raw_x


def test_identity_g_empty_structural_root_uses_document_text() -> None:
    raw_x = "whole raw document X"
    tree = TreeRecord(
        tree_id="identity-root-document-fallback",
        text=raw_x,
        root_label=dict(TARGET),
        nodes=(
            TreeNode(
                node_id="root",
                unit_type="root",
                text="",
                state=TaskState(
                    kind="fixture",
                    text="cached root summary that identity g must bypass",
                    measures=dict(TARGET),
                ),
                metadata={"split": "train"},
            ),
        ),
        metadata={"split": "train"},
    )
    family = build_dspy_family(_config())
    identity_g = {
        "kind": "treepo_identity_g",
        "g_mode": "identity",
        "operator": "identity",
        "trainable": False,
    }

    rows = family._dspy_f_examples([tree], g=identity_g)

    assert len(rows) == 1
    assert rows[0].state == raw_x


def test_identity_g_nodeless_full_document_ignores_cached_taskstate() -> None:
    raw_x = "whole raw node-less document X"
    trace = SimpleNamespace(
        tree_id="identity-node-less",
        text=raw_x,
        value=TaskState(
            kind="fixture",
            text="cached full-document summary that identity g must bypass",
            measures=dict(TARGET),
        ),
        metadata={"split": "train"},
    )
    family = build_dspy_family(_config())
    identity_g = {
        "kind": "treepo_identity_g",
        "g_mode": "identity",
        "operator": "identity",
        "trainable": False,
    }

    identity_rows = family._dspy_f_examples([trace], g=identity_g)
    reference_rows = family._dspy_f_examples([trace], g=None)

    assert len(identity_rows) == 1
    assert identity_rows[0].state == raw_x
    assert reference_rows[0].state == "cached full-document summary that identity g must bypass"


def test_identity_g_preserves_prompt_only_input_for_f_preference_rows() -> None:
    trace = _preference_trace(unit_id="identity-f-preference")
    family = build_dspy_family(_config())
    identity_g = {
        "kind": "treepo_identity_g",
        "g_mode": "identity",
        "operator": "identity",
        "trainable": False,
    }

    rows = family._dspy_f_examples([trace], g=identity_g)

    assert len(rows) == 1
    assert rows[0].state == "prompt::identity-f-preference"
    assert rows[0].state != trace.text


def test_singleton_structural_root_uses_document_root_target_for_f_and_g() -> None:
    root = TreeNode(
        node_id="root",
        unit_type="root",
        text="raw singleton text",
        label=None,
        state=TaskState(
            kind="fixture",
            text="gold singleton summary",
            measures={},
        ),
        metadata={"split": "train"},
    )
    tree = TreeRecord(
        tree_id="root-label-only",
        text="raw singleton text",
        root_label=dict(TARGET),
        nodes=(root,),
        metadata={"split": "train"},
    )
    family = build_dspy_family(_config())

    f_rows = family._dspy_f_examples([tree], g=None)
    g_rows = family._dspy_g_examples([tree])

    assert len(f_rows) == 1
    assert f_rows[0].role == "root"
    assert f_rows[0].state == "gold singleton summary"
    assert json.loads(f_rows[0].target_json) == TARGET
    assert len(g_rows) == 1
    assert g_rows[0].role == "leaf"
    assert g_rows[0].supervision_role == "root"
    assert g_rows[0].state == "gold singleton summary"
    assert json.loads(g_rows[0].target_json) == TARGET


def test_document_root_target_preempts_exclusive_node_fallback() -> None:
    tree = _singleton_tree()
    tree = TreeRecord(
        tree_id=tree.tree_id,
        text=tree.text,
        root_label=dict(TARGET),
        nodes=tree.nodes,
        metadata=tree.metadata,
    )
    family = build_dspy_family(
        _config(
            node_target_exclusive=True,
            node_target_key=None,
        )
    )

    assert len(family._dspy_f_examples([tree], g=None)) == 1
    assert len(family._dspy_g_examples([tree])) == 1


def test_zero_weight_structural_root_skips_exclusive_node_target_lookup() -> None:
    family = build_dspy_family(
        _config(
            root_weight=0.0,
            node_target_exclusive=True,
            node_target_key=None,
        )
    )

    assert family._dspy_f_examples([_singleton_tree()], g=None) == []
    assert family._dspy_g_examples([_singleton_tree()]) == []


def test_document_root_target_is_not_inherited_by_nonroot_nodes() -> None:
    def state(text: str) -> TaskState:
        return TaskState(kind="fixture", text=text, measures={})

    tree = TreeRecord(
        tree_id="root-target-routing",
        text="whole document",
        root_label=dict(TARGET),
        nodes=(
            TreeNode(
                node_id="left",
                unit_type="leaf",
                text="left",
                parent_id="root",
                state=state("left reference"),
            ),
            TreeNode(
                node_id="right",
                unit_type="leaf",
                text="right",
                parent_id="root",
                state=state("right reference"),
            ),
            TreeNode(
                node_id="root",
                unit_type="root",
                text="whole document",
                left_child_id="left",
                right_child_id="right",
                state=state("root reference"),
            ),
        ),
        metadata={"split": "train"},
    )
    family = build_dspy_family(_config())

    f_rows = family._dspy_f_examples([tree], g=None)
    g_rows = family._dspy_g_examples([tree])

    assert [(row.role, row.state) for row in f_rows] == [("root", "root reference")]
    assert [(row.role, row.state) for row in g_rows] == [("merge", "root reference")]
    assert g_rows[0].c3_eligible is True


def test_duplicate_left_right_child_edge_is_rejected() -> None:
    child = TreeNode(
        node_id="child",
        unit_type="leaf",
        text="child",
        parent_id="root",
        state=TaskState(kind="fixture", text="child state", measures=dict(TARGET)),
    )
    root = TreeNode(
        node_id="root",
        unit_type="root",
        text="root",
        left_child_id="child",
        right_child_id="child",
        state=TaskState(kind="fixture", text="root state", measures=dict(TARGET)),
    )
    tree = TreeRecord(tree_id="duplicate-edge", nodes=(child, root))
    family = build_dspy_family(_config(node_target_key="state.measures"))

    with pytest.raises(ValueError, match="duplicates child edge"):
        family._dspy_g_examples([tree])


def test_shared_child_multi_parent_dag_is_rejected() -> None:
    shared = TreeNode(
        node_id="shared",
        unit_type="leaf",
        text="shared",
        state=TaskState(kind="fixture", text="shared state", measures=dict(TARGET)),
    )
    left_parent = TreeNode(
        node_id="left-parent",
        unit_type="merge",
        text="left parent",
        parent_id="root",
        left_child_id="shared",
        state=TaskState(kind="fixture", text="left state", measures=dict(TARGET)),
    )
    right_parent = TreeNode(
        node_id="right-parent",
        unit_type="merge",
        text="right parent",
        parent_id="root",
        left_child_id="shared",
        state=TaskState(kind="fixture", text="right state", measures=dict(TARGET)),
    )
    root = TreeNode(
        node_id="root",
        unit_type="root",
        text="root",
        left_child_id="left-parent",
        right_child_id="right-parent",
        state=TaskState(kind="fixture", text="root state", measures=dict(TARGET)),
    )
    tree = TreeRecord(
        tree_id="shared-child-dag",
        nodes=(shared, left_parent, right_parent, root),
    )
    family = build_dspy_family(_config(node_target_key="state.measures"))

    with pytest.raises(ValueError, match="multiple parents"):
        family._dspy_g_examples([tree])


def test_propensity_below_declared_minimum_fails_before_compiler(tmp_path: Path) -> None:
    compiler = _Compiler()
    family = build_dspy_family(
        _config(
            compiler=compiler,
            min_propensity=0.25,
        )
    )
    trace = _preference_trace(
        unit_id="too-rare",
        weight=2.0,
        propensity=0.2,
    )

    with pytest.raises(ValueError, match="below min_propensity"):
        family.train_f(
            f_init=None,
            g=None,
            traces=[trace],
            output_dir=tmp_path,
            iteration=1,
        )

    assert compiler.calls == []


def test_propensity_is_divided_once_and_precomputed_sample_weight_is_not_redivided() -> None:
    family = build_dspy_family(
        _config(
            root_weight=11.0,
            leaf_weight=3.0,
            merge_weight=7.0,
            importance_weight_cap=None,
        )
    )
    raw = _preference_trace(
        unit_id="raw",
        supervision_role="leaf",
        weight=2.0,
        propensity=0.25,
    )
    precomputed = _preference_trace(
        unit_id="precomputed",
        supervision_role="leaf",
        sample_weight=8.0,
        propensity=0.25,
    )

    raw_row = family._dspy_f_examples([raw], g=None)[0]
    precomputed_row = family._dspy_f_examples([precomputed], g=None)[0]

    assert raw_row.raw_weight == pytest.approx(2.0)
    assert raw_row.propensity == pytest.approx(0.25)
    assert raw_row.effective_weight == pytest.approx((2.0 / 0.25) * 3.0)
    assert raw_row.propensity_source == "metadata.propensity"
    assert raw_row.weight_source == (
        "raw_weight_over_propensity_times_leaf_weight[propensity_source=metadata.propensity]"
    )

    assert precomputed_row.raw_weight == pytest.approx(8.0)
    assert precomputed_row.propensity == pytest.approx(0.25)
    assert precomputed_row.effective_weight == pytest.approx(8.0 * 3.0)
    assert precomputed_row.propensity_source == "metadata.propensity"
    assert (
        precomputed_row.weight_source == "precomputed_sample_weight_times_leaf_weight"
        "[propensity_source=metadata.propensity]"
    )
