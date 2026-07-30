from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo import fit
from treepo.methods.dspy import DSPyFamilyConfig, build_dspy_family
from treepo.state import TaskState
from treepo.tree import TreeNode, TreeRecord


class _RecordingProgram:
    def __init__(self, kind: str, *, prediction: dict[str, float] | None = None) -> None:
        self.kind = str(kind)
        self.prediction = dict(prediction or {})
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.kind == "f":
            return {"prediction_json": json.dumps(self.prediction, sort_keys=True)}
        return {"state": f"generated-state-{len(self.calls)}"}

    def save(self, path: str, save_program: bool = False) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "prediction": self.prediction,
                    "save_program": bool(save_program),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


class _RecordingCompiler:
    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []

    def compile(self, *, program, metric, trainset, valset, kind, **_kwargs):
        self.calls.append(
            SimpleNamespace(
                kind=str(kind),
                program=program,
                metric=metric,
                trainset=list(trainset),
                valset=list(valset),
            )
        )
        return program


def _load_recording_program(*, path: Path, **_kwargs):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _RecordingProgram(
        str(payload["kind"]),
        prediction=dict(payload.get("prediction") or {}),
    )


def _names(width: int) -> tuple[str, ...]:
    return tuple(f"target_{index:02d}" for index in range(width))


def _target(width: int) -> dict[str, float]:
    return {
        name: float(index + 1) / float(width + 1)
        for index, name in enumerate(_names(width))
    }


def _family_config(width: int, **overrides):
    names = _names(width)
    config = {
        "target_names": names,
        "target_oracle_ids": tuple(f"fixture:{name}" for name in names),
        "target_dim": width,
        "target_vector_key": "target_vector",
        "node_target_key": "target_vector",
        "target_min": 0.0,
        "target_max": 1.0,
        "optimizer": "bootstrap",
        "validation_fraction": 0.0,
        "allow_identity_g_targets": False,
        "audit_laws": False,
        "program_loader": _load_recording_program,
    }
    config.update(overrides)
    return config


def _state(node_id: str, width: int) -> TaskState:
    return TaskState(
        kind="manifesto_policy",
        text=f"state::{node_id}",
        measures=_target(width),
        metadata={"node_id": node_id, "source": "resolved_qsentence_bridge"},
    )


def _odd_three_leaf_tree(
    width: int,
    *,
    tree_id: str = "odd-3",
    split: str = "train",
    group_id: str | None = None,
) -> TreeRecord:
    shared = {"split": split}
    if group_id is not None:
        shared["group_id"] = group_id

    def node(
        node_id: str,
        *,
        unit_type: str,
        level: int,
        position: int,
        parent_id: str | None = None,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
    ) -> TreeNode:
        return TreeNode(
            node_id=node_id,
            unit_type=unit_type,
            text=f"raw::{node_id}",
            level=level,
            position=position,
            parent_id=parent_id,
            left_child_id=left_child_id,
            right_child_id=right_child_id,
            # The resolved bridge puts exact state text and oracle values in
            # TaskState; no legacy scalar/node-label fallback is supplied.
            label=None,
            state=_state(node_id, width),
            metadata=dict(shared),
        )

    nodes = (
        node("leaf-0", unit_type="leaf", level=0, position=0, parent_id="pair-0"),
        node("leaf-1", unit_type="leaf", level=0, position=1, parent_id="pair-0"),
        node("leaf-2", unit_type="leaf", level=0, position=2, parent_id="carry-0"),
        node(
            "pair-0",
            unit_type="merge",
            level=1,
            position=0,
            parent_id="root",
            left_child_id="leaf-0",
            right_child_id="leaf-1",
        ),
        node(
            "carry-0",
            unit_type="merge",
            level=1,
            position=1,
            parent_id="root",
            left_child_id="leaf-2",
        ),
        node(
            "root",
            unit_type="root",
            level=2,
            position=0,
            left_child_id="pair-0",
            right_child_id="carry-0",
        ),
    )
    return TreeRecord(
        tree_id=tree_id,
        doc_id=tree_id,
        text=f"document::{tree_id}",
        root_label=None,
        nodes=nodes,
        metadata=dict(shared),
    )


@pytest.mark.parametrize("width", [1, 3, 57])
def test_k1_k3_k57_share_one_strict_named_object(width: int) -> None:
    family = build_dspy_family(
        _family_config(
            width,
            f_program=_RecordingProgram("f", prediction=_target(width)),
        )
    )
    target = _target(width)

    assert family._parse_prediction(target) == target
    assert family._parse_prediction(json.dumps(target)) == target
    # A named catalog never takes a scalar or positional side path, including
    # K=1. Coordinate identity is part of the oracle contract.
    assert family._parse_prediction(list(target.values())) is None
    assert family._parse_prediction(next(iter(target.values()))) is None


@pytest.mark.parametrize("width", [1, 3, 57])
def test_reciprocal_reward_uses_exact_unnormalized_sum_l1(width: int) -> None:
    family = build_dspy_family(
        _family_config(
            width,
            f_program=_RecordingProgram("f", prediction=_target(width)),
        )
    )
    target = _target(width)

    prediction = dict(target)
    first = _names(width)[0]
    prediction[first] = target[first] + 0.25
    gold = family._dspy_example(
        state="state",
        target_json=json.dumps(target),
        effective_weight=1.0,
        inputs=("state",),
    )
    reward = family._dspy_f_metric(gold, {"prediction_json": json.dumps(prediction)})

    assert sum(
        abs(prediction[name] - target[name]) for name in _names(width)
    ) == pytest.approx(0.25)
    assert reward == pytest.approx(1.0 / 1.25)


def test_taskstate_text_and_measures_supply_node_wide_f_and_g_records() -> None:
    width = 3
    family = build_dspy_family(
        _family_config(
            width,
            root_weight=1.0,
            leaf_weight=1.0,
            merge_weight=1.0,
        )
    )
    tree = _odd_three_leaf_tree(width)

    f_rows = family._dspy_f_examples([tree], g=None)
    assert len(f_rows) == len(tree.nodes)
    assert {row.role for row in f_rows} == {"root", "leaf", "merge"}
    assert {row.state for row in f_rows} == {
        str(node.state.text) for node in tree.nodes if isinstance(node.state, TaskState)
    }
    assert all(json.loads(row.target_json) == _target(width) for row in f_rows)

    g_rows = family._dspy_g_examples([tree])
    assert len(g_rows) == len(tree.nodes)
    assert {row.role for row in g_rows} == {"leaf", "merge", "recompression"}
    assert {row.state for row in g_rows} == {
        str(node.state.text) for node in tree.nodes if isinstance(node.state, TaskState)
    }
    assert all(json.loads(row.target_json) == _target(width) for row in g_rows)

    carry = [row for row in g_rows if "carry-0" in str(row.row_id)]
    assert len(carry) == 1
    assert carry[0].role == "recompression"
    assert carry[0].c3_eligible is False
    assert "Promote" in carry[0].prompt
    assert "CHILD_STATE" in carry[0].prompt
    assert "RIGHT_STATE" not in carry[0].prompt


def test_one_g_program_runs_at_every_leaf_merge_and_odd_carry_node() -> None:
    width = 1
    family = build_dspy_family(_family_config(width))
    tree = _odd_three_leaf_tree(width)
    g = _RecordingProgram("g")

    assert family._dspy_reduce_tree(g, tree)
    assert len(g.calls) == len(tree.nodes)
    prompts = [str(call["prompt"]) for call in g.calls]
    assert sum("[TREEPO_G_CALL=leaf]" in prompt for prompt in prompts) == 3
    # Only two calls have two child states and therefore belong to the
    # merge-domain/C3 population. Odd carry is a unary recompression call to
    # the same g object, never a fabricated binary merge.
    assert sum("[TREEPO_G_CALL=merge]" in prompt for prompt in prompts) == 2
    assert sum("[TREEPO_G_CALL=recompression]" in prompt for prompt in prompts) == 1
    carry_prompts = [prompt for prompt in prompts if "CHILD_STATE" in prompt]
    assert len(carry_prompts) == 1
    assert "RIGHT_STATE" not in carry_prompts[0]


@pytest.mark.parametrize(
    ("weights", "expected_roles"),
    [
        ({"root_weight": 1.0, "leaf_weight": 0.0, "merge_weight": 0.0}, {"root"}),
        ({"root_weight": 0.0, "leaf_weight": 1.0, "merge_weight": 0.0}, {"leaf"}),
        ({"root_weight": 0.0, "leaf_weight": 0.0, "merge_weight": 1.0}, {"merge"}),
        (
            {"root_weight": 3.0, "leaf_weight": 1.0, "merge_weight": 1.0},
            {"root", "leaf", "merge"},
        ),
    ],
)
def test_dspy_supervision_weights_filter_zero_role_rows_before_compile(
    tmp_path: Path,
    weights: dict[str, float],
    expected_roles: set[str],
) -> None:
    width = 1
    compiler = _RecordingCompiler()
    family = build_dspy_family(
        _family_config(
            width,
            compiler=compiler,
            f_program=_RecordingProgram("f", prediction=_target(width)),
            **weights,
        )
    )
    family.train_f(
        f_init=None,
        g=None,
        traces=[_odd_three_leaf_tree(width)],
        output_dir=tmp_path,
        iteration=1,
    )

    call = compiler.calls[0]
    optimizer_rows = call.trainset + call.valset
    assert {row.role for row in optimizer_rows} == expected_roles
    assert all(row.effective_weight > 0.0 for row in optimizer_rows)


def test_named_leaf_supervision_uses_the_same_treepo_fit_grid_api(
    tmp_path: Path,
) -> None:
    width = 1
    compiler = _RecordingCompiler()
    tree = _odd_three_leaf_tree(width)
    result = fit(
        {
            "family": "dspy",
            "supervision_level": "leaf",
            "train_data": [tree],
            "eval_data": [tree],
            "oracle_targets": [
                {
                    "target_name": _names(width)[0],
                    "oracle_id": f"fixture:{_names(width)[0]}",
                }
            ],
            "backend_config": {
                **_family_config(
                    width,
                    compiler=compiler,
                    f_program=_RecordingProgram("f", prediction=_target(width)),
                ),
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 1, "axis_value": 3},
        }
    )

    assert result.status == "success"
    assert result.summary["supervision"]["level"] == "leaf"
    call = next(item for item in compiler.calls if item.kind == "f")
    assert call.trainset
    assert {row.role for row in call.trainset + call.valset} == {"leaf"}


def test_dspy_weight_defaults_match_named_fit_supervision_levels() -> None:
    config = DSPyFamilyConfig(
        target_names=("rile_normalized",),
        target_oracle_ids=("fixture:rile",),
    )
    assert config.root_weight == pytest.approx(1.0)
    assert config.leaf_weight == pytest.approx(1.0)
    assert config.merge_weight == pytest.approx(1.0)


def test_compiler_never_sees_test_rows_and_seeded_split_is_group_disjoint(
    tmp_path: Path,
) -> None:
    width = 1
    compiler = _RecordingCompiler()
    family = build_dspy_family(
        _family_config(
            width,
            compiler=compiler,
            f_program=_RecordingProgram("f", prediction=_target(width)),
            root_weight=1.0,
            leaf_weight=1.0,
            merge_weight=1.0,
            validation_fraction=0.34,
            split_seed=11,
        )
    )
    train = [
        _odd_three_leaf_tree(
            width,
            tree_id=f"{group}-{replicate}",
            group_id=group,
        )
        for group in ("party-a", "party-b", "party-c")
        for replicate in range(2)
    ]
    held_out = _odd_three_leaf_tree(
        width,
        tree_id="never-compile",
        split="test",
        group_id="test-party",
    )
    artifact = family.train_f(
        f_init=None,
        g=None,
        traces=[*train, held_out],
        output_dir=tmp_path,
        iteration=1,
    )

    call = compiler.calls[0]
    assert call.trainset
    assert call.valset
    assert all(row.source_split != "test" for row in call.trainset + call.valset)
    train_groups = {row.group_id for row in call.trainset}
    validation_groups = {row.group_id for row in call.valset}
    assert train_groups.isdisjoint(validation_groups)
    assert "test-party" not in train_groups | validation_groups
    assert artifact["split"]["excluded_test_count"] == len(held_out.nodes)
    assert artifact["split"]["strategy"].startswith("seeded_group_split")
