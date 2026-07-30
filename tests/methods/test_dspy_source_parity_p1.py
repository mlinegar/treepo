from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from treepo.methods._dspy_records import example_weight
from treepo.methods.dspy import DSPyFamily, DSPyFamilyConfig, build_dspy_family
from treepo.state import TaskState
from treepo.tasks.manifesto.components import (
    RILE_CMP56_GRANULARITY,
    RILE_POLARITY_GRANULARITY,
    manifesto_rile_component_fit_fragment,
)
from treepo.tree import TreeNode, TreeRecord

TARGET = {"score": 0.75}


class _Program:
    def __init__(self, kind: str) -> None:
        self.kind = str(kind)

    def __call__(self, **_kwargs: Any) -> dict[str, str]:
        if self.kind == "f":
            return {"prediction_json": json.dumps(TARGET)}
        return {"state": "candidate state"}

    def save(self, path: str, save_program: bool = False) -> None:
        Path(path).write_text(
            json.dumps({"kind": self.kind, "save_program": bool(save_program)}),
            encoding="utf-8",
        )


class _Compiler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def compile(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return kwargs["program"]


class _Field:
    def __init__(self, **kwargs: Any) -> None:
        self.metadata = dict(kwargs)


class _Signature:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.instructions = kwargs.get("instructions")


class _CapturedProgram:
    def __init__(self, signature: Any, kwargs: dict[str, Any]) -> None:
        self.signature = signature
        self.kwargs = dict(kwargs)


class _CapturingDSPy:
    Signature = _Signature
    InputField = _Field
    OutputField = _Field

    def __init__(self) -> None:
        self.programs: list[_CapturedProgram] = []

    def Predict(self, signature: Any, **kwargs: Any) -> _CapturedProgram:
        program = _CapturedProgram(signature, kwargs)
        self.programs.append(program)
        return program


def _weight_config(**overrides: Any) -> SimpleNamespace:
    values = {
        "min_propensity": 1e-8,
        "importance_weight_cap": None,
        "root_weight": 1.0,
        "leaf_weight": 1.0,
        "merge_weight": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _signature_text(program: _CapturedProgram) -> str:
    """Read instructions without prescribing string versus Signature-class construction."""

    signature = program.signature
    values = [
        signature if isinstance(signature, str) else None,
        getattr(signature, "instructions", None),
        getattr(signature, "__doc__", None),
        getattr(signature, "__name__", None),
        program.kwargs.get("instructions"),
    ]
    return "\n".join(str(value) for value in values if value)


def _optimizer_config(**overrides: Any) -> DSPyFamilyConfig:
    values = {
        "target_names": ("score",),
        "target_oracle_ids": ("fixture:score",),
        "target_dim": 1,
        "target_vector_key": "target_vector",
        "node_target_key": "target_vector",
        "optimizer": "bootstrap",
        "validation_fraction": 0.0,
        "audit_laws": False,
        # These optional fields restore the already-published no-truncation
        # contract without prescribing a particular tokenizer implementation.
        "leaf_size_tokens": 1,
        "lm_context_window_tokens": 64,
        "max_completion_tokens": 2,
        "prompt_template_overhead_tokens": 2,
    }
    values.update(overrides)
    return DSPyFamilyConfig(**values)


def _oversize_tree() -> TreeRecord:
    text = "manifesto-token " * 10_000
    state = TaskState(
        kind="fixture",
        text=text,
        measures=dict(TARGET),
    )
    return TreeRecord(
        tree_id="over-budget",
        text=text,
        root_label=dict(TARGET),
        nodes=(
            TreeNode(
                node_id="root",
                unit_type="root",
                text=text,
                label=dict(TARGET),
                state=state,
                metadata={"split": "train", "target_vector": dict(TARGET)},
            ),
        ),
        metadata={"split": "train", "target_vector": dict(TARGET)},
    )


def test_joint_propensity_outranks_generic_propensity() -> None:
    weighting = example_weight(
        _weight_config(),
        {
            "propensity": 0.5,
            "joint_propensity": 0.2,
            "node_weight": 2.0,
        },
        supervision_role="leaf",
    )

    assert weighting["propensity"] == pytest.approx(0.2)
    assert weighting["effective_weight"] == pytest.approx(2.0 / 0.2)


@pytest.mark.parametrize(
    ("sampling", "expected"),
    [
        ({"joint_propensity": 0.125}, 0.125),
        (
            {
                "document_propensity": 0.5,
                "unit_propensity": 0.25,
                "label_propensity": 0.4,
            },
            0.05,
        ),
    ],
)
def test_nested_sampling_joint_or_product_is_consumed(
    sampling: dict[str, float],
    expected: float,
) -> None:
    weighting = example_weight(
        _weight_config(),
        {"sampling": sampling},
        supervision_role="root",
    )

    assert weighting["propensity"] == pytest.approx(expected)
    assert weighting["effective_weight"] == pytest.approx(1.0 / expected)


def test_sampling_marked_unsupported_for_ipw_fails_closed() -> None:
    with pytest.raises(ValueError, match="supports_ipw_estimation"):
        example_weight(
            _weight_config(),
            {
                "sampling": {
                    "joint_propensity": 0.25,
                    "supports_ipw_estimation": False,
                }
            },
            supervision_role="root",
        )


def test_dspy_artifact_does_not_claim_horvitz_thompson_without_population_total(
    tmp_path: Path,
) -> None:
    family = build_dspy_family(
        {
            "target_names": ("score",),
            "target_oracle_ids": ("fixture:score",),
            "target_dim": 1,
            "optimizer": "none",
            "audit_laws": False,
        }
    )
    program_path = tmp_path / "program.json"
    program_path.write_text("{}", encoding="utf-8")
    row = SimpleNamespace(
        effective_weight=4.0,
        propensity=0.25,
        weight_source="unit_weight_over_propensity_times_root_weight",
    )

    artifact = family._dspy_learned_artifact(
        kind="f",
        iteration=1,
        rows=[row],
        split={"train_count": 1, "validation_count": 0},
        program_path=program_path,
    )

    estimator = str(artifact["weighting"]["estimator"]).lower()
    assert "horvitz" not in estimator
    assert "thompson" not in estimator


def test_native_programs_receive_configured_task_neutral_signature_instructions() -> None:
    f_instructions = "Read one task state and emit the declared named target vector."
    g_instructions = "Map a tagged leaf, unary, or binary call to one shared task state."
    config = DSPyFamilyConfig(
        optimizer="none",
        f_signature_instructions=f_instructions,
        g_signature_instructions=g_instructions,
    )
    dspy = _CapturingDSPy()
    family = DSPyFamily(config=config, dspy_module=dspy)

    f_program = family._dspy_new_program("f", dspy=dspy)
    g_program = family._dspy_new_program("g", dspy=dspy)

    assert f_instructions in _signature_text(f_program)
    assert g_instructions in _signature_text(g_program)


def test_default_signatures_name_joint_targets_oracles_and_shared_g_roles() -> None:
    config = DSPyFamilyConfig(
        optimizer="none",
        target_names=("first", "second"),
        target_oracle_ids=("oracle:first", "oracle:second"),
    )
    dspy = _CapturingDSPy()
    family = DSPyFamily(config=config, dspy_module=dspy)

    f_text = _signature_text(family._dspy_new_program("f", dspy=dspy)).lower()
    g_text = _signature_text(family._dspy_new_program("g", dspy=dspy)).lower()

    for value in ("first", "second", "oracle:first", "oracle:second"):
        assert value in f_text
        assert value in g_text
    assert "strict json" in f_text
    assert "sum-l1" in f_text
    assert "shared" in g_text
    assert "leaf" in g_text
    assert "unary" in g_text
    assert "binary" in g_text


@pytest.mark.parametrize(
    ("granularity", "width", "required_terms"),
    [
        (RILE_POLARITY_GRANULARITY, 3, ("RILE", "left", "right")),
        (RILE_CMP56_GRANULARITY, 57, ("RILE", "CMP", "residual")),
    ],
)
def test_rile_component_fragment_owns_nonempty_k3_k57_signature_instructions(
    granularity: str,
    width: int,
    required_terms: tuple[str, ...],
) -> None:
    fragment = manifesto_rile_component_fit_fragment(granularity)
    backend = fragment["backend_config"]

    assert len(fragment["oracle_targets"]) == width
    instructions_by_role = {}
    for key in ("f_signature_instructions", "g_signature_instructions"):
        instructions = str(backend[key]).strip()
        assert instructions
        instructions_by_role[key] = instructions
    combined = "\n".join(instructions_by_role.values()).lower()
    assert all(term.lower() in combined for term in required_terms)


def test_impossible_two_child_token_budget_is_rejected_at_configuration() -> None:
    with pytest.raises((ValueError, RuntimeError), match=r"(?i)(two.child|budget|context)"):
        DSPyFamily(
            config=_optimizer_config(
                leaf_size_tokens=8,
                lm_context_window_tokens=30,
                max_completion_tokens=16,
                prompt_template_overhead_tokens=4,
            )
        )


@pytest.mark.parametrize("kind", ["f", "g"])
def test_each_learned_artifact_discloses_exact_budget_counter(
    kind: str,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def count_tokens(text: str) -> int:
        calls.append(text)
        return len(text.split())

    family = DSPyFamily(
        config=_optimizer_config(),
        token_count_fn=count_tokens,
    )
    family._dspy_assert_input_budget(
        label="fixture",
        fields={"state": "one two three"},
    )
    path = tmp_path / f"{kind}.json"
    path.write_text("{}", encoding="utf-8")
    row = SimpleNamespace(
        effective_weight=1.0,
        propensity=1.0,
        weight_source="unit_weight_over_propensity",
    )
    artifact = family._dspy_learned_artifact(
        kind=kind,
        iteration=1,
        rows=[row],
        split={"train_count": 1, "validation_count": 0},
        program_path=path,
    )

    assert calls == ["one two three"]
    budget = artifact["no_truncation_budget"]
    assert budget["enabled"] is True
    assert budget["token_count_method"] == "injected_exact_counter"
    assert budget["coverage"] == "per_training_record_and_direct_program_input"
    assert budget["compiled_demo_stack_verified"] is False
    assert budget["absolute_prompt_fit_guarantee"] is False


@pytest.mark.parametrize("kind", ["f", "g"])
def test_actual_over_budget_record_is_rejected_before_compiler(
    kind: str,
    tmp_path: Path,
) -> None:
    compiler = _Compiler()
    family = DSPyFamily(
        config=_optimizer_config(),
        f_program=_Program("f"),
        g_program=_Program("g"),
        compiler=compiler,
    )
    tree = _oversize_tree()

    with pytest.raises((ValueError, RuntimeError), match=r"(?i)(no.truncation|budget|context)"):
        if kind == "f":
            family.train_f(
                f_init=None,
                g=None,
                traces=[tree],
                output_dir=tmp_path / "f",
                iteration=1,
            )
        else:
            family.train_g(
                g_init=None,
                f=_Program("f"),
                traces=[tree],
                output_dir=tmp_path / "g",
                iteration=1,
            )

    assert compiler.calls == []
