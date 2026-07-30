from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo.methods.dspy import DSPyFamilyConfig
from treepo.methods.families import resolve_family
from treepo.tree import TreeRecord


def _names(width: int) -> tuple[str, ...]:
    return tuple(f"target_{index:02d}" for index in range(width))


def _config(width: int, **overrides):
    names = _names(width)
    payload = {
        "target_names": names,
        "target_oracle_ids": ("fixture:oracle",) * width,
        "target_dim": width,
        "target_vector_key": "target_vector",
        "target_min": 0.0,
        "target_max": 1.0,
        "audit_laws": False,
    }
    payload.update(overrides)
    return payload


def _tree(width: int, *, text: str = "fixture document") -> TreeRecord:
    names = _names(width)
    return TreeRecord(
        tree_id=f"k{width}",
        text=text,
        metadata={
            "split": "test",
            "target_vector": {name: (index + 1) / (width + 1) for index, name in enumerate(names)},
        },
    )


@pytest.mark.parametrize("width", [1, 3, 57])
def test_all_widths_use_the_same_exact_named_vector_transport(width: int) -> None:
    expected = dict(_tree(width).metadata["target_vector"])
    family = resolve_family(
        "dspy",
        {
            **_config(width),
            "dspy_program": lambda **_kwargs: dict(expected),
        },
    )

    assert family.score_roots_with_f(
        f=None,
        g=None,
        trees=[_tree(width)],
    ) == [expected]


def test_named_k1_rejects_scalar_and_positional_sequence_transport() -> None:
    bare = resolve_family(
        "dspy",
        {
            **_config(1),
            "dspy_program": lambda **_kwargs: 0.625,
        },
    )
    assert bare.score_roots_with_f(
        f=None,
        g=None,
        trees=[_tree(1)],
    ) == [None]

    wrapped = resolve_family(
        "dspy",
        {
            **_config(1),
            "dspy_program": lambda **_kwargs: [0.625],
        },
    )
    assert wrapped.score_roots_with_f(
        f=None,
        g=None,
        trees=[_tree(1)],
    ) == [None]


def test_named_k1_training_target_requires_exact_singleton_mapping() -> None:
    family = resolve_family("dspy", {**_config(1), "optimizer": "none"})
    bare_scalar = TreeRecord(
        tree_id="bare-scalar",
        text="fixture document",
        metadata={"split": "train", "target_vector": 0.625},
    )
    exact_mapping = TreeRecord(
        tree_id="exact-singleton",
        text="fixture document",
        metadata={"split": "train", "target_vector": {"target_00": 0.625}},
    )

    assert family._dspy_f_examples([bare_scalar], g=None) == []
    rows = family._dspy_f_examples([exact_mapping], g=None)
    assert len(rows) == 1
    assert json.loads(rows[0].target_json) == {"target_00": 0.625}

    assert family._dspy_trace_target(bare_scalar) is None
    assert family._dspy_trace_target(exact_mapping) == {"target_00": 0.625}


def test_named_vectors_have_a_non_scalar_default_output_budget() -> None:
    vector = resolve_family("dspy", _config(3))
    scalar = resolve_family("dspy", {})

    assert vector.config.max_tokens == 4096
    assert scalar.config.max_tokens == 16

    explicit = resolve_family(
        "dspy",
        _config(1, lm_config={"max_tokens": 777}),
    )
    assert explicit.config.max_tokens == 777


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        "0.25",
        float("nan"),
        float("inf"),
        -0.01,
        1.01,
    ],
)
def test_vector_components_are_strict_finite_numbers_within_bounds(
    bad_value,
) -> None:
    family = resolve_family(
        "dspy",
        {
            **_config(1),
            "dspy_program": lambda **_kwargs: {"target_00": bad_value},
        },
    )
    assert family.score_roots_with_f(
        f=None,
        g=None,
        trees=[_tree(1)],
    ) == [None]


@pytest.mark.parametrize(
    "response",
    [
        '{"target_00":NaN}',
        '{"target_00":Infinity}',
        '{"target_00":0.25,"extra":0.75}',
        '{"wrong":0.25}',
    ],
)
def test_textual_vectors_fail_closed_on_invalid_json_contract(response: str) -> None:
    family = resolve_family(
        "dspy",
        {
            **_config(1),
            "dspy_program": lambda **_kwargs: response,
        },
    )
    assert family.score_roots_with_f(
        f=None,
        g=None,
        trees=[_tree(1)],
    ) == [None]


@pytest.mark.parametrize("field", ["prediction_by_target", "vector_json", "scores_json"])
def test_prediction_like_objects_are_unwrapped(field: str) -> None:
    expected = dict(_tree(3).metadata["target_vector"])
    value = expected if field == "prediction_by_target" else json.dumps(expected)
    prediction = SimpleNamespace(**{field: value})
    family = resolve_family(
        "dspy",
        {
            **_config(3),
            "dspy_program": lambda **_kwargs: prediction,
        },
    )
    assert family.score_roots_with_f(
        f=None,
        g=None,
        trees=[_tree(3)],
    ) == [expected]


def test_real_dspy_prediction_items_contract_when_dspy_is_installed() -> None:
    dspy = pytest.importorskip("dspy")
    expected = dict(_tree(3).metadata["target_vector"])
    responses = [
        dspy.Prediction(**expected),
        dspy.Prediction(vector_json=json.dumps(expected)),
        dspy.Prediction(prediction_by_target=expected),
    ]
    for response in responses:
        family = resolve_family(
            "dspy",
            {
                **_config(3),
                "dspy_program": lambda response=response, **_kwargs: response,
            },
        )
        assert family.score_roots_with_f(
            f=None,
            g=None,
            trees=[_tree(3)],
        ) == [expected]


def test_vector_default_prediction_uses_the_same_parser() -> None:
    expected = dict(_tree(3).metadata["target_vector"])
    family = resolve_family(
        "dspy",
        {
            **_config(3),
            "default_prediction": expected,
        },
    )
    assert family.score_roots_with_f(
        f=None,
        g=None,
        trees=[_tree(3)],
    ) == [expected]


def test_named_vector_supervised_examples_include_exact_target_json(
    tmp_path: Path,
) -> None:
    tree = _tree(3)
    family = resolve_family("dspy", {**_config(3), "optimizer": "none"})
    artifact = family.train_f(
        f_init=None,
        g=None,
        traces=[tree],
        output_dir=tmp_path,
        iteration=0,
    )

    assert artifact["supervision_output_contract"] == "strict_named_vector"
    rendered = artifact["supervised_examples"]
    assert "target={" in rendered
    for name in _names(3):
        assert f'"{name}"' in rendered
    assert "target=unknown" not in rendered


def test_max_prompt_chars_is_configurable_for_full_documents() -> None:
    text = "x" * 6000 + "TAIL_MARKER"
    family = resolve_family(
        "dspy",
        {
            **_config(1),
            "max_prompt_chars": 7000,
            "dspy_program": lambda **_kwargs: {"target_00": 0.5},
        },
    )
    prompt = family.render_prompt(_tree(1, text=text))

    assert family.config.max_prompt_chars == 7000
    assert "TAIL_MARKER" in prompt


@pytest.mark.parametrize(
    "config,match",
    [
        (
            {
                "target_names": ("a", "a"),
                "target_oracle_ids": ("o", "o"),
            },
            "target_names must be unique",
        ),
        (
            {
                "target_names": ("a", "b"),
                "target_oracle_ids": ("o",),
            },
            "one-for-one",
        ),
        (
            {
                "target_names": ("a", "b"),
                "target_oracle_ids": ("o", "o"),
                "target_dim": 3,
            },
            "target_dim",
        ),
    ],
)
def test_dspy_vector_schema_validation_matches_fno(config, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DSPyFamilyConfig(**config)
