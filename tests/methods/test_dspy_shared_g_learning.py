from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo import fit
from treepo.methods.dspy import build_dspy_family
from treepo.tree import TreeNode, TreeRecord


class _FakeProgram:
    def __init__(
        self,
        kind: str,
        *,
        version: int = 0,
        prediction: dict[str, float] | None = None,
    ) -> None:
        self.kind = kind
        self.version = int(version)
        self.prediction = dict(prediction or {})
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.kind == "f":
            return {"prediction_json": json.dumps(self.prediction, sort_keys=True)}
        return {"state": f"g-v{self.version}:{kwargs.get('prompt', '')}"}

    def save(self, path: str, save_program: bool = False) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "program.json").write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "version": self.version,
                    "prediction": self.prediction,
                    "save_program": bool(save_program),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


class _FakeCompiler:
    def __init__(self, *, prediction: dict[str, float]) -> None:
        self.prediction = dict(prediction)
        self.calls: list[SimpleNamespace] = []

    def compile(self, *, program, metric, trainset, valset, kind, **_kwargs):
        compiled = _FakeProgram(
            str(kind),
            version=int(getattr(program, "version", 0)) + 1,
            prediction=self.prediction if kind == "f" else None,
        )
        if kind == "f":
            prediction = compiled(state=getattr(trainset[0], "state"))
        else:
            prediction = {"state": "candidate-summary"}
        reward = metric(trainset[0], prediction)
        self.calls.append(
            SimpleNamespace(
                kind=str(kind),
                warmstart=program,
                compiled=compiled,
                metric=metric,
                reward=float(reward),
                trainset=list(trainset),
                valset=list(valset),
            )
        )
        return compiled


class _Loader:
    def __init__(self) -> None:
        self.paths: list[tuple[str, str]] = []

    def __call__(self, *, path: Path, kind: str, **_kwargs):
        self.paths.append((str(kind), str(path)))
        payload = json.loads((Path(path) / "program.json").read_text(encoding="utf-8"))
        return _FakeProgram(
            str(payload["kind"]),
            version=int(payload["version"]),
            prediction=dict(payload.get("prediction") or {}),
        )


def _names(k: int) -> tuple[str, ...]:
    return tuple(f"target_{index:02d}" for index in range(k))


def _target(k: int, value: float = 0.75) -> dict[str, float]:
    return {name: float(value) for name in _names(k)}


def _config(k: int, *, compiler=None, loader=None, validation_fraction: float = 0.0):
    names = _names(k)
    prediction = {name: 0.5 for name in names}
    return {
        "target_names": names,
        "target_oracle_ids": tuple(f"oracle:{name}" for name in names),
        "target_dim": k,
        "target_vector_key": "targets",
        "node_target_key": "targets",
        "target_min": 0.0,
        "target_max": 1.0,
        "optimizer": "bootstrap",
        "validation_fraction": validation_fraction,
        "split_seed": 17,
        "importance_weight_cap": 10.0,
        "allow_identity_g_targets": False,
        "g_target_source": "explicit_reference_text_fixture",
        "f_program": _FakeProgram("f", prediction=prediction),
        "g_program": _FakeProgram("g"),
        "compiler": compiler,
        "program_loader": loader if loader is not None else _Loader(),
    }


def _tree(k: int, *, singleton: bool = False, weight: float = 2.0, propensity: float = 0.5):
    target = _target(k)
    shared = {
        "targets": target,
        "split": "train",
        "weight": weight,
        "propensity": propensity,
    }
    if singleton:
        nodes = (
            TreeNode(
                node_id="leaf",
                unit_type="root",
                text="only leaf",
                label=target,
                metadata={**shared, "target_summary": "only-leaf-summary"},
            ),
        )
    else:
        nodes = (
            TreeNode(
                node_id="left",
                unit_type="leaf",
                text="left text",
                parent_id="root",
                label=target,
                metadata={**shared, "target_summary": "left-summary"},
            ),
            TreeNode(
                node_id="right",
                unit_type="leaf",
                text="right text",
                parent_id="root",
                label=target,
                metadata={**shared, "target_summary": "right-summary"},
            ),
            TreeNode(
                node_id="root",
                unit_type="root",
                text="whole-document",
                left_child_id="left",
                right_child_id="right",
                label=target,
                metadata={**shared, "target_summary": "root-summary"},
            ),
        )
    return TreeRecord(
        tree_id="tree-1",
        text="whole-document",
        root_label=target,
        nodes=nodes,
        metadata=shared,
    )


@pytest.mark.parametrize("k", [1, 3, 57])
def test_shared_g_compiles_leaf_and_merge_rows_with_unnormalized_sum_l1(
    tmp_path: Path,
    k: int,
) -> None:
    prediction = {name: 0.5 for name in _names(k)}
    compiler = _FakeCompiler(prediction=prediction)
    family = build_dspy_family(_config(k, compiler=compiler))
    tree = _tree(k)

    f_artifact = family.train_f(
        f_init=None,
        g={"kind": "treepo_identity_g", "g_mode": "identity", "operator": "identity"},
        traces=[tree],
        output_dir=tmp_path / "f",
        iteration=1,
    )
    assert f_artifact["f_state_sources"] == ["identity_raw_text"]
    assert f_artifact["f_state_source_training_counts"]["identity_raw_text"] == 3
    assert f_artifact["f_state_source_training_counts"]["current_g_generated"] == 0
    assert f_artifact["f_current_g_generation_executed"] is False
    outcome = family.train_g(
        g_init=None,
        f=f_artifact,
        traces=[tree],
        output_dir=tmp_path / "g",
        iteration=2,
    )

    assert outcome.update_performed is True
    artifact = outcome.artifact
    assert artifact["g_training_call_roles"] == ["leaf", "merge"]
    assert artifact["leaf_domain_training_row_count"] == 2
    assert artifact["merge_domain_training_row_count"] == 1
    assert artifact["g_training_row_count_scope"] == "optimizer_compile_trainset_only"
    assert artifact["g_training_role_evidence_source"] == "dspy_compiler_mixed_examples"
    assert artifact["same_g_across_node_roles"] is True
    assert artifact["reduce_g_is_derived"] is True
    assert artifact["universal_c3_established"] is False
    assert artifact["identity_g_target_row_count"] == 0
    assert artifact["g_target_source"] == "explicit_reference_text_fixture"
    assert Path(artifact["program_path"]).exists()
    assert len(artifact["program_sha256"]) == 64

    g_call = [call for call in compiler.calls if call.kind == "g"][0]
    prompts = [row.prompt for row in g_call.trainset]
    assert sum("[TREEPO_G_CALL=leaf]" in prompt for prompt in prompts) == 2
    assert sum("[TREEPO_G_CALL=merge]" in prompt for prompt in prompts) == 1
    assert {row.effective_weight for row in g_call.trainset} == {4.0}
    # Exact sum-L1 distance is 0.25*K, so the IPW-weighted reward is
    # 4 / (1 + 0.25*K) at every width.
    expected_reward = 4.0 / (1.0 + 0.25 * k)
    assert g_call.reward == pytest.approx(expected_reward)
    assert [call.reward for call in compiler.calls if call.kind == "f"] == [
        pytest.approx(expected_reward)
    ]

    before = len(g_call.compiled.calls)
    prediction_rows = family.score_roots_with_f(
        f=f_artifact,
        g=artifact,
        trees=[tree],
    )
    assert prediction_rows == [prediction]
    executed = g_call.compiled.calls[before:]
    assert sum(str(call["prompt"]).startswith("[TREEPO_G_CALL=leaf]") for call in executed) == 2
    assert sum(str(call["prompt"]).startswith("[TREEPO_G_CALL=merge]") for call in executed) == 1


def test_f_generated_when_available_uses_and_records_current_g_states(
    tmp_path: Path,
) -> None:
    k = 3
    compiler = _FakeCompiler(prediction={name: 0.5 for name in _names(k)})
    current_g = _FakeProgram("g", version=7)
    family = build_dspy_family(
        {
            **_config(k, compiler=compiler),
            "f_record_source": "generated_when_available",
        }
    )

    artifact = family.train_f(
        f_init=None,
        g=current_g,
        traces=[_tree(k)],
        output_dir=tmp_path,
        iteration=3,
    )

    call = next(entry for entry in compiler.calls if entry.kind == "f")
    assert len(current_g.calls) == 3
    assert len(call.trainset) == 3
    assert all(str(row.state).startswith("g-v7:") for row in call.trainset)
    assert {row.f_state_source for row in call.trainset} == {"current_g_generated"}
    assert artifact["f_record_source_requested"] == "generated_when_available"
    assert artifact["f_state_sources"] == ["current_g_generated"]
    assert artifact["f_state_source_training_counts"]["current_g_generated"] == 3
    assert artifact["f_state_source_training_counts"]["reference_state"] == 0
    assert artifact["f_current_g_generation_executed"] is True


def test_g_warmstarts_and_persisted_reload_bypasses_in_memory_cache(tmp_path: Path) -> None:
    k = 3
    prediction = {name: 0.5 for name in _names(k)}
    compiler = _FakeCompiler(prediction=prediction)
    family = build_dspy_family(_config(k, compiler=compiler))
    tree = _tree(k)
    f_artifact = family.train_f(
        f_init=None,
        g=None,
        traces=[tree],
        output_dir=tmp_path / "f",
        iteration=1,
    )
    first = family.train_g(
        g_init=None,
        f=f_artifact,
        traces=[tree],
        output_dir=tmp_path / "g1",
        iteration=2,
    ).artifact
    second = family.train_g(
        g_init=first,
        f=f_artifact,
        traces=[tree],
        output_dir=tmp_path / "g2",
        iteration=4,
    ).artifact
    g_calls = [call for call in compiler.calls if call.kind == "g"]
    assert g_calls[1].warmstart is g_calls[0].compiled
    assert g_calls[1].compiled.version == g_calls[0].compiled.version + 1
    assert second["program_sha256"] != first["program_sha256"]

    loader = _Loader()
    reloaded = build_dspy_family(_config(k, compiler=compiler, loader=loader))
    assert reloaded._dspy_program_cache == {}
    reloaded.validate_artifact(kind="g", artifact=second)
    assert loader.paths == [("g", second["program_path"])]
    assert reloaded.score_roots_with_f(f=f_artifact, g=second, trees=[tree]) == [prediction]
    assert {kind for kind, _path in loader.paths} == {"f", "g"}


def test_singleton_g_is_one_leaf_call_and_identity_full_doc_never_calls_g(tmp_path: Path) -> None:
    k = 1
    prediction = {name: 0.5 for name in _names(k)}
    compiler = _FakeCompiler(prediction=prediction)
    family = build_dspy_family(_config(k, compiler=compiler))
    singleton = _tree(k, singleton=True)
    f_artifact = family.train_f(
        f_init=None,
        g=None,
        traces=[singleton],
        output_dir=tmp_path / "f",
        iteration=1,
    )
    g_artifact = family.train_g(
        g_init=None,
        f=f_artifact,
        traces=[singleton],
        output_dir=tmp_path / "g",
        iteration=2,
    ).artifact
    assert g_artifact["g_training_call_roles"] == ["leaf"]
    assert g_artifact["leaf_domain_training_row_count"] == 1
    assert g_artifact["merge_domain_training_row_count"] == 0

    learned_g = [call for call in compiler.calls if call.kind == "g"][0].compiled
    before = len(learned_g.calls)
    family.score_roots_with_f(f=f_artifact, g=g_artifact, trees=[singleton])
    calls = learned_g.calls[before:]
    assert len(calls) == 1
    assert "[TREEPO_G_CALL=leaf]" in str(calls[0]["prompt"])

    before = len(learned_g.calls)
    identity = {
        "kind": "treepo_identity_g",
        "g_mode": "identity",
        "operator": "identity",
        "trainable": False,
    }
    assert family.score_roots_with_f(f=f_artifact, g=identity, trees=[singleton]) == [prediction]
    assert len(learned_g.calls) == before
    f_program = [call for call in compiler.calls if call.kind == "f"][0].compiled
    assert f_program.calls[-1]["state"] == "whole-document"


def test_no_node_g_preference_trace_is_explicit_and_invalid_sampling_fails(tmp_path: Path) -> None:
    k = 1
    prediction = {name: 0.5 for name in _names(k)}
    compiler = _FakeCompiler(prediction=prediction)
    family = build_dspy_family(_config(k, compiler=compiler))
    trace = SimpleNamespace(
        text="preferred candidate summary",
        metadata={
            "oracle_target": _target(k),
            "preference_target": "g",
            "preference_unit_id": "pref-1",
            "split": "train",
            "weight": 1.0,
            "propensity": 0.25,
        },
    )
    outcome = family.train_g(
        g_init=None,
        f=_FakeProgram("f", prediction=prediction),
        traces=[trace],
        output_dir=tmp_path / "preference",
        iteration=2,
    )
    assert outcome.artifact["leaf_domain_training_row_count"] == 1
    assert outcome.artifact["merge_domain_training_row_count"] == 0
    assert outcome.artifact["identity_g_target_row_count"] == 0

    bad_tree = _tree(k, propensity=1.5)
    with pytest.raises(ValueError, match="propensity must be finite and in \\(0, 1\\]"):
        family.train_g(
            g_init=None,
            f=_FakeProgram("f", prediction=prediction),
            traces=[bad_tree],
            output_dir=tmp_path / "bad",
            iteration=2,
        )


def test_missing_or_zero_weight_g_targets_fail_closed_unless_optimizer_none(tmp_path: Path) -> None:
    k = 1
    prediction = {name: 0.5 for name in _names(k)}
    compiler = _FakeCompiler(prediction=prediction)
    family = build_dspy_family(_config(k, compiler=compiler))
    missing = TreeRecord(tree_id="missing", text="no labels", metadata={"split": "train"})
    with pytest.raises(ValueError, match="at least one observed node target"):
        family.train_g(
            g_init=None,
            f=_FakeProgram("f", prediction=prediction),
            traces=[missing],
            output_dir=tmp_path / "missing",
            iteration=2,
        )
    with pytest.raises(ValueError, match="at least one observed node target"):
        family.train_g(
            g_init=None,
            f=_FakeProgram("f", prediction=prediction),
            traces=[_tree(k, weight=0.0)],
            output_dir=tmp_path / "zero",
            iteration=2,
        )

    offline = build_dspy_family(
        {
            **_config(k, compiler=compiler),
            "optimizer": "none",
            "dspy_program": lambda **_kwargs: prediction,
        }
    )
    outcome = offline.train_g(
        g_init=None,
        f=None,
        traces=[missing],
        output_dir=tmp_path / "offline",
        iteration=2,
    )
    assert outcome.update_performed is False
    assert outcome.artifact["kind"] == "treepo_dspy_g"


def test_treepo_fit_reports_direct_dspy_compiler_role_provenance(tmp_path: Path) -> None:
    k = 3
    prediction = {name: 0.5 for name in _names(k)}
    compiler = _FakeCompiler(prediction=prediction)
    tree = _tree(k)
    backend = {
        **_config(k, compiler=compiler),
        "f_record_source": "generated_when_available",
        "output_dir": str(tmp_path / "fit"),
    }
    result = fit(
        {
            "family": "dspy",
            "train_data": [tree],
            "eval_data": [tree],
            "oracle_targets": [
                {"target_name": name, "oracle_id": f"oracle:{name}"} for name in _names(k)
            ],
            "backend_config": backend,
            "axis": {"max_iterations": 3, "axis_value": 2},
        }
    )

    assert result.status == "success"
    assert [call.kind for call in compiler.calls] == ["f", "g", "f"]
    assert result.artifacts["f"]["f_current_g_generation_executed"] is True
    assert result.artifacts["f"]["f_state_sources"] == ["current_g_generated"]
    assert result.summary["g_update_count"] == 1
    contract = result.summary["g_contract"]
    assert contract["operator"] == "learned_shared"
    assert contract["g_training_call_roles"] == ["leaf", "merge"]
    assert contract["g_training_role_evidence_source"] == "dspy_compiler_mixed_examples"
    assert contract["shared_g_updated_with_merge_domain"] is True
    assert result.artifacts["g"]["universal_c3_established"] is False


def test_explicit_validation_split_is_never_moved_to_cover_a_missing_train_role(
    tmp_path: Path,
) -> None:
    k = 1
    prediction = {name: 0.5 for name in _names(k)}
    compiler = _FakeCompiler(prediction=prediction)
    family = build_dspy_family(_config(k, compiler=compiler, validation_fraction=0.5))
    train_tree = _tree(k, singleton=True)
    raw_val = _tree(k)
    val_tree = replace(
        raw_val,
        tree_id="tree-val",
        metadata={**dict(raw_val.metadata), "split": "val"},
        nodes=tuple(
            replace(node, metadata={**dict(node.metadata), "split": "val"})
            for node in raw_val.nodes
        ),
    )

    artifact = family.train_g(
        g_init=None,
        f=_FakeProgram("f", prediction=prediction),
        traces=[train_tree, val_tree],
        output_dir=tmp_path / "explicit-split",
        iteration=2,
    ).artifact

    assert artifact["g_training_call_roles"] == ["leaf"]
    assert artifact["leaf_domain_training_row_count"] == 1
    assert artifact["merge_domain_training_row_count"] == 0
    assert artifact["leaf_domain_validation_row_count"] == 2
    assert artifact["merge_domain_validation_row_count"] == 1
    assert artifact["split"]["strategy"] == "explicit_split_metadata_grouped"
    call = [entry for entry in compiler.calls if entry.kind == "g"][0]
    assert all(row.source_split == "train" for row in call.trainset)
    assert all(row.source_split == "val" for row in call.valset)


def _scheduled_family(k: int, *, rate: float, compiler: _FakeCompiler):
    current_g = _FakeProgram("g")
    family = build_dspy_family(
        {
            **_config(k, compiler=compiler),
            "g_program": current_g,
            "g_scheduled_sampling_rate": rate,
        }
    )
    return family, current_g


def test_g_scheduled_sampling_defaults_validate_and_ramp() -> None:
    from treepo.methods.dspy import DSPyFamilyConfig

    defaults = DSPyFamilyConfig()
    assert defaults.g_scheduled_sampling_rate == 0.0
    assert defaults.g_scheduled_sampling_rate_start == 0.0
    assert defaults.g_scheduled_sampling_ramp_per_iter == 0.0

    family = build_dspy_family(
        {
            **_config(1),
            "g_scheduled_sampling_rate": 0.8,
            "g_scheduled_sampling_rate_start": 0.1,
            "g_scheduled_sampling_ramp_per_iter": 0.3,
        }
    )
    assert family._dspy_scheduled_sampling_rate(iteration=0) == pytest.approx(0.1)
    assert family._dspy_scheduled_sampling_rate(iteration=1) == pytest.approx(0.4)
    assert family._dspy_scheduled_sampling_rate(iteration=2) == pytest.approx(0.7)
    assert family._dspy_scheduled_sampling_rate(iteration=10) == pytest.approx(0.8)

    with pytest.raises(ValueError, match="g_scheduled_sampling_rate must be in"):
        DSPyFamilyConfig(g_scheduled_sampling_rate=-0.1)
    with pytest.raises(ValueError, match="g_scheduled_sampling_rate_start must be in"):
        DSPyFamilyConfig(g_scheduled_sampling_rate_start=1.1)


def test_g_scheduled_sampling_rate_zero_is_prompt_identical_and_never_calls_current_g(
    tmp_path: Path,
) -> None:
    k = 1
    compiler = _FakeCompiler(prediction={name: 0.5 for name in _names(k)})
    family, current_g = _scheduled_family(k, rate=0.0, compiler=compiler)
    tree = _tree(k)
    baseline_prompts = [row.prompt for row in family._dspy_g_examples([tree])]

    artifact = family.train_g(
        g_init=None,
        f=_FakeProgram("f", prediction={name: 0.5 for name in _names(k)}),
        traces=[tree],
        output_dir=tmp_path / "rate0",
        iteration=7,
    ).artifact

    call = [entry for entry in compiler.calls if entry.kind == "g"][0]
    assert [row.prompt for row in call.trainset] == baseline_prompts
    assert current_g.calls == []
    assert all(row.scheduled_sampling_rate == 0.0 for row in call.trainset)
    assert not any(row.used_generated_children for row in call.trainset)
    assert artifact["scheduled_sampling_rate"] == 0.0
    assert artifact["g_scheduled_sampling"]["current_g_generation_executed"] is False


def test_g_scheduled_sampling_rate_one_uses_current_g_only_in_parent_prompts(
    tmp_path: Path,
) -> None:
    k = 1
    compiler = _FakeCompiler(prediction={name: 0.5 for name in _names(k)})
    family, current_g = _scheduled_family(k, rate=1.0, compiler=compiler)
    tree = _tree(k)

    artifact = family.train_g(
        g_init=None,
        f=_FakeProgram("f", prediction={name: 0.5 for name in _names(k)}),
        traces=[tree],
        output_dir=tmp_path / "rate1",
        iteration=0,
    ).artifact

    call = [entry for entry in compiler.calls if entry.kind == "g"][0]
    leaf_rows = [row for row in call.trainset if row.role == "leaf"]
    parent_rows = [row for row in call.trainset if row.role in {"recompression", "merge"}]
    assert len(current_g.calls) == len(tree.nodes)
    assert leaf_rows and parent_rows
    assert all("g-v0:" not in row.prompt for row in leaf_rows)
    assert all("g-v0:" in row.prompt for row in parent_rows)
    assert all(not row.used_generated_children for row in leaf_rows)
    assert all(row.used_generated_children for row in parent_rows)
    assert all(row.scheduled_sampling_rate == 1.0 for row in call.trainset)
    provenance = artifact["g_scheduled_sampling"]
    assert provenance["generated_node_state_count"] == len(tree.nodes)
    assert provenance["used_generated_children_row_count"] == len(parent_rows)
    assert provenance["selection_method"] == "sha256_64_per_tree_child"


def test_g_scheduled_sampling_partial_selection_is_sha256_stable(
    tmp_path: Path,
) -> None:
    k = 1
    captures = []
    for run in range(2):
        compiler = _FakeCompiler(prediction={name: 0.5 for name in _names(k)})
        family, _current_g = _scheduled_family(k, rate=0.9, compiler=compiler)
        family.train_g(
            g_init=None,
            f=_FakeProgram("f", prediction={name: 0.5 for name in _names(k)}),
            traces=[_tree(k)],
            output_dir=tmp_path / f"partial-{run}",
            iteration=3,
        )
        call = [entry for entry in compiler.calls if entry.kind == "g"][0]
        captures.append(
            [(row.row_id, row.prompt, row.used_generated_children) for row in call.trainset]
        )

    assert captures[0] == captures[1]
    parent_prompt = next(prompt for _row_id, prompt, used in captures[0] if used)
    assert "g-v0:" in parent_prompt
    assert "right-summary" in parent_prompt
    assert "left-summary" not in parent_prompt
