from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo.methods.dspy import build_dspy_family
from treepo.tree import TreeNode, TreeRecord

dspy = pytest.importorskip("dspy")


TARGET = {"score": 0.75}


class _CustomPredict(dspy.Predict):
    """Predict subclass whose behavior would be lost by bare-Predict loading."""

    def __init__(self, marker: str = "custom-architecture") -> None:
        super().__init__("state -> prediction_json")
        self.marker = marker

    def forward(self, *, state: str):
        return dspy.Prediction(
            prediction_json=json.dumps({"score": 0.75 if state.startswith("custom:") else 0.25})
        )


class _ExternalAdapter:
    def __init__(self, marker: str = "external") -> None:
        self.marker = str(marker)

    def __call__(self, **_kwargs):
        return {"prediction_json": json.dumps(TARGET)}

    def save(self, path: str, save_program: bool = False) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "marker": self.marker,
                    "save_program": bool(save_program),
                }
            ),
            encoding="utf-8",
        )


class _IdentityCompiler:
    def __init__(self) -> None:
        self.programs: list[object] = []

    def compile(self, *, program, **_kwargs):
        self.programs.append(program)
        return program


def _tree() -> TreeRecord:
    metadata = {
        "split": "train",
        "target_vector": dict(TARGET),
        "target_summary": "gold state",
    }
    return TreeRecord(
        tree_id="persistence-fixture",
        text="whole document",
        root_label=dict(TARGET),
        nodes=(
            TreeNode(
                node_id="root",
                unit_type="root",
                text="whole document",
                state="gold state",
                label=dict(TARGET),
                metadata=metadata,
            ),
        ),
        metadata=metadata,
    )


def _config(**overrides):
    values = {
        "target_names": ("score",),
        "target_oracle_ids": ("fixture:score",),
        "target_dim": 1,
        "target_vector_key": "target_vector",
        "node_target_key": "target_vector",
        "optimizer": "bootstrap",
        "validation_fraction": 0.0,
        "audit_laws": False,
    }
    values.update(overrides)
    return values


def _train_f(family, output_dir: Path, *, f_init=None, iteration: int = 1):
    return family.train_f(
        f_init=f_init,
        g=None,
        traces=[_tree()],
        output_dir=output_dir,
        iteration=iteration,
    )


def test_custom_predict_uses_opted_in_whole_program_and_preserves_behavior(
    tmp_path: Path,
) -> None:
    compiler = _IdentityCompiler()
    family = build_dspy_family(
        _config(
            f_program=_CustomPredict(),
            compiler=compiler,
            allow_pickle_program_load=True,
        )
    )
    artifact = _train_f(family, tmp_path / "first")

    path = Path(artifact["program_path"])
    assert path.is_dir()
    assert (path / "program.pkl").exists()
    assert artifact["program_format"] == "dspy_whole_program_pickle"
    assert artifact["program_reconstruction"] == "dspy_load_v1"
    assert artifact["program_requires_pickle"] is True
    assert artifact["program_class"].endswith("._CustomPredict")

    reloaded = build_dspy_family(_config(allow_pickle_program_load=True))
    loaded = reloaded._dspy_resolve_program(
        artifact,
        kind="f",
        fallback=None,
        create=False,
    )
    assert type(loaded) is _CustomPredict
    assert loaded.marker == "custom-architecture"
    assert json.loads(loaded(state="custom:input").prediction_json) == TARGET


def test_custom_dspy_module_requires_explicit_pickle_opt_in(tmp_path: Path) -> None:
    family = build_dspy_family(
        _config(
            f_program=_CustomPredict(),
            compiler=_IdentityCompiler(),
            allow_pickle_program_load=False,
        )
    )
    with pytest.raises(ValueError, match="allow_pickle_program_load=True"):
        _train_f(family, tmp_path)


def test_pickle_opt_in_rejects_truthy_non_boolean_values() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        build_dspy_family(_config(allow_pickle_program_load="true"))


def test_injected_exact_predict_is_not_mistaken_for_package_created_default(
    tmp_path: Path,
) -> None:
    family = build_dspy_family(
        _config(
            f_program=dspy.Predict("state -> prediction_json"),
            compiler=_IdentityCompiler(),
            allow_pickle_program_load=False,
        )
    )
    with pytest.raises(ValueError, match="allow_pickle_program_load=True"):
        _train_f(family, tmp_path)


def test_whole_program_reload_requires_opt_in_even_when_artifact_exists(
    tmp_path: Path,
) -> None:
    family = build_dspy_family(
        _config(
            f_program=_CustomPredict(),
            compiler=_IdentityCompiler(),
            allow_pickle_program_load=True,
        )
    )
    artifact = _train_f(family, tmp_path)

    untrusted = build_dspy_family(_config(allow_pickle_program_load=False))
    with pytest.raises(RuntimeError, match="requires pickle loading"):
        untrusted._dspy_resolve_program(
            artifact,
            kind="f",
            fallback=None,
            create=False,
        )


def test_saver_without_loader_is_rejected(tmp_path: Path) -> None:
    family = build_dspy_family(
        _config(
            f_program=_ExternalAdapter(),
            compiler=_IdentityCompiler(),
            program_saver=lambda **_kwargs: None,
        )
    )
    with pytest.raises(ValueError, match="program_saver requires program_loader"):
        _train_f(family, tmp_path)


def test_external_adapter_without_loader_is_rejected(tmp_path: Path) -> None:
    family = build_dspy_family(
        _config(
            f_program=_ExternalAdapter(),
            compiler=_IdentityCompiler(),
        )
    )
    with pytest.raises(TypeError, match="require program_loader"):
        _train_f(family, tmp_path)


def test_external_adapter_uses_loader_and_never_bare_predict(
    tmp_path: Path,
) -> None:
    def loader(*, path: Path, **_kwargs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _ExternalAdapter(str(payload["marker"]))

    family = build_dspy_family(
        _config(
            f_program=_ExternalAdapter("roundtrip"),
            compiler=_IdentityCompiler(),
            program_loader=loader,
        )
    )
    artifact = _train_f(family, tmp_path)
    assert artifact["program_format"] == "custom_program"
    assert artifact["program_reconstruction"] == "program_loader_v1"

    reloaded = build_dspy_family(_config(program_loader=loader))
    loaded = reloaded._dspy_resolve_program(
        artifact,
        kind="f",
        fallback=None,
        create=False,
    )
    assert type(loaded) is _ExternalAdapter
    assert loaded.marker == "roundtrip"

    no_loader = build_dspy_family(_config())
    with pytest.raises(RuntimeError, match="requires program_loader"):
        no_loader._dspy_resolve_program(
            artifact,
            kind="f",
            fallback=None,
            create=False,
        )


def test_legacy_state_artifact_warns_and_uses_default_predict(
    tmp_path: Path,
) -> None:
    family = build_dspy_family(
        _config(
            compiler=SimpleNamespace(
                compile=lambda program, **_kwargs: program,
            )
        )
    )
    artifact = dict(_train_f(family, tmp_path))
    artifact.pop("program_reconstruction")
    artifact.pop("program_class")
    artifact.pop("program_requires_pickle")

    reloaded = build_dspy_family(_config())
    with pytest.warns(RuntimeWarning, match="legacy DSPy artifact"):
        loaded = reloaded._dspy_resolve_program(
            artifact,
            kind="f",
            fallback=None,
            create=False,
        )
    assert type(loaded) is dspy.Predict


def test_fresh_family_warmstarts_custom_program_without_architecture_loss(
    tmp_path: Path,
) -> None:
    first = build_dspy_family(
        _config(
            f_program=_CustomPredict("warmstart"),
            compiler=_IdentityCompiler(),
            allow_pickle_program_load=True,
        )
    )
    first_artifact = _train_f(first, tmp_path / "first", iteration=1)

    compiler = _IdentityCompiler()
    second = build_dspy_family(
        _config(
            compiler=compiler,
            allow_pickle_program_load=True,
        )
    )
    second_artifact = _train_f(
        second,
        tmp_path / "second",
        f_init=first_artifact,
        iteration=2,
    )

    assert type(compiler.programs[0]) is _CustomPredict
    assert compiler.programs[0].marker == "warmstart"
    assert second_artifact["program_reconstruction"] == "dspy_load_v1"


def test_digest_and_loaded_class_are_verified(tmp_path: Path) -> None:
    family = build_dspy_family(
        _config(
            f_program=_CustomPredict(),
            compiler=_IdentityCompiler(),
            allow_pickle_program_load=True,
        )
    )
    artifact = dict(_train_f(family, tmp_path))

    digest_mismatch = {**artifact, "program_sha256": "0" * 64}
    reloaded = build_dspy_family(_config(allow_pickle_program_load=True))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        reloaded._dspy_resolve_program(
            digest_mismatch,
            kind="f",
            fallback=None,
            create=False,
        )

    class_mismatch = {**artifact, "program_class": "builtins.object"}
    reloaded = build_dspy_family(_config(allow_pickle_program_load=True))
    with pytest.raises(RuntimeError, match="class mismatch"):
        reloaded._dspy_resolve_program(
            class_mismatch,
            kind="f",
            fallback=None,
            create=False,
        )


def test_default_program_warmstart_rejects_target_catalog_mismatch(
    tmp_path: Path,
) -> None:
    trained = build_dspy_family(
        _config(
            compiler=_IdentityCompiler(),
        )
    )
    artifact = _train_f(trained, tmp_path)

    assert artifact["target_oracle_ids"] == ["fixture:score"]
    assert artifact["signature_contract"]["target_names"] == ["score"]
    incompatible = build_dspy_family(
        _config(
            target_names=("left", "right"),
            target_oracle_ids=("fixture:left", "fixture:right"),
            target_dim=2,
        )
    )

    with pytest.raises(ValueError, match="target_names"):
        incompatible._dspy_resolve_program(
            artifact,
            kind="f",
            fallback=None,
            create=False,
        )


def test_default_program_warmstart_rejects_signature_instruction_mismatch(
    tmp_path: Path,
) -> None:
    trained = build_dspy_family(
        _config(
            compiler=_IdentityCompiler(),
            f_signature_instructions="Original task-owned f instructions.",
        )
    )
    artifact = _train_f(trained, tmp_path)
    incompatible = build_dspy_family(
        _config(
            f_signature_instructions="Different task-owned f instructions.",
        )
    )

    with pytest.raises(ValueError, match="instructions_sha256"):
        incompatible._dspy_resolve_program(
            artifact,
            kind="f",
            fallback=None,
            create=False,
        )
