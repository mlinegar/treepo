from __future__ import annotations

from pathlib import Path

import pytest

from treepo.methods.dspy import build_dspy_family
from treepo.tree import TreeNode, TreeRecord


def _native_tree() -> TreeRecord:
    target = {"rile_normalized": 0.625}
    shared = {
        "split": "train",
        "target_vector": target,
        "target_summary": "compact target state",
    }
    return TreeRecord(
        tree_id="native-dspy-roundtrip",
        text="complete manifesto",
        root_label=target,
        nodes=(
            TreeNode(
                node_id="left",
                unit_type="leaf",
                text="left span",
                parent_id="root",
                label=target,
                state="left compact state",
                metadata={**shared, "target_summary": "left compact state"},
            ),
            TreeNode(
                node_id="right",
                unit_type="leaf",
                text="right span",
                parent_id="root",
                label=target,
                state="right compact state",
                metadata={**shared, "target_summary": "right compact state"},
            ),
            TreeNode(
                node_id="root",
                unit_type="root",
                text="complete manifesto",
                left_child_id="left",
                right_child_id="right",
                label=target,
                state="root compact state",
                metadata={**shared, "target_summary": "root compact state"},
            ),
        ),
        metadata=shared,
    )


def test_native_dspy3_bootstrap_save_reload_for_f_and_g_without_network(
    tmp_path: Path,
) -> None:
    dspy = pytest.importorskip("dspy")
    assert int(str(dspy.__version__).split(".", maxsplit=1)[0]) == 3

    config = {
        "target_names": ("rile_normalized",),
        "target_oracle_ids": ("fixture:rile",),
        "target_dim": 1,
        "target_vector_key": "target_vector",
        "node_target_key": "target_vector",
        "target_min": 0.0,
        "target_max": 1.0,
        "optimizer": "bootstrap",
        # Zero bootstrapped demos makes this a completely local labeled-demo
        # compile. DSPy 3.0.4 performs no LM/provider call on this path.
        "optimizer_kwargs": {
            "max_bootstrapped_demos": 0,
            "max_labeled_demos": 4,
        },
        "validation_fraction": 0.0,
        "allow_identity_g_targets": False,
        "dspy_module": dspy,
        "audit_laws": False,
    }
    tree = _native_tree()
    family = build_dspy_family(config)
    identity_g = {
        "kind": "treepo_identity_g",
        "g_mode": "identity",
        "operator": "identity",
    }

    f_artifact = family.train_f(
        f_init=None,
        g=identity_g,
        traces=[tree],
        output_dir=tmp_path / "f",
        iteration=1,
    )
    g_artifact = family.train_g(
        g_init=None,
        f=f_artifact,
        traces=[tree],
        output_dir=tmp_path / "g",
        iteration=2,
    ).artifact

    assert Path(f_artifact["program_path"]).exists()
    assert Path(g_artifact["program_path"]).exists()
    assert f_artifact["compile_status"] == "success"
    assert g_artifact["compile_status"] == "success"
    assert f_artifact["program_format"] == "dspy_json_state"
    assert g_artifact["program_format"] == "dspy_json_state"
    assert f_artifact["program_reconstruction"] == "treepo_default_predict_v1"
    assert g_artifact["program_reconstruction"] == "treepo_default_predict_v1"
    assert f_artifact["program_requires_pickle"] is False
    assert g_artifact["program_requires_pickle"] is False
    assert Path(f_artifact["program_path"]).suffix == ".json"
    assert Path(g_artifact["program_path"]).suffix == ".json"

    reloaded = build_dspy_family(config)
    reloaded.validate_artifact(kind="f", artifact=f_artifact)
    reloaded.validate_artifact(kind="g", artifact=g_artifact)
    loaded_f = reloaded._dspy_resolve_program(
        f_artifact,
        kind="f",
        fallback=None,
        create=False,
    )
    loaded_g = reloaded._dspy_resolve_program(
        g_artifact,
        kind="g",
        fallback=None,
        create=False,
    )
    assert loaded_f is not None
    assert loaded_g is not None
    assert getattr(loaded_f, "demos", None)
    assert getattr(loaded_g, "demos", None)
