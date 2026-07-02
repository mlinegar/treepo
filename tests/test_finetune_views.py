from __future__ import annotations

import json
from pathlib import Path

from treepo import Candidate, PreferenceDataset, PreferenceRecord, TaskState
from treepo.finetune import (
    build_finetune_views,
    export_finetune_views,
    export_for_adapter,
    get_finetune_adapter,
    list_finetune_adapters,
)


def _dataset() -> PreferenceDataset:
    state = TaskState(
        kind="policy_state",
        counts={"left": 2.0},
        measures={"score": -0.4},
        text="left policy evidence",
    )
    return PreferenceDataset(
        [
            PreferenceRecord(
                record_id="root_gold",
                unit_id="doc1:root",
                unit_type="root",
                target="f",
                context="Score the document.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="root",
                level=1,
                position=0,
                left_child_id="leaf0",
                right_child_id="leaf1",
                candidates=(Candidate(id="gold", value="score: -0.4", score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="leaf_gold",
                unit_id="doc1:leaf0",
                unit_type="qsentence",
                target="g",
                context="Encode qsentence leaf0.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="leaf0",
                level=0,
                position=0,
                parent_id="root",
                candidates=(Candidate(id="gold", value=state, score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="preferred_pair",
                unit_id="doc1:leaf0:pair",
                unit_type="qsentence",
                target="g",
                context="Choose a qsentence state.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="leaf0",
                level=0,
                position=0,
                parent_id="root",
                candidates=(
                    Candidate(id="chosen", value=state, score=0.9, preferred=True),
                    Candidate(id="rejected", value="generic", score=0.2),
                ),
                weight=2.0,
                propensity=0.5,
            ),
            PreferenceRecord(
                record_id="tie_group",
                unit_id="doc1:leaf1:tie",
                unit_type="qsentence",
                target="g",
                context="Tie group.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="leaf1",
                level=0,
                position=1,
                parent_id="root",
                candidates=(
                    Candidate(id="tie_a", value="tie A", score=0.5, rank=1),
                    Candidate(id="tie_b", value="tie B", score=0.5, rank=1),
                ),
            ),
            PreferenceRecord(
                record_id="ranked_group",
                unit_id="doc1:root:ranked",
                unit_type="merge",
                target="g",
                context="Rank merged states.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="root",
                level=1,
                position=0,
                left_child_id="leaf0",
                right_child_id="leaf1",
                candidates=(
                    Candidate(id="best", value="best", score=0.95, rank=1),
                    Candidate(id="ok_a", value="ok A", score=0.7, rank=2),
                    Candidate(id="ok_b", value="ok B", score=0.7, rank=2),
                ),
            ),
        ]
    )


def test_build_finetune_views_projects_all_supported_shapes() -> None:
    views = build_finetune_views(_dataset())

    assert set(views) == {
        "embedding_pairs",
        "embedding_triplets",
        "embedding_ranked",
        "sft",
        "dpo",
        "reward",
        "grpo",
    }
    assert len(views["embedding_pairs"]) == len(views["sft"]) == 6
    assert len(views["embedding_triplets"]) == len(views["dpo"]) == len(views["reward"]) == 2
    assert len(views["embedding_ranked"]) == len(views["grpo"]) == 3

    state_pair = next(row for row in views["embedding_pairs"] if row["metadata"]["unit_id"] == "doc1:leaf0")
    assert json.loads(state_pair["positive"])["kind"] == "policy_state"
    assert state_pair["score"] == 1.0

    triplet = next(row for row in views["embedding_triplets"] if row["metadata"]["unit_id"] == "doc1:leaf0:pair")
    assert triplet["positive_score"] == 0.9
    assert triplet["negative_score"] == 0.2
    assert triplet["sample_weight"] == 4.0

    assert all(row["metadata"]["unit_id"] != "doc1:leaf1:tie" for row in views["embedding_triplets"])
    tie_group = next(row for row in views["embedding_ranked"] if row["metadata"]["unit_id"] == "doc1:leaf1:tie")
    assert tie_group["ranks"] == [1, 1]
    assert tie_group["texts"] == ["tie A", "tie B"]


def test_finetune_views_preserve_tree_unit_metadata() -> None:
    views = build_finetune_views(_dataset())

    for rows in views.values():
        for row in rows:
            metadata = row["metadata"]
            for key in ("tree_id", "doc_id", "node_id", "unit_id", "unit_type", "level", "position"):
                assert key in metadata
            assert metadata["tree_id"] == "doc1"
            assert metadata["doc_id"] == "doc1"

    root_row = next(row for row in views["sft"] if row["metadata"]["unit_id"] == "doc1:root")
    assert root_row["metadata"]["left_child_id"] == "leaf0"
    assert root_row["metadata"]["right_child_id"] == "leaf1"
    leaf_row = next(row for row in views["sft"] if row["metadata"]["unit_id"] == "doc1:leaf0")
    assert leaf_row["metadata"]["parent_id"] == "root"


def test_export_finetune_views_writes_jsonl_json_and_hf_dataset(tmp_path: Path) -> None:
    artifacts = export_finetune_views(_dataset(), tmp_path)

    assert artifacts["counts"]["embedding_pairs"] == 6
    assert artifacts["counts"]["embedding_triplets"] == 2
    assert artifacts["counts"]["embedding_ranked"] == 3
    assert Path(artifacts["files"]["embedding_pairs"]).exists()
    assert Path(artifacts["files"]["embedding_ranked"]).exists()
    assert Path(artifacts["files"]["hf_dataset"]).exists()

    pair_line = Path(artifacts["files"]["embedding_pairs"]).read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(pair_line)["metadata"]["tree_id"] == "doc1"
    ranked_rows = json.loads(Path(artifacts["files"]["embedding_ranked"]).read_text(encoding="utf-8"))
    assert ranked_rows[0]["metadata"]["tree_id"] == "doc1"


def test_build_finetune_views_can_select_subset() -> None:
    views = build_finetune_views(_dataset(), views=("sft", "dpo"))

    assert set(views) == {"sft", "dpo"}
    assert views["sft"][0]["prompt"] == "Score the document."



def test_finetune_adapter_registry_exposes_builtin_exporters() -> None:
    names = {adapter.name for adapter in list_finetune_adapters()}

    assert {
        "generic_jsonl",
        "embedding",
        "trl_sft",
        "trl_dpo",
        "trl_reward",
        "trl_scalar_reward",
        "trl_grpo",
        "dspy_examples",
    } <= names
    adapter = get_finetune_adapter("trl_dpo")
    assert adapter.framework == "trl"
    assert adapter.required_views == ("dpo",)


def test_trl_adapters_export_framework_ready_rows(tmp_path: Path) -> None:
    dpo = export_for_adapter("trl_dpo", _dataset(), tmp_path / "dpo", save_hf=False)
    dpo_row = json.loads(Path(dpo["files"]["dpo"]).read_text(encoding="utf-8").splitlines()[0])
    assert set(dpo_row) >= {"prompt", "chosen", "rejected", "sample_weight", "metadata"}
    assert dpo_row["metadata"]["tree_id"] == "doc1"

    reward = export_for_adapter("trl_reward", _dataset(), tmp_path / "reward", save_hf=False)
    reward_row = json.loads(Path(reward["files"]["reward"]).read_text(encoding="utf-8").splitlines()[0])
    assert set(reward_row) >= {"prompt", "chosen", "rejected", "chosen_score", "rejected_score"}

    scalar = export_for_adapter("trl_scalar_reward", _dataset(), tmp_path / "scalar", save_hf=False)
    scalar_row = json.loads(Path(scalar["files"]["sft"]).read_text(encoding="utf-8").splitlines()[0])
    assert set(scalar_row) >= {"prompt", "response", "score", "sample_weight", "metadata"}
    assert isinstance(scalar_row["score"], float)

    grpo = export_for_adapter("trl_grpo", _dataset(), tmp_path / "grpo", save_hf=False)
    grpo_row = json.loads(Path(grpo["files"]["grpo"]).read_text(encoding="utf-8").splitlines()[0])
    assert set(grpo_row) >= {"prompt", "responses", "ranks", "scores", "sample_weight", "metadata"}


def test_embedding_and_dspy_adapters_preserve_metadata(tmp_path: Path) -> None:
    embedding = export_for_adapter("embedding", _dataset(), tmp_path / "embedding", save_hf=False)
    assert embedding["counts"]["embedding_pairs"] == 6
    pair = json.loads(Path(embedding["files"]["embedding_pairs"]).read_text(encoding="utf-8").splitlines()[0])
    assert set(pair) >= {"anchor", "positive", "score", "sample_weight", "metadata"}
    assert pair["metadata"]["doc_id"] == "doc1"

    dspy = export_for_adapter("dspy_examples", _dataset(), tmp_path / "dspy", save_hf=False)
    sft_row = json.loads(Path(dspy["files"]["sft"]).read_text(encoding="utf-8").splitlines()[0])
    dpo_row = json.loads(Path(dspy["files"]["dpo"]).read_text(encoding="utf-8").splitlines()[0])
    assert sft_row["dspy_inputs"] == ["prompt"]
    assert set(dpo_row) >= {"prompt", "summary_a", "summary_b", "preferred", "metadata"}
    assert dpo_row["preferred"] == "A"
    assert dpo_row["metadata"]["tree_id"] == "doc1"


def test_finetune_adapter_import_does_not_load_trainer_frameworks() -> None:
    import subprocess
    import sys

    code = """
import json, sys
import treepo.finetune
mods = ["sentence_transformers", "trl", "peft", "accelerate", "dspy"]
print(json.dumps({name: name in sys.modules for name in mods}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == {
        "accelerate": False,
        "dspy": False,
        "peft": False,
        "sentence_transformers": False,
        "trl": False,
    }
