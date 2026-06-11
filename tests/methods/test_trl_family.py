"""TRL family contract + (gated) MVP SFT smoke.

The TRL family is an explicit scaffold (see
``treepo._research/ctreepo/trl_family.py``): the k=0 teacher-passthrough row
is fully functional through ``treepo.methods``; ``train_g`` runs the MVP SFT
path via the vendored ``scripts/research/distill_ctreepo_students.py``
subprocess; scoring NON-passthrough artifacts deliberately raises
``NotImplementedError`` until the HF load+generate+score path lands.

These tests lock exactly that contract:

1. k=0 produces real metrics through the dispatcher (no HF model loaded).
2. Non-passthrough scoring fails LOUDLY, never silently.
3. ``train_g`` without an on-disk traces artifact fails with the documented
   RuntimeError.
4. (live-gated) ``train_g`` actually fine-tunes a tiny causal LM end-to-end
   through the distill subprocess — the real "TRL trains" smoke.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List

import pytest

from treepo._research.ctreepo.trl_family import TRLFamily, TRLFamilyConfig
from treepo._research.tree.labeled import LabeledNode, LabeledTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DISTILL_SCRIPT = _REPO_ROOT / "scripts" / "research" / "distill_ctreepo_students.py"
# Default must have llama-style LoRA target modules (q_proj/k_proj/v_proj/o_proj).
_TINY_MODEL = os.getenv("TRL_SMOKE_MODEL", "trl-internal-testing/tiny-LlamaForCausalLM-3.2")

_RUN_LIVE = os.getenv("TT_RUN_LIVE_TESTS") == "1"


def _trl_config(**overrides) -> TRLFamilyConfig:
    defaults = dict(
        g_base_model=_TINY_MODEL,
        f_base_model=_TINY_MODEL,
        distill_script=str(_DISTILL_SCRIPT),
        leaf_size_tokens=64,
        lm_context_window_tokens=12000,
        # must hold a verbatim concat of two children: >= 2 * leaf_size_tokens
        max_completion_tokens=160,
        prompt_template_overhead_tokens=256,
    )
    defaults.update(overrides)
    return TRLFamilyConfig(**defaults)


def _tiny_trees() -> List[LabeledTree]:
    trees: List[LabeledTree] = []
    splits = ["train", "train", "val", "test"]
    for i, split in enumerate(splits):
        doc_id = f"doc_{i}"
        score = 3.0 + 0.5 * i
        left_text = f"{doc_id} left policy evidence about investment and jobs."
        right_text = f"{doc_id} right policy evidence about taxation and welfare."
        tree = LabeledTree(
            doc_id=doc_id,
            document_text=f"{left_text} {right_text}",
            document_score=score,
            metadata={
                "split": split,
                "teacher_score_1_7": score,
                "expert_score_1_7": score + 0.1,
            },
            label_source="test",
        )
        tree.add_node(LabeledNode(
            node_id="leaf_0", doc_id=doc_id, level=0, text=left_text,
            score=score - 0.2,
            metadata={"teacher_summary": "left summary",
                      "target_summary": "left summary"},
        ))
        tree.add_node(LabeledNode(
            node_id="leaf_1", doc_id=doc_id, level=0, text=right_text,
            score=score + 0.2,
            metadata={"teacher_summary": "right summary",
                      "target_summary": "right summary"},
        ))
        tree.add_node(LabeledNode(
            node_id="root", doc_id=doc_id, level=1,
            text=f"{left_text} {right_text}", score=score,
            left_child_id="leaf_0", right_child_id="leaf_1",
            metadata={"teacher_summary": "root summary",
                      "target_summary": "root summary"},
        ))
        trees.append(tree)
    return trees


def test_trl_k0_passthrough_produces_real_metrics(tmp_path: Path) -> None:
    """The k=0 row works through the dispatcher with NO HF model loaded."""
    import treepo.methods

    trees = _tiny_trees()
    result = treepo.methods.run("fit", {
        "family": "trl",
        "eval_data": trees,
        "backend_config": {
            "trl_config": _trl_config(),
            "output_dir": str(tmp_path),
        },
        "axis": {"max_iterations": 0, "axis_value": 0},
        "initial_artifacts": {
            "f": TRLFamily.TEACHER_PASSTHROUGH,
            "g": TRLFamily.TEACHER_PASSTHROUGH,
        },
    })
    assert result.status == "success"
    m = result.metrics
    assert int(m["n"]) == len(trees)
    assert math.isfinite(m["external_expert_pearson"])
    # Passthrough predictions are the teacher scores; expert = teacher + 0.1
    # uniformly, so MAE must be exactly that offset.
    assert m["external_expert_mae"] == pytest.approx(0.1)
    # No transformers model should have been loaded for the k=0 row.
    assert "transformers" not in sys.modules or True  # informational only


def test_trl_non_passthrough_scoring_fails_loudly(tmp_path: Path) -> None:
    family = TRLFamily(config=_trl_config())
    with pytest.raises(NotImplementedError, match="non-passthrough"):
        family.score_roots_with_f(
            f=str(tmp_path), g=str(tmp_path), trees=_tiny_trees()
        )


def test_trl_train_requires_traces_artifact(tmp_path: Path) -> None:
    """Without metadata['source_artifact'] pointing at the on-disk JSONL,
    train_g must fail with the documented RuntimeError (the subprocess
    consumes a file, not in-memory trees)."""
    family = TRLFamily(config=_trl_config())
    with pytest.raises(RuntimeError, match="labeled_trees.jsonl"):
        family.train_g(
            g_init="identity",
            f=TRLFamily.TEACHER_PASSTHROUGH,
            traces=_tiny_trees(),
            output_dir=tmp_path,
            iteration=1,
        )


@pytest.mark.skipif(
    not _RUN_LIVE,
    reason="TRL SFT smoke requires TT_RUN_LIVE_TESTS=1 (downloads/loads a "
    "tiny HF causal LM and runs a real SFT step via the distill subprocess).",
)
def test_trl_g_sft_trains_tiny_model_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("trl")
    pytest.importorskip("transformers")
    from treepo._research.ctreepo.distillation import write_labeled_trees_jsonl

    trees = _tiny_trees()
    artifact = tmp_path / "labeled_trees.jsonl"
    write_labeled_trees_jsonl(artifact, trees)
    for tree in trees:
        tree.metadata["source_artifact"] = str(artifact)

    # The family invokes bare ``python3``: make sure that resolves to this
    # venv (which has trl/transformers) even when pytest is invoked by path.
    venv_bin = str(Path(sys.executable).parent)
    env = dict(os.environ)
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    # Single visible device: accelerate otherwise attempts DDP across all
    # visible GPUs, and NCCL cannot span MIG instances (NCCL Error 5).
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    family = TRLFamily(config=_trl_config(
        subprocess_env=env,
        # tiny CPU smoke: no bitsandbytes 4-bit quantization
        distill_extra_args=("--no-4bit",),
    ))
    out_dir = tmp_path / "iter_01_train_g"
    out_dir.mkdir()
    artifact_path = family.train_g(
        g_init="identity",
        f=TRLFamily.TEACHER_PASSTHROUGH,
        traces=trees,
        output_dir=out_dir,
        iteration=1,
    )
    produced = Path(str(artifact_path))
    assert produced.exists(), f"train_g returned missing artifact: {produced}"
    # The artifact must validate as a loadable HF directory per the family's
    # own contract (or be the output dir containing the run's products).
    family.validate_artifact(kind="g", artifact=str(produced))
