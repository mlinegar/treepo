"""LLM startup + batching pass-through verification.

The manifesto pipeline starts an LLM via either of two paths:

- ``lm_transport='batch'``: constructs ``BatchedDSPyLM`` with
  ``batch_size`` / ``batch_max_concurrent`` / ``batch_timeout`` /
  ``batch_routing_policy`` knobs and routes through the repo's central
  async batched client pool.
- ``lm_transport='litellm'``: constructs a plain ``dspy.LM`` for
  debugging.

Both modes are controlled by ``DSPyFamilyConfig`` fields. ``treepo.methods``
must forward that config through ``backend_config['dspy_config']`` to
``DSPyFamily`` *without modification* so the batching transfers to any
grid run launched via ``treepo.methods.run('fit', ...)``.

These tests verify three things end-to-end without requiring a live vLLM
server:

1. **Config propagation.** A ``DSPyFamilyConfig`` with custom batch knobs
   resolves through ``treepo.methods.families.resolve_family('dspy', ...)``
   to a ``DSPyFamily`` whose ``.config`` is the exact same instance.

2. **BatchedDSPyLM construction.** When the family lazy-constructs its
   LM (``_ensure_lm()``), it calls ``BatchedDSPyLM`` with the batch
   knobs from the config. We patch the symbol and inspect kwargs.

3. **Teacher passthrough end-to-end.** The manifesto teacher backend
   (``--backend teacher``) reads pre-computed scores from tree metadata
   and skips real LLM calls. We mirror that path through
   ``treepo.methods.run('fit', ...)`` with a minimal teacher-passthrough
   FamilyRuntime and verify MAE=0 against the metadata teacher scores.

Live-server batching efficiency (request pooling, vLLM concurrency)
remains tested by ``tests/integration/test_vllm_live.py`` under
``TT_RUN_LIVE_TESTS=1`` — that's the integration tier; this file is the
unit-tier proof that nothing in ``treepo.methods`` interposes between the
caller's batch config and the manifesto-grade LLM client.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence
from unittest.mock import patch

import pytest

from treepo._research.ctreepo.contracts import CTreePOFitResult, CTreePOLearningSpec
from treepo._research.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig
import treepo.methods
from treepo.methods.families import resolve_family


def _make_dspy_config(**overrides: Any) -> DSPyFamilyConfig:
    base = dict(
        optimizer="bootstrap_fewshot",
        lm_transport="batch",
        leaf_size_tokens=128,
        lm_context_window_tokens=12000,
        max_completion_tokens=256,
        lm_config={
            "model": "openai/test-model",
            "api_base": "http://test-host:8000/v1",
            "api_key": "EMPTY",
        },
    )
    base.update(overrides)
    return DSPyFamilyConfig(**base)


# --------------------------------------------------------------------------- #
# 1. Config propagation: knobs flow through treepo.methods unchanged.
# --------------------------------------------------------------------------- #


def test_dspy_factory_returns_dspy_family_with_same_config_instance() -> None:
    cfg = _make_dspy_config(batch_size=99, batch_max_concurrent=33)
    family = resolve_family("dspy", {"dspy_config": cfg})
    assert isinstance(family, DSPyFamily)
    assert family.config is cfg, (
        "registry must not clone or wrap the DSPyFamilyConfig — batching "
        "knobs must reach BatchedDSPyLM construction unmodified"
    )
    assert family.config.batch_size == 99
    assert family.config.batch_max_concurrent == 33
    assert family.config.lm_transport == "batch"


def test_dspy_factory_accepts_config_as_mapping_and_preserves_batch_knobs() -> None:
    """Config-loaded-from-TOML/JSON path: dict arrives, factory coerces
    to DSPyFamilyConfig. The batch knobs must survive the coercion.
    """
    cfg_mapping = {
        "optimizer": "bootstrap_fewshot",
        "lm_transport": "batch",
        "batch_size": 17,
        "batch_max_concurrent": 21,
        "batch_timeout": 0.05,
        "batch_routing_policy": "round_robin",
        "leaf_size_tokens": 128,
        "lm_context_window_tokens": 12000,
        "max_completion_tokens": 256,
        "lm_config": {
            "model": "openai/test-model",
            "api_base": "http://test-host:8000/v1",
            "api_key": "EMPTY",
        },
    }
    family = resolve_family("dspy", {"dspy_config": cfg_mapping})
    assert family.config.batch_size == 17
    assert family.config.batch_max_concurrent == 21
    assert family.config.batch_timeout == pytest.approx(0.05)
    assert family.config.batch_routing_policy == "round_robin"


def test_dspy_factory_rejects_missing_config() -> None:
    with pytest.raises(ValueError, match="dspy_config"):
        resolve_family("dspy", {})


# --------------------------------------------------------------------------- #
# 2. BatchedDSPyLM construction: family wires the batch knobs into the
#    client when the LM is lazily built.
# --------------------------------------------------------------------------- #


def test_batched_dspy_lm_constructed_with_passthrough_batch_knobs() -> None:
    """When ``lm_transport='batch'`` and the family needs an LM, it
    invokes ``BatchedDSPyLM`` with the config's batch knobs. We patch
    the symbol so no real server is touched.
    """
    cfg = _make_dspy_config(
        batch_size=42,
        batch_max_concurrent=7,
        batch_timeout=0.123,
        batch_request_timeout=99.0,
        batch_routing_policy="affinity_load_aware",
    )
    family = resolve_family("dspy", {"dspy_config": cfg})

    sentinel_lm = object()
    # BatchedDSPyLM is imported inside _ensure_lm_unlocked, so patch the
    # source module where the binding is resolved at call time.
    with patch(
        "treepo._research.core.dspy_batch_client.BatchedDSPyLM", return_value=sentinel_lm
    ) as batched_ctor:
        with patch(
            "treepo._research.ctreepo.dspy_family.normalize_base_urls",
            return_value=["http://test-host:8000/v1"],
        ):
            lm = family._ensure_lm()

    assert lm is sentinel_lm
    batched_ctor.assert_called_once()
    kwargs = batched_ctor.call_args.kwargs
    # The batching knobs must arrive unmodified.
    assert kwargs["batch_size"] == 42
    assert kwargs["max_concurrent"] == 7
    assert kwargs["batch_timeout"] == pytest.approx(0.123)
    assert kwargs["request_timeout"] == pytest.approx(99.0)
    assert kwargs["routing_policy"] == "affinity_load_aware"
    assert kwargs["api_bases"] == ["http://test-host:8000/v1"]
    assert kwargs["model"] == "openai/test-model"


def test_litellm_transport_falls_back_to_plain_dspy_lm() -> None:
    """``lm_transport='litellm'`` bypasses BatchedDSPyLM. Useful for
    debugging single-request flows; batching is not applied but the
    selection itself is verified.
    """
    cfg = _make_dspy_config(lm_transport="litellm")
    family = resolve_family("dspy", {"dspy_config": cfg})

    sentinel = object()
    # ``dspy`` is imported inside _ensure_lm_unlocked. Patching at the
    # source ensures the lookup resolves to our sentinel.
    with patch("dspy.LM", return_value=sentinel) as plain_ctor:
        with patch(
            "treepo._research.core.dspy_batch_client.BatchedDSPyLM",
            side_effect=AssertionError("BatchedDSPyLM must NOT be called for litellm"),
        ):
            lm = family._ensure_lm()
    assert lm is sentinel
    plain_ctor.assert_called_once()


# --------------------------------------------------------------------------- #
# 3. Teacher passthrough end-to-end through treepo.methods.run("fit", ...).
#    Mirrors `scripts/run_manifesto_fg_real_training_grid.py --backend teacher`:
#    no LLM calls, just metadata reads.
# --------------------------------------------------------------------------- #


class _TeacherPassthroughFamily:
    """Mirrors the manifesto teacher backend: return ``tree.metadata
    ['teacher_score_1_7']`` as the prediction. No LLM. No training.
    Same control-flow shape FNO/DSPy/TRL families follow (FamilyRuntime
    protocol), so the dispatch path through treepo.methods is identical.
    """

    name = "manifesto_teacher_passthrough"

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        return f_init

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        return g_init

    def score_roots_with_f(
        self, *, f: Any, g: Any, trees: Sequence[Any]
    ) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for tree in trees:
            meta = getattr(tree, "metadata", None) or {}
            value = meta.get("teacher_score_1_7")
            out.append(float(value) if value is not None else None)
        return out

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        return None


def _make_manifesto_trees(dim_scores: List[float]) -> List[SimpleNamespace]:
    """Manifesto-style trees: each has a precomputed teacher dimension
    score (Pearson-r-readable scalar). Matches what the teacher backend
    reads off labeled-tree artifacts on disk.
    """
    return [
        SimpleNamespace(
            leaves=[SimpleNamespace(tokens=[])],
            metadata={
                "split": "test",
                "teacher_score_1_7": float(s),
                "teacher_score_native": float(s),
                "expert_score_1_7": float(s),
                "expert_score_native": float(s),
                "expert_target_scale": "raw",
                "expert_score_for_objective": float(s),
            },
        )
        for s in dim_scores
    ]


def test_manifesto_teacher_passthrough_runs_through_treepo_methods(tmp_path: Path) -> None:
    """The teacher backend in scripts/run_manifesto_fg_real_training_grid.py
    doesn't construct an LLM client; it loads labeled-tree artifacts and
    reports teacher metrics. The same path through treepo.methods.run uses
    the FamilyRuntime escape hatch — no special teacher method needed.
    """
    trees = _make_manifesto_trees([3.0, 4.5, 6.0, 2.5])
    family = _TeacherPassthroughFamily()
    result = treepo.methods.run(
        "fit",
        {
            "family": "manifesto_teacher_passthrough",
            "eval_data": trees,
            "backend_config": {
                "family_runtime": family,
                "output_dir": str(tmp_path),
            },
        },
    )
    assert isinstance(result, CTreePOFitResult)
    assert result.status == "success"
    # Teacher = prediction by construction → MAE=0.
    assert result.metrics["internal_f_mae"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["n"] == float(len(trees))


def test_manifesto_teacher_grid_through_treepo_methods(tmp_path: Path) -> None:
    """A grid run over a manifesto-style axis (dimension) using the
    teacher path. Verifies the same shape works at scale through
    ``treepo.methods.run`` — same loop a paper grid script would write.
    """
    dim_scores_by_dim = {
        "economic": [3.0, 4.5, 6.0, 2.5],
        "social": [5.0, 5.5, 4.0, 6.5],
        "immigration": [2.0, 3.0, 4.0, 5.0],
    }
    rows = []
    for dimension, scores in dim_scores_by_dim.items():
        trees = _make_manifesto_trees(scores)
        result = treepo.methods.run(
            "fit",
            {
                "family": "manifesto_teacher_passthrough",
                "eval_data": trees,
                "backend_config": {
                    "family_runtime": _TeacherPassthroughFamily(),
                    "output_dir": str(tmp_path / dimension),
                },
            },
        )
        rows.append(
            {
                "dimension": dimension,
                "status": result.status,
                "internal_f_mae": result.metrics["internal_f_mae"],
                "n": result.metrics["n"],
            }
        )
    assert len(rows) == 3
    assert all(r["status"] == "success" for r in rows)
    # Every dimension cell produces teacher-equals-prediction → MAE=0.
    assert all(r["internal_f_mae"] == pytest.approx(0.0, abs=1e-9) for r in rows)
