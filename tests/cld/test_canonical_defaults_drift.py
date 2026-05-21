"""Pin canonical defaults and TOML loading across every family.

Parametrized so adding a new family means adding one line to the
relevant fixture — not writing a new test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TREEPO_ROOT = Path(__file__).resolve().parents[2]  # treepo project root
if str(TREEPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(TREEPO_ROOT / "src"))

from dataclasses import dataclass, field

from treepo.cld import canonical_defaults as cd

CONFIGS = TREEPO_ROOT / "configs" / "cld"
EXAMPLES = TREEPO_ROOT / "examples" / "cld"


# Module-level helpers for `test_nested_dataclass_field_recurses` —
# typing.get_type_hints needs the referenced class to be importable from
# the module's globalns, which excludes function-local definitions.
@dataclass
class _TestInner:
    x: int = 0


@dataclass
class _TestOuter:
    inner: _TestInner = field(default_factory=_TestInner)


# ===========================================================================
# 1) Cross-family constants — re-exports of upstream, not mirrors
# ===========================================================================


def test_cross_family_constants_are_upstream_objects() -> None:
    """`canonical_defaults` re-exports upstream constants; verified by `is`-identity.

    These re-exports replaced the previous mirror tests — drift is now
    impossible by construction (the constant IS the upstream value).
    """
    from treepo._research.core import batch_transport as bt
    from treepo._research.tasks.manifesto import pipeline_config as pc
    from treepo._research.training.optimization.gepa import GEPA_STRONG_DEFAULT_KWARGS

    # Re-exported batch transport.
    from treepo.cld.canonical_defaults import (
        DEFAULT_BATCH_MAX_CONCURRENT, DEFAULT_BATCH_SIZE,
        DEFAULT_BATCH_TIMEOUT_SECONDS, DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
        DEFAULT_BATCH_ROUTING_POLICY,
    )
    assert DEFAULT_BATCH_SIZE is bt.DEFAULT_BATCH_SIZE
    assert DEFAULT_BATCH_MAX_CONCURRENT is bt.DEFAULT_BATCH_MAX_CONCURRENT
    assert DEFAULT_BATCH_TIMEOUT_SECONDS is bt.DEFAULT_BATCH_TIMEOUT_SECONDS
    assert DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS is bt.DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS
    assert DEFAULT_BATCH_ROUTING_POLICY is bt.DEFAULT_BATCH_ROUTING_POLICY

    # Re-exported pipeline invariants.
    assert cd.CONCAT_RATIO is pc.CONCAT_RATIO
    assert cd.DEFAULT_TARGET_RATIO is pc.DEFAULT_TARGET_RATIO
    assert cd.DEFAULT_SCORER_MAX_TOKENS is pc.DEFAULT_SCORER_MAX_TOKENS
    assert cd.DEFAULT_PROMPT_OVERHEAD_TOKENS is pc.DEFAULT_PROMPT_OVERHEAD_TOKENS
    assert cd.DEFAULT_MANIFESTO_WORKERS is pc.DEFAULT_MANIFESTO_WORKERS
    assert cd.DEFAULT_SUMMARY_WORKERS is pc.DEFAULT_SUMMARY_WORKERS
    assert cd.DEFAULT_SCORING_WORKERS is pc.DEFAULT_SCORING_WORKERS

    # Re-exported GEPA strong defaults.
    assert cd.GEPA_STRONG_DEFAULTS is GEPA_STRONG_DEFAULT_KWARGS


def test_dspy_family_batch_defaults_match_canonical() -> None:
    """DSPyFamilyConfig must ship the canonical batch values out of the box."""
    from treepo._research.ctreepo.dspy_family import DSPyFamilyConfig

    dc = DSPyFamilyConfig()
    assert dc.batch_size == cd.BATCH_DEFAULTS["batch_size"]
    assert dc.batch_max_concurrent == cd.BATCH_DEFAULTS["batch_max_concurrent"]
    assert dc.batch_timeout == cd.BATCH_DEFAULTS["batch_timeout"]
    assert dc.batch_request_timeout == cd.BATCH_DEFAULTS["batch_request_timeout"]
    assert dc.batch_routing_policy == cd.BATCH_DEFAULTS["batch_routing_policy"]


# ===========================================================================
# 2) Every TOML loads cleanly into the right dataclass
# ===========================================================================


def _toml_load_cases():
    """(toml_filename, dataclass, section, expected_field_subset)."""
    from treepo._research.ctreepo.dspy_family import DSPyFamilyConfig
    from treepo._research.ctreepo.fno_family import FNOFamilyConfig
    from treepo._research.ctreepo.sim.core.lda_tree_recovery import LDATreeRecoveryConfig
    from treepo._research.tree.markov_changepoint_honesty_simulation import MarkovChangepointConfig

    return [
        # Manifesto TOML now leans almost entirely on canonical defaults;
        # only overrides are include_identity_targets=True and the
        # caller-side eval-pool pre-filter max_input_chars.
        ("manifesto_fg_compile.toml", DSPyFamilyConfig, "family",
         {"optimizer": "gepa", "budget": "heavy", "leaf_size_tokens": 512,
          "lm_context_window_tokens": 32000, "max_completion_tokens": 1024,
          "include_identity_targets": True, "max_input_chars": 96000}),
        ("manifesto_fg_compile.toml", cd.LmSection, "lm",
         {"model": "nvidia/Gemma-4-31B-IT-NVFP4"}),
        ("fno_smoke.toml", FNOFamilyConfig, None,
         {"hidden_channels": 8, "n_modes": 4, "n_layers": 1,
          "epochs_per_iteration": 1, "leaf_size_tokens": 64}),
        # Markov probe TOML is a flat dict (no dataclass mirror) —
        # validated separately via test_markov_probe_toml_keys_are_allowed.
        ("markov_oracle.toml", MarkovChangepointConfig, "dgp",
         {"n_regimes": 4, "vocab_size": 96}),
        ("hll_sketch.toml", cd.HllSketchConfig, None,
         {"precision": 14, "schedule": "balanced", "n_trees": 6}),
        ("lda_oracle.toml", cd.LdaOracleConfig, None,
         {"oracle_name": "leaf_local_mixture_target", "n_trees": 8}),
        ("lda_recovery_smoke.toml", LDATreeRecoveryConfig, None,
         {"n_topics": 4, "vocab_size": 64, "train_docs": 4, "test_docs": 16}),
    ]


@pytest.mark.parametrize("toml,cls,section,expected", _toml_load_cases(),
                         ids=lambda x: getattr(x, "__name__", str(x))[:40])
def test_toml_loads_with_expected_values(toml, cls, section, expected) -> None:
    inst = cd.load_dataclass(CONFIGS / toml, cls, section=section)
    for k, v in expected.items():
        assert getattr(inst, k) == v, f"{cls.__name__}.{k}: expected {v!r}, got {getattr(inst, k)!r}"


# ===========================================================================
# 3) Every upstream dataclass round-trips through load_dataclass(None, cls)
# ===========================================================================


@pytest.mark.parametrize("cls_path", [
    "treepo._research.ctreepo.fno_family.FNOFamilyConfig",
    "treepo._research.ctreepo.sim.core.lda_tree_recovery.LDATreeRecoveryConfig",
    "treepo._research.tree.markov_changepoint_honesty_simulation.MarkovChangepointConfig",
    "treepo.cld.canonical_defaults.HllSketchConfig",
    "treepo.cld.canonical_defaults.LdaOracleConfig",
    "treepo.cld.canonical_defaults.LmSection",
])
def test_load_none_returns_in_code_defaults(cls_path: str) -> None:
    """`load_dataclass(None, cls)` must equal `cls()` for every dataclass we expose."""
    import importlib

    mod, _, name = cls_path.rpartition(".")
    cls = getattr(importlib.import_module(mod), name)
    loaded = cd.load_dataclass(None, cls)
    default = cls()
    assert loaded == default, f"{cls_path}: load(None) != cls()"


# ===========================================================================
# 4) Generic loader behavior
# ===========================================================================


def test_unknown_top_level_key_raises() -> None:
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("bogus_field = 1\n")
        path = f.name
    with pytest.raises(ValueError, match="unknown field"):
        cd.load_dataclass(path, cd.HllSketchConfig)


def test_unknown_section_key_raises() -> None:
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("[family]\nthis_field_does_not_exist = 1\n")
        path = f.name
    from treepo._research.ctreepo.dspy_family import DSPyFamilyConfig

    with pytest.raises(ValueError, match="unknown field"):
        cd.load_dataclass(path, DSPyFamilyConfig, section="family")


def test_overrides_skip_none() -> None:
    cfg = cd.load_dataclass(None, cd.HllSketchConfig,
                            overrides={"precision": None, "schedule": "left_to_right"})
    assert cfg.precision == 14  # default preserved
    assert cfg.schedule == "left_to_right"  # override applied


def test_overrides_reject_unknown_path() -> None:
    with pytest.raises(ValueError, match="unknown"):
        cd.load_dataclass(None, cd.HllSketchConfig, overrides={"nope": 1})


def test_nested_dataclass_field_recurses() -> None:
    """When a wrapper has a dataclass-typed field, the loader recurses into a nested table.

    Uses module-level dataclasses below (`_TestInner`, `_TestOuter`) because
    ``typing.get_type_hints`` can't resolve forward refs declared in a function
    scope. The recursion itself works fine for real configs (see the manifesto
    TOML test loading DSPyFamilyConfig as a section).
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("[inner]\nx = 7\n")
        path = f.name
    out = cd.load_dataclass(path, _TestOuter)
    assert out.inner.x == 7


def test_dspy_family_config_ships_strong_gepa_kwargs() -> None:
    """DSPyFamilyConfig().gepa_kwargs IS the strong defaults — no monkey-patch needed."""
    from treepo._research.ctreepo.dspy_family import DSPyFamilyConfig

    cfg = DSPyFamilyConfig()
    # Every kwarg from upstream's canonical dict is on the field default.
    # Identity-checked via dict equality (the factory returns a fresh dict
    # per instance, but the values come from the same source).
    assert cfg.gepa_kwargs == cd.GEPA_STRONG_DEFAULTS
    # num_threads lives as its own dataclass field (paper canonical = 128).
    assert cfg.num_threads >= 64


def test_markov_probe_toml_keys_are_allowed() -> None:
    """Every key in markov_probe.toml is in `allowed_config_keys("probe")`."""
    import tomllib

    import treepo.cld

    payload = tomllib.loads((CONFIGS / "markov_probe.toml").read_text())
    allowed = treepo.cld.allowed_config_keys("probe")
    unknown = sorted(k for k in payload if k not in allowed)
    assert not unknown, (
        f"markov_probe.toml has keys not in allowed_config_keys('probe'): {unknown}. "
        f"Either fix the TOML or expand allowed_config_keys."
    )


def test_probe_allowed_keys_cover_probe_argparse() -> None:
    """`allowed_config_keys("probe")` is a superset of probe argparse flags.

    Prevents the case where the probe gains a new --flag but treepo.cld
    silently blocks it. Auto-extracts argparse flags by regex.
    """
    import re

    import treepo.cld

    probe_src = (TREEPO_ROOT / "scripts" / "probe_clean_unified_no.py").read_text()
    argparse_flags = set(re.findall(
        r'parser\.add_argument\(\s*"(--[a-z-][\w-]*)"', probe_src
    ))
    probe_keys = {f.lstrip("-").replace("-", "_") for f in argparse_flags}
    allowed = treepo.cld.allowed_config_keys("probe")
    # Allowed must cover every probe flag (we always add `output_root`/`timeout`).
    missing = sorted(probe_keys - allowed)
    assert not missing, (
        f"probe accepts these flags but allowed_config_keys('probe') doesn't list them: "
        f"{missing}. Add them to register_method('probe', ...) in methods.py."
    )


def test_dspy_family_config_defaults_are_paper_canonical() -> None:
    """The repo defaults themselves now match paper canonical (no override needed)."""
    from treepo._research.ctreepo.dspy_family import DSPyFamilyConfig

    cfg = DSPyFamilyConfig()
    assert cfg.optimizer == "gepa"
    assert cfg.budget == "heavy"
    assert cfg.lm_context_window_tokens == 32000
    assert cfg.max_completion_tokens == 1024
    # The two-leaf concat invariant (max_completion_tokens >= 2 × leaf_size_tokens).
    assert cfg.max_completion_tokens >= 2 * cfg.leaf_size_tokens


# ===========================================================================
# 5) Every example imports + has main()
# ===========================================================================


@pytest.mark.parametrize("example", sorted(
    p.stem for p in EXAMPLES.glob("run_*.py")
))
def test_example_imports(example: str) -> None:
    import importlib.util

    path = EXAMPLES / f"{example}.py"
    spec = importlib.util.spec_from_file_location(f"example_{example}", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "main"), f"{example} missing main()"
