"""Pin the vLLM/SGLang first-class-backend parity guarantee.

Adding a new local-chat backend means adding to the parametrize list below.
If a feature drifts (e.g. only one backend gains a venv-path helper, or its
launch script goes missing), this test fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from treepo._research.core.engines import (
    LOCAL_CHAT_MANAGED_ENGINES,
    EngineRegistry,
    EngineSurface,
    EngineType,
)
from treepo.paths import sglang_venv_path, vllm_venv_path


# Engines that must satisfy the local-chat first-class contract.
_PARITY_ENGINES = (EngineType.VLLM, EngineType.SGLANG)


@pytest.mark.parametrize("engine", _PARITY_ENGINES)
def test_engine_is_in_local_chat_managed_set(engine: EngineType) -> None:
    """Both engines must be members of LOCAL_CHAT_MANAGED_ENGINES."""
    assert engine in LOCAL_CHAT_MANAGED_ENGINES


@pytest.mark.parametrize("engine", _PARITY_ENGINES)
def test_engine_spec_satisfies_first_class_contract(engine: EngineType) -> None:
    """Load-bearing fields that define a 'first-class local backend'."""
    spec = EngineRegistry.resolve(engine)
    assert spec.launchable, f"{engine.value}: must be launchable"
    assert spec.openai_compatible, f"{engine.value}: must be OpenAI-compatible"
    assert spec.supports_profiles, f"{engine.value}: must support model profiles"
    assert spec.default_port is not None, f"{engine.value}: must have a default port"
    assert spec.default_host, f"{engine.value}: must have a default host"
    assert EngineSurface.CHAT_OPENAI in spec.surfaces, (
        f"{engine.value}: must expose the CHAT_OPENAI surface"
    )


@pytest.mark.parametrize("engine", _PARITY_ENGINES)
def test_engine_launch_script_exists_on_disk(engine: EngineType) -> None:
    """The script the registry points at must actually be present in the repo."""
    spec = EngineRegistry.resolve(engine)
    assert spec.launch_script, f"{engine.value}: launch_script not set"
    script = Path(spec.launch_script)
    assert script.exists(), (
        f"{engine.value}: launch script missing at {script}. "
        f"Did EngineRegistry._repo_root() resolve incorrectly, "
        f"or was the script removed from scripts/?"
    )
    assert script.is_file()


@pytest.mark.parametrize("engine,resolver", [
    (EngineType.VLLM, vllm_venv_path),
    (EngineType.SGLANG, sglang_venv_path),
])
def test_engine_has_venv_path_helper(engine: EngineType, resolver) -> None:
    """treepo.paths must expose a venv-path helper for the engine."""
    path = resolver()
    assert isinstance(path, str) and path, (
        f"{engine.value}: venv-path resolver returned empty"
    )


@pytest.mark.parametrize("engine,env_var", [
    (EngineType.VLLM, "TREEPO_VLLM_VENV"),
    (EngineType.SGLANG, "TREEPO_SGLANG_VENV"),
])
def test_engine_venv_env_var_overrides_default(
    engine: EngineType, env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting the env var must flow through to the resolver."""
    from treepo.paths import sglang_venv_path, vllm_venv_path

    monkeypatch.setenv(env_var, "/example/some/venv")
    resolver = vllm_venv_path if engine is EngineType.VLLM else sglang_venv_path
    assert resolver() == "/example/some/venv"


def test_orchestrator_config_consumes_venv_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """OrchestratorConfig field defaults must read from the env vars at instantiation."""
    monkeypatch.setenv("TREEPO_VLLM_VENV", "/example/vllm-env")
    monkeypatch.setenv("TREEPO_SGLANG_VENV", "/example/sglang-env")
    # Re-import via dataclass introspection to call the factories with current env.
    import dataclasses

    from treepo._research.core.gpu_orchestrator import OrchestratorConfig

    fields_by_name = {f.name: f for f in dataclasses.fields(OrchestratorConfig)}
    assert fields_by_name["venv_path"].default_factory() == "/example/vllm-env"
    assert fields_by_name["sglang_venv_path"].default_factory() == "/example/sglang-env"


def test_engines_normalize_aliases_consistently() -> None:
    """Cross-check: at minimum, both names must round-trip through EngineType.normalize."""
    assert EngineType.normalize("vllm") is EngineType.VLLM
    assert EngineType.normalize("VLLM") is EngineType.VLLM
    assert EngineType.normalize("sglang") is EngineType.SGLANG
    assert EngineType.normalize("SGLang") is EngineType.SGLANG
