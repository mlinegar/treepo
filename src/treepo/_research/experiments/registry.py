from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

from treepo._research.experiments.contracts import ExperimentSpec


class MethodAdapter(Protocol):
    adapter_id: str
    aliases: tuple[str, ...]

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec: ...

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]: ...


class ReportProfile(Protocol):
    profile_id: str
    aliases: tuple[str, ...]

    def render(
        self,
        *,
        output_root: Path,
        output_dir: Path | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


@dataclass
class MethodAdapterRegistry:
    _adapters: Dict[str, MethodAdapter]

    def register(self, adapter: MethodAdapter) -> MethodAdapter:
        keys = {str(adapter.adapter_id)}
        keys.update(str(alias) for alias in tuple(getattr(adapter, "aliases", ()) or ()))
        for key in keys:
            self._adapters[str(key)] = adapter
        return adapter

    def resolve(self, key: str) -> MethodAdapter:
        adapter = self._adapters.get(str(key))
        if adapter is None:
            raise KeyError(f"unknown method adapter: {key!r}")
        return adapter

    def available(self) -> tuple[str, ...]:
        canonical = sorted(
            {
                str(getattr(adapter, "adapter_id", key))
                for key, adapter in self._adapters.items()
            }
        )
        return tuple(canonical)


@dataclass
class ReportProfileRegistry:
    _profiles: Dict[str, ReportProfile]

    def register(self, profile: ReportProfile) -> ReportProfile:
        keys = {str(profile.profile_id)}
        keys.update(str(alias) for alias in tuple(getattr(profile, "aliases", ()) or ()))
        for key in keys:
            self._profiles[str(key)] = profile
        return profile

    def resolve(self, key: str) -> ReportProfile:
        profile = self._profiles.get(str(key))
        if profile is None:
            raise KeyError(f"unknown report profile: {key!r}")
        return profile

    def available(self) -> tuple[str, ...]:
        canonical = sorted(
            {
                str(getattr(profile, "profile_id", key))
                for key, profile in self._profiles.items()
            }
        )
        return tuple(canonical)


METHOD_ADAPTERS = MethodAdapterRegistry(_adapters={})
REPORT_PROFILES = ReportProfileRegistry(_profiles={})
_DEFAULT_ADAPTERS_REGISTERED = False


def ensure_default_method_adapters() -> None:
    """Load built-in adapter registrations without exporting them publicly."""

    global _DEFAULT_ADAPTERS_REGISTERED
    if _DEFAULT_ADAPTERS_REGISTERED:
        return
    importlib.import_module("src.experiments.adapters")
    _DEFAULT_ADAPTERS_REGISTERED = True


def register_method_adapter(adapter: MethodAdapter | type[Any]) -> MethodAdapter | type[Any]:
    if isinstance(adapter, type):
        METHOD_ADAPTERS.register(adapter())
        return adapter
    return METHOD_ADAPTERS.register(adapter)


def register_report_profile(profile: ReportProfile | type[Any]) -> ReportProfile | type[Any]:
    if isinstance(profile, type):
        REPORT_PROFILES.register(profile())
        return profile
    return REPORT_PROFILES.register(profile)


def method_adapter_factory(
    factory: Callable[[], MethodAdapter],
) -> Callable[[], MethodAdapter]:
    adapter = factory()
    register_method_adapter(adapter)
    return factory


def report_profile_factory(
    factory: Callable[[], ReportProfile],
) -> Callable[[], ReportProfile]:
    profile = factory()
    register_report_profile(profile)
    return factory
