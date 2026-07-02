"""Composable state-statistic protocol.

``ComposableStatistic`` is the small public state surface behind treepo's
``g``/``f`` split. Exact sketches and learned operators can expose it without
changing the public ``fit`` or ``PreferenceDataset`` APIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from treepo.local_law import LocalLawAuditRow


@dataclass(frozen=True)
class StatisticInfo:
    """Compact metadata for a composable statistic."""

    name: str
    state_kind: str
    exact: bool
    supports_local_laws: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata or {})
        return payload


@runtime_checkable
class ComposableStatistic(Protocol):
    """Minimal protocol for locally composable state and readout."""

    info: StatisticInfo

    def encode_leaf(self, leaf: Any) -> Any:
        ...

    def merge(self, left: Any, right: Any) -> Any:
        ...

    def readout(self, state: Any, query: Any = None) -> Any:
        ...

    def local_law_rows(
        self,
        units: Sequence[Any],
        *,
        query: Any = None,
        oracle: Any = None,
    ) -> Sequence[LocalLawAuditRow]:
        ...


def family_statistic(family: Any, *, f: Any = None, g: Any = None) -> ComposableStatistic | None:
    """Return a family's composable statistic when it exposes one."""

    hook = getattr(family, "as_statistic", None)
    if not callable(hook):
        return None
    statistic = hook(f=f, g=g)
    return statistic if statistic is not None else None


__all__ = ["ComposableStatistic", "StatisticInfo", "family_statistic"]
