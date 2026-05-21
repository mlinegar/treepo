from __future__ import annotations

from dataclasses import asdict

from treepo._research.unified_g_v1.core.contracts import MarkovRunSpec, Profile
from treepo._research.unified_g_v1.markov.program import (
    profile_overrides,
    supervision_budget_fields,
)

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig


def apply_profile_overrides(
    config: OPSCountConfig,
    *,
    profile: Profile,
) -> OPSCountConfig:
    merged = {**asdict(config), **dict(profile_overrides(profile))}
    return OPSCountConfig(**merged)


def apply_supervision_policy(
    config: OPSCountConfig,
    *,
    spec: MarkovRunSpec,
) -> OPSCountConfig:
    budget_fields = supervision_budget_fields(spec)
    merged = {**asdict(config), **budget_fields}
    return OPSCountConfig(**merged)
