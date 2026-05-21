"""Core TreePO primitives with no heavyweight optional dependencies."""

from treepo.core.experiment import ExperimentContext, NormalizedOutput, SamplingPlan
from treepo.core.refs import BenchmarkRef, MethodRef, ResultRow
from treepo.core.roles import (
    ROLE_EMBEDDER,
    ROLE_ORACLE,
    ROLE_SCORER,
    ROLE_STATE_MODEL,
    ROLE_SUMMARIZER,
    RoleRef,
    role_ref,
    roles_metadata,
)

__all__ = [
    "BenchmarkRef",
    "ExperimentContext",
    "MethodRef",
    "NormalizedOutput",
    "ROLE_EMBEDDER",
    "ROLE_ORACLE",
    "ROLE_SCORER",
    "ROLE_STATE_MODEL",
    "ROLE_SUMMARIZER",
    "ResultRow",
    "RoleRef",
    "SamplingPlan",
    "role_ref",
    "roles_metadata",
]
