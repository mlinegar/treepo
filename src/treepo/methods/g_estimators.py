"""Compatibility aliases for the unified-g estimator registry.

Prefer :mod:`treepo.methods.estimators` and the ``estimator`` config key in new
code. This module remains for older configs and tests that used
``g_estimator`` while the public axis was being introduced.
"""

from __future__ import annotations

from treepo.methods.estimators import (
    EstimatorDescriptor as GEstimatorDescriptor,
    EstimatorFactory as GEstimatorFactory,
    EstimatorSpec as GEstimatorSpec,
    list_estimators as list_g_estimators,
    register_estimator as register_g_estimator,
    resolve_estimator as resolve_g_estimator,
)

__all__ = [
    "GEstimatorDescriptor",
    "GEstimatorFactory",
    "GEstimatorSpec",
    "list_g_estimators",
    "register_g_estimator",
    "resolve_g_estimator",
]
