"""Compatibility facade for built-in neural-operator families."""

from __future__ import annotations

from treepo.methods._neural_operator_core import (
    FNOFamily,
    FNOFamilyConfig,
    NeuralOperatorFamily,
    NeuralOperatorFamilyConfig,
    build_fno_family,
    build_neural_operator_family,
)

__all__ = [
    "FNOFamily",
    "FNOFamilyConfig",
    "NeuralOperatorFamily",
    "NeuralOperatorFamilyConfig",
    "build_fno_family",
    "build_neural_operator_family",
]
