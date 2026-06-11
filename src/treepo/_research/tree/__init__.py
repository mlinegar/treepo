"""Lazy namespace for research tree modules.

The historical initializer eagerly imported nearly every tree, audit,
operator, and training helper. That made lightweight package paths such as
`treepo.methods` pull in torch/DSPy/pandas through unrelated research modules.
Keep this namespace import-light and resolve legacy top-level attributes only
when callers ask for them.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SEARCH_MODULES = (
    "treepo._research.tree.builder",
    "treepo._research.tree.auditor",
    "treepo._research.tree.ipw",
    "treepo._research.tree.ipw_simulation",
    "treepo._research.tree.ipw_toy_problems",
    "treepo._research.tree.mergeable_ablation",
    "treepo._research.tree.verification",
    "treepo._research.tree.compositional_operator",
    "treepo._research.tree.theorem_backing",
    "treepo._research.tree.contract_runner",
    "treepo._research.tree.compositional_learning",
    "treepo._research.tree.core_model",
    "treepo._research.tree.tree_model_v2",
    "treepo._research.tree.neural_operator",
    "treepo._research.tree.labeled",
    "treepo._research.tree.async_operator",
    "treepo._research.tree.state_tree",
    "treepo._research.tree.state_tree_runner",
    "treepo._research.tree.state_tree_verifiers",
    "treepo._research.tree.treepo_stack",
    "treepo._research.tree.treepo_supervision",
)


def __getattr__(name: str) -> Any:
    for module_name in _SEARCH_MODULES:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


__all__ = [
    "TreeBuilder",
    "BuildConfig",
    "BuildResult",
    "Auditor",
    "AuditConfig",
    "AuditReport",
    "NodeType",
    "TreeSample",
    "LabeledNode",
    "LabeledTree",
    "LabeledDataset",
    "StateNode",
    "StateTree",
    "state_tree_to_text_tree",
    "run_fixed_binary_state_tree",
    "arun_fixed_binary_state_tree",
    "LawVerifier",
    "TreePOStack",
    "TreePOSupervisionSpec",
]
