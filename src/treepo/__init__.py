"""treepo: composable tree-operator fitting and local-law certificates."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("treepo")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.2.0"

_LAZY_EXPORTS = {
    "fit": ("treepo.learning", "fit"),
    "Candidate": ("treepo.methods.preference", "Candidate"),
    "PreferenceDataset": ("treepo.methods.preference", "PreferenceDataset"),
    "PreferenceRecord": ("treepo.methods.preference", "PreferenceRecord"),
    "ComposableStatistic": ("treepo.statistic", "ComposableStatistic"),
    "family_statistic": ("treepo.statistic", "family_statistic"),
    "TaskState": ("treepo.state", "TaskState"),
    "TreeNode": ("treepo.tree", "TreeNode"),
    "TreeRecord": ("treepo.tree", "TreeRecord"),
    "state_from_value": ("treepo.state", "state_from_value"),
    "state_to_dict": ("treepo.state", "state_to_dict"),
    "write_tree_visualization_html": ("treepo.viz", "write_tree_visualization_html"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "Candidate",
    "ComposableStatistic",
    "PreferenceDataset",
    "PreferenceRecord",
    "TaskState",
    "TreeNode",
    "TreeRecord",
    "family_statistic",
    "fit",
    "state_from_value",
    "state_to_dict",
    "write_tree_visualization_html",
]
