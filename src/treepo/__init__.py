"""treepo: composable tree-operator fitting and local-law certificates."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("treepo")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.2.0"

_LAZY_EXPORTS = {
    "align_model_artifacts": ("treepo.methods.artifact_alignment", "align_model_artifacts"),
    "fit": ("treepo.learning", "fit"),
    "OracleTargetSpec": ("treepo.forest", "OracleTargetSpec"),
    "l1_oracle_metric_schema": ("treepo.forest", "l1_oracle_metric_schema"),
    "G_MODE_FIXED": ("treepo.methods.contracts", "G_MODE_FIXED"),
    "G_MODE_IDENTITY": ("treepo.methods.contracts", "G_MODE_IDENTITY"),
    "G_MODE_LEARNED": ("treepo.methods.contracts", "G_MODE_LEARNED"),
    "G_MODES": ("treepo.methods.contracts", "G_MODES"),
    "normalize_g_mode": ("treepo.methods.contracts", "normalize_g_mode"),
    "oracle_vector_l1": ("treepo.forest", "oracle_vector_l1"),
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
    "align_model_artifacts",
    "Candidate",
    "ComposableStatistic",
    "PreferenceDataset",
    "G_MODE_FIXED",
    "G_MODE_IDENTITY",
    "G_MODE_LEARNED",
    "G_MODES",
    "PreferenceRecord",
    "OracleTargetSpec",
    "TaskState",
    "normalize_g_mode",
    "TreeNode",
    "TreeRecord",
    "family_statistic",
    "fit",
    "l1_oracle_metric_schema",
    "oracle_vector_l1",
    "state_from_value",
    "state_to_dict",
    "write_tree_visualization_html",
]
