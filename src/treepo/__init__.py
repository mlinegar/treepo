"""treepo: composable tree-operator fitting and local-law certificates."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from treepo.local_law import (
    InfluenceWeightedAuditOverlap,
    LawKind,
    LocalLawAuditRow,
    LocalLawObjectiveSummary,
    compute_influence_weighted_overlap,
    corrected_local_law_loss,
    local_law_objective_summary,
)

try:
    __version__ = version("treepo")
except (PackageNotFoundError, TypeError, KeyError):  # pragma: no cover
    __version__ = "0.1.0"

_LAZY_EXPORTS = {
    "FitConfig": ("treepo.learning", "FitConfig"),
    "FitResult": ("treepo.learning", "FitResult"),
    "fit": ("treepo.learning", "fit"),
    "Candidate": ("treepo.methods.preference", "Candidate"),
    "PreferenceDataset": ("treepo.methods.preference", "PreferenceDataset"),
    "PreferenceRecord": ("treepo.methods.preference", "PreferenceRecord"),
    "ComposableStatistic": ("treepo.statistic", "ComposableStatistic"),
    "StatisticInfo": ("treepo.statistic", "StatisticInfo"),
    "family_statistic": ("treepo.statistic", "family_statistic"),
    "TaskState": ("treepo.state", "TaskState"),
    "TreeUnitRef": ("treepo.state", "TreeUnitRef"),
    "TreeNode": ("treepo.tree", "TreeNode"),
    "TreeRecord": ("treepo.tree", "TreeRecord"),
    "state_from_value": ("treepo.state", "state_from_value"),
    "state_to_dict": ("treepo.state", "state_to_dict"),
    "unit_ref_from": ("treepo.state", "unit_ref_from"),
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
    "FitConfig",
    "FitResult",
    "PreferenceDataset",
    "PreferenceRecord",
    "StatisticInfo",
    "TaskState",
    "TreeNode",
    "TreeRecord",
    "TreeUnitRef",
    "family_statistic",
    "fit",
    "state_from_value",
    "state_to_dict",
    "unit_ref_from",
    "LawKind",
    "LocalLawAuditRow",
    "LocalLawObjectiveSummary",
    "InfluenceWeightedAuditOverlap",
    "compute_influence_weighted_overlap",
    "corrected_local_law_loss",
    "local_law_objective_summary",
]
