"""Unit-level preference data and trainer export views.

``PreferenceDataset`` is the package data boundary for labels, scored
candidates, ranked candidates, and pairwise preferences. Pairwise DPO rows are
one projection of this data, not the storage model. The implementation is
split by responsibility:

* ``_preference_normalize``: row parsing, scalar/JSON coercion (data-prep).
* ``_preference_views``: supervised/DPO/reward/GRPO projections (data-prep).
* ``_preference_dataset``: the ``Candidate``/``PreferenceRecord``/
  ``PreferenceDataset`` data model.
* ``_preference_io``: dataset IO, export, view previews, and trainer-adapter
  export fan-out.
* ``_preference_tree``: ``TreeRecord``-derived units.
"""

from __future__ import annotations

from treepo.methods._preference_dataset import (
    Candidate,
    PreferenceDataset,
    PreferenceFormat,
    PreferenceRecord,
    PreferenceTarget,
)
from treepo.methods._preference_io import (
    export_adapter_views,
    export_preference_records,
    normalize_preference_data,
    summarize_preference_views,
)
from treepo.methods._preference_tree import (
    make_unit_id,
    preference_units_from_trees,
)

__all__ = [
    "Candidate",
    "PreferenceDataset",
    "PreferenceFormat",
    "PreferenceRecord",
    "PreferenceTarget",
    "export_adapter_views",
    "export_preference_records",
    "make_unit_id",
    "normalize_preference_data",
    "preference_units_from_trees",
    "summarize_preference_views",
]
