from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class _StrictEnum(str, Enum):
    @classmethod
    def parse(cls, value: str):
        normalized = str(value or "").strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(member.value for member in cls)
        raise ValueError(f"unsupported {cls.__name__}={value!r}; expected one of: {valid}")


class Profile(_StrictEnum):
    STANDARD = "standard"
    ROOT_ONLY = "root_only"
    FNO_CANARY = "fno_canary"
    DUPLICATE_LOCAL_LABEL_ONE_LEAF = "duplicate_local_label_one_leaf"


class SupervisionPolicy(_StrictEnum):
    ROOT_ONLY = "root_only"
    LEAF_MASS_EQ = "leaf_mass_eq"


class ComparatorPolicy(_StrictEnum):
    LOCKED_REFS = "locked_refs"


class MarkovScope(_StrictEnum):
    RECOVERABLE_V5_T128 = "recoverable_v5_t128"
    R12_P079 = "r12_p079"


_LEGACY_FORBIDDEN_TOKENS = frozenset(
    {
        "legacy",
        "v2",
        "comparison_grid_v3",
        "standard_tree",
        "half_c1",
        "fno_parity_canary",
        "unified_g_full_local_laws_v1",
        "unified_g_fno_parity_canary_v1",
        "unified_g_multi_leaf_root_only_v1",
        "root_only_matched",
        "root_only_replay",
        "root_only_opt_fix",
        "root_only_capacity_fix",
    }
)


def reject_legacy_value(value: str, *, field_name: str) -> None:
    normalized = str(value or "").strip().lower()
    if normalized in _LEGACY_FORBIDDEN_TOKENS:
        raise ValueError(
            f"{field_name}={value!r} is legacy-only in this repo. "
            "Use the Unified-G V1 lane profile/supervision policy surface instead."
        )


@dataclass(frozen=True)
class MarkovRunSpec:
    scope: MarkovScope
    train_docs: int
    root_share: int
    leaf_tokens: int
    supervision_policy: SupervisionPolicy
    profile: Profile
    seed: int = 0
    comparator_policy: ComparatorPolicy = ComparatorPolicy.LOCKED_REFS

    def __post_init__(self) -> None:
        if int(self.train_docs) <= 0:
            raise ValueError("train_docs must be positive")
        if int(self.root_share) not in {100, 90, 80, 70, 60, 50, 40, 30, 20, 10}:
            raise ValueError("root_share must be one of 100,90,80,70,60,50,40,30,20,10")
        if int(self.leaf_tokens) not in {128, 64, 32, 16, 8}:
            raise ValueError("leaf_tokens must be one of 128,64,32,16,8")
        if self.profile == Profile.FNO_CANARY and int(self.leaf_tokens) != 128:
            raise ValueError("fno_canary profile only supports leaf_tokens=128")
        if (
            self.profile == Profile.DUPLICATE_LOCAL_LABEL_ONE_LEAF
            and int(self.leaf_tokens) != 128
        ):
            raise ValueError(
                "duplicate_local_label_one_leaf profile only supports leaf_tokens=128"
            )
        if (
            self.profile == Profile.FNO_CANARY
            and self.supervision_policy != SupervisionPolicy.ROOT_ONLY
        ):
            raise ValueError("fno_canary only supports supervision_policy='root_only'")
        if (
            self.profile == Profile.DUPLICATE_LOCAL_LABEL_ONE_LEAF
            and self.supervision_policy != SupervisionPolicy.ROOT_ONLY
        ):
            raise ValueError(
                "duplicate_local_label_one_leaf only supports supervision_policy='root_only'"
            )

    @property
    def run_key(self) -> str:
        return (
            f"{self.scope.value}__train{int(self.train_docs)}__r{int(self.root_share)}"
            f"__leaf{int(self.leaf_tokens)}__{self.supervision_policy.value}"
            f"__{self.profile.value}__s{int(self.seed)}"
        )
