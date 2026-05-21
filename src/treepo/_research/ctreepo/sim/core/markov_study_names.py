"""Stable public aliases for Markov study package and ablation names."""

from __future__ import annotations

from typing import Mapping, Sequence


def _pct_alias_text(value: float) -> str:
    text = f"{float(value):.1f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


SUPERVISION_RECOVERY_PACKAGE_ALIASES: dict[str, str] = {
    "root40": "full40",
    "root10": "full10",
    "root20": "full20",
    "root30": "full30",
    "root50": "full50",
    "root60": "full60",
    "root70": "full70",
    "root80": "full80",
    "root90": "full90",
    "root100": "full100",
    "root0_extra_leaffull100_internalcount100": "full0_leaf_full100_internal_count100",
    "root10_extra_leafcount100": "full10_leaf_count100",
    "root10_extra_leaffull100": "full10_leaf_full100",
    "root10_extra_leaffull100_internalcount100_d1": "full10_leaf_full100_internal_depth1_count100",
    "root10_extra_leaffull100_internalcount100_d2": "full10_leaf_full100_internal_depth2_count100",
    "root100_extra_leaf05_internal10": "r100_superset_leaf05_internal10p0",
}

for _root_share, _rates in {
    10: (10, 20, 50, 100),
    20: (10, 20, 50, 100),
}.items():
    for _rate in _rates:
        SUPERVISION_RECOVERY_PACKAGE_ALIASES[
            f"root{int(_root_share)}_extra_leafcount{int(_rate)}_internalcount{int(_rate)}"
        ] = (
            f"full{int(_root_share)}_leaf_count{int(_rate)}_internal_count{int(_rate)}"
        )

for _root_share in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
    SUPERVISION_RECOVERY_PACKAGE_ALIASES[
        f"root{int(_root_share)}_extra_leaffull100_internalcount100"
    ] = f"full{int(_root_share)}_leaf_full100_internal_count100"

for _root_share, _rates in {
    10: (0.5, 1.0, 1.5, 2.0),
    20: (1.0, 2.0, 3.0, 4.0),
    80: (5.0, 10.0, 15.0, 16.0),
    90: (5.0, 10.0, 15.0, 18.0),
    100: (5.0, 10.0, 15.0, 20.0),
}.items():
    for _rate in _rates:
        _rate_text = _pct_alias_text(float(_rate))
        SUPERVISION_RECOVERY_PACKAGE_ALIASES[
            f"root{int(_root_share)}_mass_local{_rate_text}"
        ] = (
            f"r{int(_root_share)}_mass_local_eq_"
            f"{float(_rate):.1f}".replace('.', 'p')
        )

for _rate in (10.0, 15.0, 20.0):
    _rate_text = _pct_alias_text(float(_rate))
    SUPERVISION_RECOVERY_PACKAGE_ALIASES[
        f"root100_extra_local{_rate_text}"
    ] = f"r100_superset_local_eq_{float(_rate):.1f}".replace(".", "p")

for _node_share in range(10, 101, 10):
    _root_share = 100 - int(_node_share)
    _node_text = _pct_alias_text(float(_node_share))
    SUPERVISION_RECOVERY_PACKAGE_ALIASES[
        f"root{int(_root_share)}_nodes{_node_text}"
    ] = f"r100_node_mass_eq_{float(_node_share):.1f}".replace(".", "p")

for _root_share in range(0, 100, 10):
    _local_share = 100 - int(_root_share)
    _local_text = _pct_alias_text(float(_local_share))
    SUPERVISION_RECOVERY_PACKAGE_ALIASES[
        f"root{int(_root_share)}_leaf{_local_text}"
    ] = f"r{int(_root_share)}_leaf_mass_eq_{float(_local_share):.1f}".replace(".", "p")
    SUPERVISION_RECOVERY_PACKAGE_ALIASES[
        f"root{int(_root_share)}_levels_equal{_local_text}"
    ] = (
        f"r{int(_root_share)}_depth_equal_mass_eq_"
        f"{float(_local_share):.1f}".replace(".", "p")
    )

_PRIMARY_PACKAGE_PUBLIC_NAMES: dict[str, str] = {}
for _alias, _canonical in SUPERVISION_RECOVERY_PACKAGE_ALIASES.items():
    _PRIMARY_PACKAGE_PUBLIC_NAMES.setdefault(str(_canonical), str(_alias))

SUPERVISION_RECOVERY_PACKAGE_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "comparison_grid_v3": (
        "root100",
        "root100_extra_local10",
        "root100_extra_local15",
        "root100_extra_local20",
    ),
    "mass_r100": (
        "root100",
        "root100_mass_local5",
        "root100_mass_local10",
        "root100_mass_local15",
        "root100_mass_local20",
    ),
    "redistribution_r100_coarse": (
        "root100",
        "root80_nodes20",
        "root50_nodes50",
        "root20_nodes80",
        "root0_nodes100",
    ),
    "redistribution_r100": (
        "root100",
        "root90_nodes10",
        "root80_nodes20",
        "root70_nodes30",
        "root60_nodes40",
        "root50_nodes50",
        "root40_nodes60",
        "root30_nodes70",
        "root20_nodes80",
        "root10_nodes90",
        "root0_nodes100",
    ),
    "root_ladder_deciles": (
        "root100",
        "root90",
        "root80",
        "root70",
        "root60",
        "root50",
        "root40",
        "root30",
        "root20",
        "root10",
    ),
    "mass_preserving_leaf_only_deciles": (
        "root100",
        "root90",
        "root90_leaf10",
        "root80",
        "root80_leaf20",
        "root70",
        "root70_leaf30",
        "root60",
        "root60_leaf40",
        "root50",
        "root50_leaf50",
        "root40",
        "root40_leaf60",
        "root30",
        "root30_leaf70",
        "root20",
        "root20_leaf80",
        "root10",
        "root10_leaf90",
        "root0_leaf100",
    ),
    "mass_preserving_levels_equal_deciles": (
        "root100",
        "root90",
        "root90_levels_equal10",
        "root80",
        "root80_levels_equal20",
        "root70",
        "root70_levels_equal30",
        "root60",
        "root60_levels_equal40",
        "root50",
        "root50_levels_equal50",
        "root40",
        "root40_levels_equal60",
        "root30",
        "root30_levels_equal70",
        "root20",
        "root20_levels_equal80",
        "root10",
        "root10_levels_equal90",
        "root0_levels_equal100",
    ),
    "mass_r10_r20_r80_r90": (
        "root10",
        "root10_mass_local0p5",
        "root10_mass_local1",
        "root10_mass_local1p5",
        "root10_mass_local2",
        "root20",
        "root20_mass_local1",
        "root20_mass_local2",
        "root20_mass_local3",
        "root20_mass_local4",
        "root80",
        "root80_mass_local5",
        "root80_mass_local10",
        "root80_mass_local15",
        "root80_mass_local16",
        "root90",
        "root90_mass_local5",
        "root90_mass_local10",
        "root90_mass_local15",
        "root90_mass_local18",
    ),
    "root10_local_rate": (
        "root10",
        "root10_extra_leafcount10_internalcount10",
        "root10_extra_leafcount20_internalcount20",
        "root10_extra_leafcount50_internalcount50",
        "root10_extra_leafcount100_internalcount100",
    ),
    "root20_local_rate": (
        "root20",
        "root20_extra_leafcount10_internalcount10",
        "root20_extra_leafcount20_internalcount20",
        "root20_extra_leafcount50_internalcount50",
        "root20_extra_leafcount100_internalcount100",
    ),
    "legacy_tree_main": (
        "root100",
        "root50",
        "root30",
        "root20",
        "root10",
        "root10_extra_leafcount100",
        "root10_extra_leaffull100",
        "root10_extra_leaffull100_internalcount100_d1",
        "root10_extra_leaffull100_internalcount100_d2",
        "root10_extra_leaffull100_internalcount100",
        "root20_extra_leaffull100_internalcount100",
        "root30_extra_leaffull100_internalcount100",
        "root50_extra_leaffull100_internalcount100",
    ),
}

LAW_PACKAGE_ALIASES: dict[str, str] = {
    "root_only": "tree_root_only",
    "c2_only": "tree_c2_only",
    "all_laws": "tree_all_laws",
}


def supervision_recovery_package_public_name(name: str) -> str:
    canonical = str(name or "").strip()
    return _PRIMARY_PACKAGE_PUBLIC_NAMES.get(canonical, canonical)


def supervision_recovery_package_names(
    valid_names: Sequence[str] | None = None,
    *,
    public_only: bool = False,
) -> tuple[str, ...]:
    if public_only:
        return tuple(sorted(SUPERVISION_RECOVERY_PACKAGE_ALIASES))
    if valid_names is None:
        return tuple(sorted(set(SUPERVISION_RECOVERY_PACKAGE_ALIASES.values())))
    return tuple(sorted({str(value) for value in valid_names if str(value).strip()}))


def supervision_recovery_package_group_names() -> tuple[str, ...]:
    return tuple(sorted(SUPERVISION_RECOVERY_PACKAGE_GROUP_ALIASES))


def resolve_supervision_recovery_package_name(
    name: str,
    *,
    valid_names: Sequence[str] | None = None,
) -> str:
    requested = str(name or "").strip()
    if not requested:
        raise ValueError("supervision recovery package name must be non-empty")
    canonical = SUPERVISION_RECOVERY_PACKAGE_ALIASES.get(requested, requested)
    valid = None if valid_names is None else {str(value) for value in valid_names}
    if valid is not None and canonical not in valid:
        raise ValueError(
            "unknown supervision recovery package "
            f"{requested!r}; valid canonical names are {sorted(valid)} "
            f"and public aliases are {sorted(SUPERVISION_RECOVERY_PACKAGE_ALIASES)}"
        )
    return canonical


def resolve_supervision_recovery_package_names(
    names: Sequence[str],
    *,
    valid_names: Sequence[str] | None = None,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    def _append_entry(entry: str) -> None:
        text = str(entry or "").strip()
        if not text:
            return
        if text in SUPERVISION_RECOVERY_PACKAGE_GROUP_ALIASES:
            for nested in SUPERVISION_RECOVERY_PACKAGE_GROUP_ALIASES[text]:
                _append_entry(str(nested))
            return
        canonical = resolve_supervision_recovery_package_name(
            text,
            valid_names=valid_names,
        )
        if canonical in seen:
            return
        seen.add(canonical)
        resolved.append(canonical)

    for raw_name in names:
        _append_entry(str(raw_name))
    return resolved


def law_package_names(
    valid_names: Sequence[str] | None = None,
    *,
    public_only: bool = False,
) -> tuple[str, ...]:
    if public_only:
        return tuple(sorted(LAW_PACKAGE_ALIASES))
    if valid_names is None:
        return tuple(sorted(set(LAW_PACKAGE_ALIASES.values())))
    return tuple(sorted({str(value) for value in valid_names if str(value).strip()}))


def resolve_law_package_name(
    name: str,
    *,
    valid_names: Sequence[str] | None = None,
) -> str:
    requested = str(name or "").strip()
    if not requested:
        raise ValueError("law package name must be non-empty")
    canonical = LAW_PACKAGE_ALIASES.get(requested, requested)
    valid = None if valid_names is None else {str(value) for value in valid_names}
    if valid is not None and canonical not in valid:
        raise ValueError(
            "unknown law package "
            f"{requested!r}; valid canonical names are {sorted(valid)} "
            f"and public aliases are {sorted(LAW_PACKAGE_ALIASES)}"
        )
    return canonical


def resolve_law_package_names(
    names: Sequence[str],
    *,
    valid_names: Sequence[str] | None = None,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        canonical = resolve_law_package_name(
            str(raw_name or "").strip(),
            valid_names=valid_names,
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved
