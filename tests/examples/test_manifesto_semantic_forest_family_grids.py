from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "methods"
    / "run_manifesto_semantic_forest_grid.py"
)


def _load_example():
    module_name = "treepo_manifesto_semantic_forest_family_grids_example"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


grid = _load_example()


def test_dspy_and_fno_plans_are_isomorphic_nine_cell_grids(tmp_path: Path) -> None:
    dspy = grid.build_family_grid_plan(tmp_path, "dspy", seed=41)
    fno = grid.build_family_grid_plan(tmp_path, "fno", seed=41)

    assert len(dspy) == len(fno) == 9
    assert (
        [(cell.target_width, cell.representation_path) for cell in dspy]
        == [(cell.target_width, cell.representation_path) for cell in fno]
        == [
            (width, representation_path)
            for width in (1, 3, 57)
            for representation_path in (
                "full_doc_direct",
                "ctree_base_summary",
                "ctree_recursive",
            )
        ]
    )
    assert {cell.family for cell in dspy} == {"dspy"}
    assert {cell.family for cell in fno} == {"fno"}


def test_family_cli_plan_selects_nine_or_eighteen_cells(tmp_path: Path) -> None:
    dspy_root = tmp_path / "dspy"
    assert (
        grid.main(
            [
                "--plan-only",
                "--family",
                "dspy",
                "--output-dir",
                str(dspy_root),
            ]
        )
        == 0
    )
    dspy = json.loads(
        (dspy_root / "semantic_forest_dspy_grid_plan.json").read_text(encoding="utf-8")
    )
    assert dspy["expected_cells"] == 9
    assert dspy["dspy_execution"] == "learned"
    assert set(dspy["family_grids"]) == {"dspy"}
    assert (
        dspy["comparison_report_schema"] == dspy["family_grids"]["dspy"]["comparison_report_schema"]
    )
    assert dspy["comparison_rows_are_not_plan_measurements"] is True

    both_root = tmp_path / "both"
    assert (
        grid.main(
            [
                "--plan-only",
                "--family",
                "both",
                "--output-dir",
                str(both_root),
            ]
        )
        == 0
    )
    both = json.loads((both_root / "semantic_forest_grid_plan.json").read_text(encoding="utf-8"))
    assert both["expected_cells"] == 18
    assert both["dspy_execution"] == "learned"
    assert set(both["family_grids"]) == {"dspy", "fno"}
    assert both["comparison_report_schema_identical_across_families"] is True
    assert (
        both["comparison_report_schema"]
        == both["family_grids"]["dspy"]["comparison_report_schema"]
        == both["family_grids"]["fno"]["comparison_report_schema"]
    )
