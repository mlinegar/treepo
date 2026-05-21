#!/usr/bin/env python3
"""Progress report for adaptive learned-g overnight runs."""

from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


EXPERIMENT_OPS = "learned-segment-lda-ops-g"
EXPERIMENT_THETA = "learned-segmented-lda-theta-g"


@dataclass(frozen=True)
class _RunRow:
    round_index: int
    experiment: str
    candidate_id: str
    seed: int
    root_mae: float
    merge_mae: float
    schedule_spread_mean: float
    score: float
    path: str


@dataclass(frozen=True)
class _CandidateAgg:
    round_index: int
    experiment: str
    candidate_id: str
    n_seeds: int
    score_mean: float
    root_mae_mean: float
    merge_mae_mean: float
    schedule_spread_mean: float


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate learned-g overnight progress report.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <output-root>/figures/learned_g_overnight",
    )
    parser.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _as_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    return float(v) if math.isfinite(v) else None


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _loss_from_metrics(metrics: Mapping[str, object]) -> float:
    root = _as_float(metrics.get("root_mae"))
    merge = _as_float(metrics.get("merge_mae"))
    leaf = _as_float(metrics.get("leaf_mae"))
    spread = _as_float(metrics.get("schedule_spread_mean"))
    spread_p95 = _as_float(metrics.get("schedule_spread_p95"))
    leaf_v = _as_float(metrics.get("leaf_violation_rate"))
    merge_v = _as_float(metrics.get("merge_violation_rate"))

    root = root if root is not None else 1e9
    merge = merge if merge is not None else 1e9
    leaf = leaf if leaf is not None else 1e9
    spread = spread if spread is not None else 1e9
    spread_p95 = spread_p95 if spread_p95 is not None else 1e9
    leaf_v = leaf_v if leaf_v is not None else 1.0
    merge_v = merge_v if merge_v is not None else 1.0

    return float(
        root
        + 0.45 * merge
        + 0.20 * leaf
        + 0.30 * spread
        + 0.10 * spread_p95
        + 0.25 * (leaf_v + merge_v)
    )


def _scan_runs(output_root: Path) -> List[_RunRow]:
    rows: List[_RunRow] = []
    pattern = str(output_root / "round_*" / "*" / "*" / "seed_*" / "summary.json")
    for fp in glob.glob(pattern):
        path = Path(fp)
        rel = path.relative_to(output_root)
        parts = rel.parts
        if len(parts) < 5:
            continue
        round_part, experiment, candidate_id, seed_part = parts[0], parts[1], parts[2], parts[3]
        if not round_part.startswith("round_") or not seed_part.startswith("seed_"):
            continue
        try:
            round_index = int(round_part.split("_", 1)[1])
            seed = int(seed_part.split("_", 1)[1])
        except Exception:
            continue

        payload = _load_json(path)
        if payload is None:
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        root = _as_float(metrics.get("root_mae"))
        merge = _as_float(metrics.get("merge_mae"))
        spread = _as_float(metrics.get("schedule_spread_mean"))
        if root is None or merge is None or spread is None:
            continue
        score = _loss_from_metrics(metrics)
        rows.append(
            _RunRow(
                round_index=int(round_index),
                experiment=str(experiment),
                candidate_id=str(candidate_id),
                seed=int(seed),
                root_mae=float(root),
                merge_mae=float(merge),
                schedule_spread_mean=float(spread),
                score=float(score),
                path=str(path),
            )
        )
    rows.sort(key=lambda r: (r.round_index, r.experiment, r.candidate_id, r.seed))
    return rows


def _aggregate_candidates(rows: Sequence[_RunRow]) -> List[_CandidateAgg]:
    grouped: Dict[Tuple[int, str, str], List[_RunRow]] = {}
    for row in rows:
        key = (int(row.round_index), str(row.experiment), str(row.candidate_id))
        grouped.setdefault(key, []).append(row)

    out: List[_CandidateAgg] = []
    for (round_index, experiment, candidate_id), group in grouped.items():
        out.append(
            _CandidateAgg(
                round_index=int(round_index),
                experiment=str(experiment),
                candidate_id=str(candidate_id),
                n_seeds=int(len(group)),
                score_mean=_mean(r.score for r in group),
                root_mae_mean=_mean(r.root_mae for r in group),
                merge_mae_mean=_mean(r.merge_mae for r in group),
                schedule_spread_mean=_mean(r.schedule_spread_mean for r in group),
            )
        )
    out.sort(key=lambda r: (r.round_index, r.experiment, r.score_mean))
    return out


def _best_by_round(candidates: Sequence[_CandidateAgg]) -> Dict[Tuple[int, str], _CandidateAgg]:
    out: Dict[Tuple[int, str], _CandidateAgg] = {}
    for row in candidates:
        key = (int(row.round_index), str(row.experiment))
        prev = out.get(key)
        if prev is None or float(row.score_mean) < float(prev.score_mean):
            out[key] = row
    return out


def _best_overall(candidates: Sequence[_CandidateAgg]) -> Dict[str, _CandidateAgg]:
    out: Dict[str, _CandidateAgg] = {}
    for row in candidates:
        key = str(row.experiment)
        prev = out.get(key)
        if prev is None or float(row.score_mean) < float(prev.score_mean):
            out[key] = row
    return out


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        ["pandoc", str(md_path.name), "-o", str(pdf_path.name), "--pdf-engine=pdflatex"],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


def _parse_utc(ts: object) -> Optional[datetime]:
    if ts is None:
        return None
    text = str(ts).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt(v: float) -> str:
    if not math.isfinite(float(v)):
        return "nan"
    if abs(float(v)) < 1e-3 or abs(float(v)) >= 1e4:
        return f"{float(v):.3e}"
    return f"{float(v):.5g}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (output_root / "figures" / "learned_g_overnight")
    pages_dir = out_dir / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    rows = _scan_runs(output_root)
    candidates = _aggregate_candidates(rows)
    round_best = _best_by_round(candidates)
    overall_best = _best_overall(candidates)

    status_payload = _load_json(output_root / "overnight_status.json")
    status = "unknown"
    started_at: Optional[datetime] = None
    eta_end: Optional[datetime] = None
    if status_payload is not None:
        status = str(status_payload.get("status", "unknown"))
        started_at = _parse_utc(status_payload.get("started_at_utc"))
        max_hours = _as_float(status_payload.get("max_hours"))
        if started_at is not None and max_hours is not None and max_hours > 0.0:
            eta_end = started_at + timedelta(hours=float(max_hours))

    rounds = sorted({int(r.round_index) for r in rows})
    experiments = sorted({str(r.experiment) for r in rows})

    fig_progress = pages_dir / "completed_runs_by_round.png"
    if rounds and experiments:
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.5), constrained_layout=True)
        for exp in experiments:
            ys = [sum(1 for r in rows if r.experiment == exp and r.round_index == ridx) for ridx in rounds]
            ax.plot(rounds, ys, marker="o", linewidth=1.8, label=exp)
        ax.set_title("Completed seed-runs per round")
        ax.set_xlabel("round")
        ax.set_ylabel("completed runs")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=9)
        _save_fig(fig, fig_progress)

    fig_scores = pages_dir / "best_scores_by_round.png"
    if rounds and experiments:
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.5), constrained_layout=True)
        for exp in experiments:
            xs = [ridx for ridx in rounds if (ridx, exp) in round_best]
            ys = [float(round_best[(ridx, exp)].score_mean) for ridx in xs]
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=exp)
        ax.set_title("Best candidate score by round")
        ax.set_xlabel("round")
        ax.set_ylabel("score (lower is better)")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=9)
        _save_fig(fig, fig_scores)

    fig_metrics = pages_dir / "best_metrics_by_round.png"
    if rounds and experiments:
        fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
        specs = [
            ("root_mae_mean", "root MAE"),
            ("merge_mae_mean", "merge MAE"),
            ("schedule_spread_mean", "schedule spread"),
        ]
        for ax, (field, title) in zip(axs, specs):
            for exp in experiments:
                xs = [ridx for ridx in rounds if (ridx, exp) in round_best]
                ys = [float(getattr(round_best[(ridx, exp)], field)) for ridx in xs]
                if xs:
                    ax.plot(xs, ys, marker="o", linewidth=1.6, label=exp)
            ax.set_title(title)
            ax.set_xlabel("round")
            ax.grid(alpha=0.2)
        axs[0].legend(loc="best", fontsize=8)
        _save_fig(fig, fig_metrics)

    diagnostics: Dict[str, object] = {
        "generated_at_utc": now.isoformat(),
        "output_root": str(output_root),
        "status": status,
        "n_completed_runs": int(len(rows)),
        "n_candidate_aggregates": int(len(candidates)),
        "rounds_found": rounds,
        "experiments_found": experiments,
        "status_started_at_utc": started_at.isoformat().replace("+00:00", "Z") if started_at is not None else None,
        "status_eta_end_utc": eta_end.isoformat().replace("+00:00", "Z") if eta_end is not None else None,
        "overall_best": {
            exp: {
                "round_index": int(best.round_index),
                "candidate_id": str(best.candidate_id),
                "n_seeds": int(best.n_seeds),
                "score_mean": float(best.score_mean),
                "root_mae_mean": float(best.root_mae_mean),
                "merge_mae_mean": float(best.merge_mae_mean),
                "schedule_spread_mean": float(best.schedule_spread_mean),
            }
            for exp, best in overall_best.items()
        },
    }
    diagnostics_path = out_dir / "learned_g_overnight_latest_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append("# Learned-g Overnight Progress Report")
    md_lines.append("")
    md_lines.append(f"- generated_at_utc: `{now.isoformat().replace('+00:00', 'Z')}`")
    md_lines.append(f"- output_root: `{output_root}`")
    md_lines.append(f"- status: `{status}`")
    if started_at is not None:
        md_lines.append(f"- started_at_utc: `{started_at.isoformat().replace('+00:00', 'Z')}`")
    if eta_end is not None:
        md_lines.append(f"- configured_end_utc: `{eta_end.isoformat().replace('+00:00', 'Z')}`")
    md_lines.append(f"- completed_runs: `{len(rows)}`")
    md_lines.append(f"- completed_rounds_seen: `{len(rounds)}`")
    md_lines.append("")
    md_lines.append("## Best Observed (So Far)")
    md_lines.append("")
    for exp in (EXPERIMENT_OPS, EXPERIMENT_THETA):
        best = overall_best.get(exp)
        if best is None:
            md_lines.append(f"- `{exp}`: no completed runs yet")
            continue
        md_lines.append(
            f"- `{exp}`: round `{best.round_index}` candidate `{best.candidate_id}` "
            f"(n_seeds={best.n_seeds}) | score `{_fmt(best.score_mean)}` | "
            f"root `{_fmt(best.root_mae_mean)}` | merge `{_fmt(best.merge_mae_mean)}` | "
            f"spread `{_fmt(best.schedule_spread_mean)}`"
        )
    md_lines.append("")
    md_lines.append("## Latest Round Snapshot")
    md_lines.append("")
    if rounds:
        latest = int(max(rounds))
        md_lines.append(f"- latest_round: `{latest}`")
        for exp in (EXPERIMENT_OPS, EXPERIMENT_THETA):
            row = round_best.get((latest, exp))
            if row is None:
                md_lines.append(f"- `{exp}`: no completed candidate in latest round")
                continue
            md_lines.append(
                f"- `{exp}`: candidate `{row.candidate_id}` "
                f"(n_seeds={row.n_seeds}) | score `{_fmt(row.score_mean)}` | "
                f"root `{_fmt(row.root_mae_mean)}` | merge `{_fmt(row.merge_mae_mean)}` | "
                f"spread `{_fmt(row.schedule_spread_mean)}`"
            )
    else:
        md_lines.append("- no rounds found yet")
    md_lines.append("")
    md_lines.append("## Artifacts")
    md_lines.append("")
    md_lines.append(f"- diagnostics_json: `{diagnostics_path}`")
    md_lines.append(f"- completed_runs_by_round_png: `{fig_progress}`")
    md_lines.append(f"- best_scores_by_round_png: `{fig_scores}`")
    md_lines.append(f"- best_metrics_by_round_png: `{fig_metrics}`")

    md_path = out_dir / "learned_g_overnight_latest.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if bool(args.emit_pdf):
        pdf_path = out_dir / "learned_g_overnight_latest.pdf"
        try:
            _run_pandoc(md_path, pdf_path)
        except Exception:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
