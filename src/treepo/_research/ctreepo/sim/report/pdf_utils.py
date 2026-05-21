"""Shared PDF and numeric utilities for unified reports."""
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ── safe numerics ────────────────────────────────────────────────────────


def safe_float(value: object, default: float = float("nan")) -> float:
    """Convert *value* to float, returning *default* on failure or non-finite."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def safe_float_key(mapping: dict, key: str, default: float = float("nan")) -> float:
    """``safe_float(mapping.get(key, default))``."""
    return safe_float(mapping.get(key, default), default=default)


def safe_mean(values: Iterable[object]) -> float:
    xs = [float(v) for v in (safe_float(x) for x in values) if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(fmean(xs))


def safe_sem(values: Iterable[object]) -> float:
    """Standard error of the mean, ignoring non-finite entries."""
    vals = [float(v) for v in (safe_float(x) for x in values) if math.isfinite(float(v))]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


def normalize(value: float, *, scale: float) -> float:
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        return float("nan")
    return float(value) / float(scale)


# ── PDF page helpers ─────────────────────────────────────────────────────


def write_text_page(pdf: PdfPages, *, title: str, lines: Sequence[str]) -> None:
    """Write a multi-line text page to *pdf*, auto-paginating if needed."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=16, fontweight="bold", va="top")
    y = 0.95
    for raw in lines:
        chunks = textwrap.wrap(
            str(raw), width=105, break_long_words=False, break_on_hyphens=False,
        ) or [""]
        for chunk in chunks:
            ax.text(0.0, y, chunk, fontsize=10.0, va="top")
            y -= 0.024
            if y < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
                ax.axis("off")
                y = 0.97
    pdf.savefig(fig)
    plt.close(fig)


def write_image_page(pdf: PdfPages, *, image_path: Path, title: str) -> None:
    """Write a full-page image to *pdf*."""
    if not image_path.exists():
        return
    image = plt.imread(str(image_path))
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.90])
    ax.axis("off")
    ax.imshow(image)
    fig.suptitle(title, fontsize=14, y=0.98)
    pdf.savefig(fig)
    plt.close(fig)


def page_header(fig, title: str, subtitle: str) -> None:
    """Add a title+subtitle header to the top of *fig*."""
    title_wrapped = textwrap.fill(title, width=60)
    title_lines = title_wrapped.count("\n") + 1
    fig.text(0.06, 0.965, title_wrapped, fontsize=18, fontweight="bold", ha="left", va="top")
    subtitle_y = 0.965 - 0.046 * title_lines
    fig.text(
        0.06, subtitle_y, textwrap.fill(subtitle, width=130),
        fontsize=10.5, color="#444444", ha="left", va="top",
    )


# ── CSV helper ───────────────────────────────────────────────────────────


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write *rows* to a CSV file, preserving insertion-order field names."""
    import csv

    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


__all__ = [
    "normalize",
    "page_header",
    "safe_float",
    "safe_float_key",
    "safe_mean",
    "safe_sem",
    "write_csv",
    "write_image_page",
    "write_text_page",
]
