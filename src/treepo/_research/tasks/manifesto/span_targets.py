"""Targets derived from Manifesto Project quasi-sentence annotations.

The quasi-sentence corpus gives one CMP code per annotated span.  This module
keeps the code normalization and additive target construction in one place so
the DSPy f/g ladder can use exact annotation-derived labels without calling a
teacher model.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


RILE_LEFT_CODES = {
    "103",
    "105",
    "106",
    "107",
    "202",
    "403",
    "404",
    "406",
    "412",
    "413",
    "504",
    "506",
    "701",
}

RILE_RIGHT_CODES = {
    "104",
    "201",
    "203",
    "305",
    "401",
    "402",
    "407",
    "414",
    "505",
    "601",
    "603",
    "605",
    "606",
}

CMP_DOMAIN_KEYS = tuple(f"domain_{idx}" for idx in range(1, 8))
COMPACT_TARGET_DIMENSIONS = ("rile",) + CMP_DOMAIN_KEYS


def normalize_cmp_code(value: Any) -> Optional[str]:
    """Return a canonical CMP main code.

    Examples
    --------
    ``605.1`` and ``6051`` both normalize to ``605``.  ``H`` and ``000`` are
    kept as special non-policy labels.
    """
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        text = f"{int(value):03d}"
    else:
        text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in {"H", "HEADLINE"}:
        return "H"
    if upper in {"NAN", "NONE", "NULL"}:
        return None
    if "." in text:
        text = text.split(".", 1)[0]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if int(digits or "0") == 0:
        return "000"
    if len(digits) >= 3:
        return digits[:3]
    return digits.zfill(3)


def cmp_domain(code: Any) -> Optional[str]:
    normalized = normalize_cmp_code(code)
    if normalized is None or normalized in {"H", "000"}:
        return None
    if len(normalized) != 3 or not normalized.isdigit():
        return None
    domain = int(normalized[0])
    if 1 <= domain <= 7:
        return f"domain_{domain}"
    return None


def aggregate_cmp_codes(
    codes: Iterable[Any], *, denominator: str = "non_header"
) -> dict[str, Any]:
    """Aggregate normalized CMP codes into additive counts and compact targets."""
    counter: Counter[str] = Counter()
    for raw in codes:
        code = normalize_cmp_code(raw)
        if code is not None:
            counter[code] += 1
    return targets_from_counts(counter, denominator=denominator)


def targets_from_counts(
    counts: Mapping[str, int], *, denominator: str = "non_header"
) -> dict[str, Any]:
    """Build additive RILE/domain targets from CMP-code counts.

    ``denominator`` selects the RILE normalization convention:

    * ``"non_header"`` (default, the repo standard): all coded quasi-sentences
      except ``H`` headers. Matches the published MPDS ``rile`` best
      (Step 0 gate 2026-06-09: Pearson 0.9975 / MAE 0.49 vs 0.9944 / 1.35).
    * ``"all"``: every counted quasi-sentence including headers (the literal
      Laver & Budge denominator). Kept for comparisons only.
    """
    counter = Counter({str(k): int(v) for k, v in dict(counts).items() if int(v) > 0})
    total_items = int(sum(counter.values()))
    total_non_header = int(sum(v for k, v in counter.items() if k != "H"))
    if denominator == "non_header":
        denominator_count = total_non_header
    elif denominator == "all":
        denominator_count = total_items
    else:
        raise ValueError(
            f"denominator must be 'non_header' or 'all', got {denominator!r}"
        )
    denominator = max(1, denominator_count)
    left = int(sum(counter.get(code, 0) for code in RILE_LEFT_CODES))
    right = int(sum(counter.get(code, 0) for code in RILE_RIGHT_CODES))
    rile_raw = 100.0 * float(right - left) / float(denominator)
    rile_norm = max(0.0, min(1.0, (rile_raw + 100.0) / 200.0))

    domain_counts = {key: 0 for key in CMP_DOMAIN_KEYS}
    for code, count in counter.items():
        domain = cmp_domain(code)
        if domain is not None:
            domain_counts[domain] += int(count)
    compact = {"rile": float(rile_norm)}
    compact.update(
        {key: float(domain_counts[key]) / float(denominator) for key in CMP_DOMAIN_KEYS}
    )
    return {
        "counts": dict(sorted(counter.items())),
        "total_items": total_items,
        "total_non_header": total_non_header,
        "left_count": left,
        "right_count": right,
        "rile_raw": float(rile_raw),
        "rile": float(rile_norm),
        "domain_counts": domain_counts,
        "compact": compact,
    }


def merge_count_payloads(
    payloads: Sequence[Mapping[str, Any]], *, denominator: str = "non_header"
) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for payload in payloads:
        for code, count in dict(payload.get("counts") or {}).items():
            counter[str(code)] += int(count)
    return targets_from_counts(counter, denominator=denominator)


def render_policy_state(target: Mapping[str, Any], *, max_top_codes: int = 8) -> str:
    """Render a compact text state used as the gold summary target for g."""
    counts = Counter({str(k): int(v) for k, v in dict(target.get("counts") or {}).items()})
    top_codes = counts.most_common(max(0, int(max_top_codes)))
    compact = {
        key: round(float(value), 6)
        for key, value in dict(target.get("compact") or {}).items()
        if key in COMPACT_TARGET_DIMENSIONS
    }
    payload = {
        "cmp_state": {
            "compact_targets": compact,
            "rile_raw": round(float(target.get("rile_raw", 0.0)), 6),
            "total_non_header": int(target.get("total_non_header", 0) or 0),
            "left_count": int(target.get("left_count", 0) or 0),
            "right_count": int(target.get("right_count", 0) or 0),
            "top_codes": [[code, int(count)] for code, count in top_codes],
        }
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def parse_compact_scores_json(value: Any) -> dict[str, float]:
    """Parse a DSPy f output into the compact target vector."""
    if isinstance(value, Mapping):
        raw = value
    else:
        text = str(value or "").strip()
        if not text:
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if "scores" in raw and isinstance(raw.get("scores"), Mapping):
        raw = raw["scores"]
    if "compact_targets" in raw and isinstance(raw.get("compact_targets"), Mapping):
        raw = raw["compact_targets"]
    if "cmp_state" in raw and isinstance(raw.get("cmp_state"), Mapping):
        return parse_compact_scores_json(raw["cmp_state"])
    out: dict[str, float] = {}
    for key in COMPACT_TARGET_DIMENSIONS:
        if key not in raw:
            continue
        try:
            val = float(raw[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out[key] = max(0.0, min(1.0, val))
    return out

