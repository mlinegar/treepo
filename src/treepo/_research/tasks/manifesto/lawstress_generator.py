"""Utilities for synthetic local-law stress benchmarks for information extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from treepo._research.tasks.manifesto.data_loader import ManifestoDataset

logger = logging.getLogger(__name__)

RILE_MIN = -100.0
RILE_MAX = 100.0
RILE_RANGE = RILE_MAX - RILE_MIN

DIFFICULTY_HARD = "hard"
DIFFICULTY_CONTROL = "control"
ANCHOR_REAL = "real_anchor"
ANCHOR_SYNTHETIC = "synthetic"

LAW_TARGETS: Tuple[str, ...] = ("c1_sufficiency", "c2_idempotence", "c3_merge")
FAMILIES: Tuple[str, ...] = (
    "polarity_cancellation",
    "cross_chunk_dependency",
    "idempotence_rephrase_trap",
    "merge_order_asymmetry",
)


@dataclass(frozen=True)
class RILEBin:
    """Named raw-RILE bin with explicit inclusive/exclusive boundaries."""

    name: str
    lower: float
    upper: float
    include_lower: bool = True
    include_upper: bool = False

    def contains(self, value: float) -> bool:
        if self.include_lower:
            lower_ok = value >= self.lower
        else:
            lower_ok = value > self.lower
        if self.include_upper:
            upper_ok = value <= self.upper
        else:
            upper_ok = value < self.upper
        return bool(lower_ok and upper_ok)

    def sample(self, rng: random.Random) -> float:
        eps = 1e-6
        low = self.lower + (0.0 if self.include_lower else eps)
        high = self.upper - (0.0 if self.include_upper else eps)
        if high <= low:
            return float(self.lower)
        return float(rng.uniform(low, high))


RILE_BINS: Tuple[RILEBin, ...] = (
    RILEBin("extreme_left", -100.0, -75.0, include_lower=True, include_upper=False),
    RILEBin("far_left", -75.0, -50.0, include_lower=True, include_upper=False),
    RILEBin("left", -50.0, -25.0, include_lower=True, include_upper=False),
    RILEBin("center_left", -25.0, -10.0, include_lower=True, include_upper=False),
    RILEBin("center", -10.0, 10.0, include_lower=True, include_upper=True),
    RILEBin("center_right", 10.0, 25.0, include_lower=False, include_upper=True),
    RILEBin("right", 25.0, 50.0, include_lower=False, include_upper=True),
    RILEBin("far_right", 50.0, 75.0, include_lower=False, include_upper=True),
    RILEBin("extreme_right", 75.0, 100.0, include_lower=False, include_upper=True),
)

_BIN_BY_NAME: Dict[str, RILEBin] = {entry.name: entry for entry in RILE_BINS}


class TeacherScoreFn(Protocol):
    def __call__(self, text: str) -> float: ...


class TeacherRewriteFn(Protocol):
    def __call__(self, text: str, spec: "LawStressSpec", truth_raw: float) -> str: ...


class ReferenceSummaryFn(Protocol):
    def __call__(self, text: str, spec: "LawStressSpec", truth_raw: float) -> str: ...


@dataclass
class LawStressSpec:
    """Generation target for one synthetic sample."""

    example_id: str
    split: str
    bin_name: str
    law_target: str
    family: str
    difficulty: str
    anchor_source: str


@dataclass
class PolicyAtom:
    """Deterministic directional atom used to compute ground-truth RILE."""

    atom_id: str
    direction: int
    strength: float
    weight: float
    segment: str
    topic: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "direction": int(self.direction),
            "strength": float(self.strength),
            "weight": float(self.weight),
            "segment": str(self.segment),
            "topic": str(self.topic),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyAtom":
        return cls(
            atom_id=str(data.get("atom_id", "")),
            direction=int(data.get("direction", 0)),
            strength=float(data.get("strength", 0.0)),
            weight=float(data.get("weight", 0.0)),
            segment=str(data.get("segment", "A")),
            topic=str(data.get("topic", "policy")),
        )


@dataclass
class LawStressRecord:
    """Final synthetic benchmark record with truth labels and artifacts."""

    example_id: str
    split: str
    bin_name: str
    law_target: str
    family: str
    difficulty: str
    anchor_source: str

    text: str
    segment_a: str
    segment_b: str

    policy_atoms: List[PolicyAtom]

    target_raw: float
    y_raw: float
    y_norm: float
    yA_raw: float
    yB_raw: float
    y_merge_expected_raw: float

    teacher_score_doc: float
    teacher_score_segment_a: float
    teacher_score_segment_b: float
    naive_summary: str
    naive_score_raw: float
    naive_drift_norm: float

    reference_summary: str
    attempts_used: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "bin_name": self.bin_name,
            "law_target": self.law_target,
            "family": self.family,
            "difficulty": self.difficulty,
            "anchor_source": self.anchor_source,
            "text": self.text,
            "segment_a": self.segment_a,
            "segment_b": self.segment_b,
            "policy_atoms": [atom.to_dict() for atom in self.policy_atoms],
            "target_raw": float(self.target_raw),
            "y_raw": float(self.y_raw),
            "y_norm": float(self.y_norm),
            "yA_raw": float(self.yA_raw),
            "yB_raw": float(self.yB_raw),
            "y_merge_expected_raw": float(self.y_merge_expected_raw),
            "teacher_score_doc": float(self.teacher_score_doc),
            "teacher_score_segment_a": float(self.teacher_score_segment_a),
            "teacher_score_segment_b": float(self.teacher_score_segment_b),
            "naive_summary": self.naive_summary,
            "naive_score_raw": float(self.naive_score_raw),
            "naive_drift_norm": float(self.naive_drift_norm),
            "reference_summary": self.reference_summary,
            "attempts_used": int(self.attempts_used),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LawStressRecord":
        return cls(
            example_id=str(data.get("example_id", "")),
            split=str(data.get("split", "")),
            bin_name=str(data.get("bin_name", "")),
            law_target=str(data.get("law_target", "")),
            family=str(data.get("family", "")),
            difficulty=str(data.get("difficulty", "")),
            anchor_source=str(data.get("anchor_source", "")),
            text=str(data.get("text", "")),
            segment_a=str(data.get("segment_a", "")),
            segment_b=str(data.get("segment_b", "")),
            policy_atoms=[PolicyAtom.from_dict(atom) for atom in (data.get("policy_atoms", []) or [])],
            target_raw=float(data.get("target_raw", 0.0)),
            y_raw=float(data.get("y_raw", 0.0)),
            y_norm=float(data.get("y_norm", 0.5)),
            yA_raw=float(data.get("yA_raw", 0.0)),
            yB_raw=float(data.get("yB_raw", 0.0)),
            y_merge_expected_raw=float(data.get("y_merge_expected_raw", 0.0)),
            teacher_score_doc=float(data.get("teacher_score_doc", 0.0)),
            teacher_score_segment_a=float(data.get("teacher_score_segment_a", 0.0)),
            teacher_score_segment_b=float(data.get("teacher_score_segment_b", 0.0)),
            naive_summary=str(data.get("naive_summary", "")),
            naive_score_raw=float(data.get("naive_score_raw", 0.0)),
            naive_drift_norm=float(data.get("naive_drift_norm", 0.0)),
            reference_summary=str(data.get("reference_summary", "")),
            attempts_used=int(data.get("attempts_used", 1)),
        )

    def to_benchmark_doc(self) -> Dict[str, Any]:
        return {
            "id": self.example_id,
            "doc_id": self.example_id,
            "text": self.text,
            "reference_score": float(self.y_raw),
            "score": float(self.y_raw),
            "metadata": {
                "split": self.split,
                "bin_name": self.bin_name,
                "law_target": self.law_target,
                "family": self.family,
                "difficulty": self.difficulty,
                "anchor_source": self.anchor_source,
                "y_raw": float(self.y_raw),
                "y_norm": float(self.y_norm),
                "yA_raw": float(self.yA_raw),
                "yB_raw": float(self.yB_raw),
                "y_merge_expected_raw": float(self.y_merge_expected_raw),
                "segment_a": self.segment_a,
                "segment_b": self.segment_b,
                "reference_summary": self.reference_summary,
            },
        }


def normalize_rile(raw: float) -> float:
    return max(0.0, min(1.0, (float(raw) - RILE_MIN) / RILE_RANGE))


def denormalize_rile(norm: float) -> float:
    return max(RILE_MIN, min(RILE_MAX, RILE_MIN + float(norm) * RILE_RANGE))


def clamp_raw_rile(value: float) -> float:
    return max(RILE_MIN, min(RILE_MAX, float(value)))


def get_rile_bin(value_raw: float) -> RILEBin:
    for entry in RILE_BINS:
        if entry.contains(float(value_raw)):
            return entry
    # Guard against tiny floating errors at boundaries.
    if float(value_raw) <= RILE_MIN:
        return RILE_BINS[0]
    return RILE_BINS[-1]


def compute_raw_rile_from_atoms(policy_atoms: Sequence[PolicyAtom]) -> float:
    numerator = 0.0
    denominator = 0.0
    for atom in policy_atoms:
        mass = float(atom.strength) * float(atom.weight)
        numerator += float(atom.direction) * mass
        denominator += mass
    if denominator <= 1e-12:
        return 0.0
    return clamp_raw_rile(100.0 * numerator / denominator)


def compute_segment_raw_rile(policy_atoms: Sequence[PolicyAtom], segment: str) -> float:
    filtered = [atom for atom in policy_atoms if str(atom.segment).upper() == str(segment).upper()]
    if not filtered:
        return 0.0
    return compute_raw_rile_from_atoms(filtered)


def length_weighted_mean(value_a: float, value_b: float, len_a: int, len_b: int) -> float:
    total = int(len_a) + int(len_b)
    if total <= 0:
        return clamp_raw_rile(0.5 * (float(value_a) + float(value_b)))
    weighted = (float(value_a) * int(len_a) + float(value_b) * int(len_b)) / float(total)
    return clamp_raw_rile(weighted)


def naive_compress_text(text: str, max_chars: int = 280) -> str:
    rendered = str(text or "").strip()
    if len(rendered) <= max_chars:
        return rendered
    head = max_chars // 2
    tail = max_chars - head
    return f"{rendered[:head].rstrip()} ... {rendered[-tail:].lstrip()}"


def _sample_balanced_labels(
    n_items: int,
    labels: Sequence[str],
    rng: random.Random,
) -> List[str]:
    if n_items <= 0:
        return []
    values = [labels[idx % len(labels)] for idx in range(n_items)]
    rng.shuffle(values)
    return values


def _allocate_binary_mix(
    n_items: int,
    primary_label: str,
    secondary_label: str,
    primary_ratio: float,
    rng: random.Random,
) -> List[str]:
    if n_items <= 0:
        return []
    primary_count = int(round(float(primary_ratio) * n_items))
    primary_count = max(0, min(n_items, primary_count))
    values = [primary_label] * primary_count + [secondary_label] * (n_items - primary_count)
    rng.shuffle(values)
    return values


def _assign_anchor_sources(
    specs: List[LawStressSpec],
    *,
    real_anchor_ratio: float,
    rng: random.Random,
) -> None:
    by_difficulty: Dict[str, List[int]] = {DIFFICULTY_HARD: [], DIFFICULTY_CONTROL: []}
    for idx, spec in enumerate(specs):
        by_difficulty.setdefault(spec.difficulty, []).append(idx)

    for difficulty, indices in by_difficulty.items():
        if not indices:
            continue
        n_real = int(round(len(indices) * float(real_anchor_ratio)))
        n_real = max(0, min(len(indices), n_real))
        shuffled = list(indices)
        rng.shuffle(shuffled)
        real_set = set(shuffled[:n_real])
        for idx in indices:
            specs[idx].anchor_source = ANCHOR_REAL if idx in real_set else ANCHOR_SYNTHETIC


def generate_lawstress_specs(
    split_sizes: Dict[str, int] | None = None,
    *,
    hard_ratio: float = 0.8,
    real_anchor_ratio: float = 0.3,
    seed: int = 42,
) -> List[LawStressSpec]:
    """Create balanced generation specs for all requested splits."""

    split_sizes = split_sizes or {"train": 600, "val": 150, "test": 150}
    rng = random.Random(int(seed))

    all_specs: List[LawStressSpec] = []
    for split in ("train", "val", "test"):
        n_split = int(split_sizes.get(split, 0) or 0)
        if n_split <= 0:
            continue

        bins = _sample_balanced_labels(n_split, [entry.name for entry in RILE_BINS], rng)
        laws = _sample_balanced_labels(n_split, LAW_TARGETS, rng)
        families = _sample_balanced_labels(n_split, FAMILIES, rng)
        difficulties = _allocate_binary_mix(
            n_split,
            DIFFICULTY_HARD,
            DIFFICULTY_CONTROL,
            primary_ratio=hard_ratio,
            rng=rng,
        )

        split_specs: List[LawStressSpec] = []
        for idx in range(n_split):
            split_specs.append(
                LawStressSpec(
                    example_id=f"lawstress_{split}_{idx:04d}",
                    split=split,
                    bin_name=bins[idx],
                    law_target=laws[idx],
                    family=families[idx],
                    difficulty=difficulties[idx],
                    anchor_source=ANCHOR_SYNTHETIC,
                )
            )

        _assign_anchor_sources(split_specs, real_anchor_ratio=real_anchor_ratio, rng=rng)
        all_specs.extend(split_specs)

    return all_specs


def summarize_spec_balance(specs: Sequence[LawStressSpec]) -> Dict[str, Any]:
    """Return split-level balance counts for diagnostics/tests."""

    summary: Dict[str, Any] = {"splits": {}}
    for split in sorted({spec.split for spec in specs}):
        split_specs = [spec for spec in specs if spec.split == split]
        split_stats: Dict[str, Dict[str, int]] = {
            "bins": {},
            "laws": {},
            "families": {},
            "difficulty": {},
            "anchor_source": {},
            "anchor_by_difficulty": {},
        }
        for spec in split_specs:
            split_stats["bins"][spec.bin_name] = split_stats["bins"].get(spec.bin_name, 0) + 1
            split_stats["laws"][spec.law_target] = split_stats["laws"].get(spec.law_target, 0) + 1
            split_stats["families"][spec.family] = split_stats["families"].get(spec.family, 0) + 1
            split_stats["difficulty"][spec.difficulty] = split_stats["difficulty"].get(spec.difficulty, 0) + 1
            split_stats["anchor_source"][spec.anchor_source] = split_stats["anchor_source"].get(spec.anchor_source, 0) + 1
            key = f"{spec.difficulty}:{spec.anchor_source}"
            split_stats["anchor_by_difficulty"][key] = split_stats["anchor_by_difficulty"].get(key, 0) + 1
        summary["splits"][split] = {
            "n": len(split_specs),
            **split_stats,
        }
    return summary


_LEFT_TOPICS = (
    "public housing",
    "collective bargaining",
    "progressive taxation",
    "welfare expansion",
    "public ownership",
    "labor protections",
    "environmental regulation",
    "social equality",
)
_RIGHT_TOPICS = (
    "private enterprise",
    "tax reductions",
    "law and order",
    "border controls",
    "defense spending",
    "traditional values",
    "market competition",
    "business incentives",
)


def _sample_policy_atoms_for_target(
    spec: LawStressSpec,
    *,
    target_raw: float,
    rng: random.Random,
) -> List[PolicyAtom]:
    n_atoms = 14 if spec.difficulty == DIFFICULTY_HARD else 9
    target_norm = max(-1.0, min(1.0, float(target_raw) / 100.0))
    p_right = 0.5 * (1.0 + target_norm)
    p_right = max(0.02, min(0.98, p_right))

    atoms: List[PolicyAtom] = []
    for idx in range(n_atoms):
        direction = 1 if rng.random() < p_right else -1
        strength = rng.uniform(0.12, 1.0)
        weight = rng.uniform(0.5, 2.0)
        segment = "A" if idx % 2 == 0 else "B"
        topic = rng.choice(_RIGHT_TOPICS if direction > 0 else _LEFT_TOPICS)
        atoms.append(
            PolicyAtom(
                atom_id=f"atom_{idx:02d}",
                direction=direction,
                strength=float(strength),
                weight=float(weight),
                segment=segment,
                topic=topic,
            )
        )

    return atoms


def _atom_sentence(atom: PolicyAtom, family: str, rng: random.Random) -> str:
    polarity = "right" if atom.direction > 0 else "left"
    intensity = "strongly" if atom.strength >= 0.66 else "moderately" if atom.strength >= 0.33 else "slightly"

    if family == "polarity_cancellation":
        prefix = "While acknowledging trade-offs,"
    elif family == "cross_chunk_dependency":
        prefix = "As noted in the prior section,"
    elif family == "idempotence_rephrase_trap":
        prefix = "In qualified terms,"
    elif family == "merge_order_asymmetry":
        prefix = "Sequenced after earlier commitments,"
    else:
        prefix = ""

    qualifier_pool = (
        "with budget discipline",
        "under phased implementation",
        "through coalition oversight",
        "with measurable milestones",
        "via statutory safeguards",
    )
    qualifier = rng.choice(qualifier_pool)
    return (
        f"{prefix} the source text {intensity} emphasizes {atom.topic} "
        f"as a {polarity}-leaning policy signal, {qualifier}."
    ).strip()


def _family_postscript(family: str) -> str:
    if family == "polarity_cancellation":
        return "The document repeatedly balances opposing priorities, making directional signal easy to blur in short summaries."
    if family == "cross_chunk_dependency":
        return "Several commitments only make sense when references across distant paragraphs are preserved."
    if family == "idempotence_rephrase_trap":
        return "Many lines use layered caveats, so repeated compression can erase key qualifiers."
    if family == "merge_order_asymmetry":
        return "Meaning depends on the order that commitments and exceptions are introduced and reconciled."
    return ""


def _compose_segments(
    atoms: Sequence[PolicyAtom],
    *,
    spec: LawStressSpec,
    rng: random.Random,
) -> Tuple[str, str]:
    atoms_a = [atom for atom in atoms if atom.segment == "A"]
    atoms_b = [atom for atom in atoms if atom.segment == "B"]

    lines_a = [_atom_sentence(atom, spec.family, rng) for atom in atoms_a]
    lines_b = [_atom_sentence(atom, spec.family, rng) for atom in atoms_b]

    if spec.family == "cross_chunk_dependency" and lines_a and lines_b:
        lines_b.insert(0, "The following proposals define the antecedents referenced above, not a reversal of them.")
    if spec.family == "merge_order_asymmetry" and lines_a and lines_b:
        lines_a.append("This sequencing is deliberate and should not be inverted.")
        lines_b.append("Only after these measures does the document permit limited exceptions.")
    if spec.family == "idempotence_rephrase_trap" and lines_a and lines_b:
        lines_a.append("These claims are conditional and do not imply unconditional support.")
        lines_b.append("The caveats remain part of the core commitment, not footnotes.")
    if spec.family == "polarity_cancellation" and lines_a and lines_b:
        lines_a.append("A parallel clause in the next section intentionally offsets this stance.")
        lines_b.append("This offset is partial and should not erase the original directional signal.")

    segment_a = "\n".join(lines_a).strip()
    segment_b = "\n".join(lines_b).strip()
    return segment_a, segment_b


def _sample_anchor_snippets(
    *,
    rng: random.Random,
    max_snippets: int,
    snippet_chars: int,
) -> List[str]:
    if max_snippets <= 0:
        return []

    snippets: List[str] = []
    try:
        dataset = ManifestoDataset(require_text=True)
        ids = dataset.get_all_ids()
        rng.shuffle(ids)
        for manifesto_id in ids:
            if len(snippets) >= max_snippets:
                break
            sample = dataset.get_sample(manifesto_id)
            if sample is None:
                continue
            text = str(sample.text or "").strip()
            if len(text) < max(200, snippet_chars // 2):
                continue
            start_max = max(0, len(text) - snippet_chars)
            start = rng.randint(0, start_max) if start_max > 0 else 0
            snippet = text[start:start + snippet_chars].strip()
            if snippet:
                snippets.append(snippet)
    except Exception as exc:
        logger.warning("Failed to load real-anchor snippets from ManifestoDataset: %s", exc)
        return []

    return snippets


def _build_document_text(
    *,
    spec: LawStressSpec,
    segment_a: str,
    segment_b: str,
    anchor_text: Optional[str],
) -> str:
    intro = (
        "This synthetic policy document is designed for local-law stress testing focused on "
        "faithful information extraction under a directional scoring rubric."
    )
    family_line = f"Stress family: {spec.family}. Law target: {spec.law_target}."
    postscript = _family_postscript(spec.family)

    chunks = [intro, family_line]
    if anchor_text:
        chunks.append("Context anchor (real source fragment):")
        chunks.append(anchor_text)
    chunks.append("Segment A:")
    chunks.append(segment_a)
    chunks.append("Segment B:")
    chunks.append(segment_b)
    if postscript:
        chunks.append(postscript)
    return "\n\n".join(part for part in chunks if part).strip()


def _fallback_naive_score(
    *,
    truth_raw: float,
    difficulty: str,
    rng: random.Random,
) -> float:
    if difficulty == DIFFICULTY_HARD:
        desired = rng.uniform(45.0, 65.0)
    else:
        desired = rng.uniform(2.0, 9.0)

    truth = float(clamp_raw_rile(truth_raw))
    headroom_pos = 100.0 - truth
    headroom_neg = truth + 100.0

    viable: List[float] = []
    if headroom_pos >= desired:
        viable.append(1.0)
    if headroom_neg >= desired:
        viable.append(-1.0)
    if not viable:
        viable = [1.0 if headroom_pos >= headroom_neg else -1.0]
        if difficulty == DIFFICULTY_HARD:
            desired = max(0.0, min(max(headroom_pos, headroom_neg), 65.0))

    sign = viable[int(rng.random() * len(viable)) % len(viable)]
    return clamp_raw_rile(truth + sign * desired)


def _default_reference_summary(text: str, max_chars: int = 500) -> str:
    rendered = str(text or "").strip()
    if not rendered:
        return ""
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars].rstrip() + " ..."


def generate_lawstress_records(
    specs: Sequence[LawStressSpec],
    *,
    seed: int = 42,
    max_attempts: int = 4,
    teacher_score_fn: Optional[TeacherScoreFn] = None,
    teacher_rewrite_fn: Optional[TeacherRewriteFn] = None,
    reference_summary_fn: Optional[ReferenceSummaryFn] = None,
    real_anchor_snippets: Optional[Sequence[str]] = None,
    hard_drift_threshold_norm: float = 0.20,
    control_drift_threshold_norm: float = 0.08,
    doc_score_tolerance_raw: float = 10.0,
    segment_score_tolerance_raw: float = 12.0,
) -> List[LawStressRecord]:
    """Generate accepted records according to local-law synthetic constraints."""

    rng = random.Random(int(seed))
    snippets = list(real_anchor_snippets or [])
    if not snippets and any(spec.anchor_source == ANCHOR_REAL for spec in specs):
        snippets = _sample_anchor_snippets(
            rng=rng,
            max_snippets=max(256, len(specs)),
            snippet_chars=420,
        )

    accepted: List[LawStressRecord] = []
    dropped = 0

    for spec in specs:
        bin_spec = _BIN_BY_NAME.get(spec.bin_name)
        if bin_spec is None:
            raise ValueError(f"Unknown bin_name in spec: {spec.bin_name}")

        accepted_record: Optional[LawStressRecord] = None
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            target_raw = bin_spec.sample(rng)
            atoms: Optional[List[PolicyAtom]] = None
            y_raw: Optional[float] = None
            for _ in range(32):
                trial_atoms = _sample_policy_atoms_for_target(spec, target_raw=target_raw, rng=rng)
                trial_score = compute_raw_rile_from_atoms(trial_atoms)
                if bin_spec.contains(trial_score):
                    atoms = trial_atoms
                    y_raw = trial_score
                    break
            if atoms is None or y_raw is None:
                continue

            y_a = compute_segment_raw_rile(atoms, "A")
            y_b = compute_segment_raw_rile(atoms, "B")
            seg_a, seg_b = _compose_segments(atoms, spec=spec, rng=rng)
            y_merge_expected = length_weighted_mean(y_a, y_b, len(seg_a), len(seg_b))

            anchor_text = None
            if spec.anchor_source == ANCHOR_REAL and snippets:
                anchor_text = snippets[rng.randrange(len(snippets))]

            text = _build_document_text(spec=spec, segment_a=seg_a, segment_b=seg_b, anchor_text=anchor_text)
            if teacher_rewrite_fn is not None:
                try:
                    rewritten = str(teacher_rewrite_fn(text, spec, y_raw) or "").strip()
                    if rewritten:
                        text = rewritten
                except Exception as exc:
                    logger.debug("Teacher rewrite failed for %s: %s", spec.example_id, exc)

            if teacher_score_fn is not None:
                try:
                    score_doc = float(teacher_score_fn(text))
                    score_a = float(teacher_score_fn(seg_a))
                    score_b = float(teacher_score_fn(seg_b))
                except Exception as exc:
                    logger.debug("Teacher score call failed for %s: %s", spec.example_id, exc)
                    continue
            else:
                score_doc = float(y_raw)
                score_a = float(y_a)
                score_b = float(y_b)

            if abs(score_doc - y_raw) > float(doc_score_tolerance_raw):
                continue
            if abs(score_a - y_a) > float(segment_score_tolerance_raw):
                continue
            if abs(score_b - y_b) > float(segment_score_tolerance_raw):
                continue

            naive_summary = naive_compress_text(text)
            if teacher_score_fn is not None:
                try:
                    naive_score = float(teacher_score_fn(naive_summary))
                except Exception:
                    naive_score = _fallback_naive_score(truth_raw=y_raw, difficulty=spec.difficulty, rng=rng)
            else:
                naive_score = _fallback_naive_score(truth_raw=y_raw, difficulty=spec.difficulty, rng=rng)

            naive_drift_norm = abs(normalize_rile(naive_score) - normalize_rile(y_raw))
            if spec.difficulty == DIFFICULTY_HARD and not (naive_drift_norm > float(hard_drift_threshold_norm)):
                continue
            if spec.difficulty == DIFFICULTY_CONTROL and not (naive_drift_norm < float(control_drift_threshold_norm)):
                continue

            if reference_summary_fn is not None:
                try:
                    reference_summary = str(reference_summary_fn(text, spec, y_raw) or "").strip()
                except Exception:
                    reference_summary = _default_reference_summary(text)
            else:
                reference_summary = _default_reference_summary(text)

            accepted_record = LawStressRecord(
                example_id=spec.example_id,
                split=spec.split,
                bin_name=spec.bin_name,
                law_target=spec.law_target,
                family=spec.family,
                difficulty=spec.difficulty,
                anchor_source=spec.anchor_source,
                text=text,
                segment_a=seg_a,
                segment_b=seg_b,
                policy_atoms=list(atoms),
                target_raw=float(target_raw),
                y_raw=float(y_raw),
                y_norm=normalize_rile(y_raw),
                yA_raw=float(y_a),
                yB_raw=float(y_b),
                y_merge_expected_raw=float(y_merge_expected),
                teacher_score_doc=float(score_doc),
                teacher_score_segment_a=float(score_a),
                teacher_score_segment_b=float(score_b),
                naive_summary=naive_summary,
                naive_score_raw=float(naive_score),
                naive_drift_norm=float(naive_drift_norm),
                reference_summary=reference_summary,
                attempts_used=int(attempt),
            )
            break

        if accepted_record is None:
            dropped += 1
            continue

        accepted.append(accepted_record)

    if dropped:
        logger.warning("Lawstress generation dropped %d/%d specs after max attempts", dropped, len(specs))
    return accepted


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_lawstress_records_jsonl(path: Path, records: Sequence[LawStressRecord]) -> None:
    write_jsonl(path, (record.to_dict() for record in records))


def load_lawstress_records_jsonl(path: Path) -> List[LawStressRecord]:
    loaded: List[LawStressRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            loaded.append(LawStressRecord.from_dict(json.loads(line)))
    return loaded


def write_benchmark_docs_jsonl(path: Path, records: Sequence[LawStressRecord]) -> None:
    write_jsonl(path, (record.to_benchmark_doc() for record in records))


def build_reference_summary_rows(records: Sequence[LawStressRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "example_id": record.example_id,
                "split": record.split,
                "reference_summary": record.reference_summary,
                "y_raw": record.y_raw,
                "y_norm": record.y_norm,
                "law_target": record.law_target,
                "family": record.family,
                "difficulty": record.difficulty,
                "bin_name": record.bin_name,
            }
        )
    return rows


__all__ = [
    "ANCHOR_REAL",
    "ANCHOR_SYNTHETIC",
    "DIFFICULTY_CONTROL",
    "DIFFICULTY_HARD",
    "FAMILIES",
    "LAW_TARGETS",
    "LawStressRecord",
    "LawStressSpec",
    "PolicyAtom",
    "RILE_BINS",
    "RILEBin",
    "build_reference_summary_rows",
    "clamp_raw_rile",
    "compute_raw_rile_from_atoms",
    "compute_segment_raw_rile",
    "denormalize_rile",
    "generate_lawstress_records",
    "generate_lawstress_specs",
    "get_rile_bin",
    "length_weighted_mean",
    "load_lawstress_records_jsonl",
    "naive_compress_text",
    "normalize_rile",
    "summarize_spec_balance",
    "write_benchmark_docs_jsonl",
    "write_jsonl",
    "write_lawstress_records_jsonl",
]
