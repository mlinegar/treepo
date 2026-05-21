"""Alternating f/g optimization trampoline for the manifesto distillation ladder.

The ladder name counts alternations starting from supplied ``(f_init, g_init)``:

- ``k = 0`` -> ``fg``       (no training; evaluate ``f_init`` on ``g_init``'s output)
- ``k = 1`` -> ``fgf``      (produce ``f1`` from ``(f_init, g_init)``; g unchanged)
- ``k = 2`` -> ``fgfg``     (produce ``g1`` from ``(f1, g_init)``; f unchanged)
- ``k = 3`` -> ``fgfgf``    (produce ``f2`` from ``(f1, g1)``)
- ``k = 4`` -> ``fgfgfg``   (produce ``g2`` from ``(f2, g1)``)
- ... alternating odd -> train f, even -> train g.

**Training signal for g**: when training ``g_k``, the scoring / reward function
is the *current* student ``f_k``, NOT the teacher or the gold expert. This is
the whole point of the alternation: f and g co-adapt. The f-vs-f* gap between
"what our f says" (internal) and "what the expert says" (external) must be
measured at every iteration to surface reward-hacking.

This module defines:
- ``FamilyRuntime``: the protocol every backend family implements.
- ``stage_name_for_iteration(k)``: maps integer k to a stage id.
- ``stage_label_for_iteration(k)``: human-readable power notation.
- ``run_alternating_family(...)``: the shared loop.
- ``IterationRecord``: the output schema.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from treepo._research.tree.state_tree import (
    state_tree_skeleton_from_labeled_tree,
    state_tree_trace_metrics,
    update_state_tree_node,
    write_state_trees_jsonl,
)

LOGGER = logging.getLogger(__name__)


def _json_write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _artifact_for_manifest(artifact: Any) -> Optional[str]:
    if artifact is None:
        return None
    return str(artifact)


def _normalize_first_train_side(value: str) -> str:
    side = str(value or "f").strip().lower()
    if side not in {"f", "g"}:
        raise ValueError(f"first_train_side must be 'f' or 'g', got {value!r}")
    return side


def stage_powers_for_iteration(
    k: int,
    *,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
) -> tuple[int, int]:
    """Return the ``(f_degree, g_degree)`` at iteration ``k``."""
    if k < 0:
        raise ValueError(f"iteration must be >= 0, got {k}")
    f_degree = int(initial_f_degree)
    g_degree = int(initial_g_degree)
    if f_degree < 0 or g_degree < 0:
        raise ValueError(
            "initial degrees must be >= 0, got "
            f"f={initial_f_degree!r} g={initial_g_degree!r}"
        )
    next_side = _normalize_first_train_side(first_train_side)
    for _ in range(int(k)):
        if next_side == "f":
            f_degree += 1
            next_side = "g"
        else:
            g_degree += 1
            next_side = "f"
    return f_degree, g_degree


def _write_step_checkpoint(
    *,
    output_dir: Path,
    family: FamilyRuntime,
    axis_kind: str,
    axis_value: int,
    leaf_count: Optional[int],
    leaf_size_tokens: Optional[int],
    iteration: int,
    stage_name: str,
    stage_label: Optional[str],
    f_degree: Optional[int],
    g_degree: Optional[int],
    trained: str,
    phase: str,
    f_artifact: Any,
    g_artifact: Any,
    iteration_dir: Optional[Path] = None,
    split_metrics: Optional[Mapping[str, "SplitMetrics"]] = None,
    error: Optional[str] = None,
    artifact_validation: Optional[Mapping[str, Any]] = None,
    trace_artifacts: Optional[Mapping[str, Any]] = None,
    trace_metrics: Optional[Mapping[str, Any]] = None,
    trace_errors: Optional[Mapping[str, Any]] = None,
) -> Path:
    checkpoints_dir = Path(output_dir) / "step_checkpoints"
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family": str(family.name),
        "axis_kind": str(axis_kind),
        "axis_value": int(axis_value),
        "leaf_count": int(leaf_count) if leaf_count is not None else None,
        "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
        "iteration": int(iteration),
        "stage_name": str(stage_name),
        "stage_label": str(stage_label) if stage_label is not None else None,
        "f_degree": int(f_degree) if f_degree is not None else None,
        "g_degree": int(g_degree) if g_degree is not None else None,
        "trained": str(trained),
        "phase": str(phase),
        "f_artifact": _artifact_for_manifest(f_artifact),
        "g_artifact": _artifact_for_manifest(g_artifact),
        "iteration_dir": str(iteration_dir) if iteration_dir is not None else None,
        "error": error,
        "artifact_validation": dict(artifact_validation or {}),
    }
    if trace_artifacts:
        payload["trace_artifacts"] = dict(trace_artifacts)
    if trace_metrics:
        payload["trace_metrics"] = dict(trace_metrics)
    if trace_errors:
        payload["trace_errors"] = dict(trace_errors)
    if split_metrics is not None:
        payload["split_metrics"] = {
            str(name): asdict(metrics) for name, metrics in split_metrics.items()
        }
    checkpoint_path = checkpoints_dir / f"iter_{int(iteration):02d}_{phase}.json"
    payload["checkpoint_path"] = str(checkpoint_path)
    _json_write_atomic(checkpoint_path, payload)
    _json_write_atomic(checkpoints_dir / "latest.json", payload)
    LOGGER.info("Wrote alternating step checkpoint %s", checkpoint_path)
    return checkpoint_path


def _legacy_stage_name_for_iteration(k: int) -> str:
    if k < 0:
        raise ValueError(f"iteration must be >= 0, got {k}")
    if k == 0:
        return "fg"
    tail = "".join("f" if i % 2 == 0 else "g" for i in range(k))
    return "fg" + tail


def stage_name_for_iteration(
    k: int,
    *,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
    naming: str = "legacy",
) -> str:
    """Map iteration ``k`` to a stage identifier.

    ``naming="legacy"`` preserves the historical ``fg -> fgf -> fgfg`` labels
    for the canonical ``(f^1, g^1)`` / f-first ladder. Any other setup falls
    back to compact power ids like ``f1g0``.
    """
    mode = str(naming or "legacy").strip().lower()
    if mode not in {"legacy", "powers"}:
        raise ValueError(f"naming must be 'legacy' or 'powers', got {naming!r}")
    side = _normalize_first_train_side(first_train_side)
    if (
        mode == "legacy"
        and side == "f"
        and int(initial_f_degree) == 1
        and int(initial_g_degree) == 1
    ):
        return _legacy_stage_name_for_iteration(k)
    f_degree, g_degree = stage_powers_for_iteration(
        k,
        first_train_side=side,
        initial_f_degree=initial_f_degree,
        initial_g_degree=initial_g_degree,
    )
    return f"f{f_degree}g{g_degree}"


def stage_label_for_iteration(
    k: int,
    *,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
) -> str:
    """Return human-readable power notation for iteration ``k``."""
    f_degree, g_degree = stage_powers_for_iteration(
        k,
        first_train_side=first_train_side,
        initial_f_degree=initial_f_degree,
        initial_g_degree=initial_g_degree,
    )
    return f"f^{f_degree} g^{g_degree}"


def trains_f_at_iteration(k: int, *, first_train_side: str = "f") -> bool:
    """Return True if iteration ``k`` trains f."""
    if k < 1:
        return False
    side = _normalize_first_train_side(first_train_side)
    return (k % 2 == 1) if side == "f" else (k % 2 == 0)


def trains_g_at_iteration(k: int, *, first_train_side: str = "f") -> bool:
    """Return True if iteration ``k`` trains g."""
    if k < 1:
        return False
    side = _normalize_first_train_side(first_train_side)
    return (k % 2 == 1) if side == "g" else (k % 2 == 0)


@runtime_checkable
class FamilyRuntime(Protocol):
    """Contract every alternating-optimization backend family must satisfy.

    Families own their artifact types: ``FArtifact`` / ``GArtifact`` are opaque
    Any handles that the family constructs and consumes. The trampoline only
    threads them through.
    """

    #: Short family name, e.g. ``"dspy"`` / ``"trl"`` / ``"fno"``.
    name: str

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """Train f for one iteration. Returns the new f artifact."""
        ...

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """Train g for one iteration. Returns the new g artifact.

        ``f`` here is the *current* student f; the family must use it as the
        scoring / reward signal for g training, NOT the teacher.
        """
        ...

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> List[Optional[float]]:
        """Apply ``f(g(tree_root))`` for each tree; return 1-7 predictions.

        Returns a list aligned with ``trees``; entries may be ``None`` for
        trees the family cannot score (missing text, failed inference, etc.).
        """
        ...

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        """Raise if a returned artifact cannot be reloaded for a future step."""
        ...


@runtime_checkable
class BundleAwareFamilyRuntime(Protocol):
    """Optional extension surface for the unified TreeBundle v1 ladder.

    Families that implement this protocol can be plugged directly into the
    unified runner CLI via ``--family <name>`` without per-family CLI plumbing.

    Existing families that do not implement these methods continue to work via
    the helper-fallback path below (``family_default_g`` etc.). New families
    (oracles, sketches, CTreePO native) should implement this protocol.
    """

    @property
    def default_f(self) -> str:
        """Canonical f initializer spec, e.g. ``"raw"`` / ``"identity"`` /
        ``"oracle:hll_exact"`` / ``"external_passthrough"``."""
        ...

    @property
    def default_g(self) -> str:
        """Canonical g initializer spec.

        MUST be ``"raw_concat"`` for non-lossy families. Lossy-native sketch
        families (e.g. HLL register-max) must declare ``"raw_concat"`` here as
        well and resolve it to a ConcatSketch-equivalent wrapper internally;
        the lossy native merge is then a named opt-in
        (``"oracle:hll_max_merge"``).
        """
        ...

    def expected_bundle(self) -> Mapping[str, Any]:
        """Required ``TreeBundleManifest`` field constraints for this family.

        Returns a mapping of fields the family checks at construction time
        against the loaded bundle, e.g.::

            {"leaf_unit": "text_token",
             "domain": ("manifesto_rile",),
             "state_dim_min": 2 * summary_dim}

        Tuple values mean any-of; scalar values mean exact match. The runner
        uses this to fail loud on bundle/family mismatch.
        """
        ...

    def supported_inits(self) -> Mapping[str, frozenset[str]]:
        """Whitelist of ``<init-spec>`` prefixes the family accepts.

        Keyed by ``"f"`` and ``"g"``; values are frozensets of accepted
        init-spec prefixes (``"identity"``, ``"raw"``, ``"raw_concat"``,
        ``"oracle"``, ``"artifact"``, ``"external_passthrough"``, etc.).
        """
        ...

    def resolve_init(self, *, kind: str, spec: str) -> Any:
        """Turn an ``<init-spec>`` string into an opaque artifact handle.

        ``kind`` is ``"f"`` or ``"g"``; ``spec`` is the user-supplied string
        (e.g. ``"oracle:hll_exact"`` or ``"artifact:/path/to/f.json"``). The
        returned value is fed back to ``train_f``/``train_g``/
        ``score_roots_with_f`` as the family's opaque artifact.
        """
        ...

    def share_state_axes(self) -> frozenset[str]:
        """Axes whose state is shared across f and g.

        FNO returns ``frozenset({"f", "g"})`` because it uses one state_dict
        for both. Most families return ``frozenset()``. The runner uses this
        to decide whether ``f_artifact`` and ``g_artifact`` can be unified
        when persisting checkpoints.
        """
        ...


# Init-spec grammar accepted by ``parse_init_spec`` and ``family_resolve_init``.
INIT_SPEC_RAW_PREFIXES: frozenset[str] = frozenset(
    {
        "identity",
        "raw",
        "raw_concat",
        "external_passthrough",
        "pretuned_scorer",
        "bare_scorer",
        # Legacy alias kept for one release; emits a DeprecationWarning at
        # parse time when seen as the *resolved* default.
        "teacher_passthrough",
    }
)


@dataclass(frozen=True)
class InitSpec:
    """Parsed ``<init-spec>`` value.

    ``kind`` is ``"sentinel"`` (raw/identity/raw_concat/etc.),
    ``"oracle"``, or ``"artifact"``. ``value`` is the bare token for
    sentinels, the oracle name for oracle, or the artifact path for artifact.
    """

    kind: str
    value: str

    @property
    def raw(self) -> str:
        if self.kind == "sentinel":
            return self.value
        return f"{self.kind}:{self.value}"


def parse_init_spec(spec: str | None) -> Optional[InitSpec]:
    """Parse a user-supplied init-spec string into an ``InitSpec``.

    Returns ``None`` for ``None``/empty input so callers can fall back to a
    family default. Raises ``ValueError`` for malformed inputs.
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    if ":" in text:
        prefix, _, value = text.partition(":")
        prefix = prefix.strip().lower()
        value = value.strip()
        if not value:
            raise ValueError(f"init-spec missing value after prefix: {spec!r}")
        if prefix in {"oracle", "artifact"}:
            return InitSpec(kind=prefix, value=value)
        raise ValueError(
            f"unknown init-spec prefix {prefix!r} in {spec!r}; "
            "expected 'oracle:<name>' or 'artifact:<path>'"
        )
    bare = text.lower()
    if bare not in INIT_SPEC_RAW_PREFIXES:
        raise ValueError(
            f"unknown init-spec sentinel {bare!r}; expected one of "
            f"{sorted(INIT_SPEC_RAW_PREFIXES)} or 'oracle:<name>'/'artifact:<path>'"
        )
    return InitSpec(kind="sentinel", value=bare)


# ---------------------------------------------------------------------------
# Optional bundle-aware helpers.
#
# Existing FamilyRuntime implementations may not implement BundleAwareFamilyRuntime;
# these helpers fall back to sensible defaults via ``getattr`` so the runner
# can call them uniformly without breaking legacy families.
# ---------------------------------------------------------------------------

_DEFAULT_FALLBACK_F: str = "identity"
_DEFAULT_FALLBACK_G: str = "raw_concat"


def family_default_f(family: Any) -> str:
    """Return the family's canonical f initializer, or a fallback sentinel.

    Looks up the ``default_f`` property; falls back to ``"identity"`` if the
    family doesn't implement BundleAwareFamilyRuntime.
    """
    value = getattr(family, "default_f", None)
    if value is None:
        return _DEFAULT_FALLBACK_F
    return str(value)


def family_default_g(family: Any) -> str:
    """Return the family's canonical g initializer, or a fallback sentinel.

    Looks up the ``default_g`` property; falls back to ``"raw_concat"`` if
    the family doesn't implement BundleAwareFamilyRuntime. The fallback
    matches the universal invariant: raw concat must always be the no-op
    default unless the family explicitly opts out.
    """
    value = getattr(family, "default_g", None)
    if value is None:
        return _DEFAULT_FALLBACK_G
    return str(value)


def family_expected_bundle(family: Any) -> Mapping[str, Any]:
    """Return the family's expected TreeBundleManifest constraints, or {}."""
    method = getattr(family, "expected_bundle", None)
    if not callable(method):
        return {}
    result = method()
    if not isinstance(result, Mapping):
        return {}
    return result


def family_supported_inits(family: Any) -> Mapping[str, frozenset[str]]:
    """Return the family's supported ``<init-spec>`` prefixes, or wildcard.

    When a family doesn't implement ``supported_inits``, returns the full
    set of known prefixes (so ``family_resolve_init`` defers fully to the
    family's own validation).
    """
    method = getattr(family, "supported_inits", None)
    if not callable(method):
        wide = INIT_SPEC_RAW_PREFIXES | frozenset({"oracle", "artifact"})
        return {"f": wide, "g": wide}
    result = method()
    if not isinstance(result, Mapping):
        return {}
    return {str(k): frozenset(v) for k, v in result.items()}


def family_resolve_init(family: Any, *, kind: str, spec: str | None) -> Any:
    """Resolve an ``<init-spec>`` for a family, with helper fallback.

    If the family implements ``resolve_init``, defers to it. Otherwise
    returns the parsed ``InitSpec`` (or ``None`` for empty input) so the
    family's existing constructor logic can handle the value via its legacy
    artifact-string accepting paths.
    """
    parsed = parse_init_spec(spec)
    method = getattr(family, "resolve_init", None)
    if callable(method):
        if parsed is None:
            return None
        return method(kind=str(kind), spec=parsed.raw)
    return parsed


def family_share_state_axes(family: Any) -> frozenset[str]:
    """Return axes whose state is shared between f and g.

    Falls back to ``frozenset()`` (no sharing) for families that don't
    implement ``share_state_axes``. FNO returns ``{"f","g"}``.
    """
    method = getattr(family, "share_state_axes", None)
    if not callable(method):
        return frozenset()
    result = method()
    try:
        return frozenset(result)
    except TypeError:
        return frozenset()


def _validate_family_artifact(
    family: FamilyRuntime,
    *,
    kind: str,
    artifact: Any,
) -> Optional[str]:
    validator = getattr(family, "validate_artifact", None)
    if not callable(validator):
        return None
    validator(kind=str(kind), artifact=artifact)
    return "passed"


@dataclass
class IterationRecord:
    """One row of the alternating-optimization output history."""

    iteration: int
    stage_name: str          # "fg", "fgf", "fgfg", ...
    family: str
    trained: str             # "none", "f", "g"
    stage_label: Optional[str] = None
    f_degree: Optional[int] = None
    g_degree: Optional[int] = None
    axis_kind: str = "leaf_count"
    axis_value: int = 0
    leaf_count: Optional[int] = None
    leaf_size_tokens: Optional[int] = None
    f_artifact: Optional[str] = None
    g_artifact: Optional[str] = None
    #: Per-split metrics. Keys are split names ("train", "val", "test", "all").
    split_metrics: Dict[str, "SplitMetrics"] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitMetrics:
    """Pearson + MAE vs teacher-f (internal) and vs gold expert (external)."""

    n: int
    #: Internal: how closely current f agrees with the teacher f at the root.
    internal_f_pearson: Optional[float] = None
    internal_f_mae: Optional[float] = None
    internal_f_mae_1_7: Optional[float] = None
    #: External: Pearson / MAE vs gold expert score (the paper-facing metric).
    external_expert_pearson: Optional[float] = None
    external_expert_mae: Optional[float] = None
    external_expert_mae_1_7: Optional[float] = None
    #: ``internal_f_pearson - external_expert_pearson``. Positive = f is
    #: drifting from expert signal while still agreeing with the teacher.
    f_star_gap: Optional[float] = None
    mean_prediction: Optional[float] = None
    mean_teacher: Optional[float] = None
    mean_expert: Optional[float] = None
    mean_prediction_1_7: Optional[float] = None
    mean_teacher_1_7: Optional[float] = None
    mean_expert_1_7: Optional[float] = None
    metrics_scale: Optional[str] = None
    #: Optional vector-task breakdown. When a family exposes
    #: ``score_roots_with_f_by_dimension``, the scalar fields above are macro
    #: averages over these per-dimension metric dictionaries.
    per_dimension: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _pearson_and_mae(
    preds: Sequence[Optional[float]],
    truths: Sequence[Optional[float]],
) -> Dict[str, Optional[float]]:
    """Compute Pearson r + MAE on paired lists, dropping entries with None on either side."""
    from treepo._research.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r

    paired = [
        (float(p), float(t))
        for p, t in zip(preds, truths)
        if p is not None and t is not None
    ]
    if len(paired) < 4:
        return {
            "n": len(paired),
            "pearson_r": None,
            "mae": None,
            "mae_1_7": None,
            "mean_prediction": float(sum(p for p, _ in paired) / len(paired)) if paired else None,
            "mean_truth": float(sum(t for _, t in paired) / len(paired)) if paired else None,
            "mean_prediction_1_7": float(sum(p for p, _ in paired) / len(paired)) if paired else None,
            "mean_truth_1_7": float(sum(t for _, t in paired) / len(paired)) if paired else None,
        }
    ps, ts = zip(*paired)
    corr = compute_corpus_pearson_r(ps, ts).as_dict()
    mae = sum(abs(p - t) for p, t in paired) / len(paired)
    return {
        "n": len(paired),
        "pearson_r": corr.get("pearson_r"),
        "mae": float(mae),
        "mae_1_7": float(mae),
        "mean_prediction": float(sum(ps) / len(ps)),
        "mean_truth": float(sum(ts) / len(ts)),
        "mean_prediction_1_7": float(sum(ps) / len(ps)),
        "mean_truth_1_7": float(sum(ts) / len(ts)),
    }


def _tree_split(tree: Any) -> str:
    metadata = getattr(tree, "metadata", None) or {}
    return str(metadata.get("split") or "unknown").lower()


def _teacher_root_score(tree: Any) -> Optional[float]:
    """Extract the teacher f's root score from a LabeledTree."""
    metadata = getattr(tree, "metadata", None) or {}
    native = _safe_float(metadata.get("teacher_score_native"))
    if native is not None:
        return native
    root_level = getattr(tree, "levels", None) or []
    if root_level:
        for node_id in reversed(root_level[-1]):
            node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
            if node is not None:
                s = _safe_float(getattr(node, "score", None))
                if s is not None:
                    return s
                nm = node.metadata or {}
                return _safe_float(nm.get("teacher_score_1_7"))
    return _safe_float(metadata.get("teacher_score_1_7") or getattr(tree, "document_score", None))


def _expert_root_score(tree: Any) -> Optional[float]:
    metadata = getattr(tree, "metadata", None) or {}
    native = _safe_float(metadata.get("expert_score_native"))
    if native is not None:
        return native
    objective = _safe_float(metadata.get("expert_score_for_objective"))
    if objective is not None:
        return objective
    return _safe_float(metadata.get("expert_score_1_7"))


def _metrics_scale(trees: Sequence[Any]) -> str:
    for tree in trees:
        metadata = getattr(tree, "metadata", None) or {}
        value = metadata.get("expert_target_scale") if isinstance(metadata, Mapping) else None
        if value:
            return str(value)
    return "normalized_1_7"


def _root_dimension_scores(tree: Any, *, source: str) -> Dict[str, float]:
    """Extract root-level vector scores from a labeled tree.

    ``source`` is either ``"teacher"`` or ``"expert"``. Teacher scores live
    primarily on the root node's ``dimension_scores``; expert scores live in
    tree metadata because only roots have human/expert labels.
    """
    metadata = getattr(tree, "metadata", None) or {}
    keys = (
        ("teacher_dimension_scores_1_7", "dimension_scores")
        if source == "teacher"
        else ("expert_dimension_scores_1_7", "expert_means")
    )
    out: Dict[str, float] = {}
    for key in keys:
        value = metadata.get(key) if isinstance(metadata, Mapping) else None
        if isinstance(value, Mapping):
            for dim, raw in value.items():
                score = _safe_float(raw)
                if score is not None:
                    out[str(dim)] = float(score)
            if out:
                return out

    if source == "teacher":
        root_level = getattr(tree, "levels", None) or []
        if root_level:
            for node_id in reversed(root_level[-1]):
                node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                if node is None:
                    continue
                node_scores = getattr(node, "dimension_scores", None)
                if isinstance(node_scores, Mapping):
                    for dim, raw in node_scores.items():
                        score = _safe_float(raw)
                        if score is not None:
                            out[str(dim)] = float(score)
                    if out:
                        return out
                node_meta = getattr(node, "metadata", None) or {}
                meta_scores = node_meta.get("teacher_dimension_scores_1_7")
                if isinstance(meta_scores, Mapping):
                    for dim, raw in meta_scores.items():
                        score = _safe_float(raw)
                        if score is not None:
                            out[str(dim)] = float(score)
                    if out:
                        return out
    return out


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _tree_prediction_base_record(index: int, tree: Any) -> Dict[str, Any]:
    metadata = getattr(tree, "metadata", None) or {}
    doc_id = getattr(tree, "doc_id", None) or metadata.get("manifesto_id") or metadata.get("doc_id")
    base: Dict[str, Any] = {
        "index": int(index),
        "doc_id": str(doc_id) if doc_id is not None else None,
        "split": _tree_split(tree),
    }
    for key in (
        "manifesto_id",
        "benoit_manifesto_key",
        "party_id",
        "party_abbrev",
        "country_name",
        "year",
    ):
        value = metadata.get(key) if isinstance(metadata, Mapping) else None
        if value is not None:
            base[key] = value
    return base


def _write_prediction_records(path: Optional[Path], records: Sequence[Mapping[str, Any]]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    tmp_path.replace(path)


def _artifact_suffix(value: str) -> str:
    text = str(value or "").strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in out.split("_") if part) or "unknown"


def _fallback_vector_root_traces(
    *,
    family: FamilyRuntime,
    f: Any,
    g: Any,
    trees: Sequence[Any],
) -> Dict[str, List[Any]]:
    scorer = getattr(family, "score_roots_with_f_by_dimension", None)
    if not callable(scorer):
        return {}
    preds_by_dim = scorer(f=f, g=g, trees=list(trees))
    if not isinstance(preds_by_dim, Mapping) or not preds_by_dim:
        return {}
    out: Dict[str, List[Any]] = {}
    for dim, preds in sorted(dict(preds_by_dim).items(), key=lambda item: str(item[0])):
        if not isinstance(preds, Sequence):
            continue
        dim_key = str(dim)
        traces: List[Any] = []
        for idx, tree in enumerate(trees):
            trace = state_tree_skeleton_from_labeled_tree(
                tree,
                method_family=str(getattr(family, "name", "family")),
                state_kind=f"root_score_{_artifact_suffix(dim_key)}",
                split=_tree_split(tree),
            )
            prediction = _safe_float(preds[idx]) if idx < len(preds) else None
            target = _root_dimension_scores(tree, source="teacher").get(dim_key)
            metadata: Dict[str, Any] = {
                "dimension": dim_key,
                "state_kind": f"root_score_{_artifact_suffix(dim_key)}",
                "law_channel": "root",
                "observed": False,
                "sampled": False,
                "propensity": 0.0,
            }
            if prediction is not None:
                metadata["prediction"] = float(prediction)
                metadata["readout_prediction"] = float(prediction)
            if target is not None:
                metadata["target"] = float(target)
                metadata["proxy_loss"] = (
                    None
                    if prediction is None
                    else float((float(prediction) - float(target)) ** 2)
                )
            update_state_tree_node(trace, trace.root.id, metadata=metadata)
            traces.append(trace)
        out[dim_key] = traces
    return out


def export_ladder_full_tree_traces(
    *,
    family: FamilyRuntime,
    f: Any,
    g: Any,
    tree_sets: Mapping[str, Sequence[Any]],
    output_dir: Path,
    iteration: int,
) -> Dict[str, Any]:
    """Persist full-tree traces for one ladder iteration.

    Scalar families expose ``full_tree_traces_with_f_g``. Vector families may
    expose ``full_tree_traces_with_f_g_by_dimension`` and are written one
    scalar dimension per JSONL file.
    """

    trace_dir = Path(output_dir) / "full_tree_traces"
    artifacts: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    vector_trace_fn = getattr(family, "full_tree_traces_with_f_g_by_dimension", None)
    vector_score_fn = getattr(family, "score_roots_with_f_by_dimension", None)
    scalar_trace_fn = getattr(family, "full_tree_traces_with_f_g", None)
    if (
        not callable(vector_trace_fn)
        and not callable(vector_score_fn)
        and not callable(scalar_trace_fn)
    ):
        return {"artifacts": artifacts, "metrics": metrics, "errors": errors}

    for split_name, raw_trees in dict(tree_sets).items():
        trees = list(raw_trees or ())
        split_key = _artifact_suffix(str(split_name))
        if callable(vector_trace_fn):
            try:
                by_dim = vector_trace_fn(f=f, g=g, trees=trees)
            except Exception as exc:
                errors[split_key] = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "Failed to export vector full-tree traces for %s iteration %d split %s: %s",
                    getattr(family, "name", "family"),
                    int(iteration),
                    split_key,
                    exc,
                )
                continue
            if isinstance(by_dim, Mapping) and by_dim:
                split_metrics: Dict[str, Any] = {}
                for dim, traces in sorted(dict(by_dim).items(), key=lambda item: str(item[0])):
                    dim_key = _artifact_suffix(str(dim))
                    trace_list = list(traces or ())
                    stem = f"iter_{int(iteration):02d}_{split_key}_{dim_key}"
                    trace_path = trace_dir / f"{stem}.jsonl"
                    metrics_path = trace_dir / f"{stem}_metrics.json"
                    write_state_trees_jsonl(trace_list, trace_path)
                    trace_metrics = state_tree_trace_metrics(trace_list)
                    _json_write_atomic(metrics_path, trace_metrics)
                    artifacts[f"full_tree_traces_iter_{int(iteration):02d}_{split_key}_{dim_key}_jsonl"] = str(trace_path)
                    artifacts[f"full_tree_metrics_iter_{int(iteration):02d}_{split_key}_{dim_key}_json"] = str(metrics_path)
                    split_metrics[dim_key] = trace_metrics
                metrics[split_key] = split_metrics
                continue

        if callable(vector_score_fn) and not callable(scalar_trace_fn):
            try:
                by_dim = _fallback_vector_root_traces(
                    family=family,
                    f=f,
                    g=g,
                    trees=trees,
                )
            except Exception as exc:
                errors[split_key] = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "Failed to export fallback vector full-tree traces for %s iteration %d split %s: %s",
                    getattr(family, "name", "family"),
                    int(iteration),
                    split_key,
                    exc,
                )
                continue
            if by_dim:
                split_metrics = {}
                for dim, traces in sorted(dict(by_dim).items(), key=lambda item: str(item[0])):
                    dim_key = _artifact_suffix(str(dim))
                    trace_list = list(traces or ())
                    stem = f"iter_{int(iteration):02d}_{split_key}_{dim_key}"
                    trace_path = trace_dir / f"{stem}.jsonl"
                    metrics_path = trace_dir / f"{stem}_metrics.json"
                    write_state_trees_jsonl(trace_list, trace_path)
                    trace_metrics = state_tree_trace_metrics(trace_list)
                    _json_write_atomic(metrics_path, trace_metrics)
                    artifacts[f"full_tree_traces_iter_{int(iteration):02d}_{split_key}_{dim_key}_jsonl"] = str(trace_path)
                    artifacts[f"full_tree_metrics_iter_{int(iteration):02d}_{split_key}_{dim_key}_json"] = str(metrics_path)
                    split_metrics[dim_key] = trace_metrics
                metrics[split_key] = split_metrics
                continue

        if not callable(scalar_trace_fn):
            continue
        try:
            trace_list = list(scalar_trace_fn(f=f, g=g, trees=trees) or ())
        except Exception as exc:
            errors[split_key] = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "Failed to export full-tree traces for %s iteration %d split %s: %s",
                getattr(family, "name", "family"),
                int(iteration),
                split_key,
                exc,
            )
            continue
        stem = f"iter_{int(iteration):02d}_{split_key}"
        trace_path = trace_dir / f"{stem}.jsonl"
        metrics_path = trace_dir / f"{stem}_metrics.json"
        write_state_trees_jsonl(trace_list, trace_path)
        trace_metrics = state_tree_trace_metrics(trace_list)
        _json_write_atomic(metrics_path, trace_metrics)
        artifacts[f"full_tree_traces_iter_{int(iteration):02d}_{split_key}_jsonl"] = str(trace_path)
        artifacts[f"full_tree_metrics_iter_{int(iteration):02d}_{split_key}_json"] = str(metrics_path)
        metrics[split_key] = trace_metrics
    return {"artifacts": artifacts, "metrics": metrics, "errors": errors}


def _evaluate_vector_iteration(
    *,
    family: FamilyRuntime,
    f: Any,
    g: Any,
    trees: Sequence[Any],
    splits: Sequence[str],
    prediction_records_path: Optional[Path] = None,
) -> Optional[Dict[str, SplitMetrics]]:
    scorer = getattr(family, "score_roots_with_f_by_dimension", None)
    if not callable(scorer):
        return None
    preds_by_dim = scorer(f=f, g=g, trees=list(trees))
    if not isinstance(preds_by_dim, Mapping) or not preds_by_dim:
        return None
    dims = sorted(str(dim) for dim in preds_by_dim)
    for dim in dims:
        values = preds_by_dim.get(dim)
        if not isinstance(values, Sequence) or len(values) != len(trees):
            raise RuntimeError(
                f"{family.name}.score_roots_with_f_by_dimension[{dim!r}] returned "
                f"{len(values) if isinstance(values, Sequence) else 'non-sequence'} "
                f"predictions for {len(trees)} trees"
            )

    tree_splits = [_tree_split(t) for t in trees]
    teacher_by_tree = [_root_dimension_scores(t, source="teacher") for t in trees]
    expert_by_tree = [_root_dimension_scores(t, source="expert") for t in trees]
    metrics_scale = _metrics_scale(trees)

    if prediction_records_path is not None:
        prediction_records: List[Dict[str, Any]] = []
        for i, tree in enumerate(trees):
            base = _tree_prediction_base_record(i, tree)
            for dim in dims:
                record = dict(base)
                record.update(
                    {
                        "dimension": dim,
                        "prediction_1_7": _safe_float(preds_by_dim[dim][i]),
                        "prediction": _safe_float(preds_by_dim[dim][i]),
                        "teacher_score_1_7": teacher_by_tree[i].get(dim),
                        "teacher_score": teacher_by_tree[i].get(dim),
                        "expert_score_1_7": expert_by_tree[i].get(dim),
                        "expert_score": expert_by_tree[i].get(dim),
                        "metrics_scale": metrics_scale,
                    }
                )
                prediction_records.append(record)
        _write_prediction_records(prediction_records_path, prediction_records)

    out: Dict[str, SplitMetrics] = {}
    for split in splits:
        if split == "all":
            idxs = list(range(len(trees)))
        else:
            idxs = [i for i, s in enumerate(tree_splits) if s == split.lower()]
        if not idxs:
            out[split] = SplitMetrics(n=0)
            continue

        per_dim: Dict[str, Dict[str, Optional[float]]] = {}
        internal_rs: List[Optional[float]] = []
        external_rs: List[Optional[float]] = []
        internal_maes: List[Optional[float]] = []
        external_maes: List[Optional[float]] = []
        pred_means: List[Optional[float]] = []
        teacher_means: List[Optional[float]] = []
        expert_means: List[Optional[float]] = []
        paired_counts: List[int] = []
        for dim in dims:
            split_preds = [preds_by_dim[dim][i] for i in idxs]
            split_teacher = [teacher_by_tree[i].get(dim) for i in idxs]
            split_expert = [expert_by_tree[i].get(dim) for i in idxs]
            internal = _pearson_and_mae(split_preds, split_teacher)
            external = _pearson_and_mae(split_preds, split_expert)
            gap: Optional[float] = None
            if internal["pearson_r"] is not None and external["pearson_r"] is not None:
                gap = float(internal["pearson_r"]) - float(external["pearson_r"])
            per_dim[dim] = {
                "n_internal": internal["n"],
                "n_external": external["n"],
                "internal_f_pearson": internal["pearson_r"],
                "internal_f_mae": internal["mae"],
                "internal_f_mae_1_7": internal["mae_1_7"],
                "external_expert_pearson": external["pearson_r"],
                "external_expert_mae": external["mae"],
                "external_expert_mae_1_7": external["mae_1_7"],
                "f_star_gap": gap,
                "mean_prediction": internal["mean_prediction"],
                "mean_teacher": internal["mean_truth"],
                "mean_expert": external["mean_truth"],
                "mean_prediction_1_7": internal["mean_prediction_1_7"],
                "mean_teacher_1_7": internal["mean_truth_1_7"],
                "mean_expert_1_7": external["mean_truth_1_7"],
                "metrics_scale": metrics_scale,
            }
            paired_counts.append(int(external["n"]))
            internal_rs.append(internal["pearson_r"])
            external_rs.append(external["pearson_r"])
            internal_maes.append(internal["mae_1_7"])
            external_maes.append(external["mae_1_7"])
            pred_means.append(internal["mean_prediction_1_7"])
            teacher_means.append(internal["mean_truth_1_7"])
            expert_means.append(external["mean_truth_1_7"])

        macro_internal = _mean_optional(internal_rs)
        macro_external = _mean_optional(external_rs)
        macro_gap: Optional[float] = None
        if macro_internal is not None and macro_external is not None:
            macro_gap = float(macro_internal) - float(macro_external)
        out[split] = SplitMetrics(
            n=int(max(paired_counts) if paired_counts else len(idxs)),
            internal_f_pearson=macro_internal,
            internal_f_mae=_mean_optional(internal_maes),
            internal_f_mae_1_7=_mean_optional(internal_maes),
            external_expert_pearson=macro_external,
            external_expert_mae=_mean_optional(external_maes),
            external_expert_mae_1_7=_mean_optional(external_maes),
            f_star_gap=macro_gap,
            mean_prediction=_mean_optional(pred_means),
            mean_teacher=_mean_optional(teacher_means),
            mean_expert=_mean_optional(expert_means),
            mean_prediction_1_7=_mean_optional(pred_means),
            mean_teacher_1_7=_mean_optional(teacher_means),
            mean_expert_1_7=_mean_optional(expert_means),
            metrics_scale=metrics_scale,
            per_dimension=per_dim,
        )
    return out


def evaluate_iteration(
    *,
    family: FamilyRuntime,
    f: Any,
    g: Any,
    trees: Sequence[Any],
    splits: Sequence[str] = ("all", "train", "val", "test"),
    prediction_records_path: Optional[Path] = None,
) -> Dict[str, SplitMetrics]:
    """Produce the per-split metric dict for one iteration."""
    vector_metrics = _evaluate_vector_iteration(
        family=family,
        f=f,
        g=g,
        trees=trees,
        splits=splits,
        prediction_records_path=prediction_records_path,
    )
    if vector_metrics is not None:
        return vector_metrics

    preds = family.score_roots_with_f(f=f, g=g, trees=list(trees))
    if len(preds) != len(trees):
        raise RuntimeError(
            f"{family.name}.score_roots_with_f returned {len(preds)} predictions "
            f"for {len(trees)} trees"
        )
    tree_splits = [_tree_split(t) for t in trees]
    teacher_scores = [_teacher_root_score(t) for t in trees]
    expert_scores = [_expert_root_score(t) for t in trees]
    metrics_scale = _metrics_scale(trees)

    if prediction_records_path is not None:
        prediction_records = []
        is_native_raw = metrics_scale == "raw_benoit"
        for i, tree in enumerate(trees):
            record = _tree_prediction_base_record(i, tree)
            record.update(
                {
                    "prediction_1_7": None if is_native_raw else _safe_float(preds[i]),
                    "prediction": _safe_float(preds[i]),
                    "prediction_native": _safe_float(preds[i]) if is_native_raw else None,
                    "teacher_score_1_7": None if is_native_raw else teacher_scores[i],
                    "teacher_score": teacher_scores[i],
                    "expert_score_1_7": None if is_native_raw else expert_scores[i],
                    "expert_score": expert_scores[i],
                    "metrics_scale": metrics_scale,
                }
            )
            prediction_records.append(record)
        _write_prediction_records(prediction_records_path, prediction_records)

    out: Dict[str, SplitMetrics] = {}
    for split in splits:
        if split == "all":
            idxs = list(range(len(trees)))
        else:
            idxs = [i for i, s in enumerate(tree_splits) if s == split.lower()]
        if not idxs:
            out[split] = SplitMetrics(n=0)
            continue
        split_preds = [preds[i] for i in idxs]
        split_teacher = [teacher_scores[i] for i in idxs]
        split_expert = [expert_scores[i] for i in idxs]
        internal = _pearson_and_mae(split_preds, split_teacher)
        external = _pearson_and_mae(split_preds, split_expert)
        is_one_to_seven = metrics_scale != "raw_benoit"
        gap: Optional[float] = None
        if internal["pearson_r"] is not None and external["pearson_r"] is not None:
            gap = float(internal["pearson_r"]) - float(external["pearson_r"])
        out[split] = SplitMetrics(
            n=int(internal["n"]),
            internal_f_pearson=internal["pearson_r"],
            internal_f_mae=internal["mae"],
            internal_f_mae_1_7=internal["mae_1_7"] if is_one_to_seven else None,
            external_expert_pearson=external["pearson_r"],
            external_expert_mae=external["mae"],
            external_expert_mae_1_7=external["mae_1_7"] if is_one_to_seven else None,
            f_star_gap=gap,
            mean_prediction=internal["mean_prediction"],
            mean_teacher=internal["mean_truth"],
            mean_expert=external["mean_truth"],
            mean_prediction_1_7=internal["mean_prediction_1_7"] if is_one_to_seven else None,
            mean_teacher_1_7=internal["mean_truth_1_7"] if is_one_to_seven else None,
            mean_expert_1_7=external["mean_truth_1_7"] if is_one_to_seven else None,
            metrics_scale=metrics_scale,
        )
    return out


def run_alternating_family(
    *,
    family: FamilyRuntime,
    f_init: Any,
    g_init: Any,
    traces: Sequence[Any],
    eval_trees: Sequence[Any],
    max_iterations: int,
    axis_value: int,
    output_dir: Path,
    axis_kind: str = "leaf_count",
    leaf_count: Optional[int] = None,
    leaf_size_tokens: Optional[int] = None,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
    stage_naming: str = "legacy",
    artifact_namer: Optional[Callable[[str, int], Optional[str]]] = None,
) -> List[IterationRecord]:
    """Run the alternating loop for one ``(family, axis_value)`` row.

    Returns one ``IterationRecord`` per ``k in {0, 1, ..., max_iterations}``.

    ``artifact_namer(kind, iteration)`` lets callers customize on-disk artifact
    paths; when ``None``, artifacts inherit the family's own conventions.
    """
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_train_side = _normalize_first_train_side(first_train_side)

    records: List[IterationRecord] = []
    f_current: Any = f_init
    g_current: Any = g_init

    for k in range(0, max_iterations + 1):
        stage = stage_name_for_iteration(
            k,
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
            naming=stage_naming,
        )
        stage_label = stage_label_for_iteration(
            k,
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
        )
        f_degree, g_degree = stage_powers_for_iteration(
            k,
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
        )
        axis_label = (
            f"leaf{int(axis_value):04d}tok"
            if str(axis_kind) == "leaf_size_tokens"
            else f"leaf_{int(axis_value):03d}"
        )
        trains_f = trains_f_at_iteration(k, first_train_side=first_train_side)
        trains_g = trains_g_at_iteration(k, first_train_side=first_train_side)
        train_side = "f" if trains_f else "g" if trains_g else "none"
        LOGGER.info(
            "[%s %s] iteration %d (%s / %s)", family.name, axis_label, k, stage, stage_label
        )
        trained = "none"
        error: Optional[str] = None
        iter_dir: Optional[Path] = None
        artifact_validation: Dict[str, Any] = {}
        try:
            if trains_f:
                iter_dir = output_dir / f"iter_{k:02d}_train_f"
                iter_dir.mkdir(parents=True, exist_ok=True)
                f_current = family.train_f(
                    f_init=f_current,
                    g=g_current,
                    traces=traces,
                    output_dir=iter_dir,
                    iteration=k,
                )
                trained = "f"
                artifact_validation["f"] = _validate_family_artifact(
                    family, kind="f", artifact=f_current
                )
            elif trains_g:
                iter_dir = output_dir / f"iter_{k:02d}_train_g"
                iter_dir.mkdir(parents=True, exist_ok=True)
                g_current = family.train_g(
                    g_init=g_current,
                    f=f_current,
                    traces=traces,
                    output_dir=iter_dir,
                    iteration=k,
                )
                trained = "g"
                artifact_validation["g"] = _validate_family_artifact(
                    family, kind="g", artifact=g_current
                )
        except NotImplementedError as exc:
            LOGGER.warning(
                "[%s %s] iteration %d train_%s not implemented: %s",
                family.name, axis_label, k, train_side, exc,
            )
            error = f"NotImplementedError: {exc}"
            trained = "skipped"
        except RuntimeError as exc:
            LOGGER.warning(
                "[%s %s] iteration %d train_%s hard-error: %s",
                family.name, axis_label, k, train_side, exc,
            )
            error = f"RuntimeError: {exc}"
            trained = "skipped"
        except Exception as exc:
            LOGGER.exception(
                "[%s %s] iteration %d train/validate_%s failed",
                family.name,
                axis_label,
                k,
                train_side,
            )
            error = f"{type(exc).__name__}: {exc}"

        _write_step_checkpoint(
            output_dir=output_dir,
            family=family,
            axis_kind=axis_kind,
            axis_value=int(axis_value),
            leaf_count=leaf_count,
            leaf_size_tokens=leaf_size_tokens,
            iteration=k,
            stage_name=stage,
            stage_label=stage_label,
            f_degree=f_degree,
            g_degree=g_degree,
            trained=trained,
            phase="post_train",
            f_artifact=f_current,
            g_artifact=g_current,
            iteration_dir=iter_dir,
            error=error,
            artifact_validation=artifact_validation,
        )

        if error is None:
            try:
                split_metrics = evaluate_iteration(
                    family=family,
                    f=f_current,
                    g=g_current,
                    trees=eval_trees,
                    prediction_records_path=(
                        output_dir / "prediction_records" / f"iter_{k:02d}_post_eval.jsonl"
                    ),
                )
            except NotImplementedError as exc:
                LOGGER.warning(
                    "[%s %s] iteration %d evaluation not implemented: %s",
                    family.name, axis_label, k, exc,
                )
                split_metrics = {}
                error = error or f"evaluation NotImplementedError: {exc}"
            except Exception as exc:
                LOGGER.exception(
                    "[%s %s] iteration %d evaluation failed after %s training",
                    family.name,
                    axis_label,
                    k,
                    trained,
                )
                split_metrics = {}
                error = error or f"evaluation {type(exc).__name__}: {exc}"
        else:
            split_metrics = {}

        trace_export = export_ladder_full_tree_traces(
            family=family,
            f=f_current,
            g=g_current,
            tree_sets={"train": list(traces), "eval": list(eval_trees)},
            output_dir=output_dir,
            iteration=k,
        )
        extra: Dict[str, Any] = {}
        if error is not None:
            extra["error"] = error
        if trace_export.get("artifacts"):
            extra["trace_artifacts"] = dict(trace_export.get("artifacts") or {})
        if trace_export.get("metrics"):
            extra["trace_metrics"] = dict(trace_export.get("metrics") or {})
        if trace_export.get("errors"):
            extra["trace_errors"] = dict(trace_export.get("errors") or {})
        record = IterationRecord(
            iteration=k,
            stage_name=stage,
            stage_label=stage_label,
            family=family.name,
            f_degree=f_degree,
            g_degree=g_degree,
            axis_kind=str(axis_kind),
            axis_value=int(axis_value),
            leaf_count=int(leaf_count) if leaf_count is not None else None,
            leaf_size_tokens=int(leaf_size_tokens) if leaf_size_tokens is not None else None,
            trained=trained,
            f_artifact=_artifact_for_manifest(f_current),
            g_artifact=_artifact_for_manifest(g_current),
            split_metrics=split_metrics,
            extra=extra,
        )
        records.append(record)
        _write_step_checkpoint(
            output_dir=output_dir,
            family=family,
            axis_kind=axis_kind,
            axis_value=int(axis_value),
            leaf_count=leaf_count,
            leaf_size_tokens=leaf_size_tokens,
            iteration=k,
            stage_name=stage,
            stage_label=stage_label,
            f_degree=f_degree,
            g_degree=g_degree,
            trained=trained,
            phase="post_eval",
            f_artifact=f_current,
            g_artifact=g_current,
            iteration_dir=iter_dir,
            split_metrics=split_metrics,
            error=error,
            artifact_validation=artifact_validation,
            trace_artifacts=trace_export.get("artifacts"),
            trace_metrics=trace_export.get("metrics"),
            trace_errors=trace_export.get("errors"),
        )
        # If training was skipped due to NotImplementedError, subsequent
        # iterations would produce the same error; stop early but keep
        # the records we did gather.
        if error is not None and trained == "skipped":
            break
        if error is not None and not split_metrics:
            break
    return records
