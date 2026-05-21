from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


DEFAULT_THEOREM_FEATURE_ADAPTER = "markov_count_sketch"


@dataclass(frozen=True)
class TheoremFeaturePairSets:
    same_pairs: tuple[tuple[int, int], ...] = tuple()
    different_pairs: tuple[tuple[int, int], ...] = tuple()

    @property
    def total_pairs(self) -> int:
        return int(len(self.same_pairs) + len(self.different_pairs))


@dataclass(frozen=True)
class TheoremFeatureTargets:
    leaf: tuple[Any, ...] = tuple()
    merge: tuple[Any, ...] = tuple()
    root: tuple[Any, ...] = tuple()
    merge_join_bits: tuple[int, ...] = tuple()

    def all_labels(self) -> tuple[Any, ...]:
        return tuple(self.leaf) + tuple(self.merge) + tuple(self.root)


@dataclass(frozen=True)
class TheoremFeatureDiagnostics:
    phi_pair_same_accuracy: float = float("nan")
    phi_pair_diff_accuracy: float = float("nan")
    phi_pair_auc: float = float("nan")
    phi_replay_same_class_rate: float = float("nan")
    task_factorization_gap: float = float("nan")

    def as_dict(self) -> Dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class TheoremFeatureStage1Artifact:
    artifact_dir: str
    model_state_path: str
    metadata_path: str
    selection_metric_name: str = ""
    selection_metric_value: float = float("nan")
    best_epoch: int = 0
    epochs_completed: int = 0
    training_schedule: str = ""
    artifact_source: str = "trained"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "model_state_path": str(self.model_state_path),
            "metadata_path": str(self.metadata_path),
            "selection_metric_name": str(self.selection_metric_name),
            "selection_metric_value": float(self.selection_metric_value),
            "best_epoch": int(self.best_epoch),
            "epochs_completed": int(self.epochs_completed),
            "training_schedule": str(self.training_schedule),
            "artifact_source": str(self.artifact_source),
        }


@runtime_checkable
class TheoremFeatureAdapter(Protocol):
    name: str
    has_canonical_decode: bool

    def oracle_label(
        self,
        *,
        count: float,
        first: int | None = None,
        last: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        ...

    def same_pair(
        self,
        left: Any,
        right: Any,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        ...

    def different_pair(
        self,
        left: Any,
        right: Any,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        ...

    def diagnostic_key(self, label: Any) -> Hashable:
        ...

    def task_readout_target(self, label: Any) -> float:
        ...

    def decode_from_phi(self, phi: Any) -> Any | None:
        ...


_THEOREM_FEATURE_ADAPTER_REGISTRY: Dict[
    str, Callable[[], TheoremFeatureAdapter]
] = {}


def register_theorem_feature_adapter(
    name: str,
    factory: Callable[[], TheoremFeatureAdapter],
    *,
    overwrite: bool = False,
) -> None:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("adapter name must be non-empty")
    if normalized in _THEOREM_FEATURE_ADAPTER_REGISTRY and not overwrite:
        raise ValueError(f"adapter {normalized!r} is already registered")
    _THEOREM_FEATURE_ADAPTER_REGISTRY[normalized] = factory


def ensure_builtin_theorem_feature_adapters_registered() -> None:
    if DEFAULT_THEOREM_FEATURE_ADAPTER in _THEOREM_FEATURE_ADAPTER_REGISTRY:
        return
    from treepo._research.ctreepo.sim.core import markov_theorem_feature_adapter  # noqa: F401


def valid_theorem_feature_adapters() -> tuple[str, ...]:
    ensure_builtin_theorem_feature_adapters_registered()
    return tuple(sorted(_THEOREM_FEATURE_ADAPTER_REGISTRY))


def resolve_theorem_feature_adapter(name: str) -> TheoremFeatureAdapter:
    ensure_builtin_theorem_feature_adapters_registered()
    normalized = str(name or DEFAULT_THEOREM_FEATURE_ADAPTER).strip().lower()
    if not normalized:
        normalized = DEFAULT_THEOREM_FEATURE_ADAPTER
    factory = _THEOREM_FEATURE_ADAPTER_REGISTRY.get(normalized)
    if factory is None:
        raise ValueError(
            f"unsupported theorem_feature_adapter={name!r}; "
            f"expected one of {valid_theorem_feature_adapters()}"
        )
    adapter = factory()
    if not isinstance(adapter, TheoremFeatureAdapter):
        raise TypeError(f"registered adapter {normalized!r} does not satisfy TheoremFeatureAdapter")
    return adapter


def theorem_feature_targets_from_markov_exact_targets(
    *,
    adapter: TheoremFeatureAdapter,
    exact_targets: Mapping[str, Sequence[tuple[float, int, int]]],
    leaf_metadata: Sequence[Mapping[str, Any] | None] | None = None,
    merge_metadata: Sequence[Mapping[str, Any] | None] | None = None,
    root_metadata: Sequence[Mapping[str, Any] | None] | None = None,
) -> TheoremFeatureTargets:
    leaf_metadata_values = list(leaf_metadata or ())
    merge_metadata_values = list(merge_metadata or ())
    root_metadata_values = list(root_metadata or ())

    def _convert(
        values: Sequence[tuple[float, int, int]],
        metadata_values: Sequence[Mapping[str, Any] | None],
    ) -> tuple[Any, ...]:
        labels = []
        for idx, (count, first, last) in enumerate(list(values)):
            labels.append(
                adapter.oracle_label(
                    count=float(count),
                    first=int(first),
                    last=int(last),
                    metadata=(
                        dict(metadata_values[int(idx)])
                        if int(idx) < len(metadata_values)
                        and metadata_values[int(idx)] is not None
                        else None
                    ),
                )
            )
        return tuple(labels)

    return TheoremFeatureTargets(
        leaf=_convert(exact_targets.get("leaf", tuple()), leaf_metadata_values),
        merge=_convert(exact_targets.get("merge", tuple()), merge_metadata_values),
        root=_convert(exact_targets.get("root", tuple()), root_metadata_values),
        merge_join_bits=tuple(int(value) for value in exact_targets.get("merge_join_bits", tuple())),
    )


def theorem_feature_class_ids(
    labels: Sequence[Any],
    *,
    adapter: TheoremFeatureAdapter,
) -> tuple[int, ...]:
    label_index: Dict[Hashable, int] = {}
    ids = []
    for label in list(labels):
        key = adapter.diagnostic_key(label)
        ids.append(int(label_index.setdefault(key, len(label_index))))
    return tuple(ids)


def build_theorem_feature_pair_sets(
    labels: Sequence[Any],
    *,
    adapter: TheoremFeatureAdapter,
    same_threshold: float | None = None,
    diff_threshold: float | None = None,
) -> TheoremFeaturePairSets:
    same_pairs: list[tuple[int, int]] = []
    different_pairs: list[tuple[int, int]] = []
    values = list(labels)
    for left_idx in range(len(values)):
        for right_idx in range(left_idx + 1, len(values)):
            left = values[left_idx]
            right = values[right_idx]
            is_same = bool(
                adapter.same_pair(
                    left,
                    right,
                    same_threshold=same_threshold,
                    diff_threshold=diff_threshold,
                )
            )
            is_diff = bool(
                adapter.different_pair(
                    left,
                    right,
                    same_threshold=same_threshold,
                    diff_threshold=diff_threshold,
                )
            )
            if is_same and is_diff:
                raise ValueError("adapter marked one pair as both same and different")
            if is_same:
                same_pairs.append((int(left_idx), int(right_idx)))
            elif is_diff:
                different_pairs.append((int(left_idx), int(right_idx)))
    return TheoremFeaturePairSets(
        same_pairs=tuple(same_pairs),
        different_pairs=tuple(different_pairs),
    )


def theorem_feature_pair_metrics_from_scores(
    *,
    same_scores: Sequence[float],
    different_scores: Sequence[float],
) -> TheoremFeatureDiagnostics:
    same_arr = np.asarray(list(same_scores), dtype=np.float64)
    diff_arr = np.asarray(list(different_scores), dtype=np.float64)
    if same_arr.size <= 0 or diff_arr.size <= 0:
        return TheoremFeatureDiagnostics()
    threshold = float((np.mean(same_arr) + np.mean(diff_arr)) / 2.0)
    same_acc = float(np.mean((same_arr >= threshold).astype(np.float64)))
    diff_acc = float(np.mean((diff_arr < threshold).astype(np.float64)))
    auc = float(_pair_auc_from_score_arrays(same_arr, diff_arr))
    return TheoremFeatureDiagnostics(
        phi_pair_same_accuracy=same_acc,
        phi_pair_diff_accuracy=diff_acc,
        phi_pair_auc=auc,
    )


def write_theorem_feature_stage1_artifact(
    artifact_dir: str | Path,
    *,
    model_state: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> TheoremFeatureStage1Artifact:
    import torch

    artifact_root = Path(str(artifact_dir)).expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    model_state_path = artifact_root / "model_state.pt"
    metadata_path = artifact_root / "metadata.json"
    artifact = TheoremFeatureStage1Artifact(
        artifact_dir=str(artifact_root),
        model_state_path=str(model_state_path),
        metadata_path=str(metadata_path),
        selection_metric_name=str((metadata or {}).get("selection_metric_name", "")),
        selection_metric_value=float(
            (metadata or {}).get("selection_metric_value", float("nan"))
        ),
        best_epoch=int((metadata or {}).get("best_epoch", 0)),
        epochs_completed=int((metadata or {}).get("epochs_completed", 0)),
        training_schedule=str((metadata or {}).get("training_schedule", "")),
        artifact_source=str((metadata or {}).get("artifact_source", "trained")),
    )
    torch.save(dict(model_state), model_state_path)
    payload = {
        **artifact.as_dict(),
        **{
            str(key): value
            for key, value in dict(metadata or {}).items()
            if str(key) not in artifact.as_dict()
        },
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact


def load_theorem_feature_stage1_artifact(
    artifact_dir: str | Path,
) -> tuple[TheoremFeatureStage1Artifact, Dict[str, Any]]:
    import torch

    artifact_root = Path(str(artifact_dir)).expanduser().resolve()
    metadata_path = artifact_root / "metadata.json"
    model_state_path = artifact_root / "model_state.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"stage-1 artifact metadata not found: {metadata_path}")
    if not model_state_path.exists():
        raise FileNotFoundError(f"stage-1 artifact model state not found: {model_state_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = TheoremFeatureStage1Artifact(
        artifact_dir=str(payload.get("artifact_dir", artifact_root)),
        model_state_path=str(payload.get("model_state_path", model_state_path)),
        metadata_path=str(payload.get("metadata_path", metadata_path)),
        selection_metric_name=str(payload.get("selection_metric_name", "")),
        selection_metric_value=float(payload.get("selection_metric_value", float("nan"))),
        best_epoch=int(payload.get("best_epoch", 0)),
        epochs_completed=int(payload.get("epochs_completed", 0)),
        training_schedule=str(payload.get("training_schedule", "")),
        artifact_source=str(payload.get("artifact_source", "loaded")),
    )
    state = torch.load(artifact.model_state_path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError(
            "stage-1 artifact model_state.pt must contain a mapping-compatible state_dict"
        )
    return artifact, dict(state)


def theorem_feature_replay_same_class_rate(
    left_labels: Sequence[Any],
    right_labels: Sequence[Any],
    *,
    adapter: TheoremFeatureAdapter,
    same_threshold: float | None = None,
    diff_threshold: float | None = None,
) -> float:
    left_values = list(left_labels)
    right_values = list(right_labels)
    if len(left_values) != len(right_values) or not left_values:
        return float("nan")
    matches = []
    for left, right in zip(left_values, right_values):
        matches.append(
            float(
                int(
                    adapter.same_pair(
                        left,
                        right,
                        same_threshold=same_threshold,
                        diff_threshold=diff_threshold,
                    )
                )
            )
        )
    return float(np.mean(np.asarray(matches, dtype=np.float64)))


def _pair_auc_from_score_arrays(
    same_scores: np.ndarray,
    different_scores: np.ndarray,
) -> float:
    if same_scores.size <= 0 or different_scores.size <= 0:
        return float("nan")
    diff_sorted = np.sort(np.asarray(different_scores, dtype=np.float64))
    same_arr = np.asarray(same_scores, dtype=np.float64)
    lower = np.searchsorted(diff_sorted, same_arr, side="left")
    upper = np.searchsorted(diff_sorted, same_arr, side="right")
    wins = float(np.sum(lower, dtype=np.float64))
    ties = float(np.sum(upper - lower, dtype=np.float64))
    total = float(same_arr.size * diff_sorted.size)
    if total <= 0.0:
        return float("nan")
    return float((wins + 0.5 * ties) / total)
