from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from treepo._research.ctreepo.sim.core.theorem_feature_route import (
    DEFAULT_THEOREM_FEATURE_ADAPTER,
    TheoremFeatureAdapter,
    register_theorem_feature_adapter,
)


@dataclass(frozen=True)
class MarkovTheoremFeatureLabel:
    count: float
    first: int
    last: int


@dataclass(frozen=True)
class ScoreFiberTheoremFeatureLabel:
    score: float
    fiber_key: Any


@dataclass(frozen=True)
class MarkovCountSketchTheoremFeatureAdapter(TheoremFeatureAdapter):
    name: str = DEFAULT_THEOREM_FEATURE_ADAPTER
    has_canonical_decode: bool = True

    def oracle_label(
        self,
        *,
        count: float,
        first: int | None = None,
        last: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MarkovTheoremFeatureLabel:
        if first is None or last is None:
            raise ValueError("markov_count_sketch labels require first and last regimes")
        return MarkovTheoremFeatureLabel(
            count=float(count),
            first=int(first),
            last=int(last),
        )

    def same_pair(
        self,
        left: MarkovTheoremFeatureLabel,
        right: MarkovTheoremFeatureLabel,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        return self.diagnostic_key(left) == self.diagnostic_key(right)

    def different_pair(
        self,
        left: MarkovTheoremFeatureLabel,
        right: MarkovTheoremFeatureLabel,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        return self.diagnostic_key(left) != self.diagnostic_key(right)

    def diagnostic_key(self, label: MarkovTheoremFeatureLabel) -> tuple[int, int, int]:
        return (
            int(round(float(label.count))),
            int(label.first),
            int(label.last),
        )

    def task_readout_target(self, label: MarkovTheoremFeatureLabel) -> float:
        return float(label.count)

    def decode_from_phi(self, phi: Any) -> None:
        return None


register_theorem_feature_adapter(
    DEFAULT_THEOREM_FEATURE_ADAPTER,
    lambda: MarkovCountSketchTheoremFeatureAdapter(),
    overwrite=True,
)


SCOREFIBER_MARKOV_ENDPOINTS_ADAPTER = "markov_score_endpoints"


@dataclass(frozen=True)
class MarkovScoreEndpointsTheoremFeatureAdapter(TheoremFeatureAdapter):
    name: str = SCOREFIBER_MARKOV_ENDPOINTS_ADAPTER
    has_canonical_decode: bool = False

    def oracle_label(
        self,
        *,
        count: float,
        first: int | None = None,
        last: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ScoreFiberTheoremFeatureLabel:
        if first is None or last is None:
            raise ValueError("markov_score_endpoints labels require first and last regimes")
        return ScoreFiberTheoremFeatureLabel(
            score=float(count),
            fiber_key=(int(first), int(last)),
        )

    def same_pair(
        self,
        left: ScoreFiberTheoremFeatureLabel,
        right: ScoreFiberTheoremFeatureLabel,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        return self.diagnostic_key(left) == self.diagnostic_key(right)

    def different_pair(
        self,
        left: ScoreFiberTheoremFeatureLabel,
        right: ScoreFiberTheoremFeatureLabel,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        return self.diagnostic_key(left) != self.diagnostic_key(right)

    def diagnostic_key(self, label: ScoreFiberTheoremFeatureLabel) -> tuple[int, int]:
        first, last = label.fiber_key
        return (int(first), int(last))

    def task_readout_target(self, label: ScoreFiberTheoremFeatureLabel) -> float:
        return float(label.score)

    def decode_from_phi(self, phi: Any) -> None:
        return None


register_theorem_feature_adapter(
    SCOREFIBER_MARKOV_ENDPOINTS_ADAPTER,
    lambda: MarkovScoreEndpointsTheoremFeatureAdapter(),
    overwrite=True,
)


COARSENED_THEOREM_FEATURE_ADAPTER = "markov_count_sketch_coarsened"


@dataclass(frozen=True)
class CoarsenedMarkovTheoremFeatureAdapter(TheoremFeatureAdapter):
    """Adapter that bins counts into ranges to increase positive pair density.

    With exact (count, first, last) keys, same-class pairs are extremely rare
    within a single document's tree (~7 states).  Binning counts into width-N
    buckets dramatically increases positive pair yield while preserving the
    fiber structure that matters: states with *similar* oracle outputs should
    map to nearby phi embeddings.
    """

    name: str = COARSENED_THEOREM_FEATURE_ADAPTER
    has_canonical_decode: bool = False
    count_bin_width: int = 3
    ignore_endpoints: bool = False

    def oracle_label(
        self,
        *,
        count: float,
        first: int | None = None,
        last: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MarkovTheoremFeatureLabel:
        if first is None or last is None:
            raise ValueError("markov_count_sketch labels require first and last regimes")
        return MarkovTheoremFeatureLabel(
            count=float(count),
            first=int(first),
            last=int(last),
        )

    def _binned_count(self, count: float) -> int:
        return int(round(float(count))) // max(1, self.count_bin_width)

    def same_pair(
        self,
        left: MarkovTheoremFeatureLabel,
        right: MarkovTheoremFeatureLabel,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        return self.diagnostic_key(left) == self.diagnostic_key(right)

    def different_pair(
        self,
        left: MarkovTheoremFeatureLabel,
        right: MarkovTheoremFeatureLabel,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        lk = self.diagnostic_key(left)
        rk = self.diagnostic_key(right)
        if self.ignore_endpoints:
            return abs(lk[0] - rk[0]) >= 2
        return lk != rk

    def diagnostic_key(
        self, label: MarkovTheoremFeatureLabel
    ) -> tuple[int, int | None, int | None]:
        binned = self._binned_count(label.count)
        if self.ignore_endpoints:
            return (binned, None, None)
        return (binned, int(label.first), int(label.last))

    def task_readout_target(self, label: MarkovTheoremFeatureLabel) -> float:
        return float(label.count)

    def decode_from_phi(self, phi: Any) -> None:
        return None


register_theorem_feature_adapter(
    COARSENED_THEOREM_FEATURE_ADAPTER,
    lambda: CoarsenedMarkovTheoremFeatureAdapter(),
    overwrite=True,
)


SCOREFIBER_LENGTH_BUCKET_ADAPTER = "scorefiber_length_bucket"


@dataclass(frozen=True)
class ScoreFiberLengthBucketTheoremFeatureAdapter(TheoremFeatureAdapter):
    """Toy non-Markov fiber label: score is scalar count, fiber is a size bucket."""

    name: str = SCOREFIBER_LENGTH_BUCKET_ADAPTER
    has_canonical_decode: bool = False

    @staticmethod
    def _bucket_from_metadata(metadata: Mapping[str, Any] | None) -> int:
        payload = metadata or {}
        leaf_span_count = int(payload.get("leaf_span_count", 0))
        if leaf_span_count > 0:
            if leaf_span_count <= 2:
                return 0
            if leaf_span_count <= 4:
                return 1
            return 2
        span_length = int(payload.get("span_length", 0))
        if span_length <= 32:
            return 0
        if span_length <= 64:
            return 1
        return 2

    def oracle_label(
        self,
        *,
        count: float,
        first: int | None = None,
        last: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ScoreFiberTheoremFeatureLabel:
        return ScoreFiberTheoremFeatureLabel(
            score=float(count),
            fiber_key=int(self._bucket_from_metadata(metadata)),
        )

    def same_pair(
        self,
        left: ScoreFiberTheoremFeatureLabel,
        right: ScoreFiberTheoremFeatureLabel,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        return self.diagnostic_key(left) == self.diagnostic_key(right)

    def different_pair(
        self,
        left: ScoreFiberTheoremFeatureLabel,
        right: ScoreFiberTheoremFeatureLabel,
        *,
        same_threshold: float | None = None,
        diff_threshold: float | None = None,
    ) -> bool:
        return self.diagnostic_key(left) != self.diagnostic_key(right)

    def diagnostic_key(self, label: ScoreFiberTheoremFeatureLabel) -> int:
        return int(label.fiber_key)

    def task_readout_target(self, label: ScoreFiberTheoremFeatureLabel) -> float:
        return float(label.score)

    def decode_from_phi(self, phi: Any) -> None:
        return None


register_theorem_feature_adapter(
    SCOREFIBER_LENGTH_BUCKET_ADAPTER,
    lambda: ScoreFiberLengthBucketTheoremFeatureAdapter(),
    overwrite=True,
)
