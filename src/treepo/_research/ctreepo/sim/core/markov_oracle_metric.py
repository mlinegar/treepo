"""Markov DGP implementation of OracleMetricSpace.

The Markov oracle space Y = R × {0,...,K-1} × {0,...,K-1}
where count is continuous and first/last are categorical regime IDs.

The metric is Euclidean on the encoded vector
[count * count_scale, first * regime_scale, last * regime_scale].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from treepo._research.ctreepo.sim.core.oracle_metric import (
    OracleMetricSpace,
    register_oracle_metric,
)


MARKOV_ORACLE_METRIC_NAME = "markov"


@dataclass(frozen=True)
class MarkovOracleMetric:
    """Oracle metric for the Markov changepoint DGP.

    Encodes (count, first_regime, last_regime) as a real vector and uses
    Euclidean distance.  The scale parameters control the relative weight
    of count differences vs regime mismatches.

    With default scales (1.0, 1.0):
    - Two nodes differing by 1 in count have distance 1.0
    - Two nodes differing in first regime (e.g., 0 vs 1) have distance 1.0
    - These contribute equally, which is a reasonable default
    """

    count_scale: float = 1.0
    regime_scale: float = 1.0

    @property
    def oracle_dim(self) -> int:
        return 3

    def oracle_vector(
        self,
        *,
        count: float,
        first: int | None = None,
        last: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        return np.array(
            [
                float(count) * self.count_scale,
                float(int(first or 0)) * self.regime_scale,
                float(int(last or 0)) * self.regime_scale,
            ],
            dtype=np.float64,
        )

    def distance(self, y1: np.ndarray, y2: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(y1) - np.asarray(y2)))

    def task_readout(self, y: np.ndarray) -> float:
        scale = self.count_scale if self.count_scale != 0.0 else 1.0
        return float(np.asarray(y)[0] / scale)


# Verify protocol conformance at import time.
assert isinstance(MarkovOracleMetric(), OracleMetricSpace)


register_oracle_metric(
    MARKOV_ORACLE_METRIC_NAME,
    MarkovOracleMetric,
    overwrite=True,
)
