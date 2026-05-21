from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypeVar

Z = TypeVar("Z")
Y = TypeVar("Y")


@dataclass
class SklearnProxyOracle:
    """Adapter for scikit-learn style estimators (fit/predict) to the ProxyOracle protocol.

    Notes:
    - ``sample_weight`` is passed through when supported. If the estimator does not
      accept it, an informative TypeError is raised.
    """

    estimator: Any

    def fit(
        self,
        inputs: Sequence[Z],
        targets: Sequence[Y],
        *,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "SklearnProxyOracle":
        if sample_weight is None:
            self.estimator.fit(inputs, targets)
            return self
        try:
            self.estimator.fit(inputs, targets, sample_weight=sample_weight)
        except TypeError as exc:
            raise TypeError(
                "Estimator does not accept sample_weight in fit(). "
                "Use an estimator that supports weighting or pre-resample."
            ) from exc
        return self

    def predict(self, inputs: Sequence[Z]) -> Any:
        return self.estimator.predict(inputs)

