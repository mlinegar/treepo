from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ExperimentMethod(Protocol):
    """Method-level lifecycle used by `ExperimentContext`.

    Native frameworks keep their own idioms internally. For example, a PyTorch
    method should call `module.train()` and `module.eval()` inside these methods
    rather than exposing a raw `nn.Module` as the experiment method.
    """

    def train(
        self,
        train_data: Any,
        validation_data: Any | None = None,
        *,
        context: Any,
        config: Mapping[str, Any] | None = None,
    ) -> Any: ...

    def evaluate(
        self,
        data: Any,
        *,
        context: Any,
        split: str = "test",
        config: Mapping[str, Any] | None = None,
    ) -> Any: ...

    def predict(
        self,
        inputs: Any,
        *,
        context: Any,
        config: Mapping[str, Any] | None = None,
    ) -> Any: ...


__all__ = ["ExperimentMethod"]
