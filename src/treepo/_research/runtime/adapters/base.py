from __future__ import annotations

from typing import Iterable, Protocol

from treepo._research.runtime.contracts import AnswerSpec, NodeContract, ProblemSpec, RuntimeTaskView


class BenchmarkAdapter(Protocol):
    def load_split(self, split: str, limit: int | None = None) -> Iterable[ProblemSpec]:
        ...

    def build_contract(self, problem: ProblemSpec) -> NodeContract:
        ...

    def task_view(self, problem: ProblemSpec) -> RuntimeTaskView:
        ...

    def parse_prediction(self, problem: ProblemSpec, text: str) -> str:
        ...

    def build_answer_spec(self, problem: ProblemSpec) -> AnswerSpec:
        ...

    def score(self, problem: ProblemSpec, runtime_output: dict) -> dict[str, float]:
        ...

    def primary_metric(self) -> str:
        ...

    def supports_tools(self) -> bool:
        ...
