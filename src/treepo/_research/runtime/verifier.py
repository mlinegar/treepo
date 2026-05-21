from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from treepo._research.runtime.contracts import ModelResponse, NodeContract, VerifierResult
from treepo._research.runtime.memory import TokenCounter


@dataclass(frozen=True)
class DeterministicVerifier:
    """Minimal verifier used for early ablations.

    This is intentionally simple: it enforces budget compliance and basic
    non-emptiness, and can be extended with task-specific checks later.
    """

    counter: TokenCounter

    def check(
        self,
        *,
        contract: NodeContract,
        response: ModelResponse,
        min_nonempty_chars: int = 1,
        max_output_tokens: Optional[int] = None,
    ) -> VerifierResult:
        failures: List[str] = []

        text = response.text or ""
        if len(text.strip()) < min_nonempty_chars:
            failures.append("empty_output")

        out_budget = int(max_output_tokens or contract.max_output_tokens)
        out_tokens = self.counter.count(text)
        if out_tokens > out_budget:
            failures.append("output_over_budget")

        return VerifierResult(
            pass_=(len(failures) == 0),
            score=1.0 if len(failures) == 0 else 0.0,
            failures=failures,
            evidence={"output_tokens_est": out_tokens, "output_budget": out_budget},
        )

