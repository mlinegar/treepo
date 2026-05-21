from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from treepo._research.runtime.backbone import BackboneAdapter
from treepo._research.runtime.contracts import ModelResponse, NodeContract, VerifierResult


@dataclass(frozen=True)
class SimpleRepairPolicy:
    """Minimal repair policy: one retry with stricter instructions."""

    def apply(
        self,
        *,
        backbone: BackboneAdapter,
        messages: List[Dict[str, str]],
        contract: NodeContract,
        verifier: VerifierResult,
        max_tokens: int,
    ) -> ModelResponse:
        tightened = list(messages)
        tightened.append(
            {
                "role": "user",
                "content": (
                    "The previous output failed checks: "
                    f"{', '.join(verifier.failures)}. "
                    "Retry. Keep the output concise and within the budget. "
                    "Output only the required content."
                ),
            }
        )
        return backbone.generate(tightened, max_tokens=max_tokens, temperature=0.0)

