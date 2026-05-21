"""
Manifesto RILE task plugin registration.

Registers `manifesto_rile` so the training pipeline can target CMP RILE
scoring directly without relying on document-analysis defaults.
"""

from typing import Any, Callable, Dict, Optional

from treepo._research.tasks.prompting import (
    PromptBuilders,
    default_merge_prompt,
    default_summarize_prompt,
)

from .manifesto import (
    RILE_SCALE,
    RILE_PRESERVATION_RUBRIC,
    RILE_TASK_CONTEXT,
    RILEScorer,
    UnifiedManifestoG,
    create_rile_oracle,
)
from .registry import register_task
from .scoring import ScoringTask


def _build_score_prompt(summary: str, task_context: str) -> list[dict[str, str]]:
    """Prompt the model for a raw CMP RILE score in [-100, +100]."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert CMP manifesto coder. "
                "Return exactly one numeric RILE score between -100 and +100. "
                "Estimate as precisely as possible on the continuous scale (do not default to coarse bins)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{task_context}\n\n"
                f"SUMMARY:\n{summary}\n\n"
                "Output only the numeric RILE score in [-100, +100] "
                "(no labels like 'Score:', no extra text, no code fences)."
            ),
        },
    ]


@register_task(["manifesto_rile", "rile"])
class ManifestoRILETask(ScoringTask):
    """RILE-specific task plugin backed by the generic ScoringTask."""

    def __init__(
        self,
        name: str = "manifesto_rile",
        error_threshold_high: float = 0.15,
        error_threshold_low: float = 0.05,
        use_cot_summarizer: bool = False,
        use_cot_scorer: bool = False,
        use_cot_merge: Optional[bool] = None,
        scorer_max_tokens: int = 96,
        scorer_temperature: float = 0.0,
        scorer_strict_parse: bool = True,
        **_: Optional[object],
    ):
        if use_cot_merge is None:
            use_cot_merge = use_cot_summarizer
        self._use_cot_merge = use_cot_merge
        scorer_max_tokens = max(1, int(scorer_max_tokens))
        scorer_temperature = float(scorer_temperature)
        scorer_strict_parse = bool(scorer_strict_parse)

        prompt_builders = PromptBuilders(
            summarize=default_summarize_prompt,
            merge=default_merge_prompt,
            score=_build_score_prompt,
            audit=None,
        )

        scorer = create_rile_oracle(task_context=RILE_TASK_CONTEXT)

        super().__init__(
            name=name,
            scale=RILE_SCALE,
            id_field="doc_id",
            label_field="reference_score",
            text_field="text",
            output_field_name="score",
            error_threshold_high=error_threshold_high,
            error_threshold_low=error_threshold_low,
            rubric=RILE_PRESERVATION_RUBRIC.strip(),
            task_context=RILE_TASK_CONTEXT.strip(),
            prompt_builders=prompt_builders,
            predictor_factory=lambda: RILEScorer(
                use_cot=use_cot_scorer,
                max_tokens=scorer_max_tokens,
                temperature=scorer_temperature,
                strict_parse=scorer_strict_parse,
            ),
            summarizer_factory=lambda: UnifiedManifestoG(use_cot=use_cot_summarizer),
            oracle_scorer_factory=scorer.value_extractor,
        )

    def create_merge_summarizer(self):
        """Return the same unified g module for merge compatibility callers."""
        return self.create_summarizer()

    def describe_local_law_oracle(self) -> Dict[str, Any]:
        return {
            "available": True,
            "exact": False,
            "model_backed": True,
            "kind": "task_oracle_model_backed",
            "spec": f"{self.name}:rile_task_oracle",
        }

    def create_local_law_oracle(
        self,
        *,
        port: Optional[int] = None,
        model: Optional[str] = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        strict_parse: bool = True,
    ) -> Callable[[str], float]:
        from treepo._research.config.dspy_config import configure_dspy, create_local_engine_lm
        from treepo._research.config.local_inference import resolve_local_inference_config

        if port is not None or model:
            local_inference = resolve_local_inference_config(
                {
                    "port": int(port) if port is not None else None,
                    "model": model,
                    "temperature": float(temperature),
                    "max_tokens": max(1, int(max_tokens)),
                }
            )
            scorer_lm = create_local_engine_lm(**local_inference.dspy_kwargs())
            configure_dspy(lm=scorer_lm)

        scorer = RILEScorer(
            max_tokens=max(1, int(max_tokens)),
            temperature=float(temperature),
            strict_parse=bool(strict_parse),
        )

        def _score_span(text: str) -> float:
            result = scorer(text=text, task_context=RILE_TASK_CONTEXT)
            score01 = 0.5
            if isinstance(result, dict):
                score01 = float(result.get("score", 0.5))
            raw = self.denormalize_score(score01)
            return float(0.0 if raw is None else raw)

        return _score_span
