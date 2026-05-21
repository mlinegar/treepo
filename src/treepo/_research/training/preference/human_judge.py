"""
Interactive (human-in-the-loop) GenRM-style judge.

This provides a drop-in replacement for `GenRMJudge` for debugging or
manual labeling. It prompts on stdin for a GenRM-style `ranking_score`
(1-6), and returns a `GenRMResult`.

Ranking score interpretation (same as GenRM):
    1 = Response A is much better than Response B
    2 = Response A is better than Response B
    3 = Response A is slightly better than Response B
    4 = Response B is slightly better than Response A
    5 = Response B is better than Response A
    6 = Response B is much better than Response A
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Optional

from .engine import DEFAULT_GENRM_ENGINE
from .genrm import GenRMResult


class HumanGenRMJudge:
    """
    Interactive judge that asks a human for a GenRM-compatible ranking score.

    Notes:
    - Do not use this with concurrency > 1 (e.g., threaded audits), since
      multiple threads calling `input()` will interleave prompts.
    - Intended for small-scale debugging / spot checks, not large-scale runs.
    """

    def __init__(
        self,
        *,
        show_original: bool = False,
        max_preview_chars: int = 1200,
        ask_helpfulness: bool = False,
        ask_reasoning: bool = False,
    ) -> None:
        self.show_original = bool(show_original)
        self.max_preview_chars = max(0, int(max_preview_chars))
        self.ask_helpfulness = bool(ask_helpfulness)
        self.ask_reasoning = bool(ask_reasoning)

    def compare(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
    ) -> GenRMResult:
        print("\n" + "=" * 80)
        print(f"[HumanGenRMJudge] law_type={law_type}")
        if context:
            print("\nCONTEXT:\n" + str(context).strip())
        if extra_context:
            print("\nEXTRA CONTEXT:\n" + str(extra_context).strip())
        if original_text:
            if self.show_original:
                print("\nORIGINAL TEXT:\n" + str(original_text).strip())
            else:
                preview = str(original_text).strip()[: self.max_preview_chars]
                if len(preview) < len(str(original_text).strip()):
                    preview = preview + "\n...[truncated]"
                print("\nORIGINAL TEXT (preview):\n" + preview)

        print("\nRESPONSE A:\n" + str(summary_a).strip())
        print("\nRESPONSE B:\n" + str(summary_b).strip())
        print("=" * 80)

        ranking_score = self._prompt_ranking_score()
        helpfulness_a = 3.0
        helpfulness_b = 3.0
        if self.ask_helpfulness:
            helpfulness_a = self._prompt_optional_float(
                "Helpfulness A (1-5) [enter to skip]: ",
                default=helpfulness_a,
                min_value=1.0,
                max_value=5.0,
            )
            helpfulness_b = self._prompt_optional_float(
                "Helpfulness B (1-5) [enter to skip]: ",
                default=helpfulness_b,
                min_value=1.0,
                max_value=5.0,
            )

        reasoning = ""
        if self.ask_reasoning:
            reasoning = str(input("Reasoning [optional]: ")).strip()

        preferred, confidence = DEFAULT_GENRM_ENGINE.derive_preference(
            score_a=helpfulness_a,
            score_b=helpfulness_b,
            ranking_score=ranking_score,
        )
        return GenRMResult(
            preferred=preferred,  # "A" | "B" | "tie"
            ranking_score=int(ranking_score),
            helpfulness_a=float(helpfulness_a),
            helpfulness_b=float(helpfulness_b),
            reasoning=reasoning,
            confidence=float(confidence),
            raw_response="human",
        )

    def _prompt_ranking_score(self) -> int:
        help_text = (
            "Ranking score (1-6) [1=A≻≻B ... 6=B≻≻A; 3/4 ~ tie; 't' for tie]: "
        )
        while True:
            raw = str(input(help_text)).strip().lower()
            if raw in {"t", "tie"}:
                return 3
            if raw in {"a", "a wins", "a>", "a>>"}:
                return 1
            if raw in {"b", "b wins", "b>", "b>>"}:
                return 6
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer 1-6 (or 't' for tie).")
                continue
            if 1 <= value <= 6:
                return value
            print("Please enter an integer 1-6 (or 't' for tie).")

    def _prompt_optional_float(
        self,
        prompt: str,
        *,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        while True:
            raw = str(input(prompt)).strip()
            if raw == "":
                return float(default)
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number (or press enter to skip).")
                continue
            if min_value is not None and value < min_value:
                print(f"Value must be >= {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be <= {max_value}.")
                continue
            return float(value)


def _demo(args: argparse.Namespace) -> int:
    judge = HumanGenRMJudge(
        show_original=bool(args.show_original),
        max_preview_chars=int(args.max_preview_chars),
        ask_helpfulness=bool(args.ask_helpfulness),
        ask_reasoning=bool(args.ask_reasoning),
    )

    result = judge.compare(
        context="Choose which response better matches the rubric, using the GenRM 1-6 ranking scale.",
        original_text="The quick brown fox jumps over the lazy dog. The dog was not amused.",
        summary_a="A fox jumps over a dog.",
        summary_b="A dog jumps over a fox.",
        law_type="sufficiency",
        extra_context="Rubric: preserve key facts and avoid hallucinations.",
    )
    print("\nParsed result:\n" + json.dumps(asdict(result), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive GenRM-style human judge (demo).")
    parser.add_argument("--show-original", action="store_true", help="Print full original text (not just preview).")
    parser.add_argument("--max-preview-chars", type=int, default=1200, help="Preview length when not showing full original.")
    parser.add_argument("--ask-helpfulness", action="store_true", help="Prompt for helpfulness_a/helpfulness_b (1-5).")
    parser.add_argument("--ask-reasoning", action="store_true", help="Prompt for optional free-text reasoning.")
    args = parser.parse_args()
    return _demo(args)


if __name__ == "__main__":
    raise SystemExit(main())

