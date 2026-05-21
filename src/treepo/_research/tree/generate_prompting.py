"""Prompt templates for TreePO execution over `/generate`-style surfaces.

These prompts are intended to work for both:
- standard autoregressive decoding (no diffusion-specific engine_options), and
- diffusion-style decoding (e.g., SGLang DLLM) controlled via engine_options.

The intent is "generate-first" execution: a single surface that returns text.
"""

from __future__ import annotations

from dataclasses import dataclass

from treepo._research.core.protocols import format_merge_input


@dataclass(frozen=True)
class GenerateTreePromptTemplates:
    """Default prompt templates for fixed-binary TreePO over `/generate`."""

    leaf_system: str = (
        "You are a careful text summarizer.\n"
        "Preserve all information relevant to the rubric.\n"
        "Output ONLY the summary text."
    )
    merge_system: str = (
        "You are a careful text summarizer.\n"
        "Merge two summaries into ONE coherent summary while preserving all rubric-relevant facts.\n"
        "Output ONLY the merged summary text."
    )
    refine_system: str = (
        "You refine an existing summary.\n"
        "Rewrite it to be more concise and coherent while preserving all rubric-relevant facts.\n"
        "Output ONLY the revised summary text."
    )


def format_generate_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format a plain-text chat-style prompt for `/generate` endpoints."""
    return (
        f"<system>\n{system_prompt.strip()}\n</system>\n\n"
        f"<user>\n{user_prompt.strip()}\n</user>\n\n"
        "<assistant>\n"
    )


def leaf_prompt(text: str, rubric: str, templates: GenerateTreePromptTemplates) -> str:
    rubric_clean = str(rubric or "").strip() or "Preserve rubric-relevant content."
    return format_generate_chat_prompt(
        templates.leaf_system,
        f"Rubric:\n{rubric_clean}\n\nText:\n{str(text or '').strip()}",
    )


def merge_prompt(left_summary: str, right_summary: str, rubric: str, templates: GenerateTreePromptTemplates) -> str:
    rubric_clean = str(rubric or "").strip() or "Preserve rubric-relevant content."
    merge_input = format_merge_input(str(left_summary or ""), str(right_summary or ""))
    return format_generate_chat_prompt(
        templates.merge_system,
        f"Rubric:\n{rubric_clean}\n\nMerge input:\n{merge_input}",
    )


def refine_prompt(summary: str, rubric: str, round_index: int, templates: GenerateTreePromptTemplates) -> str:
    rubric_clean = str(rubric or "").strip() or "Preserve rubric-relevant content."
    round_value = int(round_index) if round_index is not None else 1
    return format_generate_chat_prompt(
        templates.refine_system,
        (
            f"Rubric:\n{rubric_clean}\n\n"
            f"Refinement round: {round_value}\n\n"
            f"Current summary:\n{str(summary or '').strip()}"
        ),
    )


__all__ = [
    "GenerateTreePromptTemplates",
    "format_generate_chat_prompt",
    "leaf_prompt",
    "merge_prompt",
    "refine_prompt",
]

