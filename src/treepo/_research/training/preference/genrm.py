"""
GenRM-based Preference Collection for OPS Summarization.

Uses NVIDIA's Qwen3-Nemotron-235B-A22B-GenRM model to compare summaries.
The GenRM model uses a special format with response_1 and response_2 roles
and produces helpfulness scores (1-5) and ranking scores (1-6).

Ranking score interpretation:
    1 = Response 1 is much better than Response 2
    2 = Response 1 is better than Response 2
    3 = Response 1 is slightly better than Response 2
    4 = Response 2 is slightly better than Response 1
    5 = Response 2 is better than Response 1
    6 = Response 2 is much better than Response 1

Usage:
    from treepo._research.training.preference.genrm import (
        GenRMJudge,
        GenRMPreferenceCollector,
    )
    from treepo._research.config import get_genrm_url

    # Create judge connected to GenRM server (auto-detects URL from config)
    judge = GenRMJudge(base_url=get_genrm_url())

    # Compare two summaries
    result = judge.compare(
        context="Summarize this text preserving the key information",
        original_text="...",
        summary_a="...",
        summary_b="...",
    )
    # result.preferred = "A" or "B" or "tie"
    # result.ranking_score = 1-6
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import aiohttp
import requests

from .base import BasePreferenceCollector, CandidateInfo, PreferenceResult
from .collector import GenerationConfig, PreferenceDataset
from .engine import DEFAULT_GENRM_ENGINE, PreferenceEngine

# Import batch client for type hints (actual import happens in methods to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .genrm_batch import AsyncBatchGenRMClient

logger = logging.getLogger(__name__)


@dataclass
class GenRMResult:
    """Result from GenRM comparison."""
    preferred: Literal["A", "B", "tie"]
    ranking_score: int  # 1-6
    helpfulness_a: float  # 1-5
    helpfulness_b: float  # 1-5
    reasoning: str
    confidence: float
    raw_response: str = ""


@dataclass
class GenRMErrorResult:
    """Error result from GenRM - distinct from legitimate ties.

    This type exists to prevent network/timeout errors from being confused
    with genuine 50/50 preference ties. Callers should filter these out
    of training data rather than treating them as preferences.
    """
    error_type: Literal["network", "timeout", "parse_error", "server_error"]
    error_message: str
    raw_response: str = ""

    def is_error(self) -> bool:
        """Always True for error results."""
        return True


# Type alias for GenRM comparison results
GenRMComparisonResult = Union[GenRMResult, GenRMErrorResult]


def is_genrm_error(result: GenRMComparisonResult) -> bool:
    """Check if a GenRM result is an error (not a valid preference)."""
    return isinstance(result, GenRMErrorResult)


class GenRMJudge:
    """
    Judge using NVIDIA's Qwen3-Nemotron-235B-A22B-GenRM.

    Uses the special response_1/response_2 format for comparison.
    """

    # Class-level cache for model names per server URL (avoids repeated HTTP requests)
    _model_cache: Dict[str, str] = {}

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        batch_client: Optional["AsyncBatchGenRMClient"] = None,
    ):
        """
        Initialize the GenRM judge.

        Args:
            base_url: vLLM server base URL (None = auto-detect from config)
            model_name: Model name for API requests (None = auto-detect from server)
            temperature: Generation temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens for response
            batch_client: Optional AsyncBatchGenRMClient for batched requests (better throughput)
        """
        # Auto-detect base URL from config if not provided
        if base_url is None:
            from treepo._research.config import get_genrm_url
            base_url = get_genrm_url()
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.batch_client = batch_client  # Optional batching support

        # Auto-detect model name from server if not provided
        if model_name is None:
            self.model_name = self._detect_model_name()
        else:
            self.model_name = model_name

    def _detect_model_name(self) -> Optional[str]:
        """Auto-detect the model name from the vLLM server (cached per URL)."""
        # Check class-level cache first
        if self.base_url in GenRMJudge._model_cache:
            model_id = GenRMJudge._model_cache[self.base_url]
            logger.debug(f"Using cached GenRM model: {model_id}")
            return model_id

        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                model_id = data["data"][0]["id"]
                # Cache for future instances
                GenRMJudge._model_cache[self.base_url] = model_id
                logger.info(f"Auto-detected GenRM model: {model_id}")
                return model_id
        except Exception as e:
            logger.warning(f"Failed to auto-detect model name: {e}")

        # Return None to indicate detection failed - will retry on first request
        return None

    def _ensure_model_name(self) -> str:
        """Ensure model name is detected, retrying if needed."""
        if self.model_name is None:
            self.model_name = self._detect_model_name()
        if self.model_name is None:
            raise RuntimeError(
                f"Could not detect GenRM model name from {self.base_url}/models. "
                "Is the GenRM server running?"
            )

    def _build_messages(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build messages in GenRM format with response_1 and response_2.

        The format requires:
        - Conversation history with user/assistant roles
        - Last turn must be a user message
        - Two candidate responses as response_1 and response_2
        """
        law_instructions = {
            "sufficiency": (
                "Compare which summary better preserves the oracle-relevant "
                "information from the original text."
            ),
            "idempotence": (
                "Compare which summary is more stable under re-summarization. "
                "Use the provided resummaries to judge drift."
            ),
            "merge": (
                "Compare which merged summary better preserves the information "
                "from its child summaries."
            ),
        }
        instruction = law_instructions.get(law_type, law_instructions["sufficiency"])

        original_section = ""
        if original_text.strip():
            original_section = f"\n\nOriginal Text:\n{original_text}"

        extra_section = ""
        if extra_context:
            extra_section = f"\n\nAdditional Context:\n{extra_context}"

        # Create the comparison task as user message
        user_message = (
            "Please compare the following two candidate summaries.\n"
            f"OPS law: {law_type}\n"
            f"{instruction}\n\n"
            f"Context (what to preserve): {context}"
            f"{original_section}"
            f"{extra_section}\n\n"
            "Evaluate the candidates below on:\n"
            "1. Preservation of oracle-relevant information\n"
            "2. Accuracy and faithfulness\n"
            "3. Completeness vs. conciseness tradeoff"
        )

        return [
            {"role": "user", "content": user_message},
            {"role": "response_1", "content": summary_a},
            {"role": "response_2", "content": summary_b},
        ]

    def _build_completion_prompt(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
    ) -> str:
        """
        Build a formatted prompt for the /completions endpoint.

        Uses NVIDIA's official GenRM chat template format.
        """
        law_instructions = {
            "sufficiency": (
                "Compare which summary better preserves the oracle-relevant "
                "information from the original text."
            ),
            "idempotence": (
                "Compare which summary is more stable under re-summarization. "
                "Use the provided resummaries to judge drift."
            ),
            "merge": (
                "Compare which merged summary better preserves the information "
                "from its child summaries."
            ),
        }
        instruction = law_instructions.get(law_type, law_instructions["sufficiency"])

        original_section = ""
        if original_text.strip():
            original_section = f"\n\nOriginal text being summarized:\n{original_text}"  # Use full text

        extra_section = ""
        if extra_context:
            extra_section = f"\n\n{extra_context}"

        # Build conversation context (the user's summarization task)
        user_task = (
            f"Summarize the following text while preserving: {context}\n"
            f"OPS Law: {law_type} - {instruction}"
            f"{original_section}{extra_section}"
        )

        # NVIDIA's official GenRM template format
        prompt = f"""<|im_start|>user
You are an expert evaluation judge specializing in comparative assessment of LLM responses. You are impartial, rigorous, and consistent. Given the conversation context and two assistant responses to the user's latest query, you will follow the evaluation plan and scoring guidelines exactly as written below.

#### Conversation Context ####
User: {user_task}

#### Responses to be Scored ####
[The Begin of Response 1]
{summary_a}
[The End of Response 1]

[The Begin of Response 2]
{summary_b}
[The End of Response 2]

#### Evaluation Plan ####
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

#### Scoring Guidelines ####
Based on the evaluation plan above, assign scores using these scales:

**Individual Helpfulness Scores (1-5):**
- 5: Extremely Helpful - Completely aligned with what the user was asking for
- 4: Mostly Helpful - Generally useful with minor room for improvement
- 3: Partially Helpful - Misses the overall goal in some way
- 2: Borderline Unhelpful - Mostly doesn't capture what the user wanted
- 1: Not Helpful - Completely missed the essence of the request

**Comparative Ranking (1-6):**
- 1: Response 1 is much better than Response 2
- 2: Response 1 is better than Response 2
- 3: Response 1 is slightly better than Response 2
- 4: Response 2 is slightly better than Response 1
- 5: Response 2 is better than Response 1
- 6: Response 2 is much better than Response 1

#### Output Format ####
Analyze step by step following the evaluation plan, then provide your judgment as JSON:
```json
{{
    "response_1_analysis": "Your detailed analysis of Response 1 based on the evaluation plan",
    "response_2_analysis": "Your detailed analysis of Response 2 based on the evaluation plan",
    "score_1": <1-5>,
    "score_2": <1-5>,
    "ranking": <1-6>
}}
```
<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def compare(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
    ) -> GenRMComparisonResult:
        """
        Compare two summaries using GenRM.

        Args:
            context: Description of what information to preserve
            original_text: Original text being summarized
            summary_a: First candidate summary
            summary_b: Second candidate summary

        Returns:
            GenRMResult with preference and scores
        """
        # Ensure model name is detected (lazy detection if server wasn't ready at init)
        self._ensure_model_name()

        messages = self._build_messages(
            context=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            extra_context=extra_context,
        )

        # Try chat completions first
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                },
                timeout=600,  # 10 min for large GenRM models (235B)
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return self._parse_genrm_response(content)

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # Chat completions not available, try completions endpoint
                logger.debug("Chat completions not available, trying completions endpoint")
                return self._compare_via_completions(
                    context, original_text, summary_a, summary_b, law_type, extra_context
                )
            else:
                logger.error(f"GenRM request failed: {e}")
                return self._error_result(str(e))

        except Exception as e:
            logger.error(f"GenRM request failed: {e}")
            return self._error_result(str(e))

    async def compare_async(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
    ) -> GenRMComparisonResult:
        """
        Async version of compare() - uses batch client when available for better throughput.

        If batch_client is provided during init, uses it for batched requests.
        Otherwise falls back to direct HTTP calls via aiohttp.

        Args:
            context: Description of what information to preserve
            original_text: Original text being summarized
            summary_a: First candidate summary
            summary_b: Second candidate summary
            law_type: OPS law type (sufficiency, idempotence, merge)
            extra_context: Additional context for the comparison

        Returns:
            GenRMResult with preference and scores
        """
        # Ensure model name is detected
        self._ensure_model_name()

        # Use batch client if available (better for tournament mode across multiple docs)
        if self.batch_client is not None:
            from .genrm_batch import GenRMComparisonRequest
            import uuid

            # Build the full context string for the batch client
            full_context = context
            if extra_context:
                full_context = f"{context}\n\nAdditional context:\n{extra_context}"

            request = GenRMComparisonRequest(
                request_id=f"genrm_{uuid.uuid4().hex[:12]}",
                context=full_context,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
            )

            return await self.batch_client.call(request)

        # Fall back to direct aiohttp call if no batch client
        messages = self._build_messages(
            context=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            extra_context=extra_context,
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=aiohttp.ClientTimeout(total=600),  # 10 min for large GenRM models
                ) as resp:
                    data = await resp.json()

                    if resp.status == 200:
                        content = data["choices"][0]["message"]["content"]
                        return self._parse_genrm_response(content)
                    else:
                        error_msg = f"HTTP {resp.status}: {data}"
                        logger.error(f"GenRM request failed: {error_msg}")
                        return self._error_result(error_msg)

        except Exception as e:
            logger.error(f"GenRM async request failed: {e}")
            return self._error_result(str(e))

    async def compare_batch_async(
        self,
        comparisons: List[tuple],
    ) -> List[GenRMComparisonResult]:
        """
        Batch compare multiple summary pairs with true batching when batch_client available.

        Each comparison tuple: (context, original_text, summary_a, summary_b, law_type, extra_context)

        With batch_client: Submits all requests to the batch queue, then awaits all responses.
        This enables true concurrent batching across the entire batch.

        Without batch_client: Falls back to asyncio.gather on individual HTTP calls.

        Args:
            comparisons: List of (context, original_text, summary_a, summary_b, law_type, extra_context) tuples

        Returns:
            List of GenRMComparisonResult (may include GenRMErrorResult for failed comparisons)
        """
        if not comparisons:
            return []

        # True batching with batch client
        if self.batch_client is not None:
            from .genrm_batch import GenRMComparisonRequest
            import uuid

            # Submit all requests to the batch queue
            requests = []
            for comp in comparisons:
                context = comp[0]
                extra_context = comp[5] if len(comp) > 5 else None
                if extra_context:
                    context = f"{context}\n\nAdditional context:\n{extra_context}"

                request = GenRMComparisonRequest(
                    request_id=f"genrm_{uuid.uuid4().hex[:12]}",
                    context=context,
                    original_text=comp[1],
                    summary_a=comp[2],
                    summary_b=comp[3],
                    law_type=comp[4] if len(comp) > 4 else "sufficiency",
                )
                await self.batch_client.submit(request)
                requests.append(request)

            # Await all responses concurrently so one slow request does not
            # serially delay the rest of the batch.
            return await asyncio.gather(
                *(self.batch_client.await_response(request.request_id) for request in requests)
            )

        # Fallback: Use asyncio.gather to run all comparisons concurrently
        tasks = [
            self.compare_async(
                context=comp[0],
                original_text=comp[1],
                summary_a=comp[2],
                summary_b=comp[3],
                law_type=comp[4] if len(comp) > 4 else "sufficiency",
                extra_context=comp[5] if len(comp) > 5 else None,
            )
            for comp in comparisons
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch comparison {i} failed: {result}")
                processed_results.append(self._error_result(str(result)))
            else:
                processed_results.append(result)

        return processed_results

    def _compare_via_completions(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
    ) -> GenRMComparisonResult:
        """
        Compare using the /completions endpoint as fallback.
        """
        prompt = self._build_completion_prompt(
            context, original_text, summary_a, summary_b, law_type, extra_context
        )

        try:
            response = requests.post(
                f"{self.base_url}/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                    "stop": ["<|im_end|>", "<|im_start|>"],
                },
                timeout=600,  # 10 min for large GenRM models (235B)
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["text"]
            return self._parse_genrm_response(content)

        except Exception as e:
            logger.error(f"GenRM completions request failed: {e}")
            return self._error_result(str(e))

    def _error_result(self, error_msg: str, error_type: str = "network") -> GenRMErrorResult:
        """Return an error result distinct from legitimate ties.

        Callers should filter out GenRMErrorResult from training data
        rather than treating them as preferences.
        """
        return GenRMErrorResult(
            error_type=error_type,
            error_message=error_msg,
            raw_response="",
        )

    def _parse_genrm_response(self, content: str) -> GenRMResult:
        """
        Parse GenRM response to extract scores and preference.

        First tries to parse the official JSON output format, then falls back
        to regex-based extraction for other response formats.
        """
        helpfulness_a = 3.0
        helpfulness_b = 3.0
        ranking_score = 3
        reasoning = ""

        # Try to parse JSON output format first (official template)
        json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
        if not json_match:
            # Try without code block markers
            json_match = re.search(r'(\{[^{}]*"score_1"[^{}]*\})', content, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                helpfulness_a = float(result.get('score_1', 3))
                helpfulness_b = float(result.get('score_2', 3))
                ranking_score = int(result.get('ranking', 3))
                reasoning = result.get('response_1_analysis', '') + '\n' + result.get('response_2_analysis', '')
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug(f"JSON parsing failed, falling back to regex: {e}")

        # Fallback: regex-based extraction
        if helpfulness_a == 3.0 and helpfulness_b == 3.0 and ranking_score == 3:
            # Extract numbers from response
            numbers = re.findall(r"\b([1-5](?:\.[0-9])?)\b", content)
            if len(numbers) >= 2:
                try:
                    helpfulness_a = float(numbers[0])
                    helpfulness_b = float(numbers[1])
                except ValueError:
                    pass

            # Look for ranking score (1-6)
            ranking_pattern = r"(?:ranking|overall|preference)[^\d]*([1-6])"
            ranking_match = re.search(ranking_pattern, content, re.IGNORECASE)
            if ranking_match:
                ranking_score = int(ranking_match.group(1))
            else:
                # Infer from helpfulness
                if helpfulness_a > helpfulness_b + 0.5:
                    ranking_score = 2  # A is better
                elif helpfulness_b > helpfulness_a + 0.5:
                    ranking_score = 5  # B is better
                else:
                    ranking_score = 3  # Roughly equal

        # Determine preference from ranking score (1-6 scale per Nemotron paper)
        # 1,2 = A wins, 3,4 = tie, 5,6 = B wins (symmetric)
        # NOTE: This is the canonical implementation. PreferenceEngine.RANKING_SCORE_THRESHOLD
        # strategy provides the same logic for use outside GenRMJudge.
        if ranking_score <= 2:
            preferred = "A"
            confidence = (3 - ranking_score) * 0.3 + 0.4  # 0.7-1.0
        elif ranking_score >= 5:
            preferred = "B"
            confidence = (ranking_score - 4) * 0.3 + 0.4  # 0.7-1.0
        else:  # 3 or 4 = tie (symmetric middle ground)
            preferred = "tie"
            confidence = 0.5

        return GenRMResult(
            preferred=preferred,
            ranking_score=ranking_score,
            helpfulness_a=helpfulness_a,
            helpfulness_b=helpfulness_b,
            reasoning=reasoning if reasoning else content,
            confidence=confidence,
            raw_response=content,
        )


def create_genrm_comparison_prompt(
    rubric: str,
    original_text: str,
    summary_a: str,
    summary_b: str,
    law_type: str = "sufficiency",
    extra_context: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Create a GenRM-format prompt for summary comparison.

    This is a convenience function for direct API usage.

    Returns:
        List of messages in GenRM format
    """
    original_section = ""
    if original_text.strip():
        original_section = f"\n\nOriginal text:\n{original_text}"

    extra_section = ""
    if extra_context:
        extra_section = f"\n\nAdditional context:\n{extra_context}"

    return [
        {
            "role": "user",
            "content": (
                "Compare these two candidate summaries.\n"
                f"OPS law: {law_type}\n"
                f"Preservation criteria: {rubric}"
                f"{original_section}"
                f"{extra_section}\n\n"
                "Evaluate which candidate better preserves the information specified "
                "in the criteria."
            ),
        },
        {"role": "response_1", "content": summary_a},
        {"role": "response_2", "content": summary_b},
    ]
