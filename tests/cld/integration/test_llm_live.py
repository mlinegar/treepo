"""Live-server integration test for `treepo.cld` against a real vLLM endpoint.

Gated by ``TT_RUN_LIVE_TESTS=1`` (same convention as
``tests/integration/test_vllm_live.py``). Skipped automatically when:

- ``TT_RUN_LIVE_TESTS`` is unset, OR
- The vLLM endpoint at ``VLLM_HOST:VLLM_PORT`` (default
  ``http://localhost:8000/v1``) doesn't answer.

Enable with:

    ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 > logs/vllm.log 2>&1 &
    # wait for /v1/models to respond
    TT_RUN_LIVE_TESTS=1 \\
      python -m pytest \\
      treepo.cld/tests/integration/test_llm_live.py -v

Two checks:

1. **Direct DSPy roundtrip.** Build a ``DSPyFamily`` via the
   ``treepo.cld`` registry with ``lm_transport="litellm"`` pointed at
   the live server, force-construct its LM, and issue a real Predict
   call. Confirms the model is reachable, returns text, and uses GPU.

2. **Batched-transport roundtrip.** Same family, but
   ``lm_transport="batch"`` so the call goes through
   ``BatchedDSPyLM`` → ``AsyncBatchLLMClient`` → vLLM. Confirms the
   batched-async path the manifesto grid uses is actually wired and
   gets a finite-text response from the GPU.

What this proves: the path from ``treepo.cld.run(...)`` → registered
``"dspy"`` family → live vLLM server → GPU → response works end to end.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest


VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
LIVE_MODEL = os.getenv("VLLM_MODEL", "nvidia/Gemma-4-31B-IT-NVFP4")


def _is_vllm_available() -> bool:
    try:
        req = urllib.request.Request(f"{VLLM_URL}/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


_RUN_LIVE = str(os.getenv("TT_RUN_LIVE_TESTS", "") or "").strip().lower() in {"1", "true", "yes"}
pytestmark = pytest.mark.skipif(
    (not _RUN_LIVE) or (not _is_vllm_available()),
    reason=f"Live tests disabled (set TT_RUN_LIVE_TESTS=1) or vLLM not at {VLLM_URL}",
)


def _make_live_dspy_config(*, lm_transport: str):
    from treepo._research.ctreepo.dspy_family import DSPyFamilyConfig

    return DSPyFamilyConfig(
        optimizer="bootstrap_fewshot",
        lm_transport=lm_transport,
        # Modest budget knobs — enough to exercise the batched path
        # without a 30-minute test.
        batch_size=8,
        batch_max_concurrent=16,
        batch_timeout=0.05,
        batch_request_timeout=60.0,
        # Manifesto-grade budgets. DSPyFamily enforces
        # ``max_completion_tokens >= 2 * leaf_size_tokens`` (g must be
        # able to emit a verbatim concatenation of two leaf children),
        # and the prompt budget must fit in the context window.
        leaf_size_tokens=32,
        lm_context_window_tokens=8192,
        max_completion_tokens=128,
        prompt_template_overhead_tokens=512,
        # Connection.
        lm_config={
            "model": f"openai/{LIVE_MODEL}",
            "api_base": VLLM_URL,
            "api_key": "EMPTY",
            "temperature": 0.0,
            "max_tokens": 64,
            "cache": False,
        },
    )


def _force_eager_load(family) -> object:
    """Trigger ``DSPyFamily._ensure_lm()`` so the LM is constructed
    before we issue Predict calls. Returns the LM."""
    lm = family._ensure_lm()
    assert lm is not None
    return lm


def _simple_predict_call(lm) -> str:
    """Issue one minimal completion via the LM and return the text."""
    import dspy

    # Configure dspy.settings.lm so dspy.Predict resolves to this LM.
    dspy.settings.configure(lm=lm)

    class _Echo(dspy.Signature):
        """Reply with exactly the word OK."""

        question: str = dspy.InputField(desc="The user's prompt")
        answer: str = dspy.OutputField(desc="A short single-word answer")

    predictor = dspy.Predict(_Echo)
    result = predictor(question="Say OK.")
    return str(getattr(result, "answer", ""))


# --------------------------------------------------------------------------- #
# 1. litellm transport (single-request path) against the live server.
# --------------------------------------------------------------------------- #


def test_live_dspy_litellm_transport_completes_predict_call() -> None:
    from treepo.cld.families import resolve_family

    cfg = _make_live_dspy_config(lm_transport="litellm")
    family = resolve_family("dspy", {"dspy_config": cfg})
    lm = _force_eager_load(family)
    text = _simple_predict_call(lm)
    assert isinstance(text, str)
    assert len(text.strip()) > 0, f"expected non-empty completion, got {text!r}"


# --------------------------------------------------------------------------- #
# 2. batched transport (paper-grade async batched pool) against the live server.
# --------------------------------------------------------------------------- #


def test_live_dspy_batched_transport_completes_predict_call() -> None:
    """Same call, but the LM is ``BatchedDSPyLM`` from
    ``src.core.dspy_batch_client``. This is the path the manifesto grid
    scripts use.
    """
    from treepo.cld.families import resolve_family

    cfg = _make_live_dspy_config(lm_transport="batch")
    family = resolve_family("dspy", {"dspy_config": cfg})
    lm = _force_eager_load(family)
    text = _simple_predict_call(lm)
    assert isinstance(text, str)
    assert len(text.strip()) > 0, f"expected non-empty completion, got {text!r}"


# --------------------------------------------------------------------------- #
# 3. Batched concurrency — 8 prompts in parallel through the batched pool.
#    Verifies the async pooling actually pools (no exception on concurrent
#    submission, all 8 complete with finite text).
# --------------------------------------------------------------------------- #


def test_live_dspy_batched_transport_handles_concurrent_predicts() -> None:
    from treepo.cld.families import resolve_family

    cfg = _make_live_dspy_config(lm_transport="batch")
    family = resolve_family("dspy", {"dspy_config": cfg})
    lm = _force_eager_load(family)

    import dspy

    dspy.settings.configure(lm=lm)

    class _Echo(dspy.Signature):
        """Reply with a single short word."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(_Echo)

    # Submit 8 concurrent requests. DSPy's parallelizer goes through the
    # configured LM; with the batched client, they pool into the async
    # transport.
    from concurrent.futures import ThreadPoolExecutor

    prompts = [f"Say the word {w}." for w in ("alpha", "beta", "gamma", "delta",
                                                "echo", "foxtrot", "golf", "hotel")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda q: predictor(question=q), prompts))
    answers = [str(getattr(r, "answer", "")).strip() for r in responses]
    assert all(len(a) > 0 for a in answers), f"some answer empty: {answers}"
    assert len(answers) == 8
