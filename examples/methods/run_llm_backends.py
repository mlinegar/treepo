#!/usr/bin/env python3
"""Self-contained LLM backend adapter example.

The same ``treepo.fit(..., family="llm")`` surface can use:

* OpenAI-compatible HTTP endpoints such as vLLM or SGLang;
* a direct local callable such as a Hugging Face Transformers pipeline;
* a DSPy-style callable/program.

This example is runnable without a model server: it uses tiny fake backends that
match those interfaces, so the package contracts are exercised without pulling
heavy serving stacks into the example.
"""

from __future__ import annotations

from typing import Any, Mapping

from example_setup import parse_output_dir, toy_optimizer_trees_and_preferences, write_json


class FakeResponse:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Mapping[str, Any]:
        return self.payload


class FakeOpenAICompatibleSession:
    """Tiny requests-like session for the OpenAI-compatible example path."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        del url, kwargs
        return FakeResponse({"data": [{"id": "example-chat-model"}]})

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        payload = dict(kwargs.get("json") or {})
        self.calls.append({"url": url, "payload": payload})
        text = _messages_text(payload.get("messages") or ())
        return FakeResponse({"choices": [{"message": {"content": _score_from_text(text)}}]})


def transformers_style_predict_fn(*, prompt: str, **kwargs: Any) -> str:
    """Adapter shape for ``transformers.pipeline("text-generation")``.

    Real usage would call the pipeline here and parse ``generated_text``. The
    local example returns the same shape so it stays fast and self-contained.
    """

    del kwargs
    pipeline_output = [{"generated_text": _score_from_text(prompt)}]
    return str(pipeline_output[0]["generated_text"])


class ToyDSPyProgram:
    """Callable program shape accepted by ``family="dspy"``."""

    def __call__(self, *, prompt: str, **kwargs: Any) -> str:
        del kwargs
        return _score_from_text(prompt)


def _messages_text(messages: Any) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping):
            parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


def _score_from_text(text: str) -> str:
    return "-0.5" if "negative" in str(text).lower() else "0.7"


def _fit_openai_compatible(output_dir):
    from treepo import fit

    trees, preferences = toy_optimizer_trees_and_preferences()
    session = FakeOpenAICompatibleSession()
    result = fit(
        {
            "family": "llm",
            "train_data": trees,
            "eval_data": trees,
            "preference_data": preferences,
            "backend_config": {
                "output_dir": str(output_dir / "openai_compatible"),
                "api_base": "http://localhost:8000/v1",
                "api_key": "EMPTY",
                "model": "example-chat-model",
                "session": session,
                "prompt_template": "Return only one score for this document:\n{text}",
                "min_score": -1.0,
                "max_score": 1.0,
            },
            "axis": {"max_iterations": 1, "axis_value": 0},
        }
    )
    return result, session


def _fit_transformers_callable(output_dir):
    from treepo import fit

    trees, preferences = toy_optimizer_trees_and_preferences()
    result = fit(
        {
            "family": "llm",
            "train_data": trees,
            "eval_data": trees,
            "preference_data": preferences,
            "backend_config": {
                "output_dir": str(output_dir / "transformers_callable"),
                "model": "local-transformers-pipeline",
                "predict_fn": transformers_style_predict_fn,
                "prompt_template": "Return only one score for this document:\n{text}",
                "min_score": -1.0,
                "max_score": 1.0,
            },
            "axis": {"max_iterations": 1, "axis_value": 0},
        }
    )
    return result


def _fit_dspy_program(output_dir):
    from treepo import fit

    trees, preferences = toy_optimizer_trees_and_preferences()
    result = fit(
        {
            "family": "dspy",
            "train_data": trees,
            "eval_data": trees,
            "preference_data": preferences,
            "backend_config": {
                "output_dir": str(output_dir / "dspy_program"),
                "lm_config": {"model": "example-chat-model", "api_base": "http://localhost:8000/v1"},
                "dspy_program": ToyDSPyProgram(),
                "prompt_template": "Return only one score for this document:\n{text}",
                "min_score": -1.0,
                "max_score": 1.0,
            },
            "axis": {"max_iterations": 1, "axis_value": 0},
        }
    )
    return result


def main() -> int:
    output_dir = parse_output_dir()
    openai_result, session = _fit_openai_compatible(output_dir)
    transformers_result = _fit_transformers_callable(output_dir)
    dspy_result = _fit_dspy_program(output_dir)

    payload = {
        "backends": {
            "openai_compatible": {
                "examples": ["vLLM", "SGLang", "OpenAI-compatible hosted APIs"],
                "status": openai_result.status,
                "metrics": openai_result.metrics,
                "http_calls": len(session.calls),
            },
            "transformers_callable": {
                "examples": ["Hugging Face Transformers pipeline", "custom local callable"],
                "status": transformers_result.status,
                "metrics": transformers_result.metrics,
            },
            "dspy_program": {
                "examples": ["DSPy program", "DSPy-optimized prompt module"],
                "status": dspy_result.status,
                "metrics": dspy_result.metrics,
            },
        },
        "results": {
            "openai_compatible": openai_result.to_dict(),
            "transformers_callable": transformers_result.to_dict(),
            "dspy_program": dspy_result.to_dict(),
        },
    }
    result_path = output_dir / "llm_backends_result.json"
    write_json(result_path, payload)
    print(
        "status=success "
        f"openai_compatible={openai_result.status} "
        f"transformers_callable={transformers_result.status} "
        f"dspy_program={dspy_result.status} "
        f"output={result_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
