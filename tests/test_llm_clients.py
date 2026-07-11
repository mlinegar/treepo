from __future__ import annotations

from treepo.llm import OpenAICompatibleChatClient, build_chat_client


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.posts = []

    def get(self, url, **kwargs):
        return FakeResponse({"data": [{"id": "served-chat"}]})

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return FakeResponse({"choices": [{"message": {"content": "42"}}]})


def test_openai_compatible_chat_client_resolves_and_completes() -> None:
    session = FakeSession()
    client = OpenAICompatibleChatClient(
        api_base="http://localhost:8000/v1",
        model="placeholder",
        api_key="EMPTY",
        session=session,
    )

    text = client.complete_chat([{"role": "user", "content": "Return 42."}], max_tokens=4)

    assert text == "42"
    assert client.model == "served-chat"
    url, kwargs = session.posts[0]
    assert url == "http://localhost:8000/v1/chat/completions"
    assert kwargs["json"]["model"] == "served-chat"
    assert kwargs["headers"]["Authorization"] == "Bearer EMPTY"


def test_build_chat_client_accepts_vllm_engine_alias() -> None:
    session = FakeSession()
    client = build_chat_client(
        "vllm",
        api_base="http://localhost:8000",
        model="served-chat",
        session=session,
    )

    assert client.models_url == "http://localhost:8000/v1/models"
    assert client.complete_chat([{"role": "user", "content": "Return 42."}]) == "42"
