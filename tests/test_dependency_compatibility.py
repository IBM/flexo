"""Offline HTTP smoke tests for the upgraded framework and model SDKs."""

import importlib
import json
from unittest.mock import patch

import httpx
import httpx2
import pytest
from anthropic import AsyncAnthropic
from mistralai.client import Mistral
from openai import AsyncOpenAI

from src.data_models.chat_completions import UserMessage
from src.tools.implementations.wikipedia_tool import WikipediaTool


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", ["openai", "anthropic", "mistral_ai"])
async def test_model_sdk_stream(vendor, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    module = importlib.import_module(f"src.llm.adapters.{vendor}_adapter")
    calls = []
    if vendor == "anthropic":
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        stream = f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"
        factory, adapter_class = "AsyncAnthropic", module.AnthropicAdapter
    else:
        event = {
            "id": "completion-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
        }
        stream = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        if vendor == "openai":
            factory, adapter_class = "AsyncOpenAI", module.OpenAIAdapter
        else:
            factory, adapter_class = "Mistral", module.MistralAIAdapter

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx2.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=stream
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as transport:
        if vendor == "anthropic":
            sdk = AsyncAnthropic(api_key="test-key", http_client=transport)
        elif vendor == "openai":
            sdk = AsyncOpenAI(api_key="test-key", http_client=transport)
        else:
            sdk = Mistral(api_key="test-key", async_client=transport)
        with patch.object(module, factory, return_value=sdk):
            adapter = adapter_class("test-model")
        tool = WikipediaTool(
            {"endpoint_url": "https://{lang}.wikipedia.org/{encoded_query}"}
        )
        chunks = [
            chunk
            async for chunk in adapter.gen_chat_sse_stream(
                [UserMessage(content="Hello")], tools=[tool.get_definition()]
            )
        ]
    assert any(chunk.choices[0].delta.content == "Hello" for chunk in chunks)
    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["tools"]


@pytest.mark.asyncio
async def test_api_auth_and_streaming(monkeypatch):
    monkeypatch.setenv("FLEXO_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_API_KEY", "true")
    # The application currently constructs the agent inside the serving event loop.
    from src.main import app
    from src.api.routes.chat_completions_api import get_streaming_agent
    from src.api.sse_models import SSEChunk

    class Agent:
        async def stream_step(self, **kwargs):
            yield SSEChunk.make_text_chunk("Hello")
            yield await SSEChunk.make_stop_chunk()

    app.dependency_overrides[get_streaming_agent] = lambda: Agent()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            body = {"messages": [{"role": "user", "content": "Hello"}]}
            denied = await client.post("/v1/chat/completions", json=body)
            assert denied.status_code == 403
            accepted = await client.post(
                "/v1/chat/completions", json=body, headers={"X-API-KEY": "test-key"}
            )
            assert accepted.status_code == 200
            assert accepted.headers["content-type"].startswith("text/event-stream")
            assert '"content":"Hello"' in accepted.text
    finally:
        app.dependency_overrides.clear()
