"""Regression coverage for tool-argument SSRF and Wikipedia request boundaries."""

from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlsplit

import pytest

from src.tools.implementations.wikipedia_tool import WikipediaTool

ENDPOINT = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"


@pytest.fixture
def tool():
    instance = WikipediaTool({"endpoint_url": ENDPOINT})
    instance.make_request = AsyncMock(
        return_value={
            "title": "Example",
            "extract": "Summary",
            "content_urls": {
                "desktop": {"page": "https://en.wikipedia.org/wiki/Example"}
            },
        }
    )
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lang",
    [
        "127.0.0.1:8443/flexo-ssrf-proof",
        "attacker.example/",
        "attacker.example#",
        "attacker.example?",
        "en@attacker.example/",
        "en.wikipedia.org.attacker.example/",
        "[::1]:8443/",
        "169.254.169.254/",
        "en\\attacker.example/",
        "en%2fattacker.example",
        "en\n",
        "en\r\n",
        "en\x00",
        " en",
        "EN",
        "ｅｎ",
        "",
        "en.",
        "en-",
        None,
        123,
        True,
        [],
        {},
    ],
)
async def test_rejects_invalid_language_before_network(tool, lang):
    with pytest.raises(ValueError, match="language code"):
        await tool.execute(query="Medicare", lang=lang)
    tool.make_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lang", ["en", "es", "fr", "ast", "simple", "be-tarask", "zh-min-nan"]
)
async def test_valid_languages(tool, lang):
    result = await tool.execute(query="Example", lang=lang)
    request = tool.make_request.call_args.kwargs
    assert urlsplit(request["endpoint_url"]).netloc == f"{lang}.wikipedia.org"
    assert request["allow_redirects"] is False
    assert "summary: Summary" in result.result
    assert "https://en.wikipedia.org/wiki/Example" in result.result


@pytest.mark.asyncio
async def test_default_language_and_title_encoding(tool):
    await tool.execute(query="AC/DC ?#% café")
    assert tool.make_request.call_args.kwargs["endpoint_url"] == (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        "AC%2FDC%20%3F%23%25%20caf%C3%A9"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [None, "", "  ", 42, [], {}])
async def test_rejects_invalid_query_before_network(tool, query):
    with pytest.raises(ValueError, match="non-empty string"):
        await tool.execute(query=query)
    tool.make_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://attacker.example/{encoded_query}",
        "https://{lang}.wikipedia.org.attacker.example/{encoded_query}",
        "https://{lang}.wikipedia.org@attacker.example/{encoded_query}",
        "https://user:password@{lang}.wikipedia.org/{encoded_query}",
        "https://{lang}.wikipedia.org:8443/{encoded_query}",
        "http://{lang}.wikipedia.org/{encoded_query}",
        "https://wikipedia.org/{encoded_query}",
    ],
)
async def test_rejects_unsafe_endpoint(tool, endpoint):
    tool.endpoint = endpoint
    with pytest.raises(ValueError, match="Wikipedia endpoint"):
        await tool.execute(query="Example")
    tool.make_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_redirect_is_not_followed_by_http_client(status):
    tool = WikipediaTool({"endpoint_url": ENDPOINT})
    response = AsyncMock()
    response.status = status
    response.headers = {"Location": "https://127.0.0.1/internal"}
    response.text.return_value = "Redirect"
    session = MagicMock()
    session.request.return_value.__aenter__ = AsyncMock(return_value=response)
    with patch("src.tools.core.base_rest_tool.aiohttp.ClientSession") as client:
        client.return_value.__aenter__ = AsyncMock(return_value=session)
        await tool.execute(query="Example")
    session.request.assert_called_once()
    assert session.request.call_args.kwargs["allow_redirects"] is False
    assert (
        urlsplit(session.request.call_args.kwargs["url"]).hostname == "en.wikipedia.org"
    )


@pytest.mark.asyncio
async def test_model_generated_arguments_are_rejected_during_dispatch(tool):
    import logging
    from types import SimpleNamespace

    from src.agent.chat_agent_streaming import StreamingChatAgent
    from src.data_models.chat_completions import ToolCall

    agent = StreamingChatAgent.__new__(StreamingChatAgent)
    agent.logger = logging.getLogger("wikipedia-security-test")
    agent.tool_registry = SimpleNamespace(get_tool=AsyncMock(return_value=tool))
    context = SimpleNamespace(
        current_tool_call=[
            ToolCall(
                id="call-test",
                type="function",
                function={
                    "name": "wikipedia",
                    "arguments": (
                        '{"query":"Medicare",'
                        '"lang":"127.0.0.1:8443/flexo-ssrf-proof"}'
                    ),
                },
            )
        ]
    )
    results = await agent._execute_tools_concurrently(context)
    assert len(results) == 1
    assert isinstance(results[0], ValueError)
    tool.make_request.assert_not_called()
