"""Exercise the installed MCP SDK over a real, local stdio transport."""

import asyncio
import sys

import pytest

from src.mcp.client import FlexoMCPClient
from src.mcp.mcp_tool_adapter import convert_mcp_tool_to_flexo_tool


@pytest.mark.asyncio
async def test_stdio_tools_and_notifications(tmp_path):
    server = tmp_path / "server.py"
    server.write_text("""
from mcp.server.mcpserver import MCPServer, Context

server = MCPServer("flexo-compatibility-test")

@server.tool()
async def echo(text: str, ctx: Context) -> str:
    await ctx.session.send_tool_list_changed()
    return text

server.run()
""")
    client = FlexoMCPClient(
        {"transport": "stdio", "command": sys.executable, "args": [str(server)]}
    )
    changed = asyncio.Event()
    events = []

    async def on_change(event):
        events.append(event)
        changed.set()

    client.observer.observer.subscribe(on_change)
    async with asyncio.timeout(20):
        async with client:
            listed = await client.list_tools()
            assert [tool.name for tool in listed.tools] == ["echo"]
            converted = convert_mcp_tool_to_flexo_tool(listed.tools[0])
            assert converted.parameters["properties"]["text"]["type"] == "string"
            result = await client.call_tool("echo", {"text": "hello"})
            assert result.content[0].text == "hello"
            await changed.wait()
            assert events[0].new_tool_defs[0].name == "echo"
    assert client._connected is False
