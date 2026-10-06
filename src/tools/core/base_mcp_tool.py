# src/tools/core/base_mcp_tool.py

from typing import Optional, Dict
from abc import ABC, abstractmethod
from mcp.types import Tool as MCPToolDefinition

from src.data_models.tools import ToolResponse
from src.data_models.agent import StreamContext
from src.tools.core.base_tool import BaseTool


class BaseMCPTool(BaseTool, ABC):
    """
    Abstract base class for MCP-based tools.

    This class bridges an MCP tool definition (provided by the MCP server)
    with a Flexo tool implementation. It automatically initializes the tool's
    name, description, and parameter definitions from the MCP tool definition.

    Subclasses must implement the execute and parse_output methods.
    """

    def __init__(self, mcp_tool_def: MCPToolDefinition, config: Optional[Dict] = None):
        super().__init__(config=config)
        self.mcp_tool_def = mcp_tool_def
        self.name = mcp_tool_def.name
        self.description = mcp_tool_def.description or ""
        self.parameters = {
            "properties": mcp_tool_def.input_schema.get("properties", {}),
            "required": mcp_tool_def.input_schema.get("required", []),
            "additionalProperties": mcp_tool_def.input_schema.get(
                "additionalProperties", None
            ),
        }

    @abstractmethod
    async def execute(
        self, context: Optional[StreamContext] = None, **kwargs
    ) -> ToolResponse:
        """
        Execute the MCP-based tool's functionality.

        Subclasses should implement this method with the actual logic.
        """
        pass

    @abstractmethod
    def parse_output(self, output: str):
        """
        Parse the tool's output from a raw string into a structured response.

        Subclasses should implement this method based on the tool's specifics.
        """
        pass
