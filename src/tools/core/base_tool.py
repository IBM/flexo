# src/tools/core/base_tool.py

from typing import Optional, Dict
from abc import abstractmethod
from src.data_models.tools import ToolResponse
from src.data_models.agent import StreamContext
from src.data_models.tools import Tool, Function, FunctionParameters


class BaseTool:
    """Abstract base class for all tools in the system.

    Provides the foundation for tool implementation with standard interfaces
    for execution, definition retrieval, and output parsing.
    """
    name: str

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.description = None
        self.parameters = {}
        self.strict = False

    @abstractmethod
    async def execute(self, context: Optional[StreamContext] = None, **kwargs) -> ToolResponse:
        """Execute the tool's main functionality."""
        pass

    def get_definition(self) -> Tool:
        """Get the tool's OpenAI-compatible definition.

        Returns:
            Tool: Tool definition including type, function details and parameters.
        """
        return Tool(
            type="function",
            function=Function(
                name=self.name,
                description=self.description,
                parameters=FunctionParameters(
                    type="object",
                    properties=self.parameters.get("properties", {}),
                    required=self.parameters.get("required", []),
                    additionalProperties=self.parameters.get("additionalProperties", None)
                ),
                strict=self.strict
            )
        )

    @abstractmethod
    def parse_output(self, output: str):
        """Parse the tool's output."""
        pass

    def get_tool_specific_instruction(self) -> str:
        """Get formatted tool-specific instruction."""
        return ""
