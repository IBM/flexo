# src/data_models/tools.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal


class FunctionParameters(BaseModel):
    """Schema for function parameters following JSON Schema specification.

    This class defines the structure for function parameters using JSON Schema.
    It specifies the type, properties, required fields, and whether additional
    properties are allowed.

    Attributes:
        type (str): The type of the parameters object, always "object".
        properties (Dict[str, Dict[str, Any]]): Mapping of parameter names to their
            JSON Schema definitions.
        required (Optional[List[str]]): List of required parameter names.
        additionalProperties (Optional[bool]): Whether additional properties beyond
            those specified are allowed.

    Example:
        ```python
        parameters = FunctionParameters(
            type="object",
            properties={
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                }
            },
            required=["location"]
        )
        ```
    """
    type: str = "object"
    properties: Dict[str, Dict[str, Any]]
    required: Optional[List[str]] = None
    additionalProperties: Optional[bool] = None


class Function(BaseModel):
    """Represents a function that can be called by the model.

    Defines the structure of a callable function, including its name,
    description, parameters, and validation settings.

    Attributes:
        name (str): Function identifier, must be 1-64 characters and contain only
            alphanumeric characters, underscores, and hyphens.
        description (Optional[str]): Human-readable description of what the
            function does.
        parameters (Optional[FunctionParameters]): Schema defining the function's
            parameters.
        strict (Optional[bool]): Whether to enforce strict parameter validation.
            Defaults to False.

    Example:
        ```python
        function = Function(
            name="get_weather",
            description="Get current weather for a location",
            parameters=FunctionParameters(...),
            strict=True
        )
        ```
    """
    name: str = Field(..., max_length=64, pattern="^[a-zA-Z0-9_-]+$")
    description: Optional[str] = None
    parameters: Optional[FunctionParameters] = None
    strict: Optional[bool] = Field(default=False)


class Tool(BaseModel):
    """Represents a tool that the model can use.

    A tool is a wrapper around a function that can be called by the model.
    Currently, only function-type tools are supported.

    Attributes:
        type (Literal["function"]): The type of tool, currently only "function"
            is supported.
        function (Function): The function definition for this tool.

    Example:
        ```python
        tool = Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get current weather",
                parameters=FunctionParameters(...),
                strict=True
            )
        )
        ```
    """
    type: Literal["function"] = "function"
    function: Function


class ToolsList(BaseModel):
    """Container for a list of tools.

    Manages a collection of tools that can be provided to the model,
    with a maximum limit of 128 tools.

    Attributes:
        tools (List[Tool]): List of tool definitions, maximum length of 128.
    """
    tools: List[Tool] = Field(..., max_length=128)


class ToolResponse(BaseModel):
    """Represents the standardized output of a tool execution."""

    result: str = Field(
        description="The main output or result of the tool execution"
    )
    context: Optional[Dict] = Field(
        default=None,
        description="Additional contextual information or metadata about the execution"
    )
