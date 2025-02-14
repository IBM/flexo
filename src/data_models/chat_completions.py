# src/data_models/chat_completions.py

import json
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Union, Optional, Literal, Annotated, Dict


class MessageBase(BaseModel):
    """Base class for all message types in the chat completion system.

    This class serves as the foundation for all message types, enforcing a common structure
    and validation rules. It uses Pydantic's strict mode to forbid extra attributes.

    Attributes:
        role (str): The role of the message sender. Must be implemented by child classes.

    Note:
        This class should not be used directly but rather inherited by specific message types.
    """
    role: str

    model_config = ConfigDict(extra='forbid')


class UserTextContent(BaseModel):
    """Represents the text content structure for user messages.

    This model defines the format for text-based content in user messages, ensuring
    proper typing and validation.

    Attributes:
        type (Literal["text"]): Content type identifier, always set to "text".
        text (str): The actual text content of the user message.
    """
    type: Literal["text"] = Field(default="text", description="Content type, fixed to 'text' for user messages")
    text: str = Field(..., description="The text content of the user message")


class UserImageURLContent(BaseModel):
    """Represents the image URL content structure for user messages.

    This model defines the format for image-based content in user messages, supporting
    base64 encoded images with configurable processing detail levels.

    Attributes:
        type (Literal["image_url"]): Content type identifier, always set to "image_url".
        image_url (dict): Dictionary containing a 'url' field with a base64 encoded image string.
        detail (Optional[Literal["low", "high", "auto"]]): Processing detail level for the image.
            Defaults to "auto".
    """
    type: Literal["image_url"] = Field(
        default="image_url",
        description="Content type, fixed to 'image_url' for user messages"
    )
    image_url: dict = Field(
        ...,
        description="The image URL as a dictionary containing a 'url' field with a base64 encoded string"
    )
    detail: Optional[Literal["low", "high", "auto"]] = Field(
        default="auto",
        description="Detail level for image processing"
    )


class UserMessage(MessageBase):
    """Represents a message from the user in the chat completion system.

    This model handles both simple text messages and complex content types including
    images. It supports single string content or a list of mixed content types.

    Attributes:
        role (Literal["user"]): Role identifier, always set to "user".
        content (Union[str, List[Union[UserTextContent, UserImageURLContent]]]):
            The message content, either as a simple string or a list of content objects.
    """
    role: Literal["user"] = Field(default="user", description="Role is fixed to 'user' for user messages")
    content: Union[str, List[Union[UserTextContent, UserImageURLContent]]] = Field(
        ..., description="String or detailed content of the user message"
    )


class FunctionDetail(BaseModel):
    """Defines the structure for function call details in tool calls.

    This model contains the essential information needed to execute a function
    through the tool calling system.

    Attributes:
        name (str): The name of the function to be called.
        arguments (Dict): Dictionary of arguments to be passed to the function.
    """
    name: str = Field(..., description="Name of the function")
    arguments: Dict = Field(..., description="Arguments for the function")


class ToolCall(BaseModel):
    """Represents a tool call made by the assistant.

    This model handles the structure and formatting of tool calls, including
    custom serialization of function arguments to JSON.

    Attributes:
        id (str): Unique identifier for the tool call.
        type (Literal["function"]): Type of tool call, currently only supports "function".
        function (FunctionDetail): Detailed information about the function to be called.

    Methods:
        model_dump(*args, **kwargs) -> dict:
            Custom serialization method that converts function arguments to JSON string.

        format_tool_calls() -> str:
            Formats the tool call as a JSON array string for API compatibility.
    """
    id: str = Field(..., description="ID of the tool call")
    type: Literal["function"] = Field(default="function", description="Tool type, currently only 'function' is allowed")
    function: FunctionDetail = Field(..., description="Details of the function call, including name and arguments")

    def model_dump(self, *args, **kwargs) -> dict:
        """Custom model_dump to convert 'arguments' in function to a JSON string."""
        # Call the original model_dump
        data = super().model_dump(*args, **kwargs)

        # Convert 'arguments' to a JSON string within 'function'
        if "function" in data:
            data["function"]["arguments"] = json.dumps(data["function"]["arguments"])

        return data

    def format_tool_calls(self) -> str:
        """Format tool call as a JSON array string."""
        formatted_call = {
            "name": self.function.name,
            "arguments": self.function.arguments
        }
        return json.dumps(formatted_call)


class AssistantMessage(MessageBase):
    """Represents a message from the assistant in the chat completion system.

    This model handles various types of assistant responses, including regular messages,
    tool calls, and refusals. It includes custom serialization logic to handle None values.

    Attributes:
        role (Literal["assistant"]): Role identifier, always set to "assistant".
        content (Optional[Union[str, List[dict]]]): The content of the assistant's message.
        refusal (Optional[str]): Optional refusal message if the assistant declines to respond.
        tool_calls (Optional[List[ToolCall]]): List of tool calls made by the assistant.

    Methods:
        model_dump(*args, **kwargs) -> dict:
            Custom serialization method that excludes None values and properly formats tool calls.
    """
    role: Literal["assistant"] = Field(
        default="assistant", description="Role is fixed to 'assistant' for assistant messages"
    )
    content: Optional[Union[str, List[dict]]] = Field(None, description="The content of the assistant message")
    refusal: Optional[str] = Field(None, description="The refusal message by the assistant")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="List of tool calls made by the assistant")

    def model_dump(self, *args, **kwargs) -> dict:
        """Custom model_dump that excludes fields with None values and calls model_dump on nested ToolCall models."""
        data = super().model_dump(*args, **kwargs)
        if self.tool_calls:
            data["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        return {key: value for key, value in data.items() if value is not None}


class SystemMessage(MessageBase):
    """Represents a system message in the chat completion system.

    This model handles system-level instructions and context that guide the
    conversation behavior.

    Attributes:
        role (Literal["system"]): Role identifier, always set to "system".
        content (str): The content of the system message.
    """
    role: Literal["system"] = Field(default="system", description="Role is fixed to 'system' for system messages")
    content: str = Field(..., description="The content of the system message")


class ToolMessage(BaseModel):
    """Represents a message from a tool in the chat completion system.

    This model handles responses from tool calls, including function executions
    and their results.

    Attributes:
        role (Literal["tool"]): Role identifier, always set to "tool".
        name (str): The name of the tool that generated the message.
        content (str): The content/result of the tool execution.
        tool_call_id (Optional[str]): Optional identifier linking to the original tool call.
            Defaults to '123abcdef'.
    """
    role: Literal["tool"] = Field(default="tool", description="Role is fixed to 'tool' for tool messages")
    name: str = Field(..., description="The name of the tool generating the message")
    content: str = Field(..., description="The content of the tool message")
    tool_call_id: Optional[str] = Field(default='123abcdef', description="Tool call ID")


"""Represents all possible message types in the chat completion system.

The TextChatMessage type uses Pydantic's discriminated unions to automatically
determine the correct message type based on the 'role' field during validation.

Types:
    - UserMessage: Messages from users with role="user"
    - AssistantMessage: Messages from the assistant with role="assistant"
    - SystemMessage: System-level messages with role="system"
    - ToolMessage: Messages from tools with role="tool"

Note:
    This uses Pydantic's discriminated union feature to ensure proper validation
    and serialization of different message types.
"""
TextChatMessage = Annotated[
    Union[UserMessage, AssistantMessage, SystemMessage, ToolMessage],
    Field(discriminator='role')
]
