# src/data_models/agent.py

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

from src.llm import LLMFactory
from src.data_models.tools import Tool
from src.data_models.chat_completions import TextChatMessage, ToolCall


class StreamState(Enum):
    """Defines the possible states of the streaming agent during processing.

    The StreamState enum represents different operational states that the streaming agent
    can be in at any given time during message processing and tool execution.

    Attributes:
        STREAMING: Currently streaming response content
        TOOL_DETECTION: Analyzing stream for potential tool calls
        EXECUTING_TOOLS: Currently executing detected tools
        INTERMEDIATE: Temporary state between major operations

    Example:
        ```python
        state = StreamState.IDLE
        if processing_started:
            state = StreamState.STREAMING
        ```
    """
    STREAMING = "streaming"
    TOOL_DETECTION = "detection"
    EXECUTING_TOOLS = "executing"
    INTERMEDIATE = "intermediate"
    COMPLETING = "completing"
    COMPLETED = "completed"


class StreamResult(BaseModel):
    """Represents the result of processing an individual stream chunk.

    This class encapsulates the various pieces of information that can be produced
    when processing a single chunk of a stream, including content, errors, and status updates.

    Attributes:
        content (Optional[str]): The actual content from the stream chunk. May be None if
            chunk contained no content (e.g., only status updates).
        error (Optional[str]): Error message if any issues occurred during processing.
            None if processing was successful.
        status (Optional[str]): Status message indicating state changes or completion.
            Used to communicate processing progress.
        should_continue (bool): Flag indicating if streaming should continue.
            Defaults to True, set to False for terminating streaming.

    Example:
        ```python
        result = StreamResult(
            content="Generated text response",
            status="Processing complete",
            should_continue=True
        )
        ```
    """
    content: Optional[str] = Field(
        default=None,
        description="The content of the stream chunk"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    status: Optional[str] = Field(
        default=None,
        description="Status message indicating state changes or completion"
    )
    should_continue: bool = Field(
        default=True,
        description="Flag indicating if streaming should continue"
    )


class StreamContext(BaseModel):
    """Context and state for a streaming conversation session.

    This class stores all necessary information for managing a streaming conversation,
    including conversation history, available tool definitions, state buffers, and
    session metadata. It also tracks the number of times the streaming state has been
    initiated.

    Attributes:
        conversation_history (List[TextChatMessage]): The full conversation history,
            including a system message at the start if available.
        tool_definitions (List[Tool]): Definitions of available tools for execution.
        message_buffer (str): Buffer for accumulating generated response text.
        tool_call_buffer (str): Buffer for accumulating potential tool call text until parsing.
        current_tool_call (Optional[List[ToolCall]]): The currently processing tool calls, if any.
        current_state (StreamState): The current state of the stream processing.
        streaming_entry_count (int): Counter tracking the number of times the streaming state has been entered.
        max_streaming_iterations (int): The maximum allowed number of times the streaming state can be initiated.
        context (Optional[Dict[str, Any]]): Additional metadata associated with the streaming session.
        llm_factory (Optional[LLMFactory]): LLM factory associated with the streaming agent.
    """

    conversation_history: List[TextChatMessage] = Field(
        default_factory=list,
        description="Full conversation history with system message at the start."
    )
    tool_definitions: List[Tool] = Field(
        default_factory=list,
        description="Definitions of available tools."
    )
    message_buffer: str = Field(
        default="",
        description="Buffer for accumulating generated response text."
    )
    tool_call_buffer: str = Field(
        default="",
        description="Buffer for accumulating tool call text until parsing."
    )
    current_tool_call: Optional[List[ToolCall]] = Field(
        default=None,
        description="Currently processing tool calls."
    )
    current_state: StreamState = Field(
        default=None,
        description="Current state of the stream processing."
    )
    streaming_entry_count: int = Field(
        default=0,
        description="Tracks how many times the streaming state has been entered."
    )
    max_streaming_iterations: int = Field(
        default=3,
        description="The maximum allowed number of times the streaming state can be initiated."
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata associated with the streaming session."
    )
    llm_factory: Optional[LLMFactory] = Field(
        default=None,
        description="LLM Model factory for retrieving LLM adapters."
    )

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True
