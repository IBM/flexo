# src/api/request_models.py

from typing import Optional, List
from pydantic import BaseModel, Field
from src.data_models.chat_completions import TextChatMessage


class ContextValue(BaseModel):
    """Represents a key-value pair for contextual information.

    This model is used to store individual pieces of context data
    that can be passed to tools or used in conversation.

    Attributes:
        key (str): The identifier for this piece of context
        value (str): The actual context value
    """
    key: str
    value: str


class ContextModel(BaseModel):
    """Container for multiple context values.

    This model serves as a collection of context values that can be
    passed around the system to provide additional information to
    tools and conversations.

    Attributes:
        values (List[ContextValue]): List of context key-value pairs

    Note: This model is used to define a structure for exposing in the API for allowing additional context to be passed through the system. The keys and values are used to create a dictionary which is passed to the streaming agent and anything downstream.
    """
    values: List[ContextValue]


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion endpoints.

    Attributes:
        model (Optional[str]): ID of the model to use for completion.
        messages (List[dict]): Array of message objects with role and content.
        context (Optional[ContextModel]): Additional context for API tools.
    """
    model: Optional[str] = Field(None, description="ID of the model to use")
    messages: List[TextChatMessage] = Field(..., description="Array of messages (role/content)")
    context: Optional[ContextModel] = Field(None, description="Additional context values (e.g. for API tools)")
