# src/api/request_models.py

from pydantic import BaseModel, Field
from typing import Optional, List
from src.data_models.tools import ContextModel
from src.data_models.chat_completions import TextChatMessage


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
