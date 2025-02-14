# src/data_models/wx_assistant.py

from pydantic import BaseModel
from typing import List, Optional
from src.api.request_models import ContextModel
from src.data_models.chat_completions import TextChatMessage, UserMessage, AssistantMessage, UserTextContent


class WxAssistantMessage(BaseModel):
    """
    This is a class for WxAssistant messages.

    Attributes:
        u (Optional[str]): Represents the user input.
        a (Optional[str]): Represents the assistant response.
        n (Optional[bool]): An optional boolean for additional context.
    """
    u: Optional[str] = None
    a: Optional[str] = None
    n: Optional[bool] = None

    def to_dict(self):
        return {"u": self.u, "a": self.a, "n": self.n}


class WxAssistantConversationInput(BaseModel):
    """
    This is a class for WxAssistant conversation input.

    Attributes:
        messages (List[WxAssistantMessage]): A list of WxAssistantMessage instances.
        context (Optional[ContextModel]): Optional metadata for additional context.
    """
    messages: List[WxAssistantMessage]
    context: Optional[ContextModel] = None


def convert_wx_to_conversation(wx_input: WxAssistantConversationInput) -> List[TextChatMessage]:
    """Convert Watson Assistant messages to standard conversation format.

    Converts messages from Watson Assistant format to a list of TextChatMessage
    objects suitable for general conversation processing.

    Args:
        wx_input (WxAssistantConversationInput): Watson Assistant conversation input.

    Returns:
        List[TextChatMessage]: List of converted messages in standard format.

    Example:
        ```python
        wx_input = WxAssistantConversationInput(messages=[...])
        messages = convert_wx_to_conversation(wx_input)
        ```
    """
    messages: List[TextChatMessage] = []

    for wx_message in wx_input.messages:
        if wx_message.u:
            messages.append(
                UserMessage(
                    role='user',
                    content=[UserTextContent(text=wx_message.u)]
                )
            )
        if wx_message.a:
            messages.append(
                AssistantMessage(content=wx_message.a))
        if wx_message.n:
            pass

    return messages
