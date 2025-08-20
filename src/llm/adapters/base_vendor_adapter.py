# src/llm/adapters/base_vendor_adapter.py

from abc import abstractmethod
from typing import AsyncGenerator, Optional, List

from src.api.sse_models import SSEChunk
from src.data_models.tools import Tool
from src.data_models.chat_completions import TextChatMessage


class BaseVendorAdapter:
    """
    Abstract base class for vendor adapters, defining the interface for
    text and chat completion endpoints using Server-Sent Events (SSE).
    """

    @abstractmethod
    async def gen_sse_stream(self, prompt: str) -> AsyncGenerator[SSEChunk, None]:
        """
        Generate a Server-Sent Events (SSE) stream for a text completion request.

        This method should connect to the vendor’s text-completion SSE endpoint
        and yield chunks of the response as they arrive.

        Args:
            prompt (str): The text prompt to send to the completion endpoint.

        Yields:
            SSEChunk: A single chunk of the SSE response stream.
        """
        pass

    @abstractmethod
    async def gen_chat_sse_stream(
        self,
        messages: List[TextChatMessage],
        tools: Optional[List[Tool]]
    ) -> AsyncGenerator[SSEChunk, None]:
        """
        Generate a Server-Sent Events (SSE) stream for a chat completion request.

        This method should connect to the vendor’s chat-completion SSE endpoint
        and yield chunks of the response as they arrive.

        Args:
            messages (List[TextChatMessage]): The sequence of chat messages leading up
                to the current user or assistant message.
            tools (Optional[List[Tool]]): Any optional tools (e.g., external APIs or
                functions) that the model can use during generation.

        Yields:
            SSEChunk: A single chunk of the SSE response stream.
        """
        pass

    async def gen_text(self, prompt: str) -> SSEChunk:
        """
        Generate a complete text completion in a single response.

        This method should call the text completion endpoint (non-streaming)
        and return the full result as one `SSEChunk`.

        Args:
            prompt (str): The text prompt to send to the completion endpoint.

        Returns:
            SSEChunk: The complete response from the text completion endpoint.
        """
        pass

    async def gen_chat(
        self,
        messages: List[TextChatMessage],
        tools: Optional[List[Tool]] = None
    ) -> SSEChunk:
        """
        Generate a complete chat-based completion in a single response.

        This method should call the chat completion endpoint (non-streaming)
        and return the full result as one `SSEChunk`.

        Args:
            messages (List[TextChatMessage]): The sequence of chat messages leading up
                to the current user or assistant message.
            tools (Optional[List[Tool]], optional): Any optional tools (e.g., external
                APIs or functions) that the model can use during generation.
                Defaults to None.

        Returns:
            SSEChunk: The complete response from the chat completion endpoint.
        """
        pass
