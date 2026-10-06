# src/llm/adapters/mistral_ai_adapter.py

import os
import json
import logging
from datetime import datetime
from mistralai.client import Mistral
from typing import AsyncGenerator, List, Optional

from src.data_models.tools import Tool
from src.llm.adapters import BaseVendorAdapter
from src.data_models.chat_completions import TextChatMessage
from src.api import SSEChunk, SSEChoice, SSEDelta, SSEToolCall, SSEFunction

logger = logging.getLogger(__name__)


class MistralAIAdapter(BaseVendorAdapter):
    """Adapter for interacting with Mistral AI's API.

    This class implements the BaseVendorAdapter interface for Mistral's chat models,
    handling authentication, request formatting, and response streaming. It converts
    Mistral-specific response formats into standardized SSE chunks for consistent
    handling across different LLM providers.

    Attributes:
        `api_key` (str): Mistral API key loaded from environment variables.
        `client` (Mistral): Authenticated Mistral client instance.
        `model_name` (str): The Mistral model identifier (e.g., "mistral-tiny").
        `default_params` (dict): Default parameters for Mistral API calls.
    """

    def __init__(self, model_name: str, **default_params):
        """Initialize the Mistral AI Adapter with model configuration.

        Args:
            `model_name` (str): The identifier of the Mistral model to use (e.g.,
                "mistral-tiny").
            `**default_params`: Additional parameters to include in all API calls.
                Common parameters include temperature, max_tokens, etc.

        Raises:
            `ValueError`: If MISTRAL_API_KEY environment variable is not set.
        """
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing Mistral API key. Set the MISTRAL_API_KEY environment variable."
            )

        self.client = Mistral(api_key=self.api_key)
        self.model_name = model_name
        self.default_params = default_params
        logger.info(f"Mistral AI Adapter initialized with model: {self.model_name}")
        logger.debug(f"Default parameters configured: {default_params}")

    async def gen_sse_stream(
        self, prompt: str, **kwargs
    ) -> AsyncGenerator[SSEChunk, None]:
        """Generate SSE stream from a single text prompt.

        Converts a single prompt into a chat message and streams the response.

        Args:
            `prompt` (str): The text prompt to send to the model.
            `**kwargs`: Additional parameters to override defaults for this request.

        Yields:
            `SSEChunk`: Standardized chunks of the streaming response.

        Raises:
            `RuntimeError`: If the streaming request fails.
        """
        logger.debug(f"Converting single prompt to chat format: {prompt[:50]}...")
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.gen_chat_sse_stream(messages, **kwargs):
            yield chunk

    async def gen_chat_sse_stream(
        self,
        messages: List[TextChatMessage],
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> AsyncGenerator[SSEChunk, None]:
        """Generate a streaming chat response from a sequence of messages.

        Args:
            `messages` (List[TextChatMessage]): List of chat messages for context.
            `tools` (Optional[List[Tool]]): List of tools available to the model.
            `**kwargs`: Additional parameters to override defaults for this request.

        Yields:
            `SSEChunk`: Standardized chunks of the streaming response.

        Raises:
            `RuntimeError`: If the Mistral API request fails.
        """
        mistral_messages = [msg.model_dump() for msg in messages]
        logger.debug(f"Processing chat stream request with {len(messages)} messages")

        request_payload = {
            "model": self.model_name,
            "messages": mistral_messages,
            **self.default_params,
            **kwargs,
        }

        if tools:
            logger.debug(f"Adding {len(tools)} tools to request")
            request_payload["tools"] = [tool.model_dump() for tool in tools]
            request_payload["tool_choice"] = "auto"

        try:
            logger.debug("Initiating Mistral streaming request")
            response = await self.client.chat.stream_async(**request_payload)
            async for chunk in response:
                yield await self._convert_to_sse_chunk(chunk)
        except Exception as e:
            logger.error(f"Error in Mistral streaming: {str(e)}", exc_info=True)
            raise RuntimeError(f"Mistral API streaming failed: {str(e)}") from e

    async def _convert_to_sse_chunk(self, raw_chunk) -> SSEChunk:
        """Convert Mistral's response chunk to standardized SSE format.

        Uses model_dump_json() to convert Mistral's response to a clean JSON
            representation,
        then constructs an SSEChunk from the parsed data.

        Args:
            raw_chunk: Raw chunk from Mistral's API, could be CompletionChunk or
                CompletionEvent.

        Returns:
            SSEChunk: Standardized chunk format for consistent handling.

        Raises:
            ValueError: If chunk conversion fails due to unexpected format.
        """
        try:
            # Use model_dump_json() to get a clean JSON representation where Unset
            # values are omitted
            chunk_json = raw_chunk.model_dump_json()
            chunk_data = json.loads(chunk_json)

            # Extract the 'data' field if present (based on the provided example)
            if "data" in chunk_data:
                chunk_data = chunk_data["data"]

            logger.debug(f"Converting chunk ID: {chunk_data.get('id', 'unknown')}")

            # Process choices
            choices = []
            for choice_data in chunk_data.get("choices", []):
                delta_data = choice_data.get("delta", {})

                # Process tool calls if present
                tool_calls = None
                if "tool_calls" in delta_data:
                    tool_calls = []
                    for tc_data in delta_data["tool_calls"]:
                        function = None
                        if "function" in tc_data:
                            # Ensure name and arguments are valid strings, defaulting to
                            # empty strings if missing or None
                            fn_name = tc_data["function"].get("name")
                            fn_name = "" if fn_name is None else fn_name

                            fn_args = tc_data["function"].get("arguments")
                            fn_args = "" if fn_args is None else fn_args

                            function = SSEFunction(name=fn_name, arguments=fn_args)

                        # Ensure type is always a string, defaulting to 'function' if
                        # missing or None
                        tool_call_type = tc_data.get("type")
                        if tool_call_type is None:
                            tool_call_type = "function"

                        tool_calls.append(
                            SSEToolCall(
                                index=tc_data.get("index", 0),
                                id=tc_data.get("id"),
                                type=tool_call_type,
                                function=function,
                            )
                        )

                # Create delta
                delta = SSEDelta(
                    role=delta_data.get("role"),
                    content=delta_data.get("content"),
                    tool_calls=tool_calls,
                    refusal=delta_data.get("refusal"),
                )

                # Create choice
                choices.append(
                    SSEChoice(
                        index=choice_data.get("index", 0),
                        delta=delta,
                        logprobs=choice_data.get("logprobs"),
                        finish_reason=choice_data.get("finish_reason"),
                    )
                )

            # Create and return the SSEChunk
            return SSEChunk(
                id=chunk_data.get("id", f"gen-{id(chunk_data)}"),
                object=chunk_data.get("object", "chat.completion.chunk"),
                created=chunk_data.get("created", int(datetime.now().timestamp())),
                model=chunk_data.get("model", self.model_name),
                service_tier=None,  # Default to None if not provided by Mistral
                system_fingerprint=None,  # Default to None if not provided by Mistral
                choices=choices,
            )

        except Exception as e:
            logger.error(f"Error converting Mistral chunk: {e}", exc_info=True)
            raise ValueError(
                f"Failed to convert Mistral response to SSEChunk: {str(e)}"
            ) from e
