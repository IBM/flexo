"""Chat Completions API Module.

This module implements a FastAPI router for streaming chat completion functionality,
with authentication, message handling, and integration with a streaming chat agent.

The module provides:
- API key validation
- Message format conversion
- Streaming chat completions
- SSE (Server-Sent Events) response handling
- Agent info endpoint (compatible with OpenAI's /models format)
"""

import os
import time
import yaml
import logging
from typing import List, Optional
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security.api_key import APIKeyHeader
from fastapi import APIRouter, Body, Depends, Header, HTTPException

from src.api import SSEChunk
from src.agent import StreamingChatAgent
from src.api.request_models import ChatCompletionRequest
from src.data_models.chat_completions import (
    TextChatMessage,
    UserMessage,
    UserTextContent
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Capture the agent's start time once when the app starts.
AGENT_START_TIME = int(time.time())

# Security setup
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
FLEXO_API_KEY = os.getenv("FLEXO_API_KEY")
ENABLE_API_KEY = os.getenv("ENABLE_API_KEY", "true").lower() != "false"

if not FLEXO_API_KEY:
    logger.error("FLEXO_API_KEY environment variable not set")
    raise ValueError("FLEXO_API_KEY is not set. Please set it in your environment or .env file.")

logger.info("API authentication %s", "enabled" if ENABLE_API_KEY else "disabled")


async def get_api_key(api_key: str = Depends(api_key_header)):
    """Validate the incoming API key against environment configuration.

    Args:
        api_key (str): The API key from the request header.

    Returns:
        str: The validated API key.

    Raises:
        HTTPException: If authentication is enabled and the API key is invalid.
    """
    if ENABLE_API_KEY:
        logger.debug("Validating API key")
        if not api_key or api_key != FLEXO_API_KEY:
            logger.warning("Invalid API key attempted")
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Unauthorized"
            )
    return api_key


# Agent initialization
try:
    logger.debug("Loading agent configuration from yaml")
    with open('src/configs/agent.yaml', 'r') as file:
        config = yaml.safe_load(file)
    streaming_chat_agent_instance = StreamingChatAgent(config=config)
    logger.info("StreamingChatAgent successfully initialized")
except Exception as e:
    logger.error("Failed to load agent config: %s", str(e), exc_info=True)
    raise


def get_streaming_agent() -> StreamingChatAgent:
    """Dependency provider for the streaming agent instance.

    Returns:
        StreamingChatAgent: The global streaming agent instance.
    """
    return streaming_chat_agent_instance


def convert_message_content(messages: List[TextChatMessage]) -> List[TextChatMessage]:
    """Convert plain string user messages to structured UserTextContent objects.

    Args:
        messages (List[TextChatMessage]): List of messages to convert.

    Returns:
        List[TextChatMessage]: Converted messages with proper content structure.
    """
    logger.debug("Converting %d messages", len(messages))
    converted = []
    for msg in messages:
        if msg.role == "user" and isinstance(msg.content, str):
            converted.append(
                UserMessage(
                    role="user",
                    content=[UserTextContent(text=msg.content.strip())]
                )
            )
        else:
            converted.append(msg)
    return converted


def get_non_sensitive_config(config: dict) -> dict:
    """Filter the agent configuration to include only non-sensitive parameters.

    Args:
        config (dict): The full agent configuration.

    Returns:
        dict: Filtered configuration excluding keys that may be sensitive.
    """
    # Exclude keys that include 'key', 'secret', or 'password' (case insensitive)
    return {
        k: v for k, v in config.items()
        if not any(sensitive in k.lower() for sensitive in ["key", "secret", "password"])
    }


@router.post(
    "/chat/completions",
    summary="Generate streaming chat completions",
    description="Generate a streaming response from the agent based on user input.",
    tags=["Agent Chat"],
    operation_id="chat",
    response_class=StreamingResponse
)
async def chat_completions(
        request_body: ChatCompletionRequest = Body(...),
        x_ibm_thread_id: Optional[str] = Header(None),
        agent: StreamingChatAgent = Depends(get_streaming_agent),
        api_key: Optional[str] = Depends(get_api_key)
):
    """Generate streaming chat completions from the agent.

    Processes incoming chat messages and returns a streaming response with
    token-by-token updates.

    Args:
        request_body (ChatCompletionRequest): The chat completion request.
        x_ibm_thread_id (Optional[str]): Thread ID for response correlation.
        agent (StreamingChatAgent): The chat agent instance.
        api_key (Optional[str]): Validated API key.

    Returns:
        StreamingResponse: Server-sent events stream of completion tokens.

    Raises:
        HTTPException: If processing fails or invalid input is provided.
    """
    try:
        logger.debug(f"Processing chat completion request with {len(request_body.messages)} messages")
        processed_messages = convert_message_content(request_body.messages)

        logger.debug("Initiating streaming response")
        response_stream = agent.stream_step(
            conversation_history=processed_messages,
            api_passed_context=request_body.context.model_dump() if request_body.context else None
        )

        async def sse_generator():
            """Generate SSE chunks from the response stream."""
            try:
                async for sse_chunk in response_stream:
                    if sse_chunk:
                        if x_ibm_thread_id:
                            sse_chunk.thread_id = x_ibm_thread_id
                        sse_chunk.object = "thread.message.delta"

                        logger.debug("Sending SSE chunk")
                        yield f"data: {sse_chunk.model_dump_json(exclude_none=True)}\n\n"

                        if any(choice.finish_reason in ["stop", "tool_calls"]
                               for choice in sse_chunk.choices):
                            yield await SSEChunk.done_marker()
                            logger.debug("Stream completed")
                            return

            except Exception as e:
                logger.error("Error in SSE generator: %s", str(e), exc_info=True)
                return

        logger.debug("Initializing StreamingResponse")
        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error("Error in /chat/completions: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/models",
    summary="Agent information",
    description="Returns information about the running agent including its default model and non-sensitive configuration parameters.",
    tags=["Models"],
    operation_id="getModels"
)
async def get_models(
    agent: StreamingChatAgent = Depends(get_streaming_agent),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Retrieve agent information in a format similar to OpenAI's models endpoint.

    This endpoint returns the agent's name, default model, creation time,
    owner, and a selection of non-sensitive configuration parameters.
    """
    agent_name = agent.config.get("name", "default-agent")
    default_model = agent.config.get("model", "default-model")
    created = agent.config.get("created", AGENT_START_TIME)
    owned_by = agent.config.get("owned_by", "user")
    non_sensitive_params = get_non_sensitive_config(agent.config)

    agent_info = {
        "id": agent_name,
        "object": "agent",
        "default_model": default_model,
        "created": created,
        "owned_by": owned_by,
        "params": non_sensitive_params
    }

    return {"data": [agent_info], "object": "list"}
