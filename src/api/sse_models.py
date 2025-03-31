from __future__ import annotations

import time
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    STARTING = "starting_generation"
    TOOL_DETECTED = "tool_call_detected"
    TOOLS_EXECUTED = "tools_executed"
    MAX_DEPTH = "max_depth_reached"
    CONTINUING = "continuing_generation"
    THINKING = "thinking"


class SSEFunction(BaseModel):
    name: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)


class SSEToolCall(BaseModel):
    id: str
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    index: Optional[int] = None


class SSEStepDetails(BaseModel):
    type: str
    content: Optional[str] = None
    tool_calls: Optional[List[SSEToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class SSEDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    step_details: Optional[SSEStepDetails] = None
    refusal: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[dict] = None


class SSEChoice(BaseModel):
    index: Optional[int] = 0
    delta: SSEDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[dict] = None


class SSEChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    thread_id: Optional[str] = None
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None
    choices: List[SSEChoice]

    @staticmethod
    async def assistant_response_event(
            text: str,
            thread_id: Optional[str] = None,
            model_name: Optional[str] = "flexo"
    ) -> SSEChunk:
        return SSEChunk(
            id=f"run-{time.time_ns() // 1000000:x}",
            object="thread.message.delta",
            created=int(time.time()),
            model=model_name,
            thread_id=thread_id,
            choices=[
                SSEChoice(
                    delta=SSEDelta(role="assistant", content=text),
                    finish_reason=None
                )
            ]
        )

    @staticmethod
    async def thinking_step_event(
            thinking_text: str,
            thread_id: Optional[str] = None,
            model_name: Optional[str] = "flexo"
    ) -> SSEChunk:
        return SSEChunk(
            id=f"step-{time.time_ns() // 1000000:x}",
            object="thread.run.step.delta",
            created=int(time.time()),
            model=model_name,
            thread_id=thread_id,
            choices=[
                SSEChoice(
                    delta=SSEDelta(
                        role="assistant",
                        step_details=SSEStepDetails(
                            type="thinking",
                            content=thinking_text
                        )
                    ),
                    finish_reason=None
                )
            ]
        )

    @staticmethod
    async def tool_call_event(
            tool_calls: List[SSEToolCall],
            thread_id: Optional[str] = None,
            model_name: Optional[str] = "flexo"
    ) -> SSEChunk:
        return SSEChunk(
            id=f"step-{time.time_ns() // 1000000:x}",
            object="thread.run.step.delta",
            created=int(time.time()),
            model=model_name,
            thread_id=thread_id,
            choices=[
                SSEChoice(
                    delta=SSEDelta(
                        role="assistant",
                        step_details=SSEStepDetails(
                            type="tool_calls",
                            tool_calls=tool_calls
                        )
                    ),
                    finish_reason=None
                )
            ]
        )

    @staticmethod
    async def tool_response_event(
            response_content: str,
            tool_name: str,
            tool_call_id: str,
            thread_id: Optional[str] = None,
            model_name: Optional[str] = "flexo"
    ) -> SSEChunk:
        return SSEChunk(
            id=f"step-{time.time_ns() // 1000000:x}",
            object="thread.run.step.delta",
            created=int(time.time()),
            model=model_name,
            thread_id=thread_id,
            choices=[
                SSEChoice(
                    delta=SSEDelta(
                        role="assistant",
                        step_details=SSEStepDetails(
                            type="tool_response",
                            content=response_content,
                            name=tool_name,
                            tool_call_id=tool_call_id
                        )
                    ),
                    finish_reason=None
                )
            ]
        )

    @staticmethod
    async def stop_event(
            thread_id: Optional[str] = None,
            content: Optional[str] = None,
            refusal: Optional[str] = None,
            model_name: str = "agent-02"
    ) -> SSEChunk:
        """
        Create a stop event to indicate the end of a response.
        """
        return SSEChunk(
            id=f"run-{time.time_ns() // 1000000:x}",
            object="thread.message.delta",
            created=int(time.time()),
            model=model_name,
            thread_id=thread_id,
            choices=[
                SSEChoice(
                    delta=SSEDelta(role="assistant", content=content, refusal=refusal),
                    finish_reason="stop"
                )
            ]
        )

    @staticmethod
    async def done_marker() -> str:
        """
        Create the end-of-stream marker.
        """
        return "data: [DONE]"
