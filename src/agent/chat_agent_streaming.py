# src/agent/chat_streaming_agent.py

import json
import yaml
import asyncio
import logging
from functools import wraps
from typing import List, AsyncGenerator, Dict, Optional, Any, Callable

from src.api import SSEChunk, AgentStatus
from src.data_models.agent import StreamState, StreamContext
from src.data_models.chat_completions import (
    ToolCall,
    ToolMessage,
    SystemMessage,
    TextChatMessage,
    AssistantMessage,
)
from src.llm import LLMFactory
from src.tools import ToolRegistry
from src.prompt_builders import PromptPayload, PromptBuilderOutput, BasePromptBuilder
from src.utils.factory import PromptBuilderFactory, ToolCallParserFactory, FormatType
from src.llm.tool_detection.detection_result import DetectionState, DetectionResult
from src.llm.tool_detection import ManualToolCallDetectionStrategy, VendorToolCallDetectionStrategy


# ----------------------------------------------------------------
#  Setup Standard Error Handler for State Handlers
# ----------------------------------------------------------------

def handle_streaming_errors(func: Callable[..., AsyncGenerator[SSEChunk, None]]) \
        -> Callable[..., AsyncGenerator[SSEChunk, None]]:
    @wraps(func)
    async def wrapper(self, *args, **kwargs) -> AsyncGenerator[SSEChunk, None]:
        try:
            async for item in func(self, *args, **kwargs):
                yield item
        except Exception:
            self.logger.error(f"Error in {func.__name__}", exc_info=True)
            yield await SSEChunk.make_stop_chunk(
                content="I apologize - I've encountered an unexpected error. Please try your request again.")
            return

    return wrapper


# ----------------------------------------------------------------
#  Create Streaming Chat Agent
# ----------------------------------------------------------------


class StreamingChatAgent:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.response_model_name = "main_chat_model"
        self.history_limit = self.config.get('history_limit', 3)
        self.system_prompt = self.config.get('system_prompt')

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        logging_level = self.config.get('logging_level', 'INFO')
        logging.basicConfig(level=getattr(logging, logging_level.upper(), None))
        self.logger.info(f'Logger set to {logging_level}')

        # Initialize vendor-specific components
        self.main_chat_model_config = self.config.get('models_config').get('main_chat_model')
        self.llm_factory = LLMFactory(config=self.config.get('models_config'))

        # Determine detection strategy first
        self.detection_mode = self.config.get("detection_mode", "vendor")
        self.use_vendor_chat_completions = self.config.get("use_vendor_chat_completions", True)

        # Load parser config if needed for manual detection
        self.logger.info(f" ---- IMPORTANT! ----\n\nMain Chat Model Config:\n{json.dumps(self.main_chat_model_config, indent=4)}\n")
        self.logger.info(f" ---- IMPORTANT! ---- Tool Detection Mode: {self.detection_mode}")
        self.logger.info(f" ---- IMPORTANT! ---- Vendor Chat API Mode: {self.use_vendor_chat_completions}")
        if self.detection_mode == "manual":
            with open("src/configs/parsing.yaml", "r") as f:
                parser_config = yaml.safe_load(f)
            self.tool_call_parser = ToolCallParserFactory.get_parser(
                FormatType.JSON,
                parser_config
            )
            self.detection_strategy = ManualToolCallDetectionStrategy(
                parser=self.tool_call_parser
            )
        else:
            self.detection_strategy = VendorToolCallDetectionStrategy()
            self.tool_call_parser = None

        # Initialize prompt builder with inject_tools flag based on detection mode
        self.prompt_builder: BasePromptBuilder = PromptBuilderFactory.get_prompt_builder(
            vendor=self.main_chat_model_config.get('vendor')
        )

        # Initialize ToolRegistry
        self.tool_registry = ToolRegistry(config=self.config.get("tools_config"))

    @handle_streaming_errors
    async def stream_step(
            self,
            conversation_history: List[TextChatMessage],
            api_passed_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[SSEChunk, None]:
        self.logger.debug("Starting streaming agent processing")

        context = await self._initialize_context(conversation_history, api_passed_context)

        while context.current_state != StreamState.COMPLETED:
            match context.current_state:

                case StreamState.STREAMING:
                    self.logger.info(f"--- Entering Streaming State ---")
                    self.detection_strategy.reset()
                    async for item in self._handle_streaming(context):
                        yield item

                case StreamState.TOOL_DETECTION:
                    self.logger.info(f"--- Entering Tool Detection State ---")
                    async for item in self._handle_tool_detection(context):
                        yield item

                case StreamState.EXECUTING_TOOLS:
                    self.logger.info(f"--- Entering Executing Tools State ---")
                    async for item in self._handle_tool_execution(context):
                        yield item

                case StreamState.INTERMEDIATE:
                    self.logger.info(f"--- Entering Intermediate State ---")
                    async for item in self._handle_intermediate(context):
                        yield item

                case StreamState.COMPLETING:
                    self.logger.info(f"--- Entering Completing State ---")
                    async for item in self._handle_completing(context):
                        yield item
                    context.current_state = StreamState.COMPLETED

    # ----------------------------------------------------------------
    #  STATE HANDLERS
    # ----------------------------------------------------------------

    @handle_streaming_errors
    async def _handle_streaming(self, context: StreamContext) -> AsyncGenerator[SSEChunk, None]:
        # Increment the streaming state counter and check if the maximum has been reached.
        context.streaming_entry_count += 1
        if context.streaming_entry_count > context.max_streaming_iterations:
            self.logger.error("Maximum streaming iterations reached. Aborting further streaming.")
            yield await SSEChunk.make_stop_chunk(
                content="Maximum streaming depth reached. Please try your request again."
            )
            context.current_state = StreamState.COMPLETING
            return

        prompt_payload = PromptPayload(
            conversation_history=context.conversation_history,
            tool_definitions=context.tool_definitions if self.detection_mode == "manual" else None
        )
        self.logger.debug(f"Prompt payload: {prompt_payload}")

        prompt_output: PromptBuilderOutput = (
            await self.prompt_builder.build_chat(prompt_payload) if self.use_vendor_chat_completions
            else await self.prompt_builder.build_text(prompt_payload)
        )
        llm_input = prompt_output.get_output()

        stream_kwargs = {
            'prompt': llm_input if isinstance(llm_input, str) else None,
            'messages': llm_input if isinstance(llm_input, list) else None,
            'tools': context.tool_definitions if self.detection_mode != "manual" else None
        }
        stream_kwargs = {k: v for k, v in stream_kwargs.items() if v is not None}

        self.logger.debug(f"stream_kwargs: {stream_kwargs}")
        llm_adapter = context.llm_factory.get_adapter(self.response_model_name)
        stream_gen = llm_adapter.gen_sse_stream if isinstance(llm_input, str) else llm_adapter.gen_chat_sse_stream

        accumulated_content = []
        async for sse_chunk in stream_gen(**stream_kwargs):
            detection_result = await self.detection_strategy.detect_chunk(sse_chunk, context)
            self.logger.debug(f"Detection result: {detection_result}")

            if detection_result.state in [DetectionState.NO_MATCH, DetectionState.PARTIAL_MATCH]:
                if detection_result.content:
                    accumulated_content.append(detection_result.content)
                    yield SSEChunk.make_text_chunk(detection_result.content)

            elif detection_result.state == DetectionState.COMPLETE_MATCH:
                async for chunk in self._handle_complete_match(context, detection_result, accumulated_content):
                    yield chunk
                return

        final_result = await self.detection_strategy.finalize_detection(context)

        if final_result.state == DetectionState.COMPLETE_MATCH:
            async for chunk in self._handle_complete_match(context, final_result, accumulated_content):
                yield chunk
        else:
            if final_result.content:
                accumulated_content.append(final_result.content)
                yield SSEChunk.make_text_chunk(final_result.content)

            if accumulated_content:
                context.conversation_history.append(
                    AssistantMessage(content="".join(accumulated_content))
                )

            yield await SSEChunk.make_stop_chunk()
            context.current_state = StreamState.COMPLETING

    @handle_streaming_errors
    async def _handle_tool_detection(self, context: StreamContext) -> AsyncGenerator[SSEChunk, None]:
        self.logger.debug("Tool calls detected, transitioning to EXECUTING_TOOLS")
        context.current_state = StreamState.EXECUTING_TOOLS
        yield await SSEChunk.make_status_chunk(
            AgentStatus.TOOL_DETECTED,
            {"tools": [tc.format_tool_calls() for tc in context.current_tool_call]}
        )

    @handle_streaming_errors
    async def _handle_tool_execution(
            self,
            context: StreamContext
    ) -> AsyncGenerator[SSEChunk, None]:
        """
        Execute the detected tools, produce results, and transition to INTERMEDIATE.
        """
        if context.message_buffer.strip():
            context.conversation_history.append(
                AssistantMessage(content=context.message_buffer)
            )
            context.message_buffer = ""

        results = await self._execute_tools_concurrently(context)

        tool_results = []
        for call, result in zip(context.current_tool_call, results):
            context.conversation_history.append(
                AssistantMessage(tool_calls=[call])
            )
            if isinstance(result, Exception):
                context.conversation_history.append(
                    AssistantMessage(
                        content=f"Error executing tool {call.function.name}: {str(result)}"
                    )
                )
            else:
                tool_results.append(result)
                context.conversation_history.append(
                    ToolMessage(
                        name=call.function.name,
                        content=result["result"],
                        tool_call_id=call.id
                    )
                )

        self.logger.debug("Tool execution results: %s", tool_results)
        context.current_state = StreamState.INTERMEDIATE
        yield await SSEChunk.make_status_chunk(AgentStatus.TOOLS_EXECUTED)

    @handle_streaming_errors
    async def _handle_intermediate(self, context: StreamContext) -> AsyncGenerator[SSEChunk, None]:
        context.message_buffer = ""
        self.detection_strategy.reset()
        context.current_state = StreamState.STREAMING
        yield await SSEChunk.make_status_chunk(AgentStatus.CONTINUING)

    @handle_streaming_errors
    async def _handle_completing(self, context: StreamContext) -> AsyncGenerator[SSEChunk, None]:
        self.logger.info(f"--- Entering COMPLETING State ---")

        yield await SSEChunk.make_stop_chunk()
        self.logger.info(f"Streaming process completed.")

    # ----------------------------------------------------------------
    #  HELPER METHODS
    # ----------------------------------------------------------------

    async def _initialize_context(
            self,
            conversation_history: List[TextChatMessage],
            api_passed_context: Optional[Dict]
    ) -> StreamContext:
        selected_history = (
            conversation_history[-self.history_limit:]
            if self.history_limit > 0
            else conversation_history
        )

        if self.system_prompt:
            system_message = SystemMessage(content=self.system_prompt)
            selected_history.insert(0, system_message)

        return StreamContext(
            conversation_history=selected_history,
            tool_definitions=self.tool_registry.get_tool_definitions(),
            context=api_passed_context,
            llm_factory=self.llm_factory,
            current_state=StreamState.STREAMING,
            max_streaming_iterations=self.config.get("max_streaming_iterations", 1),
            streaming_entry_count=0
        )

    async def _handle_complete_match(
            self,
            context: StreamContext,
            result: DetectionResult,
            accumulated_content: List[str]
    ) -> AsyncGenerator[SSEChunk, None]:
        if result.content:
            accumulated_content.append(result.content)
            yield SSEChunk.make_text_chunk(result.content)

        context.current_tool_call = result.tool_calls or []
        if accumulated_content:
            context.conversation_history.append(
                AssistantMessage(content="".join(accumulated_content))
            )
        context.current_state = StreamState.TOOL_DETECTION

    async def _execute_tools_concurrently(self, context: StreamContext) -> List[Any]:
        async def run_tool(tool_call: ToolCall):
            try:
                tool = self.tool_registry.get_tool(tool_call.function.name)
                if not tool:
                    raise RuntimeError(f"Tool {tool_call.function.name} not found")

                result = await tool.execute(
                    context=context.context,
                    **tool_call.function.arguments
                )
                return {"tool_name": tool_call.function.name, "result": result.result}

            except Exception as e:
                self.logger.error(f"Error executing tool {tool_call.function.name}", exc_info=True)
                return e

        tasks = [run_tool(tc) for tc in context.current_tool_call]
        return await asyncio.gather(*tasks, return_exceptions=True)
