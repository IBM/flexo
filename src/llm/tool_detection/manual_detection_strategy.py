# src/llm/tool_detection/manual_tool_call_detection.py

import logging
from typing import List

from src.api import SSEChunk
from src.data_models.agent import StreamContext
from src.tools.core.parsers import BaseToolCallParser
from src.data_models.chat_completions import ToolCall, FunctionDetail
from src.llm.tool_detection import BaseToolCallDetectionStrategy
from src.llm.pattern_detection.aho_corasick_pattern import AhoCorasickBufferedProcessor
from src.llm.tool_detection.detection_result import DetectionResult, DetectionState


class ManualToolCallDetectionStrategy(BaseToolCallDetectionStrategy):
    """A strategy for detecting tool calls in streaming LLM output using pattern matching.

    This class implements manual detection of tool calls by processing streaming chunks
    of text using the Aho-Corasick algorithm for pattern matching. It maintains internal
    state to track tool call boundaries and accumulate content.

    Args:
        parser (BaseToolCallParser): Parser instance for processing detected tool calls.
        pattern_config_path (str, optional): Path to YAML config file containing tool call patterns.
            Defaults to "src/configs/tool_call_patterns.yaml".

    Attributes:
        tool_call_parser (BaseToolCallParser): Parser for processing tool calls.
        pattern_detector (AhoCorasickBufferedProcessor): Pattern matching processor.
        pre_tool_call_content (List[str]): Buffer for content before tool call.
        tool_call_buffer (str): Buffer for accumulating tool call content.
        in_tool_call (bool): Flag indicating if currently processing a tool call.
        accumulation_mode (bool): Flag for content accumulation mode.

    Example:
        ```python
        parser = JSONToolCallParser()
        detector = ManualToolCallDetectionStrategy(parser)

        # Process streaming chunks
        async for chunk in stream:
            result = await detector.detect_chunk(chunk, context)
            if result.state == DetectionState.COMPLETE_MATCH:
                # Handle detected tool call
                process_tool_calls(result.tool_calls)
        ```
    """

    def __init__(self, parser: BaseToolCallParser, pattern_config_path: str = "src/configs/tool_call_patterns.yaml"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing ManualToolCallDetectionStrategy with config: %s", pattern_config_path)
        self.tool_call_parser = parser
        self.pattern_detector = AhoCorasickBufferedProcessor(pattern_config_path)

        self.pre_tool_call_content: List[str] = []
        self.tool_call_buffer: str = ""
        self.in_tool_call: bool = False
        self.accumulation_mode: bool = False

    def reset(self) -> None:
        """Reset all internal state of the detection strategy.

        This method clears all buffers and resets flags to their initial state.
        Should be called between processing different streams or after errors.
        """
        self.logger.debug("Resetting detector state")
        self.pattern_detector.reset_states()
        self.pre_tool_call_content.clear()
        self.tool_call_buffer = ""
        self.in_tool_call = False
        self.accumulation_mode = False

    async def detect_chunk(self, sse_chunk: SSEChunk, context: StreamContext) -> DetectionResult:
        """
        Process a single chunk of streaming content for tool call detection.

        Args:
            sse_chunk (SSEChunk): The chunk of streaming content to process.
            context (StreamContext): Context information for the current stream.

        Returns:
            DetectionResult: Result of processing the chunk, including detection state
                and any content or tool calls found.
        """
        # If the chunk has no valid delta content, return NO_MATCH.
        if not sse_chunk.choices or not sse_chunk.choices[0].delta:
            return DetectionResult(state=DetectionState.NO_MATCH, sse_chunk=sse_chunk)

        chunk_content = sse_chunk.choices[0].delta.content
        if not chunk_content:
            return DetectionResult(state=DetectionState.NO_MATCH, sse_chunk=sse_chunk)

        # Process the chunk through the pattern detector.
        result = await self.pattern_detector.process_chunk(chunk_content)

        if result.error:
            return DetectionResult(state=DetectionState.NO_MATCH, sse_chunk=sse_chunk)

        # If a pattern is matched, switch into tool call mode.
        # Return an empty string for content so that no part of the tool call leaks into the output.
        if result.matched:
            self.in_tool_call = True
            self.tool_call_buffer = result.text_with_tool_call
            return DetectionResult(
                state=DetectionState.PARTIAL_MATCH,
                content="",  # No content is emitted once a tool call is detected
                sse_chunk=sse_chunk
            )

        # If already in tool call detection mode, continue accumulating and report a partial match.
        if self.in_tool_call:
            self.tool_call_buffer += chunk_content
            return DetectionResult(
                state=DetectionState.PARTIAL_MATCH,
                sse_chunk=sse_chunk
            )

        # For regular content, if there is any output, return it as PARTIAL_MATCH.
        if result.output:
            return DetectionResult(
                state=DetectionState.PARTIAL_MATCH,
                content=result.output,
                sse_chunk=sse_chunk
            )

        # If nothing of interest is found, return NO_MATCH.
        return DetectionResult(state=DetectionState.NO_MATCH, sse_chunk=sse_chunk)

    async def finalize_detection(self, context: StreamContext) -> DetectionResult:
        """Finalize the detection process and handle any accumulated content.

        This method is called at the end of a stream to process any remaining content
        in the buffers and return final detection results.

        Args:
            context (StreamContext): Context information for the current stream.

        Returns:
            DetectionResult: Final result of the detection process, including any
                complete tool calls or remaining content.
        """
        self.logger.debug("Finalizing detection")
        # Flush any remaining content from pattern detector
        final_result = await self.pattern_detector.flush_buffer()

        if self.in_tool_call:
            self.logger.debug("Processing final tool call buffer")
            if final_result.output:
                self.tool_call_buffer += final_result.output

            # Parse accumulated tool call
            parsed_tool_call_data = self.tool_call_parser.parse(self.tool_call_buffer)

            if "error" in parsed_tool_call_data:
                self.logger.error(f"Tool call parsing failed: {parsed_tool_call_data['error']}")
                return DetectionResult(
                    state=DetectionState.NO_MATCH,
                    content="Sorry, but I was unable to complete your request. Please try again.",
                )

            parsed_tool_calls = self._extract_tool_calls(parsed_tool_call_data)
            self.logger.debug("Successfully parsed %d tool calls", len(parsed_tool_calls))

            return DetectionResult(
                state=DetectionState.COMPLETE_MATCH,
                tool_calls=parsed_tool_calls
            )

        # No tool call detected, return any final content
        if final_result.output:
            self.logger.debug("Returning final content: %s", final_result.output[:50])
            return DetectionResult(
                state=DetectionState.PARTIAL_MATCH,
                content=final_result.output
            )

        self.logger.debug("No final content to return")
        return DetectionResult(state=DetectionState.NO_MATCH)

    def _extract_tool_calls(self, parsed_output: dict) -> List[ToolCall]:
        """Extract structured tool calls from parsed JSON output.

        Converts the parsed JSON format into a list of ToolCall objects with
        appropriate structure and typing.

        Args:
            parsed_output (dict): The parsed JSON output containing tool call data.

        Returns:
            List[ToolCall]: List of structured tool call objects ready for processing.
        """
        tool_calls = []
        for tool_call_dict in parsed_output.get("tool_calls", []):
            tool_call_args = tool_call_dict.get("parameters", tool_call_dict.get("arguments"))
            self.logger.debug("Extracting tool call arguments: %s", tool_call_args)
            tool_calls.append(ToolCall(
                id='123456789',  # Placeholder ID; modify as needed
                type=tool_call_dict.get("type", "function"),
                function=FunctionDetail(
                    name=tool_call_dict.get("name"),
                    arguments=tool_call_args
                )
            ))
        return tool_calls
