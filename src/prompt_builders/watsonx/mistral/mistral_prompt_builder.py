# prompt_builders/mistral/mistral_prompt_builder.py

import yaml
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from mistral_common.protocol.instruct.messages import (
    UserMessage as MistralUserMessage,
    AssistantMessage as MistralAssistantMessage,
    ToolMessage as MistralToolMessage,
    SystemMessage as MistralSystemMessage,
)
from mistral_common.protocol.instruct.chunk import ImageURLChunk, TextChunk, ImageURL
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    Tool as MistralTool,
    ToolCall as MistralToolCall,
    FunctionCall as MistralFunctionCall,
)

from src.prompt_builders.base_prompt_builder import BasePromptBuilder
from src.prompt_builders.prompt_models import PromptPayload, PromptBuilderOutput
from src.data_models.chat_completions import TextChatMessage, ToolCall, AssistantMessage
from src.data_models.tools import Tool

logger = logging.getLogger(__file__)


class WatsonXMistralPromptBuilder(BasePromptBuilder):
    """Prompt builder for the Mistral model architecture.

    This class handles the construction of prompts and chat messages specifically
    formatted for Mistral models, including support for tool definitions,
    multi-modal content, and Mistral-specific message formatting.

    Attributes:
        config (Dict): Configuration loaded from prompt_builders.yaml.
        model_name (str): Name of the Mistral model to use.
        tokenizer (MistralTokenizer): Tokenizer instance for the specified model.
    """

    def __init__(self, model_name: str = "mistral-large"):
        """Initialize the Mistral prompt builder.

        Args:
            model_name (str): Name of the Mistral model to use.
                Defaults to 'mistral-large'.
        """
        super().__init__()
        self.config = self._load_config()
        self.model_name = model_name
        self.tokenizer = MistralTokenizer.from_model(model_name)

    async def build_chat(self, payload: PromptPayload) -> PromptBuilderOutput:
        """Build chat messages with tools in the last assistant message.

        Modifies the conversation history to include tool definitions in the
        most recent assistant message, or creates a new one if none exists.

        Args:
            payload (PromptPayload): The structured input containing conversation
                history,
                                   tool definitions, and other context-specific data

        Returns:
            PromptBuilderOutput: Contains the modified message list with tool
                information
        """
        conversation_history = payload.conversation_history
        tool_definitions = payload.tool_definitions or []

        # If no tools, return history as is
        if not tool_definitions:
            return PromptBuilderOutput(chat_messages=conversation_history)

        tool_info = self._format_tool_definitions(tool_definitions)
        modified_history = conversation_history.copy()

        last_assistant_idx = None
        for idx, msg in enumerate(modified_history):
            if msg.role == "assistant":
                last_assistant_idx = idx

        if last_assistant_idx is not None:
            # Add tool info to the last assistant message
            last_assistant = modified_history[last_assistant_idx]
            existing_content = last_assistant.content or ""
            modified_history[last_assistant_idx] = AssistantMessage(
                content=f"{existing_content}\n{tool_info}",
            )
        else:
            # Create new assistant message with tool info if none exists
            assistant_msg = AssistantMessage(content=tool_info)
            modified_history.append(assistant_msg)

        return PromptBuilderOutput(chat_messages=modified_history)

    async def build_text(self, payload: PromptPayload) -> PromptBuilderOutput:
        """Build prompt string using Mistral's chat completion format.

        Converts the conversation history and tool definitions into Mistral's
        format and generates a tokenized prompt string.

        Args:
            payload (PromptPayload): The structured input containing conversation
                history,
                                   tool definitions, and other context-specific data

        Returns:
            PromptBuilderOutput: Contains the formatted text prompt for generation
        """
        conversation_history = payload.conversation_history
        tool_definitions = payload.tool_definitions or []

        # Convert messages to Mistral format
        mistral_messages = self._process_conversation_history(conversation_history)

        # Convert tool definitions to Mistral format
        mistral_tools = self._process_tool_definitions(tool_definitions)

        chat_request = ChatCompletionRequest(
            tools=mistral_tools,
            messages=mistral_messages,
            model=self.model_name,
        )

        # Tokenize and get prompt text
        tokenized = self.tokenizer.encode_chat_completion(chat_request)
        return PromptBuilderOutput(text_prompt=tokenized.text)

    def _format_tool_definitions(self, tool_definitions: List[Tool]) -> str:
        """Format tool definitions in Mistral's required format."""
        formatted_tools = []

        for tool in tool_definitions:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters.model_dump(),
                },
            }
            formatted_tools.append(tool_dict)

        # Create the final string with the required format
        tool_names = [tool.function.name for tool in tool_definitions]
        tool_section_header = self.config["system_prompt"]["header"].format(
            tools=", ".join(tool_names), date=datetime.now().strftime("%Y-%m-%d")
        )
        tool_instructions = self.config["system_prompt"]["tool_instructions"]
        tool_json = json.dumps(formatted_tools, separators=(",", ":"))
        return (
            f"[AVAILABLE_TOOLS]{tool_section_header}\n\n{tool_json}\n\n"
            f"{tool_instructions}[/AVAILABLE_TOOLS]"
        )

    def _process_conversation_history(
        self, conversation_history: List[TextChatMessage]
    ) -> List:
        """Convert conversation history to Mistral's message format."""
        messages = []
        for msg in conversation_history:
            if msg.role == "system":
                messages.append(MistralSystemMessage(content=msg.content))
            elif msg.role == "user":
                mistral_content = self._convert_user_content(msg.content)
                messages.append(MistralUserMessage(content=mistral_content))
            elif msg.role == "assistant":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    logger.debug(f"Tool calls: {msg.tool_calls}")
                    tool_calls = [self._create_tool_call(tc) for tc in msg.tool_calls]
                    messages.append(
                        MistralAssistantMessage(content=None, tool_calls=tool_calls)
                    )
                else:
                    messages.append(MistralAssistantMessage(content=msg.content))
            elif msg.role == "tool":
                messages.append(
                    MistralToolMessage(
                        content=msg.content, tool_call_id=msg.tool_call_id
                    )
                )
        return messages

    def _convert_user_content(self, content) -> List:
        """Convert user message content to Mistral's chunk format."""
        # Handle plain string content
        if isinstance(content, str):
            return [TextChunk(text=content)]

        # Handle list content
        if not isinstance(content, list):
            raise ValueError(
                f"Content must be either string or list, got {type(content)}"
            )

        converted_content = []
        for item in content:
            # Handle string items in list
            if isinstance(item, str):
                converted_content.append(TextChunk(text=item))
                continue

            # Handle structured content items
            if not hasattr(item, "type"):
                raise ValueError(f"Content item missing 'type' attribute: {item}")

            if item.type == "text":
                converted_content.append(TextChunk(text=item.text))
            elif item.type == "image_url":
                image_url = item.image_url
                detail = getattr(item, "detail", None)
                image_url_chunk = ImageURLChunk(
                    image_url=(
                        ImageURL(url=image_url, detail=detail) if detail else image_url
                    )
                )
                converted_content.append(image_url_chunk)
            else:
                raise ValueError(f"Unsupported content type: {item.type}")

        return converted_content

    def _process_tool_definitions(
        self, tool_definitions: List[Tool]
    ) -> List[MistralTool]:
        """Convert tool definitions to Mistral's Tool format."""
        return [
            MistralTool(
                function=Function(
                    name=tool.function.name,
                    description=tool.function.description,
                    parameters=tool.function.parameters.model_dump(),
                )
            )
            for tool in tool_definitions
        ]

    def _create_tool_call(self, tool_call_data: ToolCall) -> MistralToolCall:
        """Convert tool call data to Mistral's format."""
        function_name = tool_call_data.function.name
        arguments = tool_call_data.function.arguments

        # Convert arguments to JSON string if needed
        arguments_str = (
            json.dumps(arguments) if not isinstance(arguments, str) else arguments
        )

        return MistralToolCall(
            function=MistralFunctionCall(name=function_name, arguments=arguments_str)
        )

    @staticmethod
    def _load_config() -> Dict:
        """Load Mistral-specific configuration from YAML file."""
        config_path = Path("src/configs/prompt_builders.yaml")
        with config_path.open() as f:
            config = yaml.safe_load(f)
            return config.get("watsonx-mistral")
