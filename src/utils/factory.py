# src/utils/factory.py

from enum import Enum
from typing import Dict, Any

from src.tools.core.parsers import JSONToolCallParser, NonJSONToolCallParser
from src.prompt_builders import (
    BasePromptBuilder,
    OpenAIPromptBuilder,
    AnthropicPromptBuilder,
    MistralAIPromptBuilder,
    LlamaPromptBuilder,
    GranitePromptBuilder,
    MistralPromptBuilder
)


# ----------------------------------------------------------------
#  Prompt Builder Factory
# ----------------------------------------------------------------

class PromptBuilderFactory:
    @staticmethod
    def get_prompt_builder(vendor: str) -> BasePromptBuilder:
        """Get a prompt builder instance for the specified vendor.

        Args:
            vendor (str): Vendor identifier (e.g., 'openai', 'anthropic', 'watsonx', 'mistral_ai')

        Returns:
            BasePromptBuilder: Appropriate prompt builder instance

        Raises:
            ValueError: If no prompt builder is available for the specified vendor
        """
        match vendor.lower():
            case "watsonx-granite":
                return GranitePromptBuilder()
            case "watsonx-llama":
                return LlamaPromptBuilder()
            case "watsonx-mistral":
                return MistralPromptBuilder()
            case "openai":
                return OpenAIPromptBuilder()
            case "anthropic":
                return AnthropicPromptBuilder()
            case "mistral_ai":
                return MistralAIPromptBuilder()
            case _:
                raise ValueError(f"No prompt builder available for vendor: {vendor}")


# ----------------------------------------------------------------
#  Parser Factory
# ----------------------------------------------------------------

class FormatType(Enum):
    """Enumeration of supported tool call format types.

    Attributes:
        JSON: For JSON-formatted tool calls.
        NON_JSON: For non-JSON formatted tool calls.
    """
    JSON = "json_format"
    NON_JSON = "non_json_format"


class ToolCallParserFactory:
    """Factory for creating tool call parser instances.

    Manages the creation of parsers for different tool call formats,
    including JSON and non-JSON formats.

    Attributes:
        registry (Dict): Mapping of format types to their parser classes.

    Example:
        ```python
        parser = ToolCallParserFactory.get_parser(
            FormatType.JSON,
            config={"clean_tokens": []}
        )
        ```
    """
    registry = {
        FormatType.JSON: JSONToolCallParser,
        FormatType.NON_JSON: NonJSONToolCallParser,
    }

    @staticmethod
    def get_parser(format_type: FormatType, config: Dict[str, Any]):
        """Get appropriate parser for the specified format type.

        Args:
            format_type (FormatType): Type of format to parse.
            config (Dict[str, Any]): Configuration for the parser.

        Returns:
            BaseToolCallParser: Instance of appropriate parser.

        Raises:
            ValueError: If format type is not supported.
        """
        if format_type in ToolCallParserFactory.registry:
            return ToolCallParserFactory.registry[format_type](config)
        raise ValueError(f"Unsupported format: {format_type}")
