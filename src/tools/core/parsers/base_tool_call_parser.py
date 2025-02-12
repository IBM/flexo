# src/parsers/base_tool_call_parser.py

import logging
from typing import Any, Dict
from abc import ABC, abstractmethod


class BaseToolCallParser(ABC):
    def __init__(self, config: Dict[str, Any]):
        """Abstract base class for parsing tool calls from text.

        This class provides a framework for implementing tool call parsers
        with common functionality for cleaning, validation, and error handling.

        Attributes:
            config (Dict[str, Any]): Configuration dictionary for parsing options.

        Example:
            ```python
            class MyParser(BaseToolCallParser):
                def extract(self, text: str) -> Dict[str, Any]:
                    # Implementation
                    return {"tool_calls": [...]}

            parser = MyParser({"clean_tokens": ["<START>", "<END>"]})
            result = parser.parse("some text with tool calls")
            ```
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def parse(self, text: str) -> Dict[str, Any]:
        """Main entry point for parsing tool calls from text.

        Orchestrates the parsing process through cleaning, extraction,
        and validation steps.

        Args:
            text (str): Raw input text containing tool calls.

        Returns:
            Dict[str, Any]: Parsed tool calls or error information.
                Success format: {"tool_calls": [...]}
                Error format: {"error": "error message"}
        """
        cleaned_text = self.clean_text(text)
        try:
            extracted_data = self.extract(cleaned_text)
            if "tool_calls" in extracted_data:
                for tool_call in extracted_data["tool_calls"]:
                    if "parameters" in tool_call:
                        tool_call["arguments"] = tool_call.pop("parameters")
            if "parameters" in extracted_data:
                extracted_data["arguments"] = extracted_data.pop("parameters")
                print(f"updated extracted_data: {extracted_data}")
            if "error" not in extracted_data:
                if self.validate(extracted_data):
                    return extracted_data
            return extracted_data
        except ValueError as e:
            self.logger.error(f"Validation error: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            self.logger.error("Unexpected error during parsing", exc_info=True)
            return {"error": f"Unexpected error: {str(e)}"}

    def clean_text(self, text: str) -> str:
        """Clean input text by removing specified tokens.

        Args:
            text (str): Raw input text to clean.

        Returns:
            str: Cleaned text with tokens removed and whitespace stripped.
        """
        tokens = self.config.get("clean_tokens", [])
        for token in tokens:
            text = text.replace(token, "")
        return text.strip()

    @staticmethod
    def validate(data: Dict[str, Any]) -> bool:
        """Validate the structure of extracted tool calls.

        Args:
            data (Dict[str, Any]): Extracted tool call data to validate.

        Returns:
            bool: True if validation passes.

        Raises:
            ValueError: If validation fails with specific reason.
        """
        tool_calls = data.get("tool_calls")
        if not isinstance(tool_calls, list):
            raise ValueError("Expected a list of tool calls")

        for call in tool_calls:
            if not isinstance(call, dict):
                raise ValueError("Each tool call must be a dictionary")
            if "name" not in call or "arguments" not in call:
                raise ValueError("Each tool call must contain 'name' and ('arguments' or 'parameters') keys)")

        return True

    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract tool calls from cleaned text.

        Must be implemented by subclasses to define specific extraction logic.

        Args:
            text (str): Cleaned input text.

        Returns:
            Dict[str, Any]: Extracted tool calls or error information.
        """
        pass
