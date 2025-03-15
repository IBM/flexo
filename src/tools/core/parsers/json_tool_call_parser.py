# src/tools/core/parsers/json_tool_call_parser.py

import re
from typing import Dict, Any, List

import json5

from src.tools.core.parsers.base_tool_call_parser import BaseToolCallParser


class JSONToolCallParser(BaseToolCallParser):
    """Enhanced parser for extracting and processing JSON tool calls from raw text.

    This parser handles common LLM JSON generation errors including:

    - Semicolons instead of commas between array items
    - Missing or extra commas
    - Unquoted keys
    - Single quotes instead of double quotes
    - Trailing commas
    - Other common JSON syntax errors
    """
    # Precompiled regex patterns for common JSON errors
    SEMICOLON_PATTERN = re.compile(r'(\}|\])\s*;\s*(\{|\[)')
    TRAILING_COMMA_PATTERN = re.compile(r',\s*(\}|\])')
    MISSING_COMMA_PATTERN = re.compile(r'(\}|\])\s*(\{|\[)')
    UNQUOTED_PROPERTY_PATTERN = re.compile(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:')

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON tool calls from the input text with enhanced error recovery.

        Args:
            text (str): The input text containing JSON tool calls.

        Returns:
            Dict[str, Any]: A dictionary containing parsed tool calls or error information.

            - Success format: `{"tool_calls": [{"name": "...", "arguments": {...}}, ...]}`
            - Error format: `{"error": "error message"}`
        """
        try:
            # Try to find JSON-like content
            json_strings = self.find_json_content(text)
            valid_calls = []

            for json_str in json_strings:
                try:
                    # First attempt: Standard parsing with json5
                    parsed = json5.loads(json_str)
                    parsed = self.parse_nested_json(parsed)

                    if isinstance(parsed, dict):
                        valid_calls.append(parsed)
                    elif isinstance(parsed, list):
                        valid_calls.extend(parsed)
                except Exception:
                    # Second attempt: Apply preprocessing to fix common issues
                    try:
                        fixed_json = self.preprocess_json(json_str)
                        parsed = json5.loads(fixed_json)
                        parsed = self.parse_nested_json(parsed)

                        if isinstance(parsed, dict):
                            valid_calls.append(parsed)
                        elif isinstance(parsed, list):
                            valid_calls.extend(parsed)
                    except Exception:
                        # If all else fails, try a more aggressive approach for lists
                        if json_str.startswith('[') and json_str.endswith(']'):
                            items = self.split_json_list_items(json_str)
                            for item in items:
                                try:
                                    parsed_item = json5.loads(item)
                                    parsed_item = self.parse_nested_json(parsed_item)
                                    valid_calls.append(parsed_item)
                                except Exception:
                                    continue

            return {"tool_calls": valid_calls} if valid_calls else {"error": "No valid tool calls found"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    @staticmethod
    def find_json_content(text: str) -> List[str]:
        """Extract potential JSON content from raw text using balanced delimiter matching.

        This method scans the text character by character to identify and extract
        valid JSON objects or arrays, handling nested structures correctly.

        Args:
            text (str): The raw text to search for JSON content.

        Returns:
            List[str]: A list of extracted JSON string segments.
        """
        results = []
        start = None
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            # Handle string literals correctly
            if char == '\\' and not escape_next:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string

            escape_next = False

            # Only process delimiters when not inside a string
            if not in_string:
                if char in '{[' and depth == 0:
                    start = i
                    depth += 1
                elif char in '{[':
                    depth += 1
                elif char in '}]':
                    depth -= 1
                    if depth == 0 and start is not None:
                        results.append(text[start:i + 1])
                        start = None

        return results

    def preprocess_json(self, json_str: str) -> str:
        """Preprocess JSON string to fix common LLM-generated syntax errors.

        Args:
            json_str (str): The potentially malformed JSON string.

        Returns:
            str: A corrected JSON string.
        """
        # Replace semicolons with commas between objects/arrays
        json_str = self.SEMICOLON_PATTERN.sub(r'\1,\2', json_str)

        # Fix trailing commas
        json_str = self.TRAILING_COMMA_PATTERN.sub(r'\1', json_str)

        # Fix missing commas between objects/arrays
        json_str = self.MISSING_COMMA_PATTERN.sub(r'\1,\2', json_str)

        # Fix unquoted property names
        json_str = self.UNQUOTED_PROPERTY_PATTERN.sub(r'\1"\2":', json_str)

        return json_str

    def split_json_list_items(self, json_list: str) -> List[str]:
        """Split a JSON array string into individual item strings for separate parsing.

        This handles cases where items are separated by semicolons or have other issues.

        Args:
            json_list (str): A string containing a JSON array with possibly invalid separators.

        Returns:
            List[str]: List of individual item strings.
        """
        # Remove the outer brackets
        content = json_list[1:-1].strip()

        items = []
        depth = 0
        start = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(content):
            # Handle string literals correctly
            if char == '\\' and not escape_next:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string

            escape_next = False

            # Track nesting level
            if not in_string:
                if char in '{[':
                    depth += 1
                elif char in '}]':
                    depth -= 1

                # When at top level, check for separators (comma or semicolon)
                if depth == 0 and char in ',;' and i >= start:
                    items.append(content[start:i].strip())
                    start = i + 1

        # Don't forget the last item
        if start < len(content):
            items.append(content[start:].strip())

        return items

    def parse_nested_json(self, value: Any) -> Any:
        """Recursively parses stringified JSON within a JSON structure.

        Args:
            value (Any): The input value to check and potentially parse.

        Returns:
            Any: The processed value, either as a parsed JSON object or as its original type.
        """
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed and trimmed[0] in ['{', '[']:
                try:
                    # Try to fix common issues and then parse
                    fixed_str = self.preprocess_json(trimmed)
                    parsed = json5.loads(fixed_str)
                    return self.parse_nested_json(parsed)
                except Exception:
                    # If that fails, return as is
                    return value
            else:
                return value
        elif isinstance(value, dict):
            return {k: self.parse_nested_json(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.parse_nested_json(item) for item in value]
        else:
            return value
