# src/tools/core/parsers/json_tool_call_parser.py

import json5
from typing import Dict, Any, List
from src.tools.core.parsers.base_tool_call_parser import BaseToolCallParser


class JSONToolCallParser(BaseToolCallParser):
    """Parser for extracting JSON-formatted tool calls from text.

    This parser specializes in finding and parsing JSON objects or arrays
    that represent tool calls within text content.

    Example:
        ```python
        parser = JSONToolCallParser({"clean_tokens": []})
        result = parser.parse('{"name": "my_tool", "arguments": {"arg1": "value"}}')
        ```
    """
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON tool calls from text.

        Finds potential JSON content using delimiter matching and attempts
        to parse valid JSON tool calls.

        Args:
            text (str): Cleaned input text containing JSON tool calls.

        Returns:
            Dict[str, Any]: Parsed tool calls or error information.

                - Success format: {"tool_calls": [{"name": "...", "arguments": {...}}, ...]}
                - Error format: {"error": "error message"}
        """
        try:
            # Find potential JSON strings using balanced delimiter matching
            json_strings = self.find_json_content(text)
            valid_calls = []

            # Try to parse each potential JSON string
            for json_str in json_strings:
                try:
                    parsed = json5.loads(json_str)
                    # Handle both single objects and arrays
                    if isinstance(parsed, dict):
                        valid_calls.append(parsed)
                    elif isinstance(parsed, list):
                        valid_calls.extend(parsed)
                except ValueError:
                    continue

            return {"tool_calls": valid_calls} if valid_calls else {"error": "No valid tool calls found"}

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    @staticmethod
    def find_json_content(text: str) -> List[str]:
        """Find potential JSON content using balanced delimiter matching.

        Uses a state machine approach to find balanced JSON objects or arrays
        within the text.

        Args:
            text (str): Input text to search for JSON content.

        Returns:
            List[str]: List of potential JSON strings found in the text.
        """
        results = []
        start = None
        depth = 0

        for i, char in enumerate(text):
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
