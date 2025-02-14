# src/tools/core/parsers/json_tool_call_parser.py

import json5
from typing import Dict, Any, List
from src.tools.core.parsers.base_tool_call_parser import BaseToolCallParser


class JSONToolCallParser(BaseToolCallParser):
    """Parser for extracting and processing JSON tool calls from raw text.

    This parser detects JSON structures embedded in text, extracts them,
    and parses them using `json5`. It also ensures that any nested JSON
    values stored as strings are properly converted into JSON objects.
    """
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON tool calls from the input text.

        This method scans the text for JSON segments, attempts to parse them,
        and recursively converts any nested JSON stored as string values.

        Args:
            text (str): The input text containing JSON tool calls.

        Returns:
            Dict[str, Any]: A dictionary containing parsed tool calls or error information.

            - Success format: `{"tool_calls": [{"name": "...", "arguments": {...}}, ...]}`
            - Error format: `{"error": "error message"}`
        """
        try:
            json_strings = self.find_json_content(text)
            valid_calls = []
            for json_str in json_strings:
                try:
                    parsed = json5.loads(json_str)
                    parsed = self.parse_nested_json(parsed)
                    if isinstance(parsed, dict):
                        valid_calls.append(parsed)
                    elif isinstance(parsed, list):
                        valid_calls.extend(parsed)
                except Exception:
                    continue

            return {"tool_calls": valid_calls} if valid_calls else {"error": "No valid tool calls found"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    @staticmethod
    def find_json_content(text: str) -> List[str]:
        """Extract potential JSON content from raw text using balanced delimiter matching.

        This method scans the text character by character to identify and extract
        valid JSON objects or arrays. It ensures that nested structures are handled
        correctly, but does not account for JSON within quoted strings.

        Args:
            text (str): The raw text to search for JSON content.

        Returns:
            List[str]: A list of extracted JSON string segments.
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

    def parse_nested_json(self, value):
        """Recursively parses stringified JSON within a JSON structure.

        If a value is a string that appears to be JSON (i.e., it starts with `{` or `[`),
        this method attempts to parse it. It then recursively processes the newly parsed
        object to handle any further nested JSON values.

        Args:
            value (Any): The input value to check and potentially parse.

        Returns:
            Any: The processed value, either as a parsed JSON object or as its original type.
        """
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed and trimmed[0] in ['{', '[']:
                try:
                    parsed = json5.loads(trimmed)
                    return self.parse_nested_json(parsed)
                except Exception:
                    return value
            else:
                return value
        elif isinstance(value, dict):
            return {k: self.parse_nested_json(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.parse_nested_json(item) for item in value]
        else:
            return value
