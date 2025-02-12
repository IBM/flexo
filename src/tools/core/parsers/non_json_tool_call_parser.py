# src/tools/core/parsers/non_json_tool_call_parser.py

import re
import json
from typing import Dict, Any
from src.tools.core.parsers.base_tool_call_parser import BaseToolCallParser


class NonJSONToolCallParser(BaseToolCallParser):
    """Parser for extracting non-JSON formatted tool calls from text.

    Specializes in parsing tool calls that use custom formats like
    <function=name>{args}</function>.

    Example:
        ```python
        config = {
            "formats": {
                "non_json_format": {
                    "function_call_pattern": r'<function=(.*?)>{(.*?)}</function>'
                }
            }
        }
        parser = NonJSONToolCallParser(config)
        result = parser.parse('<function=my_tool>{"arg1": "value"}</function>')
        ```
    """
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract non-JSON tool calls using regex patterns.

        Searches for tool calls using configured regex patterns and parses
        their arguments as JSON.

        Args:
            text (str): Cleaned input text containing tool calls.

        Returns:
            Dict[str, Any]: Parsed tool calls or error information.

                - Success format: {"tool_calls": [{"name": "...", "arguments": {...}}, ...]}
                - Error format: {"error": "error message"}
                - No tool calls: {"content": "original text"}
        """
        pattern = self.config.get("formats").get("non_json_format").get("function_call_pattern")
        matches = re.findall(pattern, text)

        if not matches:
            return {"content": text}

        tool_calls = []
        for match in matches:
            try:
                tool_calls.append({
                    "name": match[0],
                    "arguments": json.loads(match[1])  # Parse arguments as JSON
                })
            except json.JSONDecodeError:
                return {"error": f"Failed to parse arguments for function: {match[0]}"}

        return {"tool_calls": tool_calls}
