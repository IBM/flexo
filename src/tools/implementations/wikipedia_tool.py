# src/tools/implementations/wikipedia_tool.py

import re
from urllib.parse import quote, urlsplit
from typing import Optional, Any, Dict

from src.data_models.tools import ToolResponse
from src.data_models.agent import StreamContext
from src.utils.json_formatter import format_json_to_document
from src.tools.core.base_rest_tool import BaseRESTTool, ResponseFormat


class WikipediaTool(BaseRESTTool):
    name = "wikipedia"
    # Includes legacy Wikipedia codes such as simple, be-tarask and zh-min-nan.
    LANGUAGE_PATTERN = r"[a-z]{2,12}(?:-[a-z0-9]{1,8}){0,2}"

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the WikipediaTool with configuration options.
        """
        super().__init__(config=config)
        self.description = "Fetch a summary of a Wikipedia page for a given query."
        self.strict = False

        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The Wikipedia page title or search query "
                        "to retrieve the summary for."
                    ),
                },
                "lang": {
                    "type": "string",
                    "pattern": "^" + self.LANGUAGE_PATTERN + "$",
                    "description": (
                        "The language code for Wikipedia (e.g., 'en', 'es', 'fr'). "
                        "Defaults to 'en'."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

        # Set request settings
        self.content_type = "application/json"
        self.rate_limit = 30
        self.default_timeout = 10
        self.max_retries = 3
        self.retry_delay = 1.0

    async def execute(
        self, context: Optional[StreamContext] = None, **kwargs
    ) -> ToolResponse:
        """
        Execute the Wikipedia API call.

        Args:
            context (Optional[ContextModel]): Additional context for the request.
            **kwargs: Should include:
                - query (str): The Wikipedia page title or search query.
                - lang (str, optional): The language code (default is "en").

        Returns:
            ToolResponse: The tool's response with the Wikipedia summary.
        """
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The 'query' parameter must be a non-empty string.")

        lang = kwargs.get("lang", "en")
        if not isinstance(lang, str) or not re.fullmatch(self.LANGUAGE_PATTERN, lang):
            raise ValueError("The 'lang' parameter must be a Wikipedia language code.")
        encoded_query = quote(query, safe="")
        endpoint_url = self.endpoint.format(
            **{"lang": lang, "encoded_query": encoded_query}
        )
        destination = urlsplit(endpoint_url)
        if (
            destination.scheme != "https"
            or destination.hostname != f"{lang}.wikipedia.org"
            or destination.port not in (None, 443)
            or destination.username is not None
            or destination.password is not None
        ):
            raise ValueError(
                "The Wikipedia endpoint must use HTTPS "
                "on the selected Wikipedia language host."
            )

        self.logger.info(
            f"Fetching Wikipedia summary for query: '{query}' in language: '{lang}'"
        )
        self.logger.debug(f"Endpoint URL: {endpoint_url}")
        self.logger.debug(f"Context: {context}")

        response = await self.make_request(
            method="GET",
            endpoint_url=endpoint_url,
            response_format=ResponseFormat.JSON,
            use_token=False,  # No token required for the Wikipedia API.
            # Redirect targets must not bypass destination validation.
            allow_redirects=False,
            additional_headers={
                "Accept": "application/json",
                "User-Agent": "WikipediaTool/1.0",
            },
        )

        tool_response = ToolResponse(
            result=self.parse_output(response),
            context=None,
        )
        return tool_response

    def parse_output(self, output: Any) -> str:
        """
        Parse and format the Wikipedia API response.

        Args:
            output (Any): Raw API response data.

        Returns:
            str: A formatted string containing the page title, summary, and a link to
                the full page.
        """
        try:
            if not isinstance(output, dict):
                return str(output)

            # Check if the API indicates a missing page or error
            if (
                output.get("type")
                == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found"
            ):
                return "Error: The page for the given query was not found on Wikipedia."

            if "detail" in output:
                return f"Error: {output['detail']}"

            summary_data = {
                "title": output.get("title"),
                "summary": output.get("extract"),
                "page_url": output.get("content_urls", {})
                .get("desktop", {})
                .get("page"),
            }

            formatted_output = format_json_to_document(summary_data)
            formatted_output += self.get_tool_specific_instruction()
            return formatted_output

        except Exception as e:
            self.logger.error(f"Failed to parse Wikipedia data: {e}", exc_info=True)
            return "An error occurred while parsing the Wikipedia summary."

    def get_tool_specific_instruction(self) -> str:
        """
        Optional instructions to append to the output.

        Returns:
            str: Additional information.
        """
        return "\n\nFor more details, please visit the Wikipedia page."
