# src/tools/implementations/duck_tool.py

import re
from bs4 import BeautifulSoup
from urllib.parse import unquote
from typing import Optional, Dict

from src.data_models.tools import ToolResponse
from src.data_models.agent import StreamContext
from src.tools.core.base_rest_tool import BaseRESTTool


class DuckDuckGoSearchTool(BaseRESTTool):
    """
    A tool for searching the web using DuckDuckGo.

    This tool sends search queries to DuckDuckGo's HTML interface and extracts
    search results, including titles, snippets, and URLs. It returns up to 5
    search results formatted in Markdown.

    Attributes:
        name (str): The name of the tool, used for registry and invocation.
        description (str): Human-readable description of the tool's purpose.
        parameters (dict): JSON schema for the tool's parameters.
    """

    name = "duckduckgo_search"

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DuckDuckGo Search Tool.

        Args:
            config (Optional[Dict]): Configuration options for the tool,
                which will override the default configuration.

        Returns:
            None
        """
        default_config = {
            "endpoint_url": "https://html.duckduckgo.com/html/",
            "strict": False
        }
        if config:
            default_config.update(config)

        super().__init__(config=default_config)

        self.description = "Internet / Web search for information on a specific query."
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on DuckDuckGo."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }

    async def execute(self, context: Optional[StreamContext] = None, **kwargs) -> ToolResponse:
        """
        Execute a DuckDuckGo search.

        This method takes a query parameter and searches DuckDuckGo for relevant
        results, then formats them for display.

        Args:
            context (Optional[StreamContext]): Execution context for the tool.
            **kwargs: Keyword arguments containing the search parameters.
                Required:
                    query (str): The search query to send to DuckDuckGo.

        Returns:
            ToolResponse: A response object containing the search results
                formatted in Markdown.

        Raises:
            ValueError: If the required 'query' parameter is missing.
        """
        query = kwargs.get("query")
        if not query:
            raise ValueError("The 'query' parameter is required.")

        self.logger.info(f"Searching DuckDuckGo for: {query}")

        # Perform the search
        result = await self._search_duckduckgo(query)

        return ToolResponse(result=result, context=None)

    def parse_output(self, output: str) -> str:
        """
        Parse the raw output from DuckDuckGo.

        This method processes the HTML response from DuckDuckGo to extract
        relevant information. In this implementation, it is not directly used
        as the parsing is done in the _search_duckduckgo method for better
        error handling and control.

        Args:
            output (str): The raw HTML response from DuckDuckGo.

        Returns:
            str: Processed output as a string.

        Note:
            This method is required to implement the BaseRESTTool abstract class,
            but in this implementation, the actual parsing is done in _search_duckduckgo.
        """
        # The actual parsing is done in _search_duckduckgo for better control
        # This method is here to satisfy the BaseRESTTool abstract class requirement
        return output

    async def _search_duckduckgo(self, query: str) -> str:
        """
        Search DuckDuckGo and parse the results.

        This method sends a POST request to DuckDuckGo's HTML interface,
        parses the returned HTML to extract search results, and formats
        them in Markdown.

        Args:
            query (str): The search query to send to DuckDuckGo.

        Returns:
            str: Markdown-formatted search results, including titles,
                snippets, and links. Returns an error message if the
                search fails.

        Raises:
            No exceptions are raised; errors are caught and returned
            as error messages in the result string.
        """
        import aiohttp

        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://html.duckduckgo.com",
            "Referer": "https://html.duckduckgo.com/",
            "Connection": "keep-alive"
        }
        data = {
            "q": query,
            "b": ""
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status != 200:
                        return f"Error: DuckDuckGo returned status code {response.status}"

                    html_content = await response.text()

                    # Parse the DuckDuckGo search results
                    results = []
                    soup = BeautifulSoup(html_content, "html.parser")

                    for result in soup.select(".result"):
                        try:
                            title_element = result.select_one(".result__a")
                            snippet_element = result.select_one(".result__snippet")

                            if not title_element:
                                continue

                            title = title_element.get_text().strip()
                            url = title_element.get("href", "")

                            # Extract the actual URL from DuckDuckGo's redirect URL
                            if url.startswith("/"):
                                url_match = re.search(r'uddg=([^&]+)', url)
                                if url_match:
                                    url = unquote(url_match.group(1))

                            snippet = snippet_element.get_text().strip() if snippet_element else "No description available."

                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet
                            })

                            if len(results) >= 5:
                                break
                        except Exception as e:
                            self.logger.error(f"Error parsing result: {str(e)}")
                            continue

                    if not results:
                        return "No search results found on DuckDuckGo."

                    # Format the results
                    formatted_output = "## DuckDuckGo Search Results\n\n"
                    for i, result in enumerate(results, 1):
                        formatted_output += f"### {i}. {result['title']}\n"
                        formatted_output += f"{result['snippet']}\n"
                        formatted_output += f"[Link]({result['url']})\n\n"

                    formatted_output += "These results are from DuckDuckGo and may not reflect the latest information."
                    return formatted_output
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {str(e)}")
            return f"Error searching DuckDuckGo: {str(e)}"
