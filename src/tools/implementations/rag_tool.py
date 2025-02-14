# src/tools/implementations/rag_tool.py

"""
WARNING: SSL certificate verification is currently disabled for Elasticsearch connections in this tool.
This configuration is intended for development/testing environments only and poses
security risks in production. Before deploying to production, proper SSL certificate
verification should be implemented to ensure secure communication with the Elasticsearch
cluster. Contact your cloud provider for the appropriate CA certificates.

For more information on secure Elasticsearch connections, see:
https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html#auth-https
"""

import json
import logging
import asyncio
from typing import Optional, Dict

from src.tools.core.base_tool import BaseTool
from src.data_models.tools import ToolResponse
from src.data_models.agent import StreamContext
from src.tools.core.tool_registry import ToolRegistry
from src.database import ElasticsearchClient, ElasticQueryBuilder


# @ToolRegistry.register_tool()
class RAGTool(BaseTool):
    name = "rag_tool"

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.strict = True

        self.description = ("Tool used to retrieve information from the 'Medicare & You 2025' handbook "
                            "using natural language search. Use this tool when you need information about "
                            "Medicare coverage, enrollment, costs, and benefits.")

        self.parameters = {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': ('Search terms related to Medicare coverage, benefits, enrollment, '
                                    'costs, or other topics from the Medicare & You 2025 handbook. '
                                    'Example: "Medicare Part B coverage limits" or "prescription drug plans"'),
                }
            },
            'required': ['query'],
            'additionalProperties': False
        }

        self.logger = logging.getLogger(self.__class__.__name__)
        elasticsearch_config = self.config.get('connector_config', {})

        # TODO(security): Temporary SSL verification bypass for development.
        # Must be updated with proper certificate verification before production deployment.
        self.search_client = ElasticsearchClient(verify_certs=False)

        self.query_builder = ElasticQueryBuilder(elasticsearch_config)
        self.top_k = self.config.get('top_k', 3)
        self.query_name = self.config.get('query_name', 'basic_match')
        self.elasticsearch_timeout = self.config.get('timeouts', {}).get('elasticsearch_timeout', 30)
        self.max_retries = elasticsearch_config.get('max_retries', 3)
        self.index_name = elasticsearch_config.get('index_name', 'medicare_handbook_2025')

    async def execute(self, context: Optional[StreamContext] = None, **kwargs) -> ToolResponse:
        """Executes the RAG (Retrieval-Augmented Generation) tool to retrieve Medicare-related content.

        This method performs a search query against an Elasticsearch index to retrieve relevant
        Medicare documentation based on the provided query string.

        Args:
            context (Optional[ContextModel], optional): Context information for the execution.
                Defaults to None.
            **kwargs: Arbitrary keyword arguments.
                Required:
                    query (str): The search query string to find Medicare-related content.

        Returns:
            ToolResponse: A structured response containing:
                - result (str): The parsed and formatted retrieved documents
                - context (Optional[Dict]): Additional execution context (None in this implementation)

        Raises:
            ValueError: If the 'query' parameter is missing or empty.

        Examples:
            ```python
            tool = RAGTool(config=rag_config)
            response = await tool.execute(query="What are Medicare Part B premiums?")
            print(response.result)
            ```
        """
        query = kwargs.get('query', '')
        if not query:
            raise ValueError("The 'query' parameter is required.")

        self.logger.info("Executing RAG Tool with query about Medicare: %s", query)

        # Retrieve content from Elasticsearch
        retrieved_documents = await self._retrieve_content(
            user_input=query,
            index_name=self.index_name,
            top_k=self.top_k
        )
        response = ToolResponse(
            result=self.parse_output(retrieved_documents),
            context=None,
        )
        return response

    async def _retrieve_content(self, user_input: str, index_name: str, top_k: int = None) -> str:
        """Retrieve content from Elasticsearch based on user query.

        Args:
            user_input (str): User's query about Medicare.
            index_name (str): Name of the Elasticsearch index.
            top_k (int, optional): Number of results to return.

        Returns:
            str: Concatenated string of retrieved handbook sections.

        Raises:
            RuntimeError: If retrieval fails.
            asyncio.TimeoutError: If query times out.
        """
        self.logger.info("Querying Elasticsearch for Medicare handbook content")
        top_k = top_k if top_k is not None else self.top_k
        query_body = self.query_builder.get_query(user_input)
        self.logger.debug(f"Elastic query body for Medicare query: {json.dumps(query_body)}")
        query_results = None

        # Perform Elasticsearch query with retries and timeout
        for attempt in range(self.max_retries):
            try:
                query_results = await asyncio.wait_for(
                    self.search_client.search(query_body, index_name, top_k),
                    timeout=self.elasticsearch_timeout
                )
                break  # Exit the loop if successful
            except asyncio.TimeoutError:
                self.logger.error(f"Elasticsearch query timed out (attempt {attempt + 1}/{self.max_retries})")
                if attempt + 1 == self.max_retries:
                    raise  # Raise the exception if max retries reached

        if query_results is None:
            raise RuntimeError("Failed to retrieve Elasticsearch query results for Medicare handbook.")

        # Extract and sort hits
        extracted_hits = self.extract_and_sort_hits(query_results, "text")

        # Concatenate up to top_k results
        retrieved_content = "\n\n".join(extracted_hits[:top_k]) + "\n\n" + self. get_tool_specific_instruction()
        return retrieved_content

    @staticmethod
    def extract_and_sort_hits(response, field_name):
        """Extract and sort hits from Elasticsearch response.

        Args:
            response: Elasticsearch query response.
            field_name (str): Field name to extract from hits.

        Returns:
            List[str]: Sorted list of extracted field values.
        """
        result = []

        def extract_fields(hit, score):
            extracted_values = []
            if field_name in hit["fields"]:
                extracted_values = hit["fields"][field_name]
            else:
                for key, values in hit["fields"].items():
                    if isinstance(values, list):
                        for value in values:
                            if isinstance(value, dict) and field_name in value:
                                extracted_values = value[field_name]
                                break

            for value in extracted_values:
                result.append({field_name: value, "_score": score})

        def process_hits(hits):
            for hit in hits:
                score = hit["_score"] if hit["_score"] is not None else 0
                if "inner_hits" in hit:
                    for _, inner_hit_value in hit["inner_hits"].items():
                        process_hits(inner_hit_value["hits"]["hits"])
                else:
                    extract_fields(hit, score)

        process_hits(response["hits"]["hits"])
        sorted_result = sorted(result, key=lambda x: x["_score"], reverse=True)
        return [entry[field_name] for entry in sorted_result]

    def parse_output(self, output: str):
        """Parse and format the retrieved content.

        Args:
            output (str): Raw content from Elasticsearch.

        Returns:
            str: Formatted content with context header.
        """
        if not output:
            return "No relevant information found in the Medicare & You 2025 handbook."

        # Return the output with a context header
        return (
            "## Retrieved Content from 'Medicare & You 2025' Handbook ##\n\n"
            f"{output}\n\n"
            "Note: This content is retrieved directly from the Medicare & You 2025 handbook. "
            "For the most up-to-date information, please visit Medicare.gov or call 1-800-MEDICARE."
        )

    def get_tool_specific_instruction(self) -> str:
        return (
            "This tool searches through the content of the 'Medicare & You 2025' "
            "handbook. Please be concise and direct in your answers, basing them off of the retrieved content."
        )
