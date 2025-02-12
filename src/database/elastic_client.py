import os
from elasticsearch import AsyncElasticsearch
from src.database.base_adapter import DatabaseAdapter


class ElasticsearchClient(DatabaseAdapter):
    """Asynchronous Elasticsearch client adapter for database operations.

    This class implements the DatabaseAdapter interface to provide asynchronous
    interaction with Elasticsearch. It handles document indexing, searching,
    index management, and other essential Elasticsearch operations.

    The client requires Elasticsearch endpoint and API key to be set in environment
    variables ES_ENDPOINT and ES_API_KEY respectively.

    Attributes:
        client (AsyncElasticsearch): Async Elasticsearch client instance configured
            with the provided endpoint and API key.

    Raises:
        ValueError: If either ES_ENDPOINT or ES_API_KEY environment variables are not set.

    Example:
        ```python
        # Initialize the client
        es_client = ElasticsearchClient()

        # Add a document
        document = {"title": "Example", "content": "Sample text"}
        await es_client.add(document, "my_index")

        # Search for documents
        query = {"query": {"match": {"content": "sample"}}}
        results = await es_client.search(query, "my_index")
        ```
    """
    def __init__(self, verify_certs: bool = True):
        """Initialize the Elasticsearch client with credentials from environment variables.

        Establishes connection to Elasticsearch using endpoint and API key from
        environment variables. Sets up the async client with specific configurations
        for SSL verification and request timeout.

        Raises:
            ValueError: If required environment variables are not set.
        """
        es_endpoint = os.getenv("ES_ENDPOINT")
        es_api_key = os.getenv("ES_API_KEY")

        if not es_endpoint or not es_api_key:
            raise ValueError("Elasticsearch endpoint and API key must be provided.")

        self.client = AsyncElasticsearch(
            es_endpoint,
            api_key=es_api_key,
            verify_certs=verify_certs,
            request_timeout=60,
        )

    async def add(self, document, index_name):
        """Add a document to the specified Elasticsearch index.

        Args:
            document: Dictionary containing the document data to be indexed.
            index_name (str): Name of the index to add the document to.

        Returns:
            dict: Elasticsearch response containing the indexing result.

        Example:
            ```python
            response = await client.add(
                {"title": "Test", "content": "Content"},
                "my_index"
            )
            ```
        """
        response = await self.client.index(index=index_name, body=document)
        return response

    async def search(self, query_body: dict, index_name: str, size: int = 5):
        """Search for documents in the specified index.

        Args:
            query_body (dict): Elasticsearch query DSL dictionary.
            index_name (str): Name of the index to search in.
            size (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            dict: Elasticsearch response containing search results.

        Example:
            ```python
            query = {
                "query": {
                    "match": {
                        "content": "search text"
                    }
                }
            }
            results = await client.search(query, "my_index", size=10)
            ```
        """
        response = await self.client.search(
            index=index_name,
            body=query_body,
            size=size,
        )
        return response

    async def reset(self, index_name):
        """Delete all documents from the specified index.

        Args:
            index_name (str): Name of the index to reset.

        Example:
            ```python
            await client.reset("my_index")
            ```
        """
        await self.client.delete_by_query(
            index=index_name,
            body={"query": {"match_all": {}}},
        )

    async def create_index(self, index_name, settings=None, mappings=None):
        """Create a new Elasticsearch index if it doesn't exist.

        Args:
            index_name (str): Name of the index to create.
            settings (dict, optional): Index settings configuration. Defaults to empty dict.
            mappings (dict, optional): Index mappings configuration. Defaults to empty dict.

        Example:
            ```python
            settings = {"number_of_shards": 1}
            mappings = {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"}
                }
            }
            await client.create_index("my_index", settings, mappings)
            ```
        """
        if not await self.client.indices.exists(index=index_name):
            await self.client.indices.create(index=index_name, body={
                "settings": settings if settings else {},
                "mappings": mappings if mappings else {}
            })

    async def index_exists(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name (str): Name of the index to check.

        Returns:
            bool: True if index exists, False otherwise.

        Example:
            ```python
            exists = await client.index_exists("my_index")
            if not exists:
                await client.create_index("my_index")
            ```
        """
        return await self.client.indices.exists(index=index_name)
