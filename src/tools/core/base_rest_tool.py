# src/tools/core/base_rest_tool.py

import os
import json
import time
import aiohttp
import logging
import traceback
from enum import Enum
from abc import abstractmethod
from dotenv import load_dotenv
from asyncio import Lock, sleep
from typing import Optional, Dict, Any, Union, List

from src.tools.core.base_tool import BaseTool
from src.data_models.agent import StreamContext
from src.tools.core.utils.token_manager import OAuth2ClientCredentialsManager

load_dotenv()


class HttpMethod(Enum):
    """Supported HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ResponseFormat(Enum):
    """Supported response formats"""
    JSON = "json"
    TEXT = "text"
    BINARY = "binary"


class BaseRESTTool(BaseTool):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Base configuration
        self.endpoint: str = self.config.get("endpoint_url")
        self.token_url = self.config.get("token_url")
        self.api_key_env = self.config.get("api_key_env")
        self.client_secret_env = self.config.get("client_secret_env")

        # Additional OpenAI tool configuration
        self.strict = self.config.get("strict", False)

        # Enhanced configuration (keep existing config...)
        self.content_type = self.config.get("content_type", "application/json")
        self.rate_limit = self.config.get("rate_limit", 0)
        self.default_timeout = self.config.get("default_timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)

        # Initialize rate limiting
        self._request_lock = Lock()
        self._last_request_time = 0

        # Validate required configuration
        if not self.endpoint:
            raise ValueError("The 'endpoint_url' is required in the configuration.")

        # Setup authentication
        api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        client_secret = os.getenv(self.client_secret_env) if self.client_secret_env else None

        if self.token_url and api_key and client_secret:
            self.token_manager = OAuth2ClientCredentialsManager(
                api_key=api_key,
                client_secret_base64=client_secret,
                token_url=self.token_url
            )
        else:
            self.token_manager = None

        # Initialize middleware hooks
        self._request_middleware: List[callable] = []
        self._response_middleware: List[callable] = []

        # Ensure parameters has the correct structure
        if not self.parameters:
            self.parameters = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }

    def add_request_middleware(self, middleware: callable):
        """Add middleware function to modify requests before sending.

        Args:
            middleware (callable): Function that takes and returns a request dict.
        """
        self._request_middleware.append(middleware)

    def add_response_middleware(self, middleware: callable):
        """Add middleware function to modify responses before returning.

        Args:
            middleware (callable): Function that takes and returns a response object.
        """
        self._response_middleware.append(middleware)

    async def _apply_request_middleware(self, request_data: Dict) -> Dict:
        """Apply all request middleware functions in order."""
        for middleware in self._request_middleware:
            request_data = await middleware(request_data)
        return request_data

    async def _apply_response_middleware(self, response_data: Any) -> Any:
        """Apply all response middleware functions in order."""
        for middleware in self._response_middleware:
            response_data = await middleware(response_data)
        return response_data

    async def _enforce_rate_limit(self):
        """Enforce the configured rate limit."""
        if self.rate_limit > 0:
            async with self._request_lock:
                current_time = time.time()
                elapsed = current_time - self._last_request_time
                if elapsed < 1.0 / self.rate_limit:
                    await sleep(1.0 / self.rate_limit - elapsed)
                self._last_request_time = time.time()

    def _get_cache_key(self, method: str, endpoint_url: str, params: Optional[Dict], data: Optional[Dict]) -> str:
        """Generate a cache key for the request."""
        return f"{method}:{endpoint_url}:{hash(frozenset(params.items() if params else ()))}:{hash(frozenset(data.items() if data else ()))}"

    async def get_access_token(self) -> Optional[str]:
        """Retrieve access token for API authentication.

        Returns:
            Optional[str]: Access token if available, None otherwise.

        Raises:
            RuntimeError: If token retrieval fails.
        """
        if self.token_manager:
            try:
                access_token = await self.token_manager.get_token()
                if not access_token:
                    error_msg = "Failed to retrieve access token - token manager returned None"
                    stack_trace = ''.join(traceback.format_stack()[:-1])
                    raise RuntimeError(f"{error_msg}\nStack trace:\n{stack_trace}")
                return access_token
            except Exception as e:
                stack_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                raise RuntimeError(f"Error retrieving access token: {str(e)}\nStack trace:\n{stack_trace}") from e
        return None

    async def make_request(
            self,
            method: Union[str, HttpMethod],
            params: Optional[Dict] = None,
            data: Optional[Dict] = None,
            use_token: bool = True,
            endpoint_url: Optional[str] = None,
            additional_headers: Optional[Dict] = None,
            response_format: Union[str, ResponseFormat] = ResponseFormat.JSON,
            timeout: Optional[float] = None
    ) -> Any:
        """Make an HTTP request with enhanced features.

        Args:
            method: HTTP method to use.
            params: Query parameters.
            data: Request body data.
            use_token: Whether to use authentication token.
            endpoint_url: Override default endpoint URL.
            additional_headers: Additional HTTP headers.
            response_format: Desired response format.
            timeout: Request timeout in seconds.

        Returns:
            Any: Response data in the specified format.

        Raises:
            ValueError: If response format is unsupported.
            aiohttp.ClientError: On network errors.
        """
        # Normalize parameters
        if isinstance(method, HttpMethod):
            method = method.value
        if isinstance(response_format, ResponseFormat):
            response_format = response_format.value
        endpoint_url = endpoint_url or self.endpoint
        timeout = timeout or self.default_timeout

        # Prepare headers
        headers = {
            "Content-Type": self.content_type,
            "Cache-Control": "no-cache"
        }
        if self.api_key_env:
            headers["apikey"] = os.getenv(self.api_key_env)
        if additional_headers:
            headers.update(additional_headers)

        # Add authentication token if required
        if use_token and self.token_manager:
            headers["Authorization"] = f"Bearer {await self.get_access_token()}"

        # Prepare request data for middleware
        request_data = {
            "method": method,
            "url": endpoint_url,
            "headers": headers,
            "params": params,
            "data": data,
            "timeout": timeout
        }

        # Apply request middleware
        request_data = await self._apply_request_middleware(request_data)

        # Enforce rate limit
        await self._enforce_rate_limit()

        # Make request with retry logic
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(**request_data) as response:
                        # Handle response based on status code
                        if response.status == 200:
                            if response_format == ResponseFormat.JSON.value:
                                try:
                                    result = await response.json()
                                except json.JSONDecodeError:
                                    self.logger.error("Failed to decode JSON from response")
                                    return {"error": "Invalid JSON response from server"}
                            elif response_format == ResponseFormat.TEXT.value:
                                result = await response.text()
                            elif response_format == ResponseFormat.BINARY.value:
                                result = await response.read()
                            else:
                                raise ValueError(f"Unsupported response format: {response_format}")

                            # Apply response middleware
                            result = await self._apply_response_middleware(result)

                            return result

                        # Handle error responses
                        error_response = await response.text()
                        if response.status == 400:
                            return {"error": f"Bad Request: {error_response}"}
                        elif response.status == 401:
                            return {"error": "Unauthorized access - check API key or token."}
                        elif response.status == 403:
                            return {"error": "Forbidden - insufficient permissions."}
                        elif response.status == 404:
                            return {"error": "Resource not found - verify endpoint URL."}
                        elif response.status >= 500:
                            if attempt < self.max_retries - 1:
                                await sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                                continue
                            return {"error": "Server error - the API is currently unavailable."}
                        else:
                            return {"error": f"Unexpected status code {response.status}: {error_response}"}

            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    await sleep(self.retry_delay * (2 ** attempt))
                    continue
                self.logger.error(f"Network error: {str(e)}", exc_info=True)
                return {"error": f"Network error: {str(e)}"}

            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                return {"error": f"Unexpected error: {str(e)}"}

    @abstractmethod
    async def execute(self, context: Optional[StreamContext] = None, **kwargs):
        """Execute the tool's main functionality."""
        pass

    @abstractmethod
    def parse_output(self, output: str):
        """Parse the tool's output."""
        pass
