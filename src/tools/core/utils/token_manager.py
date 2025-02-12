# src/utils/token_manager.py

import time
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any


class OAuth2ClientCredentialsManager:
    """
    Manages OAuth2 token lifecycle including acquisition and refresh.

    Handles token expiration and thread-safe token refresh using asyncio locks.
    Implements proper logging for debugging and monitoring token lifecycle events.
    """

    def __init__(
            self,
            api_key: str,
            client_secret_base64: str,
            token_url: str,
            refresh_buffer: int = 60,
            logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize the TokenManager.

        Args:
            api_key: API key for authentication
            client_secret_base64: Base64 encoded client secret
            token_url: OAuth2 token endpoint URL
            refresh_buffer: Seconds before expiry to trigger refresh
            logger: Optional custom logger instance

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate inputs
        if not all([api_key, client_secret_base64, token_url]):
            raise ValueError("api_key, client_secret_base64, and token_url are required")
        if not token_url.startswith(('http://', 'https://')):
            raise ValueError("token_url must be a valid HTTP(S) URL")
        if refresh_buffer < 0:
            raise ValueError("refresh_buffer must be non-negative")

        self.api_key = api_key
        self.client_secret_base64 = client_secret_base64
        self.token_url = token_url
        self.refresh_buffer = refresh_buffer

        # Token state
        self.access_token: Optional[str] = None
        self.expiry_time: float = 0
        self.lock = asyncio.Lock()

        # Set up logging
        self.logger = logger or logging.getLogger(__name__)

    async def _is_token_expired(self) -> bool:
        """
        Check if the current token is expired or close to expiration.

        Returns:
            bool: True if token is expired or close to expiry, False otherwise
        """
        current_time = time.time()
        token_expired = (
                self.access_token is None or
                current_time > (self.expiry_time - self.refresh_buffer)
        )

        if token_expired:
            self.logger.debug(
                "Token status: expired or near expiry. "
                f"Current time: {current_time}, Expiry time: {self.expiry_time}"
            )
        return token_expired

    async def _refresh_token(self) -> None:
        """
        Refresh the OAuth token by making an async API request.

        Raises:
            aiohttp.ClientError: If network request fails
            ValueError: If authentication fails
            Exception: For unexpected errors during token refresh
        """
        async with self.lock:
            # Double-check expiration after acquiring lock
            if not await self._is_token_expired():
                self.logger.debug("Token was refreshed by another task")
                return

            try:
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "apikey": self.api_key,
                    "Authorization": f"Basic {self.client_secret_base64}"
                }
                payload = {
                    "grant_type": "client_credentials",
                    "scope": "public"
                }

                self.logger.debug(f"Attempting to refresh token from {self.token_url}")
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            self.token_url,
                            headers=headers,
                            data=payload
                    ) as response:
                        if response.status == 401:
                            self.logger.error(
                                "Authentication failed during token refresh. "
                                "Check credentials."
                            )
                            raise ValueError("Authentication failed")

                        response.raise_for_status()
                        token_info: Dict[str, Any] = await response.json()

                        self.access_token = token_info["access_token"]
                        expires_in = int(token_info["expires_in"])
                        self.expiry_time = time.time() + expires_in

                        self.logger.info(f"Token refreshed successfully. Expires in {expires_in} seconds.")

            except aiohttp.ClientError as e:
                self.logger.error(f"Network error during token refresh: {str(e)}")
                self.access_token = None
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during token refresh: {str(e)}")
                self.access_token = None
                raise

    async def get_token(self) -> Optional[str]:
        """
        Get the current access token, refreshing if necessary.

        Returns:
            str: The current access token if valid
            None: If token refresh fails

        Raises:
            May raise exceptions from _refresh_token() if refresh fails
        """
        if await self._is_token_expired():
            self.logger.debug("Token expired, initiating refresh")
            await self._refresh_token()
        return self.access_token
