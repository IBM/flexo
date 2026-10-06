# src/mcp/client.py

import logging
from typing import Optional, Any, Dict, Callable, Awaitable

from mcp import types
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

from src.tools.core.observer import MCPToolObserver

logger = logging.getLogger(__name__)


class FlexoMCPClient:
    """A client wrapper for the MCP SDK's ClientSession, supporting both SSE and Stdio
        transports.

    This class manages the connection, message processing, and notification handling
        with the MCP server.

    Attributes:
        config (dict[str, Any]): Configuration dictionary.
        observer (MCPToolObserver): Observer instance for handling tool notifications.
    """

    def __init__(self, config: dict[str, Any]):
        """Initializes the FlexoMCPClient.

        Args:
            config (dict[str, Any]): A dictionary that may contain:
                {
                    "transport": "sse" or "stdio",
                    "sse_url": <SSE endpoint URL>,
                    "command": <stdio command, e.g. 'python'>,
                    "args": <list of arguments for stdio process>,
                    "env": <dict of environment variables for stdio process>,
                    ...
                }
        """
        self.config = config
        self._session: Optional[ClientSession] = None
        self._streams = None
        self._transport_cm = None
        self._connected = False
        self._notification_handlers: Dict[str, Callable[[Any], Awaitable[None]]] = {}
        self.observer = MCPToolObserver()

    async def __aenter__(self) -> "FlexoMCPClient":
        """Asynchronous context manager entry.

        Returns:
            FlexoMCPClient: The connected MCP client instance.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Asynchronous context manager exit.

        Closes the MCP client connection.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        await self.close()

    async def connect(self) -> None:
        """Establishes the connection to the MCP server using SSE or Stdio transport.

        Raises:
            ValueError: If an unsupported transport type is specified.
        """
        transport = self.config.get("transport", "sse")
        if transport == "sse":
            await self._connect_sse()
        elif transport == "stdio":
            await self._connect_stdio()
        else:
            raise ValueError(f"Unsupported transport: {transport}")

        self._register_default_notification_handlers()
        self.observer.register_with_mcp_client(self)

        # The SDK owns the receive loop and dispatches messages to our callback.
        await self._session.__aenter__()

        # Initialize the session.
        init_result = await self._session.initialize()
        logger.info(
            "Connected to server: %s %s",
            init_result.server_info.name,
            init_result.server_info.version,
        )

        self._connected = True
        logger.info("MCP client session initialized successfully.")

    async def _connect_sse(self) -> None:
        """Connects to the MCP server using SSE transport.

        Uses the 'sse_url' from the configuration.

        Raises:
            Exception: If connection fails.
        """
        sse_url = self.config.get("sse_url", "http://localhost:8080/sse")
        logger.info(f"Connecting to SSE server at {sse_url}...")
        self._transport_cm = sse_client(sse_url)
        self._streams = await self._transport_cm.__aenter__()
        self._session = ClientSession(
            self._streams[0], self._streams[1], message_handler=self._handle_message
        )

    async def _connect_stdio(self) -> None:
        """Connects to the MCP server using Stdio transport.

        Uses the 'command', 'args', and 'env' from the configuration.
        """
        command = self.config.get("command", "python")
        args = self.config.get("args", [])
        env = self.config.get("env", None)
        server_params = StdioServerParameters(command=command, args=args, env=env)
        logger.info(f"Connecting to stdio server with: {command} {args} ...")
        self._transport_cm = stdio_client(server_params)
        self._streams = await self._transport_cm.__aenter__()
        self._session = ClientSession(
            self._streams[0], self._streams[1], message_handler=self._handle_message
        )

    def _register_default_notification_handlers(self) -> None:
        """Registers the built-in notification handlers."""
        handlers = {
            "notifications/tools/list_changed": self._handle_tools_list_changed,
            "notifications/resources/list_changed": self._handle_resources_list_changed,
            "notifications/prompts/list_changed": self._handle_prompts_list_changed,
            "notifications/resources/updated": self._handle_resource_updated,
            "notifications/logging/message": self._handle_logging_message,
        }
        for method, handler in handlers.items():
            self.register_notification_handler(method, handler)

    async def close(self) -> None:
        """Gracefully closes the MCP client session and transport context."""
        if self._connected and self._session:
            await self._session.__aexit__(None, None, None)
        if self._transport_cm:
            await self._transport_cm.__aexit__(None, None, None)
        self._connected = False
        self._session = None
        self._streams = None
        logger.info("MCP client closed successfully.")

    @property
    def session(self) -> ClientSession:
        """Gets the underlying ClientSession.

        Returns:
            ClientSession: The active client session.

        Raises:
            RuntimeError: If the session is not connected.
        """
        if not self._session:
            raise RuntimeError(
                "MCP client session not connected. Call connect() first."
            )
        return self._session

    async def _handle_message(
        self, message: types.ServerNotification | Exception
    ) -> None:
        """Receive notifications and transport errors from the SDK."""
        if isinstance(message, Exception):
            logger.error("Error in MCP communication: %s", message)
        else:
            await self._handle_notification(message)

    async def _handle_notification(
        self, notification: types.ServerNotification
    ) -> None:
        """Handles server notifications based on their type.

        Args:
            notification (types.ServerNotification): The notification received from the
                server.
        """
        method = notification.method
        logger.debug(f"Received notification: {method}")

        handler = self._notification_handlers.get(method)
        if handler:
            try:
                await handler(notification)
            except Exception as e:
                logger.exception(f"Error in notification handler for {method}: {e}")
        else:
            logger.info(f"Received unhandled notification: {method}")

    def register_notification_handler(
        self, method: str, handler: Callable[[Any], Awaitable[None]]
    ) -> None:
        """Registers a handler for a specific notification type.

        Args:
            method (str): The notification method name.
            handler (Callable[[Any], Awaitable[None]]): The asynchronous handler
                function.
        """
        self._notification_handlers[method] = handler
        logger.debug(f"Registered handler for {method}")

    # --- Notification Handlers ---
    async def _handle_tools_list_changed(
        self, notification: types.ServerNotification
    ) -> None:
        """Handles notifications for tools list changes by delegating to the observer.

        Args:
            notification (types.ServerNotification): The notification for tools list
                change.
        """
        logger.info("Tools list changed notification received")
        # Implementation pending

    async def _handle_resources_list_changed(
        self, notification: types.ServerNotification
    ) -> None:
        """Handles notifications for resources list changes.

        Args:
            notification (types.ServerNotification): The notification for resources list
                change.
        """
        logger.info("Resources list changed notification received")
        # Implementation pending

    async def _handle_prompts_list_changed(
        self, notification: types.ServerNotification
    ) -> None:
        """Handles notifications for prompts list changes.

        Args:
            notification (types.ServerNotification): The notification for prompts list
                change.
        """
        logger.info("Prompts list changed notification received")
        # Implementation pending

    async def _handle_resource_updated(
        self, notification: types.ServerNotification
    ) -> None:
        """Handles notifications for resource updates.

        Args:
            notification (types.ServerNotification): The notification for resource
                update.
        """
        logger.info("Resource updated notification received")
        # Implementation pending

    async def _handle_logging_message(
        self, notification: types.ServerNotification
    ) -> None:
        """Handles logging message notifications from the server.

        Args:
            notification (types.ServerNotification): The notification containing logging
                info.
        """
        params = getattr(notification, "params", None)
        if params and hasattr(params, "level") and hasattr(params, "message"):
            level = params.level
            message = params.message
            if level == "error":
                logger.error(f"MCP server log: {message}")
            elif level == "warning":
                logger.warning(f"MCP server log: {message}")
            else:
                logger.info(f"MCP server log: {message}")
        else:
            logger.info(f"MCP server log (format unknown): {params}")

    # --- Convenience Methods for Common Operations ---
    async def list_tools(self) -> types.ListToolsResult:
        """Lists available tools from the MCP server.

        Returns:
            types.ListToolsResult: The result containing available tools.
        """
        return await self.session.list_tools()

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        """Calls a tool on the MCP server.

        Args:
            name (str): The name of the tool to call.
            arguments (dict[str, Any]): A dictionary of arguments for the tool.

        Returns:
            types.CallToolResult: The result from calling the tool.
        """
        return await self.session.call_tool(name, arguments)

    async def subscribe_resource(self, uri: str) -> Any:
        """Subscribes to updates for a specific resource.

        Args:
            uri (str): The URI of the resource to subscribe to.

        Returns:
            Any: The subscription result.
        """
        return await self.session.subscribe_resource(uri)

    async def unsubscribe_resource(self, uri: str) -> Any:
        """Unsubscribes from updates for a specific resource.

        Args:
            uri (str): The URI of the resource to unsubscribe from.

        Returns:
            Any: The unsubscription result.
        """
        return await self.session.unsubscribe_resource(uri)
