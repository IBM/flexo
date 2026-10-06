# src/tools/core/observer.py

import asyncio
import logging
from enum import Enum
from typing import Callable, List, Any, Dict, Union, Coroutine


class ToolEventType(Enum):
    """Types of tool registry events."""

    ADDED = "added"
    REMOVED = "removed"
    UPDATED = "updated"
    LIST_CHANGED = "list_changed"


class ToolUpdateEvent:
    """
    Represents an event indicating changes to MCP tool definitions.
    """

    def __init__(
        self,
        event_type: ToolEventType,
        new_tool_defs: List[Any] = None,
        removed_tool_names: List[str] = None,
        updated_tool_defs: List[Any] = None,
    ):
        self.event_type = event_type
        self.new_tool_defs = new_tool_defs or []
        self.removed_tool_names = removed_tool_names or []
        self.updated_tool_defs = updated_tool_defs or []

    def __str__(self):
        return (
            f"ToolUpdateEvent(type={self.event_type.value}, "
            f"new={len(self.new_tool_defs)}, "
            f"removed={len(self.removed_tool_names)}, "
            f"updated={len(self.updated_tool_defs)})"
        )


# Type alias for observer callbacks
ToolUpdateCallback = Callable[[ToolUpdateEvent], Union[None, Coroutine[Any, Any, None]]]


class ToolRegistryObserver:
    """
    Implements an observer pattern for tool registry updates.

    Subscribers can register callback functions that will be invoked when tool updates
        occur.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._subscribers: List[ToolUpdateCallback] = []

    def subscribe(self, callback: ToolUpdateCallback) -> None:
        """Subscribe to tool update events."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)
            self.logger.debug(
                f"Subscriber added. Total subscribers: {len(self._subscribers)}"
            )

    def unsubscribe(self, callback: ToolUpdateCallback) -> None:
        """Unsubscribe from tool update events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            self.logger.debug(
                f"Subscriber removed. Total subscribers: {len(self._subscribers)}"
            )

    async def notify(self, event: ToolUpdateEvent) -> None:
        """
        Notify all subscribers about a tool update event.
        Handles both synchronous and asynchronous callbacks.
        """
        self.logger.debug(
            f"Notifying {len(self._subscribers)} subscribers about: {event}"
        )
        for callback in self._subscribers:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self.logger.error(f"Error in subscriber callback: {e}", exc_info=True)


class MCPToolObserver:
    """
    Connects MCP notifications to the ToolRegistryObserver.
    This class bridges between MCP client notifications and the tool registry.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.observer = ToolRegistryObserver()
        self._current_tools: Dict[str, Any] = {}

    def register_with_mcp_client(self, mcp_client):
        """
        Register this observer with the MCP client to receive tool notifications.
        """
        self._mcp_client = mcp_client
        mcp_client.register_notification_handler(
            "notifications/tools/list_changed", self._handle_tools_list_changed
        )
        self.logger.info("Registered with MCP client for tool notifications")

    async def _handle_tools_list_changed(self, notification):
        """
        Handle notifications when the MCP tool list changes.
        This compares the current and new tool list to determine what changed.
        """
        try:
            mcp_tools_result = await self._mcp_client.session.list_tools()
            if not mcp_tools_result or not mcp_tools_result.tools:
                self.logger.warning("Received empty tool list from MCP")
                return

            new_tools_map = {tool.name: tool for tool in mcp_tools_result.tools}

            # Identify added, removed, and updated tools
            added_tools = []
            updated_tools = []
            for name, tool in new_tools_map.items():
                if name not in self._current_tools:
                    added_tools.append(tool)
                elif self._tools_differ(self._current_tools[name], tool):
                    updated_tools.append(tool)

            removed_tools = [
                name for name in self._current_tools.keys() if name not in new_tools_map
            ]

            self._current_tools = new_tools_map

            # Determine event type and notify subscribers
            if added_tools and not removed_tools and not updated_tools:
                event = ToolUpdateEvent(ToolEventType.ADDED, new_tool_defs=added_tools)
            elif removed_tools and not added_tools and not updated_tools:
                event = ToolUpdateEvent(
                    ToolEventType.REMOVED, removed_tool_names=removed_tools
                )
            elif updated_tools and not added_tools and not removed_tools:
                event = ToolUpdateEvent(
                    ToolEventType.UPDATED, updated_tool_defs=updated_tools
                )
            else:
                # Mixed changes or initial load
                event = ToolUpdateEvent(
                    ToolEventType.LIST_CHANGED,
                    new_tool_defs=added_tools,
                    removed_tool_names=removed_tools,
                    updated_tool_defs=updated_tools,
                )

            self.logger.info(
                f"Tool changes detected: +{len(added_tools)}, -{len(removed_tools)}, "
                f"~{len(updated_tools)}"
            )
            await self.observer.notify(event)

        except Exception as e:
            self.logger.error(f"Error handling tools list changed: {e}", exc_info=True)

    def _tools_differ(self, old_tool, new_tool) -> bool:
        """
        Compare two tool definitions to determine if they differ substantially.
        This is a simple implementation - enhance based on your tool definition
            structure.
        """
        return old_tool != new_tool
