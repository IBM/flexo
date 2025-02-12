# src/tools/core/tool_registry.py

import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Optional, Callable

from src.data_models.tools import Tool
from src.tools.core.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """A registry for managing and accessing system tools with decorator support.

    This class implements a singleton pattern to ensure a single registry instance
    manages all tools across the system. It supports automatic tool discovery from
    the src/tools/implementations/ directory, registration via decorators, and
    separate handling of hidden tools.

    Attributes:
        tools (Dict[str, BaseTool]): Dictionary of registered visible tools.
        hidden_tools (Dict[str, BaseTool]): Dictionary of registered hidden tools.
        config (Dict): Configuration settings for all tools.
        initialized (bool): Flag indicating if the registry has been initialized.

    Example:
        ```python
        # Initialize registry with configuration
        config = {
            "weather_tool": {"api_key": "abc123"},
            "rag_tool": {"model": "gpt-4"}
        }
        registry = ToolRegistry(config)

        # Register a tool using the decorator
        @ToolRegistry.register_tool()
        class WeatherTool(BaseTool):
            pass

        # Access a registered tool
        weather_tool = registry.get_tool("weather")
        ```
    """

    _instance = None
    _registered_tool_classes: Dict[str, tuple[Type[BaseTool], bool]] = {}

    def __new__(cls, *args, **kwargs) -> 'ToolRegistry':
        """Ensure singleton instance of ToolRegistry.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ToolRegistry: The singleton instance of the registry.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialize the ToolRegistry with configuration settings.

        Performs one-time initialization of the registry, including tool discovery
        and configuration. Subsequent initializations with the same instance will
        not repeat these steps.

        Args:
            config: Optional dictionary containing configuration settings for tools.
                Each tool's config should be keyed by '{tool_name}_tool'.
        """
        if not hasattr(self, 'initialized'):
            logger.info("Initializing ToolRegistry")
            self.tools: Dict[str, BaseTool] = {}
            self.hidden_tools: Dict[str, BaseTool] = {}
            self.config = config or {}
            self.initialized = True

            self._discover_tools()

            if config:
                self._initialize_registered_tools()

    def _discover_tools(self) -> None:
        """Discover and import all tool modules to trigger decorators.

        Scans the src/tools/implementations/ directory for Python modules and imports them
        to trigger the registration decorators. Only registers classes that inherit from BaseTool.
        Note that actual registration is expected to occur via decorators; this method
        performs additional validation by inspecting each module.
        """
        try:
            # Get the current file's directory and navigate to implementations
            current_dir = Path(__file__).parent.parent
            implementations_path = current_dir / 'implementations'

            logger.debug(f"Scanning for tools in: {implementations_path}")

            if not implementations_path.exists():
                logger.error(f"Implementations directory not found at: {implementations_path}")
                return

            # Iterate through .py files in the implementations directory
            for file_path in implementations_path.glob('*.py'):
                if not file_path.name.startswith('_'):
                    module_name = f"src.tools.implementations.{file_path.stem}"
                    try:
                        module = importlib.import_module(module_name)

                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, BaseTool) and obj != BaseTool:
                                tool_name = obj.__name__.lower().replace('tool', '')
                                # Note: Registration still happens via decorator.
                                # This is just additional validation.
                                if tool_name in self._registered_tool_classes:
                                    tool_class, is_hidden = self._registered_tool_classes[tool_name]
                                    if not issubclass(tool_class, BaseTool):
                                        logger.warning(
                                            f"Skipping {tool_name}: not a BaseTool subclass"
                                        )
                                        del self._registered_tool_classes[tool_name]

                        logger.debug(f"Processed module: {module_name}")

                    except ImportError as e:
                        logger.error(f"Failed to import {module_name}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing {module_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")

        logger.info(
            f"Tool discovery complete. Found {len(self._registered_tool_classes)} valid tools"
        )

    def _initialize_registered_tools(self) -> None:
        """Initialize all registered tool classes with their configurations.

        Creates instances of all registered tool classes using their corresponding
        configurations from the config dictionary provided during initialization.

        Note:
            This is automatically called during initialization when config is provided
            and typically shouldn't be called directly.

        Raises:
            Exception: If tool initialization fails, the original exception is logged
                and re-raised.
        """
        logger.debug("Initializing registered tools")
        for tool_name, (tool_class, is_hidden) in self._registered_tool_classes.items():
            tool_config = self.config.get(f"{tool_name}_tool")
            if tool_config:
                try:
                    tool_instance = tool_class(config=tool_config)
                    self.register_tool_instance(tool_instance, is_hidden)
                    logger.info(f"Successfully initialized {tool_name} tool")
                except Exception as e:
                    logger.error(f"Failed to initialize {tool_name} tool: {str(e)}")
                    raise

    @classmethod
    def register_tool(cls, hidden: bool = False) -> Callable:
        """Class method decorator for registering tool classes.

        Args:
            hidden: If True, the tool will be registered as a hidden tool,
                accessible only through get_hidden_tool() and not seen by the LLM.

        Returns:
            A decorator function that registers the tool class.

        Example:
            ```python
            @ToolRegistry.register_tool(hidden=True)
            class InternalTool(BaseTool):
                pass
            ```
        """

        def decorator(tool_class: Type[BaseTool]) -> Type[BaseTool]:
            tool_name = tool_class.__name__.lower().replace('tool', '')
            cls._registered_tool_classes[tool_name] = (tool_class, hidden)
            logger.debug(f"Registered tool class: {tool_class.__name__}")
            return tool_class

        return decorator

    def register_tool_instance(self, tool: BaseTool, hidden: bool = False) -> None:
        """Register a tool instance in the registry.

        Args:
            tool: The tool instance to register.
            hidden: If True, registers the tool as a hidden tool.

        Note:
            If a tool with the same name already exists in the target registry
            (hidden or visible), it will be overwritten with a warning.
        """
        logger.debug(f"Registering tool instance: {tool.name} (hidden={hidden})")

        if hidden:
            if tool.name in self.hidden_tools:
                logger.warning(f"Hidden tool {tool.name} already registered, overwriting")
            self.hidden_tools[tool.name] = tool
        else:
            if tool.name in self.tools:
                logger.warning(f"Tool {tool.name} already registered, overwriting")
            self.tools[tool.name] = tool

        logger.debug(f"Tool {tool.name} successfully registered as {'hidden' if hidden else 'visible'} tool")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The requested tool instance if found, None otherwise.

        Note:
            This method only searches visible tools. For hidden tools,
            use get_hidden_tool().
        """
        tool = self.tools.get(name)
        if tool:
            logger.debug(f"Retrieved tool: {name}")
        else:
            logger.warning(f"Tool not found: {name}")
        return tool

    def get_hidden_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a hidden tool by name.

        Args:
            name: The name of the hidden tool to retrieve.

        Returns:
            The requested hidden tool instance if found, None otherwise.

        Note:
            This method only searches hidden tools. For visible tools,
            use get_tool().
        """
        tool = self.hidden_tools.get(name)
        if tool:
            logger.debug(f"Retrieved hidden tool: {name}")
        else:
            logger.warning(f"Hidden tool not found: {name}")
        return tool

    def get_tool_definitions(self) -> List[Tool]:
        """Get definitions for all registered non-hidden tools.

        Returns:
            A list of tool definitions for all visible tools in the registry.

        Note:
            Hidden tools are excluded from this list by design.
        """
        logger.debug(f"Retrieving tool definitions for {len(self.tools)} tools")
        definitions = [tool.get_definition() for tool in self.tools.values()]
        logger.debug("Retrieved definitions for tools: %s",
                     ", ".join(tool.name for tool in self.tools.values()))
        return definitions
