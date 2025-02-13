# src/tools/core/tool_registry.py

import inspect
import logging
import importlib
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Type, Optional, Callable

from src.data_models.tools import Tool
from src.tools.core.base_tool import BaseTool


class RegistrationSummary(BaseModel):
    """A Pydantic model that provides a summary of tool registration results.

    This class encapsulates information about the tool registration process, including
    statistics about scanned files, discovered tools, and any issues encountered during
    the registration process.

    Attributes:
        total_files_scanned (Optional[int]): The total number of Python files scanned
            during tool discovery.
        total_tools_found (Optional[int]): The total number of valid tools found across
            all scanned files.
        visible_tools (Optional[List[str]]): Names of tools that are publicly accessible
            in the registry.
        hidden_tools (Optional[List[str]]): Names of tools that are registered but hidden
            from public access.
        failed_imports (Optional[List[str]]): List of import failures encountered during
            tool discovery.
        initialization_failures (List[str]): List of tools that failed to initialize
            properly.
    """
    total_files_scanned: Optional[int] = Field(default="auto", description="Total number of files scanned.")
    total_tools_found: Optional[int] = Field(default="auto", description="Total number of tools found.")
    visible_tools: Optional[List[str]] = Field(default="auto", description="List of tools visible.")
    hidden_tools: Optional[List[str]] = Field(default="auto", description="List of tools hidden.")
    failed_imports: Optional[List[str]] = Field(default="auto", description="List of failed imports.")
    initialization_failures: List[str] = Field(default="auto", description="List of initialization failures.")

    def __str__(self) -> str:
        return f"""Tool Registration Summary:
├─ Files Scanned: {self.total_files_scanned}
├─ Total Tools: {self.total_tools_found}
│  ├─ Visible Tools ({len(self.visible_tools)}): {', '.join(self.visible_tools)}
│  └─ Hidden Tools ({len(self.hidden_tools)}): {', '.join(self.hidden_tools)}
└─ Issues:
   ├─ Import Failures ({len(self.failed_imports)}): {', '.join(self.failed_imports) if self.failed_imports else 'None'}
   └─ Init Failures ({len(self.initialization_failures)}): {', '.join(self.initialization_failures) if self.initialization_failures else 'None'}"""


class ToolRegistry:
    """A singleton registry for managing and accessing system tools with decorator support.

    This class implements the Singleton pattern and provides functionality for:
    - Discovering tools in a specified directory
    - Registering tools via a decorator pattern
    - Managing both visible and hidden tools
    - Initializing tools with configuration
    - Providing access to registered tools

    Attributes:
        tools (Dict[str, BaseTool]): Dictionary of publicly accessible tools
        hidden_tools (Dict[str, BaseTool]): Dictionary of hidden tools
        config (Dict): Configuration dictionary for tool initialization
        logger (logging.Logger): Logger instance for the registry
    """

    _instance = None
    _registered_tool_classes: Dict[str, tuple[Type[BaseTool], bool]] = {}

    def __new__(cls, *args, **kwargs) -> 'ToolRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict] = None) -> None:
        if not hasattr(self, 'initialized'):
            self.tools: Dict[str, BaseTool] = {}
            self.hidden_tools: Dict[str, BaseTool] = {}
            self.config = config or {}
            self.initialized = True
            self.logger = logging.getLogger(self.__class__.__name__)

            summary = self._discover_tools()
            if config:
                init_failures = self._initialize_registered_tools()
                summary.initialization_failures = init_failures

            self.logger.info(f" ---- IMPORTANT! ---- \n\n{summary}\n")

    def _discover_tools(self) -> RegistrationSummary:
        """Discover and import all tool modules to trigger decorators.

        This method scans the implementations directory for Python files containing tool
        definitions. It attempts to import each file and register any tool classes that
        are properly decorated and configured.

        Returns:
            RegistrationSummary: A summary object containing information about the
                discovery process, including counts of found tools and any errors
                encountered.
        """
        current_dir = Path(__file__).parent.parent
        implementations_path = current_dir / 'implementations'

        if not implementations_path.exists():
            return RegistrationSummary(initialization_failures=["Implementations directory not found"])

        python_files = [f for f in implementations_path.glob('*.py') if not f.name.startswith('_')]
        failed_imports = []
        visible_tools = []
        hidden_tools = []

        for file_path in python_files:
            module_name = f"src.tools.implementations.{file_path.stem}"
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseTool) and obj != BaseTool:
                        if hasattr(obj, 'name') and isinstance(obj.name, str):
                            tool_name = obj.name
                        else:
                            continue

                        if tool_name in self._registered_tool_classes:
                            _, is_hidden = self._registered_tool_classes[tool_name]
                            if is_hidden:
                                hidden_tools.append(tool_name)
                            else:
                                visible_tools.append(tool_name)

            except ImportError as e:
                failed_imports.append(f"{module_name}: {str(e)}")

        return RegistrationSummary(
            total_files_scanned=len(python_files),
            total_tools_found=len(visible_tools) + len(hidden_tools),
            visible_tools=visible_tools,
            hidden_tools=hidden_tools,
            failed_imports=failed_imports,
            initialization_failures=[]
        )

    def _initialize_registered_tools(self) -> List[str]:
        """Initialize all registered tool classes with their configurations.

        This method attempts to create instances of all registered tool classes using
        their corresponding configurations from the registry's config dictionary.

        Returns:
            List[str]: A list of error messages for any tools that failed to initialize
                properly. Each entry is formatted as "tool_name: error_message".
        """
        initialization_failures = []

        for tool_name, (tool_class, is_hidden) in self._registered_tool_classes.items():
            try:
                tool_config = self.config.get(tool_name)

                if tool_config is None:
                    self.logger.warning(
                        f"⚠️ No configuration found for tool '{tool_name}'. Proceeding with default settings.")

                tool_instance = tool_class(config=tool_config) if tool_config else tool_class()
                self.register_tool_instance(tool_instance, is_hidden)
            except Exception as e:
                initialization_failures.append(f"{tool_name}: {str(e)}")

        return initialization_failures

    @classmethod
    def register_tool(cls, hidden: bool = False) -> Callable:
        """Class method decorator for registering tool classes in the registry.

        This decorator registers a tool class with the registry. The decorated class
        must have a 'name' attribute defined as a string.

        Args:
            hidden (bool, optional): If True, the tool will be registered as hidden
                and only accessible via get_hidden_tool(). Defaults to False.

        Returns:
            Callable: A decorator function that registers the tool class.

        Raises:
            ValueError: If the decorated class doesn't have a valid 'name' attribute.
        """
        def decorator(tool_class: Type[BaseTool]) -> Type[BaseTool]:
            if not hasattr(tool_class, 'name') or not isinstance(tool_class.name, str):
                raise ValueError(f"Tool class '{tool_class.__name__}' must define a 'name' attribute as a string.")

            cls._registered_tool_classes[tool_class.name] = (tool_class, hidden)
            return tool_class

        return decorator

    def register_tool_instance(self, tool: BaseTool, hidden: bool = False) -> None:
        """Register a tool instance in the registry.

        Args:
            tool (BaseTool): The tool instance to register
            hidden (bool, optional): If True, registers the tool as hidden.
                Defaults to False.
        """
        target_dict = self.hidden_tools if hidden else self.tools
        target_dict[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name.

        Args:
            name (str): The name of the tool to retrieve

        Returns:
            Optional[BaseTool]: The requested tool instance if found, None otherwise
        """
        return self.tools.get(name)

    def get_hidden_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a hidden tool by name.

        Args:
            name (str): The name of the hidden tool to retrieve

        Returns:
            Optional[BaseTool]: The requested hidden tool instance if found,
                None otherwise
        """
        return self.hidden_tools.get(name)

    def get_tool_definitions(self) -> List[Tool]:
        """Get definitions for all registered non-hidden tools.

        Returns:
            List[Tool]: A list of tool definitions for all visible tools in the
                registry
        """
        return [tool.get_definition() for tool in self.tools.values()]
