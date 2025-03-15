# src/tools/core/utils/tool_builder.py

import importlib
import logging
from typing import Dict, Type, Optional

from src.tools.core.base_tool import BaseTool

logger = logging.getLogger(__name__)


def create_tool_from_config(
        tool_def: dict,
        discovered_tools: Optional[Dict[str, Type[BaseTool]]] = None
) -> BaseTool:
    """
    Create and return an instance of a tool based on its configuration.

    Supports two modes:

    1. Custom tools: Directly instantiate by name from discovered tools
    2. Config-based tools: Create based on base_tool + configuration

    Parameters:
        tool_def (dict): A dictionary containing tool configuration. Expected keys:

            - "name": Unique identifier for this tool instance.
            - Optional: "base_tool": The key to determine the underlying tool type.
            - Optional: "class": Explicit class name to use.
            - Plus any other tool-specific settings.
        discovered_tools (Optional[Dict[str, Type[BaseTool]]]):
            A mapping from tool names to tool classes, typically provided by discover_custom_tools().

    Returns:
        BaseTool: An instantiated tool configured as specified.
    """
    tool_name = tool_def["name"]
    logger.debug(f"Creating tool instance for '{tool_name}' with config: {tool_def}")

    # Check if this is a custom tool that's already been discovered
    if discovered_tools is not None and tool_name in discovered_tools:
        tool_class = discovered_tools[tool_name]
        logger.debug(f"Found discovered tool class: {tool_class.__name__} for key '{tool_name}'")

    # Otherwise, look for base_tool for configurable tools
    elif "base_tool" in tool_def:
        base_tool_key = tool_def["base_tool"]

        if discovered_tools is not None and base_tool_key in discovered_tools:
            tool_class = discovered_tools[base_tool_key]
            logger.debug(f"Found discovered base tool class: {tool_class.__name__} for key '{base_tool_key}'")
        else:
            try:
                module = importlib.import_module(f"src.tools.implementations.{base_tool_key}")
                class_name = tool_def.get("class") or "".join(
                    part.capitalize() for part in base_tool_key.split("_")) + "Tool"
                tool_class = getattr(module, class_name)
                logger.debug(
                    f"Loaded tool class via dynamic import: {tool_class.__name__} from module src.tools.implementations.{base_tool_key}")
            except Exception as e:
                logger.error(f"Error importing tool '{base_tool_key}': {e}")
                raise
    else:
        raise ValueError(f"Tool '{tool_name}' is not a discovered custom tool and has no 'base_tool' defined")

    try:
        instance = tool_class(config=tool_def)
        logger.debug(f"Created instance of {tool_class.__name__} for tool '{tool_def.get('name')}'")
        return instance
    except Exception as e:
        logger.error(f"Error instantiating tool '{tool_def.get('name')}' with class {tool_class.__name__}: {e}")
        raise
