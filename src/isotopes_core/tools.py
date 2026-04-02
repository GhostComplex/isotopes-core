"""Tool framework for isotopes-core.

Supports two decorator styles:

1. @auto_tool - Automatically generates JSON schema from type hints and docstring
2. @tool - Manual schema specification
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from isotopes_core.types import TextContent, ImageContent


# =============================================================================
# Tool Result
# =============================================================================


@dataclass
class ToolResult:
    """Result of a tool execution."""

    content: list[TextContent | ImageContent] = field(default_factory=list)
    is_error: bool = False

    @classmethod
    def text(cls, text: str, is_error: bool = False) -> ToolResult:
        """Create a ToolResult with text content."""
        return cls(content=[TextContent(text=text)], is_error=is_error)

    @classmethod
    def error(cls, message: str) -> ToolResult:
        """Create an error ToolResult."""
        return cls(content=[TextContent(text=message)], is_error=True)


# =============================================================================
# Tool Definition
# =============================================================================

ToolUpdateCallback = Callable[[ToolResult], None]

ExecuteFn = Callable[
    [str, dict[str, Any], asyncio.Event | None, ToolUpdateCallback | None],
    Awaitable[ToolResult],
]


@dataclass
class Tool:
    """Tool definition with name, description, schema, and execute function."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execute: ExecuteFn


# =============================================================================
# Type to JSON Schema
# =============================================================================

def _python_type_to_json_schema(py_type: Any) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if py_type is str:
        return {"type": "string"}
    elif py_type is int:
        return {"type": "integer"}
    elif py_type is float:
        return {"type": "number"}
    elif py_type is bool:
        return {"type": "boolean"}
    elif origin is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}
    elif origin is dict:
        return {"type": "object"}
    else:
        return {"type": "string"}  # Fallback


def _parse_docstring_args(docstring: str) -> dict[str, str]:
    """Parse argument descriptions from docstring (Google/Numpy style)."""
    args_section = re.search(r"Args?:\s*\n((?:\s+.+\n?)+)", docstring, re.IGNORECASE)
    if not args_section:
        return {}

    descriptions = {}
    for match in re.finditer(r"^\s+(\w+):\s*(.+?)(?=\n\s+\w+:|\Z)", args_section.group(1), re.MULTILINE | re.DOTALL):
        name = match.group(1)
        desc = " ".join(match.group(2).split())
        descriptions[name] = desc

    return descriptions


# =============================================================================
# Decorators
# =============================================================================


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable[[ExecuteFn], Tool]:
    """Decorator to create a Tool with manual schema."""

    def decorator(fn: ExecuteFn) -> Tool:
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            execute=fn,
        )

    return decorator


def auto_tool(fn: Callable[..., Awaitable[str | ToolResult]]) -> Tool:
    """Decorator to create a Tool with auto-generated schema from type hints.
    
    The decorated function should have type-annotated parameters and return
    either a string or ToolResult. The docstring is used for the tool
    description, and Args section is used for parameter descriptions.
    
    Example:
        @auto_tool
        async def search(pattern: str, path: str = ".", max_results: int = 10) -> str:
            \"\"\"Search for a pattern in files.
            
            Args:
                pattern: Regex pattern to search for.
                path: Directory to search in.
                max_results: Maximum number of results.
            \"\"\"
            return "results..."
    """
    sig = inspect.signature(fn)
    hints = fn.__annotations__
    docstring = fn.__doc__ or ""
    arg_descriptions = _parse_docstring_args(docstring)

    # Extract first line of docstring as description
    description = docstring.split("\n")[0].strip() if docstring else fn.__name__

    # Build JSON Schema from type hints
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        py_type = hints.get(param_name, str)
        schema = _python_type_to_json_schema(py_type)

        # Add description from docstring
        if param_name in arg_descriptions:
            schema["description"] = arg_descriptions[param_name]

        # Add default value
        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = schema

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    # Wrap the function to match ExecuteFn signature
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        try:
            result = await fn(**params)
            if isinstance(result, ToolResult):
                return result
            return ToolResult.text(str(result))
        except Exception as e:
            return ToolResult.error(str(e))

    return Tool(
        name=fn.__name__,
        description=description,
        parameters=parameters,
        execute=execute,
    )
