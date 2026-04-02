"""Tests for isotopes-core tools."""

import pytest
from isotopes_core import auto_tool, ToolResult


@auto_tool
async def greet(name: str, enthusiasm: int = 1) -> str:
    """Greet someone by name.
    
    Args:
        name: The name to greet.
        enthusiasm: Number of exclamation marks.
    """
    return f"Hello, {name}{'!' * enthusiasm}"


@auto_tool
async def divide(a: float, b: float) -> str:
    """Divide two numbers.
    
    Args:
        a: The numerator.
        b: The denominator.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return str(a / b)


class TestAutoTool:
    def test_tool_name(self):
        assert greet.name == "greet"
        assert divide.name == "divide"

    def test_tool_description(self):
        assert greet.description == "Greet someone by name."
        assert divide.description == "Divide two numbers."

    def test_tool_parameters(self):
        params = greet.parameters
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert "enthusiasm" in params["properties"]
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["enthusiasm"]["type"] == "integer"
        assert params["properties"]["enthusiasm"]["default"] == 1
        assert "name" in params["required"]
        assert "enthusiasm" not in params["required"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        result = await greet.execute("call-1", {"name": "Alice", "enthusiasm": 3}, None, None)
        assert not result.is_error
        assert len(result.content) == 1
        assert result.content[0].text == "Hello, Alice!!!"

    @pytest.mark.asyncio
    async def test_execute_default_param(self):
        result = await greet.execute("call-2", {"name": "Bob"}, None, None)
        assert result.content[0].text == "Hello, Bob!"

    @pytest.mark.asyncio
    async def test_execute_error(self):
        result = await divide.execute("call-3", {"a": 1, "b": 0}, None, None)
        assert result.is_error
        assert "divide by zero" in result.content[0].text.lower()


class TestToolResult:
    def test_text(self):
        result = ToolResult.text("hello")
        assert not result.is_error
        assert result.content[0].text == "hello"

    def test_error(self):
        result = ToolResult.error("something went wrong")
        assert result.is_error
        assert result.content[0].text == "something went wrong"
