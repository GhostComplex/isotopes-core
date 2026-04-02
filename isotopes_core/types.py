"""Core type definitions for isotopes-core.

This module defines the foundational types used throughout the library:

- ``Message`` — Conversation messages (user, assistant, tool)
- ``Tool`` — Tool definitions with JSON Schema parameters
- ``Provider`` — Abstract protocol for LLM providers
- ``StreamEvent`` — Events yielded by providers during streaming

All message types are frozen dataclasses for immutability.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from isotopes_core.events import Content, TextContent, ToolCall, Usage


# ---------------------------------------------------------------------------
# Stream events from provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContentBlockStart:
    """Provider started a new content block (text or tool_use)."""

    index: int
    block_type: str  # "text" | "tool_use"
    # For tool_use blocks, these will be set:
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class ContentBlockDelta:
    """Incremental delta within a content block."""

    index: int
    text: str | None = None        # For text blocks
    partial_json: str | None = None  # For tool_use blocks (argument JSON fragments)


@dataclass(frozen=True, slots=True)
class ContentBlockStop:
    """Provider finished a content block."""

    index: int


@dataclass(frozen=True, slots=True)
class MessageStart:
    """Provider started generating a response."""

    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class MessageDelta:
    """Provider-level message delta (e.g. stop reason update)."""

    stop_reason: str | None = None
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class MessageStop:
    """Provider finished generating the response."""

    pass


StreamEvent = (
    ContentBlockStart
    | ContentBlockDelta
    | ContentBlockStop
    | MessageStart
    | MessageDelta
    | MessageStop
)
"""Union of all events a Provider.complete() implementation may yield."""


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCallInfo:
    """A single tool invocation requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultInfo:
    """Result of a tool invocation, attached to a ``tool`` message."""

    id: str
    content: list[Content]
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """A single message in the conversation history.

    ``role`` is one of ``"user"``, ``"assistant"``, or ``"tool"``.

    For assistant messages, ``content`` holds the text parts and
    ``tool_calls`` holds any tool invocations the LLM requested.

    For tool messages, ``tool_results`` holds the results of prior
    tool calls.
    """

    role: str  # "user" | "assistant" | "tool"
    content: list[Content] = field(default_factory=list)
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    tool_results: list[ToolResultInfo] = field(default_factory=list)

    @staticmethod
    def user(text: str) -> Message:
        """Convenience constructor for a user message."""
        return Message(role="user", content=[TextContent(text=text)])

    @staticmethod
    def assistant(
        text: str = "",
        tool_calls: list[ToolCallInfo] | None = None,
    ) -> Message:
        """Convenience constructor for an assistant message."""
        content: list[Content] = [TextContent(text=text)] if text else []
        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls or [],
        )

    @staticmethod
    def tool(results: list[ToolResultInfo]) -> Message:
        """Convenience constructor for a tool-result message."""
        return Message(role="tool", tool_results=results)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


# The callable signature for a tool function.
# Accepts keyword arguments matching the JSON Schema properties.
# May be sync or async.
ToolFunction = Callable[..., Awaitable[str | list[Content]] | str | list[Content]]


@dataclass(frozen=True, slots=True)
class Tool:
    """Definition of a tool the agent can invoke.

    ``parameters`` follows JSON Schema (draft 2020-12) for the tool's
    input arguments. ``function`` is the actual callable to execute.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    function: ToolFunction


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """Abstract interface for LLM providers.

    Implementations must provide a ``complete`` method that accepts
    a message history and tool definitions, yielding ``StreamEvent``
    objects as the response is generated.
    """

    def complete(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a completion from the LLM.

        Yields ``StreamEvent`` objects as the response is generated.
        """
        ...  # pragma: no cover
