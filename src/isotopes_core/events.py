"""Event types for agent streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isotopes_core.types import Message, Usage, TextContent, ImageContent


# =============================================================================
# Stream Events
# =============================================================================


@dataclass
class TextDelta:
    """Incremental text from the assistant."""

    text: str


@dataclass
class ToolCallStart:
    """Tool call started."""

    id: str
    name: str


@dataclass
class ToolCallDelta:
    """Incremental tool call arguments (JSON string fragment)."""

    id: str
    arguments_delta: str


@dataclass
class ToolCallEnd:
    """Tool call finished parsing."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultEvent:
    """Tool execution completed."""

    id: str
    name: str
    content: list[TextContent | ImageContent]
    is_error: bool


@dataclass
class TurnEnd:
    """A turn (LLM call + tool executions) completed."""

    turn: int
    has_tool_calls: bool


@dataclass
class AgentDone:
    """Agent finished (no more tool calls)."""

    messages: list[Message]
    usage: Usage | None


@dataclass
class AgentError:
    """Agent encountered an error."""

    error: str
    messages: list[Message]


# Union type for all events
AgentEvent = (
    TextDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | ToolResultEvent
    | TurnEnd
    | AgentDone
    | AgentError
)
