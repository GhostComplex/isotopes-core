"""Event types yielded by the agent loop.

All events are frozen dataclasses for immutability. The agent_loop() async
generator yields these as it processes messages, calls tools, and completes.

Event flow for a typical multi-turn interaction:

    TextDelta* -> ToolCall+ -> ToolResult+ -> TextDelta* -> AgentDone
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Content types (used inside ToolResult and Message)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextContent:
    """Plain text content block."""

    text: str
    type: str = field(default="text", init=False)


@dataclass(frozen=True, slots=True)
class ImageContent:
    """Base64-encoded image content block."""

    data: str
    media_type: str  # e.g. "image/png", "image/jpeg"
    type: str = field(default="image", init=False)


Content = TextContent | ImageContent
"""Union of all content block types."""


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Incremental text chunk from the LLM response.

    Consumers should concatenate sequential TextDelta.text values to build
    the full assistant response.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The LLM has decided to invoke a tool.

    Yielded *after* the full tool call has been parsed from the stream
    (i.e. the arguments are complete JSON, not partial).
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of executing a single tool call.

    When ``is_error`` is True the content describes what went wrong.
    The loop will feed this back to the LLM so it can recover.
    """

    id: str
    content: list[Content]
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage statistics for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class AgentDone:
    """Terminal event: the agent loop has finished.

    ``messages`` contains the full conversation history including all
    assistant and tool messages produced during this run. ``usage``
    aggregates token counts across all LLM calls in the run.
    """

    messages: list[Any]  # list[Message] — Any to avoid circular import
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Union of all event types
# ---------------------------------------------------------------------------

AgentEvent = TextDelta | ToolCall | ToolResult | AgentDone
"""Any event that agent_loop() may yield."""
