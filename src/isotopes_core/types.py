"""Core type definitions for isotopes-core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# =============================================================================
# Content Types
# =============================================================================


@dataclass(frozen=True)
class TextContent:
    """Text content block."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ImageContent:
    """Image content block."""

    source: str  # base64 data or URL
    media_type: str = "image/png"
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class ToolCallContent:
    """Tool call content block."""

    id: str
    name: str
    arguments: dict
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True)
class ToolResultContent:
    """Tool result content block."""

    tool_call_id: str
    content: list[TextContent | ImageContent]
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


Content = TextContent | ImageContent | ToolCallContent | ToolResultContent


# =============================================================================
# Message Types
# =============================================================================


@dataclass
class UserMessage:
    """User message."""

    content: list[TextContent | ImageContent]
    role: Literal["user"] = "user"

    @classmethod
    def text(cls, text: str) -> UserMessage:
        """Create a user message with text content."""
        return cls(content=[TextContent(text=text)])


@dataclass
class AssistantMessage:
    """Assistant message."""

    content: list[TextContent | ToolCallContent]
    role: Literal["assistant"] = "assistant"

    @classmethod
    def text(cls, text: str) -> AssistantMessage:
        """Create an assistant message with text content."""
        return cls(content=[TextContent(text=text)])


@dataclass
class ToolResultMessage:
    """Tool result message."""

    content: list[ToolResultContent]
    role: Literal["tool_result"] = "tool_result"


Message = UserMessage | AssistantMessage | ToolResultMessage


# =============================================================================
# Usage
# =============================================================================


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )


# =============================================================================
# Context
# =============================================================================


@dataclass
class Context:
    """Agent context containing system prompt and message history."""

    system_prompt: str = ""
    messages: list[Message] = field(default_factory=list)
    max_tokens: int | None = None

    def add_message(self, message: Message) -> None:
        """Add a message to the context."""
        self.messages.append(message)

    def add_user_message(self, text: str) -> None:
        """Add a user message with text content."""
        self.add_message(UserMessage.text(text))

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message with text content."""
        self.add_message(AssistantMessage.text(text))

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
