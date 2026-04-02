"""LLM Provider base protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from isotopes_core.types import Message, Usage
from isotopes_core.tools import Tool


# =============================================================================
# Provider Stream Events
# =============================================================================


@dataclass
class ProviderTextDelta:
    """Text delta from provider."""
    text: str


@dataclass
class ProviderToolCallStart:
    """Tool call started."""
    id: str
    name: str


@dataclass
class ProviderToolCallDelta:
    """Tool call arguments delta."""
    id: str
    arguments_delta: str


@dataclass
class ProviderToolCallEnd:
    """Tool call finished."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderDone:
    """Provider finished streaming."""
    usage: Usage | None = None


ProviderEvent = (
    ProviderTextDelta
    | ProviderToolCallStart
    | ProviderToolCallDelta
    | ProviderToolCallEnd
    | ProviderDone
)


# =============================================================================
# Provider Protocol
# =============================================================================


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system_prompt: str = "",
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[ProviderEvent, None]:
        """Stream a completion from the provider.
        
        Args:
            messages: The conversation history.
            system_prompt: System prompt for the model.
            tools: Available tools.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Yields:
            Provider events (text deltas, tool calls, done).
        """
        ...
        yield  # Make this a generator (for type checking)
