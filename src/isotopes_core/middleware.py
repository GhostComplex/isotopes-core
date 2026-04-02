"""Middleware and lifecycle hooks for isotopes-core."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from isotopes_core.types import Context, Message
from isotopes_core.tools import Tool, ToolResult


# =============================================================================
# Hook Types
# =============================================================================

# Called when agent starts
OnAgentStartHook = Callable[[Context], Awaitable[None] | None]

# Called when agent ends (success or error)
OnAgentEndHook = Callable[[Context, list[Message]], Awaitable[None] | None]

# Called at the start of each turn (before LLM call)
OnTurnStartHook = Callable[[Context, int], Awaitable[None] | None]

# Called at the end of each turn (after tool executions)
OnTurnEndHook = Callable[[Context, int], Awaitable[None] | None]

# Called before executing a tool
BeforeToolCallHook = Callable[[Tool, dict[str, Any]], Awaitable[dict[str, Any] | None] | dict[str, Any] | None]

# Called after executing a tool
AfterToolCallHook = Callable[[Tool, ToolResult], Awaitable[ToolResult | None] | ToolResult | None]

# Called to transform context before LLM call
TransformContextHook = Callable[[Context], Awaitable[Context] | Context]

# Called on error
OnErrorHook = Callable[[Exception], Awaitable[None] | None]


# =============================================================================
# Lifecycle Hooks Container
# =============================================================================


@dataclass
class LifecycleHooks:
    """Container for all lifecycle hooks."""

    on_agent_start: OnAgentStartHook | None = None
    on_agent_end: OnAgentEndHook | None = None
    on_turn_start: OnTurnStartHook | None = None
    on_turn_end: OnTurnEndHook | None = None
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    transform_context: TransformContextHook | None = None
    on_error: OnErrorHook | None = None
