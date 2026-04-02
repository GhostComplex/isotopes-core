"""isotopes-core: Minimal, pluggable Python agent loop engine."""

from isotopes_core.agent import Agent
from isotopes_core.tools import Tool, ToolResult, auto_tool, tool
from isotopes_core.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    TextContent,
    ToolCallContent,
    ToolResultContent,
    Usage,
    UserMessage,
)
from isotopes_core.events import (
    AgentEvent,
    TextDelta,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolResultEvent,
    TurnEnd,
    AgentDone,
    AgentError,
)

__version__ = "0.1.0"

__all__ = [
    # Agent
    "Agent",
    # Tools
    "Tool",
    "ToolResult",
    "auto_tool",
    "tool",
    # Types
    "AssistantMessage",
    "Context",
    "ImageContent",
    "Message",
    "TextContent",
    "ToolCallContent",
    "ToolResultContent",
    "Usage",
    "UserMessage",
    # Events
    "AgentEvent",
    "TextDelta",
    "ToolCallStart",
    "ToolCallDelta",
    "ToolCallEnd",
    "ToolResultEvent",
    "TurnEnd",
    "AgentDone",
    "AgentError",
]
