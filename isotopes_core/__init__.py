"""isotopes-core — A minimal, pluggable Python agent loop engine.

Public API re-exports for convenient single-import usage::

    from isotopes_core import agent_loop, Message, Tool, TextDelta, AgentDone
"""

from __future__ import annotations

# Events
from isotopes_core.events import (
    AgentDone,
    AgentEvent,
    Content,
    ImageContent,
    TextContent,
    TextDelta,
    ToolCall,
    ToolResult,
    Usage,
)

# Core loop
from isotopes_core.loop import LoopConfig, MaxTurnsExceeded, agent_loop

# Types
from isotopes_core.types import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    Message,
    MessageDelta,
    MessageStart,
    MessageStop,
    Provider,
    StreamEvent,
    Tool,
    ToolCallInfo,
    ToolResultInfo,
)

__version__ = "0.1.0"

__all__ = [
    # Loop
    "agent_loop",
    "LoopConfig",
    "MaxTurnsExceeded",
    # Events
    "AgentDone",
    "AgentEvent",
    "Content",
    "ImageContent",
    "TextContent",
    "TextDelta",
    "ToolCall",
    "ToolResult",
    "Usage",
    # Types
    "ContentBlockDelta",
    "ContentBlockStart",
    "ContentBlockStop",
    "Message",
    "MessageDelta",
    "MessageStart",
    "MessageStop",
    "Provider",
    "StreamEvent",
    "Tool",
    "ToolCallInfo",
    "ToolResultInfo",
]
