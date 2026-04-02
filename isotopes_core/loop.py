"""Core agent loop implementation.

This module contains ``agent_loop()``, the heart of isotopes-core. It is an
async generator that orchestrates the LLM call → tool execution → feedback
cycle, yielding typed events at each stage.

Design goals:
    - **Async-first**: all I/O goes through ``await`` / ``async for``.
    - **Streaming**: events are yielded as they arrive, not batched.
    - **Parallel tools**: tool calls from a single LLM turn run concurrently.
    - **Cancellable**: respects ``asyncio.CancelledError`` at every await.
    - **Zero deps**: only depends on the stdlib and sibling modules.

Typical usage::

    async for event in agent_loop(
        provider=my_provider,
        messages=[Message.user("Hello!")],
        tools=[my_tool],
    ):
        match event:
            case TextDelta(text=t):
                print(t, end="")
            case AgentDone():
                break
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from isotopes_core.events import (
    AgentDone,
    AgentEvent,
    Content,
    TextContent,
    TextDelta,
    ToolCall,
    ToolResult,
    Usage,
)
from isotopes_core.types import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    Message,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamEvent,
    Tool,
    ToolCallInfo,
    ToolResultInfo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """Configuration knobs for ``agent_loop()``.

    Attributes:
        max_turns: Maximum number of LLM ↔ tool round-trips before the
            loop forces termination.  ``0`` means unlimited.
        parallel_tool_calls: When True (default), tool calls within a
            single LLM response are executed concurrently via
            ``asyncio.gather``.  Set to False for strictly sequential
            execution.
        max_tool_concurrency: Upper bound on how many tools run at once.
            ``0`` means no limit (all tool calls in a turn start together).
            Only meaningful when ``parallel_tool_calls`` is True.
    """

    max_turns: int = 0
    parallel_tool_calls: bool = True
    max_tool_concurrency: int = 0


# Default config singleton to avoid repeated allocation.
_DEFAULT_CONFIG = LoopConfig()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_tool_map(tools: list[Tool] | None) -> dict[str, Tool]:
    """Index tools by name for O(1) lookup during execution."""
    if not tools:
        return {}
    tool_map: dict[str, Tool] = {}
    for t in tools:
        if t.name in tool_map:
            raise ValueError(f"Duplicate tool name: {t.name!r}")
        tool_map[t.name] = t
    return tool_map


@dataclass(slots=True)
class _BlockAccumulator:
    """Mutable accumulator for a single content block being streamed.

    Tracks either a text block or a tool_use block as deltas arrive.
    """

    index: int
    block_type: str  # "text" | "tool_use"
    # text blocks
    text_parts: list[str] = field(default_factory=list)
    # tool_use blocks
    tool_call_id: str = ""
    tool_name: str = ""
    json_parts: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "".join(self.text_parts)

    @property
    def full_json(self) -> str:
        return "".join(self.json_parts)


class _ResponseCollector:
    """Collects ``StreamEvent``s from a single provider.complete() call.

    Tracks all content blocks, accumulates text and tool-call arguments,
    and exposes the final results once the stream completes.

    This class does NOT yield agent events — that is the caller's job.
    It only accumulates state and reports what was received.
    """

    def __init__(self) -> None:
        self.blocks: dict[int, _BlockAccumulator] = {}
        self.tool_calls: list[ToolCallInfo] = []
        self.text_parts: list[str] = []
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.stop_reason: str | None = None

    # ---- event handlers ---------------------------------------------------

    def on_message_start(self, event: MessageStart) -> None:
        if event.usage:
            self.input_tokens += event.usage.input_tokens
            self.output_tokens += event.usage.output_tokens

    def on_content_block_start(self, event: ContentBlockStart) -> None:
        acc = _BlockAccumulator(
            index=event.index,
            block_type=event.block_type,
            tool_call_id=event.tool_call_id or "",
            tool_name=event.tool_name or "",
        )
        self.blocks[event.index] = acc

    def on_content_block_delta(self, event: ContentBlockDelta) -> str | None:
        """Process a delta. Returns text if this is a text delta, else None."""
        acc = self.blocks.get(event.index)
        if acc is None:
            logger.warning("Delta for unknown block index %d", event.index)
            return None

        if acc.block_type == "text" and event.text is not None:
            acc.text_parts.append(event.text)
            self.text_parts.append(event.text)
            return event.text

        if acc.block_type == "tool_use" and event.partial_json is not None:
            acc.json_parts.append(event.partial_json)

        return None

    def on_content_block_stop(self, event: ContentBlockStop) -> ToolCallInfo | None:
        """Finalize a block. Returns a ToolCallInfo if this was a tool block."""
        acc = self.blocks.get(event.index)
        if acc is None:
            return None

        if acc.block_type == "tool_use":
            import json as _json

            raw_json = acc.full_json
            try:
                arguments = _json.loads(raw_json) if raw_json else {}
            except _json.JSONDecodeError:
                logger.error(
                    "Failed to parse tool arguments for %s: %s",
                    acc.tool_name,
                    raw_json,
                )
                arguments = {}

            info = ToolCallInfo(
                id=acc.tool_call_id,
                name=acc.tool_name,
                arguments=arguments,
            )
            self.tool_calls.append(info)
            return info

        return None

    def on_message_delta(self, event: MessageDelta) -> None:
        if event.stop_reason:
            self.stop_reason = event.stop_reason
        if event.usage:
            self.input_tokens += event.usage.input_tokens
            self.output_tokens += event.usage.output_tokens

    def on_message_stop(self, _event: MessageStop) -> None:
        pass  # Stream is done; nothing extra to track.

    # ---- final accessors --------------------------------------------------

    @property
    def full_text(self) -> str:
        return "".join(self.text_parts)

    @property
    def usage(self) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def build_assistant_message(self) -> Message:
        """Build the assistant Message from collected data."""
        return Message.assistant(
            text=self.full_text,
            tool_calls=list(self.tool_calls),
        )


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


async def _execute_tool(
    tool: Tool,
    call: ToolCallInfo,
) -> ToolResultInfo:
    """Execute a single tool call and wrap the result.

    If the tool function raises, the exception is caught and returned as
    an error result so the LLM can see what went wrong.
    """
    try:
        result = tool.function(**call.arguments)
        # Support both sync and async tool functions.
        if inspect.isawaitable(result):
            result = await result

        # Normalize return value to list[Content].
        match result:
            case str() as text:
                content: list[Content] = [TextContent(text=text)]
            case list() as items:
                content = items
            case _:
                content = [TextContent(text=str(result))]

        return ToolResultInfo(id=call.id, content=content)

    except Exception as exc:
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        error_text = f"{type(exc).__name__}: {exc}\n{''.join(tb)}"
        logger.error("Tool %s raised: %s", call.name, exc)
        return ToolResultInfo(
            id=call.id,
            content=[TextContent(text=error_text)],
            is_error=True,
        )


async def _execute_tools_parallel(
    tool_map: dict[str, Tool],
    calls: list[ToolCallInfo],
    max_concurrency: int = 0,
) -> list[ToolResultInfo]:
    """Execute multiple tool calls concurrently.

    If ``max_concurrency`` > 0, a semaphore limits parallelism.
    Unknown tools produce an error result instead of raising.
    """
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def _run(call: ToolCallInfo) -> ToolResultInfo:
        tool = tool_map.get(call.name)
        if tool is None:
            return ToolResultInfo(
                id=call.id,
                content=[TextContent(text=f"Unknown tool: {call.name!r}")],
                is_error=True,
            )
        if sem is not None:
            async with sem:
                return await _execute_tool(tool, call)
        return await _execute_tool(tool, call)

    return list(await asyncio.gather(*[_run(c) for c in calls]))


async def _execute_tools_sequential(
    tool_map: dict[str, Tool],
    calls: list[ToolCallInfo],
) -> list[ToolResultInfo]:
    """Execute tool calls one at a time in order."""
    results: list[ToolResultInfo] = []
    for call in calls:
        tool = tool_map.get(call.name)
        if tool is None:
            results.append(
                ToolResultInfo(
                    id=call.id,
                    content=[TextContent(text=f"Unknown tool: {call.name!r}")],
                    is_error=True,
                )
            )
        else:
            results.append(await _execute_tool(tool, call))
    return results


# ---------------------------------------------------------------------------
# Usage accumulator
# ---------------------------------------------------------------------------


class _UsageAccumulator:
    """Sums usage across multiple LLM calls."""

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def add(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    @property
    def total(self) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


# ---------------------------------------------------------------------------
# Public API: agent_loop
# ---------------------------------------------------------------------------


class MaxTurnsExceeded(Exception):
    """Raised internally when the turn limit is hit."""

    def __init__(self, turns: int) -> None:
        super().__init__(f"Agent loop exceeded max_turns={turns}")
        self.turns = turns


async def _stream_llm_response(
    provider: Any,  # Provider protocol
    messages: list[Message],
    *,
    system_prompt: str | None,
    tools: list[Tool] | None,
    temperature: float | None,
    max_tokens: int | None,
) -> AsyncGenerator[tuple[StreamEvent, _ResponseCollector], None]:
    """Wrap provider.complete() and pair each event with a collector.

    Yields ``(raw_event, collector)`` so the caller can both forward
    agent-level events and let the collector accumulate state.
    """
    collector = _ResponseCollector()

    stream = provider.complete(
        messages,
        system_prompt=system_prompt,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    async for event in stream:
        yield event, collector


async def agent_loop(
    *,
    provider: Any,  # implements Provider protocol
    messages: list[Message],
    system_prompt: str | None = None,
    tools: list[Tool] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    config: LoopConfig | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Core agent loop — an async generator yielding ``AgentEvent``s.

    This is the primary entry-point for isotopes-core. It implements the
    plan → act → observe cycle:

    1. Send current ``messages`` + ``tools`` to the LLM via ``provider``.
    2. Stream the response, yielding ``TextDelta`` and ``ToolCall`` events.
    3. If the LLM requested tool calls:
       a. Execute them (in parallel by default).
       b. Yield ``ToolResult`` events.
       c. Append assistant + tool messages to history.
       d. Increment turn counter; if ``max_turns`` reached, stop.
       e. Go to step 1.
    4. If no tool calls, yield ``AgentDone`` and return.

    Parameters:
        provider: LLM provider implementing the ``Provider`` protocol.
        messages: Initial conversation history. Must contain at least one
            user message. The list is mutated in-place — new assistant
            and tool messages are appended as the loop progresses.
        system_prompt: Optional system-level instruction prepended to
            every LLM call.
        tools: Available tools the LLM may invoke. ``None`` or ``[]``
            disables tool use.
        temperature: Sampling temperature override.
        max_tokens: Max tokens per LLM response.
        config: Loop configuration (turn limits, parallelism, etc.).
            Defaults to ``LoopConfig()`` if not provided.

    Yields:
        ``AgentEvent`` — one of ``TextDelta``, ``ToolCall``,
        ``ToolResult``, or ``AgentDone``.

    Raises:
        ValueError: If ``messages`` is empty or ``tools`` contains
            duplicate names.
        asyncio.CancelledError: Propagated if the task is cancelled.
    """
    if not messages:
        raise ValueError("messages must not be empty")

    cfg = config or _DEFAULT_CONFIG
    tool_map = _build_tool_map(tools)
    usage_acc = _UsageAccumulator()
    turn = 0

    while True:
        # ----- Check turn limit before calling LLM ---------------------
        if cfg.max_turns > 0 and turn >= cfg.max_turns:
            logger.info(
                "Max turns (%d) reached; stopping agent loop.", cfg.max_turns
            )
            yield AgentDone(messages=list(messages), usage=usage_acc.total)
            return

        # ----- Stream LLM response ------------------------------------
        collector = _ResponseCollector()

        stream = provider.complete(
            messages,
            system_prompt=system_prompt,
            tools=tools if tool_map else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async for event in stream:
            match event:
                case MessageStart():
                    collector.on_message_start(event)

                case ContentBlockStart():
                    collector.on_content_block_start(event)

                case ContentBlockDelta():
                    text = collector.on_content_block_delta(event)
                    if text is not None:
                        yield TextDelta(text=text)

                case ContentBlockStop():
                    info = collector.on_content_block_stop(event)
                    if info is not None:
                        yield ToolCall(
                            id=info.id,
                            name=info.name,
                            arguments=info.arguments,
                        )

                case MessageDelta():
                    collector.on_message_delta(event)

                case MessageStop():
                    collector.on_message_stop(event)

        # Accumulate usage from this call.
        usage_acc.add(collector.usage)

        # Build and append the assistant message.
        assistant_msg = collector.build_assistant_message()
        messages.append(assistant_msg)

        # ----- No tool calls → we're done! ----------------------------
        if not collector.has_tool_calls:
            yield AgentDone(messages=list(messages), usage=usage_acc.total)
            return

        # ----- Execute tool calls -------------------------------------
        if cfg.parallel_tool_calls:
            results = await _execute_tools_parallel(
                tool_map,
                collector.tool_calls,
                max_concurrency=cfg.max_tool_concurrency,
            )
        else:
            results = await _execute_tools_sequential(
                tool_map,
                collector.tool_calls,
            )

        # Yield ToolResult events.
        for r in results:
            yield ToolResult(
                id=r.id,
                content=r.content,
                is_error=r.is_error,
            )

        # Append tool results as a message.
        tool_msg = Message.tool(results)
        messages.append(tool_msg)

        # ----- Advance turn counter -----------------------------------
        turn += 1
        logger.debug("Completed turn %d", turn)
