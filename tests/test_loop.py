"""Unit tests for isotopes_core.loop.agent_loop().

Uses a mock provider that yields pre-programmed StreamEvent sequences,
allowing full deterministic testing of the agent loop without any LLM.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from isotopes_core import (
    AgentDone,
    Content,
    LoopConfig,
    Message,
    TextContent,
    TextDelta,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
    agent_loop,
)
from isotopes_core.types import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamEvent,
    ToolCallInfo,
)


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockProvider:
    """A test provider that replays pre-programmed StreamEvent sequences.

    Each call to ``complete()`` pops the next sequence from the queue.
    If ``call_log`` is provided it records each call's arguments for
    assertions.
    """

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.call_log: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.call_log.append(
            {
                "messages": list(messages),
                "system_prompt": system_prompt,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise RuntimeError("MockProvider has no more responses queued")
        events = self._responses.pop(0)
        self.call_count += 1
        for event in events:
            yield event


# ---------------------------------------------------------------------------
# Helpers to build stream event sequences
# ---------------------------------------------------------------------------


def make_text_response(
    text: str,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[StreamEvent]:
    """Build a stream event sequence for a plain text response."""
    return [
        MessageStart(usage=Usage(input_tokens=input_tokens, output_tokens=0)),
        ContentBlockStart(index=0, block_type="text"),
        ContentBlockDelta(index=0, text=text),
        ContentBlockStop(index=0),
        MessageDelta(
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=output_tokens),
        ),
        MessageStop(),
    ]


def make_text_response_chunked(
    chunks: list[str],
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[StreamEvent]:
    """Build a stream event sequence with multiple text deltas."""
    events: list[StreamEvent] = [
        MessageStart(usage=Usage(input_tokens=input_tokens, output_tokens=0)),
        ContentBlockStart(index=0, block_type="text"),
    ]
    for chunk in chunks:
        events.append(ContentBlockDelta(index=0, text=chunk))
    events += [
        ContentBlockStop(index=0),
        MessageDelta(
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=output_tokens),
        ),
        MessageStop(),
    ]
    return events


def make_tool_call_response(
    tool_calls: list[tuple[str, str, dict[str, Any]]],
    text: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[StreamEvent]:
    """Build a stream event sequence with tool calls.

    ``tool_calls`` is a list of (id, name, arguments) tuples.
    """
    events: list[StreamEvent] = [
        MessageStart(usage=Usage(input_tokens=input_tokens, output_tokens=0)),
    ]
    idx = 0
    if text:
        events += [
            ContentBlockStart(index=idx, block_type="text"),
            ContentBlockDelta(index=idx, text=text),
            ContentBlockStop(index=idx),
        ]
        idx += 1

    for call_id, name, arguments in tool_calls:
        events += [
            ContentBlockStart(
                index=idx,
                block_type="tool_use",
                tool_call_id=call_id,
                tool_name=name,
            ),
            ContentBlockDelta(
                index=idx,
                partial_json=json.dumps(arguments),
            ),
            ContentBlockStop(index=idx),
        ]
        idx += 1

    events += [
        MessageDelta(
            stop_reason="tool_use",
            usage=Usage(input_tokens=0, output_tokens=output_tokens),
        ),
        MessageStop(),
    ]
    return events


# ---------------------------------------------------------------------------
# Simple tool fixtures
# ---------------------------------------------------------------------------


async def _add_tool_fn(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


ADD_TOOL = Tool(
    name="add",
    description="Add two numbers",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
    function=_add_tool_fn,
)


def _sync_greet_fn(name: str) -> str:
    """A synchronous tool function."""
    return f"Hello, {name}!"


SYNC_GREET_TOOL = Tool(
    name="greet",
    description="Greet someone",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    function=_sync_greet_fn,
)


async def _failing_tool_fn(**kwargs: Any) -> str:
    raise RuntimeError("Tool execution failed!")


FAILING_TOOL = Tool(
    name="fail",
    description="A tool that always fails",
    parameters={"type": "object", "properties": {}},
    function=_failing_tool_fn,
)


async def _slow_tool_fn(delay: float = 0.05) -> str:
    await asyncio.sleep(delay)
    return f"slept {delay}s"


SLOW_TOOL = Tool(
    name="slow",
    description="A slow tool for concurrency testing",
    parameters={
        "type": "object",
        "properties": {"delay": {"type": "number"}},
    },
    function=_slow_tool_fn,
)


# ---------------------------------------------------------------------------
# Helper to collect all events
# ---------------------------------------------------------------------------


async def collect_events(gen: AsyncGenerator[Any, None]) -> list[Any]:
    """Drain an async generator into a list."""
    events: list[Any] = []
    async for e in gen:
        events.append(e)
    return events


# ===========================================================================
# Tests
# ===========================================================================


class TestSingleTurnNoTools:
    """Test: single LLM call with no tool invocations."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        """LLM returns plain text → TextDelta + AgentDone."""
        provider = MockProvider([make_text_response("Hello, world!")])
        messages = [Message.user("Hi")]

        events = await collect_events(
            agent_loop(provider=provider, messages=messages)
        )

        # Should get a TextDelta and an AgentDone
        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        done_events = [e for e in events if isinstance(e, AgentDone)]

        assert len(text_deltas) == 1
        assert text_deltas[0].text == "Hello, world!"

        assert len(done_events) == 1
        assert done_events[0].usage is not None
        assert done_events[0].usage.input_tokens == 10
        assert done_events[0].usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_chunked_text_response(self) -> None:
        """LLM streams text in multiple deltas."""
        chunks = ["Hello", ", ", "world", "!"]
        provider = MockProvider([make_text_response_chunked(chunks)])
        messages = [Message.user("Hi")]

        events = await collect_events(
            agent_loop(provider=provider, messages=messages)
        )

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert [d.text for d in text_deltas] == chunks

    @pytest.mark.asyncio
    async def test_messages_appended(self) -> None:
        """After the loop, messages list includes the assistant reply."""
        provider = MockProvider([make_text_response("Reply")])
        messages = [Message.user("Hi")]

        await collect_events(agent_loop(provider=provider, messages=messages))

        assert len(messages) == 2
        assert messages[1].role == "assistant"
        assert messages[1].content[0].text == "Reply"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_done_messages_snapshot(self) -> None:
        """AgentDone.messages is a snapshot copy, not a reference."""
        provider = MockProvider([make_text_response("Reply")])
        messages = [Message.user("Hi")]

        events = await collect_events(
            agent_loop(provider=provider, messages=messages)
        )
        done = [e for e in events if isinstance(e, AgentDone)][0]

        # Mutating the original should not affect the snapshot
        messages.append(Message.user("Extra"))
        assert len(done.messages) == 2


class TestMultiTurnWithTools:
    """Test: LLM invokes tools, loop feeds results back."""

    @pytest.mark.asyncio
    async def test_single_tool_call(self) -> None:
        """LLM calls one tool, then produces a final text response."""
        provider = MockProvider(
            [
                # Turn 1: LLM calls add(a=2, b=3)
                make_tool_call_response(
                    [("call_1", "add", {"a": 2, "b": 3})],
                ),
                # Turn 2: LLM returns final text
                make_text_response("The result is 5"),
            ]
        )
        messages = [Message.user("What is 2+3?")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
            )
        )

        # Check event types
        tool_calls = [e for e in events if isinstance(e, ToolCall)]
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        done_events = [e for e in events if isinstance(e, AgentDone)]

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "add"
        assert tool_calls[0].arguments == {"a": 2, "b": 3}

        assert len(tool_results) == 1
        assert not tool_results[0].is_error
        assert tool_results[0].content[0].text == "5"  # type: ignore[union-attr]

        assert len(done_events) == 1

        # Provider was called twice (tool turn + final turn)
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_turns(self) -> None:
        """LLM calls tools across multiple turns before producing text."""
        provider = MockProvider(
            [
                make_tool_call_response([("c1", "add", {"a": 1, "b": 2})]),
                make_tool_call_response([("c2", "add", {"a": 3, "b": 4})]),
                make_text_response("Done! 3 then 7."),
            ]
        )
        messages = [Message.user("Add 1+2, then 3+4")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
            )
        )

        tool_calls = [e for e in events if isinstance(e, ToolCall)]
        assert len(tool_calls) == 2
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_call_with_text(self) -> None:
        """LLM produces text AND a tool call in the same response."""
        provider = MockProvider(
            [
                make_tool_call_response(
                    [("c1", "add", {"a": 5, "b": 5})],
                    text="Let me calculate that...",
                ),
                make_text_response("It's 10!"),
            ]
        )
        messages = [Message.user("5+5?")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
            )
        )

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        # Should have text from both turns
        assert any("calculate" in d.text for d in text_deltas)
        assert any("10" in d.text for d in text_deltas)

    @pytest.mark.asyncio
    async def test_sync_tool_function(self) -> None:
        """Synchronous tool functions work too."""
        provider = MockProvider(
            [
                make_tool_call_response(
                    [("c1", "greet", {"name": "Alice"})],
                ),
                make_text_response("Done"),
            ]
        )
        messages = [Message.user("Greet Alice")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[SYNC_GREET_TOOL],
            )
        )

        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 1
        assert results[0].content[0].text == "Hello, Alice!"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_messages_history_grows(self) -> None:
        """Each tool turn appends assistant + tool messages."""
        provider = MockProvider(
            [
                make_tool_call_response([("c1", "add", {"a": 1, "b": 1})]),
                make_text_response("2"),
            ]
        )
        messages = [Message.user("1+1")]

        await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
            )
        )

        # user, assistant (tool call), tool result, assistant (final)
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "tool"
        assert messages[3].role == "assistant"


class TestParallelToolExecution:
    """Test: multiple tool calls in one turn run concurrently."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self) -> None:
        """Two tool calls in one response execute in parallel."""
        provider = MockProvider(
            [
                make_tool_call_response(
                    [
                        ("c1", "add", {"a": 1, "b": 2}),
                        ("c2", "add", {"a": 3, "b": 4}),
                    ]
                ),
                make_text_response("3 and 7"),
            ]
        )
        messages = [Message.user("Compute both")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
            )
        )

        tool_calls = [e for e in events if isinstance(e, ToolCall)]
        tool_results = [e for e in events if isinstance(e, ToolResult)]

        assert len(tool_calls) == 2
        assert len(tool_results) == 2

        # Both results present (order matches call order via gather)
        result_texts = [r.content[0].text for r in tool_results]  # type: ignore[union-attr]
        assert "3" in result_texts
        assert "7" in result_texts

    @pytest.mark.asyncio
    async def test_parallel_is_actually_concurrent(self) -> None:
        """Verify parallel execution is faster than sequential."""
        delay = 0.1
        provider = MockProvider(
            [
                make_tool_call_response(
                    [
                        ("c1", "slow", {"delay": delay}),
                        ("c2", "slow", {"delay": delay}),
                    ]
                ),
                make_text_response("done"),
            ]
        )
        messages = [Message.user("Go")]

        import time

        start = time.monotonic()
        await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[SLOW_TOOL],
                config=LoopConfig(parallel_tool_calls=True),
            )
        )
        elapsed = time.monotonic() - start

        # If parallel, should take ~delay, not ~2*delay
        assert elapsed < delay * 1.8, f"Took {elapsed:.3f}s, expected < {delay * 1.8}"

    @pytest.mark.asyncio
    async def test_sequential_execution(self) -> None:
        """With parallel_tool_calls=False, tools run sequentially."""
        provider = MockProvider(
            [
                make_tool_call_response(
                    [
                        ("c1", "add", {"a": 1, "b": 2}),
                        ("c2", "add", {"a": 3, "b": 4}),
                    ]
                ),
                make_text_response("3 and 7"),
            ]
        )
        messages = [Message.user("Compute both")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
                config=LoopConfig(parallel_tool_calls=False),
            )
        )

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 2


class TestMaxTurnsEnforcement:
    """Test: max_turns limits how many tool-use rounds happen."""

    @pytest.mark.asyncio
    async def test_stops_at_max_turns(self) -> None:
        """Loop stops after max_turns even if LLM keeps requesting tools."""
        # Provider always requests a tool call (never produces final text)
        provider = MockProvider(
            [
                make_tool_call_response([("c1", "add", {"a": 1, "b": 1})]),
                make_tool_call_response([("c2", "add", {"a": 2, "b": 2})]),
                make_tool_call_response([("c3", "add", {"a": 3, "b": 3})]),
                make_text_response("never reached"),
            ]
        )
        messages = [Message.user("Keep going")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
                config=LoopConfig(max_turns=2),
            )
        )

        # Should have executed exactly 2 tool turns
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 2

        # Should end with AgentDone
        done_events = [e for e in events if isinstance(e, AgentDone)]
        assert len(done_events) == 1

        # Provider should have been called 3 times:
        # initial + after turn 1 result + stop before turn 3 LLM call
        # Actually: call 1 (tools) -> execute -> call 2 (tools) -> execute -> turn=2 >= max_turns -> stop
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_max_turns_zero_means_unlimited(self) -> None:
        """max_turns=0 means no limit."""
        provider = MockProvider(
            [
                make_tool_call_response([("c1", "add", {"a": 1, "b": 1})]),
                make_tool_call_response([("c2", "add", {"a": 2, "b": 2})]),
                make_tool_call_response([("c3", "add", {"a": 3, "b": 3})]),
                make_text_response("finally done"),
            ]
        )
        messages = [Message.user("Keep going")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
                config=LoopConfig(max_turns=0),
            )
        )

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 3
        assert provider.call_count == 4

    @pytest.mark.asyncio
    async def test_max_turns_one(self) -> None:
        """max_turns=1 allows exactly one tool round-trip."""
        provider = MockProvider(
            [
                make_tool_call_response([("c1", "add", {"a": 1, "b": 1})]),
                make_tool_call_response([("c2", "add", {"a": 2, "b": 2})]),
            ]
        )
        messages = [Message.user("Go")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
                config=LoopConfig(max_turns=1),
            )
        )

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1


class TestToolErrorHandling:
    """Test: tool errors are caught and reported to the LLM."""

    @pytest.mark.asyncio
    async def test_tool_exception_becomes_error_result(self) -> None:
        """When a tool raises, the error is wrapped in ToolResult(is_error=True)."""
        provider = MockProvider(
            [
                make_tool_call_response([("c1", "fail", {})]),
                make_text_response("I see the error"),
            ]
        )
        messages = [Message.user("Do the thing")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[FAILING_TOOL],
            )
        )

        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 1
        assert results[0].is_error is True
        assert "RuntimeError" in results[0].content[0].text  # type: ignore[union-attr]

        # Loop should continue — LLM sees the error and responds
        done_events = [e for e in events if isinstance(e, AgentDone)]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self) -> None:
        """Calling a tool that doesn't exist produces an error result."""
        provider = MockProvider(
            [
                make_tool_call_response([("c1", "nonexistent", {"x": 1})]),
                make_text_response("Ok, that tool doesn't exist"),
            ]
        )
        messages = [Message.user("Use nonexistent")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
            )
        )

        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 1
        assert results[0].is_error is True
        assert "Unknown tool" in results[0].content[0].text  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_mixed_success_and_error(self) -> None:
        """One tool succeeds and another fails in the same turn."""
        provider = MockProvider(
            [
                make_tool_call_response(
                    [
                        ("c1", "add", {"a": 1, "b": 2}),
                        ("c2", "fail", {}),
                    ]
                ),
                make_text_response("One worked, one didn't"),
            ]
        )
        messages = [Message.user("Both please")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL, FAILING_TOOL],
            )
        )

        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 2

        success = [r for r in results if not r.is_error]
        errors = [r for r in results if r.is_error]
        assert len(success) == 1
        assert len(errors) == 1


class TestEdgeCases:
    """Test: edge cases and validation."""

    @pytest.mark.asyncio
    async def test_empty_messages_raises(self) -> None:
        """Passing empty messages list raises ValueError."""
        provider = MockProvider([])
        with pytest.raises(ValueError, match="messages must not be empty"):
            await collect_events(
                agent_loop(provider=provider, messages=[])
            )

    @pytest.mark.asyncio
    async def test_duplicate_tool_names_raises(self) -> None:
        """Two tools with the same name raises ValueError."""
        provider = MockProvider([make_text_response("Hi")])
        dup_tool = Tool(
            name="add",
            description="Duplicate",
            parameters={"type": "object", "properties": {}},
            function=_add_tool_fn,
        )
        with pytest.raises(ValueError, match="Duplicate tool name"):
            await collect_events(
                agent_loop(
                    provider=provider,
                    messages=[Message.user("Hi")],
                    tools=[ADD_TOOL, dup_tool],
                )
            )

    @pytest.mark.asyncio
    async def test_no_tools_provided(self) -> None:
        """When no tools are provided, loop works for text-only responses."""
        provider = MockProvider([make_text_response("Just text")])
        messages = [Message.user("Hi")]

        events = await collect_events(
            agent_loop(provider=provider, messages=messages)
        )

        done_events = [e for e in events if isinstance(e, AgentDone)]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_usage_accumulates_across_turns(self) -> None:
        """Usage sums across multiple LLM calls."""
        provider = MockProvider(
            [
                make_tool_call_response(
                    [("c1", "add", {"a": 1, "b": 1})],
                    input_tokens=100,
                    output_tokens=50,
                ),
                make_text_response("2", input_tokens=200, output_tokens=30),
            ]
        )
        messages = [Message.user("1+1")]

        events = await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                tools=[ADD_TOOL],
            )
        )

        done = [e for e in events if isinstance(e, AgentDone)][0]
        assert done.usage is not None
        assert done.usage.input_tokens == 300
        assert done.usage.output_tokens == 80

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_provider(self) -> None:
        """system_prompt is forwarded to the provider."""
        provider = MockProvider([make_text_response("Ok")])
        messages = [Message.user("Hi")]

        await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                system_prompt="You are a pirate.",
            )
        )

        assert provider.call_log[0]["system_prompt"] == "You are a pirate."

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_forwarded(self) -> None:
        """temperature and max_tokens are passed through to provider."""
        provider = MockProvider([make_text_response("Ok")])
        messages = [Message.user("Hi")]

        await collect_events(
            agent_loop(
                provider=provider,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
        )

        assert provider.call_log[0]["temperature"] == 0.7
        assert provider.call_log[0]["max_tokens"] == 1024


class TestCancellation:
    """Test: asyncio cancellation support."""

    @pytest.mark.asyncio
    async def test_cancellation_during_tool_execution(self) -> None:
        """Loop respects asyncio cancellation during tool execution."""

        async def slow_tool(**kwargs: Any) -> str:
            await asyncio.sleep(10)
            return "done"

        slow = Tool(
            name="slow",
            description="Slow",
            parameters={"type": "object", "properties": {}},
            function=slow_tool,
        )

        provider = MockProvider(
            [
                make_tool_call_response([("c1", "slow", {})]),
                make_text_response("never reached"),
            ]
        )
        messages = [Message.user("Go")]

        async def run_loop() -> list[Any]:
            return await collect_events(
                agent_loop(
                    provider=provider,
                    messages=messages,
                    tools=[slow],
                )
            )

        task = asyncio.create_task(run_loop())
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
