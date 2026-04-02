# 🫥 isotopes-core PRD

> Version: 0.1.0
> Date: 2026-04-02
> Status: **Draft**

## Overview

**isotopes-core** is a standalone Python library providing a pluggable agent loop engine. It is the core of the Isotopes framework, usable independently or embedded in other applications.

## Goals

1. **Minimal core** — Only essential agent loop components, no I/O, no CLI, no persistence
2. **Pluggable** — Configure provider, tools, middleware via dependency injection
3. **Zero runtime deps** — No mandatory dependencies beyond LLM SDKs
4. **Type-safe** — Full type hints, mypy/pyright compatible

## Non-Goals

- ❌ CLI / TUI
- ❌ Session persistence
- ❌ Concrete tools (bash, read, write, etc.)
- ❌ RPC protocol
- ❌ Skill system
- ❌ MCP integration

These belong in the upper-level `isotopes` package.

---

## Architecture

```
isotopes-core/
├── agent.py       # Agent class: state management, prompt(), continue_()
├── loop.py        # agent_loop(): core loop (plan → act → observe)
├── context.py     # Context management, pruning strategies
├── tools.py       # Tool definition, @auto_tool decorator
├── events.py      # Event types (AgentEvent stream)
├── middleware.py  # Lifecycle hooks
├── types.py       # Core type definitions
└── providers/     # LLM provider abstraction
    ├── base.py    # Provider interface
    ├── anthropic.py
    ├── openai.py
    ├── proxy.py   # Generic OpenAI-compatible proxy
    └── router.py  # Multi-provider routing
```

## Core Components

### 1. Agent Class

```python
from isotopes_core import Agent, auto_tool

# Create agent
agent = Agent(
    provider=AnthropicProvider(api_key="..."),
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],
    max_turns=10,
)

# Send message and stream events
async for event in agent.prompt("Hello!"):
    match event:
        case TextDelta(text=t):
            print(t, end="", flush=True)
        case ToolCall(name=n, arguments=a):
            print(f"Calling {n}({a})")
        case AgentDone():
            print("Done!")
```

### 2. Agent Loop

Core loop logic (in `loop.py`):

```
1. Receive user message
2. Build context (system prompt + history + tools)
3. Call LLM
4. If tool calls:
   a. Execute tools (parallel or sequential)
   b. Add results to history
   c. Go to step 3
5. If no tool calls: return final response
```

### 3. Tool Definition

Two styles supported:

```python
# Style 1: @auto_tool (recommended, auto-generates schema from type hints)
@auto_tool
async def search(pattern: str, path: str = ".", max_results: int = 10) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in.
        max_results: Maximum number of results.
    """
    ...

# Style 2: Manual schema
@tool(
    name="search",
    description="Search for a pattern",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
        },
        "required": ["pattern"],
    }
)
async def search(tool_call_id, params, signal, on_update):
    ...
```

### 4. Provider Abstraction

```python
class Provider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        ...
```

Built-in providers:
- `AnthropicProvider` — Anthropic Claude API
- `OpenAIProvider` — OpenAI API
- `ProxyProvider` — Any OpenAI-compatible endpoint (ollama, vllm)
- `RouterProvider` — Route to different providers by model name

### 5. Events

```python
@dataclass
class TextDelta:
    text: str

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolResult:
    id: str
    content: list[TextContent | ImageContent]
    is_error: bool

@dataclass
class AgentDone:
    messages: list[Message]  # Full history
    usage: Usage | None
```

### 6. Middleware / Hooks

```python
agent = Agent(
    ...,
    on_agent_start=lambda ctx: print("Agent started"),
    on_agent_end=lambda ctx, result: print(f"Done: {result}"),
    on_turn_start=lambda ctx, turn: print(f"Turn {turn}"),
    on_turn_end=lambda ctx, turn: print(f"Turn {turn} complete"),
    before_tool_call=lambda tool, args: print(f"Calling {tool}"),
    after_tool_call=lambda tool, result: print(f"Result: {result}"),
    transform_context=lambda ctx: ctx,  # Can modify context
)
```

### 7. Context Management

```python
from isotopes_core import Context, PruningStrategy

# Context contains full message history
ctx = Context(
    system_prompt="...",
    messages=[...],
    max_tokens=100_000,
    pruning_strategy=PruningStrategy.SLIDING_WINDOW,  # or SUMMARIZE
)

# Auto pruning
ctx.add_message(new_message)
if ctx.needs_pruning():
    ctx.prune()
```

---

## API Design Principles

1. **Async-first** — All I/O operations are async
2. **Generator-based** — prompt() returns AsyncGenerator for streaming
3. **Immutable messages** — Message objects are frozen dataclasses
4. **Explicit over implicit** — No hidden global state

---

## Line Count Estimate

| Module | Lines | Description |
|--------|-------|-------------|
| agent.py | ~400 | Agent class, state management |
| loop.py | ~900 | Core loop logic |
| context.py | ~500 | Context, pruning |
| tools.py | ~500 | Tool, @auto_tool |
| events.py | ~150 | Event types |
| middleware.py | ~250 | Hook definitions |
| types.py | ~200 | Core types |
| providers/ | ~700 | 4 providers |
| **Total** | **~3,600 lines** | |

---

## Reference Implementations

- https://github.com/GhostComplex/project-agent-core — Current prototype
- https://github.com/badlogic/pi-mono — LLM loop reference
- https://github.com/steins-z/claude-code — Claude Code CLI reference

---

## Milestones

### M0: Core Loop (~900 lines)
- `loop.py` — agent_loop() implementation

### M1: Agent Wrapper (~900 lines)
- `agent.py` — Agent class
- `context.py` — Context management + pruning

### M2: Providers (~600 lines)
- `providers/anthropic.py`
- `providers/openai.py`
- `providers/proxy.py`

### M3: Polish (~300 lines)
- `providers/router.py` — Multi-provider routing
- 100% type coverage + documentation

---

## Dependencies

```toml
[project]
dependencies = []

[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.50"]
all = ["anthropic>=0.40", "openai>=1.50"]
```

Core package has **zero dependencies**; providers are optional.

---

## Installation & Usage

```bash
# Core only (for custom providers)
pip install isotopes-core

# With Anthropic provider
pip install isotopes-core[anthropic]

# All providers
pip install isotopes-core[all]
```

```python
from isotopes_core import Agent
from isotopes_core.providers.anthropic import AnthropicProvider

agent = Agent(provider=AnthropicProvider())
async for event in agent.prompt("Hello"):
    print(event)
```
