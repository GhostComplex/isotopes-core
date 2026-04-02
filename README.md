# 🫥 isotopes-core

A minimal, pluggable Python agent loop engine.

## What is this?

**isotopes-core** is the foundation of the Isotopes agent framework. It provides:

- Agent loop (plan → act → observe → repeat)
- LLM provider abstraction (Anthropic, OpenAI, proxy)
- `@auto_tool` decorator for zero-boilerplate tool definitions
- Typed event streaming
- Context management with pruning strategies
- Composable middleware/hooks

## What this is NOT

This is a **core library**, not a full agent framework. It does NOT include:

- CLI / TUI
- Session persistence
- Concrete tools (bash, read, write)
- RPC protocol
- Skill system
- MCP integration

For those, see the main [isotopes](https://github.com/GhostComplex/isotopes) package.

## Quick Start

```bash
pip install isotopes-core[anthropic]
```

```python
from isotopes_core import Agent, auto_tool
from isotopes_core.providers.anthropic import AnthropicProvider

@auto_tool
async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

agent = Agent(
    provider=AnthropicProvider(),
    system_prompt="You are a friendly assistant.",
    tools=[greet],
)

async for event in agent.prompt("Say hi to Alice"):
    print(event)
```

## Architecture

```
isotopes-core/
├── agent.py       # Agent class: state management, prompt(), continue_()
├── loop.py        # Core loop: plan → act → observe
├── context.py     # Context management, pruning strategies
├── tools.py       # Tool definition, @auto_tool decorator
├── events.py      # Event types for streaming
├── middleware.py  # Lifecycle hooks
├── types.py       # Core type definitions
└── providers/     # LLM provider abstraction
    ├── base.py    # Provider protocol
    ├── anthropic.py
    ├── openai.py
    └── proxy.py   # OpenAI-compatible endpoints
```

## Documentation

- [PRD](https://github.com/GhostComplex/backlog/blob/main/project-isotopes/isotopes-core-prd.md)
- [Project Board](https://github.com/orgs/GhostComplex/projects/6)

## License

MIT
