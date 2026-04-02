# 🫥 isotopes-core

A minimal, pluggable Python agent loop engine.

## Overview

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

## Documentation

- [PRD](docs/PRD.md) — Product Requirements Document
- [Project Board](https://github.com/orgs/GhostComplex/projects/6)

## Status

🚧 **In Development** — See [PRD](docs/PRD.md) for milestones and progress.

## License

MIT
