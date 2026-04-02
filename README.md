# 🫥 isotopes-core

A minimal, pluggable TypeScript agent loop engine.

## Overview

**isotopes-core** is the foundation of the Isotopes agent framework. It provides:

- Agent loop (plan → act → observe → repeat)
- LLM provider abstraction (Anthropic, OpenAI, proxy)
- Type-safe tool definitions with schema generation
- Streaming events via async generators
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
npm install isotopes-core anthropic
```

```typescript
import { Agent, tool } from "isotopes-core";
import { AnthropicProvider } from "isotopes-core/providers/anthropic";

const greet = tool({
  name: "greet",
  description: "Greet someone by name",
  parameters: { name: { type: "string" } },
  execute: async ({ name }) => `Hello, ${name}!`,
});

const agent = new Agent({
  provider: new AnthropicProvider(),
  systemPrompt: "You are a friendly assistant.",
  tools: [greet],
});

for await (const event of agent.prompt("Say hi to Alice")) {
  if (event.type === "text_delta") {
    process.stdout.write(event.text);
  }
}
```

## Documentation

- [PRD](docs/PRD.md) — Product Requirements Document
- [Project Board](https://github.com/orgs/GhostComplex/projects/6)

## Status

🚧 **In Development** — See [PRD](docs/PRD.md) for milestones and progress.

## License

MIT
