# 🫥 isotopes-core PRD

> Version: 0.2.0
> Date: 2026-04-02
> Status: **Draft**
> Stack: **TypeScript** (changed from Python)

## Overview

**isotopes-core** is a standalone TypeScript library providing a pluggable agent loop engine. It is the core of the Isotopes framework, usable independently or embedded in other applications.

## Goals

1. **Minimal core** — Only essential agent loop components, no I/O, no CLI, no persistence
2. **Pluggable** — Configure provider, tools, middleware via dependency injection
3. **Zero runtime deps** — No mandatory dependencies beyond LLM SDKs
4. **Type-safe** — Full TypeScript types, strict mode compatible
5. **Tree-shakeable** — ESM-only, no side effects in core modules

## Non-Goals

- ❌ CLI / TUI
- ❌ Session persistence (JSONL, SQLite, etc.)
- ❌ Concrete tools (bash, read, write, etc.)
- ❌ RPC protocol
- ❌ Skill system
- ❌ MCP integration

These belong in the upper-level `isotopes` package.

---

## Architecture

```
isotopes-core/
├── src/
│   ├── index.ts           # Public API exports
│   ├── agent.ts           # Agent class: state management, prompt(), continue()
│   ├── loop.ts            # agentLoop(): core async generator
│   ├── context.ts         # Context management, pruning strategies
│   ├── tools.ts           # Tool definition, schema generation
│   ├── events.ts          # Event types (discriminated union)
│   ├── middleware.ts      # Lifecycle hooks
│   ├── types.ts           # Core type definitions
│   └── providers/
│       ├── index.ts       # Provider exports
│       ├── types.ts       # Provider protocol (interface)
│       ├── anthropic.ts   # Anthropic Claude
│       ├── openai.ts      # OpenAI GPT
│       └── proxy.ts       # OpenAI-compatible endpoints
├── package.json
└── tsconfig.json
```

## Core Components

### 1. EventStream

We use a push-based event stream:

```typescript
interface EventStream<TEvent, TResult> {
  [Symbol.asyncIterator](): AsyncIterator<TEvent>;
  result(): Promise<TResult>;
}

// Usage
const stream = agent.prompt("Hello!");
for await (const event of stream) {
  if (event.type === "text_delta") {
    process.stdout.write(event.text);
  }
}
const messages = await stream.result();
```

### 2. Agent Class

```typescript
import { Agent, tool } from "isotopes-core";
import { AnthropicProvider } from "isotopes-core/providers/anthropic";

const greet = tool({
  name: "greet",
  description: "Greet someone",
  parameters: { name: { type: "string" } },
  execute: async ({ name }) => `Hello, ${name}!`,
});

const agent = new Agent({
  provider: new AnthropicProvider({ apiKey: "..." }),
  systemPrompt: "You are a helpful assistant.",
  tools: [greet],
  maxTurns: 10,
});

// Stream events
for await (const event of agent.prompt("Say hi to Alice")) {
  console.log(event);
}
```

### 3. Agent Loop

Core loop in `loop.ts`:

```
1. Receive user message
2. Apply transformContext() if configured
3. Convert to LLM format via convertToLlm()
4. Call provider.complete()
5. Stream response, yielding events
6. If tool calls:
   a. Execute tools (parallel by default)
   b. Yield tool results
   c. Append to context
   d. Check max_turns
   e. Go to step 2
7. If no tool calls: yield AgentDone, return
```

### 4. Events (Discriminated Union)

```typescript
type AgentEvent =
  | { type: "agent_start" }
  | { type: "turn_start" }
  | { type: "text_delta"; text: string }
  | { type: "tool_call"; id: string; name: string; arguments: unknown }
  | { type: "tool_result"; id: string; content: Content[]; isError: boolean }
  | { type: "turn_end"; hasToolCalls: boolean }
  | { type: "agent_done"; messages: Message[]; usage: Usage }
  | { type: "agent_error"; error: Error; messages: Message[] };
```

### 5. Tool Definition

```typescript
// Option 1: Object-based definition
const search = tool({
  name: "search",
  description: "Search the web",
  parameters: {
    query: { type: "string", description: "Search query" },
    maxResults: { type: "number", default: 10 },
  },
  execute: async ({ query, maxResults }) => {
    return `Results for: ${query}`;
  },
});

// Option 2: Zod schema (via isotopes-core/zod)
import { z } from "zod";
import { zodTool } from "isotopes-core/zod";

const search = zodTool({
  name: "search",
  description: "Search the web",
  parameters: z.object({
    query: z.string(),
    maxResults: z.number().default(10),
  }),
  execute: async ({ query, maxResults }) => {
    return `Results for: ${query}`;
  },
});
```

### 6. Provider Protocol

```typescript
interface Provider {
  complete(
    messages: Message[],
    options: {
      systemPrompt?: string;
      tools?: Tool[];
      temperature?: number;
      maxTokens?: number;
      signal?: AbortSignal;
    }
  ): AsyncGenerator<StreamEvent, void, unknown>;
}

type StreamEvent =
  | { type: "message_start"; usage?: Usage }
  | { type: "content_block_start"; index: number; blockType: "text" | "tool_use"; toolCallId?: string; toolName?: string }
  | { type: "content_block_delta"; index: number; text?: string; partialJson?: string }
  | { type: "content_block_stop"; index: number }
  | { type: "message_delta"; stopReason?: string; usage?: Usage }
  | { type: "message_stop" };
```

### 7. Middleware / Hooks

```typescript
const agent = new Agent({
  // ...
  hooks: {
    onAgentStart: (ctx) => console.log("Started"),
    onAgentEnd: (ctx, messages) => console.log("Done"),
    onTurnStart: (ctx, turn) => console.log(`Turn ${turn}`),
    onTurnEnd: (ctx, turn) => console.log(`Turn ${turn} complete`),
    beforeToolCall: (tool, args) => args, // Can modify args or throw to block
    afterToolCall: (tool, result) => result, // Can modify result
    transformContext: (messages) => messages, // Pre-LLM transform
  },
});
```

### 8. Context Management

```typescript
import { Context, pruneOldest, pruneSummarize } from "isotopes-core";

const context = new Context({
  maxTokens: 100_000,
  pruningStrategy: pruneOldest({ keepLast: 10 }),
});

context.addMessage({ role: "user", content: [{ type: "text", text: "Hello" }] });

if (context.needsPruning()) {
  await context.prune();
}
```

---

## TypeScript Design Principles

1. **ESM-only** — No CommonJS, enables tree-shaking
2. **Strict mode** — `strict: true` in tsconfig
3. **Discriminated unions** — For events and content types
4. **Generics** — For provider-specific options
5. **No classes where unnecessary** — Prefer functions and interfaces
6. **Async generators** — For streaming (not callbacks)
7. **AbortSignal** — For cancellation throughout

---

## Line Count Estimate

| Module | Lines | Description |
|--------|-------|-------------|
| agent.ts | ~300 | Agent class, state management |
| loop.ts | ~400 | Core loop logic |
| context.ts | ~300 | Context, pruning |
| tools.ts | ~250 | Tool definition, schema |
| events.ts | ~80 | Event types |
| middleware.ts | ~100 | Hook definitions |
| types.ts | ~150 | Core types |
| providers/ | ~600 | 3 providers |
| **Total** | **~2,200 lines** | |

Note: TypeScript is more concise than Python for this use case due to:
- No runtime type validation boilerplate (types are compile-time)
- Discriminated unions vs dataclasses
- Async/await syntax differences

---

## Dependencies

```json
{
  "dependencies": {},
  "peerDependencies": {
    "anthropic": ">=0.40.0",
    "openai": ">=4.70.0"
  },
  "peerDependenciesMeta": {
    "anthropic": { "optional": true },
    "openai": { "optional": true }
  }
}
```

Core package has **zero dependencies**; SDK packages are peer dependencies.

---

## Package Structure

```json
{
  "name": "isotopes-core",
  "type": "module",
  "exports": {
    ".": "./dist/index.js",
    "./providers/anthropic": "./dist/providers/anthropic.js",
    "./providers/openai": "./dist/providers/openai.js",
    "./providers/proxy": "./dist/providers/proxy.js",
    "./zod": "./dist/zod.js"
  }
}
```

---

## Milestones

### M0: Core Loop (~800 lines)
- `types.ts` — Message, Content, Usage
- `events.ts` — AgentEvent discriminated union
- `tools.ts` — Tool definition, schema generation
- `loop.ts` — `agentLoop()` async generator
- `providers/types.ts` — Provider interface, StreamEvent

### M1: Agent Wrapper (~600 lines)
- `agent.ts` — Agent class
- `context.ts` — Context management + pruning
- `middleware.ts` — Hooks

### M2: Providers (~600 lines)
- `providers/anthropic.ts`
- `providers/openai.ts`
- `providers/proxy.ts`

### M3: Polish (~200 lines)
- `zod.ts` — Zod integration for tool schemas
- JSDoc documentation
- README examples

---

## Installation & Usage

```bash
# Core only (for custom providers)
npm install isotopes-core

# With Anthropic provider
npm install isotopes-core anthropic

# With OpenAI provider
npm install isotopes-core openai
```

```typescript
import { Agent } from "isotopes-core";
import { AnthropicProvider } from "isotopes-core/providers/anthropic";

const agent = new Agent({
  provider: new AnthropicProvider(),
});

for await (const event of agent.prompt("Hello")) {
  console.log(event);
}
```
