![Arka logo](assets/arka-logo.svg)

# Arka

Arka is a data insights harness for turning raw data into deep analysis, clear insights, and easy-to-understand visualizations.

It is built for teams that want agent-driven data analysis across MCP-connected systems, with structured execution, reproducible artifacts, and channel-friendly outputs.

At a high level, Arka combines:

- MCP for structured access to databases, APIs, and external capability surfaces
- sub-agent orchestration for separating discovery, computation, and local execution concerns
- planning and task breakdown through chain-of-thought-guided todo planning
- an orchestration loop with a Ralph-loop-style execution model at its core
- local tools for workspace inspection, scripting, file generation, and report delivery
- context and state management across turns, steps, sessions, and delegated work

The repository combines:

- an MCP registry, client, metadata loader, and inspection CLI
- a single-turn agent runtime with typed turn, step, and message state
- an OpenAI-backed model adapter
- a control plane with sessions, approvals, HTTP API, SSE streams, and channel integrations
- Slack ingress and threaded outbound delivery through the control plane
- a local WhatsApp Web bridge sidecar for development and operator workflows

## What This Repo Includes

| Area | Purpose |
| --- | --- |
| `mcp/config` | Loads and validates MCP registry configuration |
| `mcp/client` | Connects to MCP servers and executes MCP capabilities |
| `mcp/metadata` | Loads and resolves MCP capability metadata for prompting and routing |
| `mcp/cli` | CLI for inspecting configured MCP servers |
| `agent/runtime` | Single-turn runtime, delegation loop, local tools, and guardrails |
| `agent/openai` | OpenAI Responses API adapter for the runtime |
| `agent/controlplane` | Session orchestration, approvals, API, SSE, and channel handling |
| `bridges/whatsapp-web` | Optional local WhatsApp Web bridge using Baileys |

## Architecture

Arka is intentionally split into clear layers:

1. The MCP layer loads registry config, discovers server capabilities, and executes MCP tools or resources.
2. The runtime layer owns one turn of execution at a time, including prompting, decision validation, delegation, tool execution, and typed event emission.
3. The model adapter layer isolates provider-specific behavior behind the runtime `ModelAdapter` trait.
4. The control plane adds session state, approvals, resumability, transport adapters, and API surfaces on top of the single-turn runtime.

The runtime remains single-turn by design. Multi-turn behavior is handled by the control plane.

## Architectural Model

Arka should be read as a harness rather than a single model wrapper. The main architectural ideas are:

- **MCP-first execution**: external systems are accessed through MCP registry configuration, metadata, capability discovery, and bounded MCP execution instead of ad hoc connector code
- **Sub-agent orchestration**: the main agent delegates bounded work to purpose-built executors such as `mcp-executor` and `tool-executor`, which keeps discovery, retrieval, local computation, and report generation separated
- **Planning and task breakdown**: planning is explicit, with todo-driven decomposition and executor-specific task assignment before substantive execution begins
- **Ralph-loop-style orchestration**: the core loop repeatedly plans, delegates, executes, observes results, updates local state, and continues until the turn is resolved or requires approval
- **Tool-driven analysis**: local tools are part of the runtime contract, enabling file inspection, edits, shell execution, reproducible scripts, generated outputs, and deterministic report artifacts
- **Context management**: prompts are assembled from canonical turn state, conversation history, recent session messages, execution handoff data, tool policy context, and MCP metadata rather than from raw transcript text alone
- **State management**: Arka persists session-level records in the control plane and maintains typed turn, step, message, approval, and artifact state inside the runtime harness

In practice, this means Arka is designed to turn ambiguous data questions into a managed execution workflow: plan the work, route it to the right executor, gather evidence through MCP or tools, materialize artifacts, and return a channel-appropriate result.

The “Ralph-loop-style” phrasing here refers to an iterative plan, delegate, execute, observe, and update loop rather than a single-shot model call.

## How It Works

At a high level, one request moves through Arka like this:

1. A user request enters the control plane through the API, CLI, Slack, or WhatsApp.
2. The runtime builds planning context from conversation history, session state, available MCP capabilities, and current tool policy.
3. Planning produces an explicit task breakdown, usually as ordered todos with executor ownership.
4. The main agent delegates bounded work to `mcp-executor`, `tool-executor`, or keeps synthesis work in the main loop.
5. MCP-backed discovery and retrieval happen through the MCP layer; local computation, scripts, charts, and report generation happen through local tools.
6. The runtime updates typed turn, step, message, and artifact state after each action.
7. The control plane returns a channel-appropriate answer and can attach generated artifacts such as charts or HTML reports.

Typical outputs include concise answers, reproducible intermediate files, generated charts, and HTML reports under the session workspace `outputs/` directory.

## Channel Integrations

Arka currently supports two messaging-channel integrations through the control plane:

- Slack via the Events API, request-signature verification, thread-aware session routing, and outbound replies through the Slack Web API
- WhatsApp via a normalized gateway flow and an optional local WhatsApp Web bridge for development workflows

Slack is part of the main control-plane surface. The WhatsApp Web bridge is optional.

## Repository Layout

```text
.
|-- Cargo.toml
|-- .env.example
|-- config/
|   |-- mcp_servers.example.json
|   |-- prompt.md
|   |-- subagents.json
|   `-- tool_policy.example.json
|-- agent/
|   |-- controlplane/
|   |-- openai/
|   `-- runtime/
|-- mcp/
|   |-- cli/
|   |-- client/
|   |-- config/
|   `-- metadata/
`-- bridges/
    `-- whatsapp-web/
```

## Requirements

- Rust toolchain with `cargo`
- at least one MCP server you can reach locally or over Streamable HTTP
- an OpenAI API key if you want to run the control plane against the OpenAI adapter
- Node.js and `npm` only if you want to run the local WhatsApp Web bridge

## Quick Start

### 1. Create local config

```bash
cp .env.example .env
cp config/mcp_servers.example.json config/mcp_servers.json
```

Set `OPENAI_API_KEY` in `.env`, then update `config/mcp_servers.json` for the MCP servers you actually want to use.

### 2. Inspect an MCP server

```bash
cargo run -p mcp-cli -- inspect --server sqlite-local
```

Use `--config /path/to/mcp_servers.json` if you want a non-default registry path.

### 3. Start the control-plane server

```bash
cargo run -p agent-controlplane --bin server
```

By default, the server binds to `127.0.0.1:8080`.

### 4. Start a CLI session

```bash
cargo run -p agent-controlplane --bin cli -- session start
```

Send a message:

```bash
cargo run -p agent-controlplane --bin cli -- message send <session_id> "Analyze this dataset"
```

Watch session events:

```bash
cargo run -p agent-controlplane --bin cli -- session watch <session_id>
```

## Configuration

### Environment

The control-plane server loads `.env` automatically when present. Shell-exported variables still take precedence.

Key variables from `.env.example`:

```dotenv
OPENAI_API_KEY=
MODEL_NAME=gpt-5.4-mini

SYSTEM_PROMPT=config/prompt.md
MCP_REGISTRY_PATH=config/mcp_servers.json
SUBAGENT_REGISTRY_PATH=config/subagents.json
TOOL_POLICY_PATH=
SESSION_STORE_DIR=data/sessions

RUNTIME_DEBUG_POSTGRES_ENABLED=false
RUNTIME_DEBUG_POSTGRES_DSN=postgresql://USERNAME:PASSWORD@HOST:5432/arka

SLACK_BOT_TOKEN=
SLACK_SIGNING_SECRET=

BIND_ADDR=127.0.0.1:8080
AGENT_API_BASE_URL=http://127.0.0.1:8080
AGENT_REQUEST_TIMEOUT_SECS=240

RUNTIME_MAX_STEPS_PER_TURN=20
RUNTIME_MAX_MCP_CALLS_PER_STEP=20
RUNTIME_MAX_SUBAGENT_STEPS_PER_INVOCATION=25
RUNTIME_MAX_SUBAGENT_MCP_CALLS_PER_INVOCATION=20
RUNTIME_MAX_DUPLICATE_MCP_CALLS_PER_INVOCATION=3
RUNTIME_TURN_TIMEOUT_SECS=420
RUNTIME_REQUIRE_TODOS=true
```

`SESSION_STORE_DIR` holds local control-plane state. Session workspaces and generated artifacts live under `.arka/tmp/<session_id>/`.

`RUNTIME_DEBUG_POSTGRES_DSN` is the optional Postgres connection string used for persisted runtime debug history and tracing views.

For Slack, configure:

- `SLACK_BOT_TOKEN` for outbound delivery
- `SLACK_SIGNING_SECRET` for inbound request verification

Minimal Slack setup:

- expose `POST /channels/slack/events` to Slack Events API
- configure Slack to send app mentions or thread replies to that route
- expect replies to remain thread-scoped and user-scoped inside Arka’s session model

### Prompt templates

`SYSTEM_PROMPT` points to the base prompt file. The server reloads that file at the start of each turn and resolves supported dynamic tags.

Supported dynamic tags:

- `<dynamic variable: working_directory>`
- `<dynamic variable: current_date and time>`
- `<dynamic variable: available MCPs>`
- `<dynamic variable: available sub-agents>`
- `<dynamic variable: available tools>`

### MCP registry

The default registry path is `config/mcp_servers.json`. Start from `config/mcp_servers.example.json`.

Example stdio server:

```json
{
  "servers": [
    {
      "name": "sqlite-local",
      "transport": {
        "type": "stdio",
        "command": "uvx",
        "args": ["mcp-server-sqlite", "--db-path", "/absolute/path/to/local.db"]
      }
    }
  ]
}
```

Example Streamable HTTP server:

```json
{
  "servers": [
    {
      "name": "posthog",
      "transport": {
        "type": "streamable_http",
        "url": "https://posthog.example.com/mcp",
        "headers": {
          "Authorization": "Bearer replace-me"
        }
      }
    }
  ]
}
```

Example `mcp-remote` wrapper for OAuth-backed remote servers:

```json
{
  "servers": [
    {
      "name": "kite-remote",
      "transport": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "mcp-remote", "https://mcp.kite.trade/mcp", "--debug"]
      }
    }
  ]
}
```

Validation enforced by `mcp-config` includes:

- unique non-empty server names
- non-empty stdio commands
- valid `http://` or `https://` URLs for Streamable HTTP transports
- `args` defaulting to `[]` when omitted

### Sub-agents

The default sub-agent registry is `config/subagents.json`.

Built-in executors:

- `mcp-executor` for delegated MCP work against the selected server or capability scope
- `tool-executor` for delegated local tool work inside the session working directory

### Tool policy overlay

`TOOL_POLICY_PATH` is optional. When set, it points to a JSON overlay that adjusts the default tool policy without changing the prompt-visible catalog. Start from `config/tool_policy.example.json`.

## Local WhatsApp Web Bridge

The local bridge is optional and is only needed when running WhatsApp flows through a local Baileys-based sidecar.

Install dependencies:

```bash
npm install
```

Start the bridge:

```bash
npm run whatsapp-web
```

Relevant bridge variables:

```dotenv
ARKA_WHATSAPP_BRIDGE_PORT=8091
ARKA_WHATSAPP_CONTROLPLANE_BASE_URL=http://127.0.0.1:8080
ARKA_WHATSAPP_ACCOUNT_ID=default
ARKA_WHATSAPP_AUTH_DIR=data/whatsapp-web/auth
```

To enable the control plane to use the bridge:

```dotenv
WHATSAPP_GATEWAY_ENABLED=true
WHATSAPP_LOCAL_WEB_ENABLED=true
WHATSAPP_BRIDGE_BASE_URL=http://127.0.0.1:8091
```

## CLI Commands

Start an interactive loop:

```bash
cargo run -p agent-controlplane --bin cli -- repl
```

Submit an approval:

```bash
cargo run -p agent-controlplane --bin cli -- approve <session_id> <approval_id> approve
```

Inspect WhatsApp status:

```bash
cargo run -p agent-controlplane --bin cli -- whatsapp status
```

Start WhatsApp QR login:

```bash
cargo run -p agent-controlplane --bin cli -- whatsapp login
```

Complete login after scanning the QR:

```bash
cargo run -p agent-controlplane --bin cli -- whatsapp complete-login <login_session_id>
```

## HTTP API

Primary routes:

- `POST /sessions`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/messages`
- `POST /sessions/{session_id}/messages`
- `POST /sessions/{session_id}/approvals/{approval_id}`
- `GET /sessions/{session_id}/events`
- `POST /channels/slack/events`
- `GET /channels/whatsapp/status`
- `POST /channels/whatsapp/login/start`
- `POST /channels/whatsapp/login/complete`
- `POST /channels/whatsapp/logout`
- `POST /channels/whatsapp/inbound`
- `POST /channels/whatsapp/webhook`

Notes:

- session event streaming is exposed over SSE at `/sessions/{session_id}/events`
- Slack uses the Events API, verifies signatures, queues valid events, and replies back through the Slack Web API
- WhatsApp supports both a normalized gateway flow and a local WhatsApp Web bridge flow
- Slack sessions are keyed by workspace, channel, thread, and user

## Safety and Approvals

Arka is designed as a managed execution harness, not a free-form tool runner.

- tool availability is enforced by the runtime harness and policy engine, not only by prompt instructions
- delegated work is bounded by executor type, selected scope, and per-turn runtime limits
- approvals are part of the control-plane model and can gate sensitive or policy-restricted actions
- session work is isolated into per-session workspaces under `.arka/tmp/`
- typed state and ordered todos help prevent out-of-order execution and make partial progress explicit

## Runtime Behavior

For each turn, `agent/runtime`:

1. builds a prompt from typed conversation and turn state
2. calls the model through `ModelAdapter`
3. validates the model decision
4. delegates to `mcp-executor` or `tool-executor` when needed
5. executes MCP calls or local tools within runtime guardrails
6. evaluates tool policy and emits per-step tool masks
7. records typed messages and runtime events
8. returns a `TurnOutcome`

Local tool execution is workspace-scoped. The prompt-visible tool catalog stays stable while the harness enforces actual per-step availability.

## Observability

Arka exposes execution state and diagnostics through several layers:

- typed runtime events for turns, steps, tool calls, MCP calls, and delegated execution
- SSE session streams from the control plane for live turn progress
- persisted local session records under `SESSION_STORE_DIR`
- optional Postgres-backed debug history for deeper runtime inspection

This makes it possible to trace what the system planned, what it delegated, what it executed, and what artifacts were produced.

## Development

Run the full workspace test suite:

```bash
cargo test --workspace
```

Run package-specific tests:

```bash
cargo test -p agent-runtime
cargo test -p agent-controlplane
cargo test -p mcp-client
```

## Status and Limitations

Implemented today:

- MCP registry loading and validation
- MCP capability discovery and execution
- single-turn runtime with typed state and delegation
- local tools: `glob`, `read_file`, `write_file`, `edit_file`, and `bash`
- optional tool policy overlays
- OpenAI-backed model adapter
- session-aware control plane with approvals and SSE
- CLI client
- Slack integration
- WhatsApp gateway APIs and local WhatsApp Web bridge

Not yet addressed:

- durable primary storage beyond the local session store
- authn/authz for multi-user deployment
- long-running job orchestration or distributed workers
- broader local tool inventory beyond the current workspace-scoped set
- full end-to-end automation against a live WhatsApp account

## Deployment Notes

Today, Arka is best understood as a serious internal-platform or operator-facing system rather than a turnkey multi-tenant SaaS product.

- local session persistence exists and is usable for development and internal workflows
- the control plane and runtime are structured for production-oriented behavior, but authn/authz and broader deployment hardening are still incomplete
- Slack support is integrated into the main control plane; WhatsApp support is available but has more operational caveats, especially for live-account automation

## Repository Notes

- `.env`, session data, generated `outputs/`, and local workspace artifacts are not committed
- control-plane session state is stored under `SESSION_STORE_DIR` as local JSON and JSONL files
- optional Postgres-backed runtime debug history exists separately from the local session store

## Development Workflow

For contributors and maintainers:

- run `cargo test --workspace` before merging substantial runtime or control-plane changes
- keep config examples in sync with `.env.example`, `config/mcp_servers.example.json`, and `config/subagents.json`
- treat prompt changes, policy changes, and runtime loop changes as architectural changes and document them in the README when they affect how Arka is operated

## License

This repository is licensed under the `0BSD` license. See [LICENSE](/Users/vivek/Documents/ai/ai-data-analyst/LICENSE).
