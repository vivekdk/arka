![Arka logo](assets/arka-logo.svg)

# AI Data Analyst

Rust workspace for an agent runtime and channel control plane built around MCP.

Today the repo contains:

- an MCP client and JSON registry for local `stdio` and remote Streamable HTTP MCP servers
- a single-turn agent runtime with typed turn/step/message state
- an OpenAI-backed model adapter
- a server-hosted control plane with sessions, approvals, HTTP API, SSE events, and channel adapters
- a thin CLI that talks to the control-plane API

## Architecture

The system is layered deliberately:

1. `mcp/*`
   - `mcp/config`: loads and validates the MCP registry JSON
   - `mcp/client`: connects to MCP servers, performs handshake, lists tools, and calls tools
   - `mcp/cli`: small inspection CLI for configured MCP servers
2. `agent/runtime`
   - owns the single-turn execution loop
   - builds prompts from canonical state records
   - executes MCP actions and local tools
   - emits typed runtime events
   - evaluates tool-availability policy and produces per-step tool masks
3. `agent/openai`
   - OpenAI Responses API adapter behind the runtime `ModelAdapter` trait
4. `agent/controlplane`
   - adds sessions above the single-turn runtime
   - exposes HTTP endpoints and SSE event streams
   - normalizes API, CLI, Slack, and WhatsApp traffic into one session model
   - manages approvals and idempotency

## Workspace

```text
.
|-- Cargo.toml
|-- .env.example
|-- config/
|   `-- mcp_servers.example.json
|-- agent/
|   |-- controlplane/
|   |-- openai/
|   `-- runtime/
`-- mcp/
    |-- cli/
    |-- client/
    `-- config/
```

## Requirements

- Rust toolchain with `cargo`
- at least one local MCP server executable available on your machine
- an OpenAI API key if you want to run the control-plane server against the OpenAI adapter

## Configuration

### `.env`

Create a local `.env` by copying `.env.example`.

The control-plane server reads local configuration from `.env` automatically.

Current variables:

```dotenv
OPENAI_API_KEY=
MODEL_NAME=gpt-5.4-mini
SYSTEM_PROMPT=config/prompt.md
MCP_REGISTRY_PATH=config/mcp_servers.json
SUBAGENT_REGISTRY_PATH=config/subagents.json
TOOL_POLICY_PATH=
SESSION_STORE_DIR=data/sessions
```

`SYSTEM_PROMPT` is the path to the prompt file. The server re-reads that file at the beginning of every turn, resolves supported dynamic tags, and then uses the rendered contents as the system prompt for that turn.

Supported dynamic tags:

- `<dynamic variable: working_directory>`: renders the session workspace path, which defaults to `<cwd>/.arka/tmp/<session_id>`
- `<dynamic variable: current_date and time>`: renders the local date only, in `YYYY-MM-DD` format
- `<dynamic variable: available MCPs>`
- `<dynamic variable: available sub-agents>`
- `<dynamic variable: available tools>`: renders a generic runtime-managed tools notice rather than enumerating the tool catalog

Optional variables:

```dotenv
BIND_ADDR=127.0.0.1:8080
ENABLED_MCP_SERVERS=sqlite
AGENT_API_BASE_URL=http://127.0.0.1:8080
AGENT_REQUEST_TIMEOUT_SECS=240
SLACK_BOT_TOKEN=
SLACK_SIGNING_SECRET=
SLACK_API_BASE_URL=https://slack.com/api
SLACK_EVENT_QUEUE_CAPACITY=256
WHATSAPP_GATEWAY_ENABLED=false
WHATSAPP_LOCAL_WEB_ENABLED=false
WHATSAPP_ACCOUNT_ID=default
WHATSAPP_DM_POLICY=allowlist
WHATSAPP_DM_ALLOW_FROM=
WHATSAPP_EVENT_QUEUE_CAPACITY=256
WHATSAPP_BRIDGE_BASE_URL=http://127.0.0.1:8091
RUNTIME_MAX_STEPS_PER_TURN=8
RUNTIME_MAX_MCP_CALLS_PER_STEP=4
RUNTIME_MAX_SUBAGENT_STEPS_PER_INVOCATION=25
RUNTIME_MAX_SUBAGENT_MCP_CALLS_PER_INVOCATION=4
RUNTIME_TURN_TIMEOUT_SECS=420
RUNTIME_MCP_CALL_TIMEOUT_SECS=10
RUNTIME_REQUIRE_TODOS=true
```

Environment variables exported by your shell still override `.env`.

`AGENT_REQUEST_TIMEOUT_SECS` controls how long the CLI waits for a message-send
response from the control-plane API. Raise it if your turn budget or MCP work
regularly exceeds the default.

### MCP registry

The default MCP registry path is `config/mcp_servers.json`.
Create it by copying `config/mcp_servers.example.json`.

Current schema:

```json
{
  "servers": [
    {
      "name": "sqlite",
      "transport": {
        "type": "stdio",
        "command": "uvx",
        "args": ["mcp-server-sqlite"]
      }
    }
  ]
}
```

For remote MCP servers, use Streamable HTTP:

```json
{
  "servers": [
    {
      "name": "remote-crm",
      "transport": {
        "type": "streamable_http",
        "url": "https://example.com/mcp",
        "headers": {
          "Authorization": "Bearer ${TOKEN}"
        }
      }
    }
  ]
}
```

For OAuth-backed remote MCP servers from stdio-only hosts, wrap the remote
server with `mcp-remote` and keep the registry entry in stdio form:

```json
{
  "servers": [
    {
      "name": "kite",
      "transport": {
        "type": "stdio",
        "command": "npx",
        "args": [
          "-y",
          "mcp-remote",
          "https://mcp.kite.trade/mcp",
          "--debug"
        ]
      },
      "description": "Remote Kite MCP via mcp-remote; OAuth handled by the wrapper."
    }
  ]
}
```

Notes for `mcp-remote` in Arka:

- Arka treats it as a normal stdio MCP server
- the wrapper owns browser/callback OAuth and token storage
- use extra args such as `--resource`, `--auth-timeout`, `--host`, `--header`, or `--static-oauth-client-info` by appending them after the remote URL
- if startup is flaky, prefer `npx -y`, add `--debug`, and inspect `~/.mcp-auth/*_debug.log`

Backwards compatibility: the legacy stdio-only shape using top-level `command`, `args`, and `env` is still accepted.

Validation rules enforced by `mcp-config`:

- `name` must be non-empty
- `name` must be unique across the registry
- stdio `command` must be non-empty
- Streamable HTTP `transport.url` must be a non-empty `http://` or `https://` URL
- `args` defaults to `[]`

### Runtime state

`SESSION_STORE_DIR` defaults to `data/sessions`. That directory contains generated
session state, message transcripts, and channel metadata and should remain local
runtime data, not committed source. Generated analysis scripts and outputs live
separately under `.arka/tmp/<session_id>/`.

### Sub-agents

The default sub-agent registry is `config/subagents.json`.

Current built-in executors:

- `mcp-executor`: delegated MCP tool/resource execution
- `tool-executor`: delegated local tool execution inside the session working directory

Current built-in local tools:

- `glob`
- `read_file`
- `write_file`
- `edit_file`
- `bash`

### Tool policy

`TOOL_POLICY_PATH` is optional. When set, it should point to a JSON overlay file
that adjusts the default Rust tool policy rules without changing the static tool
catalog in the prompt.

Start from `config/tool_policy.example.json` if you want an overlay.
The default policy denies `file_write` and `command_exec` tools for WhatsApp delegated execution.

## Running It

### 1. Inspect an MCP server

```bash
cargo run -p mcp-cli -- inspect --server sqlite
```

Use a custom config path:

```bash
cargo run -p mcp-cli -- inspect --server sqlite --config /path/to/mcp_servers.json
```

### 2. Start the control-plane server

Create a local `.env` first. A minimal setup is:

```dotenv
OPENAI_API_KEY=...
MODEL_NAME=gpt-5.4-mini
SYSTEM_PROMPT=config/prompt.md
MCP_REGISTRY_PATH=config/mcp_servers.json
SUBAGENT_REGISTRY_PATH=config/subagents.json
SESSION_STORE_DIR=data/sessions
```

Optional policy overlay:

```dotenv
TOOL_POLICY_PATH=config/tool_policy.json
```

If you use a policy overlay, create it from `config/tool_policy.example.json`.

```bash
cargo run -p agent-controlplane --bin server
```

By default it binds to `127.0.0.1:8080`.

The server:

1. loads `.env` if present
2. creates an OpenAI model adapter
3. creates the single-turn runtime
4. wraps it with the session control plane
5. serves the HTTP API and SSE event stream

### 2a. Start the local WhatsApp Web bridge

```bash
npm install
npm run whatsapp-web
```

Optional bridge variables:

```dotenv
ARKA_WHATSAPP_BRIDGE_PORT=8091
ARKA_WHATSAPP_CONTROLPLANE_BASE_URL=http://127.0.0.1:8080
ARKA_WHATSAPP_ACCOUNT_ID=default
ARKA_WHATSAPP_AUTH_DIR=data/whatsapp-web/auth
```

To let the control-plane server use the local WhatsApp Web bridge, enable:

```dotenv
WHATSAPP_GATEWAY_ENABLED=true
WHATSAPP_LOCAL_WEB_ENABLED=true
WHATSAPP_BRIDGE_BASE_URL=http://127.0.0.1:8091
```

### 3. Use the CLI client

Start a session:

```bash
cargo run -p agent-controlplane --bin cli -- session start
```

Send a message:

```bash
cargo run -p agent-controlplane --bin cli -- message send <session_id> "Analyze this dataset"
```

Watch SSE events for a session:

```bash
cargo run -p agent-controlplane --bin cli -- session watch <session_id>
```

Run a simple interactive loop:

```bash
cargo run -p agent-controlplane --bin cli -- repl
```

Submit an approval:

```bash
cargo run -p agent-controlplane --bin cli -- approve <session_id> <approval_id> approve
```

Inspect WhatsApp bridge status:

```bash
cargo run -p agent-controlplane --bin cli -- whatsapp status
```

Start WhatsApp QR login:

```bash
cargo run -p agent-controlplane --bin cli -- whatsapp login
```

Complete login after scanning the QR shown by the bridge:

```bash
cargo run -p agent-controlplane --bin cli -- whatsapp complete-login <login_session_id>
```

## HTTP API

Current routes:

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

- session events are exposed over SSE from `/sessions/{session_id}/events`
- Slack uses the HTTP Events API route at `/channels/slack/events`, verifies request signatures, queues valid events, and posts replies back with the Slack Web API
- WhatsApp can now run in a stateful gateway mode with login state, DM access policy, background outbound delivery, and a normalized `/channels/whatsapp/inbound` route
- when `WHATSAPP_LOCAL_WEB_ENABLED=true`, Arka talks to a local Baileys-based WhatsApp Web bridge instead of a business API provider
- `/channels/whatsapp/webhook` remains as a compatibility alias for normalized legacy inbound payloads
- Slack replies are thread-scoped and user-scoped: one internal session is keyed by workspace, channel, thread, and user

## Runtime Model

`agent/runtime` is intentionally single-turn.

For each user turn it:

1. builds a prompt from typed conversation and turn records
2. calls the model through the `ModelAdapter` trait
3. validates the model decision
4. delegates to `mcp-executor` or `tool-executor` when requested
5. executes MCP calls or local tools through the delegated executor
6. evaluates tool policy and emits per-step tool masks
7. records typed messages and runtime events
7. returns a `TurnOutcome`

Important design points:

- MCP calls are active
- local runtime tools such as `glob`, `read_file`, `edit_file`, `write_file`, and `bash` are active and workspace-scoped
- the prompt-visible tool catalog stays static while per-step tool availability is enforced by the harness
- observability is exposed as typed events collected in memory and returned to callers

## Current State

Implemented:

- MCP registry loading and validation
- MCP `initialize`, `notifications/initialized`, `tools/list`, and `tools/call`
- single-turn runtime with guardrails and typed state
- delegated `mcp-executor` and `tool-executor` flows
- local `glob`, `read_file`, `write_file`, `edit_file`, and `bash` execution
- tool policy evaluation with optional JSON overlay config
- OpenAI adapter
- in-memory control-plane session orchestration
- API routes, SSE events, approvals, idempotency
- CLI client
- Slack ingress verification, async event processing, and outbound threaded delivery
- WhatsApp gateway status/login APIs, DM allowlist policy, normalized inbound routing, async outbound delivery worker, and a local WhatsApp Web bridge sidecar

Not implemented yet:

- persistent storage beyond the in-memory control-plane store
- authentication and authorization
- end-to-end automated tests against a real WhatsApp account; the local WhatsApp Web bridge is implemented, but live pairing still needs manual verification
- additional local tools beyond the current built-in workspace tool set
- long-running job queue or distributed workers

## Development

Run the full test suite:

```bash
cargo test --workspace
```

Run a narrower package test suite:

```bash
cargo test -p agent-runtime
cargo test -p agent-controlplane
cargo test -p mcp-client
```

## Repo Notes

- `.env` is ignored by git
- `.env.example` is reserved if you want to add a committed template later
- the control-plane server persists resumable session state under `SESSION_STORE_DIR` as JSON/JSONL files
- debug history remains a separate optional Postgres-backed observability path
