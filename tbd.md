# Session In Progress

## Current task

Investigate and fix the MCP connection failure seen when adding the remote `kite` MCP server through:

```text
cargo run -p agent-controlplane --bin cli
```

Observed runtime error:

```text
error: server negotiated unsupported protocol version `2025-03-26`
```

## What has been confirmed

- The failure is not in the interactive CLI flow itself.
- The failure comes from the shared MCP client at `mcp/client/src/lib.rs`.
- `McpConnection::initialize()` currently advertises:

```text
SUPPORTED_PROTOCOL_VERSION = "2025-11-25"
```

- After the initialize response returns, the client rejects any server protocol version that is not an exact string match.
- The remote Kite MCP server responds with:

```text
2025-03-26
```

- The HTTP transport already stores the negotiated protocol version after init and sends it in subsequent `MCP-Protocol-Version` headers.
- That means the narrow bug is the exact-equality validation during handshake, not downstream request propagation.

## Repo evidence gathered

- `mcp/client/src/lib.rs`
  - `initialize()` sends `protocolVersion: SUPPORTED_PROTOCOL_VERSION`
  - then errors with `UnsupportedProtocolVersion(...)` unless the response matches exactly
- `mcp/client/src/lib.rs`
  - Streamable HTTP transport already persists the returned protocol version and reuses it for later requests
- `agent/controlplane/src/bin/cli.rs`
  - metadata refresh uses the same shared MCP client, so the failure surfaces in the CLI add/save flow
- `mcp/config/src/lib.rs`
  - registry schema currently has no per-server protocol compatibility override
- `config/mcp_servers.example.json`
  - only transport URL/headers are configurable for remote MCPs

## Constraints / notes

- A targeted `cargo test -p mcp-client ...` run was attempted but blocked on Cargo's artifact lock from another active process in the workspace.
- No code changes have been made yet besides this note file.

## Likely fix direction

- Update the MCP client handshake logic to support compatible protocol negotiation instead of requiring one exact protocol string.
- Keep using the server-returned protocol version for subsequent HTTP requests.
- Add focused tests for:
  - accepting the older compatible protocol returned by a server
  - rejecting truly unsupported protocol versions
  - preserving header propagation after negotiation

## Next steps

1. Check MCP protocol/version negotiation expectations against the spec.
2. Patch `mcp/client/src/lib.rs` to accept the server's compatible negotiated version.
3. Add/adjust tests in `mcp/client/src/lib.rs`.
4. Re-run the focused test(s) once the Cargo lock is free.
5. Re-test the `kite` MCP add flow through the control-plane CLI.
