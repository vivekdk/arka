---
schema_version: 1
server:
  logical_name: kite
  protocol_name: fake-runtime-server
  title: Kite Test Server
  version: 1.0.0
  description: Kite auth test MCP
  instructions_summary: Use the login tool to check authentication.
capability_families:
  tools:
    supported: true
    count: 1
  resources:
    supported: false
    count: 0
tools:
- name: login
  title: Login
  description: Return the login link when authentication is required.
  input_schema:
    type: object
resources: []
extensions: {}
---

# MCP Full: kite

## Server
- Logical name: `kite`
- Protocol name: `fake-runtime-server`
- Title: Kite Test Server
- Version: `1.0.0`
- Description: Kite auth test MCP
- Instructions summary: Use the login tool to check authentication.

## Tools
- `login`: Return the login link when authentication is required.

## Resources
- None
