---
schema_version: 1
server:
  logical_name: fake
  protocol_name: fake-runtime-server
  title: Fake Runtime
  version: 1.0.0
  description: Test MCP
  instructions_summary: Use the selected capability only.
capability_families:
  tools:
    supported: true
    count: 3
  resources:
    supported: true
    count: 1
tools:
- name: run-sql
  title: Run SQL
  description: Execute a SQL query
  input_schema:
    properties:
      query:
        type: string
    type: object
- name: fail-tool
  title: Fail Tool
  description: Return an MCP-level error payload
  input_schema:
    type: object
- name: preview_leads
  title: Preview Leads
  description: Preview sample leads rows
  input_schema:
    properties:
      limit:
        type: integer
    type: object
resources:
- uri: crm://dashboards/main
  name: main_dashboard
  title: Main Dashboard
  description: Primary dashboard
  mime_type: application/json
  annotations: null
extensions: {}
---

# MCP Full: fake

## Server
- Logical name: `fake`
- Protocol name: `fake-runtime-server`
- Title: Fake Runtime
- Version: `1.0.0`
- Description: Test MCP
- Instructions summary: Use the selected capability only.

## Tools
- `run-sql`: Execute a SQL query
- `fail-tool`: Return an MCP-level error payload
- `preview_leads`: Preview sample leads rows

## Resources
- `crm://dashboards/main`: Primary dashboard
