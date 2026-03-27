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
    count: 2
  resources:
    supported: true
    count: 1
tools:
- name: run-sql
  title: Run SQL
  description: Execute a SQL query
- name: preview_leads
  title: Preview Leads
  description: Preview sample leads rows
resources:
- uri: crm://dashboards/main
  name: main_dashboard
  title: Main Dashboard
  description: Primary dashboard
  mime_type: application/json
---

# MCP Minimal: fake

## Server
- Logical name: `fake`
- Protocol name: `fake-runtime-server`
- Title: Fake Runtime
- Version: `1.0.0`
- Description: Test MCP
- Instructions summary: Use the selected capability only.

## Tools
- `run-sql`: Execute a SQL query
- `preview_leads`: Preview sample leads rows

## Resources
- `crm://dashboards/main`: Primary dashboard
