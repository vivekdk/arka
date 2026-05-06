For Kite:
- If a Kite tool returns a login or authorization link, or otherwise indicates that the user must authenticate before work can continue, stop immediately.
- Give the user the login link in the response, tell them to log in first, and end the turn.
- Do not continue to holdings, pricing, or portfolio computation until the user has completed login.

For MCPs:
If MCP is related to a database and you have to execute database queries, you can do joins and execute a single query instead of multiple single queries and combining them later

## Agent skills

### Issue tracker

Issues and PRDs for this repo live in Linear, in workspace `arka-agent`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default triage label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a multi-context repo. Use `CONTEXT-MAP.md` at the root to discover per-context `CONTEXT.md` files and context-scoped ADRs. See `docs/agents/domain.md`.
