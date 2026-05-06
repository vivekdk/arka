# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT-MAP.md` at the repo root if it exists. It should point to one `CONTEXT.md` per context. Read each one relevant to the topic.
- `docs/adr/` for system-wide decisions.
- `src/<context>/docs/adr/` for context-scoped decisions where present.

If any of these files don't exist, proceed silently. Don't flag their absence; don't suggest creating them upfront.

## File structure

This repo is treated as multi-context:

- Root `CONTEXT-MAP.md` points to context-specific `CONTEXT.md` files.
- Root `docs/adr/` is for cross-cutting or system-wide decisions.
- Context-specific ADRs may live under `src/<context>/docs/adr/`.

## Use the glossary's vocabulary

When naming domain concepts in issues, proposals, tests, or implementation notes, use the terminology defined in the relevant `CONTEXT.md`.

## Flag ADR conflicts

If a proposal or implementation direction conflicts with an existing ADR, surface the conflict explicitly rather than silently overriding it.
