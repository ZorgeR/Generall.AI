# AGENTS.md

Agent instructions for this repository live in [CLAUDE.md](./CLAUDE.md). Read that file first: it
covers the startup monkey-patching that changes which tools run in the Docker sandbox, the tool
contract and wiring steps, the on-disk memory layout under `data/`, the user-settings keys, the
models in use, and the known pitfalls. Keep both files in sync only by editing CLAUDE.md; this file
is a pointer for tools that look for AGENTS.md.
