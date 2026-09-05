# Agent architecture roadmap

Status: PR A shipped (#8); PR B implemented (transcript, caching, subagents), see the notes at the end. Written 2026-09-05 after the aiogram port (#4),
rich messages (#5, #6) and the model consolidation (#7).

## Why

The turn-level loop is the standard Claude tool loop and is sound. The cross-turn memory is
home-grown and now works against the bot:

- Tool calls and results never survive a turn. The next turn is rebuilt from summaries, a list of
  question/answer strings and the last turn's text-first messages, so "open the page you fetched a
  minute ago" fetches it again.
- The model is shown an invented conversation (summaries as user/assistant pairs, a fake intro
  exchange), and judge/critique prompts can be replayed next turn as if the user wrote them.
- Nothing is cached: the system prompt embeds the current time and the semantic-search hits, so
  the prefix changes every turn, and the full tool schema plus the growing transcript are re-sent
  on every iteration of the loop. A 20-tool turn pays for the prefix 20 times.
- Tools requested together run one after another.

Forum topics already give a fresh context per topic (the short-term files are prefixed with the
topic id), so no `/new` command is needed; the transcript below keeps that behaviour.

## PR A (small): tool trace and parallel tool execution

**Tool trace.** `agents/trace.py` records every tool call of a turn (name, argument summary,
start, duration, ok/error, result preview). `run_turn` renders it into the status message on every
status update, so the user watches the tools run, like tool cards in Claude Code:

```
💭 Thinking...
- - - -
📝 Step: executing-tools ...
🔧 Tools (3, 12 s)
✅ search_web  aiogram rich messages · 2.1s
⏳ download_webpage  https://core.telegram.org/bots/api…
❌ run_command  ls /nope · 0.4s
```

When the answer is delivered in rich mode and tools were used, the status message is kept as a
one-line record above the answer (`🔧 7 tool calls in 41 s: search_web ×3, …`) instead of being
deleted. With no tool calls it is deleted as before; legacy mode is unchanged (the answer replaces
the status). The busy notice shows the running tool through the same progress text.

**Parallel tool execution.** All `tool_use` blocks of one assistant message run concurrently
(`asyncio.gather`) and their results go back in one user message in the original order, which is
what the API expects. A tool that raises returns a `tool_result` with `is_error: true` instead of
aborting the turn. Sandbox slots and the thread pool already bound concurrency.

Follow-up once PR B lands: an expandable "tool calls" section inside the rich answer (rich
messages have a native collapsible block).

## PR B: native transcript, prompt caching, subagents

### B1. One real transcript per chat/topic

Replace the summaries + dialog-history + reasoning-context triad in the prompt with the actual
Messages API conversation, replayed as is.

- **Storage**: `data/<uid>/transcripts/[topic_<id>_]transcript.json`:
  `{version, model, created, updated, messages:[...]}` where `messages` are exact API blocks
  (`text`, `tool_use`, `tool_result`, `thinking` with signature, server `compaction` blocks).
  Written atomically under the per-user lock; one file per chat or topic, so a new topic starts
  empty.
- **Turn**: load → append the user message → run the loop, appending assistant content and tool
  results exactly as sent → append the final assistant text → save. Judge/critique injections are
  tagged and dropped before saving; they are not user turns.
- **Size control**, in this order: (1) cap each `tool_result` stored in the transcript (large
  outputs are truncated with a note, the model already saw the full text this turn); (2) clear the
  content of tool results older than the last N turns ("[result cleared]"); (3) server-side
  compaction (`compact-2026-01-12` beta, supported on Sonnet 5): pass `response.content` back
  including compaction blocks; (4) client-side fallback when compaction is unavailable: summarize
  the oldest half with the fast model into one labelled block.
- **Long-term memory stays**: conversation summaries + FAISS are still written per turn and used
  for retrieval, but injected as one labelled `<memory>` block in the newest user message, never
  as fake dialogue.
- **Settings**: `dialog_history` and `reasoning_context` are retired (files kept on disk, ignored);
  `summarization_history.size` becomes "recent summaries in the memory block"; new `transcript`
  category: `max_context_tokens`, `keep_tool_results_turns`. `/settings` UI updated; defaults
  backfilled as usual.
- **Migration**: on a user's first turn without a transcript, seed it from the existing dialog
  history so recent context is not lost.
- **Thinking blocks** are replayed unchanged (same model). If the API ever rejects stale
  signatures, drop the thinking blocks and retry once.
- **Reasoning file**: generated from this turn's slice of the transcript.

### B2. Prompt caching

- Request layout: `tools` (deterministic order) → `system` as content blocks with the static
  prompt first and one explicit `cache_control` breakpoint on it (1-hour TTL) → `messages` = the
  transcript, with top-level automatic caching for the growing tail.
- Volatile content (current time, memory hits, formatting flags) moves out of the system prompt
  into a `<context>` block at the start of the **newest** user message, after the cached prefix.
- Every response logs `usage.cache_read_input_tokens` / `cache_creation_input_tokens`; the status
  usage line can show the cache hit rate.
- Expected effect: within a turn every loop iteration reuses the prefix; across turns the
  transcript prefix is reused. Tool schemas must not change between iterations (they do not).

### B3. Subagents as a tool

- Tool `run_subagent(task, tools=[...] optional, model="main"|"fast", max_iterations)`: runs a
  fresh `AgentAnthropic` with the same providers (or the named subset), an empty transcript that
  is **not** persisted, the parent's sender for user-facing sends, and the parent's trace (nested
  entries). Returns the subagent's final text as the tool result; files it creates are in the
  same workspace.
- Depth 1 only (a subagent has no `run_subagent` tool); its iterations count against the parent's
  budget; it shares the user's queue slot, sandbox slots and cancellation.
- Parallel subagents come for free through PR A's parallel tool execution.
- Status rendering: `🤖 run_subagent  <task>` with the subagent's tool lines indented below it.

### B4. Small follow-ups

- Task budget (beta) on long agentic turns instead of the bare iteration counter.
- Expandable tool-call section in the rich answer.
- Optional: Anthropic memory tool for model-managed notes under the user's folder.

## Tests and rollout

- PR A: unit tests for the trace and for the batch runner (ordering, timing, error isolation).
- PR B: transcript store (append/load/prune/compaction block round trip), request builder (cache
  breakpoints, volatile block placement), subagent tool (depth limit, budget accounting), plus the
  existing suite. Staging checks: a multi-turn task that reuses a fetched page, cache hit rate on
  the second iteration of a turn, a subagent research task, a 100-turn topic to exercise pruning.
- Rollout: PR B ships behind the transcript setting defaulting to on with the migration seed;
  turning it off restores the current behaviour for one release, after which the old code is removed.

## Risks

- Larger contexts per turn cost more input tokens; caching and pruning are what keep this in
  check, so B1 and B2 ship together.
- Tool results can be huge (web pages, command output); the per-result cap in the transcript is
  mandatory.
- Server-side compaction is a beta; the client-side fallback must be tested independently.

## Implementation notes (PR B)

- Shipped: `agents/transcript.py` (store, cap/clear/summarize pruning, seeding), transcript mode in
  `ChainOfThoughtAgent._generate_with_transcript`, the cache-friendly request layout in
  `AgentAnthropic` (`_request_messages`, top-level automatic `cache_control`, explicit breakpoint on the
  system block), token usage in the status line, `agents/subagent.py` (`run_subagent`), the `transcript`
  setting with its `/settings` menu, and the migration seed from `dialog_history.json`.
- Deferred: server-side compaction (beta) is not used; client-side pruning + Haiku summarization covers it.
  The expandable tool-call section inside the rich answer and task budgets remain follow-ups.
- Legacy mode (`transcript.enabled` off) keeps the old context assembly untouched for one release.
