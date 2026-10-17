# Design notes: v3 feature wave (2026-10)

Status: in progress. Written before implementation as the contract between the work packages
below; updated as they land. Read `CLAUDE.md` first; this file only adds what the wave changes.

## Scope

| # | Package | Summary |
|---|---------|---------|
| F1 | reply/forward context | the quoted message and the forward origin reach the agent |
| F2 | error recovery | no dead ends: Retry on error, auto-recovery inside the loop |
| F3 | mode | `mode.type` = auto / speed / intelligence; streaming on by default |
| F4 | /usage | the user sees their own usage |
| F5 | timezone + reminders UI | local time everywhere, create/edit/snooze from the UI |
| F6 | settings UX | Fast / Balanced / Deep presets, friendly names, Advanced toggle |
| F7 | media to the agent | real image and PDF blocks, VisionTools, more extensions |
| F8 | profile & notes | agent-managed user profile, `/profile` |
| F9 | memory_search fix | host search over summaries, profile and transcript |
| F10 | web reading | trafilatura extraction + sandboxed headless Chromium |
| F11 | proactive agents | scheduled tasks that stay quiet when there is nothing new |
| F12 | MCP client | per-user and admin MCP servers, `/mcp` |
| F13 | documents out | `create_document` (pdf / docx / xlsx / md / html) in the sandbox |
| F14 | embeddings | text-embedding-3-small, cosine, auto-reindex |
| F15 | /help | command menu, `/help`, richer `/start` |

## Shared hub changes (scaffolding, done first)

These land before the packages fan out so that every package builds on the same seams.

### Provider registry (`agents/main.py`)

`AgentAnthropic` keeps its named slots (`file_ops`, `search_tools`, ... `subagents`) for the
existing providers and gains a generic list for new ones:

```python
agent.register_provider(provider, blocking=False)   # async execute_tool -> awaited
agent.register_provider(provider, blocking=True)    # sync execute_tool -> asyncio.to_thread
```

`get_tools_schema()` appends the extra providers' schemas after `subagents` (deterministic
order = registration order). `execute_tool()` falls through to them after the named slots.
`SubagentTools._child` copies `extra_providers` to the child, so subagents get the new tools
too (minus `run_subagent`). A new package adds ONE line in `ChainOfThoughtAgent.__init__`:

```python
self.agent.register_provider(VisionTools(user_id, sender))
```

Tool names must stay globally unique; `tool_integrator.py` must never name a new class or
method (see `tests/test_image_tools.py` for the guard pattern).

### Prompts as content blocks (`agents/content.py`, new)

`run_turn(prompt=...)` and `ChainOfThoughtAgent.generate_response(question)` accept either a
string or a list of blocks. Handlers build blocks with the helpers in `agents/content.py`:

```python
[{"type": "text", "text": "User sent a photo with the caption: ..."},
 content.image_ref("data/<uid>/images/image_x.jpg"),
 content.document_ref("data/<uid>/documents/report.pdf", title="report.pdf")]
```

* `media_ref` blocks (`{"type": "media_ref", "kind": "image"|"document", "path", "title"}`) are
  the *stored* form: small, JSON-safe, and what the turn record keeps for Retry.
* `content.materialize(blocks)` (blocking, run in a thread) turns them into API blocks:
  base64 `image` blocks (media type from the extension; HEIC must already be JPEG) and
  base64 `document` blocks (PDF only; anything else must be extracted to text by the caller).
  The transcript stores the materialized blocks: exactly what was sent (append-only rule).
* `content.prompt_text(blocks)` is the text view: text blocks joined, media as
  `[image: <path>]` / `[document: <title>]`. Used for the classifier, summaries, FAISS, the
  legacy path, the turn record and logs.
* `transcript.estimate_tokens` counts an image block as ~1600 tokens and a document block by
  its size, never the base64 as text (that would trigger pruning on every photo).
* F7 adds `transcript.clear_old_media` (old media blocks -> a text marker naming the file, so
  the agent can `view_image` it again) inside `prune`, next to `clear_old_tool_results`.

The timestamp prefix (`Message received time in UTC+0: ...`) goes on the first text block.
A turn whose prompt has media skips the Haiku quick path (the full agent sees the picture).

### Tool results as content blocks

A provider's `execute_tool` may return a `str` (as before) or a `list` of content blocks
(`text` and `image` blocks). `run_tool_batch` passes a list through as the `tool_result`
content, caps only its text blocks, and shows `[image]` in the trace. `view_image` (F7)
returns `[{"type": "image", ...}, {"type": "text", "text": "images/x.jpg, 1024x768"}]`.

### Settings keys (`bot/settings.py`)

| Category | Keys | Consumer |
|---|---|---|
| `mode` | `type` ("auto"; also speed, intelligence) | F3: classifier gating and model/thinking choice |
| `timezone` | `name` ("" = unknown, times shown in UTC) | F5: context block, reminders, `schedule_reminder` |
| `profile` | `enabled` (true) | F8: profile injection and tools |

`SETTINGS_PRESETS` (F6) live in `bot/settings.py` too: a preset is a dict of (category, key)
-> value applied through `apply_setting`-equivalent setters.

### Routers (`bot/handlers/__init__.py`)

Three new modules on the `ui` router (auth, no quota): `usage_ui.py` (F4), `profile_ui.py`
(F8), `mcp_ui.py` (F12). Scaffolding adds them as empty routers so the package work never
touches `__init__.py`. Callback prefixes reserved: `usage_`, `profile_`, `mcp_`, `tz_`
(timezone picker, F5), `rem_` is NOT free (`reminder_`/`reminders_` exist). Every family
gets a row in `tests/test_cross_features.py::ROUTING`.

### Turn records

`TurnRecord.prompt` stays the text view; `prompt_blocks` (list | None) keeps the media refs,
and Retry re-runs `prompt_blocks or prompt`. F2 adds `error: str | None` so a failed turn can
be retried from its status message.

## Package designs

### F1 reply / forward context (handlers/messages.py)

`messages.describe_reference(message)` builds a text preamble for every content handler:

```
[Replying to a message from <name or "you (the assistant)"> sent <local or UTC time>:]
<quoted text or caption, up to 1500 chars> [+ "(a photo)", "(a document: name.pdf)" ...]
[Forwarded from <user name | chat title | channel title> (<date>)]
```

Only the text of the quoted message is included (no media download of the quoted message in
this wave). A reply to the bot's own message says so, so the agent knows which answer is meant.
The preamble is the first text block, before the user's own text.

### F2 error recovery

* `run_turn` failure: a `TurnRecord` with `error=<class>: <message>` and the prompt is saved;
  the status becomes "❌ <plain-language reason>" with a 🔁 Retry action row (rich button row
  or inline keyboard). Plain-language reasons: rate limit / overloaded, provider down, request
  too large (context, file), network. Unknown errors keep the Trace ID.
* Inside `AgentAnthropic._stream_message`: `anthropic.RateLimitError`, `APIStatusError` with
  status >= 500 or `overloaded_error`, and `APIConnectionError` are retried with backoff
  (3 attempts: 2 s, 5 s, 12 s); a `BadRequestError` naming thinking blocks / signatures /
  prefix strips the thinking blocks from the request messages and retries once; a
  `BadRequestError` about the request size is not retried (it is reported). Each recovery
  is a status line ("Retrying after a rate limit (2/3)").
* Tool exceptions already become `is_error` tool_results; unchanged.

### F3 mode + F6 settings UX

`mode.type`:

| mode | classifier | model for the loop | thinking | notes |
|---|---|---|---|---|
| auto (default) | Haiku classifier decides quick answer vs full agent (today's behaviour) | Sonnet | user's `thinking` setting | |
| speed | no classifier | fast model (Haiku 5.5) runs the full tool loop | light | fast-model calls take no `effort`/`thinking` params (see models.py) |
| intelligence | no classifier | Sonnet | deep (effort `ANTHROPIC_EFFORT`), regardless of the `thinking` toggle | |

Presets (`SETTINGS_PRESETS` in `bot/settings.py`): Fast = mode speed, thinking off, tools
max_iteration 10; Balanced = mode auto, thinking on, max_iteration 20 (the defaults); Deep =
mode intelligence, thinking on, max_iteration 40, judge off. The settings page gets a first
row "⚡ Fast · ⚖️ Balanced · 🧠 Deep" (the matching preset is marked) and an "Advanced"
toggle (page state, stored per user in settings `ui.advanced`) that hides the power settings
(transcript sizes, dialog/reasoning/summaries, critique, judge, semantic max, system prompt,
image variants/metadata). Friendly names replace the internal ones on the page ("Quick answers
vs. deep work" instead of "mode"). `STREAMING_ENABLED` defaults to true in `bot/config.py`
and `agents/main.py` (both read it) and `.env.example`.

### F4 /usage

`/usage` (ui router) shows, for the last 30 days: actions used / limit (or "unlimited"),
turns, tool calls, tokens (input / output / cached share), estimated cost, media described,
plus active reminders and the workspace size (files, MB). Data from `stats.py`
(`usage_events`, `stats_events`) via new helpers `get_user_usage_summary(user_id, days)` and
`get_user_workspace_size`. Legacy-Markdown text via `bot.ui.answer_md`, monospace lines like
`/stats`. Admin sees the same for themselves.

### F5 timezone + reminders UI, F11 proactive agents

* Timezone: `timezone.name` (IANA). Set via `/timezone <Name>` (validated with `zoneinfo`),
  via the `set_user_timezone` tool (the agent calls it when the user mentions their city or
  zone; the system prompt tells it to ask once when unknown), or by sharing a location
  (`timezonefinder` is optional: if not installed, the bot asks for the zone name). The
  `<context>` block shows "Local time (<zone>): ..." next to UTC. `schedule_reminder` gets a
  `timezone` parameter defaulting to the user's; its `reminder_time` may be local time.
* `/reminders`: list in local time, "➕ New" (asks the user to type the reminder in plain
  language, which the agent turns into `schedule_reminder`), per-reminder page with
  ⏰ +1h / +1d, ✏️ edit text (type a new text), ⏸ / ▶️, 🗑. A fired user reminder carries
  snooze buttons (10 min, 1 h, tomorrow 09:00 local).
* Proactive tasks: agent reminders run with a framing prompt ("This is a scheduled task you
  set up earlier: <text>. It is <local time>. ...") and may answer with the marker
  `<no_update/>` alone when there is nothing worth telling the user; then nothing is sent and
  the record stores the marker as `agent_response`, `last_result` = quiet | answered and a
  `quiet_runs` counter. Snoozing a periodic reminder adds a one-off copy (`snoozed_from`) instead
  of moving the series. Periodic reminders gain `weekdays` (list of 0-6) for
  "every weekday at 9". `/reminders` separates "⏰ Reminders" and "🤖 Tasks".

### F7 media to the agent

* `_run_images`: no pre-description. The prompt is text (caption, paths, "to show one inline
  write ![caption](images/<name>)") + one `image_ref` per photo. Images are prepared as JPEG
  ≤ `MAX_IMAGE_RESOLUTION_VISION` as today (HEIC converted), saved under `images/`.
* `_run_document`: PDF -> `document_ref` (title = file name) when ≤ 20 MB and ≤ 100 pages
  (pypdf page count; otherwise the extracted text as today); txt/md/code -> text block with
  the content (capped, the rest readable with `read_file`); docx/xlsx/json -> extracted text as
  today. New image extensions png/webp/gif/bmp are accepted as documents and routed to the
  image job (converted to JPEG when needed).
* `VisionTools` (`agents/vision_tools.py`, host, unpatched):
  `view_image(path)` -> image block + text (dimensions); `describe_image(path, question,
  model="anthropic"|"openai")` -> text from `bot.media.describe_image_*`; `read_document(path,
  question?)` -> PDF as a document block (same limits) or extracted text. Paths resolve with
  `agents.paths.resolve_under`.
* Transcript: `clear_old_media` in `prune`; `estimate_tokens` media-aware (scaffolding).
* The system prompt gains a line: pictures and documents arrive as real blocks; use the
  vision tools for files in the workspace or a second opinion.

### F8 profile & notes, F9 memory_search

* `bot/profile.py`: `data/<uid>/profile.md` with sections (`## About`, `## Preferences`,
  `## Notes`); entries are bullet lines. API: `load`, `entries(section)`, `add(section,
  text)`, `remove(section, text_or_index)`, `replace`, `clear`, `fingerprint`. Atomic writes,
  per-user lock, 8 KB cap (oldest notes rotated out with a warning in the tool result).
* `agents/profile_tools.py`: `user_profile_read`, `user_profile_add(section, text)`,
  `user_profile_remove(section, text)`, `user_profile_replace(section, text)`. The system
  prompt tells the agent to save durable facts about the user (name, city, timezone, language,
  preferences, projects) and to honour "forget that".
* Injection: `_request_context` appends a `<profile>` block (the whole file) to the stored
  context when its fingerprint differs from the last `<profile fingerprint="...">` block in the
  transcript (`agents/profile_tools.last_shown_profile_fingerprint`), so an unchanged profile is
  not repeated; the quick text-only path gets it on every turn (it never sees earlier context blocks).
* `/profile`: shows the profile, buttons "🗑 Forget everything" (confirm) and a hint: "tell me
  'remember that ...' or 'forget ...'".
* `memory_search` (host): case-insensitive substring/regex search over conversation summaries,
  questions and answers (`conversations/*.json`), the profile and the transcript text view;
  returns up to 20 hits as JSON lines (file, timestamp, topic, snippet). Removed from
  `patch_search_tools` in `tool_integrator.py`.

### F10 web reading, F13 documents out (sandbox tools)

* `download_webpage` (host): `trafilatura.extract` (main content, tables as text) with the
  BeautifulSoup text as fallback; same save-to-file behaviour.
* `browse_webpage(url, wait_ms?, screenshot?, max_chars?)` (new provider `agents/web_tools.py`, blocking):
  runs `app/secure_container/scripts/browse.py` inside the sandbox with network on
  (`ContainerManager.run_script(user_id, script_name, args, timeout, network_enabled=True)`,
  a new helper that copies a bundled script into the workspace under a unique temp name and
  runs `python <file> <args>`). The script uses Playwright + Chromium to load the page, waits
  for network idle, extracts readable text (trafilatura on the rendered HTML), prints JSON
  (title, text, links, final_url) and optionally saves `downloads/screenshot_<hex>.png`. The
  tool returns the text (capped) and the screenshot path (viewable with `view_image`).
* `create_document(format, title, content, filename?)` (new provider
  `agents/document_tools.py`, blocking): content is Markdown (tables allowed); `md`/`html`
  render on the host (markdown -> HTML with a clean CSS); `pdf`, `docx`, `xlsx` render in the
  sandbox via `scripts/render_document.py` (HTML -> PDF with Chromium `page.pdf()`, docx with
  python-docx, xlsx with openpyxl from the Markdown tables / CSV blocks). Output under
  `documents/<filename>`; the tool sends the file to the user (like ImageTools) and returns
  the path.
* Sandbox image: tagged by the hash of its build context (`container_manager.image_build_fingerprint`
  over Dockerfile, entrypoint.sh, requirements.txt), so a changed sandbox build is rebuilt on the
  next startup by itself; `secure-container:latest` is re-pointed and older content tags removed.
  `playwright`, `trafilatura`, `markdown`, `python-docx`, `openpyxl` in
  `secure_container/requirements.txt`; `playwright install --with-deps chromium` in the
  Dockerfile (as root, before `USER runner`, with `PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`
  readable by runner). Container memory for browser runs: 1 GB (`run_script(mem_limit=...)`).

### F12 MCP client

* Config: admin servers in `MCP_SERVERS_FILE` (default `data/mcp_servers.json`, list of
  `{name, url, transport: "http"|"sse", headers: {...}, enabled}`), per-user servers in
  `data/<uid>/mcp.json` (same shape plus `tools` cache and `updated`).
* `/mcp` (ui router): list servers with tool counts and ✅/⬜ toggles, `/mcp add <name> <url>
  [token]` (token stored as `Authorization: Bearer`), `/mcp remove <name>`, `/mcp refresh`.
  Adding connects once, lists tools, caches `{name, description, input_schema}`.
* `agents/mcp_tools.py` (`MCPTools`, async provider; `mcp>=2.0`, the 1.x branches are untested fallbacks): tools exposed as `mcp_<server>_<tool>`
  (sanitized to `^[a-zA-Z0-9_-]{1,64}$`), schemas from the cache (no network at turn start).
  `execute_tool` opens a session (`mcp.client.streamable_http` or `sse_client`), calls the
  tool, converts the result with `anthropic.lib.tools.mcp` helpers where possible (text and
  image content -> blocks), closes the session. Errors -> "Error: ..." strings. Timeout 60 s.
* Changing the server set changes the tool list -> prefix fingerprint -> stored thinking is
  stripped once (documented).

### F14 embeddings

`EMBEDDING_MODEL` default `text-embedding-3-small` (1536 dims, same as ada-002). The index
becomes `IndexFlatIP` over L2-normalized vectors (cosine); `metadata.json` gains `model` and
`version`. On load, a metadata model different from the configured one (or a legacy index
without a tag) triggers a rebuild from `conversations/*.json` (question + answer + summary
embedded as one text, timestamp and topic stored) in a worker thread; until it finishes
`search_conversations` returns no hits (vectors of another model are not comparable). Search returns `score` and skips near-duplicates (same question).

### F15 /help, command menu, /start

`bot/handlers/commands.py`: `/help` lists what the bot does, the commands and a few example
prompts (buttons that send the example through the follow-up mechanism are a later step);
`/start` for an authorized user shows the short version plus "Try: ..." examples. `app.py
on_startup` calls `bot.set_my_commands([...])` with: start, help, settings, usage, reminders,
profile, timezone, voice, mcp, cancel; admin-only commands are not listed.

## Order of work

1. Scaffolding (this document, registry, blocks, settings keys, routers, tests).
2. Wave 1 in parallel: F1+F15, F4, F10+F13, F14.
3. Wave 2 in parallel: F7, F8+F9, F12, F3+F6.
4. Wave 3: F5+F11, F2.
5. Review, CLAUDE.md, README, .env.example, tests, push.

Each package: unit tests next to the existing ones, `pytest` green, the import smoke check
(`python -c "import bot.app, agents.main, models"` from `app/` with dummy keys) passing, and a
short CLAUDE.md delta reported back for the docs step.
