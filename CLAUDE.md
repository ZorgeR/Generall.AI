# CLAUDE.md — GALL.AI / GenerALL.AI

Guidance for AI coding agents working in this repository. Everything below was
derived from reading the code, not the README; where the two disagree, the code
wins and the discrepancy is listed under "README vs code".

## What this is

A single-process Python Telegram bot (aiogram 3, asyncio) that wraps a Claude
tool-use agent. Users send text / voice / photos / video / audio / documents in
Telegram; the bot builds a prompt of content blocks (text, plus the photos and PDFs
themselves as image/document blocks, with the replied-to or forwarded message quoted
in front), runs a per-message agent loop with 44 built-in tools (files, web search and
a headless browser, code execution in a Docker sandbox, image/video generation, documents
out as pdf/docx/xlsx, reminders and scheduled tasks in the user's time zone, SMS, S3, TTS,
button questions, a user profile) plus
the tools of the user's MCP servers, persists multi-layer memory per user on disk
(transcript, summaries + embeddings, profile), and replies in Telegram. Claude drives
the tool loop; OpenAI (`gpt-6.1-sol`) is only consulted by the optional critique step,
which can force a rewrite, and by the `describe_image` second opinion. OpenAI, Google
Gemini/Veo, Perplexity, Tavily, ElevenLabs, Twilio and Whisper otherwise serve specific
side tasks.

Every chat has its own job queue: messages are processed strictly in order per
user, each as its own turn, and users never wait for each other. See "Queues".

There is a pytest suite (`tests/`, run with `pytest` from the repo root) covering the queue,
settings, presets and the /settings page, auth, reminder store, text splitting, the model settings
module, the /stats views and /usage, the Stop buttons, the turn buttons, `ask_user_choice`, the
reply/forward preamble and /help, content blocks and media prompts, the vision tools, the profile
and memory_search, the MCP client and /mcp, the mode setting, the web/document tools and the
sandbox image tag, embeddings, time zones and the /reminders UI, scheduled tasks, the agent's retry
loop and the error path (`test_timezones.py`, `test_reminders_ui.py`, `test_proactive_tasks.py`,
`test_stream_retry.py`, `test_error_recovery.py`), whole turns on a fake client (`test_mode.py`: the tools and
`<capabilities>` notes actually sent, a photo reaching the loop as an image block), and how these features meet
(`tests/test_cross_features.py`).
There is no CI, linter, formatter or type checking. Validate changes with the tests and
the import smoke check at the end of this file.

## Repository layout

```
app/                        Python package root; the bot runs with cwd=app/ (Dockerfile CMD)
  main_bot.py               Entrypoint (≈60 lines): load .env, validate config, init sandbox, import agents, run bot.app
  models.py                 THE place for model names + request options (effort / reasoning_effort), each with an
                            env override; imported as `models` by bot/* and agents/*. No import-time side effects.
  bot/                      Telegram layer (aiogram). No import-time side effects; never imports agents/stats at import.
    app.py                  create_bot (local Bot API support), create_dispatcher, startup/shutdown hooks, run()
    config.py               Config dataclass from env (`config` singleton), validate()
    runtime.py              process-wide QueueManager instance + background task list
    queue.py                QueueManager / Job / JobContext: per-user FIFO workers, deadline, global cap, /cancel
                            (cancel) and the Stop buttons (cancel_current: the running job only, by Job.id)
    sender.py               ChatSender: the one object handlers AND tools use to send text/media/reactions;
                            send_markdown renders LLM answers as rich messages with tiered fallback
    rich.py                 rich-message helpers: split/convert (telegramify-markdown), inline media extraction,
                            sticky "unsupported" flag, sticky "buttons inside rich messages" flag, is_chat_error
    agent_runner.py         run_turn(prompt: str | content blocks): builds ChainOfThoughtAgent, status updates (Stop
                            button, "recovering" lines from the agent's retries), streaming, voice reply, answer, turn
                            record + buttons, partial answer on Stop, token accounting (record_usage), quiet_marker
                            (scheduled tasks: an answer that is only the marker sends nothing; with quiet_notice the status
                            becomes that line instead, "📭 Nothing new to report." on a Retry); a failed turn ends as
                            "❌ <friendly_error(e)> (trace <uuid>)" + 🔁 Retry over an error TurnRecord (ERROR_* texts)
    turn_actions.py         buttons of a finished turn: <follow_ups> parsing, TurnRecord store (data/<uid>/turns/; prompt
                            text view + prompt_blocks media refs for Retry; error = a failed turn, Retry only; quiet_marker = a scheduled task's
                            marker, Retry runs with it again), turn_*
                            callback data, action rows (Continue/Retry/Listen/Reasoning) and answer keyboard
    message_context.py      reply / forward preamble: describe_reference(message, bot_id) and with_reference(prompt,
                            message, bot_id), pure (metadata + quoted text only, nothing downloaded); every content job
                            puts it in front of its prompt
    profile.py              ProfileStore (`profile_store` singleton): data/<uid>/profile.md, sections About/Preferences/
                            Notes (+ any "## Title" the agent creates), "- " entries, dated Notes, 8 KB cap (oldest Notes
                            rotate out), atomic writes, per-user threading.Lock, fingerprint(text) = 12 hex; sync, callers
                            use to_thread
    mcp_store.py            MCP server records + cached tool lists: per-user data/<uid>/mcp.json, the admin file
                            (MCP_SERVERS_FILE) + its .cache.json; sync I/O, per-user lock, atomic writes, name/URL
                            validation, tokens never rendered back
    questions.py            ask_user_choice backend: in-memory pending questions (Future per question), ask_*
                            callback data, rich button rows / inline keyboard, timeout and cancel outcomes
    status.py               StatusMessage: rich status edited in place (progress blocks + Stop button → end-of-turn
                            summary with expandable tool calls, thinking, tokens/cost, action row; fail() = the error
                            text + Retry row through the same _final_edit fallbacks) with legacy-Markdown fallback
    media.py                Whisper transcription, picture preparation for the agent (full-res JPEG + downscaled
                            vision copy, image_for_vision), PDF limits (pdf_page_count / pdf_fits_as_block),
                            extract_document_text, the Claude/GPT describe_* fallbacks, video frames, TTS, ffmpeg setup
    streaming.py            throttled draft streaming (rich drafts + <tg-thinking> block, plain draft fallback);
                            draft_id = Job.id with Telegram's Stop (can_stop), follow-ups block hidden, partial_text()
    auth.py                 AuthStore (`auth` singleton): allow/block lists + invite codes in data/userlist.json
    limits.py               check_user_limits() (rolling 30-day action quota via stats.db)
    settings.py             DEFAULT_SETTINGS + UserSettings (single source of truth), MODE_TYPES, SETTINGS_PRESETS
                            (fast / balanced / deep) with apply_preset / current_preset
    timezones.py            the user's zone (settings timezone.name) and local-time helpers, stdlib only: valid_zone
                            (IANA names in any case, ALIASES of ~60 cities/countries consulted first), zone_of, format_local,
                            to_utc, COMMON_ZONES (the /timezone picker), zone_from_location (optional timezonefinder, lazy)
    ui.py                   answer_md/edit_md helpers: legacy Markdown with plain-text fallback
    jobs.py                 reminders scheduler loop (10 s): sends user reminders with snooze buttons, runs agent
                            reminders as scheduled tasks (task_prompt framing + QUIET_MARKER through run_turn, no
                            announcement; last_result / quiet_runs on the record)
    clients.py              lazily built OpenAI/Whisper/Anthropic clients + model name constants
    stats_report.py         /stats data: collect() → StatsReport (stats_tracker queries; blocking, run in to_thread)
                            and the legacy-Markdown text views (Unicode bars, daily sparkline, deltas vs previous 30 days,
                            monospace lines ≤ LINE_WIDTH so phones don't wrap them)
    stats_chart.py          /stats PNG dashboard from the same report: matplotlib object API only (Figure + Agg, never
                            pyplot, so worker-thread renders share no state), imported lazily; render in to_thread
    handlers/
      __init__.py           build_root_router(): public → admin → ui → chat, middleware per router
      middleware.py         AuthMiddleware(require_admin, check_limits) injects user_id / limit kwargs
      messages.py           text/voice/video/audio/photo(+albums)/document handlers (block prompts: reply/forward
                            preamble + text + image_ref/document_ref), busy notice, /cancel, Stop buttons (queue_stop_
                            callback, stopped_message_generation update), on_location (a shared location sets the
                            time zone; the agent never sees locations)
      commands.py           /start, /help, /invite, /listusers, /voice, /timezone (picker tz_set_<i>, name or city
                            alias, confirmation with the local time); COMMANDS = the one list of user commands
                            (the Telegram command menu via bot_commands() and the command section of /help)
      settings_ui.py        /settings: one rich page with buttons inside it (setp_ callbacks; presets row, Advanced
                            toggle, SECTIONS of Controls with friendly names, apply_setting shared with the keyboards),
                            else the settings_ inline keyboards
      usage_ui.py           /usage: the user's own last-30-day usage as ONE legacy-Markdown message (quota meter,
                            requests/answers/tool calls/media described, tokens + cached share + est. cost + 7-day cost
                            sparkline, active reminders, workspace files and MB); "🔄 Refresh" (usage_refresh) edits it
                            in place; SQLite, the directory walk and check_user_limits run in to_thread; `stats` is
                            imported lazily
      profile_ui.py         /profile: the profile as legacy Markdown, paged under Telegram's 4096-char limit (render_pages),
                            🔄 Refresh / 🗑 Forget everything (confirm step); profile_ callbacks
      mcp_ui.py             /mcp [add|remove|refresh|help] + mcp_ buttons; refresh_admin_servers() also runs once from
                            app.py:on_startup as background task "mcp-admin-refresh"
      turn_actions_ui.py    turn_ buttons: router (ui: Listen, Reasoning) + turn_router (chat, quota: Retry,
                            Continue, follow-ups; they echo the press and queue a turn like a typed message)
      questions_ui.py       ask_ buttons: resolve the pending question, edit it into its final state
      reminders_ui.py       /reminders inline keyboards: local times, "⏰ Reminders" / "🤖 Tasks" groups, detail page
                            with +1h / +1d / Edit text (the next text message, _pending_edits, 5 min) / Delete / Disable,
                            ➕ New hint, snooze handling for fired reminders (reminder_snooze_), legacy-Markdown escaping
                            of zone names
      stats_ui.py           admin /stats: dashboard photo + text view, users page, per-user view, limits, block;
                            display names cached 1 h (successful lookups only)
  reminders_store.py        RemindersStore (`reminders_store` singleton): per-user locked JSON read/modify/write;
                            calculate_next_trigger steps daily/weekly/monthly in the reminder's zone and skips days
                            outside `weekdays`; clean_weekdays, first_allowed_trigger, WEEKDAY_NAMES
  agents/
    main.py                 Agent core: AgentAnthropic (tool loop, named provider slots + register_provider list,
                            tool results as text or blocks, _stream_message = the streamed call with retries and the
                            thinking-mismatch strip) + ChainOfThoughtAgent (memory, prompts, mode, routing,
                            persistence). The 4 system prompts live here; providers add their own instructions via
                            prompt_notes (the <capabilities> block).
    content.py              prompt / tool-result content blocks: text_block, image_ref, document_ref (stored media_ref
                            form), materialize (base64 image/document API blocks), prompt_text (text view), strip_media,
                            result_preview / cap_blocks (list results), estimate_block_tokens
    file_ops.py             FileOperations: files under data/<uid>/, downloads (download_webpage = trafilatura main
                            content, BeautifulSoup fallback), S3, zip, send file
    search_tools.py         SearchTools: Tavily search, Perplexity deep_research, memory_search (host: conversation
                            files, profile.md and the transcript text view)
    vision_tools.py         VisionTools: view_image (image block + dimensions), describe_image (second opinion, Claude or
                            GPT), read_document (PDF as document block or extracted text); host, never sandboxed
    profile_tools.py        ProfileTools: user_profile_read/add/remove/replace over bot/profile.py, prompt_notes,
                            profile_block() and last_shown_profile_fingerprint() for the <context> block
    mcp_tools.py            MCPTools: the cached tools of the user's (and the admin's) MCP servers as mcp_<server>_<tool>,
                            discovery, one fresh connection per call, results as text or image blocks, URL secrets redacted
    web_tools.py            WebTools: browse_webpage (headless Chromium in the sandbox via ContainerManager.run_script),
                            get_container_manager() (process-wide, lazy) and last_json_line() for script output
    document_tools.py       DocumentTools: create_document (md/html rendered on the host with python-markdown; pdf/docx/
                            xlsx rendered in the sandbox by scripts/render_document.py), sends the file itself
    code_tools.py           CodeTools: execute_python (patched to run in Docker at startup)
    terminal_tools.py       TerminalTools: run_command (patched; gains run_shell_script/install_package)
    time_tools.py           TimeTools: timezone helpers
    sms_tools.py            SMSTools: Twilio send_sms
    image_tools.py          ImageTools: one generate_image tool (generate/edit/compose) over the
                            models.IMAGE_BACKENDS table, plus generate_story_images
    video_tools.py          VideoTools: Google Veo 3.1 text→video, image→video, extend, interpolate
    user_interactions.py    UserInteractions: mid-run Telegram messages, TTS voice, reactions, set_user_timezone,
                            schedule_reminder (local time, timezone, weekdays), send file from content,
                            ask_user_choice (via bot/questions.py); prompt_notes on zones and quiet scheduled tasks
    embeddings.py           ConversationEmbeddings: OpenAI text-embedding-3-small + FAISS IndexFlatIP (cosine) per user;
                            model-tagged metadata, stale detection, one background rebuild per user per process
    system_tools.py         SystemTools: apt/pip/service/network tools — NOT wired into the agent
    paths.py                resolve_under(): safe resolution of model-supplied paths inside data/<uid>;
                            regular_files(folder, pattern): host reads of a workspace folder that never follow
                            a link (O_NOFOLLOW folder and files, regular UTF-8 files only; memory_search, embeddings); ensure_real_dir(path, mode):
                            a workspace subfolder the bot prepares (downloads/, documents/) made a real directory
                            (a link or file there is unlinked) and chmod-ed through an O_NOFOLLOW descriptor
    trace.py                ToolTrace/ToolCall/TurnBudget: per-turn tool calls, token usage, shared tool budget
    transcript.py           TranscriptStore + size control: one real API transcript per chat/topic (transcript mode);
                            prune clears old tool results AND old media blocks (clear_old_media)
    subagent.py             SubagentTools: run_subagent tool (fresh AgentAnthropic, shared budget, nested trace; the
                            child inherits the registered providers)
    __init__.py             imports SystemTools; the package object is what the patcher needs
  secure_container/
    main.py                 initialize_secure_containers() / cleanup_containers()
    container_manager.py    ContainerManager: builds image "secure-container:<12-hex hash of the build context>" when
                            that tag is missing (re-points :latest, prunes older tags), runs one throwaway Docker
                            container per tool call, MAX_SANDBOX_CONTAINERS semaphore, run_script() for bundled scripts
    tool_integrator.py      ToolIntegrator: monkey-patches tool classes at startup (see below)
    secure_tool_wrapper.py  SecureToolWrapper: thin routing layer to ContainerManager
    scripts/                scripts that run INSIDE the sandbox (stdlib + the image's libs only; print ONE JSON line last):
      browse.py             Playwright Chromium page reader: url [--screenshot path] [--wait-ms] [--max-chars] →
                            {title, final_url, text, links, screenshot, error}; own 60 s deadline
      render_document.py    --format pdf|docx|xlsx --input --output [--title=] → {output, bytes, error}: HTML→PDF
                            (page.pdf), Markdown→DOCX (python-docx), Markdown→XLSX (openpyxl); own 75 s deadline
    Dockerfile, entrypoint.sh, requirements.txt   the sandbox image (python:3.12-slim + tools, Playwright Chromium at
                            /opt/pw-browsers, non-root user "runner", passwordless sudo for apt only)
  stats.py                  StatsTracker singleton: SQLite data/stats.db, per-user action limits, the /usage helpers
                            get_user_usage_summary(user_id, days=30, daily_days=7) and get_user_workspace_size(user_id)
                            (os.walk of data/<uid>, symlinks by their own size, never followed)
  image_utils.py            JPEG/HEIC/PNG/WebP/GIF/BMP helpers (IMAGE_EXTENSIONS, flatten_to_rgb; pillow-heif registered
                            on import)
  voice/                    VoiceManager: ElevenLabs voice ids (voices.json) + per-user choice (config.json)
  Dockerfile, requirements.txt   bot image (python:3.12-slim + git, docker.io, ffmpeg); timezonefinder is a commented,
                            optional line there (time zone from a shared location)
tests/                      pytest suite (pytest.ini sets pythonpath=app, asyncio_mode=auto)
.github/workflows/deploy.yml  manual production deploy (workflow_dispatch, environment PROD), see "Running"
deploy/                     render_env.py (PROD secrets+vars → .env, runs on the runner) and
                            remote_deploy.sh (git checkout, .env swap, compose build/up, health check; runs on the server)
requirements-dev.txt        app requirements + pytest, pytest-asyncio
docker-compose.yml          services: telegram-bot-api (local Bot API server) + bot
.env.example                template for .env (WORKSPACE_ROOT comes from docker-compose.yml; BROWSER_SERVICE_URL is unused)
docs/                       design notes: agent-roadmap.md (transcript / caching / subagents)
static/                     README images only
data/                       runtime state, git-ignored (see "On-disk data layout"); also the admin's MCP servers file
```

## Running

Docker (the intended way):

```bash
cp .env.example .env   # fill in keys
docker-compose up --build
```

Local (needs a reachable Docker daemon, ffmpeg on PATH, Python ≥ 3.12):

```bash
pip install -r app/requirements.txt
cd app && python main_bot.py      # cwd MUST be app/: all paths are relative ("./data", "temp_audio", ...)
                                  # and stats/voice/reminders_store/secure_container/image_utils are top-level modules
```

Deployment facts that are easy to get wrong:

- `docker-compose.yml` hard-codes `WORKSPACE_ROOT=/opt/generall.ai/Generall.AI`. This must be the
  **host** path of the repo checkout, because the bot container talks to the host Docker daemon
  (`/var/run/docker.sock`) and bind-mounts `$WORKSPACE_ROOT/data/<uid>` into each sandbox container.
  Cloning elsewhere without changing it makes every sandboxed tool fail or mount an empty dir.
- `.env.example` ships `TELEGRAM_USE_LOCAL_API=true`. For a plain local run without the
  `telegram-bot-api` sidecar set it to `false`, or the bot will point aiogram at `localhost:8081`.
- Rich messages (`sendRichMessage`, Bot API 10.1) need a Bot API server that knows the method. The
  compose sidecar is `aiogram/telegram-bot-api:latest`, so run `docker compose pull telegram-bot-api`
  on an old checkout; against an older server the bot logs one warning and falls back to MarkdownV2
  for the rest of the process (`bot/rich.py` flag). Old Telegram clients show rich messages as
  "unsupported message"; users turn them off per chat in `/settings` → Rich Messages.
- The `bot` service runs `privileged: true` with `network_mode: host`. Host networking is what lets it
  reach the `telegram-bot-api` sidecar at `localhost:8081`; the shared `telegram-bot-api-data` volume
  is mounted in both containers because local-mode `get_file` returns server-side absolute paths
  (aiogram's `TelegramAPIServer(is_local=True)` then reads them from disk).
- `./data` is bind-mounted to `/app/data` with `:z` (SELinux). `ContainerManager.ensure_user_directory`
  `chmod 777`s `data/<uid>` on every sandboxed call so the sandbox's non-root `runner` can write.
- `main_bot.py` fails fast: `Config.validate()` exits if `TELEGRAM_BOT_TOKEN` is missing or nobody
  could be authorized; `initialize_secure_containers()` exits if Docker is unavailable. Importing
  `agents.main` constructs `OpenAI(...)` and `genai.Client(...)` at module level, which raise when
  `OPENAI_API_KEY` or `GOOGLE_API_KEY` are unset (any non-empty value passes). `stats` creates
  `data/stats.db` on import. The `bot` package has no import-time side effects.
- Python ≥ 3.12 is required: `agents/main.py` uses nested same-quote f-strings (PEP 701).
- Polling mode only; `DROP_PENDING_UPDATES=true` (default) discards messages sent while the bot was
  down. The only scheduled work is the reminders loop every 10 s, plus a one-off refresh of the admin
  MCP servers' tool cache at startup (`mcp-admin-refresh`).
- The sandbox image is tagged by the hash of its build context (`secure-container:<12 hex>` over
  Dockerfile, entrypoint.sh, requirements.txt). A deploy that changes any of them builds the new tag on
  the next startup by itself (slow: apt layers plus Chromium); no manual `docker rmi`. Scripts under
  `secure_container/scripts/` are copied into the workspace at call time, so changing them needs no rebuild.
- Production deploys are a manual GitHub Actions run (`.github/workflows/deploy.yml`, Deploy button,
  `environment: PROD`) plus `deploy/`: `render_env.py` turns **every** PROD secret/variable into the
  server's `.env` (secrets win; `SERVER_*` are the SSH settings and are excluded), the workflow uploads
  it as `.env.new` and streams `remote_deploy.sh` over SSH (refuse on dirty tracked files, checkout the
  ref, swap `.env` keeping `.env.bak`, compose build/up, wait for "is running": `HEALTH_TIMEOUT` 90 s, or `SANDBOX_BUILD_TIMEOUT` 1200 s when the
  deploy changes the sandbox build files, which the bot rebuilds before it starts). The server's `.env` is
  therefore generated: change the PROD environment, not the file. Tests: `tests/test_render_env.py`.

## Startup order and the monkey-patching (read this first)

`main_bot.py` executes, in order:

1. `load_dotenv()`, `logging.basicConfig`, `Config.validate()`.
2. `initialize_secure_containers()` (`secure_container/main.py`): connects to Docker, builds
   `secure-container:<content hash>` from `app/secure_container/Dockerfile` when that tag is missing
   (`ContainerManager.image_build_fingerprint`; `:latest` is re-pointed and older content tags removed,
   best effort), then `ToolIntegrator.patch_all_tools(agents)` **monkey-patches the tool classes in place**:
   - `TerminalTools.run_command` → runs in a sandbox container; `run_shell_script` and
     `install_package` methods **and their tool schemas are added** to the class.
   - `CodeTools.execute_python` → sandbox; a `network_enabled` param is added to its schema.
   - `FileOperations`: only method names that exist on the class AND are in
     `SecureToolWrapper.SECURE_TOOL_METHODS` get patched. That is **only `create_directory` and
     `delete_file`**. `list_directory`, `read_text_file`, `create_text_file`, downloads, S3, zip all
     run on the host (i.e. inside the bot container) unsandboxed.
   - `SearchTools`: **not patched any more**. `patch_search_tools` only logs; `memory_search` runs on
     the host (bot process) over the conversation files, `profile.md` and the transcripts, which the old
     sandbox grep of `*.txt` never had. `SECURE_TOOL_METHODS` no longer lists it
     (`tests/test_memory_search.py` guards that `tool_integrator.py` never names it again).
   - `VisionTools`, `ProfileTools`, `MCPTools`, `WebTools`, `DocumentTools` (the v3 providers): **never
     patched**. They are registered through `AgentAnthropic.register_provider` instead of a named slot;
     the two that run anything risky (`browse_webpage`, pdf/docx/xlsx rendering) already do it in the sandbox
     via `ContainerManager.run_script`. Guard tests: `tests/test_vision_tools.py`, `test_mcp_tools.py`,
     `test_web_tools.py` assert `tool_integrator.py` never names these classes or methods.
   - `ImageTools`: **not patched at all** (it needs API access, not a sandbox). The patcher used to
     wrap `_generate_image` with a fixed `(prompt, size, quality, caption)` signature; when the
     consolidated tool happened to reuse that method name, every edit died on `unexpected keyword
     argument 'images'` and every other parameter was silently replaced. Do not add it back.
   - `ConversationEmbeddings`: wrapped with logging only.
   - `SystemTools`: fully patched, but the class is never given to the agent, so unreachable.
   `ContainerManager` is instantiated twice here (once kept for cleanup, once inside the wrapper).
3. **Only then** `import agents.main` — so every `ChainOfThoughtAgent` sees patched classes.
   `bot.agent_runner` also imports `agents.main` lazily, inside `run_turn`, for the same reason.
4. `bot.app.run()`: build `Bot` (local API session when configured), `Dispatcher`, include the root
   router, `delete_webhook(drop_pending_updates)`, `start_polling`. `on_startup` loads the auth store,
   configures ffmpeg for pydub, starts the reminders loop and the `mcp-admin-refresh` task
   (`mcp_ui.refresh_admin_servers`), and registers the command menu (`register_commands` →
   `set_my_commands(bot_commands())`; a refusal by an old local Bot API server is only logged);
   `on_shutdown` cancels background tasks and running jobs; `main_bot` then calls `cleanup_containers()`.

Consequences: the tool list the model actually sees is **not** what you read in `agents/*.py`
alone; the source of truth is source file + `tool_integrator.py`. When you add or rename a method on
a tool class, check `SECURE_TOOL_METHODS` and the `patch_*` functions, or you may silently move
execution between host and sandbox.

## Queues, isolation and cancellation

- `bot/queue.py` gives every chat a FIFO `asyncio.Queue` and a lazily started worker task. Handlers
  only validate, enqueue a `Job(user_id, label, run)` and return within milliseconds. Every job gets a
  random 31-bit `id` (`Job.token` = its hex): the draft id of its streaming drafts and the Stop callback.
- If the user already has a running or queued job, `messages.notify_busy` immediately replies
  "⏳ I'm still working on your previous request (label, elapsed, step) …" with a Stop button
  (`queue_cancel` callback). The new message is processed later as its own turn, in order.
- `/cancel` (and the busy notice's button) is `QueueManager.cancel`: cancels the running task and drops
  the queue. The Stop buttons of a turn are `cancel_current(user_id, job_id)`: the running job only, and
  only while it is still the job the button names (queued messages run next). Two Stop buttons exist:
  `queue_stop_<Job.token>` in the status message (rich button row, inline keyboard as fallback) and
  Telegram's own Stop on the streaming draft, whose `draft_id` is `Job.id` (`can_stop`/`keep_on_stop`);
  pressing it delivers a `stopped_message_generation` update. A user stop (Stop or /cancel) keeps the answer
  streamed so far as a message marked "⏹ Stopped — partial answer, not saved to memory." (streaming on only).
  `JobContext.stoppable` is cleared once the agent returned: Stop then answers "too late" and the
  delivery runs to the end; /cancel still cancels, but a status that already became the answer/summary
  is left as it is. Neither path cancels a task twice (`task.cancelling()`), so a second press or a
  /cancel never cuts the first one's cleanup.
- Cancellation is cooperative: handlers catch `CancelledError`, edit their status message to
  "🛑 Stopped." and re-raise. A tool call already inside a worker thread (Docker, HTTP) cannot be
  interrupted; the turn ends when the thread returns. Sandbox containers keep running until their own
  timeout.
- Isolation is real only because blocking work is off the event loop: `AgentAnthropic.execute_tool`
  runs synchronous providers (search, code, terminal, sms, and every provider registered with
  `blocking=True`: web, document) via `asyncio.to_thread`; image/video tools wrap SDK calls in
  `to_thread` and poll with `asyncio.sleep`; MCP calls are async; profile and MCP store I/O go through
  `to_thread`; media pre-processing uses `to_thread` for PIL/pydub/ffmpeg/pypdf/OpenAI/pandas, and
  `content.materialize` (base64 of the pictures) runs in a thread too. **Any new blocking call must be
  wrapped the same way**, otherwise it stalls every user again.
- Global caps: `MAX_CONCURRENT_TURNS` (asyncio semaphore in `QueueManager`) and
  `MAX_SANDBOX_CONTAINERS` (threading semaphore in `container_manager.py`). `TURN_TIMEOUT_SECONDS`
  cancels a stuck job and tells the user; the worker continues with the next job.
- Agent reminders go through the same per-user queue (`bot/jobs.py`, label "scheduled agent task"), so they
  never interleave with the user's own turn. The job is `run_agent_reminder`: `run_turn(prompt=task_prompt(...),
  header="🤖 *Scheduled task...*", quiet_marker=QUIET_MARKER)`, no announcement message; the framing prompt is
  "[Scheduled task] This is a task you scheduled earlier (id <id>; it is <local time>). Task: <text>. Do it
  now. If there is nothing new or worth telling the user, reply with exactly <no_update/> and nothing else."
  An answer that is exactly the marker sends nothing and deletes the status (`TurnResult.quiet`); the record
  gets `agent_response` (the marker when quiet), `last_result` quiet|answered and a `quiet_runs` counter.
  `run_turn` returning None (error) marks a one-shot task `failed`; a periodic one is rescheduled anyway.
  While queued/running the record has `status: "processing"`; a stale `processing` older than twice the
  turn timeout is reset to `pending` on the next scan. A task moved with ⏰ +1h / 📅 +1d while it runs keeps the
  new time: `_finalize` sees `time` differ from the snapshot it queued, sets it pending and records the result
  without stepping or completing it. Fired user reminders are sent directly with snooze
  buttons (`reminder_snooze_<id>_<10|60|tomorrow>`) and rescheduled/completed only once delivered.
- Forum topics: the queue key is the chat id, so all topics of one chat share one queue.

## Request flow (text message)

```
messages.on_text (bot/handlers/messages.py)      [AuthMiddleware(check_limits=True) already ran:
  user_id = str(message.chat.id)                   authorized, quota ok, kwargs user_id/limit injected]
  stats_tracker.track_message_received; submit(Job(run=_run_text))  → busy notice if needed → return
worker → _run_text → prompt = message_context.with_reference(text, message, bot.id): when the message replies to,
    quotes or forwards something, a plain-text preamble goes first (see "Media handlers" below), then a blank
    line and the user's text
  → bot.agent_runner.run_turn(bot, user_id, chat_id, prompt (str | content blocks), thread_id, reply_to, ctx, limit)
    prompt_text = content.prompt_text(prompt) (text view), prompt_blocks = the blocks when the prompt had media
    sender = ChatSender(bot, chat_id, thread_id, reply_to_message_id)
    UserSettings(user_id).save() (persists backfilled defaults) → user_settings dict
    status = sender.send_text("💭 *Thinking...*", Stop keyboard) wrapped in bot.status.StatusMessage; update_status()
    edits it in place: in rich mode editMessageText(rich_message=blocks) (header, step, iterations, an open
    details block with the last 15 tool calls, usage line, 🛑 Stop row `queue_stop_<job token>`); a rejected
    rich edit switches that status to the legacy-Markdown text (last 10 calls, Stop as inline keyboard), a
    404 marks rich unsupported process-wide. Drafts use draft_id = Job.id with Telegram's Stop
    agents.ChainOfThoughtAgent(model=models.ANTHROPIC_MODEL, user_id, sender, user_settings, thread_id)
       mode = settings.mode.type: speed → AgentAnthropic(model=ANTHROPIC_MODEL_FAST), thinking False;
       intelligence → ANTHROPIC_MODEL, thinking True; auto (default, unknown) → ANTHROPIC_MODEL, thinking = the
       `thinking` toggle. Named slots (file_ops … subagents) + register_provider(Vision, Profile, MCP, Web*, Document*)
       (* = blocking=True, run in to_thread)
    .generate_response(question, update_status, on_text_chunk)
       ├─ agent.on_recovery = update_status(step="recovering", details=note) (transcript and legacy mode alike)
       ├─ question_blocks = content.with_prefix(normalize(question), "Message received time in UTC+0: <ts>\n\n")
       │     (the prefix goes on the first text block); question = content.prompt_text(question_blocks): the text
       │     view, media as [image: <path>] / [document: <title>], used by the classifier, memory, logs, legacy path
       ├─ pick system prompt by settings.system_prompt.type (4 inline prompts; unknown → generall-ai-v2),
       │     + RICH_/LEGACY_FORMATTING_GUIDE, + FOLLOW_UPS_GUIDE when follow_ups.enabled (<follow_ups> block),
       │     + _provider_notes(): every provider's `prompt_notes` inside ONE <capabilities> block (VisionTools,
       │       ProfileTools and UserInteractions always, MCPTools only when the user has MCP tools) — all static,
       │       part of the cache prefix
       ├─ TRANSCRIPT MODE (default, transcript.enabled): _generate_with_transcript → load
       │     data/<uid>/transcripts/[topic_<id>_]transcript.json (seeded from dialog_history.json on first use);
       │     static prompt (time stripped) as ONE cached system block (ttl SYSTEM_CACHE_TTL); fingerprint of
       │     model + static prompt + tools changed → strip stored thinking; _request_context(question, status,
       │     messages) → <context> = UTC time + "Local time (<zone>): … (UTC+2)" or "User timezone: unknown (ask
       │     once and save it with set_user_timezone)" (_render_context, settings timezone.name) + <memory> (recent
       │     summaries + FAISS hits not shown before) +
       │     <profile fingerprint="..."> (profile.enabled; only when the fingerprint differs from the last one shown
       │     in this transcript), stored as the first block of the new user turn; _decide_complexity: mode auto AND
       │     no media → Haiku classifier (text view, context blocks excluded), else "complex"; "simple" → tool-less
       │     Haiku answer on the FULL context (profile every time); _build_user_message materializes the media refs
       │     (base64 image/document blocks, to_thread) behind the <context> text block; tool loop on the real
       │     transcript (tool results capped before the model sees them); history_edited → strip every thinking
       │     block of the stored transcript first (the prefix was sent stripped); then append the turn exactly as
       │     sent, judge/critique/SYSTEM NOTICE included; prune only above max_context_tokens (clear old tool results AND
       │     old image/document blocks → text markers, summarize oldest half, strip thinking); save;
       │     summaries/FAISS/dialog_history.json still written. LEGACY MODE below:
       ├─ semantic_search: FAISS hits appended to the system prompt (to_thread; failure → skipped)
       ├─ context_memory = fixed intro pair
       │     + conversation summaries (summarization_history)   [reads data/<uid>/conversations/*.json]
       │     + dialog_history.json, then context_memory[:-2]     (drops the newest pair; if the file is
       │       empty it eats the summaries pair or the intro pair instead)
       │     + short_term_memory.json messages whose FIRST block is text (reasoning_context)
       │     + the question                       (all three gated by short_term_memory.enabled too)
       │     the user turn = _legacy_context() (<context>: UTC time, local time/zone, <profile> when profile.enabled;
       │     no memory lines) + a blank line + the question, quick path too; afterwards thread_messages gets the
       │     bare question back, so short_term_memory.json and the conversation file never store a context block
       ├─ _classify_complexity (Haiku, auto mode only) → "simple": tool-less Haiku answer; on any error fall back ↓
       ├─ AgentAnthropic.generate_response: prefix_len = len(incoming messages);
       │  for iteration in range(tools.max_iteration):
       │     _stream_message → messages.stream (always) (model, system, tools=get_tools_schema(),
       │        request_options_for(model, thinking): output_config.effort + adaptive thinking for Sonnet, nothing
       │        for the fast model; max_tokens_for: 64000 deep / 16000 light / ANTHROPIC_MAX_TOKENS_FAST; thinking
       │        tokens count against it)
       │        RateLimitError / APIStatusError ≥ 500 or overloaded / APIConnectionError → retried after
       │        STREAM_RETRY_DELAYS 2, 5, 12 s (a numeric Retry-After wins, ≤ MAX_RETRY_AFTER 30 s), each retry
       │        on_recovery(note) → status step "recovering" ("Retrying after a rate limit (2/3)"); the 4th failure
       │        propagates. A 400 naming thinking blocks / signatures / prefix mismatch (is_thinking_mismatch, not the
       │        binding-beta refusal) → retried once with the thinking blocks of the first prefix_len messages removed,
       │        history_edited = True (the rest of the turn sends the prefix stripped); other 4xx propagate at once.
       │        forward_event resets the draft accumulators on message_start, so a retried stream starts over
       │     stop_reason == tool_use AND tools.enabled AND cicles < max → run ALL blocks concurrently
       │        (run_tool_batch: asyncio.gather, is_error on exceptions, trace + status refresh per call; a tool may
       │        return a str OR a list of text/image blocks: cap_blocks cuts only the text, the trace shows [image]),
       │        append ONE user msg of tool_results in block order, continue (cicles counts tool_use BLOCKS)
       │     gate fails → tool_use blocks silently dropped, text goes on to critique/judge/return
       │        (budget spent → AgentAnthropic.limit_reached, which offers ▶️ Continue)
       │     else optional critique (OpenAI gpt-6.1-sol) / judge (Claude yes/no, same request_options_for) re-prompts → return
       │     for-loop exhausted → forced "SYSTEM NOTICE" final call with the loop's tools and tool_choice none (same thinking mode;
       │        limit_reached too)
       └─ strip pre-loaded context from thread_messages; persist memory inside try/except
          (a failed Haiku summary is logged, the answer is still returned) → (response, messages)
    agent returned → drafts closed, ctx.stoppable = False, Stop removed from later status edits
    stats_tracker.track_message_sent
    record_usage(): one usage_events row per model of the turn (tokens, cache, tool calls, cost estimate)
    turn_actions.parse_follow_ups(response) → (display, suggestions): the LAST <follow_ups> block is cut from
       the visible answer (always) and becomes ≤4 suggestions (none when follow_ups is off)
    quiet_marker given (scheduled tasks: bot/jobs.run_agent_reminder passes QUIET_MARKER "<no_update/>" with the
       framing prompt, see "Queues") and display == marker → nothing sent, status deleted, TurnResult(quiet=True)
    save_turn(): TurnRecord(id 12 hex, prompt (text view), prompt_blocks (media refs, Retry re-runs them), display,
       speak, thread, limit_reached, follow_ups) + reasoning text (if reasoning_context.enabled) → data/<uid>/turns/
       (to_thread, newest 50 kept)
    speak? → ElevenLabs TTS (to_thread) → sender.send_voice
    rich mode: sender.send_markdown(display) then status_msg.finish(trace, keep, actions): the status is
       edited into the summary (headline, collapsible "Tool calls (N)" with one details block per call:
       args JSON + result excerpt, collapsible "💭 Thinking", usage + ≈$cost, then the action row), sized
       under 28 KB; keep_summary off → deleted. Legacy mode: sender.send_markdown(display, edit=status)
       rich (default): split ≤32 KB → sendRichMessage(markdown) → on parse error rich HTML
       (telegramify-markdown) → on 404 sticky MarkdownV2 → legacy Markdown → raw; status deleted after
       legacy (rich_messages off): ≤4000 edit status in place; else new chunks + notice
    buttons (success; a failed turn gets 🔁 Retry alone, see "errors" below): actions = ▶️ Continue (limit_reached) ·
       🔁 Retry · 🔊 Listen (TTS key) · 📄 Reasoning
       (stored); on the summary when it took them, else under the last answer message below the follow-up
       rows (one per row). The reasoning file is sent only by its button; if save_turn failed it is sent
       right away (send_reasoning_file) and the turn gets no buttons
    returns TurnResult(response=display, messages, turn_id)
    errors → logger.exception("Turn failed … (trace <uuid>)"); record_usage unless the agent had already returned
       (counted then); a status that already became the answer/summary is left alone; else save_turn(TurnRecord(
       prompt, prompt_blocks, response="", error=friendly_error(e, trace_id))) (no reasoning file) and
       status_msg.fail("❌ <reason> (trace <uuid>)", actions=record_actions(record) = 🔁 Retry only): rich paragraph +
       button row → inline keyboard once rows are refused → plain text + keyboard; a failed save → the text without
       buttons. friendly_error: RateLimitError → ERROR_RATE_LIMIT, APIConnectionError/APITimeoutError → ERROR_NETWORK,
       APIStatusError ≥ 500 or overloaded → ERROR_OVERLOADED, 413 or a BadRequestError saying too large / prompt is too
       long / max_tokens / exceed → ERROR_TOO_LARGE, aiogram TelegramAPIError → "Telegram refused the message: <≤120
       chars>", else "Something went wrong.". Returns None
    CancelledError → user stop before the agent returned: partial answer; status "🛑 Stopped." (user) /
       "🛑 Cancelled." (deadline, shutdown), unless it already became the answer/summary; re-raised
```

Media handlers (`_run_voice`, `_run_video`, `_run_audio`, `_run_images`, `_run_document` in
`messages.py`) are jobs with the same shape and only differ in how they build the prompt. Every job
(text included) starts its prompt with the reply/forward preamble of `bot/message_context.py` when the
message is a reply, quotes a part of one, or was forwarded (`with_reference(prompt, message, bot.id)`;
plain text, no Markdown, separated from the user's own text by a blank line; stored with the turn like
the rest of the prompt):

```
[Replying to a message from Alice Smith (the user), sent 2026-10-17 12:00 UTC]
<text or caption, capped at QUOTE_LIMIT 1500 chars, "… (truncated)">
[The user quoted this part of it: "..."]
[End of the replied-to message]
[Forwarded from Daily News, signed Editor (2026-10-17 12:00 UTC)]
```

Author: `from_user` full name, "(the user)" for the user's own earlier message, "you (the assistant)" when
`from_user.id == bot.id` (without a bot id, i.e. the test fakes, any bot counts as the assistant), the
`sender_chat` title for channel posts. "a message" becomes the media kind (a photo, a document: <name>, a
voice message, a video, a sticker <emoji>, a location <lat>, <lon>, a poll: <question>, ...). The bot's own
rich-message answers are flattened to text (`_rich_text`), dropping the thinking/buttons blocks and the
turn-summary details sections named in `SUMMARY_DETAILS_PREFIXES` (keep them in sync with `bot/status.py`
titles). `external_reply` (a reply to another chat's message) gives origin + media kind, no text. Forum
topics: Telegram sets `reply_to_message` on every topic message to the topic's creation service message;
`_is_topic_root` treats that as "no reply". Nothing is downloaded from the referenced message.

- Voice notes: downloaded to `temp_audio/`, converted, Whisper-transcribed, deleted; only the
  transcription reaches the agent (`speak=True` adds a TTS voice reply). Nothing is persisted.
- Audio files: saved to `data/<uid>/audio/<original name>`; no transcription; prompt carries metadata + path.
- Photos/albums (`_run_images`): **no pre-description** (no Claude/GPT describe calls, no
  `track_describe_used`); the agent sees the picture. Each picture is saved twice under `data/<uid>/images/`:
  `image_<uuid>.jpg` at full resolution (`media.prepare_downloaded_image_for_vision`: a JPEG is copied as is;
  PNG/WebP/GIF/BMP/HEIC are converted, transparency flattened on white) and `image_<uuid>_v.jpg`
  (`media.save_vision_copy`: ≤ `MAX_IMAGE_RESOLUTION_VISION` on the long side and under
  `content.MAX_IMAGE_BYTES`). The prompt is a block list: one text block (preamble, caption + "User attached
  N image(s)…" or "User sent N photo(s).", one `Image i: <full-res path>` line each, `IMAGE_EMBED_HINT`)
  followed by one `content.image_ref(<_v copy>)` per picture, so the transcript stores the downscaled copy
  while the model knows the full-resolution path for edits. A picture that cannot be prepared is named as
  such and skipped; with none left the job stops ("❌ Could not process the image(s)."). Albums are buffered
  per `(chat_id, media_group_id)` and flushed 10 s after the last photo into ONE job.
- Video/video notes: saved to `data/<uid>/videos/`, 4 frames to `data/<uid>/images/video_frame_*.jpg`,
  frames described by `gpt-6-luna`, audio by Whisper; `speak=True`. (Unchanged: video is still described.)
- Documents (`_run_document`, saved to `data/<uid>/documents/<lowercased name>` first, re-uploads overwrite):
  `.pdf` within `media.PDF_MAX_BYTES` (20 MB) and `PDF_MAX_PAGES` (100, counted with pypdf) → a text block +
  `content.document_ref(path, title=name)` (the model reads the PDF itself); a larger or unparseable PDF →
  the old `media.describe_document` path (`_describe_fallback`, still counts `track_describe_used`) with the
  reason in the prompt. `media.TEXT_EXTENSIONS` → the file content in the prompt, capped at
  `messages.TEXT_PROMPT_CAP` (30000 chars, "use read_file for the rest"). `media.OFFICE_EXTENSIONS`
  (.docx/.xlsx/.xls/.json) → `media.extract_document_text`, capped at `EXTRACTED_TEXT_CAP` (60000); longer or
  failed extraction → describe fallback. `on_document` reroutes video extensions to the video job and every
  `image_utils.IMAGE_EXTENSIONS` (jpg/jpeg/heic/heif/png/webp/gif/bmp) to the image job (single or album);
  everything else is rejected with the supported list.
- Pre-processing failures inside the image and document jobs (download, conversion, extraction) are not turn
  errors: they edit the status to "❌ An error occurred while processing the images/document. Trace ID: …" with
  no Retry button. Only a failure inside `run_turn` gets the friendly reason, the error record and 🔁 Retry.

Return-type note: both `generate_response` methods are annotated `-> str` but return
`(text, messages)` tuples. `ChainOfThoughtAgent.generate_response(question)` and `run_turn(prompt=...)`
take a string or a list of blocks built with `agents/content.py`; the stored `media_ref` form
(`{"type": "media_ref", "kind": "image"|"document", "path", "title"}`) is small and JSON-safe (turn records),
`content.materialize` turns it into base64 API blocks right before the request (image media type from the
extension or sniffed; documents must be PDF, anything else must be extracted to text by the caller; a file that
cannot be embedded becomes a text block saying why). `materialize(blocks, base)` confines refs to `base`
(`_build_user_message` passes `data/<uid>`, for new turns and Retry alike): a ref resolving outside it
(`..`, a symlink out) or not a regular file reads as `[<kind> not found: <path>]`; without `base` the path is
read as given (`vision_tools.read_document` resolves it first). `transcript.estimate_tokens` counts an image block as
`content.IMAGE_TOKENS` (1600) and a document by its size, never the base64 as text.

## The tool contract

A tool provider is duck-typed. It must expose:

- `tools_schema` → `list[dict]` of Anthropic tool definitions `{name, description, input_schema}`.
  Either a `@property` (file/search/code/terminal/time/sms/vision/profile/mcp/web/document) or an
  instance attribute set in `__init__` (image/video/user_interactions). If you make it a property, the
  patcher can wrap it.
- `execute_tool(self, tool_name, tool_args) -> str | list[dict]`, sync or async. A list is content
  blocks (`text` and `image`) and is passed through as the `tool_result` content (`view_image`, MCP
  image results); everything else is `str()`-ed.
- Optional `prompt_notes` (str attribute or property): instructions about the provider's tools, appended
  by `ChainOfThoughtAgent._provider_notes` inside the `<capabilities>` block of the static system prompt
  (so it is part of the cache prefix and of the transcript fingerprint: adding or changing notes strips
  stored thinking once). Per-turn text never belongs there.

Tools that talk to the user receive a `sender` (`bot.sender.ChatSender`) in their constructor, never
a Telegram update: `sender.send_text(md)`, `send_markdown(md)` (splits, fallback), `send_document(path|bytes,
filename, caption)`, `send_photo`, `send_video`, `send_voice(bytes)`, `react(emoji)` (only when the turn
has a user message), `typing()`. Captions are truncated to 1024. This is what makes tool sends work in
scheduled agent reminders too. `sender` may be `None` for unit tests; guard before using it.

Wiring lives in `agents/main.py`. The original providers sit in **named slots**; everything added since
the v3 wave goes through the **registry**:

- Named slots (file_ops, search_tools, code_tools, terminal_tools, time_tools, image_tools, video_tools,
  sms_tools, user_interactions, subagents): `AgentAnthropic.__init__` has a `self.<name> = None` slot,
  `get_tools_schema` extends in that fixed order, `execute_tool` has an `elif` per slot (async providers
  awaited: file_ops, image, video, user_interactions, subagents; synchronous ones via `asyncio.to_thread`:
  search, code, terminal, sms; time called directly), and `ChainOfThoughtAgent.__init__` assigns
  `self.agent.<name> = self.<name>`. `ChainOfThoughtAgent._PROVIDER_SLOTS` and `subagent.PROVIDER_SLOTS`
  list them.
- Registry: `agent.register_provider(provider, blocking=False)` appends to `agent.extra_providers`
  (`[(provider, blocking)]`). `get_tools_schema` appends their schemas after `subagents` in registration
  order; `execute_tool` falls through to them after the slots (`blocking=True` → `asyncio.to_thread`,
  else awaited); `allowed_tools`, when set, filters the whole list (subagent subsets). A new provider is
  ONE line in `ChainOfThoughtAgent.__init__`, after the SubagentTools wiring: `VisionTools(user_id,
  sender)`, `ProfileTools(user_id)`, `MCPTools(user_id)` (awaited), `WebTools(user_id, sender)` and
  `DocumentTools(user_id, sender)` (`blocking=True`). `SubagentTools._child` copies `extra_providers`,
  so subagents get the registered tools too (minus `run_subagent`).
- First match wins in both, so tool names must be globally unique (`mcp_*` names are sanitized and
  de-duplicated per server set).

Tool results: a `str` becomes the single `tool_result` content (cut to `transcript.max_tool_result_chars`
first), a list of blocks is passed through with only its text blocks capped (`content.cap_blocks`) and
shown as `[image]` in the trace (`content.result_preview`). Every call is recorded via
`stats_tracker.track_tool_used` (and counts against the user's action limit). Tool schemas are sent to the
API even when `tools.enabled` is false. All `tool_use` blocks of one assistant message run **concurrently**
(`AgentAnthropic.run_tool_batch`), so a provider must be safe to call from several tasks at once; an
exception becomes a `tool_result` with `is_error: true`, a returned string starting with "Error" is passed
through as content (shown ❌ in the trace).

Effective tool list as the model sees it (after patching; 44 built-in tools plus the user's `mcp_*` ones):

| Provider | Tools | Runs where |
|---|---|---|
| FileOperations | `list_files`, `create_file`, `read_file`, `download_file`, `download_webpage` (readable main content: `trafilatura.extract(include_tables=True)`, lazy import cached in `file_ops._TRAFILATURA`, BeautifulSoup page text when trafilatura is missing, raises or finds < `MIN_EXTRACTED_CHARS` 200 chars), `upload_to_s3`, `create_zip_archive`, `send_file_path_to_user_via_telegram` | bot process, paths resolved under `data/<uid>/` by `agents/paths.resolve_under` (escapes refused); the names download_file / download_webpage(save_to_file) / create_zip_archive write keep only their base name (`FileOperations._download_target` → `downloads/<name>`, so a link planted in downloads/ that points outside is refused) |
| FileOperations | `create_directory`, `delete_file` | sandbox container (a fresh `docker run` per call), paths relative to `data/<uid>` |
| SearchTools | `search_web` (Tavily), `deep_research` (Perplexity `sonar*`) | bot process, worker thread |
| SearchTools | `memory_search` (query with `" | "` alternatives, case_sensitive, max_results 1..100 default 20): substring search over `conversations/*.json` (summary, topic, question, response), `profile.md` (field = section) and the text view of `transcripts/*.json` (user/assistant text only: no context blocks, tool results, thinking or media); JSON array of `{source, timestamp, topic, field, snippet}` newest first, or "No matches for …" | bot process, worker thread (reads every conversation file and transcript on each call through `agents.paths.regular_files`: a symlinked folder or file, or a FIFO, is skipped, because data/<uid> is the sandbox's workspace) |
| CodeTools | `execute_python` (+`network_enabled`) | sandbox |
| TerminalTools | `run_command` (no network option), `run_shell_script` (+`network_enabled`), `install_package` (apt, persisted, network on) | sandbox |
| TimeTools | `get_time_in_timezone`, `list_timezones` | bot process |
| ImageTools | `generate_image` (no `images` → generate, one → edit, several → compose; engine `auto`/`best`/`fast`/`story`), `generate_story_images` | bot process (SDK calls in worker threads), never sandboxed, delivers each result itself as a document, once |
| VideoTools | `video_generator`, `image_to_video_generator`, `video_from_reference_images`, `video_interpolation_generator`, `video_extension_generator` | bot process (Veo in worker threads, async polling up to 5 min) |
| SMSTools | `send_sms` | bot process, worker thread |
| UserInteractions | `send_user_telegram_message`, `send_voice_message`, `set_message_reaction`, `set_user_timezone` (IANA name or city/country alias via `timezones.valid_zone`; saves settings `timezone.name` in to_thread and returns the user's local time, `Error: unknown time zone …` otherwise), `schedule_reminder` (reminder_text; reminder_time `YYYY-MM-DD HH:MM[:SS]` read in the user's zone, or in the optional `timezone` param, unless it carries a `Z`/`±HH:MM` offset, which wins; reminder_type user\|agent; is_periodic + period_type hourly\|daily\|weekly\|monthly + period_interval; `weekdays` [0=Mon..6=Sun] for daily periodic reminders only, the first run moved to an allowed day by `first_allowed_trigger`; stores UTC plus the zone; confirms in local time; user reminders fire with snooze buttons, agent tasks run as scheduled tasks and may answer `<no_update/>`), `send_file_content_to_user_via_telegram`, `ask_user_choice` (question + 2-8 option buttons, waits for the tap up to `timeout_seconds` 30..900, default 300; returns `User chose: "<option>"`, a no-answer notice or `Error: …`); its `prompt_notes` (static) tell the model to infer or ask for the zone once and save it, to schedule and confirm in local time, and to answer exactly `<no_update/>` on an uneventful scheduled run | bot process |
| SubagentTools | `run_subagent` (task, tools subset, model main/fast, max_tool_calls) | bot process: a fresh `AgentAnthropic` with the same providers (registered ones included), empty non-persisted context, budget slice of the parent's `TurnBudget`, trace depth 1; main agent only (a subagent has no `run_subagent`); model "main" is `ANTHROPIC_MODEL` whatever the mode |
| VisionTools (registered, awaited) | `view_image` (path, max_px 128..2048, default `MAX_IMAGE_RESOLUTION_VISION`) → `[image block, text "<rel path> · WxH px (shown at wxh)"]` (JPEG/PNG/GIF/WebP within the limits and upright pass through unchanged, else re-encoded as JPEG); `describe_image` (path, question?, model anthropic\|openai) → text from `bot.media.describe_image_anthropic/openai` (temp JPEG in the system temp dir; the OpenAI call writes its own usage row); `read_document` (path, question?) → a PDF within the limits as `[document block, text "<name> · N pages"]`, else extracted text capped at `vision_tools.EXTRACTED_TEXT_CAP` (60000), unknown types through `describe_document` | bot process, never sandboxed; paths via `resolve_under` with images/downloads/documents fallbacks |
| ProfileTools (registered, awaited) | `user_profile_read`, `user_profile_add(section, text)`, `user_profile_remove(section, needle)`, `user_profile_replace(section, old, new)`; needle = exact entry, a unique substring or a 1-based index; results = confirmation + the numbered section; registered always (the `profile` setting only gates the `<profile>` block in the context) | bot process, file I/O in to_thread |
| MCPTools (registered, awaited) | `mcp_<server>_<tool>` for every tool of the user's enabled MCP servers and the admin's (names sanitized to `[a-zA-Z0-9_-]{1,64}`, collisions `_2`, `_3`…; description `[MCP <server>] …`; input_schema passed through, an empty object schema when missing/invalid); `prompt_notes` only when the user has MCP tools | bot process: schemas from the cached tool lists (NO network at turn start); one fresh connection per call (streamable HTTP or SSE by `transport`, stored headers) → `initialize` → `call_tool` → closed; `CALL_TIMEOUT` 60 s (`DISCOVER_TIMEOUT` 30 s); text → joined text, images → base64 `image` blocks (list result), embedded resources → their text or `[resource <uri>]`, `isError` → `Error: …`, unreachable → `Error: MCP server <name> unreachable: <reason>` (never raises; URLs in the reason redacted to scheme+host+path; the error is recorded in the store and shown by `/mcp`) |
| WebTools (registered, blocking) | `browse_webpage` (url, wait_ms 0..10000 default 1500, screenshot, max_chars 500..100000 default 20000): page title, final URL, readable text (trafilatura on the rendered HTML), first 20 links, optional screenshot `downloads/screenshot_<hex>.png` (view with `view_image`; a failed screenshot keeps the page read and says "Screenshot: not saved") | sandbox: ONE container with network on, `mem_limit` 1g, timeout 75 s (script deadline 60 s) via `ContainerManager.run_script("browse.py", …)`; the bot process only formats the JSON |
| DocumentTools (registered, blocking) | `create_document` (format md\|html\|pdf\|docx\|xlsx, title, content Markdown, filename?) → `documents/<sanitized name>.<fmt>` (existing names get `-2`, `-3`); the tool sends the file itself and returns "Document saved to documents/… (<size> KB) and sent to the user." | md/html: bot process (python-markdown: tables, fenced_code, sane_lists, toc + embedded CSS); pdf/docx/xlsx: sandbox via `run_script("render_document.py", …)`, network off, 1g for pdf, timeout 90 s (script deadline 75 s) |

Unreachable: all 11 `SystemTools` tools (`install_package`, `run_shell_script` duplicates plus
`check_system_info`, `manage_service`, `monitor_process`, `network_diagnostics`,
`list_installed_packages`, `remove_package`, `install_python_package`,
`list_installed_python_packages`, `remove_python_package`). `ContainerManager` does honor a
persisted pip list (`data/<uid>/installed_python_packages.txt`) at every sandbox run; no bot code
writes it, but the model can create the file with `create_file` and it will then be replayed.

## Sandbox (secure_container) mechanics

- The image is `secure-container:<tag>` where the tag is `container_manager.image_build_fingerprint()`:
  12 hex of sha256 over `IMAGE_BUILD_FILES` (Dockerfile, entrypoint.sh, requirements.txt). `_ensure_image_exists`
  builds it from `app/secure_container/` when that tag is missing (slow: many apt layers plus
  `python -m playwright install --with-deps chromium`), then `_retag_latest_and_prune` points
  `secure-container:latest` at it and removes older `secure-container:*` images (best effort). So a change
  to any of the three files is picked up at the next startup by itself; the scripts in `scripts/` are copied
  in at call time and need no rebuild. The image installs playwright, trafilatura, markdown, python-docx, openpyxl and Chromium
  (`PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`, world-readable, installed as root before `USER runner`).
  If building a new content tag fails and `secure-container:latest` exists, the bot logs an error and runs on
  `:latest` (`_fall_back_to_latest`; the tag goes into `_FAILED_BUILDS`, so the other ContainerManager
  instances do not build again; the next start retries). Only without any image does the constructor raise.
- Every sandboxed call = one new container `secure-container-<uid>-<8 hex>`: bind-mount
  `$WORKSPACE_ROOT/data/<uid>` → `/home/runner/workspace` (rw, shared), cwd there, user `runner`,
  `mem_limit` = `DEFAULT_MEM_LIMIT` ("512m") unless the caller passes one (`run_command`/`run_script`
  take `mem_limit`; the browser and the PDF renderer use "1g" because Chromium does not fit in 512 MB),
  one CPU, `network_mode=none` unless `network_enabled` or packages must be installed, removed in
  `finally`. `run_command` first acquires one of `MAX_SANDBOX_CONTAINERS` slots (blocking, in the calling
  worker thread). `browse_webpage` and a pdf render hold a slot for 10-60 s each.
- `ContainerManager.run_script(user_id, script_name, args=(), timeout=90, network_enabled=False, mem_limit=None)`
  copies ONE bundled script from `secure_container/scripts/` (`SCRIPTS_DIR`) into the workspace as
  `temp_tool_<hex>.py` (unique per call), runs `python -I <file> <shlex-quoted args>` through `run_command` (isolated mode: the script's directory is the
  user's workspace, so a workspace `json.py` or a `pip install --user` copy of a library would otherwise shadow
  its imports)
  (same slots, mount, package prelude), deletes the copy in `finally` and returns the raw logs.
  `script_name` must be a bare file name of a bundled script; anything with a directory part (`../x.py`,
  absolute paths) is refused with "Error: sandbox script not found". Script output contract: the scripts
  print their result as the LAST line, one JSON object with `json.dumps` (`ensure_ascii=True`, so the line
  never contains a Unicode line break), and the host reads it with `agents.web_tools.last_json_line(logs)`
  (splits on "\n" only: `str.splitlines` also breaks on U+2028/U+2029/U+0085). Both scripts enforce their
  own deadline (browse 60 s, render 75 s) with a watchdog thread and print an error JSON instead of running
  into the container timeout, which would lose the logs. Chromium runs with `--no-sandbox
  --disable-dev-shm-usage --disable-gpu`. `web_tools.writable_dir()` creates `downloads/` (screenshots) and
  `documents/` (rendered files) and chmods them 777 so the container's `runner` can write there, through
  `paths.ensure_real_dir`: the sandbox can replace those folders with links, and a plain chmod by the root bot
  would follow them (e.g. to `/var/run/docker.sock`, which is the host's socket).
- **`run_command` does not go through a shell**: docker-py `shlex.split`s the string and
  `entrypoint.sh` does `exec "$@"`. Pipes, `&&`, redirects, `cd`, globs need `bash -c '...'` or
  `run_shell_script`. Exception: when a persisted package list exists the command is embedded in a
  bash prelude script, so shell syntax suddenly works.
- Python code is written to `data/<uid>/temp_code_<hex>.py`, shell scripts to `temp_script_<hex>.sh`,
  and a package-install prelude to `temp_setup_<hex>.sh`; all are deleted after the run. The name is
  unique per call (`ContainerManager._workspace_file`) because a batch of tool calls runs concurrently:
  fixed names were overwritten by whichever call wrote last, so every container ran that one command.
- Persistence across runs is by re-installing: `installed_packages.txt` (apt, via `sudo apt-get`)
  and `installed_python_packages.txt` (`pip install --user`) are replayed before **every** sandboxed
  call (including `delete_file`/`create_directory`/`browse_webpage`/`create_document` renders) when
  present, which forces network on and adds many seconds per call. `python3-*` apt packages also get a
  best-effort `pip install --user <name-without-prefix>`.
- Output = `container.logs()` (stdout+stderr), always prefixed by the entrypoint banner (~10 lines).
  Tool results are therefore never clean JSON (the script tools parse the last JSON line, see above).
  Non-zero exit → `Error (exit code N):\n<logs>`; a timeout → `Error executing command: <exc>` and the
  logs are lost with the container.
- Security boundary: the sandbox covers `run_command`, `execute_python`, `run_shell_script`,
  `install_package`, `delete_file`, `create_directory`, `browse_webpage` and the pdf/docx/xlsx renders of
  `create_document`. `list_files`, `read_file`, `create_file`, downloads, zip, S3, `memory_search`, the
  vision/profile tools, md/html rendering, MCP calls and every API-backed tool run in the bot container,
  which is privileged with the Docker socket mounted.

## Memory and on-disk data layout

All paths are relative to the process cwd (`app/` in Docker → `/app/data`, bind-mounted to `./data`).

```
data/
  userlist.json                     {users:[chat_id...], blocked_users:[...], invites:{inviter:{code:{created_at,used_by}}}}
  stats.db (+ -wal/-shm)            SQLite: stats_events(user_id,event_type,event_subtype,extra_data,timestamp), user_limits(user_id,action_limit),
                                    usage_events(user_id,timestamp,model,api_calls,input/output/cache_read/cache_write_tokens,tool_calls,duration_s,cost_usd)
  mcp_servers.json                  (or MCP_SERVERS_FILE) the admin's MCP servers, hand-edited, offered to every user, read-only in /mcp,
                                    never rewritten by the bot: {"servers": [{name, url, transport: "http"|"sse", headers: {...}, enabled}]};
                                    may be absent
  mcp_servers.cache.json            (<admin file stem>.cache.json) their discovered tools: {"servers": [{name, tools, updated, error}]},
                                    written at startup (mcp-admin-refresh) and by the admin's /mcp refresh
  <chat_id>/
    settings.json                   per-user settings (see table); merged over defaults on load, unknown keys preserved
    profile.md                      agent-managed profile: "## About", "## Preferences", "## Notes" (entries "- ...", Notes dated
                                    "[YYYY-MM-DD] "), other "## Title" sections after them; ≤ 8 KB (oldest Notes rotated out); deleted
                                    when empty; edited only through bot/profile.ProfileStore (user_profile_* tools, /profile); read only
                                    as a regular file (read_regular_text: O_NOFOLLOW|O_NONBLOCK, so a symlink or
                                    FIFO reads as empty), written via a mkstemp tmp file + os.replace
    mcp.json                        the user's MCP servers: {"servers": [{name, url, transport, headers: {Authorization: "Bearer …"},
                                    enabled, tools: [{name, description, input_schema}], updated (iso), error}]}; atomic writes
    conversations/conversation_<YYYYmmdd_HHMMSS>_<topic>.json
                                    one file per turn: {timestamp, thread_id, topic, summary, question, response, full_history}
                                    (full_history with media blocks replaced by their markers: content.strip_media)
    short_term_memory/[topic_<thread>_]dialog_history.json
                                    last dialog_history.size Q/A PAIRS (size*2 entries) as plain strings
    short_term_memory/[topic_<thread>_]short_term_memory.json
                                    block-structured messages of (mostly) the last turn ("reasoning context")
    transcripts/[topic_<thread>_]transcript.json
                                    transcript mode: {version, user_id, thread_id, created, updated, model, seeded_from,
                                    fingerprint, messages:[exact API messages, as sent]}; append-only, pruned only above
                                    max_context_tokens; atomic write
    embeddings/faiss_index.bin, metadata.json
                                    IndexFlatIP(1536) over L2-normalized vectors (cosine) of "Topic/Question/Answer/Summary" texts with
                                    text-embedding-3-small; metadata.json = {version: 2, model, dimension, updated, items: [{question,
                                    answer (≤ 2000 chars each), timestamp, topic, text_hash}]}; a bare list (legacy) is treated as stale;
                                    both files written tmp + os.replace
    turns/<turn id>.json, <turn id>.reasoning.txt
                                    one TurnRecord per turn (prompt = text view, prompt_blocks = media refs when the prompt had
                                    media, displayed answer, speak, thread_id, limit_reached, follow_ups, has_reasoning, error =
                                    the friendly reason of a FAILED turn: response "", no reasoning file, Retry only; records
                                    without the key read as null; quiet_marker = the scheduled task's marker or null) + its reasoning transcript; what the turn_ buttons load;
                                    newest 50 kept (pruned on save); atomic writes
    reminders/reminders.json        list of {id (8 hex), user_id, text, time (ISO UTC), type: user|agent,
                                    status: pending|processing|completed|failed, created_at, is_periodic, period_type,
                                    period_interval, last_triggered, next_trigger, enabled, [completed_at],
                                    [agent_response: the task's answer, "<no_update/>" after a quiet run], [processing_since],
                                    [timezone: IANA zone it was scheduled in, "" = UTC], [weekdays: [0..6], daily periodic only],
                                    [last_result: quiet|answered], [quiet_runs: int], [snoozed_from: id of the periodic
                                    reminder this one-off copy was snoozed from]}; written atomically under a per-user lock
    images/                         image_<uuid>.jpg (uploads, full resolution, the path the prompt names) + image_<uuid>_v.jpg (the
                                    downscaled copy the model saw; Retry re-sends it, so it must stay), video_frame_<uuid>.jpg,
                                    generated_*/transformed_*/composed_*/gpt_generated_*/gpt_edited_*/story_image_*, edit_input_* (transient)
    videos/                         video_<uuid>.mp4 (uploads), veo3_*.mp4
    audio/, documents/              uploads under their ORIGINAL file names (documents lowercased) → re-uploads overwrite;
                                    documents/ also holds create_document outputs <name>.<md|html|pdf|docx|xlsx> (-2, -3 on a clash)
                                    and the transient render sources .render_<hex>.html|.md
    downloads/                      download_file / download_webpage(save_to_file) / create_zip_archive outputs, screenshot_<hex>.png
                                    (browse_webpage)
    installed_packages.txt, installed_python_packages.txt, temp_code_<hex>.py, temp_script_<hex>.sh,
  temp_setup_<hex>.sh, temp_tool_<hex>.py (run_script copy) (unique per call: tool calls of one message run concurrently)
temp_audio/ temp_photos/ temp_docs/   transient, cwd-relative, deleted in finally; describe_image's temp JPEG is in the system temp dir
app/voice/config.json               MUTATED at runtime by /voice (per-user voice id). Lives inside the image, not data/.
```

Memory semantics worth knowing before touching `ChainOfThoughtAgent.generate_response`:

- Transcript mode is the default: the prompt is the stored API transcript (tool calls, results and thinking
  blocks included) plus the `<context>` block; the bullets below describe the legacy files, which are still
  written (summaries, FAISS, dialog_history.json) but only read for the prompt when `transcript.enabled` is off.
- Forum topics: `message_thread_id` prefixes the transcript and the two short-term files (`topic_<id>_...`), so
  a new topic is a fresh context; there is no `/new` command on purpose.
  Conversation summaries and the FAISS index are shared across threads.
- Summaries are read with `Path.glob` (unsorted) and sliced
  `[:-dialog_history.size][-summarization_history.size:]`. Sizes come clamped 1..50 from the UI;
  a hand-edited `dialog_history.size` of 0 yields nothing, and `summarization_history.size` of 0
  injects **every** conversation file.
- Old conversation files without `summary`/`topic`/`timestamp` are tolerated; unreadable JSON is
  skipped. There is no migration. The conversation filename embeds the raw Haiku "topic" unsanitized.
- Persistence at the end of a turn is wrapped in try/except: a failed Haiku topic/summary or
  embedding call is logged and the answer is still delivered (previously the user saw an error).
  `_save_conversation` embeds AFTER the Haiku topic and summary exist (`add_conversation(question, answer,
  timestamp, topic, summary)`), so they are part of the vector text at once.
- Embeddings (`agents/embeddings.py`): the index is a derived cache of `conversations/*.json`.
  `ConversationEmbeddings(user_id)` does no I/O in its constructor (files load on first use; the OpenAI client
  is built lazily, `client=` injects a fake). On load, a different `metadata.model`, the legacy layout, a
  dimension mismatch, a non-cosine index or index/metadata disagreement marks the index **stale**:
  `search_conversations` returns `[]` and one daemon rebuild thread per user per process is started
  (`AUTO_REINDEX`), which embeds every conversation in batches of ≤ 64 texts and ≤ `BATCH_BYTES` (250k) UTF-8 bytes (a batch the API
  refuses with a 400 is retried one text at a time; a text still refused is skipped and logged, any other error
  fails the rebuild; conversations/ is read through `regular_files`), writes both files atomically and swaps
  the index under a per-user `RLock`; other live instances reload via a generation counter. `add_conversation`
  during a rebuild is queued and appended after the swap (deduped by `text_hash`); while stale with no rebuild
  running the add is dropped (logged): the next rebuild reads the conversation file anyway. A failed automatic
  rebuild is not retried until the next process start (`reindex_status(uid)`; `reindex(blocking=True)` forces
  one). Every embedded text and the query are cut to `EMBED_TEXT_BYTES` (8000 UTF-8 bytes). Searches strip the
  `Message received time in UTC+0:` prefix from the query, collapse hits with the
  same normalized question and return `{question, answer, timestamp, topic, score 0..1}`; adds strip that
  prefix from the question and the last `<follow_ups>` block from the answer. Neither method ever raises.
  Embedding calls write no `usage_events` row.
- The `<context>` block of a user turn may end with `<profile fingerprint="<12 hex>">\n<profile.md>\n</profile>`
  (after `<memory>`; `agents.profile_tools.profile_block`). It is added to the stored turn only when the
  fingerprint differs from the last `<profile fingerprint=...>` in the transcript's context blocks
  (`last_shown_profile_fingerprint`), so an unchanged profile costs nothing per turn; the quick (Haiku) path
  gets the full context, profile included, every time, because it never sees earlier context blocks. If
  `prune` summarized the turn that showed it away, the profile is shown again next turn (intended).
  `_request_context(question, update_status, messages)` takes the transcript messages and computes
  `shown_memory_lines` itself.
- Media in the transcript: the user turn stores the materialized `image`/`document` blocks (exactly what was
  sent). `transcript.clear_old_media(messages, keep_turns)` runs inside `prune` right after
  `clear_old_tool_results` (same `keep_tool_results_turns`): image/document blocks in user messages older than
  the last N user turns (top level and inside `tool_result` content lists) become `IMAGE_CLEARED_MARKER` /
  `DOCUMENT_CLEARED_MARKER` text blocks naming `view_image` / `read_document`; `prune` reports
  `stats["cleared_media"]` and it counts as an edit that strips thinking. Tool results of those turns are
  cleared wholesale first, so pictures inside them count under `cleared`. The last N turns are never touched.
- Reloading `short_term_memory.json` keeps only messages whose first block is `text`. With thinking
  on, intermediate tool-calling assistant turns start with a `thinking` block and are dropped; the
  final answer (re-appended as text-only) is kept. Tool-result user messages are never reloaded.
- Scheduled tasks are ordinary turns for the memory: the framing prompt and the answer, `<no_update/>`
  included, are appended to the transcript, written to dialog_history.json, summarized by Haiku and embedded
  like any turn; the agent core knows nothing about the marker (`run_turn` checks it after `generate_response`
  returned, and only skips the delivery).
- Prompt caching (transcript mode): the static system prompt is one `text` block with an explicit
  `cache_control` breakpoint (`SYSTEM_CACHE_TTL`), covering the tool schemas before it; every request also
  passes the top-level `cache_control={"type": "ephemeral"}` so the API places an automatic breakpoint on
  the conversation tail. Everything volatile (time, memory hits) goes into the `<context>` block, the first
  block of each user turn; it is stored with the turn (see "Preserved thinking" in Known pitfalls), and only
  memory lines the transcript has not shown yet are included, so the previous turn is a cache read.
  `ToolTrace.add_usage(usage, model)` accumulates `usage` per turn and per model; the status line shows the
  cached share and an estimated cost (`models.MODEL_PRICES`, cache reads ×0.1, writes ×1.25), and
  `agent_runner.record_usage` stores one `usage_events` row per model; `/stats` (admin) shows tokens, cost
  and top spenders, per day too (`get_daily_series`, the dashboard's cost chart); `/usage` shows the user their
  own numbers (`get_user_usage_summary`). The OpenAI critique call is added to the turn's trace; the GPT
  `describe_image` second opinion and the video-frame descriptions (`bot/media.py`) each write their own
  `usage_events` row (OpenAI usage has no cache split: all prompt tokens are priced as input). Image and video calls are added to the turn's trace too
  (`ImageTools.trace` / `VideoTools.trace`, set per turn by `ChainOfThoughtAgent.generate_response`): OpenAI and
  Gemini images priced by modality with `models.estimate_image_cost` / `IMAGE_TOKEN_PRICES`, Veo per generated
  second with `estimate_video_cost` / `VIDEO_PRICES_PER_SECOND` (the tools leave `duration_seconds` unset, so
  `VEO_DEFAULT_SECONDS` / `VEO_EXTENSION_SECONDS` are assumed). That cost is stored as the bucket's `cost_usd`,
  which `ToolTrace.model_cost` prefers over `MODEL_PRICES`. Haiku side calls (topic/summary/classifier), the
  judge, Claude image/document descriptions, Whisper, embeddings and the Gemini Pro image model (no price) are
  not counted.
  Legacy mode still has no caching (its system prompt embeds the time).

## User settings (`data/<uid>/settings.json`)

`bot/settings.py` holds `DEFAULT_SETTINGS`, the single source of truth. `UserSettings` deep-copies
the defaults, merges the file over them (unknown keys are kept, never `KeyError`), and `save()`s on
every `set`. `run_turn` saves once per turn so new defaults reach disk. Agent code reads them as
`user_settings.get(cat).get(key)`, so every category must stay in `DEFAULT_SETTINGS`.

| Category | Keys (default, UI clamp) | Consumer |
|---|---|---|
| `summarization_history` | enabled (true), size (5, 1..50) | conversation summaries in context |
| `dialog_history` | enabled (true), size (10, 1..50) | rolling Q/A window |
| `reasoning_context` | enabled (true) | reload last turn's blocks; also gates the stored reasoning file and its 📄 button |
| `short_term_memory` | enabled (true) | master switch for the three above |
| `critique` | enabled (false), max_iteration (5, 1..300) | OpenAI critique re-prompts |
| `judge` | enabled (false), max_iteration (5, 1..300) | Claude yes/no completeness re-prompts |
| `tools` | enabled (true), max_iteration (20, 1..300) | the agent loop bound (this, not the env var) |
| `semantic_search` | enabled (true), max_results (3, 1..20) | FAISS hits in system prompt |
| `thinking` | enabled (true) | on ("deep"): adaptive thinking, `display: summarized`, effort `ANTHROPIC_EFFORT` (high), max_tokens `ANTHROPIC_MAX_TOKENS` (64000); off ("light"): still adaptive, `display: omitted`, effort `ANTHROPIC_EFFORT_LIGHT` (low), max_tokens 16000. Never `disabled` (`models.anthropic_request_options` / `anthropic_max_tokens`). Overridden by `mode` speed (off) and intelligence (on) |
| `mode` | type (`auto`; also `speed`, `intelligence`; `MODE_TYPES`) | `ChainOfThoughtAgent.__init__` (`_mode_of`: unknown → auto) and `_decide_complexity`. auto: the Haiku classifier picks quick answer vs full agent, model `ANTHROPIC_MODEL`, thinking = the toggle; speed: no classifier, the fast model runs the whole tool loop (`request_options_for` gives it no effort/thinking, `ANTHROPIC_MAX_TOKENS_FAST`), thinking off; intelligence: no classifier, `ANTHROPIC_MODEL`, thinking on. A prompt with media is always "complex" |
| `timezone` | name ("" = unknown → times in UTC; a canonical IANA name otherwise) | set by `/timezone` (picker `tz_set_<i>` over `COMMON_ZONES`, or `/timezone <name\|city>` through `timezones.valid_zone`), by sharing a location (`messages.on_location`, needs the optional `timezonefinder`) or by the agent's `set_user_timezone`; not on the `/settings` page. Read by `_render_context` (the local-time line of the `<context>` block, transcript and legacy mode), `schedule_reminder` (default zone of `reminder_time`), `jobs.task_prompt` (the time in the framing prompt) and the `/reminders` pages (`timezones.format_local`) |
| `profile` | enabled (true) | the `<profile>` block in the `<context>` (both modes; legacy shows it every turn); the `user_profile_*` tools work regardless (the model then sees the file only via `user_profile_read`) |
| `ui` | advanced (false) | `/settings` page only: off hides the power settings (`Control.advanced`); the section summaries still show their values |
| `rich_messages` | enabled (true) | answers as Telegram rich messages (native GFM); off = legacy Markdown v1 path; also selects the `<formatting>` prompt section and rich vs plain streaming drafts |
| `image` | engine (`auto`), quality (`auto`), size (`2K`), max_variants (4), metadata (true) | defaults for `generate_image`; the agent may override engine/quality from the request, `max_variants` is a hard cap, `metadata` puts model/quality/size in the delivered file's caption |
| `trace` | keep_summary (true) | end of turn: status shortened into the rich summary above the answer (expandable tool calls, thinking, tokens, action buttons); off = deleted (actions go under the answer) |
| `follow_ups` | enabled (true) | `FOLLOW_UPS_GUIDE` in the system prompt and follow-up buttons under the answer; off = neither (a block the model still writes is hidden anyway) |
| `transcript` | enabled (true), max_context_tokens (120000, 20k..400k), keep_tool_results_turns (3), max_tool_result_chars (20000, applied when the tool returns, before the model sees it) | transcript mode (real conversation replayed as is); when on, `dialog_history`/`reasoning_context` are ignored for the prompt (dialog_history.json is still kept current) and `summarization_history.size` = recent summaries in the `<memory>` block |
| `system_prompt` | type ("generall-ai-v2"; also generall-ai-v1, perplexity-deep-research, perplexity-r1) | prompt selection |

Presets (`bot/settings.py` `SETTINGS_PRESETS`, `{(category, key): value}`): fast = mode speed, thinking off,
tools.max_iteration 10; balanced = mode auto, thinking on, 20 (= the defaults); deep = mode intelligence,
thinking on, 40, judge.enabled false. `apply_preset(user_settings, name) -> bool` (one save, unknown names
change nothing, other settings untouched: a preset is a starting point, not a reset), `current_preset` (every
key must match exactly, else "custom" and no button is marked), `PRESET_NAMES` keeps the order.

Edited through `/settings` (`bot/handlers/settings_ui.py`). With rich messages on (and button rows
available) it is ONE rich page: a bold "Preset" paragraph with ONE row `⚡ Fast · ⚖️ Balanced · 🧠 Deep`
(the active one "✅ …", callback `setp_preset_<name>`) and the `⚙️ Advanced ✅/⬜` toggle (`setp_ui_advanced`,
`Control ADVANCED` in `CONTROLS` but in no section), then four collapsible sections (Memory, Agent, Replies,
Images) built from `SECTIONS` (`Control(token, action, category, key, kind toggle|number|choice, name, hint, …,
advanced=False)`), each setting a bold paragraph with a friendly name ("🎛 Quick answers vs. deep work" for
mode, "Deep reasoning", "Tool calls per answer", "Remember this chat in full", "Memory window",
"Completeness check", "Second opinion rewrite (GPT)", "Personality prompt", …; tests address rows by these
names) plus button rows inside the message (`control_rows`: toggle "✅ On"/"⬜ Off", number "−  value  +",
choice = one button per option, the current one "✅ …" and not tappable, spread over rows by `split_rows`
so labels stay readable on a phone, with `CHOICE_LABELS` for values that make poor button text, e.g. the
system prompt types); `Control.advanced` rows (transcript sizes, summaries size, dialog, reasoning,
short-term memory, semantic max, critique, judge, system prompt, image variants/metadata) are hidden while
`ui.advanced` is off (`visible_controls`). A tap applies and edits the page in place with that section open.
Its callback_data is `setp_<token>_<action>[_<value>]`, `setp_sysprompt_<type>`, `setp_preset_<name>`,
`setp_ui_advanced`; a server that refuses sections gets the flat layout from then on, and a refused flat page
(not a chat-specific error) switches button rows off. Otherwise the inline keyboards: callback_data
`settings_<token>[_<action>[_<value>]]` parsed with a plain `split("_")`; the overview has a `🎛 Mode` line
and button (`settings_mode` → `show_mode_menu`, `settings_mode_set_<value>`) and a presets row
(`settings_preset_<name>`, special-cased in `settings_button`). In both, `<token>` is a short name, not the
JSON key: summarization, dialog, reasoning, memory (= short_term_memory), critique, judge, tools, semantic,
thinking, rich, transcript, trace, image, followups, mode, ui, main and the pseudo-token preset.
`system_prompt` is special-cased with `startswith`. Both UIs change values only through
`apply_setting(user_settings, token, action, value)` (same clamps and allowed choices; also `("mode", "set",
value)`, `("ui", "advanced")` and `("preset", "set", name)` → `apply_preset`).

## Models and external services

**Single source of truth: `app/models.py`.** Every model name is a module constant there with an
environment override of the same name (read once at import, after `load_dotenv()`), and the request
options that belong to a model live next to it: `anthropic_request_options(thinking, effort=None)` returns
`{"output_config": {"effort": ...}}` plus adaptive `thinking` (`display` `summarized` for `True`, `omitted` for
`False`, omitted parameter for `None`; effort `ANTHROPIC_EFFORT` vs `ANTHROPIC_EFFORT_LIGHT`), `anthropic_max_tokens(thinking)`
the matching ceiling, `request_options_for(model, thinking)` / `max_tokens_for(model, thinking)` the same
for ANY Anthropic model (`{}` and `ANTHROPIC_MAX_TOKENS_FAST` for `ANTHROPIC_MODEL_FAST`, which is what the
loop, the final call and the judge use, so speed mode works), `anthropic_text(message)` the text blocks (the
first block may be thinking), and `openai_reasoning_options(model)` returns `{"reasoning_effort":
OPENAI_REASONING_EFFORT}` for the two OpenAI reasoning models and `{}` for anything else. No other file holds
a model literal.

| Purpose | Model / API (default; env override) | Request options | Used by |
|---|---|---|---|
| Agent loop, judge, final compile (modes auto and intelligence) | `claude-sonnet-5-5` via `anthropic.AsyncAnthropic` (`ANTHROPIC_MODEL`) | adaptive thinking always; loop and final call: effort `high` + summarized display when the user's `thinking` setting is on (or mode intelligence), effort `low` + omitted display when off; judge: light mode, max_tokens 2048; the loop and final call always stream; all built with `request_options_for(self.model, …)` | `agents/main.py` |
| Document & image description fallbacks (PDFs over the block limits, failed extraction, `describe_image(model="anthropic")`, `read_document` on unknown types) | `claude-sonnet-5-5` (`ANTHROPIC_MODEL`) | light mode (adaptive, effort `low`, display omitted); max_tokens 4096 / 8192 / 16000 leave room for thinking; text read with `anthropic_text` | `bot/media.py` via `bot/clients.py` clients |
| Topic/summary, complexity classifier, "simple" answers, fast subagents, transcript summaries; the whole tool loop in mode speed | `claude-haiku-5-5` (`ANTHROPIC_MODEL_FAST`) | none: no `effort`, no `thinking` parameter (accepted by every model; Haiku 4.5 rejected both; `request_options_for` returns `{}`). Answers are read with `anthropic_text` and short calls get `ANTHROPIC_MAX_TOKENS_FAST_SHORT` (2048), in case the model thinks by default; the loop gets `ANTHROPIC_MAX_TOKENS_FAST` (16000) | `agents/main.py` |
| Critique | `gpt-6.1-sol` structured output (`beta.chat.completions.parse`) (`OPENAI_MODEL`) | `reasoning_effort` = `high` (`OPENAI_REASONING_EFFORT`); no `temperature`/`max_tokens` | `agents/main.py` |
| GPT vision: `describe_image(model="openai")` second opinion only (photos are no longer pre-described) | `gpt-6.1-sol` (`OPENAI_MODEL`) | `reasoning_effort` `high`; writes its own `usage_events` row | `bot/media.py` via `agents/vision_tools.py` |
| Video frame description | `gpt-6-luna` (`VIDEO_FRAMES_MODEL`) | `reasoning_effort` `high`; no `max_completion_tokens` (it would cap reasoning + answer together) | `bot/media.py` |
| Transcription | `whisper-1` (`WHISPER_MODEL`) via a second client keyed by `OPENAI_API_KEY_WHISPER` (falls back to `OPENAI_API_KEY`); >24 MB chunked | none | `bot/media.py` `transcribe_audio` |
| Embeddings | `text-embedding-3-small` (`EMBEDDING_MODEL`), dim 1536 (`EMBEDDING_DIMENSION`, must match the model's vector size); a model change makes every existing index stale and it is rebuilt from `conversations/*.json` in the background (no manual migration; searches return nothing for that user until it finishes) | none; cosine over L2-normalized vectors (`IndexFlatIP`) | `agents/embeddings.py` |
| MCP tool servers (the user's and the admin's) | `mcp>=2.0` client: streamable HTTP (`streamable_http_client` with an httpx client carrying the stored headers) or legacy SSE (`sse_client`), `ClientSession(read, write, read_timeout_seconds=…)`; the 1.x camelCase/`streamablehttp_client` branches are untested fallbacks | one connection per call, `initialize` → `call_tool`; the `mcp` import (~0.8 s) is lazy inside `open_transport` / `client_session` (tests replace them) | `agents/mcp_tools.py`, `bot/mcp_store.py` |
| Image gen/edit/compose | engine → model via `models.IMAGE_BACKENDS`: `auto`/`best` → `gpt-image-2.5-sunburst` (`GPT_IMAGE_MODEL`), `fast` → `gpt-image-2.5-flare` (`GPT_IMAGE_MODEL_FAST`, same price, lower quality), `story` → `gemini-3.1-flash-image-preview` (`GEMINI_IMAGE_MODEL_FLASH`), escalating to `gemini-3-pro-image-preview` (`GEMINI_IMAGE_MODEL_PRO`) at quality xhigh/max | OpenAI: `quality` from the `auto…max` ladder, pixel `size` from ratio+tier, `n=variants`, `output_format`, and `input_fidelity=high` on edits so the original survives (any knob the model refuses is dropped and remembered for the process). Gemini: `ImageConfig(aspect_ratio, image_size)`, no quality knob | `agents/image_tools.py` |
| Video | `veo-3.1-generate-preview` for all five tools (`VEO_MODEL`) | none | `agents/video_tools.py` |
| Web search / research | Tavily; Perplexity `sonar` default (`PERPLEXITY_MODEL`), enum `PERPLEXITY_MODELS` = sonar-reasoning-pro / sonar-pro / sonar, via raw HTTP | Perplexity payload keeps its own `temperature`/`max_tokens` | `agents/search_tools.py` |
| TTS | ElevenLabs `eleven_multilingual_v2` (`TTS_MODEL`), voices in `app/voice/voices.json` | none | `bot/media.py`, `agents/user_interactions.py` |
| SMS | Twilio | `agents/sms_tools.py` |
| Object storage | boto3 S3-compatible, presigned URL 1 h | `agents/file_ops.py` |

Client construction: `bot/clients.py` builds OpenAI/Whisper/Anthropic lazily on first use. The agent
package still builds `OpenAI`, `genai.Client`, `AsyncAnthropic` and `TavilyClient` at import time;
the first two raise immediately if their key is missing. Twilio, boto3, `TavilyClient` in
SearchTools and the embeddings `OpenAI` are per instance; ElevenLabs is per call in a worker thread.

## Environment variables

| Variable | Used by | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | bot/config | required (validated at startup) |
| `TELEGRAM_CHAT_ID` | bot/config, bot/auth | comma list of always-authorized chat ids; optional if admin or allow-all is set |
| `TELEGRAM_ADMIN_ID` | bot/auth, commands, stats | admin: authorized for chat, unlimited invites, no action limit, `/stats`, `/listusers`, join notifications |
| `TELEGRAM_ALLOWED_ALL_USERS` | bot/auth | `true`/`1`/`yes` (case-insensitive) → everyone not blocked is authorized |
| `INVITE_LIMIT` | commands | default 3 (`.env.example` says 5) |
| `TELEGRAM_USE_LOCAL_API`, `TELEGRAM_LOCAL_API_URL` | bot/app | local Bot API server (compose sidecar); default URL `http://localhost:8081` |
| `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` | compose only | for the `telegram-bot-api` sidecar |
| `STREAMING_ENABLED` (true) | bot/config AND agents/main (read independently, both after dotenv; the two defaults must agree, `tests/test_mode.py` checks both in a fresh interpreter) | draft streaming: `send_rich_message_draft` in rich mode (thinking as `<tg-thinking>`), else `send_message_draft`; errors swallowed; `draft_id` = the job id with `can_stop` (Telegram's Stop on the draft); `false` = answers arrive whole |
| `MCP_SERVERS_FILE` (`data/mcp_servers.json`) | bot/mcp_store (read at call time) | the admin's MCP servers file, offered to every user; may be absent; its tool cache is `<stem>.cache.json` next to it |
| `MAX_CONCURRENT_TURNS` (8), `TURN_TIMEOUT_SECONDS` (1800), `DROP_PENDING_UPDATES` (true) | bot/config | queue caps, see "Queues" |
| `MAX_SANDBOX_CONTAINERS` (4) | secure_container/container_manager | concurrent sandbox containers |
| `THREAD_POOL_SIZE` (32, min 8) | bot/app | size of the default executor used by `asyncio.to_thread` |
| `WORKSPACE_ROOT` | secure_container | host path of the checkout (see Running); set in compose, not `.env.example` |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_API_KEY_WHISPER`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `PERPLEXITY_API_KEY`, `ELEVENLABS_API_KEY` | various | see models table; `OPENAI_API_KEY` and `GOOGLE_API_KEY` needed to import `agents.main` |
| `ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT`, `ANTHROPIC_MODEL_FAST`, `OPENAI_MODEL`, `VIDEO_FRAMES_MODEL`, `OPENAI_REASONING_EFFORT`, `WHISPER_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `GEMINI_IMAGE_MODEL_FLASH`, `GEMINI_IMAGE_MODEL_PRO`, `GPT_IMAGE_MODEL`, `GPT_IMAGE_MODEL_FAST`, `VEO_MODEL`, `PERPLEXITY_MODEL`, `TTS_MODEL` | `models` | optional overrides of the model defaults (blank = default), read once at import; see "Models and external services" and `.env.example` |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | sms_tools | optional; tool returns an error string if unset |
| `S3_HOST`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME`, `S3_PATH_TO_STORE` | file_ops | optional |
| `MAX_IMAGE_RESOLUTION_VISION` (1024), `MAX_IMAGE_RESOLUTION_EDIT` (4096) | bot/media, image_tools | downscale before vision / edit |
| `FFMPEG_DIR` | bot/media (Windows only) | directory holding ffmpeg.exe when not on PATH |
| `MAX_AGENT_TOOLS_ITERATIONS`, `MAX_AGENT_CRITIQUE_ITERATIONS` | read, never used | obsolete; per-user settings replaced them |
| `BROWSER_SERVICE_URL` | nobody | listed in `.env.example`, unused |
| `THINKING_PREFIX_MISMATCH` (drop_block) | `models` | preserved-thinking safety net sent with the `thinking-binding-controls-2026-08-01` beta: `drop_block` drops a stale thinking block instead of failing, `error` fails (finds history edits while testing), `off` sends neither |
| `PROMPT_CACHING` (true), `SYSTEM_CACHE_TTL` (1h), `ANTHROPIC_MAX_TOKENS_FAST` (16000), `ANTHROPIC_MAX_TOKENS_FAST_SHORT` (2048) | `models` | explicit breakpoint on the static system block + top-level automatic caching of the conversation tail (`cache_control` kwarg); max_tokens for fast-model subagents; max_tokens for the short fast-model calls (topic, summary, classifier) |

Optional dependency, not an env var: `timezonefinder` (a commented line in `app/requirements.txt`). When it is
installed, a shared Telegram location sets the time zone (`timezones.zone_from_location`: lazy import, one
cached instance, the polygon load runs in a worker thread); without it the bot answers `LOCATION_NO_FINDER`
and asks for the zone name. Nothing else depends on it.

Never commit `.env`, `data/`, `temp_photos/`, `temp_docs/`, `temp_audio/` (git-ignored by exact
name; a new temp dir needs its own `.gitignore` line).

## Telegram layer conventions (aiogram)

- Routers and middleware (`bot/handlers/__init__.py`): `public` (/start, /help, /invite: no auth; /help
  shows unauthorized users the invite flow), `admin` (/listusers, /stats: `AuthMiddleware(require_admin=True)`),
  `ui` (/voice, /timezone + `tz_` buttons, /settings, /reminders + the pending-edit text handler, /usage,
  /profile, /mcp, /cancel, `queue_cancel`, Stop, shared locations (`F.location` → time zone), `ask_` answers,
  Listen/Reasoning, `usage_`/`profile_`/`mcp_` buttons: `AuthMiddleware()`, also on the
  `stopped_message_generation` observer), `chat` (content handlers and the turn-starting buttons
  Retry/Continue/follow-ups: `AuthMiddleware(check_limits=True)` on message AND callback_query). Handlers
  never check auth themselves; they receive `user_id` (str chat id) and, in `chat`, `limit` as kwargs.
  Unknown `/commands` match nothing and are ignored. Middleware is registered as inner middleware, so
  it only runs for a handler whose filters matched. Polling uses `dp.resolve_used_update_types()`, which
  must keep `stopped_message_generation` (Telegram's Stop on a draft) next to message and callback_query.
- Commands: `/start [invite_<code>]` (authorized users get `START_AUTHORIZED`: welcome + 3 "Try:" examples +
  pointers to /help and /settings via `answer_md`; the unauthorized branch is unchanged HTML), `/help`
  (`HELP_ABOUT` capabilities, the `COMMANDS` with one-line descriptions, tips, 6 example prompts; legacy
  Markdown), `/invite [code]`, `/voice`, `/settings`, `/usage`, `/profile`, `/mcp [add <name> <url> [token]
  [sse] | remove <name> | refresh [name] | help]`, `/reminders`, `/timezone [name|city]` (`commands.py`,
  `voice_router`: no argument → the saved zone and local time, a picker of `timezones.COMMON_ZONES` two per
  row (`tz_set_<index>`) and the hint to share a location or type a name; an argument → `valid_zone`, save,
  confirm with the local time (`timezone_set_text`), or `TIMEZONE_UNKNOWN`), `/cancel`, `/stats` (admin),
  `/listusers` (admin).
- Command menu: `commands.COMMANDS` (start, help, settings, usage, reminders, profile, timezone, voice, mcp,
  cancel; descriptions ≤ 256 chars) is the one list: `bot_commands()` feeds `set_my_commands` at startup and
  `help_text` renders the command section of /help from it. /invite and the admin-only commands are not
  listed. Add a user command there and both the menu and /help show it.
- Callback prefixes: `voice_`, `settings_` (incl. `settings_preset_<name>`, `settings_mode[_set_<v>]`),
  `setp_` (settings page, incl. `setp_preset_<name>`, `setp_ui_advanced`), `reminder_`/`reminders_`, `noop`,
  `queue_cancel`, `queue_stop_<job token>` (Stop in the status), `ask_<qid>_<i>` (ask_user_choice),
  `turn_<retry|cont|listen|reason>_<turn id>` / `turn_fu_<turn id>_<i>` (turn buttons; a `turn_` catch-all
  answers "expired"), `stats_`, `usage_refresh` (`usage_ui.usage_button` matches `usage_`), `profile_`
  (`profile_refresh`, `profile_clear`, `profile_clear_yes`, `profile_cancel` → `profile_ui.profile_button`),
  `mcp_<action>[_<name>]` (`mcp_toggle_`, `mcp_refresh_`, `mcp_remove_` → confirm `mcp_rm_` / `mcp_back`,
  `mcp_refreshall` → `mcp_ui.mcp_button`; actions carry no underscore, names are `[a-z0-9_-]`), `tz_set_<index>`
  (the /timezone picker → `commands.timezone_button`, an index into `COMMON_ZONES`). The reminders family
  (`reminders_ui.reminder_button`, parsed `split("_", 2)` → action + payload, a trailing `_<value>` taken off
  the payload with `rpartition`): `reminders_page_<n>`, `reminders_summary`, `reminders_new`, `reminder_info_<id>`,
  `reminder_delete_<id>`, `reminder_toggle_<id>`, `reminder_delay_<id>_<minutes>` (+1h = 60, +1d = 1440),
  `reminder_edit_<id>`, `reminder_editcancel_<id>`, `reminder_snooze_<id>_<10|60|tomorrow>` (on fired
  reminders, built by `jobs.snooze_keyboard`); group titles and spacers are `noop`. `rem_` is NOT free. A new
  family needs a prefix that is not a prefix of (and does not start with) any other, and a
  `@router.callback_query(F.data.startswith(...))` handler on the router with the right middleware;
  `tests/test_cross_features.py::ROUTING` checks that every family reaches its own handler in router order
  (`MESSAGE_ROUTING` does the same for message handlers: plain text, text during a pending reminder edit,
  /reminders and /cancel during one, a location, a photo)
  (rows exist for usage_, profile_, mcp_, setp_preset_, setp_ui_advanced, settings_preset_, settings_mode_set_,
  tz_set_, reminder_snooze_, reminder_delay_, reminder_edit_, reminders_new). Read the chat from
  `callback.message.chat.id` and guard `isinstance(callback.message, Message)`. Answer the callback best-effort
  (an old query raises).
- Message handlers outside `chat`: `reminders_ui.reminder_edit_text` (`F.text & ~F.text.startswith("/")` AND
  `_edit_is_pending`: the chat has a pending `✏️ Edit text`, `_pending_edits[user_id] = (reminder id, monotonic
  deadline)`, `EDIT_TIMEOUT` 300 s) sits on the `ui` router, which runs before `chat`, so the next plain text
  becomes the reminder text and never reaches the agent; it is not a queue job, so it also works while a turn
  runs. Commands pass through the filter; only `/reminders` and `✖️ Cancel edit` clear the pending state.
  `messages.on_location` (`F.location`, `control_router`) sets the zone from a shared location
  (`zone_from_location` in `to_thread`) and confirms with `commands.timezone_set_text`, or asks for the name
  (`LOCATION_NO_FINDER`) when `timezonefinder` is missing; the location never reaches the agent.
- Buttons inside rich messages (`InputRichBlockButtons` rows: the settings page, the status Stop and action
  rows, questions) depend on the sticky process-wide `rich.buttons_available()` flag; a refusal proven to
  be about the row (the same blocks without it pass) or naming buttons sets `mark_buttons_unavailable`,
  and from then on inline keyboards are used. `rich.is_chat_error(e)` (topic closed, no rights, chat not
  found) must never set a process-wide flag.
- Static UI text goes through `bot.ui.answer_md` / `edit_md` (legacy Markdown, plain fallback,
  "message is not modified" swallowed). LLM text goes through `ChatSender.send_markdown`, which in
  rich mode (`ChatSender(rich=True)`, set by `run_turn` from `rich_messages.enabled`) sends
  `InputRichMessage(markdown=...)`; a plain message cannot be edited into a rich one, so the status
  message is deleted instead of edited. `bot/rich.py` owns the tiers and the process-wide
  "server has no rich support" flag (`rich.reset()` in tests). Markdown images in an answer
  (`![alt](images/x.jpg)` or an https URL) are resolved by `rich.extract_media` against the sender's
  `media_root` (`data/<uid>`): in the rich tier they are uploaded with the message as `tg://photo?id=`
  media, in every other tier the caption stays in the text and the file is sent as a photo/video after
  it. Unresolvable images (missing, outside the workspace, too big) are replaced by their caption. Escape
  user/LLM text embedded in Markdown with `bot.ui.escape_markdown`. Invite/admin replies use HTML. Zone
  names carry `_` (`America/New_York`): the reminders pages and `/timezone` escape them outside bold spans
  (`_zone_line`, `_format_time`, `snooze_text`, `timezone_overview`) and leave them bare inside `*…*`
  (`timezone_set_text`), where legacy Markdown takes `_` literally and a `\` would show; button labels are
  plain text (`tests/test_reminders_ui.py::test_zone_names_with_underscores_are_escaped_on_every_page`).
- Keyboards: `InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=..., callback_data=...)]])`
  (keyword arguments are mandatory in aiogram).
- Every content handler: track_message_received → `submit(message, Job(...))`. The job function
  signature is `(bot, message, user_id, limit, ..., ctx)` via `functools.partial`; it creates its own
  `ChatSender`, its own status message, sets `ctx.set_progress(...)`, builds its prompt (reply/forward
  preamble via `with_reference`, media as `content.image_ref`/`document_ref` blocks:
  `_prompt_with_reference(text, media_blocks, message, bot_id)`) and calls `run_turn`. Retry /
  Continue / follow-up buttons do the same: they post a short echo ("🔁 Retry", "💬 <suggestion>") and
  queue the turn on it, so the busy notice and the reply-to work as for a typed message; Retry re-runs
  `record.prompt_blocks or record.prompt` with `record.quiet_marker` (and `QUIET_NOTICE`), so a retried
  scheduled task with nothing new never shows `<no_update/>`.
- Action-limit accounting counts **every** `stats_events` row in 30 days: each received message,
  sent message, tool call (`memory_search`, `view_image`, `user_profile_*` and `mcp_*` included), describe
  call (now only the document fallback path and video frames) and media group. A turn with 20 tool calls
  costs ~22 of the default 50. Any new `stats_tracker.track_*` call burns user quota. `/usage` (ui router,
  no quota check) shows the user the same meter; for admins `check_user_limits` returns `used=0`, so it
  shows the summary's `actions` count next to "unlimited".
- `/mcp add` with a token deletes the command message (best effort) as soon as a token is present, before
  validation; server names (`^[a-z0-9][a-z0-9_-]{0,31}$`, lower-cased) go raw inside code spans in the
  legacy-Markdown page, every other user-derived string is escaped, URLs are shown without user info/query.
  `/profile` sends the profile as pages of ≤ `PAGE_CHARS` (4000 UTF-16 units; the file may hold 8 KB),
  sections packed whole; the Refresh/Keep-it callbacks edit ONE message and therefore show `render_one`
  (the first page plus "… N more entries not shown here").
- Reminders: created only by the agent tool `schedule_reminder` (via `reminders_store.add`; `➕ New` in
  `/reminders` only explains that); fired by `bot/jobs.py`. All reads/writes go through
  `reminders_store.update()` (per-user lock, atomic rename), including the `/reminders` UI (delay, edit
  text, snooze, delete, toggle). Times are stored in UTC and shown in the user's zone everywhere
  (`timezones.format_local`; "Times in UTC. Set your zone with /timezone." when unknown); the list shows the
  active (`pending`/`processing`) reminders grouped "⏰ Reminders" (type user) then "🤖 Tasks" (type agent),
  soonest first, 5 per page; the detail page adds "On: Mon, Tue, …" for `weekdays`, the last run (quiet or
  answered, quiet runs so far) for tasks and "Snoozed copy" for `snoozed_from`. `⏰ +1h` / `📅 +1d` push the
  time from its scheduled time, or from now when that is past (a completed one-shot becomes pending again).
  Snooze (`snooze_time`: N minutes from now, or "tomorrow" = 09:00 local, `SNOOZE_TOMORROW_HOUR`) re-schedules
  a one-shot in place and gives a periodic reminder a one-off `type: user` copy with `snoozed_from`, so the
  series does not drift; the fired message is edited into `snooze_text` (no keyboard).
- Logging: `logging.getLogger(__name__)` in the `bot` package; `agents/main.py` is still
  `print`-only and dumps every API response and tool result to stdout. Nothing writes log files.

## README vs code

- README model names (Claude Sonnet 5.5 / GPT-6.1 / GPT-6, gpt-image-2.5 / Gemini image models,
  text-embedding-3-small) now match the defaults in `app/models.py`; if they drift again, `app/models.py` wins.
- The README's feature list was realigned with the code in the v3 wave: there are no SSH or Shodan tools
  (SSH is only possible via `run_shell_script` inside the sandbox, where openssh-client is installed, with
  `network_enabled`), the accepted image formats are `image_utils.IMAGE_EXTENSIONS`, the invite default is 3.
- `.env.example` says `INVITE_LIMIT=5`; code default is 3. It also still lists `BROWSER_SERVICE_URL` (unused)
  and the obsolete `MAX_AGENT_*_ITERATIONS`.

## Known pitfalls (do not "fix" casually without checking callers)

- The iteration counter `cicles` counts individual `tool_use` blocks (a whole batch is added at once,
  so it can overshoot the limit by one batch) while the `for` loop counts API round-trips; hitting
  either limit drops pending tool calls and may return an incomplete
  "Let me check..." text. The forced final call sends the loop's tool definitions with `tool_choice: none` (thinking mode as the loop):
  the model cannot call a tool, and the request prefix (tools, system, messages) stays the one the turn's
  thinking blocks were made against. A tool-less final call made every thinking block stale (a 400 on
  enforced accounts, after which the stored thinking was stripped).
- `judge_response` treats exceptions as "yes" (accept); `critique_response` treats them as "no rewrite".
  That is why the loop, the final call AND the judge build their options with
  `models.request_options_for(self.model, …)` / `max_tokens_for`: in speed mode `self.model` is Haiku, which
  rejects `effort`/`thinking`, and a judge calling `anthropic_request_options` directly failed silently and
  accepted every answer. Never call `anthropic_request_options` with `self.model`.
- `perplexity-*` system prompts contain hard-coded 2025 dates and "You are Perplexity" identity text;
  the `generall-ai-*` prompts say "20 previous messages" regardless of `dialog_history.size`.
- Model-supplied paths go through `agents/paths.py:resolve_under` (file, image and video tools): it
  refuses anything outside `data/<uid>/` (`..`, absolute paths, symlinks out), understands the sandbox
  spelling `/home/runner/workspace/...` and the host spelling `data/<uid>/...`, and falls back to a bare
  file name in images/videos/downloads. Sandbox-patched tools rely on the container mount instead.
- `check_reminders` skips users `auth.is_authorized` rejects (blocked or removed); their reminders
  stay `pending` and fire once they are authorized again.
- A pending `✏️ Edit text` (`reminders_ui._pending_edits`: in memory, keyed by chat id, so forum topics share
  it) swallows the user's next plain text for up to 5 minutes (`EDIT_TIMEOUT`) as the new reminder text: it
  never reaches the agent, even while a turn is running (the handler is not a queue job). Voice notes, photos
  and commands are unaffected; `/reminders` and `✖️ Cancel edit` clear it, other commands do not; a restart
  forgets it (`reminders_ui.reset()` in tests).
- `set_user_timezone` writes `settings.json` but not the running turn's `user_settings` dict, and the
  `<context>` block was rendered at turn start: the model keeps seeing "User timezone: unknown" until the next
  turn. `schedule_reminder` in the same turn is fine: `UserInteractions._user_zone()` returns the zone
  `set_user_timezone` accepted this turn (`_zone_override`, set before its save's first await, put back if the
  save fails), else re-reads `UserSettings`; `schedule_reminder` yields once (`asyncio.sleep(0)`) before reading
  the zone, so the two in one concurrent tool batch agree in either block order. The same holds for `/timezone` pressed while a turn runs.
- Snoozing a periodic reminder adds a separate one-off `type: user` reminder (`snoozed_from`, its own id, no
  `weekdays`) and leaves the series alone: deleting or disabling the series does not remove the copy, which
  has its own row in `/reminders`. Snoozing a completed one-shot makes it `pending` again.
- `timezones.valid_zone` consults `ALIASES` first ("Turkey" → `Europe/Istanbul`, "Japan" → `Asia/Tokyo`, "la" →
  `America/Los_Angeles`) and only then the case- and separator-insensitive table of `available_timezones()`,
  which on Debian's tzdata (the bot image) includes the backward links ("GB", "Egypt", "US/Eastern", …): a link
  without an alias is accepted and stored as given. The table is `lru_cache`d for the process (a tzdata update
  needs a restart); `reminders_store.zone_or_utc` falls back to UTC with a warning for a stored name this host
  does not know. `to_utc` takes a trailing `Z`/`±HH:MM` as given, so a model that writes UTC times with a `Z`
  is not shifted twice.
- Quiet scheduled runs still cost: `track_message_sent`, the tool calls and the usage rows are recorded before
  `run_turn` looks at the marker, and the turn is stored in memory (see "Memory semantics"), so an hourly quiet
  task grows the transcript until `prune` summarizes it. The `agent_response` of a periodic task is overwritten
  by every run; `quiet_runs` only ever grows.
- Cancellation cannot interrupt a tool call already inside a worker thread; the turn only ends when
  that call returns (sandbox containers run to their own timeout).
- `stats_ui.py` imports `stats` at module level, so importing `bot.handlers` creates `data/stats.db`
  in the cwd (the other modules import it lazily).
- Dead code: `JudgeResponse`, `tavily_client` in `agents/main.py`, `describe_document_openai` was removed;
  `ToolIntegrator.patch_search_tools` only logs. `bot.limits.check_user_limits` is blocking SQLite with a
  side effect (`ensure_user_limit` inserts the default row): outside the middleware call it in `to_thread`.
- `MCPTools` reads the user's `mcp.json` and the admin cache synchronously on the first `tools_schema`
  access, i.e. on the event loop at turn start (KB-sized files, fine; do not grow them). `tools_schema` is
  computed once per instance (= per turn); tool discovery is cached, so a server whose tools changed shows
  the old tools until `/mcp refresh` (users) or a restart / admin `/mcp refresh` (admin servers).
- A user's MCP server must be https to a public host: `mcp_store.validate_url` refuses http, localhost /
  *.localhost and loopback/private/link-local/shared/reserved/unspecified/multicast (IPv4-mapped included)
  literals for non-admins, and `MCPTools._check_host` runs `mcp_store.check_public_host` (getaddrinfo via
  `_resolve`, in to_thread; an unresolvable name passes) before every discovery and call of a non-admin
  user's server, which also covers records written straight into `mcp.json`. The admin (shared servers and
  the admin's own `/mcp add`) may use http and private hosts. Not pinned: httpx resolves again (a DNS
  rebinding window). `download_file` / `download_webpage` still reach any host. Tests stub `mcp_store._resolve`.
- Adding, removing, toggling or refreshing an MCP server (or an admin file change), switching `mode` to or
  from `speed` (the model changes; `transcript.model` is overwritten with the latest turn's model) and any new
  or changed provider `prompt_notes` change the tool list / static prompt → `transcript.prefix_fingerprint` →
  the user's stored thinking is stripped once at the next turn. Expected; `/mcp help` tells the user.
- `_provider_notes()` is appended AFTER the formatting and follow-ups guides, so the static system prompt
  no longer ends with those guides once any provider has notes (ProfileTools and VisionTools always do); a
  test asserting `system_context.endswith(RICH_FORMATTING_GUIDE …)` must account for the `<capabilities>` block.
- Pictures are stored twice (`images/image_<uuid>.jpg` + `_v.jpg`). `prepare_downloaded_image_for_vision`
  never downscales the permanent file (it is for edits); the model's copy is always the `_v.jpg` / in-memory
  version, and the `_v.jpg` must stay because Retry re-materializes the stored `prompt_blocks`.
  `content.IMAGE_TOKENS` assumes ≤ ~1 MP, so a `view_image` with `max_px` 2048 is slightly underestimated.
  `image_for_vision` re-encodes a JPEG whose EXIF orientation is not 1 so the model sees it upright.
  `content.MAX_IMAGE_BYTES` (5 MiB) is the API's per-image limit on the base64 data; files are judged by
  `content.max_image_file_bytes()` (3/4 of it): `materialize` refuses a larger image file, `image_for_vision`
  passes a file through only up to it, `_fit_jpeg` / `save_vision_copy` target it, and an MCP image whose
  base64 is over the limit becomes the text "[image <mime> omitted: too large to show (N MB)]".
- `memory_search` is a plain substring scan of every conversation file, `profile.md` and every transcript
  of the user on each call (worker thread), and like any tool call it burns an action. Conversation
  timestamps are local time (`datetime.now()`) while transcript/context times are UTC, which only affects
  the ordering across sources in its results. Transcript messages before the first `<context>` block (the
  `<earlier_conversation_summary>` prune leaves on top, strings of a transcript seeded from dialog_history.json)
  are stamped with the transcript's `created` time ("" when absent, so they sort last).
- `ProfileStore.add` refuses an empty entry (`ValueError`), cuts entries at `MAX_ENTRY_CHARS` (600) and silently
  skips an exact duplicate (case-insensitive, Notes date ignored): check `Saved.duplicate` / `Saved.rotated`.
- `DocumentTools.execute_tool` is synchronous and sends the file from the worker thread via
  `asyncio.run_coroutine_threadsafe` on the loop captured in `__init__` (constructed on the loop in
  `run_turn`, like every provider). Calling it on the loop thread is refused ("must run in a worker
  thread"); always register it `blocking=True`. Output names are reserved process-wide between
  `unique_path()` and `release_path()` so a concurrent batch never gets the same file; `--title=<value>`
  is passed as one token (a title starting with "-" is not an option); `has_h1()` decides whether the title
  is prepended. `render_document.py`'s Markdown parser is a deliberate subset (#..###### headings at column 0,
  paragraphs, -/*/+ and 1. lists, **bold**, *italic* only when not adjacent to word characters, `code`,
  fenced code where a ```csv fence becomes an xlsx sheet, GFM pipe tables; `_text_` is NOT italic so
  snake_case survives; numeric-looking cells become numbers except identifiers (`looks_like_identifier`: a leading zero, a leading "+"
  or more than 15 significant digits stay text, so zip codes and phone numbers survive), cells starting with "=" are forced to text,
  sheet names come from the nearest heading or "Table N"); richer Markdown renders fine in md/html/pdf
  (python-markdown) but degrades to paragraphs in docx/xlsx.
- `run_script` shell-quotes every argument because `run_command` does not go through a shell: docker-py
  `shlex.split`s the string again, and with a package prelude the same string is embedded in a bash script,
  so quoting must stay POSIX. `browse_webpage` and the pdf render need ~1 GB and 10-60 s and hold a
  `MAX_SANDBOX_CONTAINERS` slot for that long.
- `ConversationEmbeddings` process-wide state (`_auto_rebuilds`, `_pending`, generation counters) lives in
  module dicts: tests call `embeddings.reset()` and set `embeddings.AUTO_REINDEX = False` for determinism
  (`tests/test_embeddings.py`).
- `message_context.SUMMARY_DETAILS_PREFIXES` ("Tool calls (", "🔧 Tools (", "💭 Thinking") must follow the
  titles `bot/status.py` gives the turn-summary sections, or a reply to the bot's own answer quotes the tool
  log. `describe_image`'s temporary JPEG lives in the system temp dir, not the workspace.
- `bot.rich` marks rich messages unsupported for the whole process on the first 404 from
  `sendRichMessage`/`sendRichMessageDraft`/a rich `editMessageText`; a wrong verdict (e.g. a transient 404) needs a restart.
  `TelegramBadRequest` on a rich send is treated as "this text", not "this server". The "buttons inside
  rich messages" flag is just as sticky: once set, every button row (settings page, Stop, actions,
  questions) is an inline keyboard until a restart.
- `ask_user_choice` blocks the user's queue while it waits (up to its timeout, 15 min at most): the user's
  next messages get the busy notice and run after it; Stop or /cancel ends the wait ("🛑 Cancelled" on the
  question). Pending questions live in memory, so after a restart a tap only gets "expired" and the buttons
  are removed. All tool calls of one message run concurrently, so several questions can be open at once.
- The `<follow_ups>` block is part of the model's answer and stays in everything stored from it (transcript,
  dialog_history.json, summaries, FAISS, the reasoning file); it is only cut for display (`TurnResult.response`,
  the turn record, TTS, drafts, the partial answer). Toggling `follow_ups` changes the static system prompt,
  so the transcript fingerprint changes and stored thinking is stripped once, like the rich-messages toggle.
- Anything the patcher rebinds imposes **its own signature** on the method. Adding a method to a tool
  class whose name a `patch_*` function targets silently replaces it; that is how image editing broke.
  `tests/test_image_tools.py` guards the image tools by asserting `tool_integrator.py` never names them.
- Image tools: `generate_image` resolves engine/quality/size as tool call → user's `image` settings → default,
  so the agent can override a preference from the request. `xhigh`/`max` are in no published SDK type for images
  and are sent as given, with a single retry at `IMAGE_QUALITY_FALLBACK` when the provider refuses them. File
  extensions are derived from the returned bytes, never asserted, and the inline-embed hint is only offered for
  formats `bot/rich.py` can actually inline.
- `agents/main.py` appends `RICH_FORMATTING_GUIDE` / `LEGACY_FORMATTING_GUIDE` to every system
  prompt after selection, then `FOLLOW_UPS_GUIDE` when `follow_ups` is on, then the providers'
  `<capabilities>` block (none of it to the subagent prompt); the `generall-ai-*` prompts still say nothing
  about formatting, pictures or the profile themselves.
- Roadmap and design notes for transcript / prompt caching / subagents: `docs/agent-roadmap.md`.
- Status edits are coalesced: while one edit is in flight further progress updates are dropped (the
  next one carries newer state), so a batch of parallel tool calls cannot storm the message. A rejected
  rich *progress* payload only disables rich progress after `MAX_PROGRESS_FAILURES`; `finish()` always
  tries the rich summary once, because it is different content. "not modified" (either phrasing, see
  `bot.ui.is_not_modified`) counts as success.
- The status message starts as a plain legacy-Markdown message (handlers create it before the settings are
  read) and becomes rich on its first `editMessageText(rich_message=...)`; if that edit is rejected the
  status stays plain for the turn (`StatusMessage.rich = False`). Headers like `💭 *Thinking...*` are
  legacy Markdown; `status.plain_header` strips the marks for rich blocks.
- Transcript mode: `AgentAnthropic.generate_response` returns ALL messages (incoming + appended); the transcript
  path slices `[base_len:]` and keeps every one of them, judge/critique/SYSTEM NOTICE (`_ephemeral`) included,
  so never insert or drop messages in the request path. `_request_messages` only strips keys starting with `_`.
- **Preserved thinking (Sonnet 5.5): the transcript is append-only.** The API binds each thinking block to the
  model, system prompt, tools and every message before it; a changed one is a 400 for accounts created on or
  after 2026-08-31 (older accounts are not enforced). So the stored transcript is exactly what was sent: the
  `<context>` block is stored with its turn, tool results are cut (`transcript.cap_text`) before the model sees
  them, empty text blocks are dropped when the assistant message is appended, and redacted_thinking blocks go
  back unchanged. The final answer is stored text-only; that is safe because no later block depends on it.
  The only edits are boundaries that strip every thinking block (`transcript.strip_thinking`): `prune` above
  `max_context_tokens` (clear old tool results to `CLEARED_MARKER` and old image/document blocks to the
  `*_CLEARED_MARKER` texts, then summarize the oldest half into one `<earlier_conversation_summary>` user
  message, down to `PRUNE_TARGET_RATIO` of the budget), a change of `transcript.prefix_fingerprint`
  (model, static system prompt or tool definitions, e.g. a new image preference, prompt type, rich-messages or
  follow-ups toggle, a mode switch to/from speed, an MCP server change, new provider notes, a new tool or a
  deploy), and the API refusing the stored blocks anyway (`AgentAnthropic.history_edited`, next bullet). Any
  new code that edits stored history must strip thinking the same way. Safety net:
  `models.thinking_binding` adds `block_binding.prefix_mismatch_behavior` (`THINKING_PREFIX_MISMATCH`) on the
  beta client; `log_input_transformations` prints any block the API dropped (that means something still edits
  history), and a refused beta is switched off for the process.
- **Stale thinking recovery** (`AgentAnthropic._stream_message`): a 400 that names thinking blocks, signatures
  or a prefix mismatch (`is_thinking_mismatch`; the binding-beta refusal is a different 400 handled inside
  `_stream_once`) is retried ONCE with the thinking blocks of the first `prefix_len` messages removed
  (`_without_thinking` on a copy; `prefix_len` is set by `generate_response` to the number of messages the turn
  started with, `None` = the whole request for direct calls). This turn's own assistant blocks are kept on
  purpose: the API wants the last tool-calling assistant message to keep its thinking. `history_edited` then
  makes every later call of the turn strip the prefix first, and `_generate_with_transcript` runs
  `strip_thinking(transcript.messages)` BEFORE appending the turn, so stored == sent. A second mismatch, or a
  prefix with nothing to strip (legacy mode reloads text-only messages), propagates. A turn that fails after
  the strip leaves the stored blocks in place and the next turn repeats the strip-and-retry. Any new code that
  strips history must keep this prefix-only rule; `tests/test_stream_retry.py` pins it.
- **What is and is not retried**: only the loop's calls and the forced final call go through `_stream_message`.
  It retries `anthropic.APIError` subclasses that `is_transient_error` accepts (RateLimitError, APIStatusError
  ≥ 500 or `overloaded`, APIConnectionError incl. timeouts) three times (2, 5, 12 s; a numeric Retry-After
  header wins up to 30 s); a raw `httpx` exception the SDK did not wrap, a `CancelledError`, 401/403/404/413
  and any other 400 propagate at once. The Haiku side calls (classifier, topic/summary, quick answers,
  transcript summaries), the judge, the critique, `describe_*` and the embeddings have no retry. The note
  reaches `on_recovery` only where `ChainOfThoughtAgent.generate_response` wired it (the user's turn,
  transcript AND legacy mode: status step `recovering`); a subagent child (`SubagentTools._child`) retries
  silently, and a failing hook is logged, never fatal. A retry after a mid-stream failure replays the whole
  stream: `forward_event` resets its text/thinking accumulators on `message_start`, so the draft (and the
  partial answer on Stop) is not doubled; keep that reset if the closure is refactored. The usage of the
  failed attempt is not counted (`trace.add_usage` sees only the final message).
- Retry on an error record replays `prompt_blocks or prompt` even when the failure came after the agent had
  answered (TTS, the Telegram send, the summary edit): that answer is already in the transcript, so the user
  gets a second turn on the same prompt. `friendly_error` matches by class first: an `APIConnectionError` wins
  over the status checks, a 413 is "too large" whatever its text, and the "(trace <uuid>)" suffix is the id of
  the `logger.exception` line.
- Subagents share the parent's `TurnBudget`: the child gets `min(max_tool_calls, remaining)` and its used calls
  are charged to the parent afterwards; a child never persists a transcript and runs with judge/critique off.
  It inherits the registered providers (vision, profile, MCP, web, document) but not `run_subagent`.

## How to make common changes

- **Add a tool**: new class in `app/agents/`, implement the contract (take `sender` if it talks to
  the user; `tools_schema`, `execute_tool` returning a string or text/image blocks; optional `prompt_notes`
  for what the model must know about the tools, instead of editing the system prompts), then ONE line in
  `ChainOfThoughtAgent.__init__` after the SubagentTools wiring: `self.agent.register_provider(Provider(...))`
  (awaited) or `register_provider(..., blocking=True)` (sync `execute_tool`, run in `asyncio.to_thread`).
  Subagents inherit it. Keep tool names unique; never let `tool_integrator.py` name the class or a method
  (add a guard test like `tests/test_web_tools.py`); if part of it must run in the sandbox, call
  `ContainerManager.run_script` with a bundled script rather than patching. The four named-slot steps are
  only for the original providers.
- **Add a sandbox script**: a file in `secure_container/scripts/` (stdlib + the image's requirements; add a
  package to `secure_container/requirements.txt` if needed, which rebuilds the image on the next start), argparse
  arguments, its own deadline (watchdog thread) and ONE JSON object as the last printed line (`json.dumps`,
  `ensure_ascii` default); the host runs it via `web_tools.get_container_manager().run_script(user_id,
  "<name>.py", args, timeout, network_enabled, mem_limit)` and reads `web_tools.last_json_line(logs)`. Files the
  script writes go under a directory made writable with `web_tools.writable_dir`.
- **Add an output document format**: `document_tools.FORMATS` (+ `SANDBOX_FORMATS` when it renders in the
  sandbox) and a branch in `scripts/render_document.py` (`build_<fmt>`), tests in `tests/test_document_tools.py`.
- **Add an MCP server**: users: `/mcp add <name> <url> [token] [sse]`; for everyone: a record in the file named
  by `MCP_SERVERS_FILE` (`data/mcp_servers.json`, see the data layout), discovered at the next start or admin
  `/mcp refresh`. Tool names become `mcp_<server>_<tool>`; the tool list change strips stored thinking once.
  Code changes: `bot/mcp_store.py` for records/validation, `agents/mcp_tools.py` for transport/result
  conversion (`convert_content`, `error_reason`), `bot/handlers/mcp_ui.py` for the command.
- **Add a user setting**: add the category to `DEFAULT_SETTINGS` in `bot/settings.py`; in `settings_ui.py`
  add a `Control` to a section of `SECTIONS` with a friendly name + hint (`advanced=True` to hide it until the
  Advanced toggle is on; this gives the page row and `apply_setting` support), add the short token to the
  token list and, for the keyboards, the overview line/button and a `show_<cat>_menu` registered in
  `_LEGACY_MENUS`; if a preset should set it, add the `(category, key)` to `SETTINGS_PRESETS`; then read it in
  `agents/main.py` via the `user_settings` dict.
- **Add a memory source to the prompt**: extend `ChainOfThoughtAgent._request_context(question, update_status,
  messages)` (transcript mode); dedupe against the transcript `messages` (`shown_memory_lines`, or a fingerprint
  tag like the profile's `<profile fingerprint="...">`) and start memory lines with `transcript.MEMORY_BULLET`
  so a line already shown in this transcript is not repeated. Never put per-turn text into the static system
  prompt (it breaks the cache and strips every stored thinking block).
- **Add a system prompt**: define `system_context_<name>` inside
  `ChainOfThoughtAgent.generate_response`, add the selection `elif`, the display-name branch, add the
  name to `SYSTEM_PROMPT_TYPES` in `bot/settings.py` (the menu is generated from it). Instructions about a
  tool belong in that provider's `prompt_notes`, not in the prompts.
- **Add a document extension**: `TEXT_EXTENSIONS` (content goes into the prompt as text) or `OFFICE_EXTENSIONS`
  (`extract_document_text` branch, extracted text into the prompt) in `bot/media.py`; `DOCUMENT_EXTENSIONS` is
  derived. For the fallback path write a `describe_<kind>` following `describe_txt` including the
  `process_large_text` fallback and route it in `describe_document`. New picture formats go into
  `image_utils.IMAGE_EXTENSIONS` (converted to JPEG) and, if the API takes them natively, into
  `content.IMAGE_MEDIA_TYPES` / `media.VISION_MEDIA_TYPES`.
- **Add a slash command / callback family**: a handler on the router with the right middleware in
  `bot/handlers/__init__.py`; copy the nearest existing handler; a user command also goes into
  `commands.COMMANDS` (menu + /help); add the family to `ROUTING` (a new message handler to `MESSAGE_ROUTING`)
  in `tests/test_cross_features.py`.
- **Add a turn button**: an action name, label and `_CALLBACK_RE` alternative in `bot/turn_actions.py`
  (`record_actions` decides when it shows; an error record gets Retry only), and a handler in
  `handlers/turn_actions_ui.py`: on `router` when it only reads the stored record, on `turn_router` (quota
  checked) when it starts an agent turn.
- **Add a reminder button action**: a button in `reminders_ui._info_keyboard` (detail page) or
  `jobs.snooze_keyboard` (fired reminders) with callback `reminder_<action>_<id>[_<value>]` (the action carries
  no underscore: `reminder_button` splits `"_", 2` and `rpartition`s the value off the payload), an `elif action
  ==` branch in `reminder_button` that mutates through `reminders_store.update()` (never `load` + `save`), times
  through `timezones` (store UTC, show `format_local`, escape the zone name in legacy Markdown), a row in
  `tests/test_cross_features.py::ROUTING` and a test in `tests/test_reminders_ui.py`.
- **Add a time zone alias or picker entry**: `timezones.ALIASES` (key = lower case without spaces, `_` or `-`,
  see `_key`; value = the canonical IANA name; aliases win over the tz database, which is how backward links
  such as "Turkey" stay out of `settings.json`) or `COMMON_ZONES` (the /timezone picker; buttons are indexed,
  so append rather than reorder); `tests/test_timezones.py::test_every_alias_and_common_zone_is_a_real_zone`
  checks that each resolves on this host.
- **Change what is retried or how an error reads**: inside the loop, `agents/main.py` (`STREAM_RETRY_DELAYS`,
  `MAX_RETRY_AFTER`, `is_transient_error`, `transient_reason`, `retry_delay`, `is_thinking_mismatch`; tests in
  `tests/test_stream_retry.py`); what the user reads, `bot/agent_runner.friendly_error` and the `ERROR_*`
  constants (`tests/test_error_recovery.py`); which buttons a failed turn gets, `turn_actions.record_actions`
  (`record.error` → Retry only). A call that should recover must go through `AgentAnthropic._stream_message`.
- **Add a line to the `<context>` block**: `ChainOfThoughtAgent._render_context` (per turn, never the static
  prompt); the local-time line reads `self.user_settings` through `timezones.zone_of`.
- **Add a media type**: a `_run_<kind>(bot, message, user_id, limit, ctx)` job plus an
  `@router.message(F.<kind>)` handler in `messages.py` that tracks and `submit`s it.
- **Add a background job**: an `async` loop started from `bot/app.py:on_startup` and appended to
  `runtime.background_tasks`; to run the agent from it, submit a `Job` to `runtime.queue`.
- **Add an image engine**: a row in `models.IMAGE_BACKENDS` (model id, provider, which operations and knobs it
  supports, variant strategy, file prefix) and, if it is a new provider, one `_<provider>_images` method in
  `agents/image_tools.py`. Nothing else dispatches on the engine name. For cost tracking add the model's
  per-modality rates to `models.IMAGE_TOKEN_PRICES`.
- **Change models**: edit the default in `app/models.py` or set the env var of the same name
  (`ANTHROPIC_MODEL`, `OPENAI_MODEL`, `VIDEO_FRAMES_MODEL`, ...; see `.env.example`). No other file
  holds a model name. Keep the option helpers honest when the new model's API differs: a
  non-reasoning OpenAI model rejects `reasoning_effort` (drop it from `OPENAI_REASONING_MODELS`), a
  pre-4.6 Claude model needs `budget_tokens` instead of adaptive thinking, and the fast model's calls stay
  plain (`ANTHROPIC_MODEL_FAST` never gets `anthropic_request_options`; Haiku 4.5 rejected `effort`; use
  `request_options_for` / `max_tokens_for` wherever the model may be either one). Read answers with
  `models.anthropic_text`, never `content[0].text`: newer models may return a thinking block first. A new
  `EMBEDDING_MODEL` needs no migration step: every user's index rebuilds itself on first use (search returns
  nothing for that user until the rebuild finishes); `EMBEDDING_DIMENSION` must still match the model's vector
  size (`_embed` refuses mismatched vectors, which fails the rebuild and is logged). Then update the table
  above and `tests/test_models.py`.
- **Change sandbox limits/mounts/network**: `ContainerManager._run_command_in_slot`; the slot cap is
  `MAX_SANDBOX_CONTAINERS`, the default memory `DEFAULT_MEM_LIMIT` (callers may pass `mem_limit`).
- **Change the sandbox image**: edit `secure_container/Dockerfile`, `entrypoint.sh` or `requirements.txt`; the
  tag changes with the content, so the next startup builds it (several minutes with Chromium) and re-points
  `:latest`. `tests/test_image_tag.py` covers the tagging.
- **Run the tests**: `pip install -r requirements-dev.txt && pytest` from the repo root (Python ≥ 3.12).
- **Smoke-check without Telegram or Docker**: from `app/`, with `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
  `TAVILY_API_KEY` set to any non-empty value, run `python -c "import bot.app, agents.main, models"`. It
  catches syntax and import errors in the whole bot and tool code. Running `main_bot.py` additionally
  needs Docker and a real token.
