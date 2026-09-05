# CLAUDE.md — GALL.AI / GenerALL.AI

Guidance for AI coding agents working in this repository. Everything below was
derived from reading the code, not the README; where the two disagree, the code
wins and the discrepancy is listed under "README vs code".

## What this is

A single-process Python Telegram bot (aiogram 3, asyncio) that wraps a Claude
tool-use agent. Users send text / voice / photos / video / audio / documents in
Telegram; the bot turns media into a text prompt, runs a per-message agent loop
with 35 tools (files, web search, code execution in a Docker sandbox,
image/video generation, reminders, SMS, S3, TTS), persists multi-layer memory
per user on disk, and replies in Telegram. Claude drives the tool loop; OpenAI
(`gpt-5.6-terra`) is only consulted by the optional critique step, which can force a
rewrite. OpenAI, Google Gemini/Veo, Perplexity, Tavily, ElevenLabs, Twilio and
Whisper otherwise serve specific side tasks.

Every chat has its own job queue: messages are processed strictly in order per
user, each as its own turn, and users never wait for each other. See "Queues".

There is a small pytest suite (`tests/`, run with `pytest` from the repo root)
covering the queue, settings, auth, reminder store, text splitting and the model settings module. There is
no CI, linter, formatter or type checking. Validate changes with the tests and
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
    queue.py                QueueManager / Job / JobContext: per-user FIFO workers, deadline, /cancel, global cap
    sender.py               ChatSender: the one object handlers AND tools use to send text/media/reactions;
                            send_markdown renders LLM answers as rich messages with tiered fallback
    rich.py                 rich-message helpers: split/convert (telegramify-markdown), inline media extraction,
                            sticky "unsupported" flag
    agent_runner.py         run_turn(): builds ChainOfThoughtAgent, status edits, streaming, voice reply, answer, reasoning file
    media.py                Whisper transcription, Claude/GPT image+document description, video frames, TTS, ffmpeg setup
    streaming.py            throttled draft streaming (rich drafts + <tg-thinking> block, plain draft fallback)
    auth.py                 AuthStore (`auth` singleton): allow/block lists + invite codes in data/userlist.json
    limits.py               check_user_limits() (rolling 30-day action quota via stats.db)
    settings.py             DEFAULT_SETTINGS + UserSettings (single source of truth)
    ui.py                   answer_md/edit_md helpers: legacy Markdown with plain-text fallback
    jobs.py                 reminders scheduler loop (10 s): sends user reminders, queues agent reminders
    clients.py              lazily built OpenAI/Whisper/Anthropic clients + model name constants
    handlers/
      __init__.py           build_root_router(): public → admin → ui → chat, middleware per router
      middleware.py         AuthMiddleware(require_admin, check_limits) injects user_id / limit kwargs
      messages.py           text/voice/video/audio/photo(+albums)/document handlers, busy notice, /cancel
      commands.py           /start, /invite, /listusers, /voice
      settings_ui.py        /settings inline keyboards
      reminders_ui.py       /reminders inline keyboards
      stats_ui.py           admin /stats
  reminders_store.py        RemindersStore (`reminders_store` singleton): per-user locked JSON read/modify/write
  agents/
    main.py                 Agent core: AgentAnthropic (tool loop) + ChainOfThoughtAgent
                            (memory, prompts, routing, persistence). The 4 system prompts live here.
    file_ops.py             FileOperations: files under data/<uid>/, downloads, S3, zip, send file
    search_tools.py         SearchTools: Tavily search, Perplexity deep_research, memory_search
    code_tools.py           CodeTools: execute_python (patched to run in Docker at startup)
    terminal_tools.py       TerminalTools: run_command (patched; gains run_shell_script/install_package)
    time_tools.py           TimeTools: timezone helpers
    sms_tools.py            SMSTools: Twilio send_sms
    image_tools.py          ImageTools: Gemini / GPT Image 2 / DALL-E generation, editing, composition
    video_tools.py          VideoTools: Google Veo 3.1 text→video, image→video, extend, interpolate
    user_interactions.py    UserInteractions: mid-run Telegram messages, TTS voice, reactions,
                            schedule_reminder, send file from content
    embeddings.py           ConversationEmbeddings: OpenAI ada-002 + FAISS per user
    system_tools.py         SystemTools: apt/pip/service/network tools — NOT wired into the agent
    paths.py                resolve_under(): safe resolution of model-supplied paths inside data/<uid>
    __init__.py             imports SystemTools; the package object is what the patcher needs
  secure_container/
    main.py                 initialize_secure_containers() / cleanup_containers()
    container_manager.py    ContainerManager: builds image "secure-container:latest", runs one
                            throwaway Docker container per tool call, MAX_SANDBOX_CONTAINERS semaphore
    tool_integrator.py      ToolIntegrator: monkey-patches tool classes at startup (see below)
    secure_tool_wrapper.py  SecureToolWrapper: thin routing layer to ContainerManager
    Dockerfile, entrypoint.sh, requirements.txt   the sandbox image (python:3.12-slim + tools,
                            non-root user "runner", passwordless sudo for apt only)
  stats.py                  StatsTracker singleton: SQLite data/stats.db, per-user action limits
  image_utils.py            JPEG/HEIC helpers (pillow-heif registered on import)
  voice/                    VoiceManager: ElevenLabs voice ids (voices.json) + per-user choice (config.json)
  Dockerfile, requirements.txt   bot image (python:3.12-slim + git, docker.io, ffmpeg)
tests/                      pytest suite (pytest.ini sets pythonpath=app, asyncio_mode=auto)
.github/workflows/deploy.yml  manual production deploy (workflow_dispatch, environment PROD), see "Running"
deploy/                     render_env.py (PROD secrets+vars → .env, runs on the runner) and
                            remote_deploy.sh (git checkout, .env swap, compose build/up, health check; runs on the server)
requirements-dev.txt        app requirements + pytest, pytest-asyncio
docker-compose.yml          services: telegram-bot-api (local Bot API server) + bot
.env.example                template for .env (WORKSPACE_ROOT comes from docker-compose.yml; BROWSER_SERVICE_URL is unused)
static/                     README images only
data/                       runtime state, git-ignored (see "On-disk data layout")
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
  down. The only scheduled work is the reminders loop every 10 s.
- Production deploys are a manual GitHub Actions run (`.github/workflows/deploy.yml`, Deploy button,
  `environment: PROD`) plus `deploy/`: `render_env.py` turns **every** PROD secret/variable into the
  server's `.env` (secrets win; `SERVER_*` are the SSH settings and are excluded), the workflow uploads
  it as `.env.new` and streams `remote_deploy.sh` over SSH (refuse on dirty tracked files, checkout the
  ref, swap `.env` keeping `.env.bak`, compose build/up, wait for "is running"). The server's `.env` is
  therefore generated: change the PROD environment, not the file. Tests: `tests/test_render_env.py`.

## Startup order and the monkey-patching (read this first)

`main_bot.py` executes, in order:

1. `load_dotenv()`, `logging.basicConfig`, `Config.validate()`.
2. `initialize_secure_containers()` (`secure_container/main.py`): connects to Docker, builds
   `secure-container:latest` from `app/secure_container/Dockerfile` if missing, then
   `ToolIntegrator.patch_all_tools(agents)` **monkey-patches the tool classes in place**:
   - `TerminalTools.run_command` → runs in a sandbox container; `run_shell_script` and
     `install_package` methods **and their tool schemas are added** to the class.
   - `CodeTools.execute_python` → sandbox; a `network_enabled` param is added to its schema.
   - `FileOperations`: only method names that exist on the class AND are in
     `SecureToolWrapper.SECURE_TOOL_METHODS` get patched. That is **only `create_directory` and
     `delete_file`**. `list_directory`, `read_text_file`, `create_text_file`, downloads, S3, zip all
     run on the host (i.e. inside the bot container) unsandboxed.
   - `SearchTools.memory_search` → sandbox version that only greps `**/*.txt` and prints a JSON list
     of `{file, context, position}`. The host implementation in `search_tools.py` is dead once patched.
   - `ImageTools`: patch targets `_generate_image`, which no longer exists → no-op.
   - `ConversationEmbeddings`: wrapped with logging only.
   - `SystemTools`: fully patched, but the class is never given to the agent, so unreachable.
   `ContainerManager` is instantiated twice here (once kept for cleanup, once inside the wrapper).
3. **Only then** `import agents.main` — so every `ChainOfThoughtAgent` sees patched classes.
   `bot.agent_runner` also imports `agents.main` lazily, inside `run_turn`, for the same reason.
4. `bot.app.run()`: build `Bot` (local API session when configured), `Dispatcher`, include the root
   router, `delete_webhook(drop_pending_updates)`, `start_polling`. `on_startup` loads the auth store,
   configures ffmpeg for pydub, and starts the reminders loop; `on_shutdown` cancels background tasks
   and running jobs; `main_bot` then calls `cleanup_containers()`.

Consequences: the tool list the model actually sees is **not** what you read in `agents/*.py`
alone; the source of truth is source file + `tool_integrator.py`. When you add or rename a method on
a tool class, check `SECURE_TOOL_METHODS` and the `patch_*` functions, or you may silently move
execution between host and sandbox.

## Queues, isolation and cancellation

- `bot/queue.py` gives every chat a FIFO `asyncio.Queue` and a lazily started worker task. Handlers
  only validate, enqueue a `Job(user_id, label, run)` and return within milliseconds.
- If the user already has a running or queued job, `messages.notify_busy` immediately replies
  "⏳ I'm still working on your previous request (label, elapsed, step) …" with a Stop button
  (`queue_cancel` callback). The new message is processed later as its own turn, in order.
- `/cancel` (and the Stop button) cancels the running task and drops the queue. Cancellation is
  cooperative: handlers catch `CancelledError`, edit their status message to "🛑 Stopped." and re-raise.
  A tool call already inside a worker thread (Docker, HTTP) cannot be interrupted; the turn ends when
  the thread returns. Sandbox containers keep running until their own timeout.
- Isolation is real only because blocking work is off the event loop: `AgentAnthropic.execute_tool`
  runs synchronous providers (search, code, terminal, sms) via `asyncio.to_thread`; image/video tools
  wrap SDK calls in `to_thread` and poll with `asyncio.sleep`; media pre-processing uses `to_thread`
  for pydub/ffmpeg/OpenAI/pandas. **Any new blocking call must be wrapped the same way**, otherwise it
  stalls every user again.
- Global caps: `MAX_CONCURRENT_TURNS` (asyncio semaphore in `QueueManager`) and
  `MAX_SANDBOX_CONTAINERS` (threading semaphore in `container_manager.py`). `TURN_TIMEOUT_SECONDS`
  cancels a stuck job and tells the user; the worker continues with the next job.
- Agent reminders go through the same per-user queue (`bot/jobs.py`), so they never interleave with
  the user's own turn. While queued/running the record has `status: "processing"`; a stale
  `processing` older than twice the turn timeout is reset to `pending` on the next scan.
- Forum topics: the queue key is the chat id, so all topics of one chat share one queue.

## Request flow (text message)

```
messages.on_text (bot/handlers/messages.py)      [AuthMiddleware(check_limits=True) already ran:
  user_id = str(message.chat.id)                   authorized, quota ok, kwargs user_id/limit injected]
  stats_tracker.track_message_received; submit(Job(run=_run_text))  → busy notice if needed → return
worker → _run_text → bot.agent_runner.run_turn(bot, user_id, chat_id, prompt, thread_id, reply_to, ctx, limit)
    sender = ChatSender(bot, chat_id, thread_id, reply_to_message_id)
    UserSettings(user_id).save() (persists backfilled defaults) → user_settings dict
    status = sender.send_text("💭 *Thinking...*"); update_status() edits it (usage line, step, iteration)
    agents.ChainOfThoughtAgent(model=models.ANTHROPIC_MODEL, user_id, sender, user_settings, thread_id)
    .generate_response(question, update_status, on_text_chunk)
       ├─ question = "Message received time in UTC+0: <ts>\n\n" + text     (stored everywhere in this form)
       ├─ pick system prompt by settings.system_prompt.type (4 inline prompts; unknown → generall-ai-v2)
       ├─ semantic_search: FAISS hits appended to the system prompt (to_thread; failure → skipped)
       ├─ context_memory = fixed intro pair
       │     + conversation summaries (summarization_history)   [reads data/<uid>/conversations/*.json]
       │     + dialog_history.json, then context_memory[:-2]     (drops the newest pair; if the file is
       │       empty it eats the summaries pair or the intro pair instead)
       │     + short_term_memory.json messages whose FIRST block is text (reasoning_context)
       │     + the question                       (all three gated by short_term_memory.enabled too)
       ├─ _classify_complexity (Haiku) → "simple": tool-less Haiku answer; on any error fall back ↓
       ├─ AgentAnthropic.generate_response: for iteration in range(tools.max_iteration):
       │     messages.stream (always) (model, system, tools=get_tools_schema(), output_config.effort, adaptive thinking,
       │        max_tokens 64000 deep / 16000 light; thinking tokens count against it)
       │     stop_reason == tool_use AND tools.enabled AND cicles < max → run each block,
       │        append ONE user msg of tool_results, continue      (cicles counts tool_use BLOCKS)
       │     gate fails → tool_use blocks silently dropped, text goes on to critique/judge/return
       │     else optional critique (OpenAI gpt-5.6-terra) / judge (Claude yes/no) re-prompts → return
       │     for-loop exhausted → forced "SYSTEM NOTICE" final call without tools (same thinking mode)
       └─ strip pre-loaded context from thread_messages; persist memory inside try/except
          (a failed Haiku summary is logged, the answer is still returned) → (response, messages)
    stats_tracker.track_message_sent
    speak? → ElevenLabs TTS (to_thread) → sender.send_voice
    sender.send_markdown(response, edit=status)
       rich (default): split ≤32 KB → sendRichMessage(markdown) → on parse error rich HTML
       (telegramify-markdown) → on 404 sticky MarkdownV2 → legacy Markdown → raw; status deleted after
       legacy (rich_messages off): ≤4000 edit status in place; else new chunks + notice
    send_reasoning_file (if reasoning_context.enabled; document from bytes)
    errors → status edited to "❌ An error occurred. Trace ID: …", returns None; CancelledError → "🛑 Stopped."
```

Media handlers (`_run_voice`, `_run_video`, `_run_audio`, `_run_images`, `_run_document` in
`messages.py`) are jobs with the same shape and only differ in how they build the prompt:

- Voice notes: downloaded to `temp_audio/`, converted, Whisper-transcribed, deleted; only the
  transcription reaches the agent (`speak=True` adds a TTS voice reply). Nothing is persisted.
- Audio files: saved to `data/<uid>/audio/<original name>`; no transcription; prompt carries metadata + path.
- Photos/albums: JPEG copy to `data/<uid>/images/image_<uuid>.jpg`, described twice (Claude, then GPT)
  before the agent runs. Albums are buffered per `(chat_id, media_group_id)` and flushed 10 s after
  the last photo into ONE job.
- Video/video notes: saved to `data/<uid>/videos/`, 4 frames to `data/<uid>/images/video_frame_*.jpg`,
  frames described by `gpt-5.6-luna`, audio by Whisper; `speak=True`.
- Documents: saved to `data/<uid>/documents/<lowercased name>`, then `media.describe_document`
  (PDF via Claude document block; txt/json/docx/xlsx extracted; >100k chars map-reduced). Video
  extensions are rerouted to the video job, JPG/JPEG/HEIC/HEIF to the image job, everything else rejected.

Return-type note: both `generate_response` methods are annotated `-> str` but return
`(text, messages)` tuples.

## The tool contract

A tool provider is duck-typed. It must expose:

- `tools_schema` → `list[dict]` of Anthropic tool definitions `{name, description, input_schema}`.
  Either a `@property` (file/search/code/terminal/time/sms) or an instance attribute set in
  `__init__` (image/video/user_interactions). If you make it a property, the patcher can wrap it.
- `execute_tool(self, tool_name, tool_args) -> str`, sync or async.

Tools that talk to the user receive a `sender` (`bot.sender.ChatSender`) in their constructor, never
a Telegram update: `sender.send_text(md)`, `send_markdown(md)` (splits, fallback), `send_document(path|bytes,
filename, caption)`, `send_photo`, `send_video`, `send_voice(bytes)`, `react(emoji)` (only when the turn
has a user message), `typing()`. Captions are truncated to 1024. This is what makes tool sends work in
scheduled agent reminders too. `sender` may be `None` for unit tests; guard before using it.

Wiring is entirely manual in `agents/main.py` and there is no registry:

1. `AgentAnthropic.__init__`: a `self.<name> = None` slot.
2. `AgentAnthropic.get_tools_schema`: `tools.extend(self.<name>.tools_schema)` — fixed order
   file_ops, search, code, terminal, time, image, video, sms, user_interactions.
3. `AgentAnthropic.execute_tool`: an `elif` that checks `tool_name in [t["name"] for t in
   provider.tools_schema]`. Async providers are awaited (file_ops, image, video, user_interactions);
   synchronous ones are run via `asyncio.to_thread` (search, code, terminal, sms); time is called
   directly. First match wins, so tool names must be globally unique.
4. `ChainOfThoughtAgent.__init__`: construct the provider and assign `self.agent.<name> = self.<name>`.
   Constructor shapes: `(user_id, sender)` for file/terminal/image/video/user_interactions,
   `(user_id)` for search/code, no args for time/sms.

Tool results are `str(result)` into a single `tool_result` block; return strings. Every call is
recorded via `stats_tracker.track_tool_used` (and counts against the user's action limit). Tool
schemas are sent to the API even when `tools.enabled` is false.

Effective tool list as the model sees it (after patching):

| Provider | Tools | Runs where |
|---|---|---|
| FileOperations | `list_files`, `create_file`, `read_file`, `download_file`, `download_webpage`, `upload_to_s3`, `create_zip_archive`, `send_file_path_to_user_via_telegram` | bot process, paths resolved under `data/<uid>/` by `agents/paths.resolve_under` (escapes refused) |
| FileOperations | `create_directory`, `delete_file` | sandbox container (a fresh `docker run` per call), paths relative to `data/<uid>` |
| SearchTools | `search_web` (Tavily), `deep_research` (Perplexity `sonar*`) | bot process, worker thread |
| SearchTools | `memory_search` | sandbox; `*.txt` only |
| CodeTools | `execute_python` (+`network_enabled`) | sandbox |
| TerminalTools | `run_command` (no network option), `run_shell_script` (+`network_enabled`), `install_package` (apt, persisted, network on) | sandbox |
| TimeTools | `get_time_in_timezone`, `list_timezones` | bot process |
| ImageTools | `image_generator`, `image_editing`, `image_composition` (Gemini or GPT Image 2), `generate_multimodal_image_and_text` (Gemini), `generate_image_dall_e` (obsolete) | bot process (SDK calls in worker threads), sends results itself as documents |
| VideoTools | `video_generator`, `image_to_video_generator`, `video_from_reference_images`, `video_interpolation_generator`, `video_extension_generator` | bot process (Veo in worker threads, async polling up to 5 min) |
| SMSTools | `send_sms` | bot process, worker thread |
| UserInteractions | `send_user_telegram_message`, `send_voice_message`, `set_message_reaction`, `schedule_reminder`, `send_file_content_to_user_via_telegram` | bot process |

Unreachable: all 11 `SystemTools` tools (`install_package`, `run_shell_script` duplicates plus
`check_system_info`, `manage_service`, `monitor_process`, `network_diagnostics`,
`list_installed_packages`, `remove_package`, `install_python_package`,
`list_installed_python_packages`, `remove_python_package`). `ContainerManager` does honor a
persisted pip list (`data/<uid>/installed_python_packages.txt`) at every sandbox run; no bot code
writes it, but the model can create the file with `create_file` and it will then be replayed.

## Sandbox (secure_container) mechanics

- Image `secure-container:latest` is built lazily on first startup from `app/secure_container/`
  (slow, many apt layers). Rebuild by deleting the image; the code never rebuilds an existing one.
- Every sandboxed call = one new container `secure-container-<uid>-<8 hex>`: bind-mount
  `$WORKSPACE_ROOT/data/<uid>` → `/home/runner/workspace` (rw, shared), cwd there, user `runner`,
  `mem_limit=512m`, one CPU, `network_mode=none` unless `network_enabled` or packages must be
  installed, removed in `finally`. `run_command` first acquires one of `MAX_SANDBOX_CONTAINERS`
  slots (blocking, in the calling worker thread).
- **`run_command` does not go through a shell**: docker-py `shlex.split`s the string and
  `entrypoint.sh` does `exec "$@"`. Pipes, `&&`, redirects, `cd`, globs need `bash -c '...'` or
  `run_shell_script`. Exception: when a persisted package list exists the command is embedded in a
  bash prelude script, so shell syntax suddenly works.
- Python code is written to `data/<uid>/temp_code.py`, shell scripts to `temp_script.sh`, and a
  package-install prelude to `temp_setup.sh`; all are deleted after the run.
- Persistence across runs is by re-installing: `installed_packages.txt` (apt, via `sudo apt-get`)
  and `installed_python_packages.txt` (`pip install --user`) are replayed before **every** sandboxed
  call (including `delete_file`/`create_directory`/`memory_search`) when present, which forces
  network on and adds many seconds per call. `python3-*` apt packages also get a best-effort
  `pip install --user <name-without-prefix>`.
- Output = `container.logs()` (stdout+stderr), always prefixed by the entrypoint banner (~10 lines).
  Tool results are therefore never clean JSON. Non-zero exit → `Error (exit code N):\n<logs>`;
  a timeout → `Error executing command: <exc>` and the logs are lost with the container.
- Security boundary: the sandbox covers `run_command`, `execute_python`, `run_shell_script`,
  `install_package`, `delete_file`, `create_directory`, `memory_search`. `list_files`, `read_file`,
  `create_file`, downloads, zip, S3 and every API-backed tool run in the bot container, which is
  privileged with the Docker socket mounted.

## Memory and on-disk data layout

All paths are relative to the process cwd (`app/` in Docker → `/app/data`, bind-mounted to `./data`).

```
data/
  userlist.json                     {users:[chat_id...], blocked_users:[...], invites:{inviter:{code:{created_at,used_by}}}}
  stats.db (+ -wal/-shm)            SQLite: stats_events(user_id,event_type,event_subtype,extra_data,timestamp), user_limits(user_id,action_limit)
  <chat_id>/
    settings.json                   per-user settings (see table); merged over defaults on load, unknown keys preserved
    conversations/conversation_<YYYYmmdd_HHMMSS>_<topic>.json
                                    one file per turn: {timestamp, thread_id, topic, summary, question, response, full_history}
    short_term_memory/[topic_<thread>_]dialog_history.json
                                    last dialog_history.size Q/A PAIRS (size*2 entries) as plain strings
    short_term_memory/[topic_<thread>_]short_term_memory.json
                                    block-structured messages of (mostly) the last turn ("reasoning context")
    embeddings/faiss_index.bin, metadata.json
                                    IndexFlatL2(1536) over "Question: ..\nAnswer: .." with text-embedding-ada-002
    reminders/reminders.json        list of {id (8 hex), user_id, text, time (ISO UTC), type: user|agent,
                                    status: pending|processing|completed|failed, created_at, is_periodic, period_type,
                                    period_interval, last_triggered, next_trigger, enabled, [completed_at],
                                    [agent_response], [processing_since]}; written atomically under a per-user lock
    images/                         image_<uuid>.jpg (uploads), video_frame_<uuid>.jpg, generated_*/transformed_*/composed_*/
                                    gpt_generated_*/gpt_edited_*/story_image_*, edit_input_* (transient)
    videos/                         video_<uuid>.mp4 (uploads), veo3_*.mp4
    audio/, documents/              uploads under their ORIGINAL file names (documents lowercased) → re-uploads overwrite
    downloads/                      download_file / download_webpage(save_to_file) / create_zip_archive outputs
    installed_packages.txt, installed_python_packages.txt, temp_code.py, temp_script.sh, temp_setup.sh
temp_audio/ temp_photos/ temp_docs/   transient, cwd-relative, deleted in finally
app/voice/config.json               MUTATED at runtime by /voice (per-user voice id). Lives inside the image, not data/.
```

Memory semantics worth knowing before touching `ChainOfThoughtAgent.generate_response`:

- Forum topics: `message_thread_id` only prefixes the two short-term files (`topic_<id>_...`).
  Conversation summaries and the FAISS index are shared across threads.
- Summaries are read with `Path.glob` (unsorted) and sliced
  `[:-dialog_history.size][-summarization_history.size:]`. Sizes come clamped 1..50 from the UI;
  a hand-edited `dialog_history.size` of 0 yields nothing, and `summarization_history.size` of 0
  injects **every** conversation file.
- Old conversation files without `summary`/`topic`/`timestamp` are tolerated; unreadable JSON is
  skipped. There is no migration. The conversation filename embeds the raw Haiku "topic" unsanitized.
- Persistence at the end of a turn is wrapped in try/except: a failed Haiku topic/summary or
  embedding call is logged and the answer is still delivered (previously the user saw an error).
- Reloading `short_term_memory.json` keeps only messages whose first block is `text`. With thinking
  on, intermediate tool-calling assistant turns start with a `thinking` block and are dropped; the
  final answer (re-appended as text-only) is kept. Tool-result user messages are never reloaded.
- No prompt caching anywhere (`cache_control` count is 0); the full tool schema and system prompt
  (the Perplexity ones are very long) are resent every iteration.

## User settings (`data/<uid>/settings.json`)

`bot/settings.py` holds `DEFAULT_SETTINGS`, the single source of truth. `UserSettings` deep-copies
the defaults, merges the file over them (unknown keys are kept, never `KeyError`), and `save()`s on
every `set`. `run_turn` saves once per turn so new defaults reach disk. Agent code reads them as
`user_settings.get(cat).get(key)`, so every category must stay in `DEFAULT_SETTINGS`.

| Category | Keys (default, UI clamp) | Consumer |
|---|---|---|
| `summarization_history` | enabled (true), size (5, 1..50) | conversation summaries in context |
| `dialog_history` | enabled (true), size (10, 1..50) | rolling Q/A window |
| `reasoning_context` | enabled (true) | reload last turn's blocks; also gates the reasoning file |
| `short_term_memory` | enabled (true) | master switch for the three above |
| `critique` | enabled (false), max_iteration (5, 1..300) | OpenAI critique re-prompts |
| `judge` | enabled (false), max_iteration (5, 1..300) | Claude yes/no completeness re-prompts |
| `tools` | enabled (true), max_iteration (20, 1..300) | the agent loop bound (this, not the env var) |
| `semantic_search` | enabled (true), max_results (3, 1..20) | FAISS hits in system prompt |
| `thinking` | enabled (true) | on ("deep"): adaptive thinking, `display: summarized`, effort `ANTHROPIC_EFFORT` (high), max_tokens `ANTHROPIC_MAX_TOKENS` (64000); off ("light"): still adaptive, `display: omitted`, effort `ANTHROPIC_EFFORT_LIGHT` (low), max_tokens 16000. Never `disabled` (`models.anthropic_request_options` / `anthropic_max_tokens`) |
| `rich_messages` | enabled (true) | answers as Telegram rich messages (native GFM); off = legacy Markdown v1 path; also selects the `<formatting>` prompt section and rich vs plain streaming drafts |
| `system_prompt` | type ("generall-ai-v2"; also generall-ai-v1, perplexity-deep-research, perplexity-r1) | prompt selection |

Edited through `/settings` (`bot/handlers/settings_ui.py`). callback_data is
`settings_<token>[_<action>[_<value>]]` parsed with a plain `split("_")`, where `<token>` is a short
name, not the JSON key: summarization, dialog, reasoning, memory (= short_term_memory), critique,
judge, tools, semantic, thinking, rich, main. `system_prompt` is special-cased with `startswith`.

## Models and external services

**Single source of truth: `app/models.py`.** Every model name is a module constant there with an
environment override of the same name (read once at import, after `load_dotenv()`), and the request
options that belong to a model live next to it: `anthropic_request_options(thinking, effort=None)` returns
`{"output_config": {"effort": ...}}` plus adaptive `thinking` (`display` `summarized` for `True`, `omitted` for
`False`, omitted parameter for `None`; effort `ANTHROPIC_EFFORT` vs `ANTHROPIC_EFFORT_LIGHT`), `anthropic_max_tokens(thinking)`
the matching ceiling, `anthropic_text(message)` the text blocks (the first block may be thinking), and
`openai_reasoning_options(model)` returns `{"reasoning_effort": OPENAI_REASONING_EFFORT}` for the two
OpenAI reasoning models and `{}` for anything else. No other file holds a model literal.

| Purpose | Model / API (default; env override) | Request options | Used by |
|---|---|---|---|
| Agent loop, judge, final compile | `claude-sonnet-5` via `anthropic.AsyncAnthropic` (`ANTHROPIC_MODEL`) | adaptive thinking always; loop and final call: effort `high` + summarized display when the user's `thinking` setting is on, effort `low` + omitted display when off; judge: light mode, max_tokens 2048; the loop and final call always stream | `agents/main.py` |
| Document & image description | `claude-sonnet-5` (`ANTHROPIC_MODEL`) | light mode (adaptive, effort `low`, display omitted); max_tokens 4096 / 8192 / 16000 leave room for thinking; text read with `anthropic_text` | `bot/media.py` via `bot/clients.py` clients |
| Topic/summary, complexity classifier, "simple" answers | `claude-haiku-4-5` (`ANTHROPIC_MODEL_FAST`) | none (Haiku rejects `effort`) | `agents/main.py` |
| Critique | `gpt-5.6-terra` structured output (`beta.chat.completions.parse`) (`OPENAI_MODEL`) | `reasoning_effort` = `high` (`OPENAI_REASONING_EFFORT`); no `temperature`/`max_tokens` | `agents/main.py` |
| GPT vision on photos (second description after Claude) | `gpt-5.6-terra` (`OPENAI_MODEL`) | `reasoning_effort` `high` | `bot/media.py` |
| Video frame description | `gpt-5.6-luna` (`VIDEO_FRAMES_MODEL`) | `reasoning_effort` `high`; no `max_completion_tokens` (it would cap reasoning + answer together) | `bot/media.py` |
| Transcription | `whisper-1` (`WHISPER_MODEL`) via a second client keyed by `OPENAI_API_KEY_WHISPER` (falls back to `OPENAI_API_KEY`); >24 MB chunked | none | `bot/media.py` `transcribe_audio` |
| Embeddings | `text-embedding-ada-002` (`EMBEDDING_MODEL`), dim 1536 (`EMBEDDING_DIMENSION`, must match; existing FAISS indexes are not migrated) | none | `agents/embeddings.py` |
| Image gen/edit | `gemini-3.1-flash-image-preview` (Normal, `GEMINI_IMAGE_MODEL_FLASH`), `gemini-3-pro-image-preview` (Pro, `GEMINI_IMAGE_MODEL_PRO`), `gpt-image-2-2026-04-21` (GPT, `GPT_IMAGE_MODEL`), `dall-e-3` (legacy, `DALLE_MODEL`) | none | `agents/image_tools.py` |
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
| `STREAMING_ENABLED` | bot/config AND agents/main (read independently, both after dotenv) | draft streaming: `send_rich_message_draft` in rich mode (thinking as `<tg-thinking>`), else `send_message_draft`; errors swallowed |
| `MAX_CONCURRENT_TURNS` (8), `TURN_TIMEOUT_SECONDS` (1800), `DROP_PENDING_UPDATES` (true) | bot/config | queue caps, see "Queues" |
| `MAX_SANDBOX_CONTAINERS` (4) | secure_container/container_manager | concurrent sandbox containers |
| `THREAD_POOL_SIZE` (32, min 8) | bot/app | size of the default executor used by `asyncio.to_thread` |
| `WORKSPACE_ROOT` | secure_container | host path of the checkout (see Running); set in compose, not `.env.example` |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_API_KEY_WHISPER`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `PERPLEXITY_API_KEY`, `ELEVENLABS_API_KEY` | various | see models table; `OPENAI_API_KEY` and `GOOGLE_API_KEY` needed to import `agents.main` |
| `ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT`, `ANTHROPIC_MODEL_FAST`, `OPENAI_MODEL`, `VIDEO_FRAMES_MODEL`, `OPENAI_REASONING_EFFORT`, `WHISPER_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `GEMINI_IMAGE_MODEL_FLASH`, `GEMINI_IMAGE_MODEL_PRO`, `GPT_IMAGE_MODEL`, `DALLE_MODEL`, `VEO_MODEL`, `PERPLEXITY_MODEL`, `TTS_MODEL` | `models` | optional overrides of the model defaults (blank = default), read once at import; see "Models and external services" and `.env.example` |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | sms_tools | optional; tool returns an error string if unset |
| `S3_HOST`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME`, `S3_PATH_TO_STORE` | file_ops | optional |
| `MAX_IMAGE_RESOLUTION_VISION` (1024), `MAX_IMAGE_RESOLUTION_EDIT` (4096) | bot/media, image_tools | downscale before vision / edit |
| `FFMPEG_DIR` | bot/media (Windows only) | directory holding ffmpeg.exe when not on PATH |
| `MAX_AGENT_TOOLS_ITERATIONS`, `MAX_AGENT_CRITIQUE_ITERATIONS` | read, never used | obsolete; per-user settings replaced them |
| `BROWSER_SERVICE_URL` | nobody | listed in `.env.example`, unused |

Never commit `.env`, `data/`, `temp_photos/`, `temp_docs/`, `temp_audio/` (git-ignored by exact
name; a new temp dir needs its own `.gitignore` line).

## Telegram layer conventions (aiogram)

- Routers and middleware (`bot/handlers/__init__.py`): `public` (/start, /invite: no auth),
  `admin` (/listusers, /stats: `AuthMiddleware(require_admin=True)`), `ui` (/voice, /settings,
  /reminders, /cancel, `queue_cancel`: `AuthMiddleware()`), `chat` (content handlers:
  `AuthMiddleware(check_limits=True)`). Handlers never check auth themselves; they receive `user_id`
  (str chat id) and, in `chat`, `limit` as kwargs. Unknown `/commands` match nothing and are ignored.
- Commands: `/start [invite_<code>]`, `/invite [code]`, `/voice`, `/settings`, `/reminders`,
  `/cancel`, `/stats` (admin), `/listusers` (admin).
- Callback prefixes: `voice_`, `settings_`, `reminder_`/`reminders_`, `noop`, `queue_cancel`, `stats_`.
  A new keyboard family needs a unique prefix and a `@router.callback_query(F.data.startswith(...))`
  handler on the router with the right middleware. Read the chat from `callback.message.chat.id`
  and guard `isinstance(callback.message, Message)`.
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
  user/LLM text embedded in Markdown with `bot.ui.escape_markdown`. Invite/admin replies use HTML.
- Keyboards: `InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=..., callback_data=...)]])`
  (keyword arguments are mandatory in aiogram).
- Every content handler: track_message_received → `submit(message, Job(...))`. The job function
  signature is `(bot, message, user_id, limit, ..., ctx)` via `functools.partial`; it creates its own
  `ChatSender`, its own status message, sets `ctx.set_progress(...)`, and calls `run_turn`.
- Action-limit accounting counts **every** `stats_events` row in 30 days: each received message,
  sent message, tool call, describe call and media group. A turn with 20 tool calls costs ~22 of the
  default 50. Any new `stats_tracker.track_*` call burns user quota.
- Reminders: created only by the agent tool `schedule_reminder` (via `reminders_store.add`); fired
  by `bot/jobs.py`. All reads/writes go through `reminders_store.update()` (per-user lock, atomic
  rename), including the `/reminders` UI.
- Logging: `logging.getLogger(__name__)` in the `bot` package; `agents/main.py` is still
  `print`-only and dumps every API response and tool result to stdout. Nothing writes log files.

## README vs code

- README model names (Claude Sonnet 5 / GPT-5.6) now match the defaults in `app/models.py`
  (`claude-sonnet-5`, `claude-haiku-4-5`, `gpt-5.6-terra`, `gpt-5.6-luna`); if they drift again, `app/models.py` wins.
- README lists SSH and Shodan tools; neither exists. SSH is only possible via `run_shell_script`
  inside the sandbox (openssh-client is installed there) with `network_enabled`.
- README lists PNG/GIF/BMP/WEBP as supported images; as documents only JPG/JPEG/HEIC/HEIF are.
- `.env.example` says `INVITE_LIMIT=5`; code default is 3.

## Known pitfalls (do not "fix" casually without checking callers)

- The iteration counter `cicles` counts individual `tool_use` blocks while the `for` loop counts
  API round-trips; hitting either limit drops pending tool calls and may return an incomplete
  "Let me check..." text. The forced final call has no tools (thinking mode as the loop).
- `judge_response` treats exceptions as "yes" (accept); `critique_response` treats them as "no rewrite".
- `perplexity-*` system prompts contain hard-coded 2025 dates and "You are Perplexity" identity text;
  the `generall-ai-*` prompts say "20 previous messages" regardless of `dialog_history.size`.
- Model-supplied paths go through `agents/paths.py:resolve_under` (file, image and video tools): it
  refuses anything outside `data/<uid>/` (`..`, absolute paths, symlinks out), understands the sandbox
  spelling `/home/runner/workspace/...` and the host spelling `data/<uid>/...`, and falls back to a bare
  file name in images/videos/downloads. Sandbox-patched tools rely on the container mount instead.
- `check_reminders` skips users `auth.is_authorized` rejects (blocked or removed); their reminders
  stay `pending` and fire once they are authorized again.
- Cancellation cannot interrupt a tool call already inside a worker thread; the turn only ends when
  that call returns (sandbox containers run to their own timeout).
- `stats_ui.py` imports `stats` at module level, so importing `bot.handlers` creates `data/stats.db`
  in the cwd (the other modules import it lazily).
- Dead code: `JudgeResponse`, `tavily_client` in `agents/main.py`, `describe_document_openai` was removed.
- `bot.rich` marks rich messages unsupported for the whole process on the first 404 from
  `sendRichMessage`/`sendRichMessageDraft`; a wrong verdict (e.g. a transient 404) needs a restart.
  `TelegramBadRequest` on a rich send is treated as "this text", not "this server".
- `agents/main.py` appends `RICH_FORMATTING_GUIDE` / `LEGACY_FORMATTING_GUIDE` to every system
  prompt after selection; the `generall-ai-*` prompts still say nothing about formatting themselves.

## How to make common changes

- **Add a tool**: new class in `app/agents/`, implement the contract (take `sender` if it talks to
  the user), do the four wiring steps in `agents/main.py` (use `await` for async providers,
  `asyncio.to_thread` for blocking ones), decide host vs sandbox (`tool_integrator.py` /
  `SECURE_TOOL_METHODS`), keep the name unique, return a string.
- **Add a user setting**: add the category to `DEFAULT_SETTINGS` in `bot/settings.py`, render it in
  `settings_ui.py` (overview text, keyboard, `show_<cat>_menu`, `elif category ==` branch with the
  short token), then read it in `agents/main.py` via the `user_settings` dict.
- **Add a system prompt**: define `system_context_<name>` inside
  `ChainOfThoughtAgent.generate_response`, add the selection `elif`, the display-name branch, add the
  name to `SYSTEM_PROMPT_TYPES` in `bot/settings.py` (the menu is generated from it).
- **Add a document extension**: `TEXT_EXTENSIONS`/`DOCUMENT_EXTENSIONS` and `describe_document` in
  `bot/media.py`; write a `describe_<kind>` following `describe_txt` including the `process_large_text` fallback.
- **Add a slash command / callback family**: a handler on the router with the right middleware in
  `bot/handlers/__init__.py`; copy the nearest existing handler.
- **Add a media type**: a `_run_<kind>(bot, message, user_id, limit, ctx)` job plus an
  `@router.message(F.<kind>)` handler in `messages.py` that tracks and `submit`s it.
- **Add a background job**: an `async` loop started from `bot/app.py:on_startup` and appended to
  `runtime.background_tasks`; to run the agent from it, submit a `Job` to `runtime.queue`.
- **Change models**: edit the default in `app/models.py` or set the env var of the same name
  (`ANTHROPIC_MODEL`, `OPENAI_MODEL`, `VIDEO_FRAMES_MODEL`, ...; see `.env.example`). No other file
  holds a model name. Keep the option helpers honest when the new model's API differs: a
  non-reasoning OpenAI model rejects `reasoning_effort` (drop it from `OPENAI_REASONING_MODELS`), a
  pre-4.6 Claude model needs `budget_tokens` instead of adaptive thinking, and Haiku-class models
  reject `effort` (which is why `ANTHROPIC_MODEL_FAST` calls never get `anthropic_request_options`).
  Then update the table above and `tests/test_models.py`.
- **Change sandbox limits/mounts/network**: `ContainerManager._run_command_in_slot`; the slot cap is
  `MAX_SANDBOX_CONTAINERS`.
- **Run the tests**: `pip install -r requirements-dev.txt && pytest` from the repo root (Python ≥ 3.12).
- **Smoke-check without Telegram or Docker**: from `app/`, with `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
  `TAVILY_API_KEY` set to any non-empty value, run `python -c "import bot.app, agents.main, models"`. It
  catches syntax and import errors in the whole bot and tool code. Running `main_bot.py` additionally
  needs Docker and a real token.
