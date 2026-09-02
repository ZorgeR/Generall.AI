# CLAUDE.md — GALL.AI / GenerALL.AI

Guidance for AI coding agents working in this repository. Everything below was
derived from reading the code, not the README; where the two disagree, the code
wins and the discrepancy is listed under "README vs code".

## What this is

A single-process Python Telegram bot (`python-telegram-bot` v22, asyncio) that
wraps a Claude tool-use agent. Users send text / voice / photos / video / audio /
documents in Telegram; the bot turns media into a text prompt, runs a
per-message agent loop with 35 tools (files, web search, code execution in a
Docker sandbox, image/video generation, reminders, SMS, S3, TTS), persists
multi-layer memory per user on disk, and replies in Telegram. Claude drives the
tool loop; OpenAI (`gpt-5.2`) is only consulted by the optional critique step,
which can force a rewrite. OpenAI, Google Gemini/Veo, Perplexity, Tavily,
ElevenLabs, Twilio and Whisper otherwise serve specific side tasks.

There are **no tests, no CI, no linter/formatter config, and no type checking**.
Validate changes by running the bot (see "Running") or with the import smoke
check at the end of this file.

## Repository layout

```
app/                        Python package root; the bot runs with cwd=app/ (Dockerfile CMD)
  main_bot.py               Telegram layer: handlers, auth/invites, /settings UI,
                            media pre-processing, reminders scheduler, admin /stats (≈4k lines)
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
    __init__.py             imports SystemTools; the package object is what the patcher needs
  secure_container/
    main.py                 initialize_secure_containers() / cleanup_containers()
    container_manager.py    ContainerManager: builds image "secure-container:latest", runs one
                            throwaway Docker container per tool call
    tool_integrator.py      ToolIntegrator: monkey-patches tool classes at startup (see below)
    secure_tool_wrapper.py  SecureToolWrapper: thin routing layer to ContainerManager
    Dockerfile, entrypoint.sh, requirements.txt   the sandbox image (python:3.12-slim + tools,
                            non-root user "runner", passwordless sudo for apt only)
  stats.py                  StatsTracker singleton: SQLite data/stats.db, per-user action limits
  image_utils.py            JPEG/HEIC helpers (pillow-heif registered on import)
  voice/                    VoiceManager: ElevenLabs voice ids (voices.json) + per-user choice (config.json)
  Dockerfile, requirements.txt   bot image (python:3.12-slim + git, docker.io, ffmpeg)
docker-compose.yml          services: telegram-bot-api (local Bot API server) + bot
.env.example                template for .env (not exhaustive: STREAMING_ENABLED and WORKSPACE_ROOT
                            come from docker-compose.yml; BROWSER_SERVICE_URL is listed but unused)
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
                                  # and stats/voice/secure_container/image_utils are top-level modules
```

Deployment facts that are easy to get wrong:

- `docker-compose.yml` hard-codes `WORKSPACE_ROOT=/opt/generall.ai/Generall.AI`. This must be the
  **host** path of the repo checkout, because the bot container talks to the host Docker daemon
  (`/var/run/docker.sock`) and bind-mounts `$WORKSPACE_ROOT/data/<uid>` into each sandbox container.
  Cloning elsewhere without changing it makes every sandboxed tool fail or mount an empty dir.
- `.env.example` ships `TELEGRAM_USE_LOCAL_API=true`. For a plain local run without the
  `telegram-bot-api` sidecar set it to `false`, or `main()` will point PTB at `localhost:8081`.
- The `bot` service runs `privileged: true` with `network_mode: host`. Host networking is what lets it
  reach the `telegram-bot-api` sidecar at `localhost:8081`; the shared `telegram-bot-api-data` volume
  is mounted in both containers because local-mode `get_file` returns server-side absolute paths.
- `./data` is bind-mounted to `/app/data` with `:z` (SELinux). `ContainerManager.ensure_user_directory`
  `chmod 777`s `data/<uid>` on every sandboxed call so the sandbox's non-root `runner` can write.
- Importing `app/main_bot.py` has side effects: it calls `initialize_secure_containers()` and
  `exit(1)`s if Docker is unavailable, and it does `os.getenv("TELEGRAM_CHAT_ID").split(",")`, which
  raises if that var is unset. Importing `agents.main` alone constructs `OpenAI(...)`,
  `genai.Client(...)` and `telegram.Bot(...)` at module level, which **raise at import** when
  `OPENAI_API_KEY`, `GOOGLE_API_KEY` or `TELEGRAM_BOT_TOKEN` are unset (any non-empty value passes).
  `stats` creates `data/stats.db` on import.
- Python ≥ 3.12 is required in practice: `agents/main.py` uses nested same-quote f-strings (PEP 701)
  and fails to even parse on 3.11. `requirements.txt` adds `audioop-lts` for 3.13.
- `STREAMING_ENABLED` is read in `main_bot.py` **before** `load_dotenv()` but in `agents/main.py`
  after it. Under compose it is passed via `environment`, so both agree; in a local run with the flag
  only in `.env` the agent streams into a callback that `main_bot` never creates (no-op).
- Polling mode only, `drop_pending_updates=True`: messages sent while the bot is down are lost on
  restart. Nothing else runs on a schedule except two `job_queue.run_repeating` jobs every 10 s
  (user reminders and agent reminders).

## Startup order and the monkey-patching (read this first)

`main_bot.py` executes, at import time and in this order:

1. Top-level imports: `stats` (singleton, creates `data/stats.db`), `secure_container.main`
   (runs `logging.basicConfig`, so the later one in `main_bot` is a no-op), `voice`, `image_utils`.
   `STREAMING_ENABLED` is read here.
2. ffmpeg/ffprobe detection for pydub (`in_docker` or PATH; otherwise `./ffmpeg/ffmpeg/` on POSIX
   and a hard-coded Windows path). Then `load_dotenv()`.
3. `initialize_secure_containers()` (`secure_container/main.py`): connects to Docker, builds
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
     of `{file, context, position}`. The host implementation in `search_tools.py` (reads
     .txt/.json/.md, returns `{full_match, word_match}`) is dead once patched.
   - `ImageTools`: patch targets `_generate_image`, which no longer exists → no-op.
   - `ConversationEmbeddings`: wrapped with logging only.
   - `SystemTools`: fully patched, but the class is never given to the agent, so unreachable.
   `ContainerManager` is instantiated twice here (once kept for cleanup, once inside the wrapper).
4. **Only then** `import agents.main` — so every `ChainOfThoughtAgent` sees patched classes.
5. Module-level clients: two `OpenAI` (normal and Whisper key), `AsyncAnthropic`, `VoiceManager()`.
   ElevenLabs, Twilio, boto3 and the embeddings client are built per call or per instance.

Consequences: the tool list the model actually sees is **not** what you read in `agents/*.py`
alone; the source of truth is source file + `tool_integrator.py`. When you add or rename a method on
a tool class, check `SECURE_TOOL_METHODS` and the `patch_*` functions, or you may silently move
execution between host and sandbox.

## Request flow (text message)

```
handle_message (main_bot.py)
  user_id = str(chat_id)            # chat id, never from_user id; used everywhere as the user key
  is_user_authorized → check_user_limits → async with get_user_lock(user_id)
  stats_tracker.track_message_received; reply "💭 Thinking..." (edited in place with status)
  get_answer(text, user_id, update_status, update, context, on_text_chunk, message_thread_id)
    UserSettings(user_id) + backfill defaults (in memory only; the "save" writes a fresh instance)
    agents.ChainOfThoughtAgent(model="claude-sonnet-4-6", user_id, telegram_update, user_settings, thread_id)
    .generate_response(question)
       ├─ question = "Message received time in UTC+0: <ts>\n\n" + text     (stored everywhere in this form)
       ├─ pick system prompt by settings.system_prompt.type (4 inline prompts; unknown → generall-ai-v2)
       ├─ semantic_search: FAISS hits appended to the system prompt (sync OpenAI embedding call)
       ├─ context_memory = fixed intro pair
       │     + conversation summaries (summarization_history)   [reads data/<uid>/conversations/*.json]
       │     + dialog_history.json, then context_memory[:-2]     (drops the newest pair; if the file is
       │       empty it eats the summaries pair or the intro pair instead)
       │     + short_term_memory.json messages whose FIRST block is text (reasoning_context)
       │     + the question                       (all three gated by short_term_memory.enabled too)
       ├─ _classify_complexity (Haiku) → "simple": tool-less Haiku answer; on any error fall back ↓
       ├─ AgentAnthropic.generate_response: for iteration in range(tools.max_iteration):
       │     messages.create/stream(model, system, tools=get_tools_schema(), thinking?)
       │     stop_reason == tool_use AND tools.enabled AND cicles < max → run each block,
       │        append ONE user msg of tool_results, continue      (cicles counts tool_use BLOCKS)
       │     gate fails → tool_use blocks silently dropped, text goes on to critique/judge/return
       │     else optional critique (OpenAI gpt-5.2) / judge (Claude yes/no) re-prompts → return
       │     for-loop exhausted → forced "SYSTEM NOTICE" final call without tools or thinking
       └─ strip pre-loaded context from thread_messages; persist (see memory) → (response, messages)
  send_response_to_user (markdown, 4000-char split, plain-text fallback)
  send_reasoning_file (if reasoning_context.enabled: reasoning_<uid>_<uuid>.txt as a document)
```

Media handlers follow the same skeleton and only differ in how they build the prompt:

- Voice notes (`handle_voice_message`): downloaded to `temp_audio/`, converted, Whisper-transcribed,
  deleted; only the transcription reaches the agent. Nothing is persisted.
- Audio files (`handle_audio_message`): saved to `data/<uid>/audio/<original name>`; no transcription;
  the prompt carries metadata plus the path.
- Photos/albums (`handle_photo_message` → `process_image_message`): JPEG copy to
  `data/<uid>/images/image_<uuid>.jpg`, described twice (Claude, then GPT) before the agent runs;
  albums are buffered per user and flushed 10 s after the last photo.
- Video/video notes (`handle_video_message`): saved to `data/<uid>/videos/`, 4 frames to
  `data/<uid>/images/video_frame_*.jpg`, frames described by `gpt-5-mini`, audio by Whisper.
- Documents (`handle_document_message`): saved to `data/<uid>/documents/<lowercased name>`, then
  `describe_document` (PDF via Claude document block; txt/json/docx/xlsx extracted; >100k chars
  map-reduced). Video extensions are rerouted to the video handler, JPG/JPEG/HEIC/HEIF to the photo
  pipeline, everything else unsupported.

Voice and video replies additionally get an ElevenLabs TTS voice note. Every saved path is put into
the prompt so file tools can use it later.

Return-type note: both `generate_response` methods are annotated `-> str` but return
`(text, messages)` tuples. `get_answer`'s shortcut for the exact texts "time", "current time" and
"what time is it" returns a bare string, which breaks every caller's tuple unpacking (existing bug).

## The tool contract

A tool provider is duck-typed. It must expose:

- `tools_schema` → `list[dict]` of Anthropic tool definitions `{name, description, input_schema}`.
  Either a `@property` (file/search/code/terminal/time/sms) or an instance attribute set in
  `__init__` (image/video/user_interactions). If you make it a property, the patcher can wrap it.
- `execute_tool(self, tool_name, tool_args) -> str`, sync or async.

Wiring is entirely manual in `agents/main.py` and there is no registry:

1. `AgentAnthropic.__init__`: a `self.<name> = None` slot.
2. `AgentAnthropic.get_tools_schema`: `tools.extend(self.<name>.tools_schema)` — fixed order
   file_ops, search, code, terminal, time, image, video, sms, user_interactions.
3. `AgentAnthropic.execute_tool`: an `elif` that checks `tool_name in [t["name"] for t in
   provider.tools_schema]` and calls `execute_tool` — **with `await` only for file_ops, image,
   video, user_interactions**. First match wins, so tool names must be globally unique.
4. `ChainOfThoughtAgent.__init__`: construct the provider and assign `self.agent.<name> = self.<name>`.
   Constructor shapes differ: `(user_id, telegram_update)` for file/terminal/image/video/
   user_interactions, `(user_id)` for search/code, no args for time/sms.

Tool results are `str(result)` into a single `tool_result` block; return strings. Every call is
recorded via `stats_tracker.track_tool_used` (and counts against the user's action limit). Tool
schemas are sent to the API even when `tools.enabled` is false.

Effective tool list as the model sees it (after patching):

| Provider | Tools | Runs where |
|---|---|---|
| FileOperations | `list_files`, `create_file`, `read_file`, `download_file`, `download_webpage`, `upload_to_s3`, `create_zip_archive`, `send_file_path_to_user_via_telegram` | bot process, `data/<uid>/ / <arg>` via pathlib (an absolute arg escapes the base; no traversal check) |
| FileOperations | `create_directory`, `delete_file` | sandbox container (a fresh `docker run` per call), paths relative to `data/<uid>` |
| SearchTools | `search_web` (Tavily), `deep_research` (Perplexity `sonar*`) | bot process |
| SearchTools | `memory_search` | sandbox; `*.txt` only |
| CodeTools | `execute_python` (+`network_enabled`) | sandbox |
| TerminalTools | `run_command` (no network option), `run_shell_script` (+`network_enabled`), `install_package` (apt, persisted, network on) | sandbox |
| TimeTools | `get_time_in_timezone`, `list_timezones` | bot process |
| ImageTools | `image_generator`, `image_editing`, `image_composition` (Gemini or GPT Image 2), `generate_multimodal_image_and_text` (Gemini), `generate_image_dall_e` (obsolete) | bot process, sends results to Telegram itself |
| VideoTools | `video_generator`, `image_to_video_generator`, `video_from_reference_images`, `video_interpolation_generator`, `video_extension_generator` | bot process (Veo, polls up to 5 min) |
| SMSTools | `send_sms` | bot process |
| UserInteractions | `send_user_telegram_message`, `send_voice_message`, `set_message_reaction`, `schedule_reminder`, `send_file_content_to_user_via_telegram` | bot process |

Unreachable: all 11 `SystemTools` tools (`install_package`, `run_shell_script` duplicates plus
`check_system_info`, `manage_service`, `monitor_process`, `network_diagnostics`,
`list_installed_packages`, `remove_package`, `install_python_package`,
`list_installed_python_packages`, `remove_python_package`). `ContainerManager` does honor a
persisted pip list (`data/<uid>/installed_python_packages.txt`) at every sandbox run; no bot code
writes it, but the model can create the file with `create_file` and it will then be replayed.

Event-loop warning: most tool implementations are synchronous network or Docker calls executed
directly on the asyncio loop (Tavily, Perplexity, Gemini, OpenAI images, the FAISS embedding call,
`container.wait(timeout)` for every sandboxed call, and `time.sleep(20)` polling in
`video_tools.py`). One user's long tool call stalls **all** users; the per-user lock only
serializes one user's own messages. Wrap new blocking work in `asyncio.to_thread`.

## Sandbox (secure_container) mechanics

- Image `secure-container:latest` is built lazily on first startup from `app/secure_container/`
  (slow, many apt layers). Rebuild by deleting the image; the code never rebuilds an existing one.
- Every sandboxed call = one new container `secure-container-<uid>-<8 hex>`: bind-mount
  `$WORKSPACE_ROOT/data/<uid>` → `/home/runner/workspace` (rw, shared), cwd there, user `runner`,
  `mem_limit=512m`, one CPU, `network_mode=none` unless `network_enabled` or packages must be
  installed, removed in `finally`.
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
- Output = `container.logs()` (stdout+stderr), always prefixed by the entrypoint banner (~10 lines:
  "Secure container environment initialized", tool versions, "Executing command: ...", dashes).
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
    settings.json                   per-user settings (see table); merged over defaults on load
    conversations/conversation_<YYYYmmdd_HHMMSS>_<topic>.json
                                    one file per turn: {timestamp, thread_id, topic, summary, question, response, full_history}
                                    "long-term memory"; topic+summary come from Haiku; also read by the agent's file tools
    short_term_memory/[topic_<thread>_]dialog_history.json
                                    last dialog_history.size Q/A PAIRS (size*2 entries) as plain strings
    short_term_memory/[topic_<thread>_]short_term_memory.json
                                    block-structured messages of (mostly) the last turn ("reasoning context")
    embeddings/faiss_index.bin, metadata.json
                                    IndexFlatL2(1536) over "Question: ..\nAnswer: .." with text-embedding-ada-002
    reminders/reminders.json        list of {id,user_id,text,time(ISO UTC),type:user|agent,status,created_at,is_periodic,
                                    period_type,period_interval,last_triggered,next_trigger,[enabled],[completed_at],[agent_response]}
    images/                         image_<uuid>.jpg (uploads), video_frame_<uuid>.jpg, generated_*/transformed_*/composed_*/
                                    gpt_generated_*/gpt_edited_*/story_image_*, edit_input_* (transient)
    videos/                         video_<uuid>.mp4 (uploads), veo3_*.mp4
    audio/, documents/              uploads under their ORIGINAL file names (documents lowercased) → re-uploads overwrite
    downloads/                      download_file / download_webpage(save_to_file) / create_zip_archive outputs
    installed_packages.txt, installed_python_packages.txt, temp_code.py, temp_script.sh, temp_setup.sh
temp_audio/ temp_photos/ temp_docs/ reasoning_<uid>_<uuid>.txt     transient, cwd-relative, deleted in finally
app/voice/config.json               MUTATED at runtime by /voice (per-user voice id). Lives inside the image, not data/.
```

Memory semantics worth knowing before touching `ChainOfThoughtAgent.generate_response`:

- Forum topics: `message_thread_id` only prefixes the two short-term files (`topic_<id>_...`).
  Conversation summaries and the FAISS index are shared across threads.
- Summaries are read with `Path.glob` (unsorted) and sliced
  `[:-dialog_history.size][-summarization_history.size:]`. Sizes come clamped 1..50 from the UI;
  a hand-edited `dialog_history.size` of 0 yields nothing, and `summarization_history.size` of 0
  injects **every** conversation file.
- Old conversation files without `summary`/`topic`/`timestamp` are tolerated (commit d836a2d);
  unreadable JSON is skipped. There is no migration.
- The conversation filename embeds the raw Haiku "topic" string unsanitized.
- `_save_conversation` runs inside `generate_response` before it returns and has no try/except: if
  the Haiku topic/summary or embedding call fails, the whole turn errors out and the user sees
  "❌ An error occurred" even though the answer was already generated (and paid for).
- Reloading `short_term_memory.json` keeps only messages whose first block is `text`. With thinking
  on, intermediate tool-calling assistant turns start with a `thinking` block and are dropped; the
  final answer (re-appended as text-only) is kept. Tool-result user messages are never reloaded.
  De-duplication of reloaded messages is by dict equality and misses multi-block originals, so
  text-only copies of the previous turn's tool-calling messages survive one extra turn.
- No prompt caching anywhere (`cache_control` count is 0); the full tool schema and system prompt
  (the Perplexity ones are very long) are resent every iteration.

## User settings (`data/<uid>/settings.json`)

Defaults are duplicated in `main_bot.py` (`default_settings` **and** `UserSettings.__init__`) and
must be kept in sync by hand. Agent code reads them as `user_settings.get(cat).get(key)` with no
default for the category, so every category must exist. `get_answer` backfills missing categories
in memory only; the file on disk is not migrated.

| Category | Keys (default, UI clamp) | Consumer |
|---|---|---|
| `summarization_history` | enabled (true), size (5, 1..50) | conversation summaries in context |
| `dialog_history` | enabled (true), size (10, 1..50) | rolling Q/A window |
| `reasoning_context` | enabled (true) | reload last turn's blocks; also gates `send_reasoning_file` |
| `short_term_memory` | enabled (true) | master switch for the three above |
| `critique` | enabled (false), max_iteration (5, 1..300) | OpenAI critique re-prompts |
| `judge` | enabled (false), max_iteration (5, 1..300) | Claude yes/no completeness re-prompts |
| `tools` | enabled (true), max_iteration (20, 1..300) | the agent loop bound (this, not the env var) |
| `semantic_search` | enabled (true), max_results (3, 1..20) | FAISS hits in system prompt |
| `thinking` | enabled (true) | extended thinking: max_tokens 20000 / budget 16000, else max_tokens 4096 |
| `system_prompt` | type ("generall-ai-v2"; also generall-ai-v1, perplexity-deep-research, perplexity-r1) | prompt selection |

Edited through `/settings` inline keyboards. callback_data is `settings_<token>[_<action>[_<value>]]`
parsed with a plain `split("_")`, where `<token>` is a short name, not the JSON key: summarization,
dialog, reasoning, memory (= short_term_memory), critique, judge, tools, semantic, thinking, main.
`system_prompt` is special-cased with `startswith` because of its underscore.

## Models and external services

| Purpose | Model / API | Where set |
|---|---|---|
| Agent loop, judge, final compile, document & image description | `claude-sonnet-4-6` via `anthropic.AsyncAnthropic` | `agents/main.py` `anthropic_model`; `main_bot.py` duplicates the constant and passes it in |
| Topic/summary, complexity classifier, "simple" answers | `claude-haiku-4-5` | `agents/main.py` `anthropic_model_fast` |
| Critique | `gpt-5.2` structured output (`beta.chat.completions.parse`) | `agents/main.py` `openai_model` |
| GPT vision on photos (second description after Claude) | `gpt-5.2` | `main_bot.py` `openai_model` |
| Video frame description | `gpt-5-mini` | `main_bot.py` `describe_video_screenshots` |
| Transcription | `whisper-1` via a second client keyed by `OPENAI_API_KEY_WHISPER` (falls back to `OPENAI_API_KEY` if unset); >24 MB chunked | `main_bot.py` `transcribe_audio` |
| Embeddings | `text-embedding-ada-002`, dim 1536 | `agents/embeddings.py` |
| Image gen/edit | `gemini-3.1-flash-image-preview` (Normal), `gemini-3-pro-image-preview` (Pro), `gpt-image-2-2026-04-21` (GPT), `dall-e-3` (legacy) | `agents/image_tools.py` |
| Video | `veo-3.1-generate-preview` for all five tools (two descriptions still say "Veo 3.0") | `agents/video_tools.py` |
| Web search / research | Tavily; Perplexity `sonar` (default), `sonar-pro`, `sonar-reasoning-pro` via raw HTTP | `agents/search_tools.py` |
| TTS | ElevenLabs `eleven_multilingual_v2` (literal in `main_bot.py` twice and `user_interactions.py`), voices in `app/voice/voices.json` | |
| SMS | Twilio | `agents/sms_tools.py` |
| Object storage | boto3 S3-compatible, presigned URL 1 h | `agents/file_ops.py` |

Client construction: `OpenAI`, `genai.Client`, `telegram.Bot`, `AsyncAnthropic` and `TavilyClient`
are module-level (import time); the first three raise immediately if their key is missing.
Twilio, boto3, `TavilyClient` in SearchTools and the embeddings `OpenAI` are per instance;
ElevenLabs is per call.

## Environment variables

| Variable | Used by | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | main_bot, file_ops | required at import |
| `TELEGRAM_CHAT_ID` | main_bot | comma list; **required** (import crashes without it); always-authorized ids; the ONLY ids allowed to press /reminders buttons |
| `TELEGRAM_ADMIN_ID` | main_bot | unlimited invites, no action limit, `/stats`, `/listusers`, join notifications. NOT auto-authorized for chat unless also in `TELEGRAM_CHAT_ID` |
| `TELEGRAM_ALLOWED_ALL_USERS` | main_bot | exact lowercase `true` → everyone not blocked is authorized |
| `INVITE_LIMIT` | main_bot | default in code is 3 (`.env.example` says 5) |
| `TELEGRAM_USE_LOCAL_API`, `TELEGRAM_LOCAL_API_URL` | main_bot `main()` | local Bot API server (compose sidecar); default URL `http://localhost:8081` |
| `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` | compose only | for the `telegram-bot-api` sidecar |
| `STREAMING_ENABLED` | main_bot (before dotenv) AND agents/main (after) | draft-message streaming via `bot.send_message_draft`, errors swallowed |
| `WORKSPACE_ROOT` | secure_container | host path of the checkout (see Running); set in compose, not `.env.example` |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_API_KEY_WHISPER`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `PERPLEXITY_API_KEY`, `ELEVENLABS_API_KEY` | various | see models table; `OPENAI_API_KEY` and `GOOGLE_API_KEY` needed at import |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | sms_tools | optional; tool returns an error string if unset |
| `S3_HOST`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME`, `S3_PATH_TO_STORE` | file_ops | optional |
| `MAX_IMAGE_RESOLUTION_VISION` (1024), `MAX_IMAGE_RESOLUTION_EDIT` (4096) | main_bot, image_tools | downscale before vision / edit |
| `MAX_AGENT_TOOLS_ITERATIONS`, `MAX_AGENT_CRITIQUE_ITERATIONS` | read, never used | obsolete; per-user settings replaced them |
| `BROWSER_SERVICE_URL` | nobody | listed in `.env.example`, unused |

Never commit `.env`, `data/`, `temp_photos/`, `temp_docs/`, `temp_audio/` (git-ignored by exact
name; a new temp dir needs its own `.gitignore` line).

## Telegram layer conventions

- Commands: `/start [invite_<code>]`, `/invite [code]`, `/voice`, `/settings`, `/reminders`,
  `/stats` (admin), `/listusers` (admin). Unknown commands are silently ignored.
- Callback prefixes and handlers: `voice_` → `voice_button`, `settings_` → `settings_button`,
  `reminder_`/`reminders_` → `reminder_button`, `stats_` → `stats_button`. A new keyboard family
  needs a unique prefix, a `CallbackQueryHandler(pattern="^prefix_")` in `main()`, and its own auth
  check inside the handler (nothing is centralized). Callback handlers must read
  `update.callback_query.message.chat_id`; `update.message` is `None` there.
- Every message handler: `user_id = str(update.message.chat_id)` → `is_user_authorized` →
  `check_user_limits` (50 actions / rolling 30 days by default, admin exempt, 0 = unlimited) →
  `async with get_user_lock(user_id)` → `stats_tracker.track_message_received(user_id, kind)` →
  status message edited in place → `get_answer` → `send_response_to_user` → `send_reasoning_file`
  → broad `except` that replies `❌ ... Trace ID: <uuid>` → temp files removed in `finally`.
  (Photo and document handlers track before taking the lock.) The
  `update_thinking_message(step, details, iteration, critique)` closure is copy-pasted per handler.
- Action-limit accounting counts **every** `stats_events` row in 30 days: each received message,
  sent message, tool call, describe call and media group. A turn with 20 tool calls costs ~22 of the
  default 50. Any new `stats_tracker.track_*` call burns user quota.
- Replies use Telegram Markdown v1 (`parse_mode="markdown"`) inside try/except with a plain-text
  fallback; invite/admin replies use HTML. Anything over 4000 chars is split by
  `split_text_intelligently`. The "message was split" notice is hard-coded in Russian.
- Authorization state (`authorized_users`, `blocked_users`, `user_invites`) is module-global and
  flushed to `data/userlist.json` after each mutation.
- Reminders: created only by the agent tool `schedule_reminder`; fired only by the two 10-second
  jobs in `main_bot.py`. Agent reminders (`type: "agent"`) build a mock `Update` and run the full
  agent **without** the user lock, without auth or limit checks, without a thread id, over every
  directory under `data/`, and with an inline copy of the reply/reasoning-file logic (changes to
  `send_response_to_user` do not reach them). Both jobs rewrite whole `reminders.json` files and
  can race each other and the `/reminders` UI.
- Media groups (albums) are keyed by user, not by `media_group_id`: a second album within the 10 s
  window discards the first. Only JPG/JPEG/HEIC/HEIF are accepted as image *documents*; PNG/WEBP
  sent as files are rejected. Telegram photos (compressed) are always JPEG and fine.
- Logging is a mix of `logging.getLogger(__name__)`, bare `logging.*`, and a lot of `print`.
  `agents/main.py` is `print`-only and dumps every API response and tool result to stdout. Prefer
  `logger` in new code. Nothing writes log files.

## README vs code

- README says Claude 3.7 / GPT-4o; code uses `claude-sonnet-4-6`, `claude-haiku-4-5`, `gpt-5.2`, `gpt-5-mini`.
- README documents `MAX_AGENT_*_ITERATIONS` as active; they are dead. The real bound is
  `settings.tools.max_iteration` (default 20, not 65).
- README lists SSH and Shodan tools; neither exists. SSH is only possible via `run_shell_script`
  inside the sandbox (openssh-client is installed there) with `network_enabled`.
- README lists PNG/GIF/BMP/WEBP as supported images; as documents only JPG/JPEG/HEIC/HEIF are.
- README never mentions `TELEGRAM_ALLOWED_ALL_USERS`, which bypasses the allow-list entirely.
- `.env.example` says the admin is authorized by default and `INVITE_LIMIT=5`; code: not
  auto-authorized, default 3.

## Known pitfalls (do not "fix" casually without checking callers)

- `get_answer` returns a bare string for "time"/"current time"/"what time is it"; every caller
  unpacks a tuple.
- `reminder_button` authorizes against the raw `TELEGRAM_CHAT_ID` list, so invited users can open
  `/reminders` but every button answers "Unauthorized chat". The `noop` callback_data used for
  spacer buttons matches no handler pattern.
- `UserSettings.load_settings` does `self.settings[key].update(value)` → a `settings.json` with an
  unknown dict-valued key raises `KeyError` on every message for that user. Removing or renaming a
  settings category requires migrating existing files.
- `validate_iteration` returns `None` for unknown types; `calculate_next_trigger` leaves
  `next_trigger` unbound for unknown `period_type`; "monthly" means 30 days.
- Reminder ids are `str(len(all)+1)` at creation → collisions after deletions.
- The iteration counter `cicles` counts individual `tool_use` blocks while the `for` loop counts
  API round-trips; hitting either limit drops pending tool calls and may return an incomplete
  "Let me check..." text. The forced final call has no tools and no thinking.
- `judge_response` treats exceptions as "yes" (accept); `critique_response` treats them as "no rewrite".
- `Application.post_shutdown` branch in `main()` never runs; container cleanup relies on `atexit`.
- `sendvoice_to_user` leaks its `NamedTemporaryFile(delete=False)` mp3.
- Dead code: `describe_document_openai` (never called, invalid `files=` kwarg), `JudgeResponse`,
  `tavily_client` in `agents/main.py`, `telegram_bot` in `file_ops.py`, `send_reasoning` in
  `main_bot.py`, the `set_semantic_max_`/`toggle_semantic_enabled`/`semantic_max_results` and
  `category == "system_prompt"` branches in `settings_button`.
- `perplexity-*` system prompts contain hard-coded 2025 dates and "You are Perplexity" identity text;
  the `generall-ai-*` prompts say "20 previous messages" regardless of `dialog_history.size`.
- `handle_message` logs the first 50 chars of every text before the auth check.
- File tools join under `data/<uid>/` with pathlib, so an absolute path escapes the base;
  `send_file_path_to_user_via_telegram`, `_resolve_image_path` and the video tools accept any
  existing path on the bot filesystem.

## How to make common changes

- **Add a tool**: new class in `app/agents/`, implement the contract, do the four wiring steps in
  `agents/main.py`, decide host vs sandbox (`tool_integrator.py` / `SECURE_TOOL_METHODS`), keep the
  name unique, return a string, and use `await` in `execute_tool` only if yours is async.
- **Add a user setting**: `default_settings` + `UserSettings.__init__` + both overview texts and
  keyboards in `settings_command`/`settings_button` + a `show_<cat>_menu` + an `elif category ==`
  branch (using the short token); then read it in `agents/main.py` via the `user_settings` dict.
- **Add a system prompt**: define `system_context_<name>` inside
  `ChainOfThoughtAgent.generate_response`, add the selection `elif`, the display-name branch, and a
  `settings_system_prompt_set_<name>` button in `show_system_prompt_menu`.
- **Add a document extension**: three parallel lists in `main_bot.py`: `supported_extensions` and
  `describe_type_map` in `handle_document_message`, and the dispatch in `describe_document`; write a
  `describe_<kind>` following `describe_txt` including the >100k-char `process_large_text` fallback.
- **Add a slash command / callback family / media type / background job**: copy the nearest existing
  handler in `main_bot.py`; the conventions section lists what each needs.
- **Change models**: constants at the top of `agents/main.py` and `main_bot.py` (both!), literals in
  `image_tools.py`, `video_tools.py`, `embeddings.py`, `search_tools.py` (sonar enum + defaults),
  `describe_video_screenshots`, `transcribe_audio`, and the three `eleven_multilingual_v2` literals.
- **Change sandbox limits/mounts/network**: `ContainerManager.run_command`.
- **Smoke-check without Telegram or Docker**: from `app/`, with `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
  `TELEGRAM_BOT_TOKEN` and `TAVILY_API_KEY` set to any non-empty value, run
  `python -c "import agents.main"` (Python ≥ 3.12). It catches syntax and import errors in the agent
  and tool modules. Importing `main_bot` additionally needs Docker and `TELEGRAM_CHAT_ID`.
