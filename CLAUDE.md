# CLAUDE.md — GALL.AI / GenerALL.AI

Guidance for AI coding agents working in this repository. Everything below was
derived from reading the code, not the README; where the two disagree, the code
wins and the discrepancy is listed under "README vs code".

## What this is

A single-process Python Telegram bot (aiogram 3, asyncio) that wraps a Claude
tool-use agent. Users send text / voice / photos / video / audio / documents in
Telegram; the bot builds a prompt of content blocks (text, plus the photos and PDFs
themselves as image/document blocks, with the replied-to or forwarded message quoted
in front), runs a per-message agent loop with 51 built-in tools (files, web search and
a headless page reader, code execution in a persistent per-user Docker sandbox, image/video
generation, documents out as pdf/docx/xlsx, reminders and scheduled tasks in the user's time
zone, SMS, S3, TTS, button questions, a user profile, the sandbox status, background tasks) plus Anthropic's
browser toolset (an interactive Chromium per user, in its own container, with saved logins)
plus the tools of the user's MCP servers (a bearer token or an OAuth sign-in in one tap), persists multi-layer memory per user on disk
(transcript, summaries + embeddings, profile), and replies in Telegram. Claude drives
the tool loop; OpenAI (`gpt-6.1-sol`) is only consulted by the optional critique step,
which can force a rewrite, and by the `describe_image` second opinion. OpenAI, Google
Gemini/Veo, Perplexity, Tavily, ElevenLabs, Twilio and Whisper otherwise serve specific
side tasks.

Every chat has its own job queue: messages are processed strictly in order per
user, and users never wait for each other. A message sent while the agent of the same chat and
forum topic is working joins that running turn (the agent reads it after its next tool call)
instead of waiting for a turn of its own, and a tool call that runs long moves to the background
and reports back when it finishes ("live turns": `bot/inbox.py`, `bot/background.py`). Messages sent
together (a comment and the messages it forwards, quick texts in a row) become ONE turn: a message that
starts a turn waits 1.5 s for the next one, each new one extends the wait (the message window,
`bot/handlers/messages.py`). See "Queues".

There is a pytest suite (`tests/`, run with `pytest` from the repo root) covering the queue,
settings, presets and the /settings page, auth, reminder store, text splitting, the model settings
module, the /stats views and /usage, the Stop buttons, the turn buttons, `ask_user_choice`, the
reply/forward preamble and /help, content blocks and media prompts, the vision tools, the profile
and memory_search, forum topic names and the `<context>` line naming the topic (`test_topics.py`), the MCP client and /mcp, MCP OAuth sign-ins (`test_mcp_oauth.py`: the SDK's provider against a
fake OAuth + MCP server, storage, keys, refresh, copy-paste fallback, the move of the records out of the
workspace, the URL binding, the shared-server guard, sign-out / removal racing a refresh or a sign-in) and their callback endpoint
(`test_oauth_web.py`: routes, headers, TLS), the mode setting, the web/document tools and the
sandbox image tag, embeddings, time zones and the /reminders UI, scheduled tasks, the agent's retry
loop and the error path (`test_timezones.py`, `test_reminders_ui.py`, `test_proactive_tasks.py`,
`test_stream_retry.py`, `test_error_recovery.py`), whole turns on a fake client (`test_mode.py`: the tools and
`<capabilities>` notes actually sent, a photo reaching the loop as an image block), the persistent sandbox
sessions (`test_sessions.py`, with a fake Docker client), /sandbox and the idle warnings (`test_sandbox_ui.py`),
the browser server inside the container (`test_browser_server.py`), the browser toolset in the agent loop
(`test_browser_tools.py`, `test_toolset_loop.py`), live turns (`test_inbox.py`: joining the running turn,
👀/✍, leftovers as one turn, Stop / deadline / /cancel / shutdown, the turn record, typed answers, follow-ups, the
message window and `inbox.combine`; `test_mode.py`: what the agent reads before its first model call;
`test_background_tasks.py`: the background host, notifications and notification turns; `test_live_turn_loop.py`:
injection at tool boundaries and the background watcher in the agent loop; `test_task_tools.py`;
`test_transcript_live_turn.py`: tool-result messages with injected blocks, system notices), and how these features meet
(`tests/test_cross_features.py`).
`tests/conftest.py` sets `MESSAGE_WINDOW_SECONDS=0` for the whole run (handlers queue or inject at once); the
window's tests turn it on with `monkeypatch.setattr(messages, "MESSAGE_WINDOW", …)`.
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
                            (cancel) and the Stop buttons (cancel_current: the running job only, by Job.id);
                            Job.seq = process-wide arrival number (next_seq): a submitted job goes before every pending
                            job with a larger one; submit_nowait = submit without a suspension point (cleanup code);
                            Job.thread_id + from_user → has_pending_user_jobs(user_id, thread_id); Job.on_drop is called
                            when cancel() drops the job unrun (a requeued inbox message cancels its preparation)
    inbox.py                live turns: TurnInbox (one per running turn) holds what the user sends meanwhile and finished
                            background tasks as InboxItems sorted by seq (a user item's future gives a Prepared, still
                            preparing or ready); drain() at the agent's tool boundaries (waits DRAIN_WAIT 3 s for an
                            unready head, never skips it; an item leaves only once read, so a cancel mid-read leaves it a
                            leftover), wait(), has_user_messages (a ready, drainable one only), consumed_user_items,
                            close() → leftovers; registry open_inbox / current /
                            accepts / deliver_task / finish (read= consumed items handed back with the leftovers) /
                            requeue (the user items as ONE turn: InboxItem.requeue_group) / drop / discard / reset;
                            combine(list[Prepared]) → one Prepared (COMBINED_HEAD + <message n="i"> per message, media refs
                            inside the tags; speak only when all speak); reactions mark_seen 👀 and
                            ✍ once drained; SENT_WHILE_WORKING "[Sent while I was working]"; a TASK item's future may give
                            a callable (the notice, rendered at drain time)
    background.py           live turns: BackgroundHost per user (host_for binds bot/chat/topic each turn): adopts tool calls
                            the agent loop moved to the background (ids t1, t2, … per user, MAX_BACKGROUND_TASKS 4 running
                            in all topics), running / list / cancel / unread / read (the bound turn's topic only), cancel_all,
                            cancel_turn (a failed turn's adoptions); a finished call keeps its raw result on the record
                            (read → result_blocks, what get_task_result returns) and sends only a notice of facts
                            (notice_blocks: {"type": "system_notice", text: notice_text}, none once the result was read) to
                            the running turn of its topic, else a queued "background task finished" turn
                            (submit_notification(records), notification_prompt: NOTIFICATION_PROMPT with the facts, tasks
                            read meanwhile left out, quiet marker); shutdown() (also refuses and cancels later adoptions)
    sender.py               ChatSender: the one object handlers AND tools use to send text/media/reactions;
                            send_markdown renders LLM answers as rich messages with tiered fallback
    rich.py                 rich-message helpers: split/convert (telegramify-markdown), inline media extraction,
                            sticky "unsupported" flag, sticky "buttons inside rich messages" flag, is_chat_error
    agent_runner.py         run_turn(prompt: str | content blocks): builds ChainOfThoughtAgent, status updates (Stop
                            button, "recovering" lines from the agent's retries), streaming, voice reply, answer, turn
                            record + buttons, partial answer on Stop, token accounting (record_usage; record_late_usage:
                            what background calls add to the trace later, one row per addition), quiet_marker
                            (scheduled tasks: an answer that is only the marker sends nothing; with quiet_notice the status
                            becomes that line instead, "📭 Nothing new to report." on a Retry; messages it read are
                            requeued); a failed turn ends as "❌ <friendly_error(e)> (trace <uuid>)" + 🔁 Retry over an
                            error TurnRecord (ERROR_* texts) and cancels the calls it moved to the background;
                            opens the turn's inbox and binds the user's background host (inbox= / background=, when
                            takes_live_turn(ChainOfThoughtAgent)), closes the inbox the moment the agent returned, failed
                            or was stopped; the record keeps what the agent read mid-turn (with_drained)
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
    topics.py               TopicStore (`topic_store` singleton): data/<uid>/topics.json, the names of the user's forum
                            topics (threads) by thread id, as seen in Telegram's service messages: set (created/renamed,
                            a rename wins) / learn (the creation message a topic message points at, only fills a gap) /
                            get; names cleaned to one line ≤ MAX_NAME_CHARS 128 on write AND read, MAX_TOPICS 500,
                            read_regular_text (no link, ≤ 256 KB), mkstemp + os.replace; sync, callers use to_thread
    mcp_store.py            MCP server records + cached tool lists: per-user data/_mcp/<uid>.json (bot-only, outside the
                            workspace; _migrate moves an old data/<uid>/mcp.json there on the user's first load), the admin
                            file (MCP_SERVERS_FILE) + its .cache.json; sync I/O, per-user lock, atomic writes (mkstemp, 0600,
                            data/_mcp 0700; read O_NOFOLLOW, ≤ MAX_FILE_BYTES), name/URL validation, tokens never rendered
                            back; OAuth sign-ins: token_cipher() (Fernet: MCP_TOKEN_KEY, else data/.mcp_token_key;
                            MCP_TOKEN_KEY_OLD), read_oauth / write_oauth / clear_oauth / seal_oauth / mark_signin_needed
                            (values bound to user + server + URL, used only while the stored record has the URL of the
                            connection: _locate), all_servers(with_state=True) → oauth_state
    oauth_flow.py           MCP OAuth sign-in: pending logins (register / resolve / wait_for_code, keyed by the SDK's state,
                            one-shot, LOGIN_TIMEOUT 600 s, one per user + server), OAuthStorage (the SDK's TokenStorage,
                            expires_at + discovered metadata kept, refresh lock), OAuthConnection (→ OAuthClientProvider,
                            non-interactive without handlers: NeedsSignIn), request_guard (https + public, httpx2 hook;
                            trust_origin: a shared server's configured origin exempt), install_log_filter, start_sign_in
                            (the background task: 🔐 URL button → "✅ <name> connected" once the store took tokens and tools,
                            else REMOVED_TEXT), cancel_sign_in, sign_out / remove_server (cancel a running sign-in, wait for
                            a refresh in flight: the refresh lock), redirect_uri (public callback or LOOPBACK_REDIRECT),
                            shutdown
    oauth_web.py            the sign-in callback endpoint (aiohttp in the bot process, task "oauth-web", only with
                            PUBLIC_BASE_URL): GET /oauth/callback + /oauth/health, settings_from_env (OAUTH_LISTEN, OAUTH_TLS,
                            OAUTH_TLS_CERT/KEY), load_tls (admin files, else a generated self-signed pair), callback_url()
    questions.py            ask_user_choice backend: in-memory pending questions (Future per question, chat + topic), ask_*
                            callback data, rich button rows / inline keyboard, timeout and cancel outcomes; ask() returns an
                            Answer (str subclass, .tapped: a tap, or text typed instead: answer_in_text / resolve_text /
                            waiting_in, typed_result; .delivered: the typed text also went to the running turn as a
                            message of its own → typed_result = TYPED_DELIVERED, a pointer, never the text)
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
                            preamble + text + image_ref/document_ref), each split into prepare_<kind> (→ Prepared) and
                            the _run_<kind> job; accept / inject_target / inject (join the running turn: PREPARE_SLOTS 2
                            preparations per user, _path_lock per saved document/audio file) / _run_prepared (prepared
                            messages as ONE turn: a turn's leftovers, a message window); the message window (_Window per
                            chat + topic, MESSAGE_WINDOW 1.5 s from MESSAGE_WINDOW_SECONDS, extended by each message up to
                            MESSAGE_WINDOW_MAX 30 s, held by an album still arriving: _album_arriving; _gather, _window_timer,
                            _close_window, _drop_window / drop_windows; "typing…", then 👀 + WINDOW_STATUS count line);
                            a typed text answers a waiting question; busy notice, /cancel (also drops
                            the inbox and the message windows and cancels background tasks), Stop buttons (queue_stop_ callback,
                            stopped_message_generation update), on_location (a shared location sets the time zone; the
                            agent never sees locations), on_topic_service (forum_topic_created / forum_topic_edited → the
                            topic's name in bot/topics.py, no answer, no quota) and remember_topic (called by accept: the
                            name of the creation message reply_to_message points at, when none is known)
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
      mcp_ui.py             /mcp [add|remove|refresh|help] + mcp_ buttons (mcp_signin_ / mcp_signout_ for OAuth servers;
                            removal and sign-out through oauth_flow.remove_server / sign_out);
                            add without a token probes for OAuth and starts the background sign-in (begin_sign_in);
                            pasted_callback (the copy-paste fallback: a pasted http://localhost:8765/oauth/callback URL
                            while a sign-in is pending; async filter); refresh_admin_servers() also runs once from
                            app.py:on_startup as background task "mcp-admin-refresh" (with the admin's own sign-ins)
      turn_actions_ui.py    turn_ buttons: router (ui: Listen, Reasoning) + turn_router (chat, quota: Retry,
                            Continue, follow-ups; they echo the press and queue a turn like a typed message; a follow-up
                            joins a running turn of its topic instead: _start_turn(join_running=True))
      questions_ui.py       ask_ buttons: resolve the pending question, edit it into its final state
      reminders_ui.py       /reminders inline keyboards: local times, "⏰ Reminders" / "🤖 Tasks" groups, detail page
                            with +1h / +1d / Edit text (the next text message, _pending_edits, 5 min) / Delete / Disable,
                            ➕ New hint, snooze handling for fired reminders (reminder_snooze_), legacy-Markdown escaping
                            of zone names
      sandbox_ui.py         /sandbox: ONE legacy-Markdown page from manager.status (sandbox, saved packages, browser tabs,
                            stored logins, idle timeout; admins: server caps, free memory, egress filter) with sandbox_
                            buttons (Keep open, Close, Forget logins + confirm, Refresh); reaper_loop = the "sandbox-reaper"
                            task (every REAP_INTERVAL 60 s: reap_idle, then one warning per due_warnings entry with
                            "👌 OK" / "⏳ Keep open 30 min", remembered in _warnings); session_manager() imports
                            secure_container.sessions lazily (None = no manager: /sandbox says unavailable, hooks no-op)
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
                            prompt_notes (the <capabilities> block). Live turns: _drain_inbox after each tool batch,
                            the background watcher (_watch_call, TOOL_BACKGROUND_AFTER_SECONDS, NEVER_BACKGROUND,
                            background_placeholder), split_notices + notice_messages (the inbox's notice blocks → one
                            {"role": "system"} message after the user message, or a <bot_notice> block for a model without
                            them: models.supports_system_messages; _request_messages(messages, model) converts stored ones
                            for such a model), ChainOfThoughtAgent(inbox=, background=), with_messages_read (the
                            question + what was read mid-turn: memory files, judge, critique)
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
                            discovery, one fresh connection per call, results as text or image blocks, URL secrets redacted;
                            OAuth servers over the stored sign-in (non-interactive, one forced-refresh retry, "needs you to
                            sign in again"), the OAuth probe of credential-less servers (SignInRequired), the request guard
                            and provider handed to open_transport through the _CONNECTION context variable
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
                            ask_user_choice (via bot/questions.py; a typed answer comes back as typed_result: a pointer
                            to the <user_message> that follows the tool results, or the text when no turn took it);
                            prompt_notes on zones and quiet scheduled tasks
    task_tools.py           TaskTools (wait_for_tasks, get_task_result, list_tasks, cancel_task; static <live_turn>
                            prompt_notes; registered only with a background host, inherit_to_subagents = False), the
                            TurnInbox / BackgroundHost protocols the agent core codes against (host: unread / read too),
                            NOTICE_BLOCK "system_notice" (an inbox block the agent core sends as a system message),
                            CallState + in_background() (a ContextVar of the call)
    embeddings.py           ConversationEmbeddings: OpenAI text-embedding-3-small + FAISS IndexFlatIP (cosine) per user;
                            model-tagged metadata, stale detection, one background rebuild per user per process
    system_tools.py         SystemTools: apt/pip/service/network tools — NOT wired into the agent
    paths.py                resolve_under(): safe resolution of model-supplied paths inside data/<uid>;
                            regular_files(folder, pattern): host reads of a workspace folder that never follow
                            a link (O_NOFOLLOW folder and files, regular UTF-8 files only; memory_search, embeddings); ensure_real_dir(path, mode):
                            a workspace subfolder the bot prepares (downloads/, documents/) made a real directory
                            (a link or file there is unlinked) and chmod-ed through an O_NOFOLLOW descriptor
    trace.py                ToolTrace/ToolCall/TurnBudget: per-turn tool calls, token usage, shared tool budget;
                            on_late_usage (set after the turn's usage was recorded: later additions go there too)
    transcript.py           TranscriptStore + size control: one real API transcript per chat/topic (transcript mode);
                            prune clears old tool results (old <task_notification> blocks of transcripts made before the
                            system notices too, to their opening tag) AND
                            old media blocks (clear_old_media); answers_tool_call
                            (a user message whose FIRST block is a tool_result, injected blocks may follow) vs
                            is_tool_result_message (tool_result blocks only)
    browser_tools.py        BrowserTools: Anthropic's browser toolset (browser_toolset_20260801, javascript_exec/file_upload/
                            read_console/read_network disabled) executed by SessionManager.browser_call, browser_confirm
                            (screenshot + ✅ Allow / ❌ Cancel before a sensitive action; only a tap allows, typed text never
                            does), browser_show_screenshot, the live
                            view photo (after_batch). SandboxTools: sandbox_status + the sandbox facts in prompt_notes.
                            Finds secure_container.sessions through sys.modules only (never imports docker)
    subagent.py             SubagentTools: run_subagent tool (fresh AgentAnthropic, shared budget, nested trace; the
                            child inherits the registered providers except toolsets and inherit_to_subagents = False ones,
                            never a NEVER_BACKGROUND tool (ask_user_choice, send_user_telegram_message, …), and gets no
                            inbox or background host; its status updates stop once the call is in the background)
    __init__.py             imports SystemTools; the package object is what the patcher needs
  secure_container/
    main.py                 initialize_secure_containers() / cleanup_containers()
    container_manager.py    ContainerManager: builds image "secure-container:<12-hex hash of the build context>" when
                            that tag is missing (re-points :latest, prunes older tags), creates the SessionManager
                            (self.session_manager), run_command → the user's persistent session (one-off container as
                            fallback: run_one_shot, MAX_SANDBOX_CONTAINERS semaphore), run_script() for bundled scripts
    sessions.py             SessionManager (init_manager / get_manager): one persistent shell container and one browser
                            container per user, memory-aware admission (CapacityError), idle reaping + warnings
                            (reap_idle, due_warnings, extend), package persistence, the egress filter (iptables), the
                            browser's unix-socket protocol (browser_call, browser_peek); sync and thread-safe
    tool_integrator.py      ToolIntegrator: monkey-patches tool classes at startup (see below)
    secure_tool_wrapper.py  SecureToolWrapper: thin routing layer to ContainerManager
    scripts/                scripts that run INSIDE the sandbox (stdlib + the image's libs only; print ONE JSON line last):
      browse.py             Playwright Chromium page reader: url [--screenshot path] [--wait-ms] [--max-chars] →
                            {title, final_url, text, links, screenshot, error}; own 60 s deadline
      render_document.py    --format pdf|docx|xlsx --input --output [--title=] → {output, bytes, error}: HTML→PDF
                            (page.pdf), Markdown→DOCX (python-docx), Markdown→XLSX (openpyxl); own 75 s deadline
      browser_server.py     the long-running browser of the browser container: Playwright Chromium with a persistent
                            profile, one JSON request per line on a unix socket, executes the browser toolset members,
                            guards sensitive actions, sends all traffic through its own filtering proxy (no private
                            addresses), keeps session cookies across restarts, exits after its own idle time
                            (copied in at every browser start; not a run_script script)
    Dockerfile, entrypoint.sh, requirements.txt   the sandbox image (python:3.12-slim + tools, Playwright Chromium at
                            /opt/pw-browsers, non-root user "runner", passwordless sudo for apt only)
  stats.py                  StatsTracker singleton: SQLite data/stats.db, per-user action limits, the /usage helpers
                            get_user_usage_summary(user_id, days=30, daily_days=7) and get_user_workspace_size(user_id)
                            (os.walk of data/<uid>, symlinks by their own size, never followed)
  image_utils.py            JPEG/HEIC/PNG/WebP/GIF/BMP helpers (IMAGE_EXTENSIONS, flatten_to_rgb; pillow-heif registered
                            on import)
  voice/                    VoiceManager: ElevenLabs voice ids (voices.json) + per-user choice (config.json)
  Dockerfile, requirements.txt   bot image (python:3.12-slim + git, docker.io, ffmpeg, iptables for the egress filter); timezonefinder is a commented,
                            optional line there (time zone from a shared location)
tests/                      pytest suite (pytest.ini sets pythonpath=app, asyncio_mode=auto)
.github/workflows/deploy.yml  manual production deploy (workflow_dispatch, environment PROD), see "Running"
deploy/                     render_env.py (PROD secrets+vars → .env, runs on the runner) and
                            remote_deploy.sh (git checkout, .env swap, compose build/up, health check; runs on the server)
requirements-dev.txt        app requirements + pytest, pytest-asyncio
docker-compose.yml          services: telegram-bot-api (local Bot API server) + bot, and cloudflared (profile "tunnel":
                            the Cloudflare Tunnel to the sign-in callback, started only with COMPOSE_PROFILES=tunnel)
.env.example                template for .env (WORKSPACE_ROOT comes from docker-compose.yml; BROWSER_SERVICE_URL is unused)
docs/                       design notes: agent-roadmap.md (transcript / caching / subagents)
static/                     README images only: schema/memory-*.svg (the memory diagrams, light + dark through
                            prefers-color-scheme; the defaults they quote: 5 summaries, 3 similar turns, 120k tokens,
                            40 photos, 3 turns of tool results, 20,000 chars, 8 KB) and the donation QR codes
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
  down. The scheduled work is the reminders loop every 10 s, the `sandbox-reaper` every 60 s (closes idle
  sandbox/browser sessions, sends the "closing in 5 minutes" warnings), plus a one-off refresh of the admin
  MCP servers' tool cache at startup (`mcp-admin-refresh`) and, with `PUBLIC_BASE_URL`, the sign-in callback
  endpoint (`oauth-web`, `bot/oauth_web.py`), the only port the bot itself listens on.
- MCP OAuth sign-ins come back to `PUBLIC_BASE_URL/oauth/callback`. Recommended: a Cloudflare Tunnel, no
  inbound port. Zero Trust → Networks → Tunnels → Create (Cloudflared) → copy the token → Public hostname
  `gall.zor.gg`, path `oauth/*`, service `HTTP` `localhost:8090` (delete an existing A record of that host
  first; the tunnel makes its own DNS record). PROD environment: `CLOUDFLARE_TUNNEL_TOKEN` (secret),
  `COMPOSE_PROFILES=tunnel` (Compose reads it from the rendered `.env` and starts the `cloudflared` service,
  host network, so it reaches the bot's default listener `127.0.0.1:8090`), `PUBLIC_BASE_URL=https://gall.zor.gg`
  and `MCP_TOKEN_KEY` (secret, generated once with `python -c "from cryptography.fernet import Fernet;
  print(Fernet.generate_key().decode())"`, never changed afterwards: rotate with `MCP_TOKEN_KEY_OLD`). Check:
  `https://gall.zor.gg/oauth/health` → `ok`. Alternatives: the bot on 443 directly (`OAUTH_LISTEN=0.0.0.0:443`,
  a proxied DNS record; Cloudflare SSL mode "Full" accepts the self-signed certificate the bot generates in
  `data/tls/<host>.selfsigned.pem` / `.key`, "Full (strict)" needs a Cloudflare Origin Certificate saved as
  `data/tls/<host>.pem` / `.key`, or `OAUTH_TLS_CERT` / `OAUTH_TLS_KEY`; set the mode for that hostname with a
  Configuration Rule rather than switching the zone; "Flexible" is not supported: it talks plain HTTP to port 80),
  or any reverse proxy forwarding `/oauth/` to `127.0.0.1:8090`. Without `PUBLIC_BASE_URL` (or when the endpoint
  cannot bind: one ERROR line) sign-ins use the copy-paste fallback. `docker compose up --remove-orphans` keeps a
  container of a disabled profile: turning the tunnel off is `docker compose --profile tunnel stop cloudflared`.
- `MCP_TOKEN_KEY` missing: the bot generates `data/.mcp_token_key` once (0600, O_EXCL) and logs a WARNING; put
  that value (or a new key, with the file kept as a decrypt-only fallback) into the PROD secret so a lost `data/`
  does not sign everyone out. A changed or lost key never breaks anything: stored sign-ins read as "signed out".
- Persistent sandbox sessions (`secure_container/sessions.py`) need, on the Docker host: a free subnet for the
  `generall-sandbox` bridge (`SANDBOX_SUBNET`, default 172.30.240.0/24), and for the egress filter iptables
  rules written from the bot container (privileged, host network) into the same backend Docker uses
  (`DOCKER-USER`, else `FORWARD`, plus `INPUT`). `SANDBOX_EGRESS_FILTER=best_effort` (default) logs one warning
  and runs unfiltered when that fails; `required` refuses to start sessions instead and gives their one-off
  fallback containers no network; the reaper re-checks the rules every 300 s (a host firewall reload drops
  them; `required` then stops every session); check with /sandbox as
  admin ("Private networks blocked"). Sandboxes resolve names through `SANDBOX_DNS` (public resolvers by
  default) because the filter blocks private resolvers. Memory: a session starts only while
  `MemAvailable` (host-wide, `/proc/meminfo`) minus `SANDBOX_MEMORY_RESERVE_MB` covers its limit (768 MB shell,
  1536 MB browser) and the limits of all running sessions of both kinds, its own included, fit in `MemTotal`
  minus that reserve (one shared budget); the automatic count caps follow `MemTotal` (8 GB: 8 shells, 4 browsers,
  but not both at once: the budget holds e.g. 3 browsers and 2 shells). Two bot
  instances must not share one Docker host: startup removes every container labelled `generall.sandbox`.
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
   configures ffmpeg for pydub, starts the reminders loop, the `mcp-admin-refresh` task
   (`mcp_ui.refresh_admin_servers`) and the `oauth-web` task (`oauth_web.serve`: the sign-in callback endpoint
   when `PUBLIC_BASE_URL` is set; a bind or TLS failure is one ERROR line and the copy-paste fallback),
   `start_sandbox_sessions` (when a SessionManager exists: `cleanup_stale`
   removes leftover session containers, awaited up to 120 s, then the `sandbox-reaper` task), registers the
   command menu (`register_commands` → `set_my_commands(bot_commands())`; a refusal by an old local Bot API
   server is only logged) and gives `oauth_web` the bot's username (the sign-in page's `tg://resolve` link);
   `on_shutdown` cancels the `runtime.background_tasks` (reminders loop, reaper, the endpoint, …), then
   `oauth_flow.shutdown()` cancels the sign-ins still waiting for a browser (nothing announced), then
   `background.shutdown()` cancels every user's background tool calls (nothing announced; up to `SHUTDOWN_GRACE`
   3 s for them to end; from then on `can_adopt()` is False and a task handed to `adopt` is cancelled, so a turn
   still running hands nothing over) BEFORE `runtime.queue.shutdown()` cancels the running jobs (cancel reason "shutdown": their
   inboxes drop the unread messages instead of requeueing them), then `sandbox_ui.shutdown_sessions()` stops every
   session (browsers first asked to save their cookies);
   `main_bot` then calls `cleanup_containers()`. The SessionManager itself is created in
   `ContainerManager.__init__` (`sessions.init_manager`, failures logged: commands then run one-off).

Consequences: the tool list the model actually sees is **not** what you read in `agents/*.py`
alone; the source of truth is source file + `tool_integrator.py`. When you add or rename a method on
a tool class, check `SECURE_TOOL_METHODS` and the `patch_*` functions, or you may silently move
execution between host and sandbox.

## Queues, isolation and cancellation

- `bot/queue.py` gives every chat a queue and a lazily started worker task. Handlers only validate,
  enqueue a `Job(user_id, label, run)` (or hand the message to the running turn, below) and return within
  milliseconds. Every job gets a random 31-bit `id` (`Job.token` = its hex): the draft id of its streaming
  drafts and the Stop callback.
- Order is by arrival: every job carries `seq` (`queue.next_seq()`, process-wide, shared with inbox items), and
  `submit` / `submit_nowait` insert it before every pending job with a larger `seq` (`_UserQueue.pending` is the
  job list, the `asyncio.Queue` only holds one token per pending job). A job made when it is submitted just goes
  to the end; one submitted later for something that arrived earlier keeps its place (an inbox leftover, an
  album flushed 10 s after its first photo, whose `_Album.seq` is that photo's arrival, a message window: the
  smallest `seq` of its messages).
- **The message window: messages sent together are one turn.** A content message that does not join a running
  turn goes into the window of its chat and topic (`messages.accept` → `_gather`; `MESSAGE_WINDOW` =
  `config.message_window_seconds`, env `MESSAGE_WINDOW_SECONDS`, default 1.5, 0 = off). The window closes
  `MESSAGE_WINDOW` after its last message (each message moves `_Window.due`), at most `MESSAGE_WINDOW_MAX` (30 s)
  after its first, and not while a photo album of that chat and topic is still being collected or handed over
  (`_album_arriving`: `_albums` and `_albums_handing`; the album's 10 s flush adds it to the window, which then
  waits `MESSAGE_WINDOW` more; with no window open an album goes straight on as before). Its messages are kept in
  Telegram `message_id` order (concurrent forum handlers may finish `remember_topic` out of order). What the user
  sees: the first message gets "typing…" (`ChatSender.typing`) and nothing else; from the second on every message
  gets 👀 (`inbox.react`), its `prepare_<kind>` starts at once (`_start_preparing` → `_prepare_for_inbox`: the
  PREPARE_SLOTS, the echo of a voice transcription), an album's "waiting for images" line is deleted, and ONE
  status line counts them (`WINDOW_STATUS` "📥 *Got N messages.* I'll read them all together.", `_update_count`: one
  Telegram call at a time). At the close (`_close_window`, no await between taking the window out of `_windows`
  and handing its messages over): a turn of that topic running by now takes them (`inject` each, with its running
  preparation as `future`; 👀 for the unseen; the count line deleted); else ONE message is queued as exactly the
  job it was (`Job(run=entry.run)`, its own `_run_<kind>`), two or more as ONE job (label "N messages",
  `_run_prepared` over their preparations with `status=` the count line, which `run_turn` makes the turn's status;
  `on_drop` = `_drop_window`). One busy notice at most, on the first message. While a window collects for a topic,
  `inject_target` returns None for it, so a later message of that topic waits in the window too. `/cancel` drops
  every open window of the user (`drop_windows`: preparations cancelled, the count line edited to "🛑 Cancelled.",
  counted as dropped messages). Not windowed: a message that joins a running turn, button presses (Retry,
  Continue, follow-ups), scheduled tasks and notification turns.
- **Live turns: joining the running turn.** `run_turn` opens an inbox (`bot/inbox.py`) right before it builds
  the agent and closes it the moment `generate_response` returned, failed or was cancelled. Meanwhile a new
  message of the same chat AND forum topic joins that turn instead of queueing: `messages.inject_target(user_id,
  thread_id)` = `inbox.current(user_id)` accepts that topic and `queue.has_pending_user_jobs(user_id, thread_id)`
  is false (an earlier message or Retry/Continue/follow-up press of that topic still queued keeps the new one
  queued behind it; jobs with `from_user` False, i.e. scheduled tasks and notification turns, never block).
  Joining = `messages.inject` (an `InboxItem` with the arrival `seq` whose future is the `prepare_<kind>` task,
  or an already ready `Prepared`) + a 👀 reaction (`inbox.mark_seen`, awaited); no busy notice. At most
  `messages.PREPARE_SLOTS` (2) preparations of one user run at once (`_prepare_slot`, a per-user semaphore inside
  the prepare task, so a leftover's turn awaiting it is bounded too), and a document or audio file is downloaded by
  one of them at a time (`_path_lock`: same-name uploads overwrite). The agent reads it at its next tool boundary (see
  "Request flow") and the message gets ✍. Scheduled tasks and notification turns run through `run_turn`, so
  they take messages too. A message of another topic, one sent while a `_run_<kind>` job still prepares its
  media (no inbox yet) or while the answer is being delivered (inbox closed) is queued as before.
- If the message does not join and the user already has a running or queued job, `messages.notify_busy`
  replies (once its message window closed) "⏳ I'm still working on your previous request (label, elapsed, step) …"
  with a Stop button (`queue_cancel` callback). The new message is processed later as its own turn, in order.
- Leftovers: what the inbox still holds when it closes (`inbox.finish`) is handed back in arrival order. The
  user items together become ONE queued turn (`inbox.requeue` → the first item's `requeue_group` =
  `messages._requeue_group` → `Job(seq=first.seq, from_user=True, label "N messages" for several)` running
  `_run_prepared` on the `Prepared`s they already have, put into one prompt by `inbox.combine`: nothing is
  downloaded or transcribed twice; a failed one is a note to the agent, all failed just sends their `error_text`;
  one leftover is its own prompt as before), each task notice a notification turn (`background.submit_notification`, none
  when the result was read meanwhile). That
  happens after a normal end, a failure, Stop and the turn deadline (a Stop while `drain` is still reading a
  message leaves it a leftover: an item leaves the inbox only once read). `/cancel` drops them instead
  (`inbox.discard` runs before `queue.cancel`, so the cancelled turn finds its inbox closed), and so does
  shutdown (`run_turn` closes with `discard=True` when `ctx.cancel_reason == "shutdown"`). A requeued job carries
  `Job.on_drop` = `inbox.drop(items)`: `/cancel` dropping it also cancels a preparation still in flight (no
  echo, no transcription or description after the cancel); cancelling it while it runs cancels them too
  (`_run_prepared` awaits them in one `asyncio.gather`). A quiet run (a scheduled task or notification turn
  that answers the marker) hands back the user messages it DID read too (`inbox.consumed_user_items`, passed
  to `inbox.finish(read=…)` with the unread ones: one turn), so they get a real answer; the run itself stays silent.
- **Live turns: background tool calls.** A call the agent loop moved to the background (see "The tool
  contract") belongs to the user's `BackgroundHost` (`bot/background.py`) from then on, across turns. When it
  ends, its raw result stays on its record until the agent reads it (`get_task_result`, or `wait_for_tasks` returns
  it: `host.read` → `result_blocks`, text capped at the call's `max_chars`, images kept, always inside a
  `tool_result`), and only a notice of facts announces the end ("The background task t3 (run_command) completed
  after 42 s. Read its result with get_task_result (task_id "t3")."; failed: "Read the error …"; cancelled: no
  read hint). The notice goes to the inbox of the user's running turn in the topic it was started from
  (`inbox.deliver_task`; its blocks are built when drained, so a result read meanwhile gives none), and the agent
  core sends it as a `system` message; with no such turn it queues a "background task finished" turn:
  `Job(label "background task result", thread_id)` → `_run_notification` → `run_turn(prompt=notification_prompt(…)`
  (`NOTIFICATION_PROMPT` with one "- The background task … completed after N s." line per task, a plain string),
  `header=NOTIFICATION_HEADER, quiet_marker=QUIET_MARKER)` ("🔔 *Background task finished*"; `<no_update/>` keeps it
  silent). Tasks that finish while that job is still queued join it (`_Batch.records`); one read before the job
  starts is left out, and with none left there is no turn; a task the agent cancelled itself (`cancel_task`) is
  reported to a running turn only. A turn with an unread result (`host.unread()`) or a notice never takes the
  quick tool-less path (`_decide_complexity(needs_tools=…)`). Finished records are kept up to `MAX_LISTED` (20),
  read ones pruned first. Like scheduled tasks, notification turns
  call no `track_message_received`. Stop never touches background tasks; `/cancel` cancels the user's silently
  and forgets a queued notification turn (`background.cancel_all`; the reply adds "cancelled N background
  task(s)"); a turn that fails cancels, silently, the ones it adopted itself (`host.cancel_turn(host.turn)`; a
  turn whose answer already went out keeps them), since its Retry starts them again; shutdown cancels everyone's
  (`background.shutdown`). The cap of 4 running tasks is per user, but the task tools see the running turn's
  topic only (`running` / `list` / `cancel` / `unread` / `read` filter on the topic `host_for` bound):
  `wait_for_tasks` in topic B neither waits for, lists nor reads a task of topic A.
- `/cancel` (and the busy notice's button) is `messages._cancel_for`: `inbox.discard`, `drop_windows`,
  `background.cancel_all`, then `QueueManager.cancel`, which cancels the running task and drops the queue (a
  queued job of several messages counts as one in the reply). The Stop buttons of a turn are
  `cancel_current(user_id, job_id)`: the running job only, and
  only while it is still the job the button names (queued messages run next, the turn's unread messages
  requeued among them by arrival, `seq`; its background tasks keep running). Two Stop buttons exist:
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
  timeout. In a live turn a watched call that was not handed over is cancelled with the turn (`_watch_call`
  cancels its task; a worker thread under it still runs to its end); a handed-over one is the host's.
- Isolation is real only because blocking work is off the event loop: `AgentAnthropic.execute_tool`
  runs synchronous providers (search, code, terminal, sms, and every provider registered with
  `blocking=True`: web, document) via `asyncio.to_thread`; image/video tools wrap SDK calls in
  `to_thread` and poll with `asyncio.sleep`; MCP calls are async; profile and MCP store I/O go through
  `to_thread`; media pre-processing uses `to_thread` for PIL/pydub/ffmpeg/pypdf/OpenAI/pandas, and
  `content.materialize` (base64 of the pictures) runs in a thread too. **Any new blocking call must be
  wrapped the same way**, otherwise it stalls every user again.
- Global caps: `MAX_CONCURRENT_TURNS` (asyncio semaphore in `QueueManager`), the session caps of
  `secure_container/sessions.py` (memory admission + `SANDBOX_MAX_SESSIONS` / `BROWSER_MAX_SESSIONS`) and
  `MAX_SANDBOX_CONTAINERS` (threading semaphore in `container_manager.py`, one-off containers only:
  `run_script`, the fallback). `TURN_TIMEOUT_SECONDS`
  cancels a stuck job and tells the user; the worker continues with the next job.
- Agent reminders go through the same per-user queue (`bot/jobs.py`, label "scheduled agent task"), so they
  never interleave with the user's own turn (it runs with `thread_id` None, so a message sent outside any forum
  topic while it runs joins it; when the agent read one and still answers the quiet marker, the run stays silent
  and the message is requeued, with any it did not read, as one turn). The job is `run_agent_reminder`: `run_turn(prompt=task_prompt(...),
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
- Forum topics: the queue key is the chat id, so all topics of one chat share one queue; joining a running turn
  is per topic (`TurnInbox.accepts(thread_id)`), and so is the delivery of a task notification.

## Request flow (text message)

```
messages.on_text (bot/handlers/messages.py)      [AuthMiddleware(check_limits=True) already ran:
  user_id = str(message.chat.id)                   authorized, quota ok, kwargs user_id/limit injected]
  stats_tracker.track_message_received
  questions.waiting_in(chat, topic) → questions.answer_in_text (the text answers that ask_user_choice) → return
  accept(): inject_target(user_id, topic) → inject(prepare=prepare_text) + 👀 → return   [joins the running turn]
            else _gather(): into the message window of chat + topic → return   ["typing…"; from the 2nd: 👀 + count]
            (MESSAGE_WINDOW 0: submit(Job(run=_run_text, thread_id, from_user=True)) → busy notice if needed → return)
  window closes (_close_window, see "Queues"): one message → submit(Job(run=_run_text, …)) → busy notice if needed;
            several → Job(_run_prepared over their preparations, status=count line); a turn of the topic running by
            now → inject each
worker → _run_text → prompt = message_context.with_reference(text, message, bot.id): when the message replies to,
    quotes or forwards something, a plain-text preamble goes first (see "Media handlers" below), then a blank
    line and the user's text
  → bot.agent_runner.run_turn(bot, user_id, chat_id, prompt (str | content blocks), thread_id, reply_to, ctx, limit)
    prompt_text = content.prompt_text(prompt) (text view), prompt_blocks = _record_blocks(prompt): the blocks when
       the prompt is a list, a materialized image/document block (a notification turn's) as its marker
    sender = ChatSender(bot, chat_id, thread_id, reply_to_message_id)
    UserSettings(user_id).save() (persists backfilled defaults) → user_settings dict
    status = sender.send_text("💭 *Thinking...*", Stop keyboard) wrapped in bot.status.StatusMessage; update_status()
    edits it in place: in rich mode editMessageText(rich_message=blocks) (header, step, iterations, an open
    details block with the last 15 tool calls, usage line, 🛑 Stop row `queue_stop_<job token>`); a rejected
    rich edit switches that status to the legacy-Markdown text (last 10 calls, Stop as inline keyboard), a
    404 marks rich unsupported process-wide. Drafts use draft_id = Job.id with Telegram's Stop
    takes_live_turn(ChainOfThoughtAgent) (it takes inbox= and background=) → box = inbox.open_inbox(user_id,
       thread_id, base=data/<uid>) (messages of this chat and topic join from now on) and
       host = background.host_for(user_id, bot, chat_id, thread_id) (a call adopted this turn reports to this topic)
    agents.ChainOfThoughtAgent(model=models.ANTHROPIC_MODEL, user_id, sender, user_settings, thread_id, inbox=box,
       background=host)
       mode = settings.mode.type: speed → AgentAnthropic(model=ANTHROPIC_MODEL_FAST), thinking False;
       intelligence → ANTHROPIC_MODEL, thinking True; auto (default, unknown) → ANTHROPIC_MODEL, thinking = the
       `thinking` toggle. Named slots (file_ops … subagents) + register_provider(Vision, Profile, MCP, Web*, Document*,
       Sandbox†, Browser†, Task‡) (* = blocking=True, run in to_thread; † = with a session manager, see the tool table;
       ‡ = TaskTools(host, box), only with a host); agent.inbox = box, agent.background = host
    .generate_response(question, update_status, on_text_chunk)
       ├─ agent.on_recovery = update_status(step="recovering", details=note) (transcript and legacy mode alike)
       ├─ question_blocks = content.with_prefix(normalize(question), "Message received time in UTC+0: <ts>\n\n")
       │     (the prefix goes on the first text block); question = content.prompt_text(question_blocks): the text
       │     view, media as [image: <path>] / [document: <title>], used by the classifier, memory, logs, legacy path
       ├─ pick system prompt by settings.system_prompt.type (4 inline prompts; unknown → generall-ai-v2),
       │     + RICH_/LEGACY_FORMATTING_GUIDE, + FOLLOW_UPS_GUIDE when follow_ups.enabled (<follow_ups> block),
       │     + _provider_notes(): every provider's `prompt_notes` inside ONE <capabilities> block (VisionTools,
       │       ProfileTools and UserInteractions always, MCPTools only when the user has MCP tools, TaskTools'
       │       <live_turn> with a host) — all static, part of the cache prefix
       ├─ TRANSCRIPT MODE (default, transcript.enabled): _generate_with_transcript → load
       │     data/<uid>/transcripts/[topic_<id>_]transcript.json (seeded from dialog_history.json on first use);
       │     static prompt (time stripped) as ONE cached system block (ttl SYSTEM_CACHE_TTL); fingerprint of
       │     model + static prompt + tools changed → strip stored thinking; _request_context(question, status,
       │     messages) → <context> = UTC time + "Local time (<zone>): … (UTC+2)" or "User timezone: unknown (ask
       │     once and save it with set_user_timezone)" (_render_context, settings timezone.name) + in a forum topic
       │     'Telegram thread (forum topic): "<name>", thread id <id>' (_topic_line from bot/topics.py, every turn;
       │     "thread id <id> (name not known yet)" when no name was seen) + <memory> (recent
       │     summaries + FAISS hits not shown before) +
       │     <profile fingerprint="..."> (profile.enabled; only when the fingerprint differs from the last one shown
       │     in this transcript), stored as the first block of the new user turn; _read_early_messages: ONE
       │     _drain_inbox before the first model call (what the user sent while the turn was starting; status step
       │     "reading your message"), its blocks go behind the question in this turn's user message (not sent yet, so
       │     nothing stored changes), a task notice among them as a {"role": "system"} message right after it
       │     (split_notices / notice_messages); _decide_complexity on with_messages_read(question, inbox): mode auto AND
       │     no media (refs or drained image/document blocks) AND no notice or unread background result
       │     (needs_tools: host.unread()) → Haiku classifier (text view, context blocks and system messages excluded),
       │     else "complex"; "simple" → tool-less Haiku answer on the FULL context (profile every time, the drained
       │     texts too); _build_user_message materializes the media refs
       │     (base64 image/document blocks, to_thread) behind the <context> text block; tool loop on the real
       │     transcript (tool results capped before the model sees them); history_edited → strip every thinking
       │     block of the stored transcript first (the prefix was sent stripped); then append the turn exactly as
       │     sent, judge/critique/SYSTEM NOTICE included; prune only above max_context_tokens or 40 pictures (clear old
       │     tool results AND old image/document blocks → text markers, pictures down to 20, summarize oldest half,
       │     strip thinking); save;
       │     summaries/FAISS/dialog_history.json still written. LEGACY MODE below:
       ├─ semantic_search: FAISS hits appended to the system prompt (to_thread; failure → skipped)
       ├─ context_memory = fixed intro pair
       │     + conversation summaries (summarization_history)   [reads data/<uid>/conversations/*.json]
       │     + dialog_history.json, then context_memory[:-2]     (drops the newest pair; if the file is
       │       empty it eats the summaries pair or the intro pair instead)
       │     + short_term_memory.json messages whose FIRST block is text (reasoning_context)
       │     + the question                       (all three gated by short_term_memory.enabled too)
       │     the user turn = _legacy_context() (<context>: UTC time, local time/zone, <profile> when profile.enabled;
       │     no memory lines) + a blank line + the question (+ the TEXT blocks _read_early_messages drained, no
       │     pictures on this path), quick path too; afterwards thread_messages gets the
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
       │        return a str OR a list of text/image blocks: cap_blocks cuts only the text, the trace shows [image]);
       │        with a host, each normal call outside NEVER_BACKGROUND runs as its own task under _watch_call: still
       │        running after TOOL_BACKGROUND_AFTER_SECONDS (30), or after BACKGROUND_ON_MESSAGE_AFTER (3 s) while a
       │        user message can be read from the inbox (has_user_messages: ready, not one still being
       │        prepared), and host.can_adopt() → host.adopt(task, …) and its tool_result is
       │        background_placeholder(tN) (trace: "moved to the background as tN")
       │        then injected, notices = split_notices(_drain_inbox()) (the user's messages and task notices, ready
       │        head only, in arrival order) and append ONE user msg = tool_results in block order + injected (the
       │        user's words never inside a tool_result), then the notices as ONE {"role": "system"} message
       │        (facts only: no tool output; a <bot_notice> text block at the end of that user message instead for a
       │        model without system messages, models.supports_system_messages), stored as sent;
       │        a message read → status step "reading your message" ("Reading your new message" / "… N new
       │        messages"); continue (cicles counts tool_use BLOCKS). Nothing is injected anywhere else in the loop:
       │        not after end_turn, not into judge/critique re-prompts or the SYSTEM NOTICE call (before the first
       │        call ChainOfThoughtAgent._read_early_messages read the inbox once, into the turn's own user message)
       │     gate fails → tool_use blocks silently dropped, text goes on to critique/judge/return
       │        (budget spent → AgentAnthropic.limit_reached, which offers ▶️ Continue)
       │     else optional critique (OpenAI gpt-6.1-sol) / judge (Claude yes/no, same request_options_for) re-prompts → return
       │        (both judge the answer against with_messages_read(question, inbox): the messages read mid-turn included)
       │     for-loop exhausted → forced "SYSTEM NOTICE" final call with the loop's tools and tool_choice none (same thinking mode;
       │        limit_reached too)
       └─ strip pre-loaded context from thread_messages; persist memory inside try/except
          (a failed Haiku summary is logged, the answer is still returned; the memory files get _memory_question =
          with_messages_read: the question + "\n\n[Sent while I was working]\n" + the texts the agent read mid-turn)
          → (response, messages)
    agent returned → quiet = quiet_marker and the answer (follow-ups cut) is exactly it → close_inbox(read=
       box.consumed_user_items if quiet) (inbox.finish: leftovers become the next turn, see "Queues"), drafts closed,
       ctx.stoppable = False, Stop removed from later status edits
    stats_tracker.track_message_sent
    record_usage(): one usage_events row per model of the turn (tokens, cache, tool calls, cost estimate); then
       record_late_usage() sets trace.on_late_usage: whatever a call moved to the background adds to this trace later
       (Veo, a subagent's model calls, a slow image) is stored as it arrives, one row per addition, exactly that usage
    turn_actions.parse_follow_ups(response) → (display, suggestions): the LAST <follow_ups> block is cut from
       the visible answer (always) and becomes ≤4 suggestions (none when follow_ups is off)
    quiet_marker given (scheduled tasks: bot/jobs.run_agent_reminder passes QUIET_MARKER "<no_update/>" with the
       framing prompt, see "Queues"; background.py's notification turns pass it too) and display == marker →
       nothing sent, status deleted, TurnResult(quiet=True); such a turn runs with browser.stream off (no live-view
       photo to leave behind). The user messages the agent read mid-turn (box.consumed_user_items) were handed back
       with the unread ones by close_inbox above (one turn on their prepared prompts, at the first one's seq)
    save_turn(): TurnRecord(id 12 hex, prompt (text view), prompt_blocks (media refs, Retry re-runs them), display,
       speak, thread, limit_reached, follow_ups) + reasoning text (if reasoning_context.enabled) → data/<uid>/turns/
       (to_thread, newest 50 kept). with_drained(prompt_text, prompt_blocks, box): prompt + "\n\n[Sent while I was
       working]\n" + the texts read mid-turn, prompt_blocks + their text and media-ref blocks (still None when the
       prompt was a plain string and nothing read carried media), so Retry replays them too
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
    errors → close_inbox() (leftovers requeued); logger.exception("Turn failed … (trace <uuid>)"); record_usage +
       record_late_usage unless the agent had already returned (counted then); a status that already became the
       answer/summary is left alone; else host.cancel_turn (the background calls this turn adopted end silently:
       Retry starts them again) and save_turn(TurnRecord(prompt, prompt_blocks (both with_drained), response="",
       error=friendly_error(e, trace_id))) (no reasoning file) and
       status_msg.fail("❌ <reason> (trace <uuid>)", actions=record_actions(record) = 🔁 Retry only): rich paragraph +
       button row → inline keyboard once rows are refused → plain text + keyboard; a failed save → the text without
       buttons. friendly_error: RateLimitError → ERROR_RATE_LIMIT, APIConnectionError/APITimeoutError → ERROR_NETWORK,
       APIStatusError ≥ 500 or overloaded → ERROR_OVERLOADED, 413 or a BadRequestError saying too large / prompt is too
       long / max_tokens / exceed → ERROR_TOO_LARGE, aiogram TelegramAPIError → "Telegram refused the message: <≤120
       chars>", else "Something went wrong.". Returns None
    CancelledError → close_inbox() (Stop and the deadline requeue the unread messages; shutdown drops them;
       after /cancel the inbox is already closed and emptied); record_late_usage (the turn's own usage is not
       recorded on a stop, its background calls' later usage is); user stop before the agent returned: partial answer;
       status "🛑 Stopped." (user) / "🛑 Cancelled." (deadline, shutdown), unless it already became the
       answer/summary; re-raised
    finally → close_inbox() again (idempotent)
```

Media handlers (`_run_voice`, `_run_video`, `_run_audio`, `_run_images`, `_run_document` in
`messages.py`) are jobs with the same shape and only differ in how they build the prompt. Each kind is split
in two: `prepare_<kind>(bot, message, user_id[, …, progress])` does the download, conversion, transcription or
extraction and returns a `bot.inbox.Prepared` (`prompt`, `speak`, `error_text` for the user, `reason` for the
agent, `echo` = the voice transcription or the video analysis shown once ready, `inject` = what a running turn
reads instead, only for voice: "(voice message, transcribed) …"); `_run_<kind>` wraps it in the status messages
and calls `run_turn`. The `on_<kind>` handlers call `accept(message, …, prepare=, run=)`: joining a running turn,
`prepare_<kind>` runs as a task without status messages (`_prepare_for_inbox`: the echo goes out as a reply to the
message; an exception or an unusable result becomes "(the user sent a <kind> that could not be processed:
<reason>)" for the agent), else it goes into the message window and, when that closes, the `_run_<kind>` job is
queued (alone) or its `prepare_<kind>` (started with the window's second message) becomes part of one
`_run_prepared` turn. Leftovers run as ONE `_run_prepared` turn on the same `Prepared`s (one: its `prompt`, not
`inject`, and a failed one only sends its `error_text`; several: `inbox.combine`, each message's `inject_prompt()`
inside `<message n="i">`). An album joins, queues or enters an open window as ONE item at
its 10 s flush, with its first photo's `seq` (the "waiting for images" message is deleted when it joins). Every job
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
- Optional toolset (only `BrowserTools` today): a `toolset` attribute (a nameless `tools` entry such as
  `{"type": "browser_toolset_20260801", "configs": {...}}`, appended by `get_tools_schema` after the provider's
  own tools), `toolset_name` ("browser"), `async execute_member(name, input) -> (content, is_error)` and an
  optional `async after_batch()`. The model's `tool_use` blocks for it carry `toolset_name`; they are stored
  with it (append-only transcript) and their `tool_result` must echo `"toolset_name"`, normal results must
  not. Member results may hold `text`, `image` and `browser_state` blocks. `allowed_tools` keeps a toolset
  only when its `toolset_name` is allowed (subagents never get one). Helpers in `agents/main.py`:
  `toolset_of`, `tool_display_name` (trace: `browser.<member>`), `block_input`, `toolset_provider`, `batch_units`.

Tools that talk to the user receive a `sender` (`bot.sender.ChatSender`) in their constructor, never
a Telegram update: `sender.send_text(md)`, `send_markdown(md)` (splits, fallback), `send_document(path|bytes,
filename, caption)`, `send_photo`, `send_video`, `send_voice(bytes)`, `react(emoji)` (only when the turn
has a user message), `typing()`. Captions are truncated to 1024. This is what makes tool sends work in
scheduled agent reminders too. `sender` may be `None` for unit tests; guard before using it. A tool whose point is
to talk to the user right now or to wait for them (a question, a confirmation, a message, a reaction) must be
listed in `agents.main.NEVER_BACKGROUND`, or a live turn may move it to the background (below).

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
  `DocumentTools(user_id, sender)` (`blocking=True`), then `SandboxTools` / `BrowserTools` (with a session
  manager) and `TaskTools(background, inbox)` (only when a background host was given). `SubagentTools._child`
  copies `extra_providers`, so subagents get the registered tools too (minus `run_subagent`, toolset providers and
  any provider with `inherit_to_subagents = False`: `TaskTools`), and never a tool in `NEVER_BACKGROUND`
  (`_available_tool_names` drops them and `_child` always sets `allowed_tools` without them).
- First match wins in both, so tool names must be globally unique (`mcp_*` names are sanitized and
  de-duplicated per server set).

Tool results: a `str` becomes the single `tool_result` content (cut to `transcript.max_tool_result_chars`
first), a list of blocks is passed through with only its text blocks capped (`content.cap_blocks`) and
shown as `[image]` in the trace (`content.result_preview`). Every call is recorded via
`stats_tracker.track_tool_used` (and counts against the user's action limit). Tool schemas are sent to the
API even when `tools.enabled` is false. All `tool_use` blocks of one assistant message run **concurrently**
(`AgentAnthropic.run_tool_batch`), so a provider must be safe to call from several tasks at once; an
exception becomes a `tool_result` with `is_error: true`, a returned string starting with "Error" is passed
through as content (shown ❌ in the trace). Toolset members are the exception: one toolset's calls run in
block order, one at a time (alongside the concurrent normal tools), and the first failure halts the rest,
which get `is_error` with `NOT_EXECUTED` ("Not executed: an earlier action in this turn failed."); then
`after_batch()` runs. A message with browser calls costs ONE `TurnBudget` unit and ONE
`track_tool_used(uid, "browser")`, however many members it holds (a failing quota write, e.g. SQLite "database is
locked", is logged and the batch runs on; for a normal tool it becomes that tool's error result).

Background calls (live turns: `AgentAnthropic.background` is the user's `BackgroundHost`, which `run_turn` passes
on every turn, `takes_live_turn`): every normal call whose name is not in `NEVER_BACKGROUND` (`ask_user_choice`,
`browser_confirm`, `browser_show_screenshot`, `send_user_telegram_message`, `set_message_reaction`,
`send_voice_message` and the four task tools) runs as its own asyncio task, created in a copied context that carries a `task_tools.CallState`, and
`_watch_call` waits for it. Once it has run `TOOL_BACKGROUND_AFTER_SECONDS` (30; 0 = never by time), or
`BACKGROUND_ON_MESSAGE_AFTER` (3 s) while `inbox.has_user_messages()` (a message ready to read; one still being
prepared does not count, the watcher polls every 0.5 s while `has_pending()`), and `host.can_adopt()` (fewer than 4
running, not shutting down), the task is handed over: `host.adopt(task, tool_name, tool_use_id, display_args, started, max_chars)`
returns `tN`, `CallState.background_id` is set (so `task_tools.in_background()` is true inside the call), the
`tool_result` is `background_placeholder` ("Still running in the background as task tN (started HH:MM:SS UTC). A
system message will say when it has ended, … then read its result with get_task_result. …") and the trace shows
"moved to the background as tN". A full host
keeps the call in the foreground until it has room (re-checked every 0.5 s); a host that raises or returns no id
leaves it there to its end (during shutdown the host cancels the task it refuses: the call ends as cancelled). The host gets the raw `execute_tool` result (or exception), keeps it, and caps its text at the
turn's `max_tool_result_chars` itself when the agent reads it (`result_blocks`: "Result of the background task tN
(<tool>) completed after N s:" + the text, images kept; a result starting with "Error" counts as failed); the end
itself is announced by a notice of facts only, a `system` message (see "Queues"). The call was
counted by `track_tool_used` when it started. Toolset members never go to the background. Without a host (tests,
subagents) every call runs to its end as before.

Effective tool list as the model sees it (after patching; 51 built-in tools, the browser toolset and the user's
`mcp_*` ones; with no session manager there is no `sandbox_status` and no browser, with `browser.enabled` off no
browser, with no background host no task tools: 44 at the least):

| Provider | Tools | Runs where |
|---|---|---|
| FileOperations | `list_files`, `create_file`, `read_file`, `download_file`, `download_webpage` (readable main content: `trafilatura.extract(include_tables=True)`, lazy import cached in `file_ops._TRAFILATURA`, BeautifulSoup page text when trafilatura is missing, raises or finds < `MIN_EXTRACTED_CHARS` 200 chars), `upload_to_s3`, `create_zip_archive`, `send_file_path_to_user_via_telegram` | bot process, paths resolved under `data/<uid>/` by `agents/paths.resolve_under` (escapes refused); the names download_file / download_webpage(save_to_file) / create_zip_archive write keep only their base name (`FileOperations._download_target` → `downloads/<name>`, so a link planted in downloads/ that points outside is refused) |
| FileOperations | `create_directory`, `delete_file` | the user's sandbox session (`run_command`), paths relative to `data/<uid>` |
| SearchTools | `search_web` (Tavily), `deep_research` (Perplexity `sonar*`) | bot process, worker thread |
| SearchTools | `memory_search` (query with `" | "` alternatives, case_sensitive, max_results 1..100 default 20): substring search over `conversations/*.json` (summary, topic, question, response), `profile.md` (field = section) and the text view of `transcripts/*.json` (user/assistant text only: no context blocks, tool results, thinking or media); JSON array of `{source, timestamp, topic, field, snippet}` newest first, or "No matches for …" | bot process, worker thread (reads every conversation file and transcript on each call through `agents.paths.regular_files`: a symlinked folder or file, or a FIFO, is skipped, because data/<uid> is the sandbox's workspace) |
| CodeTools | `execute_python` (+`network_enabled`, ignored in the session: it always has filtered internet) | the user's sandbox session |
| TerminalTools | `run_command` (bash syntax works), `run_shell_script` (+`network_enabled`, ignored in the session), `install_package` (package_name, manager apt\|pip; persisted and reinstalled at every session start) | the user's sandbox session (a one-off container when none can start) |
| TimeTools | `get_time_in_timezone`, `list_timezones` | bot process |
| ImageTools | `generate_image` (no `images` → generate, one → edit, several → compose; engine `auto`/`best`/`fast`/`story`), `generate_story_images` | bot process (SDK calls in worker threads), never sandboxed, delivers each result itself as a document, once |
| VideoTools | `video_generator`, `image_to_video_generator`, `video_from_reference_images`, `video_interpolation_generator`, `video_extension_generator` | bot process (Veo in worker threads, async polling up to 5 min) |
| SMSTools | `send_sms` | bot process, worker thread |
| UserInteractions | `send_user_telegram_message`, `send_voice_message`, `set_message_reaction`, `set_user_timezone` (IANA name or city/country alias via `timezones.valid_zone`; saves settings `timezone.name` in to_thread and returns the user's local time, `Error: unknown time zone …` otherwise), `schedule_reminder` (reminder_text; reminder_time `YYYY-MM-DD HH:MM[:SS]` read in the user's zone, or in the optional `timezone` param, unless it carries a `Z`/`±HH:MM` offset, which wins; reminder_type user\|agent; is_periodic + period_type hourly\|daily\|weekly\|monthly + period_interval; `weekdays` [0=Mon..6=Sun] for daily periodic reminders only, the first run moved to an allowed day by `first_allowed_trigger`; stores UTC plus the zone; confirms in local time; user reminders fire with snooze buttons, agent tasks run as scheduled tasks and may answer `<no_update/>`), `send_file_content_to_user_via_telegram`, `ask_user_choice` (question + 2-8 option buttons, waits for the tap up to `timeout_seconds` 30..900, default 300; returns `User chose: "<option>"`, `TYPED_DELIVERED` ("User answered in text instead of tapping a button: their message follows the tool results (a <user_message> block); read it as their reply to this question") when the user typed a plain text in that chat and topic instead and the running turn took it as a message, `User answered in text instead of tapping a button: "<text>"` when no turn took it, a no-answer notice or `Error: …`); its `prompt_notes` (static) tell the model to infer or ask for the zone once and save it, to schedule and confirm in local time, and to answer exactly `<no_update/>` on an uneventful scheduled run | bot process |
| SubagentTools | `run_subagent` (task, tools subset, model main/fast, max_tool_calls) | bot process: a fresh `AgentAnthropic` with the same providers (registered ones included), empty non-persisted context, budget slice of the parent's `TurnBudget`, trace depth 1; main agent only (a subagent has no `run_subagent`); model "main" is `ANTHROPIC_MODEL` whatever the mode |
| VisionTools (registered, awaited) | `view_image` (path, max_px 128..2048, default `MAX_IMAGE_RESOLUTION_VISION`) → `[image block, text "<rel path> · WxH px (shown at wxh)"]` (JPEG/PNG/GIF/WebP within the limits and upright pass through unchanged, else re-encoded as JPEG); `describe_image` (path, question?, model anthropic\|openai) → text from `bot.media.describe_image_anthropic/openai` (temp JPEG in the system temp dir; the OpenAI call writes its own usage row); `read_document` (path, question?) → a PDF within the limits as `[document block, text "<name> · N pages"]`, else extracted text capped at `vision_tools.EXTRACTED_TEXT_CAP` (60000), unknown types through `describe_document` | bot process, never sandboxed; paths via `resolve_under` with images/downloads/documents fallbacks |
| ProfileTools (registered, awaited) | `user_profile_read`, `user_profile_add(section, text)`, `user_profile_remove(section, needle)`, `user_profile_replace(section, old, new)`; needle = exact entry, a unique substring or a 1-based index; results = confirmation + the numbered section; registered always (the `profile` setting only gates the `<profile>` block in the context) | bot process, file I/O in to_thread |
| MCPTools (registered, awaited) | `mcp_<server>_<tool>` for every tool of the user's enabled MCP servers and the admin's (names sanitized to `[a-zA-Z0-9_-]{1,64}`, collisions `_2`, `_3`…; description `[MCP <server>] …`; input_schema passed through, an empty object schema when missing/invalid); `prompt_notes` only when the user has MCP tools | bot process: schemas from the cached tool lists (NO network at turn start); one fresh connection per call (streamable HTTP or SSE by `transport`, stored headers) → `initialize` → `call_tool` → closed; `CALL_TIMEOUT` 60 s (`DISCOVER_TIMEOUT` 30 s); text → joined text, images → base64 `image` blocks (list result), embedded resources → their text or `[resource <uri>]`, `isError` → `Error: …`, unreachable → `Error: MCP server <name> unreachable: <reason>` (never raises; URLs in the reason redacted to scheme+host+path; the error is recorded in the store and shown by `/mcp`); an OAuth server (`auth: "oauth"`) connects with the SDK's `OAuthClientProvider` over the user's stored sign-in, non-interactive (an expired access token is refreshed first; a refused one gets one forced refresh), else `Error: MCP server <name> needs you to sign in again (use /mcp).` with the record's error / the per-user flag "sign in needed" (no request at all when nothing is stored, nor when the stored record's URL is no longer the URL this provider loaded at turn start); every request of a non-admin user's connection (the OAuth endpoints and redirects included) must be https to a public address (`oauth_flow.request_guard`), on an admin's shared server too except at its configured origin; the admin's own connections are not guarded |
| WebTools (registered, blocking) | `browse_webpage` (url, wait_ms 0..10000 default 1500, screenshot, max_chars 500..100000 default 20000): page title, final URL, readable text (trafilatura on the rendered HTML), first 20 links, optional screenshot `downloads/screenshot_<hex>.png` (view with `view_image`; a failed screenshot keeps the page read and says "Screenshot: not saved") | sandbox: ONE container with network on, `mem_limit` 1g, timeout 75 s (script deadline 60 s) via `ContainerManager.run_script("browse.py", …)`; the bot process only formats the JSON |
| SandboxTools (registered, awaited; only while sessions are on) | `sandbox_status`: when the sandbox and the browser close if unused (user's local time), memory, open tabs, stored logins size, saved packages, whether the user was warned; its `prompt_notes` carry the sandbox facts (30 min idle, the warning, what persists, fresh bash per command, `nohup` for background jobs, filtered internet, apt/pip persistence) | bot process (`manager.status` in to_thread); inherited by subagents |
| BrowserTools (registered, awaited; only with `browser.enabled`) | the browser toolset members (`navigate`, `screenshot`, `zoom`, `left_click`, `right_click`, `middle_click`, `double_click`, `triple_click`, `hover`, `left_click_drag`, `left_mouse_down`, `left_mouse_up`, `mouse_move`, `scroll`, `scroll_to`, `type`, `key`, `hold_key`, `wait`, `read_page`, `find`, `get_page_text`, `form_input`, `new_tab`, `list_tabs`, `switch_tab`, `close_tab`; `javascript_exec`, `file_upload`, `read_console`, `read_network` disabled); `browser_confirm(summary)` (sends the current screenshot with ✅ Allow / ❌ Cancel through `bot/questions.ask`, 300 s; Allow = one sensitive action within 5 minutes, this turn only; only a tap grants it: a typed answer, `Answer.tapped` False, returns the text with "That is not a permission" and clears any allowance); `browser_show_screenshot(caption, region?)` (crop scaled up to 3x, e.g. a captcha for the user) | members: the user's browser container (`SessionManager.browser_call` in to_thread, 75 s); the two tools: bot process. Never given to subagents |
| DocumentTools (registered, blocking) | `create_document` (format md\|html\|pdf\|docx\|xlsx, title, content Markdown, filename?) → `documents/<sanitized name>.<fmt>` (existing names get `-2`, `-3`); the tool sends the file itself and returns "Document saved to documents/… (<size> KB) and sent to the user." | md/html: bot process (python-markdown: tables, fenced_code, sane_lists, toc + embedded CSS); pdf/docx/xlsx: sandbox via `run_script("render_document.py", …)`, network off, 1g for pdf, timeout 90 s (script deadline 75 s) |
| TaskTools (registered, awaited; only with a background host, never inherited by subagents) | `wait_for_tasks(timeout_seconds 5..600, default 120)`: the unread results of finished tasks at once (as blocks, marked read), "No background tasks are running." at once when none runs and the inbox holds nothing, else `inbox.wait` until something can be drained: a finished task → its result (or "A background task ended with nothing to read …" for a cancel or a result read already), "A new user message arrived; it follows." or the timeout (names the tasks still running); without an inbox it polls the host every second; `get_task_result(task_id)` (a finished task's result as tool-result content, marked read: `host.read`; a running one: "Task t3 (tool) is still running (…). A system message will say when it ends …"; unknown: `Error: no background task …`; pruned or cancelled: `Error: the result of task t3 is no longer available.`); `list_tasks` (id · tool · status running\|done\|failed\|cancelled · duration · "result not read yet" · start · args); `cancel_task(task_id)` (`t3`, `T3` or `3`; "Task t3 was cancelled." or `Error: no running background task …`). Static `<live_turn>` `prompt_notes`: how injected `<user_message sent_during_turn="true" time="…">` blocks look (also a typed answer to `ask_user_choice`), never ignore a message, answer quick questions with `send_user_telegram_message`, do not restart a backgrounded call, the system message when a task ends carries no output (read it with `get_task_result`), say a result will follow when answering while a task runs | bot process; the tasks themselves are the host's (`bot/background.py`) |

Unreachable: all 11 `SystemTools` tools (`install_package`, `run_shell_script` duplicates plus
`check_system_info`, `manage_service`, `monitor_process`, `network_diagnostics`,
`list_installed_packages`, `remove_package`, `install_python_package`,
`list_installed_python_packages`, `remove_python_package`). The persisted package lists
(`installed_packages.txt`, `installed_python_packages.txt`) are written by `install_package` (manager apt or
pip) and replayed once at every session start; a file the model writes with `create_file` is replayed too.

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
- **Persistent sessions** (`secure_container/sessions.py`, `SANDBOX_SESSIONS=true`, the default): every user
  has at most one shell container `sandbox-<uid>-<hex>` (`sleep infinity`, `init`, bind-mount
  `$WORKSPACE_ROOT/data/<uid>` → `/home/runner/workspace` rw, user `runner`, `SANDBOX_SESSION_MEM_MB` 768 MB with
  swap equal to it, 1 CPU, 512 pids, `oom_score_adj` 800, labels `generall.sandbox` / `generall.user`; no
  `no-new-privileges`, because `sudo apt-get` needs setuid) and at most one browser container
  `browser-<uid>-<hex>` (below). `run_command`, `execute_python`, `run_shell_script`, `install_package`,
  `create_directory` and `delete_file` all go through `ContainerManager.run_command` →
  `SessionManager.exec(uid, command, timeout)`: `exec_run` as `runner` in the workspace of
  `bash -c <capture wrapper> sandbox-exec <t> <cmd>`, which runs `timeout -k 5 <t> bash -lc <cmd>` with
  output to a temp file and stdin from /dev/null, then prints its first 1 MB. So shell syntax always works,
  a background job (`nohup … &`) does not hold the call open, and a huge output never reaches the bot.
  Each command is a fresh bash: files, installed packages and running background processes persist,
  `cd` and exported variables do not. A session is started on demand (first command), replays the saved
  packages once (900 s cap), and is stopped after `SANDBOX_IDLE_SECONDS` (1800) without use by the
  `sandbox-reaper`; `SANDBOX_WARN_BEFORE_SECONDS` (300) before that the user gets a warning with
  "👌 OK" / "⏳ Keep open 30 min" (`manager.extend`). A container found gone before the command started is
  replaced and the command run there once; one that goes away while it runs (/sandbox Close or Forget, the
  reaper, a shutdown, a crash) ends it with "Error: the sandbox was closed while the command was running; it did
  not finish." and it is never run again (`_exec_raw` makes docker-py's `exec_create` / `exec_start` /
  `exec_inspect` calls one by one: a 404/409 before the stream means nothing ran). A session whose start
  completes after `shutdown_all` (or after `required` lost the filter) is stopped at once. Timeouts are clamped
  to 3600 s.
- **Admission** (`SessionManager`): a start needs `MemAvailable` − `SANDBOX_MEMORY_RESERVE_MB` ≥ the session's
  limit, a free slot under the count cap (`SANDBOX_MAX_SESSIONS` / `BROWSER_MAX_SESSIONS`, 0 = automatic
  from `MemTotal`: 8 GB → 8 shells and 4 browsers, at most 16 / 8), and room in ONE budget shared by both kinds:
  the memory limits of every running and starting session plus its own ≤ `MemTotal` − the reserve (each count
  cap alone assumes the whole budget, and `MemAvailable` hardly moves when an idle container starts). Refused →
  one least-recently-used session of
  the same kind idle ≥ 120 s is stopped and the start retried once, else `CapacityError` with a readable
  reason (`SessionStartError`, a subclass, for Docker or filesystem failures). Busy sessions (a call in
  flight) are never evicted or reaped.
- **One-off containers** remain for `run_script` (always) and as the fallback of `run_command` /
  `install_package` on `CapacityError` or with `SANDBOX_SESSIONS=false` (`run_one_shot`, `bash -lc`, a note
  appended to the result): `secure-container-<uid>-<8 hex>`, the same mount, `mem_limit` =
  `DEFAULT_MEM_LIMIT` ("512m") unless the caller passes one (the page reader and the PDF renderer use "1g"),
  one CPU, `network_mode=none` unless `network_enabled`, removed in `finally`, behind one of
  `MAX_SANDBOX_CONTAINERS` slots (blocking, in the calling worker thread). With network they join the
  filtered `generall-sandbox` network whenever the SessionManager exists; with `SANDBOX_EGRESS_FILTER=required`
  they get no network while the filter is not `on` (or there is no manager), so the fallback of a refused
  session never runs unfiltered. The fallback reinstalls the saved packages first (`run_script` never) exactly
  like a session start: `sessions.persisted_packages` (validated names, `read_regular`: no link, no FIFO) through
  `sessions.replay_script` in a subshell of `temp_setup_<hex>.sh`, which forces network on (still none in the
  case above).
- **Network**: sessions run on the `SANDBOX_NETWORK` bridge (`SANDBOX_SUBNET`, `enable_icc` false, no IPv6)
  with `SANDBOX_DNS` resolvers, so commands and the browser always have internet; the parameter
  `network_enabled` is ignored in the session. The egress filter (`apply_egress_filter`, at manager setup)
  inserts host iptables rules (idempotent: `-C` first) that drop traffic from the subnet to private,
  loopback, link-local, CGNAT and multicast ranges (`BLOCKED_RANGES`; cloud metadata included) in
  `DOCKER-USER` (or `FORWARD`) and to the host itself in `INPUT`. The backend (`iptables`, `-legacy`, `-nft`)
  is the one whose tables hold Docker's chains. Result `on` / `off` / `failed` in `server_status()`; with
  `required` a failed filter refuses every start. `reap_idle` re-applies the rules every
  `EGRESS_RECHECK_SECONDS` (300; `-C` first, `-I` only for a missing rule, logged as a warning) and updates the
  state; in `required` mode a failed re-check stops every session (an error in the log), and a later good one
  allows starts again.
- **Browser container** `browser-<uid>-<hex>` (`BROWSER_MEM_MB` 1536 MB, `no-new-privileges`, same network):
  mounts only `data/_sandbox/<uid>/browser` → `/run/browser` (`ctl/browser.sock`, `profile/`, its own
  `downloads/`), so the shell never sees the cookies and the browser never sees the workspace. No workspace
  path is ever a mount source: the sandbox controls the links there, and Docker resolves a bind-mount source on
  the host. Each `download_completed` an answer reports is moved by `browser_call` (`_collect_downloads` →
  `move_download`: source a regular file opened `O_NOFOLLOW`, target created `O_EXCL | O_NOFOLLOW` in the real
  workspace `downloads/`, a taken name gets `-2`, `-3`, at most `DOWNLOAD_MAX_BYTES` 512 MB) and its `path`
  rewritten; one that cannot be moved becomes `download_failed`. Leftovers are swept in at the next start. At
  each start `scripts/browser_server.py` is written in fresh, `ctl/`, `profile/` and `downloads/` are made real
  0o777 directories, a stale socket and Chromium's `Singleton*` locks are removed, and the manager waits up to
  40 s for `_ping`.
  The server runs with `--idle-seconds` = the idle timeout + 60 and exits by itself after that.
  `browser.remember` off = a throwaway profile. `close` / `forget_browser_profile` / the reaper ask it to
  `_shutdown` first so the cookies are written (session cookies too: Chromium keeps only those with an
  expiry, so the server writes the others to `profile/gall-session-cookies.json`, 0600, and adds them back at
  the next start, deleting the file; never for a throwaway profile).
- `ContainerManager.run_script(user_id, script_name, args=(), timeout=90, network_enabled=False, mem_limit=None)`
  copies ONE bundled script from `secure_container/scripts/` (`SCRIPTS_DIR`) into the workspace as
  `temp_tool_<hex>.py` (unique per call), runs `python -I <file> <shlex-quoted args>` in a one-off container
  (isolated mode: the script's directory is the user's workspace, so a workspace `json.py` or a
  `pip install --user` copy of a library would otherwise shadow its imports;
  `run_one_shot`, same slots and mount, no package replay: the scripts use only the image's libraries),
  deletes the copy in `finally` and returns the raw logs.
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
- Commands always run through `bash -lc`, in the session and in the one-off fallback alike, so pipes, `&&`,
  redirects and globs work. `run_script` alone is `shlex`-split by docker-py (`entrypoint.sh` does
  `exec "$@"`), which is why it quotes every argument.
- Python code is written to `data/<uid>/temp_code_<hex>.py`, shell scripts to `temp_script_<hex>.sh`,
  `run_script`'s copy to `temp_tool_<hex>.py` and a package-install prelude to `temp_setup_<hex>.sh`, each
  created with its mode (0o755) by `sessions.write_new_file`, never chmodded by path (the session could swap
  the file for a link in between); all are deleted in `finally` (unlink, which never follows). The name is
  unique per call (`ContainerManager._workspace_file`) because a batch of tool calls runs concurrently:
  fixed names were overwritten by whichever call wrote last, so every container ran that one command.
- Packages persist by re-installing: `install_package(package_name, manager="apt"|"pip", timeout=300)`
  validates the names (no options, URLs or paths reach apt/pip, nor pip archive names such as `x.whl` or
  `x.tar.gz`, which pip would install from the workspace), installs them in the session and, on
  success, saves them to `installed_packages.txt` (`sudo apt-get`) or `installed_python_packages.txt`
  (`pip install --user`), a new version replacing the old line. Both lists are replayed once per session
  start; a failed replay is reported once, at the end of the next command's result. `python3-*` apt packages
  also get a best-effort `pip install --user <name-without-prefix>`.
- Session output = the wrapper's capture (stdout+stderr), no banner: the output, `(no output)`,
  `Error (exit code N):\n<output>`, the timeout text with the `nohup` hint (exit 124 or 137 once the session's
  time is up; a 124 earlier, e.g. from an inner `timeout 10 curl`, is a plain `Error (exit code 124)`),
  "Error (exit code 137: the process was killed, probably for using more than the memory …)" (137 earlier) or
  "Error: the command's exit status is unknown (the sandbox may have been closed):\n<output>" when Docker
  reports no exit code (`install_package` then saves nothing). One-off output = `container.logs()`, prefixed by the entrypoint banner (~10 lines),
  so the script tools parse the last JSON line; a one-off timeout → `Error executing command: <exc>` and the
  logs are lost with the container.
- Security boundary: the sandbox covers `run_command`, `execute_python`, `run_shell_script`,
  `install_package`, `delete_file`, `create_directory`, `browse_webpage`, the pdf/docx/xlsx renders of
  `create_document` and the whole browser toolset (its own container). `list_files`, `read_file`,
  `create_file`, downloads, zip, S3, `memory_search`, the vision/profile tools, md/html rendering, MCP calls,
  `browser_confirm` / `browser_show_screenshot` / `sandbox_status` and every API-backed tool run in the bot
  container, which is privileged with the Docker socket mounted, so every bot-side file operation inside
  `data/<uid>` or `data/_sandbox/<uid>` (both writable by a container) goes through the link-safe helpers of
  `sessions.py` (`real_dir`, `read_regular`, `write_new_file`, `move_download`; `_dir_size` walks the browser
  profile with `os.fwalk` from an `O_NOFOLLOW` descriptor; `_connect_unix` opens `ctl/` `O_NOFOLLOW` and the
  socket `O_PATH | O_NOFOLLOW`, requires `S_ISSOCK` and connects through `/proc/self/fd`) or `agents.paths`
  (`ensure_real_dir`, `regular_files`, `resolve_under`): a chmod, copy or read that followed a planted link
  would act on its target as root. The shell session keeps `sudo` for apt;
  the browser container has `no-new-privileges`; neither can reach private addresses while the egress filter
  is on. The users' MCP records (servers, static Bearer headers, cached tools, OAuth sign-ins) are bot-only state
  in `data/_mcp/<uid>.json`, like `data/_sandbox/<uid>/` outside every workspace: never mounted into a container,
  unreachable by `resolve_under` (so not by `read_file` / `create_file` either). They used to be
  `data/<uid>/mcp.json`, where the sandbox (data/<uid> is 777, and its shell has sudo for apt) and a
  prompt-injected agent could rewrite them, e.g. point a signed-in record at their own server for one turn, swap
  the file blind during a sign-in or write back an old copy after a sign-out; `mcp_store._migrate` moves that file
  on the user's first load (see the data layout) and a `mcp.json` in the workspace is never read afterwards.
  Defence in depth stays: the file is read `O_NOFOLLOW` (≤ `MAX_FILE_BYTES`), written through `mkstemp` +
  `os.replace` (0600, directory 0700), and the OAuth tokens and client registrations are Fernet-encrypted with a
  key outside the workspace (`MCP_TOKEN_KEY` or `data/.mcp_token_key`); each encrypted value carries the user id,
  server name, URL and admin flag and is refused for any other record, and a user server's value is used or
  written only while the stored record is an OAuth server whose URL is also the URL of the connection asking
  (`mcp_store._locate`), so a provider that loaded a record before it changed gets no token and a sign-in that
  finishes after its record changed stores nothing. `data/_mcp/`, `data/tls/` and `data/.mcp_token_key` are never
  mounted into a container.

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
                                    [, auth: "oauth" (each user signs in with their own account)]
  mcp_servers.cache.json            (<admin file stem>.cache.json) their discovered tools: {"servers": [{name, tools, updated, error
                                    [, auth: "oauth" when a discovery asked for a sign-in]}]}, written at startup (mcp-admin-refresh,
                                    with the admin's sign-ins), by the admin's /mcp refresh and by the admin's sign-in to such a server
  .mcp_token_key                    the generated Fernet key of stored MCP sign-ins when MCP_TOKEN_KEY is unset (0600, O_EXCL, kept;
                                    with MCP_TOKEN_KEY set it only decrypts what it sealed before)
  _mcp/<chat_id>.json               the user's MCP servers (bot-only: the directory is 0700, never mounted, outside the workspace):
                                    {"servers": [{name, url, transport, headers: {Authorization: "Bearer …"},
                                    enabled, tools: [{name, description, input_schema}], updated (iso), error[, auth: "oauth",
                                    oauth: {tokens: "<fernet>", client: "<fernet>", account: null, updated[, needs_signin: true]}]}],
                                    "oauth_admin": {"<admin server>": {same entry}}} (the user's own sign-ins to admin servers);
                                    tokens = {tokens (the SDK's OAuthToken), expires_at, metadata (discovered AS + resource metadata:
                                    the refresh endpoint)}, client = the dynamic client registration; each value bound to user +
                                    server + URL + admin flag (see the security boundary); mkstemp + os.replace (0600), read
                                    O_NOFOLLOW, at most MAX_FILE_BYTES (4 MB). Replaces <chat_id>/mcp.json: the user's first load
                                    with this file missing moves the old one here (read O_NOFOLLOW | O_NONBLOCK, a regular file
                                    ≤ 4 MB, normalized like any record, then unlinked; a link, FIFO, oversized or broken one is not
                                    read, just unlinked, with a WARNING; a moved one is logged once at INFO) or, when there is none,
                                    writes {"servers": []}: from then on the file exists and a mcp.json in the workspace is ignored
                                    (neither read nor deleted)
  tls/                              the sign-in endpoint's certificates (only with TLS: port 443 or OAUTH_TLS=on): <host>.pem / <host>.key
                                    (the admin's, e.g. a Cloudflare Origin Certificate; they win) or the generated <host>.selfsigned.pem /
                                    <host>.selfsigned.key (EC P-256, CN + SAN = the host of PUBLIC_BASE_URL, 10 years, key 0600, O_EXCL,
                                    reused; the admin's files are never touched)
  _sandbox/<chat_id>/browser/       the browser container's mount (/run/browser), outside the user's workspace: ctl/browser.sock
                                    (the control socket), profile/ (Chromium's persistent profile: cookies and logins, plus
                                    gall-session-cookies.json between a shutdown and the next start; deleted by
                                    🧹 Forget browser logins), downloads/ (in progress; finished files move to the workspace),
                                    browser_server.py (written in at each start)
  <chat_id>/
    settings.json                   per-user settings (see table); merged over defaults on load, unknown keys preserved
    profile.md                      agent-managed profile: "## About", "## Preferences", "## Notes" (entries "- ...", Notes dated
                                    "[YYYY-MM-DD] "), other "## Title" sections after them; ≤ 8 KB (oldest Notes rotated out); deleted
                                    when empty; edited only through bot/profile.ProfileStore (user_profile_* tools, /profile); read only
                                    as a regular file (read_regular_text: O_NOFOLLOW|O_NONBLOCK, so a symlink or
                                    FIFO reads as empty), written via a mkstemp tmp file + os.replace; something the user wants
                                    remembered for one forum topic goes into a section "## Topic <thread id>" (ProfileTools
                                    prompt_notes), shown in every topic like the rest
    topics.json                     {"topics": {"<thread id>": {name, implicit (Telegram's is_name_implicit: not chosen by the
                                    user), updated (iso UTC)}}}: the forum topic names (bot/topics.py); the bot learns them only
                                    from service messages, so a topic renamed while the bot was down keeps its old name until
                                    the next rename; read link-safe, cleaned again on read (the sandbox can write it)
    mcp.json                        legacy: the user's MCP servers before they moved to data/_mcp/<chat_id>.json (moved away on the
                                    first load; a file of that name written here later is never read)
    conversations/conversation_<YYYYmmdd_HHMMSS>_<topic>.json
                                    one file per turn: {timestamp, thread_id, topic, summary, question, response, full_history}
                                    (full_history with media blocks replaced by their markers: content.strip_media; question =
                                    _memory_question, i.e. with "[Sent while I was working]" + the messages read mid-turn)
    short_term_memory/[topic_<thread>_]dialog_history.json
                                    last dialog_history.size Q/A PAIRS (size*2 entries) as plain strings
    short_term_memory/[topic_<thread>_]short_term_memory.json
                                    block-structured messages of (mostly) the last turn ("reasoning context")
    transcripts/[topic_<thread>_]transcript.json
                                    transcript mode: {version, user_id, thread_id, created, updated, model, seeded_from,
                                    fingerprint, messages:[exact API messages, as sent]}; append-only, pruned only above
                                    max_context_tokens; atomic write; in a live turn a tool-result user message may carry
                                    injected blocks behind its tool_result blocks (<user_message …> and its images/documents)
                                    and be followed by {"role": "system", "content": "<the bot's notice>"} (a background task
                                    ended); transcripts written before the system notices may hold <task_notification …> texts
    embeddings/faiss_index.bin, metadata.json
                                    IndexFlatIP(1536) over L2-normalized vectors (cosine) of "Topic/Question/Answer/Summary" texts with
                                    text-embedding-3-small; metadata.json = {version: 2, model, dimension, updated, items: [{question,
                                    answer (≤ 2000 chars each), timestamp, topic, text_hash}]}; a bare list (legacy) is treated as stale;
                                    both files written tmp + os.replace
    turns/<turn id>.json, <turn id>.reasoning.txt
                                    one TurnRecord per turn (prompt = text view, prompt_blocks = media refs when the prompt had
                                    media, displayed answer, speak, thread_id, limit_reached, follow_ups, has_reasoning, error =
                                    the friendly reason of a FAILED turn: response "", no reasoning file, Retry only; records
                                    without the key read as null; quiet_marker = the scheduled task's marker or null; prompt /
                                    prompt_blocks include "[Sent while I was working]" + what the agent read mid-turn,
                                    with_drained) + its reasoning transcript; what the turn_ buttons load;
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
                                    (browse_webpage), files the interactive browser downloaded (moved here by the bot)
    installed_packages.txt, installed_python_packages.txt   the persisted apt / pip names (install_package), one per line,
                                    replayed at every session start
    temp_code_<hex>.py, temp_script_<hex>.sh,
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
  Conversation summaries and the FAISS index are shared across threads. The `<context>` block names the topic
  (`ChainOfThoughtAgent._topic_line`, transcript and legacy mode); the profile is shared, with "Topic <id>" sections
  for what belongs to one topic.
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
  cleared wholesale first, so pictures inside them count under `cleared`; a browser member's result keeps its
  `browser_state` blocks (`_cleared_result_content`: a result of state blocks only, i.e. every tab member's,
  stays as it is because the API wants exactly one; others become `[marker, state]`) and its `is_error`.
  `clear_old_media` never touches the last N turns.
  Screenshots make a second trigger necessary: `prune` also runs when the transcript holds more than
  `transcript.MAX_TRANSCRIPT_IMAGES` (40) images (`count_images`, tool results included). Any compaction (either
  trigger) clears old tool results and media as above, then the oldest pictures of the more recent turns until
  at most `IMAGE_PRUNE_TARGET` (20) remain (`clear_images_down_to`; the newest turn keeps all of its own), so the
  next browsing turns do not cross the cap again right away (clearing to just under it cost a cache miss and the
  thinking blocks on every turn); cleared browser images become `SCREENSHOT_CLEARED_MARKER` ("take a new
  screenshot"). A prune triggered only by images summarizes nothing; thinking is stripped as for any prune. The
  cap is checked at turn end only.
- Reloading `short_term_memory.json` keeps only messages whose first block is `text`. With thinking
  on, intermediate tool-calling assistant turns start with a `thinking` block and are dropped; the
  final answer (re-appended as text-only) is kept. Tool-result user messages are never reloaded.
- Scheduled tasks are ordinary turns for the memory: the framing prompt and the answer, `<no_update/>`
  included, are appended to the transcript, written to dialog_history.json, summarized by Haiku and embedded
  like any turn; the agent core knows nothing about the marker (`run_turn` checks it after `generate_response`
  returned, and only skips the delivery). "Background task finished" notification turns are ordinary turns too.
- Live turns in memory: a message the agent read mid-turn sits in the transcript where it was read, in the
  tool-result user message behind the `tool_result` blocks (`<user_message sent_during_turn="true" time="…">` text
  plus its materialized image/document blocks), stored exactly as sent; one read before the first model call sits
  behind the question in the turn's own user message (a turn start, as always). A background task's notice is the
  `{"role": "system"}` message right after that user message (facts only; its output is in the `tool_result` of the
  `get_task_result` / `wait_for_tasks` call that read it). `AgentAnthropic.generate_response` keeps a system message
  where it is (only a LEADING one is the system prompt); `text_only_view`, `memory_search`, the summaries and the
  legacy reasoning-context reload leave system messages out (without their tool results they would follow an
  assistant message, where the API refuses one); a prune never starts the kept part with one (it cuts at turn
  starts); `_request_messages(…, model)` sends a stored one as a `<bot_notice>` user block to a model without
  system messages. Messages that arrived together (the message window, leftovers) are one user turn of
  `<message n="i">` sections (`inbox.combine`). Such a message is NOT a turn start: tell it by its FIRST block (`transcript.answers_tool_call`), which
  `is_user_turn`, `clear_old_tool_results` (clears its `tool_result` blocks and any old-format `<task_notification>`
  text, down to the opening tag + `CLEARED_MARKER`, here and in a notification turn's user message; an injected
  `<user_message>` stays, old injected media goes through `clear_old_media`) and the prune's turn split use, so a prune never separates a
  `tool_use` from its result. `is_tool_result_message` (tool_result blocks only) still decides what `text_only_view`
  and `memory_search` skip, so the injected text is searchable. The memory files (conversation file, its Haiku
  topic and summary, the embedding, dialog_history.json; transcript and legacy mode) get
  `ChainOfThoughtAgent._memory_question(question)` (= `agents.main.with_messages_read`, which the judge and the
  critique get too): the question + "\n\n[Sent while I was working]\n" + the drained texts
  (`inbox.drained_user_texts`) joined by blank lines; with nothing read, the bare question.
- Prompt caching (transcript mode): the static system prompt is one `text` block with an explicit
  `cache_control` breakpoint (`SYSTEM_CACHE_TTL`), covering the tool schemas before it; every request also
  passes the top-level `cache_control={"type": "ephemeral"}` so the API places an automatic breakpoint on
  the conversation tail. Everything volatile (time, memory hits) goes into the `<context>` block, the first
  block of each user turn; it is stored with the turn (see "Preserved thinking" in Known pitfalls), and only
  memory lines the transcript has not shown yet are included, so the previous turn is a cache read.
  `ToolTrace.add_usage(usage, model)` accumulates `usage` per turn and per model; the status line shows the
  cached share and an estimated cost (`models.MODEL_PRICES`, cache reads ×0.1, writes ×1.25), and
  `agent_runner.record_usage` stores one `usage_events` row per model (plus, through `record_late_usage`, one row
  per later addition from a call still running in the background); `/stats` (admin) shows tokens, cost
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
| `browser` | enabled (true), stream (true), remember (true) | enabled: `BrowserTools` is registered (the toolset, `browser_confirm`, `browser_show_screenshot`); stream: the live view photo during browsing (never in a turn with a quiet marker, i.e. scheduled tasks); remember: the browser keeps its profile (cookies, logins) in `data/_sandbox/<uid>/browser/profile`, off = a throwaway profile. /settings section "🌐 Browser" (token `browser`, actions toggle/stream/remember; legacy `show_browser_menu`); none advanced, not in presets. Toggling `enabled` changes the tool list, so stored thinking is stripped once |
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
first block may be thinking), `supports_system_messages(model)` whether the model takes a mid-conversation
`{"role": "system"}` message (`SYSTEM_MESSAGE_MODEL_PREFIXES`: Opus 5.x, Opus 4.8, Fable 5, Mythos 5, Sonnet 5.5,
Haiku 5.5; not Sonnet 5; a Bedrock `…anthropic.` prefix is ignored; a change of model there needs a new prefix),
and `openai_reasoning_options(model)` returns `{"reasoning_effort": OPENAI_REASONING_EFFORT}` for the two OpenAI
reasoning models and `{}` for anything else. No other file holds a model literal.

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
| MCP tool servers (the user's and the admin's) | `mcp>=2.2` client: streamable HTTP (`streamable_http_client` with an httpx2 client from `create_mcp_http_client` carrying the stored headers, the OAuth provider as `auth` and the request guard as an event hook) or legacy SSE (`sse_client(auth=…, httpx_client_factory=…)`), `ClientSession(read, write, read_timeout_seconds=…)`; the 1.x camelCase/`streamablehttp_client` branches are untested fallbacks | one connection per call, `initialize` → `call_tool`; the `mcp` import (~0.8 s) is lazy inside `open_transport` / `client_session` (tests replace them, or put an `httpx2.MockTransport` in `mcp_tools.HTTP_TRANSPORT`) | `agents/mcp_tools.py`, `bot/mcp_store.py` |
| MCP OAuth sign-in | `mcp.client.auth.OAuthClientProvider` (RFC 9728 protected resource metadata, RFC 8414 AS metadata, RFC 7591 dynamic registration as a public client "GALL.AI" with `token_endpoint_auth_method: none`, PKCE S256, its own `state`, RFC 8707 `resource`, refresh); our `TokenStorage` (`oauth_flow.OAuthStorage`); the SDK's default `validate_resource_url`; no `client_metadata_url` (no CIMD document is hosted) | redirect URI `PUBLIC_BASE_URL/oauth/callback` (`application_type: web`) or `http://localhost:8765/oauth/callback` (`native`, the copy-paste fallback) | `bot/oauth_flow.py`, `bot/oauth_web.py` |
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
| `PUBLIC_BASE_URL` (unset) | bot/oauth_web (read at startup) | e.g. `https://gall.zor.gg`: starts the sign-in callback endpoint; redirect URI = this + `/oauth/callback`; unset (or the endpoint failed) = the copy-paste fallback (`http://localhost:8765/oauth/callback`) |
| `OAUTH_LISTEN` (`127.0.0.1:8090`) | bot/oauth_web | `host:port` (`[::]:443` for IPv6) the endpoint binds; the default is loopback plain HTTP for a Cloudflare Tunnel or a reverse proxy |
| `OAUTH_TLS` (auto) | bot/oauth_web | auto = TLS when the port is 443; `on` forces TLS on any port, `off` plain HTTP even on 443 |
| `OAUTH_TLS_CERT`, `OAUTH_TLS_KEY` (`data/tls/<host>.pem` / `.key`) | bot/oauth_web | the admin's certificate + key (PEM; e.g. a Cloudflare Origin Certificate for "Full (strict)"); used when both exist, else a generated self-signed pair; TLS ≥ 1.2; the startup log names the certificate and its expiry, never the key |
| `MCP_TOKEN_KEY` (unset) | bot/mcp_store (first use) | Fernet key (urlsafe base64 of 32 bytes) encrypting stored MCP OAuth tokens and client registrations; a PROD secret, generated once; unset → `data/.mcp_token_key` (generated, WARNING) |
| `MCP_TOKEN_KEY_OLD` (unset) | bot/mcp_store | comma-separated decrypt-only keys (rotation: values move to `MCP_TOKEN_KEY` on their next write); an invalid key is logged and skipped |
| `COMPOSE_PROFILES`, `CLOUDFLARE_TUNNEL_TOKEN` | docker compose (from `.env`) | `COMPOSE_PROFILES=tunnel` starts the `cloudflared` service with `TUNNEL_TOKEN=$CLOUDFLARE_TUNNEL_TOKEN` (the bot ignores both) |
| `MAX_CONCURRENT_TURNS` (8), `TURN_TIMEOUT_SECONDS` (1800), `DROP_PENDING_UPDATES` (true) | bot/config | queue caps, see "Queues" |
| `MESSAGE_WINDOW_SECONDS` (1.5) | bot/config (read once at import; blank or not a number = 1.5, negative = 0) | the message window: a message that does not join a running turn waits this long for the next one of its chat and topic, each new one extends it (at most `messages.MESSAGE_WINDOW_MAX` 30 s), and they become one turn; 0 = off (every message queued at once, as before); `tests/conftest.py` sets 0 |
| `TOOL_BACKGROUND_AFTER_SECONDS` (30) | agents/main (read once at import; blank or not a number = 30 with a printed warning for the latter, negative = 0) | live turns: a tool call still running after this many seconds moves to the background (`_watch_call`); 0 = no time trigger, a call still moves when the user writes meanwhile (`BACKGROUND_ON_MESSAGE_AFTER` 3 s, not configurable) |
| `MAX_SANDBOX_CONTAINERS` (4) | secure_container/container_manager | concurrent ONE-OFF sandbox containers (`run_script`, the fallback); sessions have their own caps below |
| `SANDBOX_SESSIONS` (true) | secure_container/sessions | persistent per-user shell sessions; false = a throwaway container per command (and no `sandbox_status`) |
| `SANDBOX_IDLE_SECONDS` (1800, min 60), `SANDBOX_WARN_BEFORE_SECONDS` (300, 0 = no warning) | sessions, sandbox_ui | idle time before a session closes; the warning with Keep open comes this long before |
| `SANDBOX_SESSION_MEM_MB` (768), `BROWSER_MEM_MB` (1536), `SANDBOX_MEMORY_RESERVE_MB` (1536) | sessions | memory limit per shell / browser container; host memory kept free (admission reads `MemAvailable`, and the limits of all sessions together must fit in `MemTotal` minus the reserve) |
| `SANDBOX_MAX_SESSIONS` (0), `BROWSER_MAX_SESSIONS` (0) | sessions | count caps; 0 = automatic from `MemTotal` (at most 16 / 8; 4 / 2 when `/proc/meminfo` is unreadable) |
| `SANDBOX_NETWORK` (generall-sandbox), `SANDBOX_SUBNET` (172.30.240.0/24), `SANDBOX_DNS` (1.1.1.1,8.8.8.8; empty = Docker's) | sessions | the sessions' bridge network, its subnet (must not overlap other networks), their resolvers |
| `SANDBOX_EGRESS_FILTER` (best_effort) | sessions | iptables rules keeping the sandbox subnet off private ranges and the host: best_effort (warn and continue when they cannot be written) \| required (no session starts then, one-off containers get no network, a failed re-check every 300 s stops every session) \| off |
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
  /profile, /mcp, /sandbox, /cancel, `queue_cancel`, Stop, shared locations (`F.location` → time zone), `ask_`
  answers, Listen/Reasoning, `usage_`/`profile_`/`mcp_`/`sandbox_` buttons: `AuthMiddleware()`, also on the
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
  confirm with the local time (`timezone_set_text`), or `TIMEZONE_UNKNOWN`), `/sandbox` (`sandbox_ui`: the
  sandbox and browser state, Keep open / Close / Forget logins), `/cancel`, `/stats` (admin), `/listusers` (admin).
- Command menu: `commands.COMMANDS` (start, help, settings, usage, reminders, profile, timezone, voice, mcp,
  sandbox, cancel; descriptions ≤ 256 chars) is the one list: `bot_commands()` feeds `set_my_commands` at startup and
  `help_text` renders the command section of /help from it. /invite and the admin-only commands are not
  listed. Add a user command there and both the menu and /help show it.
- Callback prefixes: `voice_`, `settings_` (incl. `settings_preset_<name>`, `settings_mode[_set_<v>]`),
  `setp_` (settings page, incl. `setp_preset_<name>`, `setp_ui_advanced`), `reminder_`/`reminders_`, `noop`,
  `queue_cancel`, `queue_stop_<job token>` (Stop in the status), `ask_<qid>_<i>` (ask_user_choice),
  `turn_<retry|cont|listen|reason>_<turn id>` / `turn_fu_<turn id>_<i>` (turn buttons; a `turn_` catch-all
  answers "expired"), `stats_`, `usage_refresh` (`usage_ui.usage_button` matches `usage_`), `profile_`
  (`profile_refresh`, `profile_clear`, `profile_clear_yes`, `profile_cancel` → `profile_ui.profile_button`),
  `mcp_<action>[_<name>]` (`mcp_toggle_`, `mcp_refresh_`, `mcp_remove_` → confirm `mcp_rm_` / `mcp_back`,
  `mcp_refreshall`, `mcp_signin_` (OAuth: start the background sign-in, a new "Connecting…" message becomes the
  🔐 URL button) / `mcp_signout_` (delete the tokens, keep the server) → `mcp_ui.mcp_button`; actions carry no
  underscore, names are `[a-z0-9_-]`). The sign-in button itself is an `InlineKeyboardButton(url=<authorization
  URL>)`: no callback; its message is edited into the result (✅ / ⌛ / ❌) without a keyboard. `tz_set_<index>`
  (the /timezone picker → `commands.timezone_button`, an index into `COMMON_ZONES`), `sandbox_<close|forget|
  forget_yes|back|refresh|keep|ok>` (→ `sandbox_ui.sandbox_button`; `sandbox_keep` / `sandbox_ok` act on the idle
  warning when the message is the remembered warning or starts with "⏳ Your ", else on the /sandbox page).
  The reminders family
  (`reminders_ui.reminder_button`, parsed `split("_", 2)` → action + payload, a trailing `_<value>` taken off
  the payload with `rpartition`): `reminders_page_<n>`, `reminders_summary`, `reminders_new`, `reminder_info_<id>`,
  `reminder_delete_<id>`, `reminder_toggle_<id>`, `reminder_delay_<id>_<minutes>` (+1h = 60, +1d = 1440),
  `reminder_edit_<id>`, `reminder_editcancel_<id>`, `reminder_snooze_<id>_<10|60|tomorrow>` (on fired
  reminders, built by `jobs.snooze_keyboard`); group titles and spacers are `noop`. `rem_` is NOT free. A new
  family needs a prefix that is not a prefix of (and does not start with) any other, and a
  `@router.callback_query(F.data.startswith(...))` handler on the router with the right middleware;
  `tests/test_cross_features.py::ROUTING` checks that every family reaches its own handler in router order
  (`MESSAGE_ROUTING` does the same for message handlers: plain text, text during a pending reminder edit,
  /reminders and /cancel during one, a location, a photo, a pasted sign-in URL with and without a pending
  sign-in, a forum topic created / renamed)
  (rows exist for usage_, profile_, mcp_ incl. mcp_signin_ / mcp_signout_, sandbox_, setp_preset_, setp_ui_advanced, settings_preset_,
  settings_mode_set_, tz_set_, reminder_snooze_, reminder_delay_, reminder_edit_, reminders_new). Read the chat from
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
  `messages.on_topic_service` (`F.forum_topic_created | F.forum_topic_edited`, `control_router`) stores the topic's
  name (`topic_store.set`; an edit without a name, i.e. only the icon, is ignored) and answers nothing; the service
  message never reaches the agent and uses no quota.
  `mcp_ui.pasted_callback` (`F.text` AND the async filter `_pasted_while_pending`: the text starts with
  `http://localhost:8765/oauth/callback` and the chat has a pending MCP sign-in) is the copy-paste fallback: it
  deletes the message (it holds the code), resolves the login by its `state` (only that user's) and answers
  "🔐 Got it…", or "⌛ … expired" for an unknown or used state; without a pending sign-in such a text goes to the
  agent. A filter that reads loop-owned state must be `async`: aiogram runs a plain function filter in a worker
  thread (`asyncio.to_thread`).
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
- Every content handler: track_message_received → `accept(message, bot, user_id, limit, label=, kind=,
  error_text=, prepare=partial(prepare_<kind>, …), run=partial(_run_<kind>, …))`, which joins the running turn
  (`inject_target` → `inject` → `inbox.mark_seen`) or puts it into the message window (`_gather`), whose close
  `submit`s `Job(…, thread_id=, from_user=True)` (with `MESSAGE_WINDOW` 0 at once, as before). The job
  function signature is `(bot, message, user_id, limit, ..., ctx)` via `functools.partial`; it creates its own
  `ChatSender`, its own status message, sets `ctx.set_progress(...)`, builds its prompt through `prepare_<kind>`
  (reply/forward preamble via `with_reference`, media as `content.image_ref`/`document_ref` blocks:
  `_prompt_with_reference(text, media_blocks, message, bot_id)`) and calls `run_turn`. Retry /
  Continue / follow-up buttons do the same: they post a short echo ("🔁 Retry", "💬 <suggestion>") and
  queue the turn on it, so the busy notice and the reply-to work as for a typed message; a follow-up (a plain
  string prompt) pressed while a turn of the record's topic is working joins that turn instead
  (`_start_turn(join_running=True)`: a ready `Prepared`, 👀 on the echo); Retry and Continue always queue. Retry
  re-runs `record.prompt_blocks or record.prompt` with `record.quiet_marker` (and `QUIET_NOTICE`), so a retried
  scheduled task with nothing new never shows `<no_update/>`.
- Live-turn reactions: 👀 (`inbox.SEEN_REACTION`) when a message joins the running turn (awaited by the handler),
  ✍ (`READ_REACTION`) once the agent drained it (a task, after the 👀 call finished); both through
  `bot.set_message_reaction`, best effort (`inbox.react` never raises: old clients, no rights).
- Typed answers: a plain text (not a command) in a chat and topic where an `ask_user_choice` question is on screen
  and waiting answers the newest such question (`on_text` checks `questions.waiting_in` and calls
  `answer_in_text` BEFORE `accept`): the question message ends as "💬 Answered in text" without buttons. The user's
  words never go inside the tool result (the model resists instructions there, so a reply quoted in it could be
  ignored): `messages._answer_item` first hands the text to the running turn of that topic as an ordinary inbox item
  (no 👀, ✍ once read; no suspension point before the question is settled), the answer is resolved with
  `delivered=True` and the tool reads `questions.TYPED_DELIVERED` (a pointer to the `<user_message>` that follows the
  tool results). With no turn to take it the tool reads `User answered in text instead of tapping a button: "<text>"`
  as before (cut at `MAX_TYPED` 4000 characters), and the text never queues as a message; a question that went away
  between the check and the answer leaves the delivered text an ordinary message of the turn (👀). Voice notes,
  photos and other kinds do not answer it; a pending reminder `✏️ Edit text` (ui router) still takes the text first.
  `browser_confirm` allows only on a tap (`isinstance(answer, questions.Answer) and answer.tapped`): a typed "yes"
  reaches the agent as a message (or as text) and allows nothing.
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
  task grows the transcript until `prune` summarizes it. A user message such a run read is in that stored turn
  AND in the requeued turn after it, so the transcript holds it twice. The `agent_response` of a periodic task is overwritten
  by every run; `quiet_runs` only ever grows.
- Cancellation cannot interrupt a tool call already inside a worker thread; the turn only ends when
  that call returns (sandbox containers run to their own timeout).
- `stats_ui.py` imports `stats` at module level, so importing `bot.handlers` creates `data/stats.db`
  in the cwd (the other modules import it lazily).
- Dead code: `JudgeResponse`, `tavily_client` in `agents/main.py`, `describe_document_openai` was removed;
  `ToolIntegrator.patch_search_tools` only logs. `bot.limits.check_user_limits` is blocking SQLite with a
  side effect (`ensure_user_limit` inserts the default row): outside the middleware call it in `to_thread`.
- `MCPTools` reads the user's `data/_mcp/<uid>.json` and the admin cache synchronously on the first `tools_schema`
  access, i.e. on the event loop at turn start (KB-sized files, fine; do not grow them). The very first load of a
  user after the move out of the workspace writes too (`mcp_store._migrate`: the moved file, or an empty one), once.
  `tools_schema` is computed once per instance (= per turn); tool discovery is cached, so a server whose tools
  changed shows the old tools until `/mcp refresh` (users) or a restart / admin `/mcp refresh` (admin servers).
- A user's MCP server must be https to a public host: `mcp_store.validate_url` refuses http, localhost / *.localhost
  and loopback/private/link-local/shared/reserved/unspecified/multicast (IPv4-mapped included) literals for
  non-admins, and `MCPTools._check_host` runs `mcp_store.check_public_host` (getaddrinfo via `_resolve`, in
  to_thread; an unresolvable name passes) before every discovery and call of a non-admin user's server, which also
  covers records that never went through `validate_url` (moved from the old workspace file). The admin (shared
  servers and the admin's own `/mcp add`) may use http and private hosts. Not pinned: httpx resolves again (a DNS
  rebinding window). `download_file` / `download_webpage` still reach any host. Tests stub `mcp_store._resolve`.
  Every request of a non-admin user's MCP connection also passes `oauth_flow.request_guard` (an httpx2 request event
  hook, which httpx2 runs for the OAuth flow's own requests too): https only, and every host other than the server's
  own (already checked) through `check_public_host` once per connection; a refused request raises `UnsafeEndpoint`
  and is never sent (fail closed). That covers the metadata, registration and token endpoints the SDK discovers,
  which a malicious server could point at the bot's host network. A non-admin user's connection to an admin's shared
  server is guarded too (`request_guard(trust_origin=True)`): only the configured origin (scheme, host, port) is
  exempt, since the admin may run it on http or a private address; every other origin, the same host on another port
  or scheme included, needs https and a public address, so a shared server's metadata cannot send a user's
  registration or token request to the bot's network. Consequence: a private shared server that redirects (http →
  https on the same host included) fails for non-admins; configure its final URL. Only the admin's own connections
  run unguarded. The authorization URL (opened by the user's browser, never fetched) must be https for non-admins,
  except at a shared server's configured origin.
- **MCP OAuth** (`bot/oauth_flow.py`): one `OAuthClientProvider` per connection, so whatever the SDK keeps in
  memory is lost after each call; `OAuthStorage` makes up for two gaps: the SDK restores no expiry from storage
  (`_initialize` only loads the token, and only a token it thinks expired is refreshed) and, before a 401 made it
  discover metadata, refreshes at the server origin's `/token`. So the stored tokens carry `expires_at` and the
  discovered metadata (restored into `provider.context`), and `get_tokens` hands an expired token (60 s margin)
  over with an empty `access_token`, which makes the SDK refresh first. A server that sends no `expires_in`
  answers 401 instead: `MCPTools._with_stored_sign_in` then retries once with a forced refresh. Refreshes are
  serialized per user + server + process (`_refresh_lock`, held from `get_tokens` to `set_tokens` or the end of the
  connection; re-entrant per storage; the interactive sign-in drops it in its redirect handler, before it waits
  for the user): a provider that rotates refresh tokens would otherwise refuse the second of two concurrent
  refreshes.
- A non-interactive OAuth connection (tool calls, `/mcp refresh`, the startup refresh, the add probe) raises
  `NeedsSignIn` from the SDK's redirect handler; it surfaces wrapped in anyio ExceptionGroups and httpx2, hence
  `mcp_tools.find_exception`. The SDK logs every flow failure with `logger.exception("OAuth flow error")`, whose
  text can hold a state (`State parameter mismatch: …`) or a server's response body: `install_log_filter` drops
  the record for our own expected ends (`NeedsSignIn`, `LoginError`) and cuts the others to the exception class.
  The interactive session waits for the browser inside the first `initialize`: its `read_timeout_seconds` is
  `LOGIN_TIMEOUT + DISCOVER_TIMEOUT`, and `wait_for_code` enforces `LOGIN_TIMEOUT` itself.
- The `/mcp add` probe registers a client (dynamic registration happens before the authorization step); the
  registration is kept (sealed into the new record) and reused by the sign-in, and by later sign-ins after a
  sign-out. A stored registration whose redirect URIs do not contain the current one (`PUBLIC_BASE_URL` set,
  changed or the endpoint down) is ignored by a sign-in (registered again) but still used for refreshes.
- Sign-in state: a user server is "sign in needed" while it has no decryptable tokens, its oauth entry says
  `needs_signin` (set by a failed refresh, cleared by a refresh or a successful call) or its error is "sign in
  needed"; an admin server's state is the user's own `oauth_admin` entry, never the shared cache (a user's failed
  refresh does not write the cache; `record_error` would). Every successful interactive sign-in clears the flag
  (`write_oauth(needs_signin=False)`), also one that needed no login because the stored token was accepted again.
  The tools of an OAuth admin server come from the admin cache, filled only by the admin's sign-in, the admin's
  `/mcp refresh` and the startup refresh with the admin's tokens (`refresh_admin_servers` uses `auth.admin_id`); a
  user who signs in first is told so.
- Pending sign-ins, their Futures and the sign-in tasks live in memory, bound to the event loop (cleared when it
  changes, `oauth_flow.reset()` in tests): a restart loses them, the link then shows the expired page. One
  pending sign-in per user + server: starting another cancels the running task ("replaced"); removing the server
  (`oauth_flow.remove_server`) or signing out of it (`sign_out`) cancels it too (`cancel_sign_in`: the message ends
  as `REMOVED_TEXT` / `SIGNED_OUT_TEXT`, the link as the expired page). Both also take the refresh lock around the
  store call, so a token refresh in flight finishes first and cannot write its tokens back after the sign-out;
  keep any new path that deletes tokens or records behind that lock. A sign-in whose writes the store refuses
  (`write_oauth` False, `update_user_tools(url=…)` None: the record is gone, no longer OAuth or has another URL,
  e.g. re-added with a token meanwhile) keeps going in memory but stores nothing and ends as `REMOVED_TEXT`
  ("❌ The server was removed while you were signing in."), never "✅ connected".
- The sign-in callback is not tied to the Telegram user: the `state` in the link identifies the pending sign-in,
  and whoever completes the provider's login with that link connects THEIR account to the bot user who started it
  (their tokens end up in that user's record and the agent acts with them). The link is single-use and expires
  after `LOGIN_TIMEOUT` (10 minutes); never forward a sign-in link, and treat a forwarded one as a phishing attempt
  (a user who signs in through someone else's link hands them access to their account).
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
- `ask_user_choice` holds its turn while it waits (up to its timeout, 15 min at most; it is in `NEVER_BACKGROUND`):
  a plain text in that chat and topic answers it; any other message of that topic joins the turn (👀) but is read
  only after the whole batch, the question included, is done (other calls of the batch that have run 3 s move to
  the background once it is ready); messages of other topics queue behind the turn with the busy notice. Stop or /cancel ends
  the wait ("🛑 Cancelled" on the question). Pending questions live in memory, so after a restart a tap only gets
  "expired" and the buttons are removed. All tool calls of one message run concurrently, so several questions can
  be open at once; a typed text answers the newest.
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
  so never insert or drop messages in the request path. `_request_messages` only strips keys starting with `_`
  (and, for a model without system messages, sends a stored system notice as a `<bot_notice>` user block).
- **Preserved thinking (Sonnet 5.5): the transcript is append-only.** The API binds each thinking block to the
  model, system prompt, tools and every message before it; a changed one is a 400 for accounts created on or
  after 2026-08-31 (older accounts are not enforced). So the stored transcript is exactly what was sent: the
  `<context>` block is stored with its turn, tool results are cut (`transcript.cap_text`) before the model sees
  them, empty text blocks are dropped when the assistant message is appended, and redacted_thinking blocks go
  back unchanged. The final answer is stored text-only; that is safe because no later block depends on it.
  The only edits are boundaries that strip every thinking block (`transcript.strip_thinking`): `prune` above
  `max_context_tokens` or `MAX_TRANSCRIPT_IMAGES` (clear old tool results to `CLEARED_MARKER`, a browser result
  keeping its `browser_state` so it stays a valid toolset result, old image/document blocks and the pictures
  beyond `IMAGE_PRUNE_TARGET` to the `*_CLEARED_MARKER` texts, then summarize the oldest half into one
  `<earlier_conversation_summary>` user message, down to `PRUNE_TARGET_RATIO` of the budget), a change of
  `transcript.prefix_fingerprint`
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
  It inherits the registered providers (vision, profile, MCP, web, document, sandbox status) but not
  `run_subagent`, never `BrowserTools` (one browser per user, driven by the main agent only) and never `TaskTools`
  (`inherit_to_subagents = False`); `_child` sets `inbox` and `background` to None, so the user's messages reach
  the main loop only and the child's own calls always run to their end.
- **Sandbox sessions are shared state.** Every command of a user runs in the same container, so a background
  process, a full disk or a broken apt state survives until the session closes (/sandbox → Close, or idle).
  Only the workspace and the saved package lists survive the session; `cd` and environment variables do not
  even survive the command (each is a fresh `bash -lc`). Tests that build a `ContainerManager` without
  `__init__` always use one-off containers; tests that build a manager call `sessions.reset_manager()`.
- `sessions.check_user_id` accepts `^[A-Za-z0-9-][A-Za-z0-9_-]{0,63}$` (chat ids, negative ones included); a
  leading `_` is refused because `data/_sandbox` lives next to the user directories.
- The idle timer is per kind (`reap_idle` and `closes_at` use each session's own last use) while the warning is
  per user (`due_warnings` uses the latest activity of both). A user who keeps using only the shell sees the
  browser close at its own time without a warning; one `extend` lines both up again. Warnings live in memory
  (`sandbox_ui._warnings`): after a restart an old button only answers with the current closing time.
- The patcher's `install_package` signature is `(package_name, manager="apt", timeout=300)` end to end
  (`tool_integrator`, `SecureToolWrapper`, `ContainerManager`, `SessionManager.install_package`); pass `manager`
  and `timeout` by keyword if you touch it.
- **Browser toolset ordering**: the model may put several browser actions in one message; they run one at a
  time in block order and stop at the first failure (`NOT_EXECUTED`). Never move them into the concurrent
  `gather` of normal tools. A `tool_result` of a member must carry `toolset_name` and only `text` / `image` /
  `browser_state` blocks (a tab member's success: exactly one `browser_state`, which a cleared old result keeps);
  a normal tool's result must not carry it. Any code that rewrites stored results must keep them valid.
  Other members get `browser_state` only when the tabs changed since the server's last report, which may have
  gone to another conversation (forum topics share one browser) or a cancelled turn, so `execute_member` sends
  `force_state: true` until a member call of the turn succeeded (never a `browser_state` on an error).
- Element refs (`ref_N`) come from ONE server-wide counter: a ref of another tab or an earlier page is reported
  stale, never resolved to a different element.
- The confirmation guard is best effort. `browser_server.py` refuses an action whose target or page looks like
  buying, paying, sending a message, deleting or changing account security (`needs_confirmation`), and
  `BrowserTools` allows ONE such action within 5 minutes after the user tapped ✅ Allow on `browser_confirm`,
  in the same turn only. Registering, logging in and accepting terms are deliberately not sensitive (a bare
  "Отправить" counts only in a message context: a message box, chat, comment or reply form; a click into a text
  field never counts). Covered: left/middle/double clicks (the ref's element AND what is under its click point),
  `left_mouse_down` / `left_mouse_up` / drags (press and release points), `form_input` on non-text controls,
  Enter and Space in `key` / `type` / `hold_key` (re-checked on the element focused right before each one: "Tab
  Enter" is caught; a second consequential key in one allowed call stops there), page confirm() dialogs, and
  back/forward/reload that would re-send a POST (aborted in the route filter). The point and the focus are
  resolved through DevTools (`_describe_at`: `DOM.getNodeForLocation`; `_describe_focused`: `DOM.describeNode`
  shadow roots), so closed shadow roots and cross-site (out-of-process) frames are looked into; a cross-origin
  frame that cannot be entered counts only when its URL/title/name looks like a payment form. Limits: names are
  heuristics; a CSS-transformed iframe gets the point translated without the transform; a page that acts on
  other keys or moves the focus by itself while letters are typed is not re-checked.
- The live view is ONE silent photo per turn, edited in place (`ChatSender.edit_photo`, at most every 2.5 s,
  skipped when nothing changed; a refused edit sends nothing more). Its caption never shows typed text. A turn
  with a `quiet_marker` (scheduled tasks, their Retry) has none: `run_turn` hands the agent its settings with
  `browser.stream` off for that turn (not saved; no fingerprint change), so a quiet run leaves nothing behind.
- Typed text is masked wherever people see a call: `type`'s `text` and `form_input`'s `value`
  (`browser_tools.TYPED_INPUTS`) show as `<N characters>` in the trace (status line, rich summary args JSON:
  `main.display_input` → `masked_input`) and the reasoning file (`agent_runner._call_args`). The request and the
  stored `tool_use` keep the real value (append-only), and so do stdout (`print("\nResponse:", response)`), the
  transcript and the conversation files.
- `secure_container.sessions` is found by `agents/browser_tools.py` through `sys.modules` only: the agent package
  never imports docker. Without a manager (no Docker, tests, `init_manager` failed) neither `SandboxTools` nor
  `BrowserTools` is registered and /sandbox says the sandbox is unavailable.
- Two bot processes must not share a Docker host: `cleanup_stale` and the per-start orphan removal match on the
  `generall.sandbox` label only.
- **Live turns read messages at tool boundaries only**, plus once before the first model call
  (`ChainOfThoughtAgent._read_early_messages`, into the turn's own user message, which is not sent yet; like any
  drain it waits up to `DRAIN_WAIT` 3 s for a message at the head that is still being prepared). A message
  that arrives after that while the model writes its final answer, during a turn without tool calls (the quick
  Haiku path, `tools.enabled` off) or after the last batch is not read by that turn: it becomes the next turn
  (leftover). Never inject anywhere else (into an already sent message, judge/critique re-prompts, the SYSTEM
  NOTICE call): the injected blocks must ride in the tool-result message (or the unsent first one), and a notice in
  the system message right after it, which is what keeps the transcript append-only and the thinking blocks valid.
  Never put tool output or a background result into either (it belongs in a `tool_result`: `get_task_result`), and
  never the user's words into a `tool_result` (a typed answer to a question is a `<user_message>` of its own).
  A mid-conversation system message must follow a user message and be followed by an assistant message: the loop
  always calls the model right after one (the forced final call appends its "Processing..." assistant message first).
- **The message window delays every turn a user starts by `MESSAGE_WINDOW`** (1.5 s; the "typing…" action covers
  it), a photo album by its own 10 s flush plus nothing more unless a window is open for its topic. A window only
  ever holds messages of one chat and topic. Its state (`_windows`, `_albums_handing`) lives in memory per event
  loop: a restart loses an open window's messages (like `DROP_PENDING_UPDATES` would). Everything that changes a
  window happens before the handler's first await, and `_close_window` hands the messages over without awaiting
  after it took the window out of `_windows`; keep both, or a message can slip between a window and its turn. A
  window's messages are prepared from its second message on, so a single message never is (its `_run_<kind>` does
  that, with the status lines it always had).
- Tools that talk to the user or wait for them go in `NEVER_BACKGROUND`. A question, a confirmation or a
  "working on it" message moved to the background would be answered with a placeholder while the user still
  sees it pending, and its real result would arrive as a notification later. A subagent never gets them at all:
  its `run_subagent` call may be moved to the background at any moment (even while the child waits for a tap), and
  a question still open after the turn would take the user's next text as its answer.
- A call moved to the background outlives its turn: Stop never cancels it (only `/cancel`, `cancel_task`, a
  failure of the turn that adopted it and shutdown do), its usage after the turn's `record_usage` (a subagent's API
  tokens, an image or video cost) reaches the same trace and is recorded through `trace.on_late_usage`, one
  `usage_events` row per addition with `tool_calls` 0 (the `track_tool_used` row was written when the call
  started), and a backgrounded `run_subagent` stops editing the status message (`SubagentTools._status_hook` checks
  `task_tools.in_background()`), since its turn's status may already be the summary. Background tasks live in
  memory: a restart loses them and their results. Ids (`t1`, `t2`, …) count up per user for the life of the
  process; at most `MAX_BACKGROUND_TASKS` (4) run per user, a fifth long call stays in the foreground until one
  ends; `list()` keeps the last `MAX_LISTED` (20) finished ones.
- `TurnInbox.wait` returns only for what `drain` would hand over now: a user message counts once it is prepared,
  so `wait_for_tasks` and a watcher listening on the inbox neither spin nor wake while a video is still being
  prepared; the watcher moves its call when the message is ready (if the call has run 3 s by then).
  `has_user_messages` counts a ready, drainable user message only (a message still being prepared moves no call
  to the background); `has_pending` counts preparing items too, so the watcher polls (0.5 s) instead of listening
  while one is preparing. `drain` waits at most `DRAIN_WAIT` (3 s) for an unready head and then
  stops there: nothing behind it (a notification included) overtakes it; it is read at a later boundary or
  becomes a leftover.
- An album's slot is its first photo's `seq`, but it reaches the turn (or the queue) only at the 10 s flush: a
  text sent during that window and drained before the flush is read first. A message window open for its topic
  waits for it (`_album_arriving`) and orders it by its first photo's `message_id`.
- Messages of different forum topics are not ordered against each other: a topic-B message queued behind a
  running topic-A turn waits while later topic-A messages join that turn and are answered first.
- The inbox and host registries (`inbox._open`, `background._hosts` / `_batches`) belong to one event loop and are
  cleared when the running loop changes (every test; `inbox.reset()` / `background.reset()` exist too).
  `open_inbox` never replaces an inbox that is still open for the user: the new one is simply not registered.
- Requeueing leftovers and queueing notification turns must never suspend: they run in cleanup paths (a
  cancelled task, `finally`) and right after a check they depend on, hence `queue.submit_nowait`; likewise there
  is no `await` between `inject_target` and `inject`. Notification turns call no `track_message_received`, like
  scheduled tasks.
- The task tools and their `<live_turn>` notes are part of the tool list and the static prompt, so the first turn
  of every transcript after a deploy that changes them (adding `get_task_result` and the system notices did) strips
  the stored thinking once (fingerprint change). For
  the same reason `run_turn` must pass the same `inbox=` / `background=` on every turn: a path that gave the agent
  no host would drop the task tools and flip the fingerprint back and forth.

## How to make common changes

- **Add a tool**: new class in `app/agents/`, implement the contract (take `sender` if it talks to
  the user; `tools_schema`, `execute_tool` returning a string or text/image blocks; optional `prompt_notes`
  for what the model must know about the tools, instead of editing the system prompts), then ONE line in
  `ChainOfThoughtAgent.__init__` after the SubagentTools wiring: `self.agent.register_provider(Provider(...))`
  (awaited) or `register_provider(..., blocking=True)` (sync `execute_tool`, run in `asyncio.to_thread`).
  Subagents inherit it. Keep tool names unique; never let `tool_integrator.py` name the class or a method
  (add a guard test like `tests/test_web_tools.py`); if part of it must run in the sandbox, call
  `ContainerManager.run_script` with a bundled script rather than patching. The four named-slot steps are
  only for the original providers. A tool that talks to the user or waits for them (a question, a confirmation, a
  message sent now) goes into `agents.main.NEVER_BACKGROUND` (`tests/test_live_turn_loop.py::test_never_background_names`;
  subagents never get those tools);
  any other call may be moved to the background in a live turn, so it must not depend on the turn still running
  when it returns. A provider that subagents must not get sets `inherit_to_subagents = False`.
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
  An OAuth server needs no token: `/mcp add notion https://mcp.notion.com/mcp` → 🔐 button; a shared one gets
  `"auth": "oauth"` in the admin file (or is found by the startup refresh), then the admin signs in once from
  `/mcp` to fill its tool cache and every user signs in with their own account.
  Code changes: `bot/mcp_store.py` for records/validation/encrypted sign-ins, `agents/mcp_tools.py` for
  transport/result conversion (`convert_content`, `error_reason`) and the OAuth connection policy,
  `bot/oauth_flow.py` for the sign-in, `bot/oauth_web.py` for the callback page, `bot/handlers/mcp_ui.py` for the
  command. Tests with a fake OAuth + MCP server: `tests/mcp_oauth_fakes.py` (an httpx2 MockTransport handler,
  put into `agents.mcp_tools.HTTP_TRANSPORT`). Every read of a user's records goes through `mcp_store._read_user`
  under the user's lock (it runs the one-time migration), every write through `_write_user_file` (private
  directory, 0600); never put MCP state back under `data/<uid>/`, and delete tokens or records only through
  `oauth_flow.sign_out` / `remove_server` (the refresh lock).
- **Host the sign-in callback**: see "Running" (Cloudflare Tunnel recommended); code in `bot/oauth_web.py`
  (routes, headers, TLS rule), tests in `tests/test_oauth_web.py`. Never add an access log or a route that
  echoes the query: the request line carries the authorization code and the state.
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
  checked) when it starts an agent turn (`_start_turn`; `join_running=True` lets a plain-text prompt join a
  running turn of its topic, as follow-ups do).
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
- **Add a media type**: in `messages.py`, a `prepare_<kind>(bot, message, user_id[, progress])` that returns a
  `bot.inbox.Prepared` (prompt, `speak`, `error_text`, `reason`, optional `echo` / `inject`) without sending
  status messages itself (report through `progress`), a `_run_<kind>(bot, message, user_id, limit, ctx)` job that
  wraps it in the status messages and calls `run_turn`, and an `@router.message(F.<kind>)` handler that tracks it
  and calls `accept(..., prepare=, run=)`, so the message can join a running turn (the inject path), go into the
  message window with others (`prepare`, then `inbox.combine`) or queue alone (`run`). `prepare_<kind>` must not
  depend on running inside the job: it also runs as a task while the window is open. Tests in `tests/test_inbox.py`.
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
- **Change sandbox limits/mounts/network**: persistent sessions in `secure_container/sessions.py` (the env
  variables above, `SessionManager._start_shell` / `_start_browser` for mounts and container options,
  `BLOCKED_RANGES` / `egress_rules` for the filter; `tests/test_sessions.py` uses a fake Docker client);
  one-off containers in `ContainerManager._run_command_in_slot` (slot cap `MAX_SANDBOX_CONTAINERS`, default
  memory `DEFAULT_MEM_LIMIT`, callers may pass `mem_limit`).
- **Change what the browser can do**: the member implementations, the sensitive-action patterns and the
  private-address check (`HostPolicy` + `EgressProxy`: Chromium's `--proxy-server` with `<-loopback>`, so
  redirects, popups and workers are checked too; the route filter is the second layer) live in
  `secure_container/scripts/browser_server.py` (`tests/test_browser_server.py`,
  real Chromium when Playwright is installed); enabling a disabled member is the `configs` of
  `BrowserTools.toolset` plus its implementation there. What the agent is told: `BrowserTools` /
  `SandboxTools.prompt_notes`. Changing the server needs no image rebuild (it is copied in at every browser
  start); a running browser keeps the old script until it closes.
- **Change the sandbox image**: edit `secure_container/Dockerfile`, `entrypoint.sh` or `requirements.txt`; the
  tag changes with the content, so the next startup builds it (several minutes with Chromium) and re-points
  `:latest`. `tests/test_image_tag.py` covers the tagging.
- **Run the tests**: `pip install -r requirements-dev.txt && pytest` from the repo root (Python ≥ 3.12).
- **Smoke-check without Telegram or Docker**: from `app/`, with `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
  `TAVILY_API_KEY` set to any non-empty value, run `python -c "import bot.app, agents.main, models"`. It
  catches syntax and import errors in the whole bot and tool code. Running `main_bot.py` additionally
  needs Docker and a real token.
