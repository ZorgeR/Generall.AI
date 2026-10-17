"""How the message-button features work together: Stop, turn buttons, questions, the settings page.

Each feature has its own test file; this one covers the seams between them: a late
/cancel against a stop in progress or a delivered answer, a tap racing a cancel,
the settings page reaching the agent, the reminder job storing the displayed
answer, and every callback family reaching its own handler.
"""
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import Chat, InputRichBlockButtons, Message

import bot.agent_runner as agent_runner
import bot.streaming as streaming
from bot import questions, rich
from bot.handlers import messages
from bot.handlers import settings_ui as su
from bot.queue import Job, JobContext, QueueManager
from bot.sender import ChatSender
from bot.settings import UserSettings

APP = Path(__file__).resolve().parents[1] / "app"
USER = "42"
ANSWER = "Here is X.\n\n<follow_ups>\n- Make it shorter\n- Compare with 2024\n</follow_ups>"


class FakeBot:
    """Records the Bot API calls; ``markup_gate`` makes edit_message_reply_markup wait."""

    def __init__(self, *, markup_gate: asyncio.Event | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.markup_gate = markup_gate
        self.markup_started = asyncio.Event()
        self.sent_ids: list[int] = []  # message ids returned by send_message, in order
        self._id = 100

    def _msg(self, **extra):
        self._id += 1
        return SimpleNamespace(message_id=self._id, **extra)

    async def send_chat_action(self, **kw):
        return True

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        message = self._msg()
        self.sent_ids.append(message.message_id)
        return message

    async def send_rich_message(self, **kw):
        self.calls.append(("send_rich_message", kw))
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        return SimpleNamespace(message_id=kw["message_id"])

    async def edit_message_reply_markup(self, **kw):
        self.calls.append(("edit_message_reply_markup", kw))
        self.markup_started.set()
        if self.markup_gate is not None:
            await self.markup_gate.wait()
        return True

    async def delete_message(self, **kw):
        self.calls.append(("delete_message", kw))
        return True

    async def send_document(self, **kw):
        self.calls.append(("send_document", kw))
        return self._msg()

    async def send_message_draft(self, **kw):
        return True

    async def send_rich_message_draft(self, **kw):
        return True

    def of(self, name):
        return [kw for n, kw in self.calls if n == name]


def use_agent(monkeypatch, response=ANSWER, *, error: Exception | None = None, gate: asyncio.Event | None = None):
    """A ChainOfThoughtAgent stand-in; records the user_settings it was built with."""
    built: list[dict] = []

    class Agent:
        def __init__(self, **kw):
            built.append(kw["user_settings"])
            self.agent = SimpleNamespace(limit_reached=False)

        async def generate_response(self, prompt, update_status, on_text_chunk=None, trace=None):
            await update_status("thinking", "Generating", 0, 0)
            if gate is not None:
                await gate.wait()
            if error is not None:
                raise error
            return response, [{"role": "assistant", "content": [{"type": "text", "text": response}]}]

    monkeypatch.setattr(agent_runner, "_agents", lambda: SimpleNamespace(ChainOfThoughtAgent=Agent))
    return built


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    """Settings, turns and reminders under tmp_path; a no-op stats tracker; drafts off; rich on."""
    monkeypatch.chdir(tmp_path)
    stats = ModuleType("stats")
    stats.stats_tracker = SimpleNamespace(
        track_message_sent=lambda *a, **k: None,
        track_usage=lambda *a, **k: None,
        get_user_action_count=lambda *a, **k: 0,
        track_message_received=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "stats", stats)
    monkeypatch.setattr(streaming, "config", SimpleNamespace(streaming_enabled=False))
    monkeypatch.setitem(su._page_state, "flat", False)
    rich.reset()
    questions.reset()
    yield
    rich.reset()
    questions.reset()


def buttons(kw) -> list[str]:
    """Callback data of every button on one send/edit: rich button rows and the inline keyboard."""
    data = []
    message = kw.get("rich_message")
    if message is not None and message.blocks:
        data += [b.callback_data for blk in message.blocks if isinstance(blk, InputRichBlockButtons) for b in blk.buttons]
    if kw.get("reply_markup") is not None:
        data += [b.callback_data for row in kw["reply_markup"].inline_keyboard for b in row]
    return data


def edit_text(kw) -> str:
    """The visible text of one edit_message_text call (rich paragraph or plain text)."""
    if kw.get("rich_message") is not None:
        return str(kw["rich_message"].blocks[0].text)
    return kw.get("text") or ""


async def wait_idle(qm: QueueManager, user_id: str = USER) -> None:
    for _ in range(200):
        if not qm.is_busy(user_id):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the queue did not drain")


# ---- /cancel, Stop and the end of a turn ------------------------------------------------
async def test_cancel_does_not_interrupt_a_stop_that_is_cleaning_up():
    """/cancel (or the busy notice's button) while a Stop is being handled must not cut its cleanup."""
    qm = QueueManager(max_concurrent_turns=2, turn_timeout=30)
    events: list[str] = []
    started, cleaning = asyncio.Event(), asyncio.Event()

    async def job_fn(ctx):
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cleaning.set()
            await asyncio.sleep(0.05)  # the partial answer, then "🛑 Stopped."
            events.append(f"cleaned-up:{ctx.cancel_reason}")
            raise

    async def queued(ctx):
        events.append("queued-ran")

    job = Job(user_id=USER, label="long", run=job_fn)
    await qm.submit(job)
    await asyncio.wait_for(started.wait(), 5)
    assert await qm.cancel_current(USER, job.id) is True
    await qm.submit(Job(user_id=USER, label="queued", run=queued))
    await asyncio.wait_for(cleaning.wait(), 5)
    assert await qm.cancel(USER) == (True, 1)  # still reported as stopping; the queue is dropped
    await wait_idle(qm)
    assert events == ["cleaned-up:user"]
    await qm.shutdown()


async def test_late_cancel_never_overwrites_a_delivered_legacy_answer(monkeypatch):
    """Legacy mode: the status message IS the answer; a cancel while its buttons go on must leave it."""
    UserSettings(USER).set("rich_messages", False, "enabled")
    use_agent(monkeypatch)
    gate = asyncio.Event()
    bot = FakeBot(markup_gate=gate)
    ctx = JobContext(Job(user_id=USER, label="text", run=None))
    task = asyncio.create_task(run_turn_task(bot, ctx))
    await asyncio.wait_for(bot.markup_started.wait(), 5)
    ctx.cancel_reason = "user"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    status_id = bot.sent_ids[0]  # the status, which the answer replaced
    texts = [edit_text(kw) for kw in bot.of("edit_message_text") if kw["message_id"] == status_id]
    assert texts[-1] == "Here is X."  # the answer stays; no "🛑 Stopped." over it
    assert not any(d.startswith("queue_stop_") for d in buttons(bot.of("edit_message_text")[-1]))


async def test_late_cancel_keeps_the_rich_summary(monkeypatch):
    use_agent(monkeypatch)
    gate = asyncio.Event()
    bot = FakeBot(markup_gate=gate)
    ctx = JobContext(Job(user_id=USER, label="text", run=None))
    task = asyncio.create_task(run_turn_task(bot, ctx))
    await asyncio.wait_for(bot.markup_started.wait(), 5)  # follow-ups going under the answer
    ctx.cancel_reason = "user"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    last = bot.of("edit_message_text")[-1]
    assert "🛑" not in edit_text(last)  # the summary with its action row is what stays
    assert any(d.startswith("turn_retry_") for d in buttons(last))


async def test_cancel_before_delivery_still_marks_the_status(monkeypatch):
    use_agent(monkeypatch, gate=asyncio.Event())  # the agent never returns
    bot = FakeBot()
    ctx = JobContext(Job(user_id=USER, label="text", run=None))
    task = asyncio.create_task(run_turn_task(bot, ctx))
    for _ in range(100):
        if bot.of("edit_message_text"):
            break
        await asyncio.sleep(0.01)
    ctx.cancel_reason = None  # e.g. the turn deadline
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    last = bot.of("edit_message_text")[-1]
    assert edit_text(last) == "🛑 Cancelled." and buttons(last) == []


async def test_a_failed_turn_has_no_stop_and_only_retry(monkeypatch):
    use_agent(monkeypatch, error=RuntimeError("boom"))
    bot = FakeBot()
    ctx = JobContext(Job(user_id=USER, label="text", run=None))
    assert await run_turn_task(bot, ctx) is None
    last = bot.of("edit_message_text")[-1]
    assert edit_text(last).startswith("❌ Something went wrong.")
    assert len(buttons(last)) == 1 and buttons(last)[0].startswith("turn_retry_")  # Retry only (tests/test_error_recovery.py)
    assert not any(d.startswith("queue_stop_") for d in buttons(last))
    assert bot.of("edit_message_reply_markup") == [] and bot.of("send_document") == []


async def test_stop_during_delivery_says_the_answer_is_on_its_way(monkeypatch):
    qm = QueueManager(max_concurrent_turns=2, turn_timeout=30)
    monkeypatch.setattr(messages, "queue", qm)
    release = asyncio.Event()
    ran = asyncio.Event()

    async def delivering(ctx):
        ctx.stoppable = False  # what run_turn does once the agent returned
        ran.set()
        await release.wait()

    job = Job(user_id=USER, label="text", run=delivering)
    await qm.submit(job)
    await asyncio.wait_for(ran.wait(), 5)
    cb = SimpleNamespace(data=f"queue_stop_{job.token}", message=None, answers=[])

    async def answer(text=None, **kw):
        cb.answers.append(text)

    cb.answer = answer
    await messages.stop_button(cb, user_id=USER)
    assert cb.answers == [messages.STOP_TOO_LATE_TEXT]
    release.set()
    await wait_idle(qm)
    await messages.stop_button(cb, user_id=USER)
    assert cb.answers[-1] == messages.STOP_FINISHED_TEXT
    await qm.shutdown()


async def run_turn_task(bot, ctx):
    return await agent_runner.run_turn(bot=bot, user_id=USER, chat_id=42, prompt="Tell me about X", ctx=ctx)


# ---- ask_user_choice racing a cancel ----------------------------------------------------
async def test_cancel_right_after_a_tap_keeps_the_answer_line():
    bot = FakeBot()
    sender = ChatSender(bot, 42, rich=True)
    task = asyncio.create_task(questions.ask(sender, "Send it?", ["Send", "Cancel"], 60))
    for _ in range(100):
        if questions._pending:
            break
        await asyncio.sleep(0.01)
    (qid,) = list(questions._pending)
    assert questions.resolve(qid, 42, 0) == "Send"  # the tap (its handler edits "✅ Send")
    task.cancel()  # /cancel lands before the tool call wakes up
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not any(questions.CANCELLED_LINE in str(kw) for kw in bot.of("edit_message_text"))
    assert questions.state(qid) == "answered"


async def test_cancel_while_waiting_still_closes_the_question():
    bot = FakeBot()
    sender = ChatSender(bot, 42, rich=True)
    task = asyncio.create_task(questions.ask(sender, "Send it?", ["Send", "Cancel"], 60))
    for _ in range(100):
        if questions._pending and next(iter(questions._pending.values())).message_id:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert any(questions.CANCELLED_LINE in str(kw) for kw in bot.of("edit_message_text"))


# ---- the settings page --------------------------------------------------------------------
def _settings_message(message_id=7):
    return Message(message_id=message_id, date=datetime.now(timezone.utc), chat=Chat(id=int(USER), type="private"))


def _bad_request(text):
    return TelegramBadRequest(method=SendMessage(chat_id=1, text="x"), message=text)


@pytest.mark.parametrize("error", [
    "Bad Request: message thread not found",
    "Bad Request: TOPIC_CLOSED",
    "Bad Request: not enough rights to send text messages to the chat",
    "Bad Request: chat not found",
])
async def test_a_chat_specific_refusal_keeps_page_buttons_for_everyone(monkeypatch, error):
    calls = []

    async def answer_md(message, text, reply_markup=None):
        calls.append(text)

    monkeypatch.setattr(su, "answer_md", answer_md)
    bot = SimpleNamespace(send_rich_message=AsyncMock(side_effect=[_bad_request(error)] * 2))
    await su.settings_command(_settings_message(), bot)
    assert bot.send_rich_message.await_count == 2 and len(calls) == 1  # the keyboards, this once
    assert rich.buttons_available()  # one chat's problem is not the server's


def test_chat_errors_are_told_apart_from_payload_errors():
    for text in ("Bad Request: message thread not found", "Bad Request: TOPIC_CLOSED",
                 "Bad Request: have no rights to send a message", "Bad Request: CHAT_WRITE_FORBIDDEN"):
        assert rich.is_chat_error(_bad_request(text)), text
    for text in ("Bad Request: unknown block type", "Bad Request: BUTTON_DATA_INVALID",
                 "Bad Request: can't parse rich message", "Bad Request: message is not modified"):
        assert not rich.is_chat_error(_bad_request(text)), text


async def test_a_refused_button_block_still_switches_page_buttons_off(monkeypatch):
    monkeypatch.setattr(su, "answer_md", AsyncMock())
    bot = SimpleNamespace(send_rich_message=AsyncMock(side_effect=[_bad_request("Bad Request: unknown block type")] * 2))
    await su.settings_command(_settings_message(), bot)
    assert not rich.buttons_available()


async def test_a_page_tap_is_applied_even_when_its_answer_fails(monkeypatch):
    too_old = TelegramBadRequest(method=AnswerCallbackQuery(callback_query_id="1"), message="Bad Request: query is too old")
    bot = SimpleNamespace(edit_message_text=AsyncMock())
    cb = SimpleNamespace(data="setp_thinking_toggle", message=_settings_message(), answer=AsyncMock(side_effect=too_old))
    await su.settings_page_button(cb, bot)
    assert UserSettings(USER).get("thinking", "enabled") is False
    bot.edit_message_text.assert_awaited_once()


async def test_the_page_toggle_reaches_the_agent_and_the_answer_buttons(monkeypatch):
    """setp_followups_toggle -> the agent gets follow_ups off -> the block is hidden, no follow-up rows."""
    page_bot = SimpleNamespace(edit_message_text=AsyncMock())
    await su.settings_page_button(SimpleNamespace(data="setp_followups_toggle", message=_settings_message(), answer=AsyncMock()), page_bot)
    assert UserSettings(USER).get("follow_ups", "enabled") is False

    built = use_agent(monkeypatch)
    bot = FakeBot()
    result = await agent_runner.run_turn(bot=bot, user_id=USER, chat_id=42, prompt="Tell me about X")
    assert built[0]["follow_ups"] == {"enabled": False}  # what ChainOfThoughtAgent reads for FOLLOW_UPS_GUIDE
    assert result.response == "Here is X."
    assert [kw["rich_message"].markdown for kw in bot.of("send_rich_message")] == ["Here is X."]
    assert not any(d.startswith("turn_fu_") for name, kw in bot.calls for d in buttons(kw))
    assert any(d.startswith("turn_retry_") for d in buttons(bot.of("edit_message_text")[-1]))  # actions still there


def test_the_agent_prompt_follows_the_stored_setting(tmp_path):
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    import agents.main as agents_main

    us = UserSettings(USER)
    assert su.apply_setting(us, "followups", "toggle") is True
    agent = agents_main.ChainOfThoughtAgent(user_id=USER, user_settings=UserSettings(USER).as_dict())
    seen: list[str] = []

    async def capture(question, system_context, *a, **k):
        seen.append(system_context)
        return "ok", []

    agent._generate_with_transcript = capture
    asyncio.run(agent.generate_response("hi"))
    assert agents_main.FOLLOW_UPS_GUIDE not in seen[0]


# ---- the reminder job stores the displayed answer ---------------------------------------
async def test_an_agent_reminder_stores_the_answer_without_the_follow_ups_block(monkeypatch):
    from bot import jobs
    from reminders_store import reminders_store

    use_agent(monkeypatch)
    monkeypatch.setattr(jobs, "check_user_limits", lambda uid: (True, 0, None))
    reminder = await reminders_store.add(USER, {
        "text": "Report on X", "type": "agent", "status": "processing", "time": datetime.now(timezone.utc).isoformat(),
    })
    bot = FakeBot()
    await jobs.run_agent_reminder(bot, USER, reminder, JobContext(Job(user_id=USER, label="reminder", run=None)))
    (stored,) = await reminders_store.load(USER)
    assert stored["status"] == "completed" and stored["agent_response"] == "Here is X."


# ---- callback routing ---------------------------------------------------------------------
ROUTING = {
    "settings_main": "settings_ui.settings_button",
    "settings_system_prompt_set_generall-ai-v1": "settings_ui.settings_button",
    "setp_followups_toggle": "settings_ui.settings_page_button",
    "setp_sysprompt_perplexity-r1": "settings_ui.settings_page_button",
    "turn_retry_0123456789ab": "turn_actions_ui.on_retry",
    "turn_cont_0123456789ab": "turn_actions_ui.on_continue",
    "turn_fu_0123456789ab_3": "turn_actions_ui.on_follow_up",
    "turn_listen_0123456789ab": "turn_actions_ui.on_listen",
    "turn_reason_0123456789ab": "turn_actions_ui.on_reasoning",
    "turn_something_else": "turn_actions_ui.on_unknown",
    "ask_0123abcd_7": "questions_ui.on_question_answer",
    "queue_cancel": "messages.cancel_button",
    "queue_stop_1f": "messages.stop_button",
    "reminder_toggle_abcd1234": "reminders_ui.reminder_button",
    "reminders_page_1": "reminders_ui.reminder_button",
    "noop": "reminders_ui.noop_button",
    "voice_rachel": "commands.voice_button",
    "stats_back_main": "stats_ui.stats_button",
    "usage_refresh": "usage_ui.usage_button",
    "profile_refresh": "profile_ui.profile_button",
    "profile_clear_yes": "profile_ui.profile_button",
    "mcp_toggle_github": "mcp_ui.mcp_button",
    "mcp_rm_github": "mcp_ui.mcp_button",
    "mcp_refreshall": "mcp_ui.mcp_button",
    "setp_preset_fast": "settings_ui.settings_page_button",
    "setp_ui_advanced": "settings_ui.settings_page_button",
    "settings_preset_fast": "settings_ui.settings_button",
    "settings_mode_set_speed": "settings_ui.settings_button",
    "tz_set_0": "commands.timezone_button",
    "reminder_snooze_abcd1234_10": "reminders_ui.reminder_button",
    "reminder_delay_abcd1234_60": "reminders_ui.reminder_button",
    "reminder_edit_abcd1234": "reminders_ui.reminder_button",
    "reminders_new": "reminders_ui.reminder_button",
}


def test_every_callback_family_reaches_its_own_handler(tmp_path):
    """The first matching handler, in the router order of build_root_router(), for each family.

    Run in a fresh interpreter: build_root_router() attaches module-level routers, which
    aiogram allows only once per process. A family's specific handler must win; the only
    second match allowed is the turn_ catch-all behind the specific turn_ handlers.
    """
    code = (
        "import asyncio, json, sys\n"
        "from aiogram.types import CallbackQuery, User\n"
        "from bot.handlers import build_root_router\n"
        "root = build_root_router()\n"
        "def walk(r):\n"
        "    for h in r.callback_query.handlers:\n"
        "        yield h\n"
        "    for s in r.sub_routers:\n"
        "        yield from walk(s)\n"
        "handlers = list(walk(root))\n"
        "async def matches(data):\n"
        "    cq = CallbackQuery(id='1', from_user=User(id=1, is_bot=False, first_name='a'), chat_instance='c', data=data)\n"
        "    out = []\n"
        "    for h in handlers:\n"
        "        ok, _ = await h.check(cq)\n"
        "        if ok:\n"
        "            out.append(h.callback.__module__.rsplit('.', 1)[-1] + '.' + h.callback.__name__)\n"
        "    return out\n"
        "async def main():\n"
        "    return {d: await matches(d) for d in json.loads(sys.argv[1])}\n"
        "print(json.dumps(asyncio.run(main())))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(APP)}
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"):
        env.setdefault(key, "x")
    out = subprocess.run(
        [sys.executable, "-c", code, json.dumps(list(ROUTING))],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    found = json.loads(out.stdout.strip().splitlines()[-1])
    for data, expected in ROUTING.items():
        assert found[data] and found[data][0] == expected, (data, found[data])
        extra = found[data][1:]
        assert extra in ([], ["turn_actions_ui.on_unknown"]), (data, found[data])


# ---- message routing ----------------------------------------------------------------------
# scenario -> (message spec, first handler in router order). "pending": the chat has a pending
# ✏️ Edit text (reminders_ui._pending_edits), whose handler sits on the ui router before chat.
MESSAGE_ROUTING = {
    "plain text": ({"text": "hello"}, "messages.on_text"),
    "text while an edit is pending": ({"text": "call mom", "pending": True}, "reminders_ui.reminder_edit_text"),
    "location": ({"location": [52.52, 13.405]}, "messages.on_location"),
    "/timezone": ({"text": "/timezone Berlin"}, "commands.timezone_command"),
    "/reminders while an edit is pending": ({"text": "/reminders", "pending": True}, "reminders_ui.reminders_command"),
    "/cancel while an edit is pending": ({"text": "/cancel", "pending": True}, "messages.cancel_command"),
    "photo": ({"photo": True}, "messages.on_photo"),
}


def test_every_message_kind_reaches_its_own_handler(tmp_path):
    """Like the callback routing above, for messages: the pending-edit handler must take a plain text
    only while an edit is pending. Without its filter it would swallow every user's text (the ui
    router runs before chat), and commands must pass it even while an edit is pending."""
    code = (
        "import asyncio, json, sys, time\n"
        "from datetime import datetime, timezone\n"
        "from types import SimpleNamespace\n"
        "from aiogram.types import Chat, Location, Message, PhotoSize, User\n"
        "from bot.handlers import build_root_router, reminders_ui\n"
        "root = build_root_router()\n"
        "def walk(r):\n"
        "    for h in r.message.handlers:\n"
        "        yield h\n"
        "    for s in r.sub_routers:\n"
        "        yield from walk(s)\n"
        "handlers = list(walk(root))\n"
        "bot = SimpleNamespace(id=999)\n"
        "async def matches(spec):\n"
        "    reminders_ui.reset()\n"
        "    if spec.get('pending'):\n"
        "        reminders_ui._pending_edits['42'] = ('abcd1234', time.monotonic() + 60)\n"
        "    extra = {}\n"
        "    if 'text' in spec:\n"
        "        extra['text'] = spec['text']\n"
        "    if 'location' in spec:\n"
        "        extra['location'] = Location(latitude=spec['location'][0], longitude=spec['location'][1])\n"
        "    if spec.get('photo'):\n"
        "        extra['photo'] = [PhotoSize(file_id='f', file_unique_id='u', width=10, height=10)]\n"
        "    msg = Message(message_id=1, date=datetime.now(timezone.utc), chat=Chat(id=42, type='private'),\n"
        "                  from_user=User(id=42, is_bot=False, first_name='a'), **extra)\n"
        "    out = []\n"
        "    for h in handlers:\n"
        "        ok, _ = await h.check(msg, bot=bot)\n"
        "        if ok:\n"
        "            out.append(h.callback.__module__.rsplit('.', 1)[-1] + '.' + h.callback.__name__)\n"
        "    return out\n"
        "async def main():\n"
        "    specs = json.loads(sys.argv[1])\n"
        "    return {name: await matches(spec) for name, spec in specs.items()}\n"
        "print(json.dumps(asyncio.run(main())))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(APP)}
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"):
        env.setdefault(key, "x")
    specs = {name: spec for name, (spec, _) in MESSAGE_ROUTING.items()}
    out = subprocess.run(
        [sys.executable, "-c", code, json.dumps(specs)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    found = json.loads(out.stdout.strip().splitlines()[-1])
    for name, (_, expected) in MESSAGE_ROUTING.items():
        assert found[name] and found[name][0] == expected, (name, found[name])
    # the pending-edit handler never sees a text when no edit is pending, nor a command when one is
    for name in ("plain text", "/reminders while an edit is pending", "/cancel while an edit is pending"):
        assert "reminders_ui.reminder_edit_text" not in found[name], (name, found[name])
