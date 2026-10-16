"""Stopping a running turn: the status-message Stop button, Telegram's Stop on a draft, the partial answer."""
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from aiogram.types import Chat, InputRichBlockButtons, MessageGenerationStopped

import bot.agent_runner as agent_runner
import bot.streaming as streaming
from bot import rich
from bot.agent_runner import STOPPED_NOTE, run_turn
from bot.handlers import messages
from bot.queue import Job, JobContext, QueueManager

APP = Path(__file__).resolve().parents[1] / "app"
USER = "42"


class FakeBot:
    """Records every Bot API call run_turn makes; returns message-like objects."""

    def __init__(self, *, rich_send_error: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.rich_send_error = rich_send_error
        self._id = 100

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_chat_action(self, **kw):
        return True

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        return self._msg()

    async def send_rich_message(self, **kw):
        self.calls.append(("send_rich_message", kw))
        if self.rich_send_error is not None:
            raise self.rich_send_error
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        return self._msg()

    async def delete_message(self, **kw):
        self.calls.append(("delete_message", kw))
        return True

    async def send_document(self, **kw):
        self.calls.append(("send_document", kw))
        return self._msg()

    async def send_message_draft(self, **kw):
        self.calls.append(("draft", kw))
        return True

    async def send_rich_message_draft(self, **kw):
        self.calls.append(("draft", kw))
        return True

    def of(self, name):
        return [kw for n, kw in self.calls if n == name]


def fake_agents(behaviour):
    """An ``agents.main`` stand-in whose agent runs ``behaviour(update_status, on_text_chunk, trace)``."""

    class Agent:
        def __init__(self, **kw):
            self.kw = kw

        async def generate_response(self, prompt, update_status, on_text_chunk=None, trace=None):
            return await behaviour(update_status, on_text_chunk, trace)

    return SimpleNamespace(ChainOfThoughtAgent=Agent)


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    """Settings/workspace under tmp_path, a no-op stats tracker, drafts on, rich support on."""
    monkeypatch.chdir(tmp_path)
    stats = ModuleType("stats")
    stats.stats_tracker = SimpleNamespace(
        track_message_sent=lambda *a, **k: None,
        track_usage=lambda *a, **k: None,
        get_user_action_count=lambda *a, **k: 0,
    )
    monkeypatch.setitem(sys.modules, "stats", stats)
    monkeypatch.setattr(streaming, "config", SimpleNamespace(streaming_enabled=True))
    rich.reset()
    yield
    rich.reset()


def stop_rows(kw) -> list[str]:
    """Callback data of the Stop buttons on one status edit/send (rich row or inline keyboard)."""
    data = []
    message = kw.get("rich_message")
    if message is not None and message.blocks:
        data += [b.callback_data for blk in message.blocks if isinstance(blk, InputRichBlockButtons) for b in blk.buttons]
    if kw.get("reply_markup") is not None:
        data += [b.callback_data for row in kw["reply_markup"].inline_keyboard for b in row]
    return data


# ---- run_turn wiring ------------------------------------------------------------
async def test_turn_status_and_drafts_name_the_job_and_the_button_goes_away(monkeypatch):
    async def behaviour(update_status, on_text_chunk, trace):
        await update_status("thinking", "Generating", 0, 0)
        await on_text_chunk("Hel", False)
        return "Hello world", [{"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}]

    monkeypatch.setattr(agent_runner, "_agents", lambda: fake_agents(behaviour))
    bot = FakeBot()
    job = Job(user_id=USER, label="text", run=None)
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi", ctx=JobContext(job))
    assert result is not None and result.response == "Hello world"

    expected = f"queue_stop_{job.token}"
    first_status = bot.of("send_message")[0]
    assert stop_rows(first_status) == [expected]  # the button is there from the first moment
    edits = bot.of("edit_message_text")
    assert stop_rows(edits[0]) == [expected]  # rich progress: the button row
    # The end-of-turn summary has no Stop button (it may carry the turn's own action buttons).
    assert not any(d.startswith("queue_stop_") for d in stop_rows(edits[-1]))
    drafts = bot.of("draft")
    assert drafts and all(d["draft_id"] == job.id and d["can_stop"] and d["keep_on_stop"] for d in drafts)
    assert bot.of("send_rich_message")[0]["rich_message"].markdown == "Hello world"


async def test_turn_without_a_job_has_no_stop_button(monkeypatch):
    async def behaviour(update_status, on_text_chunk, trace):
        await update_status("thinking", "Generating", 0, 0)
        await on_text_chunk("Hel", False)
        return "Hello", []

    monkeypatch.setattr(agent_runner, "_agents", lambda: fake_agents(behaviour))
    bot = FakeBot()
    await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi")
    assert stop_rows(bot.of("send_message")[0]) == []
    assert not any(d.startswith("queue_stop_") for kw in bot.of("edit_message_text") for d in stop_rows(kw))
    draft = bot.of("draft")[0]
    assert draft["draft_id"] > 0 and "can_stop" not in draft and "keep_on_stop" not in draft


async def test_user_stop_keeps_the_streamed_partial_answer(monkeypatch):
    started = asyncio.Event()

    async def behaviour(update_status, on_text_chunk, trace):
        await update_status("thinking", "Generating", 0, 0)
        await on_text_chunk("weighing options", True)
        await on_text_chunk("Partial answer", False)
        started.set()
        await asyncio.sleep(10)

    monkeypatch.setattr(agent_runner, "_agents", lambda: fake_agents(behaviour))
    bot = FakeBot()
    qm = QueueManager(max_concurrent_turns=2, turn_timeout=30)
    queued_ran: list[str] = []

    async def queued(ctx):
        queued_ran.append("next")

    job = Job(user_id=USER, label="text", run=lambda ctx: run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi", ctx=ctx))
    await qm.submit(job)
    await qm.submit(Job(user_id=USER, label="next", run=queued))
    await asyncio.wait_for(started.wait(), 5)

    assert await qm.cancel_current(USER, job.id) is True
    for _ in range(100):
        if not qm.is_busy(USER):
            break
        await asyncio.sleep(0.02)
    assert queued_ran == ["next"]  # the stop is for this turn only

    names = [n for n, _ in bot.calls if n != "draft"]
    partial = bot.of("send_rich_message")
    assert len(partial) == 1 and partial[0]["rich_message"].markdown == f"Partial answer\n\n{STOPPED_NOTE}"
    stopped = bot.of("edit_message_text")[-1]
    assert stopped["rich_message"].blocks[0].text == "🛑 Stopped." and stop_rows(stopped) == []
    assert names[-2:] == ["send_rich_message", "edit_message_text"]  # the partial answer, then the notice
    await qm.shutdown()


async def test_a_failed_partial_send_never_replaces_the_cancellation(monkeypatch):
    started = asyncio.Event()

    async def behaviour(update_status, on_text_chunk, trace):
        await on_text_chunk("Partial", False)
        started.set()
        await asyncio.sleep(10)

    monkeypatch.setattr(agent_runner, "_agents", lambda: fake_agents(behaviour))
    bot = FakeBot(rich_send_error=RuntimeError("network down"))
    ctx = JobContext(Job(user_id=USER, label="text", run=None))
    task = asyncio.create_task(run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi", ctx=ctx))
    await asyncio.wait_for(started.wait(), 5)
    ctx.cancel_reason = "user"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(bot.of("send_rich_message")) == 1  # it was tried
    assert bot.of("edit_message_text")[-1]["rich_message"].blocks[0].text == "🛑 Stopped."


async def test_other_cancellations_send_no_partial_answer(monkeypatch):
    started = asyncio.Event()

    async def behaviour(update_status, on_text_chunk, trace):
        await on_text_chunk("Partial", False)
        started.set()
        await asyncio.sleep(10)

    monkeypatch.setattr(agent_runner, "_agents", lambda: fake_agents(behaviour))
    bot = FakeBot()
    ctx = JobContext(Job(user_id=USER, label="text", run=None))
    task = asyncio.create_task(run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi", ctx=ctx))
    await asyncio.wait_for(started.wait(), 5)
    ctx.cancel_reason = "shutdown"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bot.of("send_rich_message") == []
    assert bot.of("edit_message_text")[-1]["rich_message"].blocks[0].text == "🛑 Cancelled."


# ---- handlers -------------------------------------------------------------------
class FakeCallback:
    def __init__(self, data):
        self.data = data
        self.message = None  # not a Message: no markup cleanup attempted
        self.answers: list[str] = []

    async def answer(self, text=None, **kw):
        self.answers.append(text)


async def _running(monkeypatch):
    """A QueueManager in place of the handlers' one, with a long job running and one queued for USER."""
    qm = QueueManager(max_concurrent_turns=2, turn_timeout=30)
    monkeypatch.setattr(messages, "queue", qm)
    events: list[str] = []

    async def long_job(ctx):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            events.append(f"cancelled:{ctx.cancel_reason}")
            raise

    async def queued(ctx):
        events.append("queued-ran")

    job = Job(user_id=USER, label="long", run=long_job)
    await qm.submit(job)
    await qm.submit(Job(user_id=USER, label="queued", run=queued))
    await asyncio.sleep(0.02)
    return qm, job, events


async def test_status_stop_button_cancels_its_own_job_only(monkeypatch):
    qm, job, events = await _running(monkeypatch)
    for data in ("queue_stop_zz", "queue_stop_", f"queue_stop_{format(job.id ^ 1, 'x')}"):
        cb = FakeCallback(data)
        await messages.stop_button(cb, user_id=USER)
        assert cb.answers == ["This task has already finished."]
    assert events == [] and qm.status(USER).current is job

    cb = FakeCallback(f"queue_stop_{job.token}")
    await messages.stop_button(cb, user_id=USER)
    assert cb.answers == ["🛑 Stopping…"]
    await asyncio.sleep(0.05)
    assert events == ["cancelled:user", "queued-ran"]
    await qm.shutdown()


async def test_draft_stop_update_cancels_the_job_with_that_draft_id(monkeypatch):
    qm, job, events = await _running(monkeypatch)
    chat = Chat(id=42, type="private")
    await messages.draft_stopped(MessageGenerationStopped(chat=chat, draft_id=job.id ^ 1), user_id=USER)
    await asyncio.sleep(0.02)
    assert events == []
    await messages.draft_stopped(MessageGenerationStopped(chat=chat, draft_id=job.id), user_id=USER)
    await asyncio.sleep(0.05)
    assert events == ["cancelled:user", "queued-ran"]
    await qm.shutdown()


def test_dispatcher_polls_for_draft_stop_updates(tmp_path):
    """bot/app.py polls with allowed_updates=dp.resolve_used_update_types().

    Run in a fresh interpreter: build_root_router() attaches module-level routers, which
    aiogram allows only once per process.
    """
    code = (
        "from aiogram import Dispatcher\n"
        "from bot.handlers import build_root_router\n"
        "dp = Dispatcher()\n"
        "dp.include_router(build_root_router())\n"
        "print(','.join(dp.resolve_used_update_types()))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(APP)}
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"):
        env.setdefault(key, "x")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    used = out.stdout.strip().splitlines()[-1].split(",")
    assert {"message", "callback_query", "stopped_message_generation"} <= set(used)


# ---- regressions: a second Stop, a Stop during delivery, stale buttons -----------
async def test_a_second_stop_does_not_interrupt_the_first_ones_cleanup():
    qm = QueueManager(max_concurrent_turns=2, turn_timeout=30)
    events: list[str] = []
    started = asyncio.Event()
    cleaning = asyncio.Event()

    async def job_fn(ctx):
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cleaning.set()
            await asyncio.sleep(0.05)  # e.g. sending the partial answer, then "🛑 Stopped."
            events.append("cleaned-up")
            raise

    job = Job(user_id=USER, label="long", run=job_fn)
    await qm.submit(job)
    await asyncio.wait_for(started.wait(), 5)
    assert await qm.cancel_current(USER, job.id) is True
    await asyncio.wait_for(cleaning.wait(), 5)
    assert await qm.cancel_current(USER, job.id) is True  # the draft's Stop after the status's Stop
    for _ in range(100):
        if not qm.is_busy(USER):
            break
        await asyncio.sleep(0.02)
    assert events == ["cleaned-up"]
    await qm.shutdown()


async def test_stop_after_the_answer_is_complete_does_not_cut_the_delivery(monkeypatch):
    """Once the agent returned (answer in memory), Stop no longer applies: no "not saved" partial."""
    tts_started = asyncio.Event()
    release = asyncio.Event()

    async def behaviour(update_status, on_text_chunk, trace):
        await on_text_chunk("Partial", False)
        return "Full answer", []

    async def slow_tts(text, voice_id):
        tts_started.set()
        await release.wait()
        return b"audio"

    voice = ModuleType("voice")
    voice.VoiceManager = lambda: SimpleNamespace(get_user_voice=lambda uid: "v")
    monkeypatch.setitem(sys.modules, "voice", voice)
    monkeypatch.setattr(agent_runner, "_agents", lambda: fake_agents(behaviour))
    monkeypatch.setattr(agent_runner, "synthesize_speech", slow_tts)
    bot = FakeBot()

    async def send_voice(**kw):
        bot.calls.append(("send_voice", kw))
        return bot._msg()

    bot.send_voice = send_voice
    qm = QueueManager(max_concurrent_turns=2, turn_timeout=30)
    job = Job(user_id=USER, label="voice", run=lambda ctx: run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi", ctx=ctx, speak=True))
    await qm.submit(job)
    await asyncio.wait_for(tts_started.wait(), 5)

    assert await qm.cancel_current(USER, job.id) is False  # too late: the answer exists
    audio_status = bot.of("edit_message_text")[-1]
    assert not any(d.startswith("queue_stop_") for d in stop_rows(audio_status))  # the button is gone
    release.set()
    for _ in range(100):
        if not qm.is_busy(USER):
            break
        await asyncio.sleep(0.02)
    sent = [kw["rich_message"].markdown for kw in bot.of("send_rich_message")]
    assert sent == ["Full answer"] and bot.of("send_voice")
    await qm.shutdown()


async def test_user_cancel_after_the_answer_sends_no_partial(monkeypatch):
    """/cancel during delivery: the streamed part is not resent as an unsaved partial answer."""
    tts_started = asyncio.Event()

    async def behaviour(update_status, on_text_chunk, trace):
        await on_text_chunk("Partial", False)
        return "Full answer", []

    async def slow_tts(text, voice_id):
        tts_started.set()
        await asyncio.sleep(10)

    voice = ModuleType("voice")
    voice.VoiceManager = lambda: SimpleNamespace(get_user_voice=lambda uid: "v")
    monkeypatch.setitem(sys.modules, "voice", voice)
    monkeypatch.setattr(agent_runner, "_agents", lambda: fake_agents(behaviour))
    monkeypatch.setattr(agent_runner, "synthesize_speech", slow_tts)
    bot = FakeBot()
    ctx = JobContext(Job(user_id=USER, label="voice", run=None))
    task = asyncio.create_task(run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi", ctx=ctx, speak=True))
    await asyncio.wait_for(tts_started.wait(), 5)
    ctx.cancel_reason = "user"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(STOPPED_NOTE not in kw["rich_message"].markdown for kw in bot.of("send_rich_message"))
    assert bot.of("edit_message_text")[-1]["rich_message"].blocks[0].text == "🛑 Stopped."


def _message_with(markup):
    from datetime import datetime

    from aiogram.types import Message

    return Message(message_id=7, date=datetime.now(), chat=Chat(id=42, type="private"), text="x", reply_markup=markup)


async def test_stale_stop_press_leaves_other_keyboards_alone(monkeypatch):
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

    qm = QueueManager(max_concurrent_turns=2, turn_timeout=30)
    monkeypatch.setattr(messages, "queue", qm)
    removed: list[int] = []

    async def edit_reply_markup(self, **kw):
        removed.append(self.message_id)

    monkeypatch.setattr(Message, "edit_reply_markup", edit_reply_markup)
    data = "queue_stop_1a2b"

    # A legacy answer (the old status) now carrying the turn buttons: not touched.
    turn_kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔁 Retry", callback_data="turn_retry_0123456789ab")]])
    cb = FakeCallback(data)
    cb.message = _message_with(turn_kb)
    await messages.stop_button(cb, user_id=USER)
    assert cb.answers == ["This task has already finished."] and removed == []

    # A leftover status with just its Stop button: cleaned up.
    stop_kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🛑 Stop", callback_data=data)]])
    cb = FakeCallback(data)
    cb.message = _message_with(stop_kb)
    await messages.stop_button(cb, user_id=USER)
    assert removed == [7]
