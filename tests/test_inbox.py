"""Messages sent while a turn works: the running turn's inbox (bot/inbox.py) and the handlers around it.

A message of the same chat and topic joins the running turn (👀, no busy notice) and is prepared
in the background; the agent drains it at a tool boundary (✍); what it did not read becomes the
next turns, in arrival order, on the prompt already prepared. Also: the queue's arrival order,
the thread rule, /cancel vs Stop, the turn record, the quiet-marker override, typed answers to a
waiting question and follow-up buttons.
"""
import asyncio
import io
import shutil
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Message
from PIL import Image

import bot.agent_runner as agent_runner
import bot.streaming as streaming
from bot import background, inbox, media, questions, rich, turn_actions
from bot.handlers import messages
from bot.handlers import turn_actions_ui as ui
from bot.inbox import Prepared
from bot.queue import Job, QueueManager, next_seq
from bot.sender import ChatSender

USER = "42"
CHAT = {"id": 42, "type": "private"}
TOPIC_CHAT = {"id": 42, "type": "supergroup", "is_forum": True}
ALICE = {"id": 42, "is_bot": False, "first_name": "Alice"}
PHOTO = [{"file_id": "p1", "file_unique_id": "pu", "width": 10, "height": 10}]


def msg(message_id: int = 10, topic: int | None = None, **fields) -> Message:
    extra = {"message_thread_id": topic, "is_topic_message": True} if topic else {}
    chat = TOPIC_CHAT if topic else CHAT
    return Message.model_validate({"message_id": message_id, "date": 1760788860, "chat": chat, "from": ALICE, **extra, **fields})


def png_bytes(width: int = 40, height: int = 30) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def pdf_bytes(pages: int = 1) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class FakeBot:
    """Records Bot API calls; ``download`` writes the payload registered for the file id."""

    id = 777

    def __init__(self, payloads: dict[str, bytes] | None = None):
        self.payloads = payloads or {}
        self.calls: list[tuple[str, dict]] = []
        self.downloads: list[str] = []
        self._id = 500

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
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        return SimpleNamespace(message_id=kw.get("message_id"))

    async def edit_message_reply_markup(self, **kw):
        self.calls.append(("edit_message_reply_markup", kw))
        return True

    async def delete_message(self, **kw):
        self.calls.append(("delete_message", kw))
        return True

    async def send_document(self, **kw):
        self.calls.append(("send_document", kw))
        return self._msg()

    async def set_message_reaction(self, **kw):
        self.calls.append(("reaction", kw))
        return True

    async def download(self, file, destination=None, **kw):
        self.downloads.append(file)
        with open(destination, "wb") as f:
            f.write(self.payloads.get(file, b"data"))

    def of(self, name):
        return [kw for n, kw in self.calls if n == name]

    def reactions(self) -> list[tuple[int, str]]:
        return [(kw["message_id"], kw["reaction"][0].emoji) for kw in self.of("reaction")]

    def texts(self) -> list[str]:
        out = []
        for name, kw in self.calls:
            if name in ("send_message", "edit_message_text") and kw.get("text"):
                out.append(kw["text"])
            message = kw.get("rich_message")
            if message is not None and getattr(message, "markdown", None):
                out.append(message.markdown)
        return out


class _Stats:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        if name == "get_user_action_count":
            return lambda *a, **k: 0
        return lambda *a, **k: self.calls.append((name, a))


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    stats = ModuleType("stats")
    stats.stats_tracker = _Stats()
    monkeypatch.setitem(sys.modules, "stats", stats)
    monkeypatch.setattr(streaming, "config", SimpleNamespace(streaming_enabled=False))
    monkeypatch.setattr(inbox, "DRAIN_WAIT", 2.0)
    rich.reset()
    inbox.reset()
    background.reset()
    questions.reset()
    yield stats.stats_tracker
    inbox.reset()
    background.reset()


@pytest.fixture
async def qm(monkeypatch):
    """A fresh queue for the handlers; busy notices are recorded in ``qm.notices`` (message ids)."""
    queue = QueueManager(max_concurrent_turns=4, turn_timeout=30)
    monkeypatch.setattr(messages, "queue", queue)
    queue.notices = []

    async def notify_busy(message, result):
        queue.notices.append(message.message_id)

    monkeypatch.setattr(messages, "notify_busy", notify_busy)
    yield queue
    await queue.shutdown()


@pytest.fixture
def quiet_media(monkeypatch):
    """No ffmpeg, Whisper, PIL conversions or vision calls; counts the transcriptions."""
    transcribe = AsyncMock(return_value="call mom tonight")
    monkeypatch.setattr(media, "convert_audio", AsyncMock())
    monkeypatch.setattr(media, "transcribe_audio", transcribe)
    monkeypatch.setattr(media, "describe_document", AsyncMock(return_value="quarterly numbers"))
    monkeypatch.setattr(media, "prepare_downloaded_image_for_vision", lambda src, dst: shutil.copy(src, dst))
    monkeypatch.setattr(media, "save_vision_copy", lambda src, dst, max_px=None: (shutil.copy(src, dst), (1, 1))[1])
    return SimpleNamespace(transcribe=transcribe)


@pytest.fixture
def later_turns(monkeypatch) -> list[dict]:
    """``messages.run_turn`` (the turns queued for leftovers and new messages) records its arguments."""
    turns: list[dict] = []

    async def fake(**kw):
        turns.append(kw)
        return None

    monkeypatch.setattr(messages, "run_turn", fake)
    return turns


def use_agent(monkeypatch, behaviour, *, live: bool = True) -> dict:
    """A ChainOfThoughtAgent stand-in running ``behaviour(kw, prompt, update_status)``; ``kw`` holds its
    constructor arguments (inbox, background...). ``live=False``: a constructor without those."""
    seen: dict = {}

    async def generate(self, prompt, update_status, on_text_chunk=None, trace=None):
        seen["prompt"] = prompt
        return await behaviour(seen, prompt, update_status)

    if live:
        class Agent:
            def __init__(self, **kw):
                seen.update(kw)
                self.agent = SimpleNamespace(limit_reached=False)

            generate_response = generate
    else:
        class Agent:
            def __init__(self, model_type=None, model=None, user_id=None, sender=None, user_settings=None, message_thread_id=None):
                seen.update(user_id=user_id)
                self.agent = SimpleNamespace(limit_reached=False)

            generate_response = generate

    monkeypatch.setattr(agent_runner, "_agents", lambda: SimpleNamespace(ChainOfThoughtAgent=Agent))
    return seen


def turn_job(bot, prompt="first", *, thread_id=None, quiet_marker=None) -> Job:
    """A queued turn running the real run_turn (which opens the inbox)."""
    async def run(ctx):
        return await agent_runner.run_turn(
            bot=bot, user_id=USER, chat_id=42, prompt=prompt, thread_id=thread_id, ctx=ctx, quiet_marker=quiet_marker,
        )

    return Job(user_id=USER, label="text message", run=run, thread_id=thread_id, from_user=True)


async def until(predicate, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


async def idle(qm: QueueManager):
    await until(lambda: not qm.is_busy(USER))


def ready_item(text: str, *, seq=None, thread_id=None, kind=inbox.USER, message_id=None, bot=None) -> inbox.InboxItem:
    value = Prepared(prompt=text) if kind == inbox.USER else [{"type": "text", "text": text}]
    return inbox.InboxItem(
        kind=kind, future=inbox.ready_future(value), thread_id=thread_id, bot=bot, chat_id=42, message_id=message_id,
        **({"seq": seq} if seq is not None else {}),
    )


def texts_of(blocks: list[dict]) -> list[str]:
    return [b["text"] for b in blocks if b.get("type") == "text"]


# ---- the inbox itself ---------------------------------------------------------------------------
async def test_a_preparing_item_keeps_its_place_and_drain_waits_for_it_only_so_long(monkeypatch):
    monkeypatch.setattr(inbox, "DRAIN_WAIT", 0.2)
    box = inbox.open_inbox(USER, None)
    slow = asyncio.get_running_loop().create_future()
    box.deliver(ready_item("one"))
    box.deliver(inbox.InboxItem(kind=inbox.USER, future=slow))
    box.deliver(ready_item("three"))
    assert box.has_user_messages() and box.has_pending()

    first = await box.drain()  # waits DRAIN_WAIT for the second, then stops: the third must not overtake it
    assert len(first) == 1 and "one" in first[0]["text"]
    assert box.has_pending() and box.drained_user_texts == ["one"]

    asyncio.get_running_loop().call_later(0.05, slow.set_result, Prepared(prompt="two"))
    rest = await box.drain()  # ready within DRAIN_WAIT: both come, in order
    assert ["two" in t for t in texts_of(rest)] == [True, False] and "three" in texts_of(rest)[1]
    assert box.drained_user_texts == ["one", "two", "three"] and not box.has_pending()
    assert await box.drain() == []


async def test_user_message_format_and_media_blocks(tmp_path):
    images = tmp_path / "data" / USER / "images"
    images.mkdir(parents=True)
    (images / "x_v.jpg").write_bytes(png_bytes())
    box = inbox.open_inbox(USER, None, base=media.user_dir(USER))
    prompt = [{"type": "text", "text": "[Replying to ...]\n\nlook at this"},
              {"type": "media_ref", "kind": "image", "path": f"data/{USER}/images/x_v.jpg", "title": "Image 1"}]
    from datetime import datetime, timezone

    item = inbox.InboxItem(kind=inbox.USER, future=inbox.ready_future(Prepared(prompt=prompt)),
                           sent_at=datetime(2026, 10, 18, 12, 1, 30, tzinfo=timezone.utc))
    box.deliver(item)
    blocks = await box.drain()
    assert blocks[0] == {"type": "text", "text": (
        '<user_message sent_during_turn="true" time="2026-10-18 12:01 UTC">\n'
        "[Replying to ...]\n\nlook at this\n</user_message>")}
    assert blocks[1]["type"] == "image" and blocks[1]["source"]["type"] == "base64"
    assert box.drained_user_texts == [f"[Replying to ...]\n\nlook at this\n\n[image: data/{USER}/images/x_v.jpg]"]
    assert box.drained_record_blocks == [
        {"type": "text", "text": f"{inbox.SENT_WHILE_WORKING}\n[Replying to ...]\n\nlook at this"}, prompt[1],
    ]


async def test_an_item_that_arrived_earlier_goes_first_and_a_failed_one_is_a_short_note():
    box = inbox.open_inbox(USER, None)
    seq_early = next_seq()
    box.deliver(ready_item("later"))
    box.deliver(inbox.InboxItem(kind=inbox.USER, seq=seq_early, future=inbox.ready_future(
        Prepared(prompt=None, kind="voice message", reason="it could not be transcribed"))))
    blocks = await box.drain()
    assert "(the user sent a voice message that could not be processed: it could not be transcribed)" in blocks[0]["text"]
    assert "later" in blocks[1]["text"]


async def test_wait_reports_what_can_be_drained():
    box = inbox.open_inbox(USER, None)
    assert await box.wait(0.05) == "timeout"
    preparing = asyncio.get_running_loop().create_future()
    box.deliver(inbox.InboxItem(kind=inbox.USER, future=preparing))
    assert not box.has_user_messages() and box.has_pending()  # pending, but nothing to read yet
    assert await box.wait(0.05) == "timeout"  # not ready yet: a drain would return nothing
    asyncio.get_running_loop().call_later(0.02, preparing.set_result, Prepared(prompt="hi"))
    assert await box.wait(2) == "message"
    assert box.has_user_messages()
    await box.drain()
    asyncio.get_running_loop().call_later(0.02, box.deliver, ready_item("done", kind=inbox.TASK))
    assert await box.wait(2) == "task"
    assert not box.has_user_messages() and box.has_pending()
    assert await box.drain() == [{"type": "text", "text": "done"}]


async def test_close_hands_back_leftovers_and_stops_accepting():
    box = inbox.open_inbox(USER, 5)
    assert inbox.accepts(USER, 5) and not inbox.accepts(USER, None) and not inbox.accepts(USER, 9)
    requeued: list[str] = []
    item = ready_item("unread", thread_id=5)
    item.requeue = lambda: requeued.append("unread")
    box.deliver(item)
    inbox.finish(box)
    assert requeued == ["unread"] and box.closed and inbox.current(USER) is None
    assert not box.deliver(ready_item("late")) and await box.drain() == [] and await box.wait(5) == "timeout"
    inbox.finish(box)  # idempotent
    assert requeued == ["unread"]


async def test_a_second_open_does_not_steal_the_first_inbox():
    first = inbox.open_inbox(USER, None)
    second = inbox.open_inbox(USER, None)
    assert inbox.current(USER) is first
    second.close()
    assert inbox.current(USER) is first


# ---- the queue keeps arrival order ----------------------------------------------------------------------
async def test_a_job_that_arrived_earlier_is_queued_before_later_ones():
    queue = QueueManager(max_concurrent_turns=2, turn_timeout=5)
    order: list[str] = []
    gate = asyncio.Event()

    async def running(ctx):
        await gate.wait()

    def job(name, seq=None, **kw):
        async def run(ctx):
            order.append(name)
        return Job(user_id=USER, label=name, run=run, **({"seq": seq} if seq is not None else {}), **kw)

    early = next_seq()
    await queue.submit(Job(user_id=USER, label="running", run=running))
    await until(lambda: queue.status(USER).current is not None)
    queue.submit_nowait(job("b", thread_id=3, from_user=True))
    result = queue.submit_nowait(job("a", seq=early))
    assert result.was_busy and result.position == 1  # right behind the running job
    assert queue.has_pending_user_jobs(USER, 3) and not queue.has_pending_user_jobs(USER, None)
    gate.set()
    await until(lambda: not queue.is_busy(USER))
    assert order == ["a", "b"]
    await queue.shutdown()


# ---- handlers: joining the running turn ----------------------------------------------------------------------
async def test_a_text_joins_the_running_turn_with_reactions_and_no_busy_notice(monkeypatch, qm, env):
    started, gate, drained = asyncio.Event(), asyncio.Event(), {}

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        drained["blocks"] = await kw["inbox"].drain()
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    await messages.on_text(msg(11, text="also check the weather"), bot, USER)
    assert qm.status(USER).pending == 0 and bot.reactions() == [(11, "👀")]
    assert qm.notices == []  # no busy notice
    assert ("track_message_received", (USER, "text")) in env.calls
    gate.set()
    await idle(qm)
    await inbox.flush_reactions()
    assert texts_of(drained["blocks"]) == [
        '<user_message sent_during_turn="true" time="2025-10-18 12:01 UTC">\nalso check the weather\n</user_message>'
    ]
    assert bot.reactions() == [(11, "👀"), (11, "✍")]


async def test_only_the_running_turns_topic_joins_it(monkeypatch, qm, later_turns):
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot, thread_id=5))
    await started.wait()
    await messages.on_text(msg(11, topic=9, text="other topic"), bot, USER)
    assert qm.status(USER).pending == 1 and bot.reactions() == [] and qm.notices == [11]  # queued, busy notice
    await messages.on_text(msg(12, topic=5, text="same topic"), bot, USER)
    assert qm.status(USER).pending == 1 and bot.reactions() == [(12, "👀")]
    gate.set()
    await idle(qm)
    # "same topic" was sent after "other topic" but before it ran: arrival order holds
    assert [t["prompt"] for t in later_turns] == ["other topic", "same topic"]
    assert [t["thread_id"] for t in later_turns] == [9, 5]


async def test_an_earlier_queued_message_of_the_topic_keeps_later_ones_queued(monkeypatch, qm, later_turns):
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    queued = Job(user_id=USER, label="retry", run=lambda ctx: asyncio.sleep(0), from_user=True)
    await qm.submit(queued)  # e.g. a Retry pressed earlier
    await messages.on_text(msg(11, text="after it"), bot, USER)
    assert qm.status(USER).pending == 2 and bot.reactions() == []
    gate.set()
    await idle(qm)
    assert [t["prompt"] for t in later_turns] == ["after it"]


async def test_unread_messages_become_the_next_turns_in_order(monkeypatch, qm, later_turns, quiet_media):
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        return "Done.", []  # never drains

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    await messages.on_voice(msg(11, voice={"file_id": "v", "file_unique_id": "vu", "duration": 3}), bot, USER)
    await messages.on_text(msg(12, text="second"), bot, USER)
    await until(lambda: quiet_media.transcribe.await_count == 1)
    gate.set()
    await until(lambda: inbox.current(USER) is None)
    await messages.on_text(msg(13, text="sent after the turn"), bot, USER)
    await idle(qm)
    assert [t["prompt"] for t in later_turns] == ["call mom tonight", "second", "sent after the turn"]
    assert [t["reply_to_message_id"] for t in later_turns] == [11, 12, 13]
    assert later_turns[0]["speak"] is True and later_turns[1]["speak"] is False
    assert quiet_media.transcribe.await_count == 1 and bot.downloads == ["v"]  # prepared once, reused
    assert any("Transcription" in t for t in bot.texts())  # the transcription is shown once ready


async def test_a_leftover_still_preparing_is_awaited_not_redone(monkeypatch, qm, later_turns, quiet_media):
    monkeypatch.setattr(inbox, "DRAIN_WAIT", 0.1)
    release = asyncio.Event()

    async def slow_transcription(path):
        await release.wait()
        return "slow words"

    quiet_media.transcribe.side_effect = slow_transcription
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        assert await kw["inbox"].drain() == []  # still preparing after DRAIN_WAIT: nothing yet
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    await messages.on_voice(msg(11, voice={"file_id": "v", "file_unique_id": "vu", "duration": 3}), bot, USER)
    await messages.on_text(msg(12, text="behind the voice"), bot, USER)
    gate.set()
    await until(lambda: inbox.current(USER) is None)
    await asyncio.sleep(0.05)
    assert later_turns == []  # the voice turn waits for its transcription; the text waits behind it
    release.set()
    await idle(qm)
    assert [t["prompt"] for t in later_turns] == ["slow words", "behind the voice"]
    assert quiet_media.transcribe.await_count == 1


async def test_photo_and_document_leftovers_reuse_their_files(monkeypatch, qm, later_turns, quiet_media):
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot({"p1": png_bytes(), "d": pdf_bytes()})
    await qm.submit(turn_job(bot))
    await started.wait()
    await messages.on_photo(msg(11, photo=PHOTO, caption="what is it?"), bot, USER)
    await messages.on_document(msg(12, document={"file_id": "d", "file_unique_id": "du", "file_name": "Report.pdf"}), bot, USER)
    await asyncio.sleep(0.05)
    gate.set()
    await idle(qm)
    photo, document = (t["prompt"] for t in later_turns)
    assert photo[0]["text"].startswith("what is it?\n\nUser attached 1 image(s)") and photo[1]["kind"] == "image"
    assert document[1] == {"type": "media_ref", "kind": "document", "path": f"data/{USER}/documents/report.pdf", "title": "report.pdf"}
    assert sorted(bot.downloads) == ["d", "p1"]


async def test_an_album_flush_joins_the_running_turn_as_one_item(monkeypatch, qm, quiet_media, later_turns):
    monkeypatch.setattr(messages, "MEDIA_GROUP_TIMEOUT", 0.01)
    started, gate, drained = asyncio.Event(), asyncio.Event(), {}

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        await until(lambda: kw["inbox"].has_pending())
        drained["blocks"] = await kw["inbox"].drain()
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot({"p1": png_bytes(), "p2": png_bytes()})
    waiting = SimpleNamespace(message_id=900, edit_text=AsyncMock(), delete=AsyncMock())
    monkeypatch.setattr(messages, "answer_md", AsyncMock(return_value=waiting))
    await qm.submit(turn_job(bot))
    await started.wait()
    for i, file_id in enumerate(("p1", "p2")):
        photo = [{"file_id": file_id, "file_unique_id": f"{file_id}u", "width": 10, "height": 10}]
        await messages.on_photo(msg(20 + i, photo=photo, media_group_id="g1"), bot, USER)
    gate.set()
    await idle(qm)
    blocks = drained["blocks"]
    assert [b["type"] for b in blocks] == ["text", "image", "image"] and "User sent 2 photo(s)." in blocks[0]["text"]
    assert later_turns == [] and waiting.delete.await_count == 1  # the "waiting for images" line goes away
    assert bot.reactions()[0] == (20, "👀")


async def test_an_agent_that_takes_no_inbox_keeps_todays_queue(monkeypatch, qm, later_turns):
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        assert inbox.current(USER) is None
        await gate.wait()
        return "Done.", []

    seen = use_agent(monkeypatch, behaviour, live=False)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    await messages.on_text(msg(11, text="queued as before"), bot, USER)
    assert qm.status(USER).pending == 1 and qm.notices == [11]
    gate.set()
    await idle(qm)
    assert "inbox" not in seen and [t["prompt"] for t in later_turns] == ["queued as before"]


# ---- stopping ----------------------------------------------------------------------------------------
async def test_stop_requeues_the_unread_messages(monkeypatch, qm, later_turns):
    started = asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await asyncio.sleep(30)

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    job = turn_job(bot)
    await qm.submit(job)
    await started.wait()
    await messages.on_text(msg(11, text="unread"), bot, USER)
    assert await qm.cancel_current(USER, job.id) is True
    await idle(qm)
    assert [t["prompt"] for t in later_turns] == ["unread"]


async def test_the_turn_deadline_requeues_the_unread_messages(monkeypatch, qm, later_turns):
    started = asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await asyncio.sleep(30)

    use_agent(monkeypatch, behaviour)
    qm.turn_timeout = 0.3
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    await messages.on_text(msg(11, text="unread"), bot, USER)
    await idle(qm)
    assert [t["prompt"] for t in later_turns] == ["unread"]


async def test_slash_cancel_drops_the_unread_messages(monkeypatch, qm, later_turns):
    started = asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await asyncio.sleep(30)

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    await messages.on_text(msg(11, text="unread"), bot, USER)
    text = await messages._cancel_for(USER)
    assert text == "🛑 Stopped the current task and dropped 1 queued message(s)."
    await idle(qm)
    assert later_turns == [] and inbox.current(USER) is None
    assert await messages._cancel_for(USER) == "Nothing is running right now."


async def test_shutdown_drops_the_unread_messages(monkeypatch, qm, later_turns):
    started = asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await asyncio.sleep(30)

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    await messages.on_text(msg(11, text="unread"), bot, USER)
    await qm.shutdown()
    assert later_turns == [] and inbox.current(USER) is None


def slow_reading(monkeypatch) -> tuple[threading.Event, threading.Event]:
    """``_user_blocks`` (run in a thread by drain) blocks until released: (entered, release)."""
    entered, release = threading.Event(), threading.Event()
    original = inbox._user_blocks

    def slow(*args):
        entered.set()
        release.wait(5)
        return original(*args)

    monkeypatch.setattr(inbox, "_user_blocks", slow)
    return entered, release


async def test_a_stop_while_the_agent_reads_a_message_requeues_it(monkeypatch, qm, later_turns):
    entered, release = slow_reading(monkeypatch)  # e.g. materializing a photo or a PDF
    started = asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await until(lambda: kw["inbox"].has_pending())
        await kw["inbox"].drain()
        await asyncio.sleep(30)

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    job = turn_job(bot)
    await qm.submit(job)
    await started.wait()
    await messages.on_text(msg(11, text="unread"), bot, USER)
    await until(entered.is_set)  # drain is reading it in its thread now
    assert await qm.cancel_current(USER, job.id) is True
    release.set()
    await idle(qm)
    await inbox.flush_reactions()
    assert [t["prompt"] for t in later_turns] == ["unread"]
    assert bot.reactions() == [(11, "👀")]  # never marked read: it was not


async def test_a_cancelled_drain_leaves_the_message_in_its_place(monkeypatch):
    entered, release = slow_reading(monkeypatch)
    bot = FakeBot()
    box = inbox.open_inbox(USER, None)
    first, second = ready_item("hello", message_id=11, bot=bot), ready_item("world", message_id=12, bot=bot)
    box.deliver(first)
    box.deliver(second)
    task = asyncio.create_task(box.drain())
    await until(entered.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert box._items == [first, second] and box.drained_user_texts == [] and box.consumed_user_items == []
    blocks = await box.drain()  # the next drain reads both, once, in order
    assert ["hello" in t for t in texts_of(blocks)] == [True, False] and "world" in texts_of(blocks)[1]
    assert box.drained_user_texts == ["hello", "world"] and box.consumed_user_items == [first, second]
    await inbox.flush_reactions()
    assert bot.reactions() == [(11, "✍"), (12, "✍")]


async def test_an_inbox_closed_while_a_message_is_read_hands_it_back_only(monkeypatch):
    entered, release = slow_reading(monkeypatch)
    bot = FakeBot()
    box = inbox.open_inbox(USER, None)
    item = ready_item("hello", message_id=11, bot=bot)
    box.deliver(item)
    task = asyncio.create_task(box.drain())
    await until(entered.is_set)
    assert box.close() == [item]  # e.g. /cancel: the item is the leftovers' now
    release.set()
    assert await task == []
    await inbox.flush_reactions()
    assert box.drained_user_texts == [] and bot.reactions() == []


async def test_messages_sent_during_a_turn_are_prepared_two_at_a_time(monkeypatch, qm, later_turns, quiet_media):
    live = {"now": 0, "max": 0}

    async def transcribe(path):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.05)
        live["now"] -= 1
        return "words"

    quiet_media.transcribe.side_effect = transcribe
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    for i in range(10):
        await messages.on_voice(msg(100 + i, voice={"file_id": f"v{i}", "file_unique_id": f"u{i}", "duration": 3}), bot, USER)
    await asyncio.sleep(0.1)
    gate.set()  # the turn ends: the rest are leftovers awaited by their own turns, still two at a time
    await idle(qm)
    assert live["max"] == 2 and messages.PREPARE_SLOTS == 2
    assert quiet_media.transcribe.await_count == 10 and len(later_turns) == 10


async def test_two_documents_of_the_same_name_are_never_downloaded_at_once(monkeypatch, qm, later_turns, quiet_media):
    writers: dict[str, int] = {}
    peak = {"value": 0}

    class SlowBot(FakeBot):
        async def download(self, file, destination=None, **kw):
            writers[destination] = writers.get(destination, 0) + 1
            peak["value"] = max(peak["value"], writers[destination])
            await asyncio.sleep(0.05)
            await super().download(file, destination=destination)
            writers[destination] -= 1

    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    bot = SlowBot({"d1": pdf_bytes(1), "d2": pdf_bytes(2)})
    await qm.submit(turn_job(bot))
    await started.wait()
    for i, file_id in enumerate(("d1", "d2")):
        await messages.on_document(msg(20 + i, document={"file_id": file_id, "file_unique_id": f"{file_id}u", "file_name": "Report.pdf"}), bot, USER)
    gate.set()
    await idle(qm)
    assert bot.downloads == ["d1", "d2"] and peak["value"] == 1
    assert len(later_turns) == 2


async def test_slash_cancel_stops_preparing_the_messages_requeued_from_a_turn(monkeypatch, qm, later_turns, quiet_media):
    release = asyncio.Event()
    finished: list[str] = []

    async def transcribe(path):
        await release.wait()
        finished.append(path)
        return "words"

    quiet_media.transcribe.side_effect = transcribe
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        return "Done.", []  # never drains: both voice notes become turns of their own

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()
    for i in range(2):
        await messages.on_voice(msg(11 + i, voice={"file_id": f"v{i}", "file_unique_id": f"u{i}", "duration": 3}), bot, USER)
    gate.set()

    def leftovers_queued() -> bool:
        status = qm.status(USER)
        return inbox.current(USER) is None and status.current is not None and status.current.label == messages.VOICE_KIND and status.pending == 1

    await until(leftovers_queued)  # the first awaits its transcription, the second waits behind it
    assert await messages._cancel_for(USER) == "🛑 Stopped the current task and dropped 1 queued message(s)."
    release.set()
    await idle(qm)
    await asyncio.sleep(0.05)
    assert finished == [] and later_turns == []
    assert not [t for t in bot.texts() if "Transcription" in t]  # no echo after the cancel


# ---- the turn record and the quiet marker ----------------------------------------------------------------------
async def test_the_turn_record_carries_what_was_read_during_the_turn(monkeypatch, qm, quiet_media):
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        await until(lambda: kw["inbox"].has_pending())
        await kw["inbox"].drain()
        return "Answer.", []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot({"p1": png_bytes()})
    await qm.submit(turn_job(bot, prompt="first"))
    await started.wait()
    await messages.on_text(msg(11, text="and this"), bot, USER)
    await messages.on_photo(msg(12, photo=PHOTO, caption="and that"), bot, USER)
    gate.set()
    await idle(qm)
    (record_file,) = (p for p in (__import__("pathlib").Path("data") / USER / "turns").glob("*.json"))
    record = turn_actions.load(USER, record_file.stem)
    assert record.prompt.startswith(f"first\n\n{inbox.SENT_WHILE_WORKING}\nand this\n\nand that\n\nUser attached 1 image(s)")
    blocks = record.prompt_blocks
    assert blocks[0] == {"type": "text", "text": "first"}
    assert blocks[1] == {"type": "text", "text": f"{inbox.SENT_WHILE_WORKING}\nand this"}
    assert blocks[2]["text"].startswith("and that\n\nUser attached") and blocks[3]["type"] == "media_ref"
    assert blocks[3]["path"].endswith("_v.jpg")


async def test_a_failed_turn_record_keeps_them_too(monkeypatch, qm):
    started, gate = asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        await kw["inbox"].drain()
        raise RuntimeError("boom")

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    await qm.submit(turn_job(bot, prompt="first"))
    await started.wait()
    await messages.on_text(msg(11, text="and this"), bot, USER)
    gate.set()
    await idle(qm)
    (record_file,) = (__import__("pathlib").Path("data") / USER / "turns").glob("*.json")
    record = turn_actions.load(USER, record_file.stem)
    assert record.error and record.prompt == f"first\n\n{inbox.SENT_WHILE_WORKING}\nand this" and record.prompt_blocks is None


async def test_a_quiet_task_that_read_a_message_stays_quiet_and_requeues_it(monkeypatch, qm, later_turns):
    """The task's run says nothing; the message it read gets a turn of its own on its prepared
    prompt, ahead of a message the run did not read and of one sent after it."""
    started, gate, read, finish = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        await kw["inbox"].drain()
        read.set()
        await finish.wait()
        return agent_runner.QUIET_MARKER, []

    use_agent(monkeypatch, behaviour)
    bot = FakeBot()
    results = []

    async def run(ctx):
        results.append(await agent_runner.run_turn(bot=bot, user_id=USER, chat_id=42, prompt="task", ctx=ctx,
                                                   quiet_marker=agent_runner.QUIET_MARKER))

    await qm.submit(Job(user_id=USER, label="scheduled agent task", run=run))
    await started.wait()
    await messages.on_text(msg(11, text="are you there?"), bot, USER)
    gate.set()
    await read.wait()
    await messages.on_text(msg(12, text="and this"), bot, USER)  # joins the run, never read by it
    finish.set()
    await until(lambda: results)
    await messages.on_text(msg(13, text="sent after it"), bot, USER)
    await idle(qm)
    (result,) = results
    assert result.quiet is True and result.response == ""
    # the run sent nothing but its status line (which it removes): no answer of its own
    assert [kw["text"] for kw in bot.of("send_message")] == [agent_runner.THINKING] and bot.of("send_rich_message") == []
    assert [t["prompt"] for t in later_turns] == ["are you there?", "and this", "sent after it"]
    assert [t["reply_to_message_id"] for t in later_turns] == [11, 12, 13]


async def test_a_quiet_task_nobody_wrote_to_stays_quiet(monkeypatch):
    async def behaviour(kw, prompt, update_status):
        assert await kw["inbox"].drain() == []
        return agent_runner.QUIET_MARKER, []

    use_agent(monkeypatch, behaviour)
    result = await agent_runner.run_turn(bot=FakeBot(), user_id=USER, chat_id=42, prompt="task", quiet_marker=agent_runner.QUIET_MARKER)
    assert result.quiet is True


# ---- a waiting ask_user_choice question takes a typed answer ----------------------------------------------------
async def test_a_typed_text_answers_the_waiting_question_instead_of_joining_the_turn(qm, env, later_turns):
    box = inbox.open_inbox(USER, None)
    bot = FakeBot()
    task = asyncio.create_task(questions.ask(ChatSender(bot, 42, rich=False), "Which one?", ["A", "B"], 60))
    await until(lambda: questions.waiting_in(42, None) is not None)
    await messages.on_text(msg(11, topic=9, text="not here"), bot, USER)  # another topic: not an answer
    assert not task.done()
    await messages.on_text(msg(12, text="neither, use C"), bot, USER)
    answer = await task
    assert answer == "neither, use C" and answer.tapped is False
    await idle(qm)
    assert not box.has_pending() and [t["prompt"] for t in later_turns] == ["not here"]  # the other topic: a turn
    assert questions.final_text(SimpleNamespace(question="Which one?"), questions.TYPED_LINE) in bot.texts()
    assert [c for c in env.calls if c[0] == "track_message_received"] == [("track_message_received", (USER, "text"))] * 2


async def test_the_tool_reports_a_typed_answer_and_a_tap_as_before(monkeypatch):
    from agents.user_interactions import UserInteractions

    script = [questions.Answer("use C", tapped=False), questions.Answer("A", tapped=True)]

    async def ask(sender, question, options, timeout):
        return script.pop(0)

    monkeypatch.setattr(questions, "ask", ask)
    tools = UserInteractions(USER, ChatSender(FakeBot(), 42))
    args = {"question": "Which one?", "options": ["A", "B"]}
    assert await tools.execute_tool("ask_user_choice", args) == 'User answered in text instead of tapping a button: "use C"'
    assert await tools.execute_tool("ask_user_choice", args) == 'User chose: "A"'


async def test_a_tap_resolves_with_a_tapped_answer():
    bot = FakeBot()
    task = asyncio.create_task(questions.ask(ChatSender(bot, 42, thread_id=5), "Q?", ["A", "B"], 60))
    await until(lambda: questions.waiting_in(42, 5) is not None)
    assert questions.waiting_in(42, None) is None
    pending = questions.waiting_in(42, 5)
    assert questions.resolve(pending.id, 42, 1) == "B"
    answer = await task
    assert answer == "B" and answer.tapped is True


# ---- follow-up buttons ------------------------------------------------------------------------------------
async def test_a_follow_up_button_joins_the_running_turn_and_retry_queues(monkeypatch, qm, later_turns):
    started, gate, drained = asyncio.Event(), asyncio.Event(), {}

    async def behaviour(kw, prompt, update_status):
        started.set()
        await gate.wait()
        drained["blocks"] = await kw["inbox"].drain()
        return "Done.", []

    use_agent(monkeypatch, behaviour)
    monkeypatch.setattr(ui, "run_turn", messages.run_turn)  # the recorder
    record = turn_actions.TurnRecord(id=turn_actions.new_turn_id(), prompt="old", response="x", follow_ups=["Make it shorter"])
    await turn_actions.save(USER, record, None)
    bot = FakeBot()
    await qm.submit(turn_job(bot))
    await started.wait()

    class Callback:
        def __init__(self, data):
            self.data, self.message = data, msg(30)

        async def answer(self, *a, **k):
            return True

    await ui.on_follow_up(Callback(f"turn_fu_{record.id}_0"), bot=bot, user_id=USER, limit=None)
    assert qm.status(USER).pending == 0
    await ui.on_retry(Callback(f"turn_retry_{record.id}"), bot=bot, user_id=USER, limit=None)
    assert qm.status(USER).pending == 1
    gate.set()
    await idle(qm)
    assert "Make it shorter" in drained["blocks"][0]["text"]
    assert [t["prompt"] for t in later_turns] == ["old"]  # Retry queued and ran after the turn
    echo = next(kw for kw in bot.of("send_message") if kw["text"] == "💬 Make it shorter")
    assert echo and bot.reactions()[0][1] == "👀"
