"""Inbound content handlers: text, voice, video, audio, photos/albums, documents.

Handlers validate quickly, enqueue a job on the user's queue and return.
If the user already has a job running they get a busy notice;
the new message is processed as its own turn once the queue reaches it.

Messages sent together become ONE turn (the message window): a message that does not join a
running turn waits ``MESSAGE_WINDOW`` seconds (1.5) for the next one of its chat and topic, each
new one extends the wait (at most ``MESSAGE_WINDOW_MAX`` in all), and a photo album still
arriving holds it. A comment and the messages it forwards, or three short texts typed in a row,
are one request. While it waits the user sees "typing…" for the first message, then 👀 on each
and one status line counting them ("📥 Got 3 messages. …"), which becomes the turn's status.
When the window closes, one message is queued as the job it always was (``_run_<kind>``); two or
more were prepared meanwhile (``prepare_<kind>``) and run as one turn on one prompt
(``_run_prepared``, ``bot.inbox.combine``). Messages that arrive while a turn of their topic
runs join that turn instead (below); a window whose topic got a running turn meanwhile hands its
messages to that turn when it closes.

Unless that job is a turn of the same chat and forum topic whose agent is still working
(``inject_target``: its inbox is open and no earlier message of that topic is still
queued): then the message joins that turn instead (``bot.inbox``). It gets a 👀 reaction,
no busy notice, and is prepared in the background (``prepare_<kind>``: the download,
conversion and transcription a job would do, without the status messages) while the agent
goes on (at most ``PREPARE_SLOTS`` preparations per user at once; a document or audio file is
never written by two of them at the same time: ``_path_lock``); the agent reads it at its next
tool boundary (✍). Every kind of message is split
that way: ``prepare_<kind>`` builds a ``bot.inbox.Prepared`` (the prompt, the voice reply
flag, the error to show), and the ``_run_<kind>`` job wraps it in status messages and runs
the turn. The messages the turn did not read are queued together as ONE turn on the prompts
they already prepared (``_run_prepared``), at the first one's place. A plain text sent while an
``ask_user_choice`` question waits in that chat and topic answers the question instead.

``control_router`` (auth only) stops work: ``/cancel`` and the busy notice's
button stop the running job and clear the queue (the messages waiting in the running
turn's inbox or in a message window, and the user's background tasks too); the Stop button in a turn's
status (``queue_stop_<job token>``) and Telegram's Stop button on its streaming
draft (a ``stopped_message_generation`` update) stop only that job (unread messages
then become the next turns; background tasks keep running). It also keeps the names of
forum topics (``on_topic_service``: a topic created or renamed), and ``accept`` learns a
topic's name from the creation message every topic message points at (``remember_topic``),
so the agent's ``<context>`` block can name the topic (``bot.topics``).

Every job puts the reply / forward preamble (``bot.message_context``) in front of
its prompt, so a reply to an earlier message or a forwarded post reaches the agent
with what it refers to.

Photos and PDFs are not described beforehand: the image job and the document job
build a block prompt (``agents.content``: one text block, then an ``image_ref`` per
picture or a ``document_ref`` for a PDF within the limits) and the agent sees the
media itself. Text, docx, xlsx and json documents go in as
extracted text; a PDF too large to attach falls back to the description path.
"""
from __future__ import annotations

import asyncio
import bisect
import logging
import os
import shutil
import uuid
import weakref
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Awaitable, Callable

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, MessageGenerationStopped

from bot import background, inbox, media, questions
from bot.agent_runner import THINKING, run_turn
from bot.config import config
from bot.inbox import Prepared
from bot.message_context import with_reference
from bot.queue import STOP_CALLBACK_PREFIX, Job, JobContext, SubmitResult, job_id_from_token, next_seq
from bot.runtime import queue
from bot.sender import ChatSender
from bot.topics import topic_store
from bot.ui import answer_md, delete_quietly
from image_utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

router = Router(name="messages")
control_router = Router(name="queue-control")

MEDIA_GROUP_TIMEOUT = 10.0
CANCEL_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🛑 Stop current task", callback_data="queue_cancel")]])


TEXT_PROMPT_CAP = 30_000  # characters of a text document put in the prompt
EXTRACTED_TEXT_CAP = 60_000  # characters of extracted docx/xlsx/json text put in the prompt
IMAGE_EMBED_HINT = "The images are saved in your workspace; to show one inline in your answer, write ![caption](images/<file name>)."


def _stats():
    from stats import stats_tracker

    return stats_tracker


def _content():
    """``agents.content`` on first use (bot/ imports nothing from agents at module level)."""
    from agents import content

    return content


def _prompt_with_reference(text: str, media_blocks: list, message: Message, bot_id: int | None) -> str | list:
    """The job's prompt: the preamble + ``text`` as one string when there is no media (exactly
    ``with_reference``), else that string as the first text block followed by the media refs."""
    text = with_reference(text, message, bot_id)
    if not media_blocks:
        return text
    return [_content().text_block(text), *media_blocks]


def _thread_id(message: Message) -> int | None:
    return getattr(message, "message_thread_id", None) if getattr(message, "is_topic_message", None) else None


def _bot_id(bot: Bot) -> int | None:
    """``Bot.id`` (from the token); None for the bot stand-ins of the tests."""
    return getattr(bot, "id", None)


def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _md_safe(text: str) -> str:
    """Make free text safe inside legacy-Markdown bold/italic spans."""
    return text.replace("_", "-").replace("*", "").replace("`", "'")


async def notify_busy(message: Message, result: SubmitResult) -> None:
    current = result.current
    if current is not None:
        label = _md_safe(current.label)
        if current.started_at is None:
            text = f"⏳ I'm about to start on your previous request (*{label}*).\nI'll handle this message right after it finishes."
        else:
            step = _md_safe(current.progress or "starting")
            text = (
                f"⏳ I'm still working on your previous request (*{label}*, "
                f"{_format_elapsed(current.elapsed)} elapsed, step: _{step}_).\n"
                "I'll handle this message right after it finishes."
            )
    else:
        text = "⏳ Your previous messages are still queued. I'll handle this one in turn."
    ahead = result.position - 1
    if ahead > 0:
        text += f"\n📬 {ahead} more message(s) are queued ahead of it."
    text += "\nSend /cancel to stop the current task and clear the queue."
    try:
        await message.answer(text, parse_mode="Markdown", reply_markup=CANCEL_KEYBOARD)
    except Exception:  # noqa: BLE001
        await message.answer(text.replace("*", "").replace("_", ""), reply_markup=CANCEL_KEYBOARD)


async def submit(message: Message, job: Job) -> None:
    result = await queue.submit(job)
    if result.was_busy:
        await notify_busy(message, result)


# ---------------------------------------------------------------------------
# joining the running turn (bot.inbox)
# ---------------------------------------------------------------------------
Progress = Callable[[str | None, str | None], Awaitable[None]]  # (status text, job progress)

PREPARE_SLOTS = 2  # messages of one user prepared at once for the running turn (downloads, Whisper, vision)
_prepare_slots: dict[str, asyncio.Semaphore] = {}
_path_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()
_locks_loop: asyncio.AbstractEventLoop | None = None


def _same_loop() -> None:
    """asyncio locks and tasks belong to one event loop: a new loop (every test) starts with none."""
    global _locks_loop
    loop = asyncio.get_running_loop()
    if loop is not _locks_loop:
        _prepare_slots.clear()
        _path_locks.clear()
        _windows.clear()
        _albums_handing.clear()
        _locks_loop = loop


def _prepare_slot(user_id: str) -> asyncio.Semaphore:
    """The user's ``PREPARE_SLOTS`` slots: twenty voice notes sent during a turn are not transcribed all at once."""
    _same_loop()
    slot = _prepare_slots.get(str(user_id))
    if slot is None:
        slot = _prepare_slots[str(user_id)] = asyncio.Semaphore(PREPARE_SLOTS)
    return slot


def _path_lock(path: str) -> asyncio.Lock:
    """One writer per saved file (documents and audio keep their names): two uploads of the same
    name are downloaded one after the other, never into the same file at the same time."""
    _same_loop()
    key = os.path.abspath(path)
    lock = _path_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _path_locks[key] = lock
    return lock


async def _no_progress(status_text: str | None = None, step: str | None = None) -> None:
    """Preparing for the running turn: no status message, no job to report to."""
    return None


def _job_progress(sender: ChatSender, status: Message | None, ctx: JobContext | None) -> Progress:
    """Preparing as a queued job: the status message and the job's progress line follow along."""

    async def progress(status_text: str | None = None, step: str | None = None) -> None:
        if step and ctx is not None:
            ctx.set_progress(step)
        if status_text and status is not None:
            await sender.edit_text(status, status_text)

    return progress


def inject_target(user_id: str, thread_id: int | None) -> inbox.TurnInbox | None:
    """The running turn a new message of this topic joins, or None (queue it as today).

    The turn's inbox must be open for this topic and no earlier message (or button press) of
    the user in this topic may still be queued or waiting in its message window: that one is
    answered first, and this one after it (or with it).
    """
    target = inbox.current(user_id)
    if target is None or not target.accepts(thread_id):
        return None
    if _collecting(user_id, thread_id) is not None:
        return None
    pending = getattr(queue, "has_pending_user_jobs", None)
    if pending is not None and pending(user_id, thread_id):
        return None
    return target


async def _prepare_for_inbox(
    prepare: Callable[[], Awaitable[Prepared]], *, kind: str, label: str, error_text: str, echo: ChatSender | None,
    user_id: str | None = None,
) -> Prepared:
    """Run a ``prepare_<kind>`` for the running turn in one of the user's ``PREPARE_SLOTS``; never
    raises (a failure is a short note the agent reads). A leftover queued as its own turn awaits
    this same task (``_run_prepared``), so the slots cover that path too."""
    try:
        async with _prepare_slot(user_id or ""):
            prepared = await prepare()
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Could not prepare a %s for the running turn: %s", kind, e)
        return Prepared(prompt=None, kind=kind, label=label, error_text=error_text, reason=inbox.failure_reason(e))
    if prepared.ok and prepared.echo and echo is not None:
        try:
            await echo.send_text(prepared.echo)  # e.g. the transcription of a voice message
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not show what was prepared: %s", e)
    return prepared


def inject(
    target: inbox.TurnInbox, bot: Bot, user_id: str, limit: int | None, *, chat_id: int, thread_id: int | None,
    message_id: int | None, sent_at: Any = None, label: str, kind: str, error_text: str,
    prepare: Callable[[], Awaitable[Prepared]] | None = None, prepared: Prepared | None = None, seq: int | None = None,
    future: asyncio.Future | None = None,
) -> inbox.InboxItem:
    """Deliver a message to the running turn: ``prepared`` as it is, ``future`` (a preparation already
    running), or ``prepare`` run in the background.

    No suspension point between ``inject_target`` and this call, so nothing can slip in between.
    The leftovers (the turn ended before reading them) are queued together as one turn on the
    same prepared prompts (``_run_prepared``), at the first one's place.
    """
    if prepared is not None:
        future = inbox.ready_future(prepared)
    elif future is None:
        echo = ChatSender(bot, chat_id, thread_id, message_id)
        future = asyncio.create_task(
            _prepare_for_inbox(prepare, kind=kind, label=label, error_text=error_text, echo=echo, user_id=user_id),
            name=f"prepare-{user_id}-{label}",
        )
    item = inbox.InboxItem(
        kind=inbox.USER, future=future, thread_id=thread_id, label=label, bot=bot, chat_id=chat_id, message_id=message_id,
        sent_at=sent_at, **({"seq": seq} if seq is not None else {}),
    )
    item.requeue_group = partial(_requeue_group, bot, user_id, chat_id, thread_id, limit)
    target.deliver(item)
    return item


def _requeue_group(bot: Bot, user_id: str, chat_id: int, thread_id: int | None, limit: int | None, items: list[inbox.InboxItem]) -> None:
    """Messages the running turn did not read, as ONE queued turn at the first one's place (``inbox.requeue``)."""
    # /cancel dropping the queued job also stops their preparations (no echo, no transcription or
    # description after the cancel); a running one is stopped with the job (_run_prepared awaits it).
    first = items[0]
    queue.submit_nowait(Job(
        user_id=user_id, label=first.label if len(items) == 1 else inbox.combined_label(len(items)), seq=first.seq,
        thread_id=thread_id, from_user=True,
        run=partial(_run_prepared, bot, user_id, chat_id, thread_id, first.message_id, limit, [i.future for i in items]),
        on_drop=partial(inbox.drop, list(items)),
    ))


async def _run_prepared(
    bot: Bot, user_id: str, chat_id: int, thread_id: int | None, reply_to: int | None, limit: int | None,
    futures: list[asyncio.Future], ctx: JobContext, *, status: Message | None = None,
) -> None:
    """Messages prepared ahead as ONE turn of their own: what a running turn did not read, or the
    messages of a message window. Their preparations are reused (awaited when still running), their
    prompts put together (``inbox.combine``; one message is its own prompt). ``status`` (the
    window's count line) becomes the turn's status."""
    sender = ChatSender(bot, chat_id, thread_id, reply_to)
    ctx.set_progress("preparing")
    live = [f for f in futures if not f.cancelled()]
    try:
        # Usually done long ago. One gather: a cancel of this job cancels every preparation still
        # running at once (no echo, no transcription after a /cancel); one dropped elsewhere is skipped.
        results = await asyncio.gather(*live, return_exceptions=True)
    except asyncio.CancelledError:
        if status is not None:
            await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    prepared: list[Prepared] = []
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            continue  # that preparation was dropped, not this job
        if isinstance(result, BaseException):
            result = Prepared(prompt=None, reason=inbox.failure_reason(result))
        prepared.append(result if isinstance(result, Prepared) else Prepared(prompt=result))
    if not prepared:
        return
    combined = inbox.combine(prepared)
    if not combined.ok:
        text = combined.error_text or "❌ Could not process this message."
        if status is not None:
            await sender.edit_text(status, text, markdown=False)
        else:
            await sender.send_text(text, markdown=False)
        return
    await run_turn(
        bot=bot, user_id=user_id, chat_id=chat_id, prompt=combined.prompt, thread_id=thread_id,
        reply_to_message_id=reply_to, ctx=ctx, limit=limit, speak=combined.speak, status=status,
    )


async def accept(
    message: Message, bot: Bot, user_id: str, limit: int | None, *, label: str, kind: str, error_text: str,
    prepare: Callable[[], Awaitable[Prepared]], run: Callable[[JobContext], Awaitable[Any]], seq: int | None = None,
    window: bool = True, waiting: Message | None = None,
) -> None:
    """A content message: into the running turn of its topic (👀); else into the message window of its
    chat and topic (``window`` and ``MESSAGE_WINDOW`` on: messages sent together become one turn);
    else a queued job (busy notice as before). ``waiting``: an album's "waiting for images" line."""
    await remember_topic(message, user_id)
    thread_id = _thread_id(message)
    target = inject_target(user_id, thread_id)
    if target is not None:
        item = inject(
            target, bot, user_id, limit, chat_id=message.chat.id, thread_id=thread_id, message_id=message.message_id,
            sent_at=message.date, label=label, kind=kind, error_text=error_text, prepare=prepare, seq=seq,
        )
        await inbox.mark_seen(item)
        return
    if window and MESSAGE_WINDOW > 0:
        entry = _Entry(
            message=message, label=label, kind=kind, error_text=error_text, prepare=prepare, run=run,
            seq=seq if seq is not None else next_seq(), waiting=waiting,
        )
        await _gather(bot, user_id, limit, thread_id, entry)
        return
    job = Job(user_id=user_id, label=label, run=run, thread_id=thread_id, from_user=True, **({"seq": seq} if seq is not None else {}))
    await submit(message, job)


# ---------------------------------------------------------------------------
# messages sent together: one turn (the message window)
# ---------------------------------------------------------------------------
MESSAGE_WINDOW = config.message_window_seconds  # seconds a message waits for the next one (MESSAGE_WINDOW_SECONDS; 0 = off)
MESSAGE_WINDOW_MAX = 30.0  # the window never stays open longer, however often it is extended
WINDOW_STATUS = "📥 *Got {count} messages.* I'll read them all together."
WINDOW_CANCELLED = "🛑 Cancelled."


@dataclass(eq=False)
class _Entry:
    """One message waiting in a window: what ``accept`` would have queued or injected."""

    message: Message
    label: str
    kind: str
    error_text: str
    prepare: Callable[[], Awaitable[Prepared]]
    run: Callable[[JobContext], Awaitable[Any]]
    seq: int
    waiting: Message | None = None  # an album's "waiting for images" line (the count replaces it)
    future: asyncio.Future | None = None  # its preparation, started once the window holds two messages


@dataclass(eq=False)
class _Window:
    """The messages of one chat and topic that arrived together (``MESSAGE_WINDOW`` apart at most)."""

    bot: Bot
    user_id: str
    chat_id: int
    thread_id: int | None
    limit: int | None
    opened: float
    due: float
    entries: list[_Entry] = field(default_factory=list)  # by Telegram message id: the order they were sent in
    status: Message | None = None  # the count line, once there are two
    shown: int = 0  # the count it shows
    status_task: asyncio.Task | None = None
    timer: asyncio.Task | None = None
    closed: bool = False
    handed: bool = False  # its messages are queued or in a running turn: nothing may drop them any more


_windows: dict[tuple[str, int | None], _Window] = {}
_albums_handing: dict[tuple[int, int | None], int] = {}  # albums of a chat and topic being handed over (flushed)


def _collecting(user_id: str, thread_id: int | None) -> _Window | None:
    """The open message window of the user's topic, if any."""
    _same_loop()
    return _windows.get((str(user_id), thread_id))


def _album_arriving(chat_id: int, thread_id: int | None) -> bool:
    """A photo album of this chat and topic is still being collected or handed over: a window waits for it."""
    if _albums_handing.get((chat_id, thread_id)):
        return True
    return any(a.message.chat.id == chat_id and _thread_id(a.message) == thread_id for a in _albums.values())


def _start_preparing(window: _Window, entry: _Entry) -> asyncio.Task:
    echo = ChatSender(window.bot, window.chat_id, window.thread_id, entry.message.message_id)
    return asyncio.create_task(
        _prepare_for_inbox(
            entry.prepare, kind=entry.kind, label=entry.label, error_text=entry.error_text, echo=echo, user_id=window.user_id,
        ),
        name=f"prepare-{window.user_id}-{entry.label}",
    )


async def _gather(bot: Bot, user_id: str, limit: int | None, thread_id: int | None, entry: _Entry) -> None:
    """Put ``entry`` into the window of its chat and topic (opened when there is none) and extend it.

    The user sees "typing…" for the first message. From the second on, the window is one turn of
    several messages: each message gets 👀 and its preparation starts (download, transcription), and
    one status line counts them. Everything that changes the window happens before the first await.
    """
    loop = asyncio.get_running_loop()
    now = loop.time()
    key = (str(user_id), thread_id)
    window = _collecting(user_id, thread_id)
    if window is None:
        window = _windows[key] = _Window(
            bot=bot, user_id=str(user_id), chat_id=entry.message.chat.id, thread_id=thread_id, limit=limit, opened=now, due=now,
        )
        window.timer = asyncio.create_task(_window_timer(key, window), name=f"message-window-{user_id}")
    ids = [e.message.message_id for e in window.entries]
    window.entries.insert(bisect.bisect_right(ids, entry.message.message_id), entry)
    window.due = min(now + MESSAGE_WINDOW, window.opened + MESSAGE_WINDOW_MAX)
    if len(window.entries) == 1:
        await ChatSender(bot, window.chat_id, thread_id).typing()
        return
    fresh = [e for e in window.entries if e.future is None]  # the first message too, once the second arrives
    for e in fresh:
        e.future = _start_preparing(window, e)
    waiting = [e.waiting for e in fresh if e.waiting is not None]
    for e in fresh:
        e.waiting = None
    _show_count(window)
    await asyncio.gather(
        *(inbox.react(bot, window.chat_id, e.message.message_id, inbox.SEEN_REACTION) for e in fresh),
        *(delete_quietly(w) for w in waiting),  # the count says it now
    )


def _show_count(window: _Window) -> None:
    if window.status_task is None or window.status_task.done():
        window.status_task = asyncio.create_task(_update_count(window), name=f"message-window-status-{window.user_id}")


async def _update_count(window: _Window) -> None:
    """The count line follows the window: one Telegram call at a time, the latest count wins."""
    while not window.closed and window.shown != len(window.entries):
        count = len(window.entries)
        sender = ChatSender(window.bot, window.chat_id, window.thread_id, window.entries[0].message.message_id)
        text = WINDOW_STATUS.format(count=count)
        try:
            if window.status is None:
                window.status = await sender.send_text(text)
            else:
                await sender.edit_text(window.status, text)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the count is a courtesy
            logger.debug("Could not show the message count to %s: %s", window.user_id, e)
            return
        window.shown = count


async def _window_timer(key: tuple[str, int | None], window: _Window) -> None:
    """Close the window once nothing arrived for ``MESSAGE_WINDOW`` (or at ``MESSAGE_WINDOW_MAX``);
    an album of the same chat and topic still arriving keeps it open."""
    loop = asyncio.get_running_loop()
    deadline = window.opened + MESSAGE_WINDOW_MAX
    while True:
        delay = window.due - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
            continue
        if _album_arriving(window.chat_id, window.thread_id) and loop.time() < deadline:
            window.due = min(loop.time() + MESSAGE_WINDOW, deadline)
            continue
        break
    try:
        await _close_window(key, window)
    except Exception as e:  # noqa: BLE001 - e.g. the queue is closed (shutdown)
        if window.handed:
            logger.warning("After handing over the messages of %s: %s", window.user_id, e)
        else:
            logger.warning("Could not hand over the messages of %s (%d): %s", window.user_id, len(window.entries), e)
            _drop_window(window, notify=False)


async def _close_window(key: tuple[str, int | None], window: _Window) -> None:
    """The window closed: its messages join a turn of their topic that started meanwhile, or become ONE
    queued turn (a single message: exactly the job it would have been without the window)."""
    while window.status_task is not None and not window.status_task.done():
        await asyncio.wait({window.status_task})
    # From here on nothing is awaited until the messages are handed over: none can slip in between.
    if _windows.get(key) is window:
        del _windows[key]
    window.closed = True
    entries = list(window.entries)
    if not entries:
        return
    first = entries[0]
    seq = min(e.seq for e in entries)
    target = inject_target(window.user_id, window.thread_id)
    if target is not None:
        # A turn of this topic started while the window was open (it came right after an earlier
        # window): the messages join it like any message sent during a turn.
        items = [
            inject(
                target, window.bot, window.user_id, window.limit, chat_id=window.chat_id, thread_id=window.thread_id,
                message_id=e.message.message_id, sent_at=e.message.date, label=e.label, kind=e.kind,
                error_text=e.error_text, prepare=e.prepare, future=e.future, seq=e.seq,
            )
            for e in entries
        ]
        window.handed = True
        unseen = [item for item, e in zip(items, entries) if e.future is None]
        await asyncio.gather(
            *(inbox.mark_seen(item) for item in unseen),
            ChatSender(window.bot, window.chat_id, window.thread_id).delete([window.status]),  # the turn has its own status
            *(delete_quietly(e.waiting) for e in entries if e.waiting is not None),
        )
        return
    if len(entries) == 1:
        job = Job(user_id=window.user_id, label=first.label, run=first.run, thread_id=window.thread_id, from_user=True, seq=seq)
    else:
        job = Job(
            user_id=window.user_id, label=inbox.combined_label(len(entries)), thread_id=window.thread_id, from_user=True, seq=seq,
            run=partial(
                _run_prepared, window.bot, window.user_id, window.chat_id, window.thread_id, first.message.message_id,
                window.limit, [e.future for e in entries], status=window.status,
            ),
            on_drop=partial(_drop_window, window),
        )
    result = queue.submit_nowait(job)
    window.handed = True
    if result.was_busy:
        await notify_busy(first.message, result)


def _drop_window(window: _Window, notify: bool = True) -> int:
    """Throw a window's messages away (/cancel, or its queued turn dropped): their preparations stop,
    the count line says so. Never suspends; the number of messages."""
    window.closed = True
    for task in (window.timer, window.status_task):
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
    for e in window.entries:
        if e.future is not None and not e.future.done():
            e.future.cancel()
    leftovers = [e.waiting for e in window.entries if e.waiting is not None]
    if leftovers or (notify and window.status is not None):
        task = asyncio.create_task(_tidy_dropped(window, leftovers, notify))
        _tidying.add(task)
        task.add_done_callback(_tidying.discard)
    return len(window.entries)


_tidying: set[asyncio.Task] = set()


async def _tidy_dropped(window: _Window, waiting: list[Message], notify: bool) -> None:
    for message in waiting:
        await delete_quietly(message)
    if notify and window.status is not None:
        await ChatSender(window.bot, window.chat_id, window.thread_id).edit_text(window.status, WINDOW_CANCELLED, markdown=False)


def drop_windows(user_id: str) -> int:
    """/cancel: the user's open message windows (every topic) are dropped; the number of messages."""
    _same_loop()
    dropped = 0
    for key in [k for k in _windows if k[0] == str(user_id)]:
        dropped += _drop_window(_windows.pop(key))
    return dropped


# ---------------------------------------------------------------------------
# /cancel
# ---------------------------------------------------------------------------
async def _cancel_for(user_id: str) -> str:
    # Messages waiting in the running turn's inbox or in a message window are dropped with the
    # queue (not requeued when the turn stops), and the user's background tasks end without a
    # notification.
    dropped_unread = inbox.discard(user_id)
    dropped_unread += drop_windows(user_id)
    stopped_tasks = background.cancel_all(user_id)
    cancelled, dropped = await queue.cancel(user_id, reason="user")
    dropped += dropped_unread
    if not cancelled and not dropped and not stopped_tasks:
        return "Nothing is running right now."
    parts = []
    if cancelled:
        parts.append("stopped the current task")
    if dropped:
        parts.append(f"dropped {dropped} queued message(s)")
    if stopped_tasks:
        parts.append(f"cancelled {stopped_tasks} background task(s)")
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return "🛑 " + text.capitalize() + "."


@control_router.message(F.text, Command("cancel"))
async def cancel_command(message: Message, user_id: str) -> None:
    await message.answer(await _cancel_for(user_id))


@control_router.callback_query(F.data == "queue_cancel")
async def cancel_button(callback: CallbackQuery, user_id: str) -> None:
    text = await _cancel_for(user_id)
    await callback.answer(text, show_alert=False)
    if isinstance(callback.message, Message):
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Stop buttons of one turn: its status message and its streaming draft
# ---------------------------------------------------------------------------
STOP_STOPPING_TEXT = "🛑 Stopping…"
STOP_TOO_LATE_TEXT = "Too late to stop: the answer is already being sent."
STOP_FINISHED_TEXT = "This task has already finished."


def _only_button(markup: InlineKeyboardMarkup | None, data: str | None) -> bool:
    """True when ``markup`` is just the pressed button (a leftover Stop keyboard)."""
    buttons = [b for row in markup.inline_keyboard for b in row] if markup is not None else []
    return bool(buttons) and all(b.callback_data == data for b in buttons)


@control_router.callback_query(F.data.startswith(STOP_CALLBACK_PREFIX))
async def stop_button(callback: CallbackQuery, user_id: str) -> None:
    """The status message's Stop: cancel that turn only; queued messages still run."""
    job_id = job_id_from_token((callback.data or "")[len(STOP_CALLBACK_PREFIX):])
    cancelled = job_id is not None and await queue.cancel_current(user_id, job_id, reason="user")
    current = queue.status(user_id).current
    running = job_id is not None and current is not None and current.id == job_id
    if cancelled:
        notice = STOP_STOPPING_TEXT
    else:
        # Still running but refused: the agent is done and the answer is being delivered.
        notice = STOP_TOO_LATE_TEXT if running else STOP_FINISHED_TEXT
    try:
        await callback.answer(notice)
    except Exception as e:  # noqa: BLE001 - the query may have expired
        logger.debug("Could not answer the stop button: %s", e)
    if (
        not cancelled
        and job_id is not None
        and not running  # a still-running turn clears its own button
        and isinstance(callback.message, Message)
        and _only_button(callback.message.reply_markup, callback.data)
    ):
        # A status left behind by a turn that ended without its final edit (e.g. a restart).
        # Any other keyboard (the turn buttons on a legacy answer) is left alone.
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass


@control_router.stopped_message_generation()
async def draft_stopped(event: MessageGenerationStopped, user_id: str) -> None:
    """Telegram's Stop button on a streaming draft; the draft id is the job id (see run_turn)."""
    cancelled = await queue.cancel_current(user_id, job_id=event.draft_id, reason="user")
    logger.info("Draft %s stopped by %s: %s", event.draft_id, user_id, "turn cancelled" if cancelled else "no matching turn")


# ---------------------------------------------------------------------------
# location → time zone; the agent never sees locations
# ---------------------------------------------------------------------------
LOCATION_NO_FINDER = (
    "I can't tell the time zone from a location on this server.\n"
    "Please type it: /timezone Europe/Berlin (any city name works too)."
)


@control_router.message(F.location)
async def on_location(message: Message, user_id: str) -> None:
    """A shared location sets ``timezone.name`` (timezonefinder, optional; else the zone name is asked for)."""
    from bot import timezones
    from bot.handlers.commands import timezone_set_text
    from bot.settings import UserSettings

    location = message.location
    name = await asyncio.to_thread(timezones.zone_from_location, location.latitude, location.longitude)
    if name is None:
        await answer_md(message, LOCATION_NO_FINDER)
        return
    UserSettings(user_id).set("timezone", name, "name")
    await answer_md(message, timezone_set_text(name))


# ---------------------------------------------------------------------------
# forum topics: their names for the agent's <context> block (bot/topics.py)
# ---------------------------------------------------------------------------
async def remember_topic(message: Message, user_id: str) -> None:
    """Learn the name of the message's forum topic from the creation message it points at
    (``reply_to_message``, see ``message_context._is_topic_root``). Only fills a gap: after a
    rename that message still carries the old name. Best effort, never stops the message."""
    thread_id = _thread_id(message)
    if thread_id is None:
        return
    created = getattr(getattr(message, "reply_to_message", None), "forum_topic_created", None)
    if created is None:
        return
    try:
        await asyncio.to_thread(topic_store.learn, user_id, thread_id, created.name, implicit=bool(created.is_name_implicit))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not store the name of topic %s of %s: %s", thread_id, user_id, e)


@control_router.message(F.forum_topic_created | F.forum_topic_edited)
async def on_topic_service(message: Message, user_id: str) -> None:
    """A forum topic was created or renamed: keep its name. Nothing is answered, nothing reaches
    the agent and no quota is used; an edit that only changed the icon carries no name."""
    created = message.forum_topic_created
    event = created or message.forum_topic_edited
    thread_id = message.message_thread_id or (message.message_id if created is not None else None)
    if event is None or not event.name or thread_id is None:
        return
    implicit = bool(getattr(event, "is_name_implicit", None))  # a rename (ForumTopicEdited) is always chosen
    try:
        await asyncio.to_thread(topic_store.set, user_id, thread_id, event.name, implicit=implicit)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not store the name of topic %s of %s: %s", thread_id, user_id, e)


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------
TEXT_KIND = "text message"


async def prepare_text(bot: Bot, message: Message, user_id: str) -> Prepared:
    return Prepared(prompt=with_reference(message.text or "", message, _bot_id(bot)), kind=TEXT_KIND, label=TEXT_KIND)


async def _run_text(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    await run_turn(
        bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=with_reference(message.text or "", message, _bot_id(bot)),
        thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
    )


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    logger.info("Text message from %s", user_id)
    _stats().track_message_received(user_id, "text")
    thread_id = _thread_id(message)
    if questions.waiting_in(message.chat.id, thread_id):
        # An ask_user_choice question waits here: the text is its answer, not a new request.
        if await questions.answer_in_text(bot, message.chat.id, thread_id, message.text or ""):
            return
    await accept(
        message, bot, user_id, limit, label=TEXT_KIND, kind=TEXT_KIND, error_text="❌ Error processing text message",
        prepare=partial(prepare_text, bot, message, user_id), run=partial(_run_text, bot, message, user_id, limit),
    )


# ---------------------------------------------------------------------------
# voice
# ---------------------------------------------------------------------------
VOICE_KIND = "voice message"
VOICE_ERROR = "❌ Error processing voice message"


async def prepare_voice(bot: Bot, message: Message, user_id: str) -> Prepared:
    """Download, convert and transcribe; only the transcription reaches the agent (nothing is kept)."""
    media.ensure_temp_dirs()
    temp_ogg = os.path.join(media.TEMP_AUDIO, f"voice_{uuid.uuid4()}.oga")
    temp_mp3 = os.path.join(media.TEMP_AUDIO, f"voice_{uuid.uuid4()}.mp3")
    try:
        await bot.download(message.voice.file_id, destination=temp_ogg)
        await media.convert_audio(temp_ogg, temp_mp3, "ogg")
        transcription = await media.transcribe_audio(temp_mp3)
    finally:
        media.remove_files([temp_ogg, temp_mp3])
    if not transcription:
        return Prepared(
            prompt=None, kind=VOICE_KIND, label=VOICE_KIND, error_text="❌ Failed to transcribe audio",
            reason="it could not be transcribed",
        )
    shown = transcription[:512] + "..." if len(transcription) > 1500 else transcription
    bot_id = _bot_id(bot)
    return Prepared(
        prompt=with_reference(transcription, message, bot_id),
        inject=with_reference(f"(voice message, transcribed) {transcription}", message, bot_id),
        kind=VOICE_KIND, label=VOICE_KIND, speak=True, echo=f"🎙️ *Transcription:*\n{shown}",
    )


async def _run_voice(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    status = await sender.send_text("🎙️ *Transcribing audio...*")
    try:
        ctx.set_progress("transcribing voice")
        prepared = await prepare_voice(bot, message, user_id)
        if not prepared.ok:
            await sender.edit_text(status, prepared.error_text or VOICE_ERROR)
            return
        await sender.edit_text(status, prepared.echo or "🎙️")
        await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit, speak=True,
        )
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing voice message: %s", e)
        await sender.edit_text(status, VOICE_ERROR)


@router.message(F.voice)
async def on_voice(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "voice")
    await accept(
        message, bot, user_id, limit, label=VOICE_KIND, kind=VOICE_KIND, error_text=VOICE_ERROR,
        prepare=partial(prepare_voice, bot, message, user_id), run=partial(_run_voice, bot, message, user_id, limit),
    )


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
VIDEO_KIND = "video"
VIDEO_ERROR = "❌ Error processing video message"
VIDEO_MISSING = "❌ Could not process video message."


async def prepare_video(bot: Bot, message: Message, user_id: str, progress: Progress = _no_progress) -> Prepared:
    """Save the video, transcribe its audio and describe four frames (``gpt-6-luna``)."""
    is_video_note = message.video_note is not None
    video = message.video or message.video_note or message.document
    if not video:
        return Prepared(prompt=None, kind=VIDEO_KIND, label=VIDEO_KIND, error_text=VIDEO_MISSING, reason="the file is missing")
    caption = message.caption or ""
    media.ensure_temp_dirs()
    temp_video = os.path.join(media.TEMP_AUDIO, f"video_{uuid.uuid4()}.mp4")
    temp_mp3 = os.path.join(media.TEMP_AUDIO, f"video_audio_{uuid.uuid4()}.mp3")
    screenshots: list[str] = []
    try:
        await progress(None, "downloading video")
        await bot.download(video.file_id, destination=temp_video)
        saved_video = str(media.user_dir(user_id, "videos") / f"video_{uuid.uuid4()}.mp4")
        await asyncio.to_thread(shutil.copy, temp_video, saved_video)

        await progress("🎬 *Extracting audio and analyzing frames...*", "transcribing video")

        async def extract_audio() -> str | None:
            try:
                await media.convert_audio(temp_video, temp_mp3, "mp4")
            except Exception as e:  # noqa: BLE001
                logger.warning("No audio track or conversion failed: %s", e)
                return None
            return await media.transcribe_audio(temp_mp3)

        transcription, screenshots = await asyncio.gather(extract_audio(), media.extract_video_screenshots(temp_video))

        visual = ""
        saved_frames: list[str] = []
        if screenshots:
            images_dir = media.user_dir(user_id, "images")
            for sp in screenshots:
                dest = str(images_dir / f"video_frame_{uuid.uuid4()}.jpg")
                await asyncio.to_thread(shutil.copy, sp, dest)
                saved_frames.append(dest)
            await progress("🖼️ *Analyzing video frames...*", "describing frames")
            visual = await media.describe_video_screenshots(screenshots, transcription=(transcription or "") if is_video_note else "", caption=caption, user_id=user_id)

        if not (transcription or visual or saved_frames):
            return Prepared(
                prompt=None, kind=VIDEO_KIND, label=VIDEO_KIND, error_text="❌ Could not extract any content from video",
                reason="no audio or picture could be extracted",
            )

        display = "🎬 *Video analysis:*\n"
        if transcription:
            shown = transcription[:512] + "..." if len(transcription) > 1500 else transcription
            display += f"🎙️ *Audio:* {shown}\n"
        if visual:
            display += f"🖼️ *Visual:* {visual[:200]}..."
        if caption:
            display += f"\n📝 *Caption:* {caption}"
        if len(display) > 4000:
            display = display[:2048] + "..."

        parts = [f"User sent a video file. Saved to: {saved_video}"]
        if caption:
            parts.append(f"User caption: {caption}")
        if transcription:
            parts.append(f"Audio transcription from video: {transcription}")
        if visual:
            parts.append(f"Visual description of video: {visual}")
        if saved_frames:
            lines = []
            for i, sp in enumerate(saved_frames):
                label = media.FRAME_LABELS[i] if i < len(media.FRAME_LABELS) else f"frame {i + 1}"
                lines.append(f"  - Frame at {label}: {sp}")
            parts.append("Video screenshots saved:\n" + "\n".join(lines))
        prompt = with_reference("\n\n".join(parts), message, _bot_id(bot))
        return Prepared(prompt=prompt, kind=VIDEO_KIND, label=VIDEO_KIND, speak=True, echo=display)
    finally:
        media.remove_files([temp_video, temp_mp3, *screenshots])


async def _run_video(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    if not (message.video or message.video_note or message.document):
        await sender.send_text(VIDEO_MISSING, markdown=False)
        return
    status = await sender.send_text("🎬 *Processing video...*")
    try:
        prepared = await prepare_video(bot, message, user_id, _job_progress(sender, status, ctx))
        if not prepared.ok:
            await sender.edit_text(status, prepared.error_text or VIDEO_ERROR)
            return
        await sender.edit_text(status, prepared.echo or "🎬")
        await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit, speak=True,
        )
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing video message: %s", e)
        await sender.edit_text(status, VIDEO_ERROR)


async def _accept_video(message: Message, bot: Bot, user_id: str, limit: int | None, label: str) -> None:
    await accept(
        message, bot, user_id, limit, label=label, kind=VIDEO_KIND, error_text=VIDEO_ERROR,
        prepare=partial(prepare_video, bot, message, user_id), run=partial(_run_video, bot, message, user_id, limit),
    )


@router.message(F.video | F.video_note)
async def on_video(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "video")
    await _accept_video(message, bot, user_id, limit, "video message")


# ---------------------------------------------------------------------------
# audio files
# ---------------------------------------------------------------------------
AUDIO_KIND = "audio file"
AUDIO_ERROR = "❌ Error processing audio file"
AUDIO_MISSING = "❌ Could not process audio file."


async def prepare_audio(bot: Bot, message: Message, user_id: str) -> Prepared:
    """Save the file under ``audio/<original name>``; the prompt carries its metadata and path (no transcription)."""
    audio = message.audio
    if not audio:
        return Prepared(prompt=None, kind=AUDIO_KIND, label=AUDIO_KIND, error_text=AUDIO_MISSING, reason="the file is missing")
    caption = message.caption or ""
    file_name = os.path.basename(audio.file_name or f"audio_{uuid.uuid4()}.mp3")
    audio_path = str(media.user_dir(user_id, "audio") / file_name)
    async with _path_lock(audio_path):
        await bot.download(audio.file_id, destination=audio_path)
    duration = f"{audio.duration // 60}m{audio.duration % 60}s" if audio.duration else "unknown"
    prompt = f"User sent an audio file.\nFile: {file_name}\nPath: {audio_path}\nDuration: {duration}"
    if audio.performer:
        prompt += f"\nPerformer: {audio.performer}"
    if audio.title:
        prompt += f"\nTitle: {audio.title}"
    if caption:
        prompt += f"\nUser message: {caption}"
    return Prepared(prompt=with_reference(prompt, message, _bot_id(bot)), kind=AUDIO_KIND, label=AUDIO_KIND)


async def _run_audio(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    if not message.audio:
        await sender.send_text(AUDIO_MISSING, markdown=False)
        return
    status = await sender.send_text("🎵 *Saving audio file...*")
    try:
        ctx.set_progress("saving audio")
        prepared = await prepare_audio(bot, message, user_id)
        await sender.edit_text(status, "🎵 *Audio saved, processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🎵 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing audio file: %s", e)
        await sender.edit_text(status, AUDIO_ERROR)


@router.message(F.audio)
async def on_audio(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "audio")
    await accept(
        message, bot, user_id, limit, label=AUDIO_KIND, kind=AUDIO_KIND, error_text=AUDIO_ERROR,
        prepare=partial(prepare_audio, bot, message, user_id), run=partial(_run_audio, bot, message, user_id, limit),
    )


# ---------------------------------------------------------------------------
# photos and albums
# ---------------------------------------------------------------------------
IMAGES_ERROR = "❌ Could not process the image(s)."


def _images_kind(count: int) -> str:
    return "photo" if count == 1 else f"photo album ({count} images)"


async def prepare_images(
    bot: Bot, message: Message, user_id: str, refs: list[Any], caption: str | None, progress: Progress = _no_progress,
) -> Prepared:
    """Photos (one message or a flushed album) go to the agent as image blocks, not descriptions.

    Each picture is kept twice under ``images/``: ``image_<uuid>.jpg`` at full resolution (the
    path the prompt names, for edits and re-sends) and ``image_<uuid>_v.jpg``, downscaled to
    ``MAX_IMAGE_RESOLUTION_VISION`` and under the API size limit, which is what the ``image_ref``
    points at and what the transcript stores. A picture that cannot be prepared is named as such.
    """
    media.ensure_temp_dirs()
    temp_files: list[str] = []
    lines: list[str] = []
    media_blocks: list[dict] = []
    kind = _images_kind(len(refs))
    try:
        images_dir = media.user_dir(user_id, "images")
        for i, ref in enumerate(refs, 1):
            try:
                await progress(None, f"preparing image {i}/{len(refs)}")
                ext = os.path.splitext(getattr(ref, "file_name", "") or "")[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    ext = ".jpg"
                downloaded = os.path.join(media.TEMP_PHOTOS, f"photo_{uuid.uuid4()}{ext}")
                temp_files.append(downloaded)
                await bot.download(ref.file_id, destination=downloaded)
                stem = f"image_{uuid.uuid4()}"
                permanent = str(images_dir / f"{stem}.jpg")
                await asyncio.to_thread(media.prepare_downloaded_image_for_vision, downloaded, permanent)
                vision_copy = str(images_dir / f"{stem}_v.jpg")
                await asyncio.to_thread(media.save_vision_copy, permanent, vision_copy)
                lines.append(f"Image {i}: {permanent}")
                media_blocks.append(_content().image_ref(vision_copy, caption=f"Image {i}"))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("Error processing photo %d: %s", i, e)
                lines.append(f"Image {i}: could not be processed ({type(e).__name__})")
    finally:
        media.remove_files(temp_files)

    if not media_blocks:
        return Prepared(prompt=None, kind=kind, label=kind, error_text=IMAGES_ERROR, reason="no picture could be read")
    n = len(refs)
    if caption is None:
        head = f"User sent {n} photo(s)."
    else:
        head = f"{caption}\n\nUser attached {n} image(s) to this message."
    text = f"{head} They are shown below as images and saved in your workspace:\n" + "\n".join(lines) + f"\n\n{IMAGE_EMBED_HINT}"
    return Prepared(prompt=_prompt_with_reference(text, media_blocks, message, _bot_id(bot)), kind=kind, label=kind)


async def _run_images(
    bot: Bot, message: Message, user_id: str, limit: int | None, refs: list[Any], caption: str | None,
    status: Message | None, ctx: JobContext,
) -> None:
    """The queued form of ``prepare_images``: a status message while the pictures are prepared, then the turn."""
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    if status is not None:
        await sender.edit_text(status, "🖼️ *Preparing images...*")
    else:
        status = await sender.send_text("🖼️ *Preparing images...*")
    try:
        prepared = await prepare_images(bot, message, user_id, refs, caption, _job_progress(sender, None, ctx))
        if not prepared.ok:
            await sender.edit_text(status, prepared.error_text or IMAGES_ERROR, markdown=False)
            return
        await sender.edit_text(status, "🤖 *Processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🤖 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        trace_id = str(uuid.uuid4())
        logger.exception("Error processing images (trace %s): %s", trace_id, e)
        await sender.edit_text(status, f"❌ An error occurred while processing the images. Trace ID: {trace_id}", markdown=False)


async def _accept_images(
    message: Message, bot: Bot, user_id: str, limit: int | None, refs: list[Any], caption: str | None, *,
    label: str, status: Message | None = None, seq: int | None = None, window: bool = True,
) -> None:
    await accept(
        message, bot, user_id, limit, label=label, kind=_images_kind(len(refs)), error_text=IMAGES_ERROR,
        prepare=partial(prepare_images, bot, message, user_id, list(refs), caption),
        run=partial(_run_images, bot, message, user_id, limit, list(refs), caption, status), seq=seq,
        window=window, waiting=status,
    )


@dataclass
class _Album:
    message: Message
    user_id: str
    limit: int | None
    refs: list[Any] = field(default_factory=list)
    caption: str | None = None
    waiting: Message | None = None
    flush_task: asyncio.Task | None = None
    seq: int = field(default_factory=next_seq)  # the first picture's arrival: the album keeps that place


_albums: dict[tuple[int, str], _Album] = {}


async def _flush_album(bot: Bot, key: tuple[int, str]) -> None:
    try:
        await asyncio.sleep(MEDIA_GROUP_TIMEOUT)
    except asyncio.CancelledError:
        return
    album = _albums.pop(key, None)
    if album is None or not album.refs:
        return
    thread_id = _thread_id(album.message)
    topic = (album.message.chat.id, thread_id)
    _albums_handing[topic] = _albums_handing.get(topic, 0) + 1  # a message window of this topic waits for it
    try:
        _stats().track_media_group_processed(album.user_id, len(album.refs))
        label = f"photo album ({len(album.refs)} images)"
        if inject_target(album.user_id, thread_id) is not None and album.waiting is not None:
            # The album joins the running turn: the "waiting for images" line has nothing more to say.
            await delete_quietly(album.waiting)
            album.waiting = None
        # It waited for its own pictures already: it goes into a message window only when one is
        # open for its topic (messages sent with it, e.g. a comment or forwards, wait for it there).
        await _accept_images(
            album.message, bot, album.user_id, album.limit, album.refs, album.caption, label=label, status=album.waiting,
            seq=album.seq, window=_collecting(album.user_id, thread_id) is not None,
        )
    finally:
        left = _albums_handing.get(topic, 0) - 1
        if left > 0:
            _albums_handing[topic] = left
        else:
            _albums_handing.pop(topic, None)


async def collect_album(message: Message, bot: Bot, user_id: str, limit: int | None, ref: Any, caption: str | None) -> None:
    key = (message.chat.id, str(message.media_group_id))
    album = _albums.get(key)
    if album is None:
        album = _Album(message=message, user_id=user_id, limit=limit)
        _albums[key] = album
        album.waiting = await answer_md(message, "🖼️ *Image media group received... Waiting for images...*")
    if caption and album.caption is None:
        album.caption = caption
    album.refs.append(ref)
    if album.waiting is not None:
        try:
            await album.waiting.edit_text(f"🖼️ *Image {len(album.refs)} received... Waiting for other images...*", parse_mode="Markdown")
        except Exception:  # noqa: BLE001
            pass
    if album.flush_task is not None:
        album.flush_task.cancel()
    album.flush_task = asyncio.create_task(_flush_album(bot, key))


@router.message(F.photo)
async def on_photo(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "photo")
    ref = message.photo[-1]
    if message.media_group_id:
        await collect_album(message, bot, user_id, limit, ref, message.caption)
        return
    await _accept_images(message, bot, user_id, limit, [ref], message.caption, label="photo")


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def _cap_document_text(text: str, limit: int, hint: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…(truncated: {hint})"


async def _describe_fallback(
    progress: Progress, user_id: str, ext: str, doc_type: str, file_name: str, saved: str, caption: str | None, reason: str,
) -> str:
    """The pre-F7 path: a model's analysis of the document goes in the prompt (a PDF over the
    block limits, text that could not be extracted or is too long to include)."""
    default = "Analyze this document and describe its contents in detail."
    question = default if caption is None else f"{default} When you are done, answer the following question: {caption}"
    await progress(f"🤖 *Analyzing {doc_type} content...*", None)
    description = await media.describe_document(question, saved)
    _stats().track_describe_used(user_id, media.describe_type_for(ext))
    return (
        f"{caption or default}\n\nUser attached a {doc_type} document to this message.\nFile: {file_name}\nSaved to: {saved}\n"
        f"{reason}\n\nHere is the analysis of document contents:\n\n{description}"
    )


async def _document_prompt(
    progress: Progress, user_id: str, ext: str, doc_type: str, file_name: str, saved: str, caption: str | None,
) -> tuple[str, list]:
    """(prompt text, media blocks) for a saved document, by extension: a PDF within the limits
    as a ``document_ref``; text files and extracted docx/xlsx/json text in the prompt (capped);
    everything else through ``_describe_fallback``."""
    user_line = f"\nUser message: {caption}" if caption else ""
    if ext == ".pdf":
        try:
            pages = await asyncio.to_thread(media.pdf_page_count, saved)
        except Exception as e:  # noqa: BLE001 - not a PDF pypdf can read
            logger.warning("Could not read %s with pypdf: %s", saved, e)
            pages = None
        if pages is not None and media.pdf_fits_as_block(saved, pages):
            text = (
                f"User sent a PDF: {file_name} ({pages} pages), saved at {saved}.{user_line}\n"
                "Read it from the attached document block; read_document gives it to you again later."
            )
            return text, [_content().document_ref(saved, title=file_name)]
        if pages is None:
            reason = "The PDF could not be attached directly (it could not be parsed as a PDF)."
        else:
            size_mb = os.path.getsize(saved) / (1024 * 1024)
            reason = (
                f"The PDF was too large to attach directly ({pages} pages, {size_mb:.1f} MB; the limits are "
                f"{media.PDF_MAX_PAGES} pages and {media.PDF_MAX_BYTES // (1024 * 1024)} MB), so it was analyzed instead."
            )
        return await _describe_fallback(progress, user_id, ext, doc_type, file_name, saved, caption, reason), []
    if ext in media.TEXT_EXTENSIONS:
        content = await asyncio.to_thread(media.extract_document_text, saved)
        shown = _cap_document_text(content, TEXT_PROMPT_CAP, "use read_file for the rest")
        text = (
            f"User sent a {doc_type} file: {file_name} ({len(content)} characters), saved at {saved}.{user_line}\n\n"
            f"File content:\n<file_content>\n{shown}\n</file_content>"
        )
        return text, []
    if ext in media.OFFICE_EXTENSIONS:
        try:
            extracted = await asyncio.to_thread(media.extract_document_text, saved)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not extract text from %s: %s", saved, e)
            extracted = None
        if extracted is not None and len(extracted) <= EXTRACTED_TEXT_CAP:
            text = (
                f"User sent a {doc_type} document: {file_name}, saved at {saved}.{user_line}\n\n"
                f"Extracted text:\n<document_text>\n{extracted}\n</document_text>"
            )
            return text, []
        reason = (
            "The document's text could not be extracted, so it was analyzed instead." if extracted is None
            else f"The document's text ({len(extracted)} characters) is too long to include, so it was analyzed instead; read_file / execute_python give you the details."
        )
        return await _describe_fallback(progress, user_id, ext, doc_type, file_name, saved, caption, reason), []
    return await _describe_fallback(progress, user_id, ext, doc_type, file_name, saved, caption, ""), []


def _document_names(message: Message) -> tuple[str, str, str]:
    """(file name, extension, type) of a document: the name lowercased (re-uploads overwrite)."""
    document = message.document
    file_name = os.path.basename((document.file_name or f"document_{uuid.uuid4()}").lower())
    ext = os.path.splitext(file_name)[1].lower()
    return file_name, ext, ext.replace(".", "").upper()


async def prepare_document(bot: Bot, message: Message, user_id: str, progress: Progress = _no_progress) -> Prepared:
    """Save the document under ``documents/<lowercased name>`` and build its block prompt (``_document_prompt``)."""
    file_name, ext, doc_type = _document_names(message)
    await progress(None, "reading document")
    saved = str(media.user_dir(user_id, "documents") / file_name)
    async with _path_lock(saved):  # the same name sent twice meanwhile: one download (and read) at a time
        await bot.download(message.document.file_id, destination=saved)
        text, media_blocks = await _document_prompt(progress, user_id, ext, doc_type, file_name, saved, message.caption)
    kind = f"{doc_type} document"
    return Prepared(prompt=_prompt_with_reference(text, media_blocks, message, _bot_id(bot)), kind=kind, label=kind)


async def _run_document(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    """A document goes to the agent as a block prompt (``prepare_document``) after a status message."""
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    _, _, doc_type = _document_names(message)
    status = await sender.send_text(f"📄 *Processing {doc_type} document...*")
    try:
        prepared = await prepare_document(bot, message, user_id, _job_progress(sender, status, ctx))
        await sender.edit_text(status, "🤖 *Processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🤖 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        trace_id = str(uuid.uuid4())
        logger.exception("Error processing document (trace %s): %s", trace_id, e)
        await sender.edit_text(status, f"❌ An error occurred while processing the document. Trace ID: {trace_id}", markdown=False)


@router.message(F.document)
async def on_document(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    file_name = (message.document.file_name or "").lower()
    ext = os.path.splitext(file_name)[1].lower()
    if ext in media.VIDEO_EXTENSIONS:
        _stats().track_message_received(user_id, "video")
        await _accept_video(message, bot, user_id, limit, "video file")
        return
    if ext in IMAGE_EXTENSIONS:
        _stats().track_message_received(user_id, "photo")
        if message.media_group_id:
            await collect_album(message, bot, user_id, limit, message.document, message.caption)
            return
        await _accept_images(message, bot, user_id, limit, [message.document], message.caption, label="image file")
        return
    if ext not in media.DOCUMENT_EXTENSIONS:
        supported = ", ".join(e.replace(".", "").upper() for e in media.DOCUMENT_EXTENSIONS)
        images = "/".join(e.replace(".", "").upper() for e in IMAGE_EXTENSIONS)
        await message.answer(f"❌ Only {supported} documents and {images} images are supported.")
        return
    _stats().track_message_received(user_id, "document")
    kind = f"{ext.replace('.', '').upper()} document"
    await accept(
        message, bot, user_id, limit, label=kind, kind=kind, error_text="❌ An error occurred while processing the document.",
        prepare=partial(prepare_document, bot, message, user_id), run=partial(_run_document, bot, message, user_id, limit),
    )
