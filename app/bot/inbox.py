"""The running turn's inbox: what the user sends while the agent works.

While a turn runs (``bot.agent_runner.run_turn`` opens one inbox per turn and closes it
when the agent has returned, failed or was stopped), a new message of the same chat and
the same forum topic is not queued as a turn of its own: the handler delivers it here
(``bot/handlers/messages.py``) and reacts 👀. The agent loop calls ``drain()`` at each
tool boundary and appends what it returns to the user message that carries the tool
results, so the model reads the message between two steps; the user then sees ✍.
Finished background tasks (``bot.background``) arrive here too: their item gives a notice
block (``agents.task_tools.NOTICE_BLOCK``, facts only), which the agent core sends as a
separate ``system`` message, never in the user's words; the output itself is read with
``get_task_result``.

Order is the user's: items are kept by their arrival number (``bot.queue.next_seq``), an
item may still be *preparing* (a photo download, a voice transcription: an asyncio task
producing a ``Prepared``) and ``drain`` stops at the first item that is not ready (after
waiting up to ``DRAIN_WAIT`` for it), so nothing overtakes it. What the turn did not
consume is handed back at ``close()``: the user items together become ONE queued turn
(``requeue_group``, with the prompts they already prepared: nothing is downloaded or
transcribed twice, ``combine`` puts them in one prompt), inserted in the queue at the first
one's arrival number, and task notifications become "background task finished" turns.
``/cancel`` drops them instead (``discard``). A run that ends quiet (a scheduled task or a
notification turn answering the quiet marker) hands back the user messages it did read as
well (``consumed_user_items``, ``bot.agent_runner``).

The formats, which the agent core passes through without parsing::

    <user_message sent_during_turn="true" time="2026-10-18 12:01 UTC">
    <reply/forward preamble, if any>
    <text or caption, "(voice message, transcribed) ...", or the photo/document prompt text>
    </user_message>

followed by the message's image/document blocks (materialized), and a task's notice
(built by ``bot.background`` when it is read, so a result the agent has already read gives
nothing) as ``{"type": NOTICE_BLOCK, "text": "The background task t3 (…) completed after 42 s. …"}``.

Several messages that become one turn (sent together within the message window of
``bot/handlers/messages.py``, or a turn's leftovers) are one prompt (``combine``)::

    The user sent 3 messages one after another; read them together as one request.

    <message n="1">
    <its text, the reply/forward preamble included>
    </message>
    <message n="2">
    ... its image/document refs inside the tags ...
    </message>

No import-time side effects and no ``agents`` imports at module level.
"""
from __future__ import annotations

import asyncio
import bisect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bot.queue import next_seq

logger = logging.getLogger(__name__)

DRAIN_WAIT = 3.0  # seconds drain() waits for an item still being prepared
USER = "user"
TASK = "task"
SEEN_REACTION = "👀"  # the message reached the running turn
READ_REACTION = "✍"  # the agent has read it
SENT_WHILE_WORKING = "[Sent while I was working]"  # turn records and memory files: what arrived during the turn
MAX_FAILURE_REASON = 160


# ---- what a handler prepared ----------------------------------------------------------------
@dataclass
class Prepared:
    """A user message made ready for the agent (``prepare_<kind>`` in ``bot/handlers/messages.py``).

    ``prompt`` is what a turn of its own runs on (a string, or blocks from ``agents.content``:
    text plus media refs); None when nothing usable came out of it. ``inject`` is what the
    running turn reads instead, when it differs (a voice message is marked as transcribed).
    ``error_text`` is what the user is told when it failed (a turn of its own shows it instead
    of running), ``reason`` the short cause the agent reads, ``echo`` a line shown to the user
    once it is ready (a voice message's transcription).
    """

    prompt: str | list | None
    kind: str = "message"
    label: str = "message"
    speak: bool = False
    inject: str | list | None = None
    error_text: str | None = None
    reason: str | None = None
    echo: str | None = None

    @property
    def ok(self) -> bool:
        return self.prompt is not None

    def inject_prompt(self) -> str | list:
        if not self.ok:
            return failed_text(self.kind, self.reason)
        return self.inject if self.inject is not None else self.prompt


COMBINED_HEAD = "The user sent {count} messages one after another; read them together as one request."
COMBINED_KIND = "messages"


def combined_label(count: int) -> str:
    return f"{count} messages"


def combine(items: list[Prepared]) -> Prepared:
    """Several prepared messages as ONE prompt, in the given order; one message is returned as it is.

    Each message is wrapped in ``<message n="i">`` with its media refs inside the tags; it is its
    ``inject_prompt()`` (a voice message marked as transcribed, a failed one as a short note the
    agent reads). A string when there is no media, else blocks. The answer is spoken only when
    every usable message asked for it (all voice or video). Not ``ok`` when none was usable: its
    ``error_text`` lists theirs.
    """
    if len(items) == 1:
        return items[0]
    from agents import content

    label = combined_label(len(items))
    usable = [p for p in items if p.ok]
    if not usable:
        errors = list(dict.fromkeys(p.error_text for p in items if p.error_text))
        return Prepared(
            prompt=None, kind=COMBINED_KIND, label=label, error_text="\n".join(errors) or None,
            reason="none of the messages could be processed",
        )
    blocks: list[dict] = [content.text_block(COMBINED_HEAD.format(count=len(items)))]
    for n, prepared in enumerate(items, 1):
        part = content.normalize(prepared.inject_prompt())
        body = "\n\n".join(str(b.get("text", "")) for b in part if b.get("type") == "text" and str(b.get("text", "")).strip())
        blocks.append(content.text_block(f'<message n="{n}">\n{body}'))
        blocks.extend(b for b in part if b.get("type") != "text")
        blocks.append(content.text_block("</message>"))
    merged: list[dict] = []
    for block in blocks:  # adjacent text blocks as one: a string prompt when there is no media
        if merged and block.get("type") == "text" and merged[-1].get("type") == "text":
            glue = "\n" if block["text"] == "</message>" else "\n\n"
            merged[-1] = content.text_block(f"{merged[-1]['text']}{glue}{block['text']}")
        else:
            merged.append(block)
    prompt: str | list = merged[0]["text"] if len(merged) == 1 else merged
    return Prepared(prompt=prompt, kind=COMBINED_KIND, label=label, speak=all(p.speak for p in usable))


def failed_text(kind: str, reason: str | None) -> str:
    """What the running turn reads for a message that could not be prepared."""
    reason = " ".join(str(reason or "unknown error").split())
    if len(reason) > MAX_FAILURE_REASON:
        reason = reason[: MAX_FAILURE_REASON - 1] + "…"
    return f"(the user sent a {kind} that could not be processed: {reason})"


def failure_reason(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def ready_future(value: Any) -> asyncio.Future:
    """A future that is already done (an item with nothing left to prepare)."""
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


@dataclass(eq=False)
class InboxItem:
    """One delivery: a user message (its future gives a ``Prepared``) or a finished task (its future
    gives the blocks, or a callable that builds them when the item is read). ``requeue`` turns a leftover into a queued turn;
    ``requeue_group`` queues several user leftovers (this one first) as ONE turn and wins over
    ``requeue``; ``drop`` is called when it is thrown away (/cancel, shutdown)."""

    kind: str
    future: asyncio.Future
    seq: int = field(default_factory=next_seq)
    thread_id: int | None = None
    label: str = "message"
    requeue: Callable[[], None] | None = None
    requeue_group: Callable[[list["InboxItem"]], None] | None = None
    drop: Callable[[], None] | None = None
    # reactions (user items): where the message is, and the 👀 call made on arrival
    bot: Any = None
    chat_id: int | None = None
    message_id: int | None = None
    sent_at: datetime | None = None
    seen: asyncio.Task | None = None

    def ready(self) -> bool:
        return self.future.done()


# ---- reactions ---------------------------------------------------------------------------------
_reaction_tasks: set[asyncio.Task] = set()


async def react(bot: Any, chat_id: int | None, message_id: int | None, emoji: str) -> bool:
    """Best-effort reaction on a message; never raises (an old client, a bot without rights)."""
    if bot is None or chat_id is None or not message_id:
        return False
    try:
        from aiogram.types import ReactionTypeEmoji

        await bot.set_message_reaction(chat_id=chat_id, message_id=message_id, reaction=[ReactionTypeEmoji(emoji=emoji)])
        return True
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not react %s on %s/%s: %s", emoji, chat_id, message_id, e)
        return False


async def mark_seen(item: InboxItem) -> None:
    """👀 on the message that reached the running turn (the handler awaits it)."""
    item.seen = asyncio.create_task(react(item.bot, item.chat_id, item.message_id, SEEN_REACTION))
    await item.seen


async def _mark_read(item: InboxItem) -> None:
    if item.seen is not None and not item.seen.done():
        try:
            await item.seen  # the 👀 must not land after the ✍
        except BaseException:  # noqa: BLE001 - only the order matters here
            pass
    await react(item.bot, item.chat_id, item.message_id, READ_REACTION)


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _reaction_tasks.add(task)
    task.add_done_callback(_reaction_tasks.discard)


async def flush_reactions() -> None:
    """Wait for the ✍ reactions in flight (tests)."""
    while _reaction_tasks:
        await asyncio.gather(*list(_reaction_tasks), return_exceptions=True)


# ---- the inbox -----------------------------------------------------------------------------------
def _when(moment: datetime | None) -> str:
    moment = moment or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _user_blocks(prepared: Prepared, sent_at: datetime | None, base: Path | None) -> tuple[list[dict], str, list[dict]]:
    """(API blocks, text view, record blocks) of one user item. Blocking (reads the media): run in a thread."""
    from agents import content

    blocks = content.normalize(prepared.inject_prompt())
    texts = [str(b.get("text", "")) for b in blocks if b.get("type") == "text" and str(b.get("text", "")).strip()]
    refs = [b for b in blocks if b.get("type") != "text"]
    body = "\n\n".join(texts)
    head = f'<user_message sent_during_turn="true" time="{_when(sent_at)}">'
    api = [content.text_block(f"{head}\n{body}\n</user_message>"), *content.materialize(refs, base=base)]
    record = [content.text_block(body), *refs] if refs else [content.text_block(body)]
    return api, content.prompt_text(blocks), record


class TurnInbox:
    """The inbox of one running turn (the ``TurnInbox`` protocol of the agent core)."""

    def __init__(self, user_id: str, thread_id: int | None, *, base: str | Path | None = None) -> None:
        self.user_id = str(user_id)
        self.thread_id = thread_id
        self.base = Path(base) if base is not None else None
        self.closed = False
        self._items: list[InboxItem] = []  # not consumed yet, sorted by seq
        self._consumed: list[InboxItem] = []  # the user items consumed so far, in the order they were read
        self._texts: list[str] = []  # text views of the consumed user messages
        self._record: list[dict] = []  # their text + media ref blocks (turn record)
        self._changed = asyncio.Event()
        self._lock = asyncio.Lock()

    # -- delivery ----------------------------------------------------------------------------------
    def accepts(self, thread_id: int | None) -> bool:
        return not self.closed and thread_id == self.thread_id

    def deliver(self, item: InboxItem) -> bool:
        """Add ``item`` in arrival order; False when this inbox is closed (the caller queues it)."""
        if self.closed:
            return False
        keys = [i.seq for i in self._items]
        self._items.insert(bisect.bisect_right(keys, item.seq), item)
        item.future.add_done_callback(lambda _f: self._changed.set())
        self._changed.set()
        return True

    # -- the protocol ----------------------------------------------------------------------------------
    def has_user_messages(self) -> bool:
        """A user message can be drained now: ready, with nothing still being prepared ahead of it.

        One still being prepared (a video being described can take minutes) does not count until
        it is ready: the agent loop moves calls to the background for a message it can read now,
        not for one it cannot read yet (it looks again while ``has_pending`` says something waits).
        """
        return self._drainable() == "message"

    def has_pending(self) -> bool:
        """Anything is waiting: user messages or task notifications."""
        return bool(self._items)

    @property
    def drained_user_texts(self) -> list[str]:
        """Text views of the user messages consumed so far this turn (memory files, the turn record)."""
        return list(self._texts)

    @property
    def drained_record_blocks(self) -> list[dict]:
        """The consumed user messages as prompt blocks (text + media refs), for the turn record."""
        return [dict(b) for b in self._record]

    @property
    def drained_media(self) -> bool:
        return any(b.get("type") != "text" for b in self._record)

    @property
    def consumed_user_items(self) -> list[InboxItem]:
        """The user items read so far, in reading order (a quiet run hands them back: ``requeue``)."""
        return list(self._consumed)

    def _drainable(self) -> str | None:
        """"message" / "task" for what drain() would return right now (the ready head), else None."""
        kinds = set()
        for item in self._items:
            if not item.ready():
                break
            kinds.add(item.kind)
        if USER in kinds:
            return "message"
        if TASK in kinds:
            return "task"
        return None

    async def wait(self, timeout: float) -> str:
        """"message" | "task" | "timeout": returns at once when something can be drained.

        A user message that is still being prepared does not count until it is ready (a caller
        that drains right away would get nothing and ask again in a loop).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout or 0))
        while True:
            found = self._drainable()
            if found is not None:
                return found
            remaining = deadline - loop.time()
            if self.closed or remaining <= 0:
                return "timeout"
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except asyncio.TimeoutError:
                pass

    async def drain(self) -> list[dict]:
        """API blocks of everything ready at the head, in arrival order; marks it consumed.

        Waits up to ``DRAIN_WAIT`` in total for an item still being prepared and stops at the
        first one that is still not ready, so nothing overtakes it. [] when empty.

        An item leaves the inbox only once it has been read: a cancel (Stop, the deadline) while a
        user item is being read (``_user_blocks`` in a thread) leaves it in its place, and the
        turn's ``close()`` hands it back like any other leftover.
        """
        async with self._lock:
            out: list[dict] = []
            loop = asyncio.get_running_loop()
            deadline = loop.time() + DRAIN_WAIT
            while self._items and not self.closed:
                item = self._items[0]
                if not item.ready():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    await asyncio.wait({item.future}, timeout=remaining)
                    if not item.ready():
                        break
                    continue  # the head may have changed while waiting
                blocks = await self._consume(item)
                if blocks is None:
                    break  # the inbox closed while the item was read: it is a leftover now
                out.extend(blocks)
            return out

    def _take(self, item: InboxItem) -> bool:
        """Remove ``item`` from the waiting items; False when it is no longer there (``close()`` took it)."""
        if self.closed or item not in self._items:
            return False
        self._items.remove(item)
        return True

    async def _consume(self, item: InboxItem) -> list[dict] | None:
        """Read ``item`` and mark it consumed (its API blocks); None when it was taken away meanwhile."""
        if item.future.cancelled():
            return [] if self._take(item) else None
        if item.kind == TASK:
            if not self._take(item):
                return None
            try:
                blocks = item.future.result()
                if callable(blocks):
                    blocks = blocks()  # rendered now: the agent may have read the result meanwhile
            except Exception as e:  # noqa: BLE001
                logger.error("A task notification for %s could not be read: %s", self.user_id, e)
                return []
            return [dict(b) for b in blocks or [] if isinstance(b, dict)]
        try:
            prepared = item.future.result()
        except Exception as e:  # noqa: BLE001 - the handlers' wrapper never raises; a stand-in might
            prepared = Prepared(prompt=None, kind=item.label, reason=failure_reason(e))
        if not isinstance(prepared, Prepared):
            prepared = Prepared(prompt=prepared, kind=item.label)
        try:
            api, text, record = await asyncio.to_thread(_user_blocks, prepared, item.sent_at, self.base)
        except Exception as e:  # noqa: BLE001
            logger.error("Could not read an injected message of %s: %s", self.user_id, e)
            fallback = Prepared(prompt=None, kind=prepared.kind, reason=failure_reason(e))
            api, text, record = _user_blocks(fallback, item.sent_at, None)
        if not self._take(item):
            return None  # /cancel or the turn's end took it while it was read: not consumed, no ✍
        self._consumed.append(item)
        self._texts.append(text)
        if self._record:
            self._record.extend(record)
        else:
            self._record = [{**record[0], "text": f"{SENT_WHILE_WORKING}\n{record[0]['text']}"}, *record[1:]]
        if item.message_id:
            _spawn(_mark_read(item))
        return api

    # -- the end ----------------------------------------------------------------------------------
    def close(self) -> list[InboxItem]:
        """Stop accepting; the items not consumed, in arrival order (the caller requeues or drops them)."""
        self.closed = True
        leftovers, self._items = self._items, []
        self._changed.set()
        if _open.get(self.user_id) is self:
            del _open[self.user_id]
        return leftovers


# ---- registry (one open inbox per user: the user's turns run one at a time) ----------------------
_open: dict[str, TurnInbox] = {}
_registry_loop: asyncio.AbstractEventLoop | None = None


def _same_loop() -> None:
    """Inboxes belong to the event loop they were opened on: a new loop (a restart of the loop,
    every test) starts with none, so a turn that died with its loop cannot swallow messages."""
    global _registry_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if loop is not _registry_loop:
        _open.clear()
        _registry_loop = loop


def open_inbox(user_id: str, thread_id: int | None, *, base: str | Path | None = None) -> TurnInbox:
    """The inbox of a turn that starts now. If one is somehow still open for the user, the new one
    is not registered (it receives nothing) rather than taking the other's messages."""
    _same_loop()
    inbox = TurnInbox(user_id, thread_id, base=base)
    _open.setdefault(str(user_id), inbox)
    return inbox


def current(user_id: str) -> TurnInbox | None:
    _same_loop()
    return _open.get(str(user_id))


def accepts(user_id: str, thread_id: int | None) -> bool:
    inbox = current(user_id)
    return inbox is not None and inbox.accepts(thread_id)


def deliver_task(user_id: str, thread_id: int | None, item: InboxItem) -> bool:
    """A finished background task reaches the user's running turn of that topic; False when there is none."""
    inbox = current(user_id)
    return inbox is not None and inbox.accepts(thread_id) and inbox.deliver(item)


def requeue(leftovers: list[InboxItem]) -> None:
    """Hand leftovers back in order (never raises, never suspends): the user messages become ONE
    queued turn together (``requeue_group`` of the first, at its arrival number), each task
    notification and each item with only ``requeue`` a queued turn of its own."""
    group = sorted((i for i in leftovers if i.kind == USER and i.requeue_group is not None), key=lambda i: i.seq)
    if group:
        try:
            group[0].requeue_group(group)
        except Exception as e:  # noqa: BLE001 - e.g. the queue is closed (shutdown)
            logger.warning("Could not requeue %d leftover message(s): %s", len(group), e)
            for item in group:
                _drop_one(item)
    grouped = {id(i) for i in group}
    for item in leftovers:
        if id(item) in grouped or item.requeue is None:
            continue
        try:
            item.requeue()
        except Exception as e:  # noqa: BLE001 - e.g. the queue is closed (shutdown)
            logger.warning("Could not requeue a leftover %s (%s): %s", item.kind, item.label, e)
            _drop_one(item)


def _drop_one(item: InboxItem) -> None:
    if not item.future.done():
        item.future.cancel()  # a preparation nobody needs any more
    if item.drop is not None:
        try:
            item.drop()
        except Exception as e:  # noqa: BLE001
            logger.debug("Dropping an inbox item failed: %s", e)


def drop(leftovers: list[InboxItem]) -> int:
    """Throw leftovers away (/cancel, shutdown); the number of user messages among them."""
    for item in leftovers:
        _drop_one(item)
    return sum(1 for i in leftovers if i.kind == USER)


def finish(inbox: TurnInbox | None, *, discard: bool = False, read: list[InboxItem] | tuple = ()) -> None:
    """Close ``inbox`` (a no-op when it is None or closed) and requeue, or drop, what is left.
    ``read``: consumed user items to hand back with the leftovers, as the same turn (a quiet run)."""
    if inbox is None or inbox.closed:
        return
    leftovers = inbox.close()
    if discard:
        drop(leftovers)
    else:
        requeue([*read, *leftovers])


def discard(user_id: str) -> int:
    """/cancel: close the user's open inbox and drop what it holds; the number of user messages dropped."""
    inbox = current(user_id)
    if inbox is None:
        return 0
    return drop(inbox.close())


def reset() -> None:
    """Forget every open inbox (tests)."""
    _open.clear()
