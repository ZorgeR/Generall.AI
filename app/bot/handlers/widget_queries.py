"""Taps inside a widget that ask the bot (Bot API 10.4 ``widget_query``) become agent turns.

aiogram 3.31 does not know the update: it keeps the payload in ``Update.model_extra`` and would only
warn "unknown update type" (a later aiogram with Bot API 10.4 declares it as a field: ``query_of`` reads both). ``WidgetQueryMiddleware``, an outer middleware on ``dp.update``
(``bot/app.py``; ``allowed_updates`` gets ``widget_query`` there too), takes it first and hands it to
``handle_query``.

Every query is answered at once (the server waits 15 s for ``answerWidgetQuery``): ``ACK`` when the
request reached the agent, ``{"ok":false,"error":...}`` otherwise (not a widget the agent sent with
``allow_queries``, a forwarded copy, an unauthorized chat, the action quota). What the agent does
about it reaches the widget as a change of its state (``update_widget`` → ``chatDataChanged`` /
``publicDataChanged``), never in the answer.

The request itself becomes, like a follow-up button:
- part of the running turn of the widget's topic, when there is one (``messages.inject_target`` /
  ``inject``, a ready ``Prepared``; no 👀, the widget message is not a user message);
- otherwise a queued turn (label ``WIDGET_LABEL``, header ``WIDGET_HEADER``, replying to the widget,
  the quiet marker: an answer that is only ``<no_update/>`` sends nothing, so a tap that only changes
  the widget leaves no message behind). Requests of the same widget that arrive while that turn is
  still queued join it (``_Batch.events``, at most ``MAX_EVENTS``); the same data again within
  ``DEDUPE_SECONDS`` is acknowledged and dropped (a double tap), and past ``RATE_LIMIT`` accepted requests
  of one widget per ``RATE_WINDOW`` the answer is an error (a widget asking in a loop).

Each request that starts or joins a turn counts against the action quota like a message
(``track_message_received(uid, "widget")``). State here lives in memory per event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

from bot import inbox, widgets
from bot.auth import auth
from bot.queue import Job, JobContext

logger = logging.getLogger(__name__)

UPDATE_TYPE = "widget_query"  # requests from an original message (the only kind the agent's widgets send)
COPY_UPDATE_TYPE = "shared_widget_query"  # a forwarded copy of a shared widget (the agent sends none)
WIDGET_LABEL = "widget request"
WIDGET_HEADER = "🧩 *Widget request*"
WIDGET_REASONING_CAPTION = "Widget request reasoning history."
ERROR_TEXT = "❌ Could not process the widget request."
DEDUPE_SECONDS = 2.0
MAX_EVENTS = 10
RATE_WINDOW = 60.0  # seconds
RATE_LIMIT = 6  # requests of one widget accepted per RATE_WINDOW (a widget asking in a loop costs a turn each)
MAX_DATA_CHARS = 64

ACK = {"ok": True}


def _error(reason: str) -> dict:
    return {"ok": False, "error": reason}


@dataclass
class Event:
    data: str
    who: str
    at: str  # HH:MM:SS UTC


@dataclass
class _Batch:
    user_id: str
    widget_id: str
    thread_id: int | None
    record: dict
    events: list[Event] = field(default_factory=list)
    started: bool = False


_batches: dict[tuple[str, str], _Batch] = {}
_last: dict[tuple[str, str], tuple[str, float]] = {}
_recent: dict[tuple[str, str], list[float]] = {}  # when the accepted requests of each widget came
_loop: asyncio.AbstractEventLoop | None = None


def _same_loop() -> None:
    global _loop
    loop = asyncio.get_running_loop()
    if loop is not _loop:
        _batches.clear()
        _last.clear()
        _recent.clear()
        _loop = loop


def reset() -> None:
    """Tests: forget pending batches, the dedupe memory and the rate counts."""
    _batches.clear()
    _last.clear()
    _recent.clear()


def _who(user: Any) -> str:
    if not isinstance(user, dict):
        return "someone"
    name = " ".join(" ".join(str(user.get(k) or "").split()) for k in ("first_name", "last_name")).strip()
    name = name[:64] or "someone"
    if user.get("username"):
        name += f" (@{''.join(str(user['username']).split())[:32]})"
    if user.get("id") is not None:
        name += f", user id {user['id']}"
    return name


def _state_lines(record: dict) -> list[str]:
    lines = []
    for key in ("chat_data", "public_data"):
        value = record.get(key)
        if value:
            lines.append(f"{key}: {value}")
    return lines or ["(no state set)"]


def widget_prompt(record: dict, events: list[Event]) -> str:
    """The framing of a widget request for the agent: which widget, what it asked, its last state."""
    head = f'[Widget request] Widget {record["id"]} "{record.get("name", "widget")}" (message {record["message_id"]})'
    if len(events) == 1:
        e = events[0]
        lines = [f"{head} sent the request {json.dumps(e.data, ensure_ascii=False)} (a tap by {e.who} at {e.at} UTC)."]
    else:
        lines = [f"{head} sent {len(events)} requests:"]
        lines += [f"- {json.dumps(e.data, ensure_ascii=False)} (a tap by {e.who} at {e.at} UTC)" for e in events]
    lines.append("Its state as you last set it:")
    lines += _state_lines(record)
    lines.append(
        "Do what the request asks and change the widget with update_widget (it redraws in place). Answer in the "
        "chat only when there is something to say beyond that; otherwise reply with exactly <no_update/>."
    )
    return "\n".join(lines)


def _stats():
    from stats import stats_tracker

    return stats_tracker


async def _answer(bot: Any, query_id: str, outcome: dict) -> None:
    try:
        await bot(widgets.AnswerWidgetQuery(widget_query_id=query_id, data=json.dumps(outcome, separators=(",", ":"))))
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - an expired query (over 15 s) or a server without widgets
        logger.debug("Could not answer widget query %s: %s", query_id, e)


async def handle_query(bot: Any, query: dict, *, copy: bool = False) -> dict:
    """Answer one widget query and hand its request to the agent; returns the answer. Never raises."""
    query_id = str(query.get("id") or "")
    try:
        outcome = await _route(bot, query, copy=copy)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - every query gets an answer, or the widget waits for TIMEOUT
        logger.exception("Widget query failed: %s", e)
        outcome = _error("the bot failed, try again")
    if query_id and bot is not None:
        await _answer(bot, query_id, outcome)
    return outcome


async def _route(bot: Any, query: dict, *, copy: bool) -> dict:
    from bot.limits import check_user_limits

    if copy:
        return _error("a forwarded copy cannot ask; open the original")
    chat = query.get("chat") if isinstance(query.get("chat"), dict) else {}
    chat_id, message_id = chat.get("id"), query.get("message_id")
    if chat_id is None or not isinstance(message_id, int):
        return _error("not available here")
    user_id = str(chat_id)
    if not auth.is_authorized(user_id):
        logger.warning("Widget query from an unauthorized chat %s", user_id)
        return _error("not authorized")
    data = query.get("data")
    if not isinstance(data, str) or not data:
        return _error("empty request")
    data = data[:MAX_DATA_CHARS]
    record = await asyncio.to_thread(widgets.by_message, user_id, message_id)
    if record is None or not record.get("allow_queries"):
        return _error("this widget does not take requests")
    allowed, used, limit = await asyncio.to_thread(check_user_limits, user_id)
    if not allowed:
        return _error(f"action limit reached ({used}/{limit})")

    # From here on no suspension point until the request is queued or delivered.
    _same_loop()
    key = (user_id, record["id"])
    now = time.monotonic()
    last = _last.get(key)
    _last[key] = (data, now)
    if last is not None and last[0] == data and now - last[1] < DEDUPE_SECONDS:
        return ACK
    recent = [t for t in _recent.get(key, ()) if now - t < RATE_WINDOW]
    if len(recent) >= RATE_LIMIT:
        _recent[key] = recent
        return _error("too many requests, wait a minute")
    _recent[key] = recent + [now]
    event = Event(data=data, who=_who(query.get("from")), at=datetime.now(timezone.utc).strftime("%H:%M:%S"))
    batch = _batches.get(key)
    if batch is not None and not batch.started:
        if len(batch.events) < MAX_EVENTS:
            batch.events.append(event)
        return ACK
    thread_id = record.get("thread_id")
    from bot.handlers import messages  # the running turn and the queue of the chat

    target = messages.inject_target(user_id, thread_id)
    if target is not None:
        messages.inject(
            target, bot, user_id, limit, chat_id=int(user_id), thread_id=thread_id, message_id=None,
            label=WIDGET_LABEL, kind=WIDGET_LABEL, error_text=ERROR_TEXT,
            prepared=inbox.Prepared(prompt=widget_prompt(record, [event]), kind=WIDGET_LABEL, label=WIDGET_LABEL),
        )
    else:
        batch = _Batch(user_id, record["id"], thread_id, record, [event])
        messages.queue.submit_nowait(Job(  # the chat queue (bot.runtime.queue, as messages binds it)
            user_id=user_id, label=WIDGET_LABEL, thread_id=thread_id, from_user=True,
            run=partial(_run_widget_turn, bot, batch), on_drop=partial(_forget, batch),
        ))
        _batches[key] = batch
    await asyncio.to_thread(_track, user_id, record["id"])
    return ACK


def _track(user_id: str, widget_id: str) -> None:
    """Count an accepted request (quota row, the widget's tap count). Never raises: the request is
    already queued or delivered, and the widget must not be told it failed."""
    try:
        _stats().track_message_received(user_id, "widget")
    except Exception as e:  # noqa: BLE001
        logger.error("Could not count a widget request of %s: %s", user_id, e)
    try:
        widgets.count_tap(user_id, widget_id)
    except Exception as e:  # noqa: BLE001
        logger.error("Could not record a request of widget %s of %s: %s", widget_id, user_id, e)


def _forget(batch: _Batch) -> None:
    """The queued turn was dropped (/cancel): later requests start a new one."""
    batch.started = True
    if _batches.get((batch.user_id, batch.widget_id)) is batch:
        del _batches[(batch.user_id, batch.widget_id)]


async def _run_widget_turn(bot: Any, batch: _Batch, ctx: JobContext) -> None:
    from bot.agent_runner import QUIET_MARKER, run_turn
    from bot.limits import check_user_limits

    _forget(batch)  # requests from now on join the running turn or start the next one
    record = await asyncio.to_thread(widgets.find, batch.user_id, batch.widget_id) or batch.record
    _, _, limit = await asyncio.to_thread(check_user_limits, batch.user_id)
    await run_turn(
        bot=bot, user_id=batch.user_id, chat_id=int(batch.user_id), prompt=widget_prompt(record, batch.events),
        thread_id=batch.thread_id, reply_to_message_id=record["message_id"], ctx=ctx, limit=limit,
        header=WIDGET_HEADER, reasoning_caption=WIDGET_REASONING_CAPTION, quiet_marker=QUIET_MARKER,
    )


def query_of(update: TelegramObject) -> tuple[dict | None, bool]:
    """(the raw widget query of ``update``, whether it came from a forwarded copy), or (None, False).
    Read from ``model_extra`` (aiogram 3.31) or a declared field (an aiogram that knows Bot API 10.4)."""
    for name, copy in ((UPDATE_TYPE, False), (COPY_UPDATE_TYPE, True)):
        raw = widgets.raw_field(update, name)
        if raw is not None:
            return raw, copy
    return None, False


class WidgetQueryMiddleware(BaseMiddleware):
    """Outer middleware on ``dp.update``: widget queries are handled here (aiogram does not know
    them); every other update goes on as usual."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        query, copy = query_of(event)
        if query is None:
            return await handler(event, data)
        await handle_query(data.get("bot"), query, copy=copy)
        return None
