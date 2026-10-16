"""Draft-message streaming of the agent's partial output.

Telegram drafts are an ephemeral preview; the final answer is always sent as
a normal message afterwards. The callback is throttled but always flushes the
last chunk, so the preview never stalls on stale text.

With rich messages enabled the draft is a rich draft (``sendRichMessageDraft``):
the answer streams with live Markdown rendering and the model's thinking shows
in Telegram's native "Thinking…" block. When the Bot API server has no rich
support the plain ``sendMessageDraft`` is used, exactly as before.

Every draft of a turn uses one ``draft_id`` (the queue job id when the turn runs
as a job). With ``can_stop`` the drafts carry Telegram's own Stop button; pressing
it delivers a ``stopped_message_generation`` update naming that draft id, which
``bot/handlers/messages.py`` turns into ``QueueManager.cancel_current``.

The answer's trailing ``<follow_ups>`` block becomes buttons under the final
message (``bot/turn_actions.py``), so drafts and ``partial_text()`` never show it,
not even a half-typed opening tag.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import Awaitable, Callable

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound

from bot import rich as rich_render
from bot.config import config
from bot.turn_actions import strip_follow_ups_draft

logger = logging.getLogger(__name__)

DRAFT_THROTTLE = 0.3
DRAFT_MAX = 4000

# The returned callback is called as ``await cb(text, is_thinking)`` with the text so far
# (cumulative within one model call) and also carries the attributes documented in
# ``create_streaming_callback``.
StreamCallback = Callable[[str, bool], Awaitable[None]]


def random_draft_id() -> int:
    """A random non-zero 31-bit draft id (Telegram rejects 0)."""
    return secrets.randbelow(2**31 - 1) + 1


def create_streaming_callback(
    bot: Bot,
    chat_id: int,
    thread_id: int | None = None,
    *,
    enabled: bool | None = None,
    rich: bool = False,
    draft_id: int | None = None,
    can_stop: bool = False,
) -> StreamCallback | None:
    """The draft callback for one turn, or None when streaming is off.

    ``draft_id`` is used for every draft of the turn (default: a random non-zero id);
    ``can_stop`` shows Telegram's Stop button on the drafts and keeps the draft on
    screen briefly after it is pressed (``can_stop`` + ``keep_on_stop``).

    The returned function also exposes:

    - ``draft_id`` (int): the draft id in use;
    - ``partial_text()`` -> str: the latest answer text streamed so far, thinking
      and the follow-ups block excluded ("" if none), e.g. to keep a stopped answer
      as a real message;
    - ``close()``: cancel a pending throttled flush and ignore later chunks, so no
      draft reappears once the turn has delivered (or abandoned) its answer.
    """
    if not (config.streaming_enabled if enabled is None else enabled):
        return None
    rich_mode = rich  # the module is imported as rich_render to avoid shadowing
    draft_id = draft_id or random_draft_id()

    state = {"last": 0.0, "latest": None, "pending": None, "text": "", "closed": False}
    thread_kw = {"message_thread_id": thread_id} if thread_id else {}
    stop_kw = {"can_stop": True, "keep_on_stop": True} if can_stop else {}

    async def _send_plain(text: str, is_thinking: bool) -> None:
        display = text[:DRAFT_MAX]
        if is_thinking:
            display = f"💭 {display}"[:DRAFT_MAX]
        await bot.send_message_draft(chat_id=int(chat_id), draft_id=draft_id, text=display, **thread_kw, **stop_kw)

    async def _send_rich(text: str, is_thinking: bool) -> bool:
        """Returns False when the caller should fall back to a plain draft."""
        payload = rich_render.thinking_draft(text) if is_thinking else rich_render.text_draft(text)
        try:
            await bot.send_rich_message_draft(
                chat_id=int(chat_id), draft_id=draft_id, rich_message=payload, **thread_kw, **stop_kw
            )
            return True
        except (TelegramBadRequest, TelegramNotFound) as e:
            if rich_render.is_unsupported_error(e):
                rich_render.mark_unavailable(str(e))
            else:
                logger.debug("Rich draft rejected, sending plain draft: %s", e)
            return False

    async def _flush() -> None:
        item = state["latest"]
        if item is None or state["closed"]:
            return
        text, is_thinking = item
        state["last"] = time.monotonic()
        try:
            if rich_mode and rich_render.is_available() and await _send_rich(text, is_thinking):
                return
            await _send_plain(text, is_thinking)
        except Exception as e:  # noqa: BLE001 - drafts are best-effort
            logger.debug("Draft update failed: %s", e)

    async def _delayed_flush(delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            state["pending"] = None
            await _flush()
        except asyncio.CancelledError:
            pass

    async def on_text_chunk(text: str, is_thinking: bool = False) -> None:
        if not text or state["closed"]:
            return
        if not is_thinking:
            state["text"] = text
            text = strip_follow_ups_draft(text)
            if not text or state["latest"] == (text, False):
                return  # nothing visible yet, or only the hidden follow-ups block grew
        state["latest"] = (text, is_thinking)
        since = time.monotonic() - state["last"]
        if since >= DRAFT_THROTTLE:
            await _flush()
        elif state["pending"] is None:
            state["pending"] = asyncio.create_task(_delayed_flush(DRAFT_THROTTLE - since))

    def partial_text() -> str:
        return strip_follow_ups_draft(state["text"])

    def close() -> None:
        state["closed"] = True
        pending = state["pending"]
        if pending is not None and not pending.done():
            pending.cancel()
        state["pending"] = None

    on_text_chunk.draft_id = draft_id
    on_text_chunk.partial_text = partial_text
    on_text_chunk.close = close
    return on_text_chunk
