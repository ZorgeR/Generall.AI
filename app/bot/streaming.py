"""Draft-message streaming of the agent's partial output.

Telegram drafts are an ephemeral preview; the final answer is always sent as
a normal message afterwards. The callback is throttled but always flushes the
last chunk, so the preview never stalls on stale text.

With rich messages enabled the draft is a rich draft (``sendRichMessageDraft``):
the answer streams with live Markdown rendering and the model's thinking shows
in Telegram's native "Thinking…" block. When the Bot API server has no rich
support the plain ``sendMessageDraft`` is used, exactly as before.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound

from bot import rich as rich_render
from bot.config import config

logger = logging.getLogger(__name__)

DRAFT_THROTTLE = 0.3
DRAFT_MAX = 4000
DRAFT_ID = 1

StreamCallback = Callable[[str, bool], Awaitable[None]]


def create_streaming_callback(
    bot: Bot,
    chat_id: int,
    thread_id: int | None = None,
    *,
    enabled: bool | None = None,
    rich: bool = False,
) -> StreamCallback | None:
    if not (config.streaming_enabled if enabled is None else enabled):
        return None
    rich_mode = rich  # the module is imported as rich_render to avoid shadowing

    state = {"last": 0.0, "latest": None, "pending": None}
    thread_kw = {"message_thread_id": thread_id} if thread_id else {}

    async def _send_plain(text: str, is_thinking: bool) -> None:
        display = text[:DRAFT_MAX]
        if is_thinking:
            display = f"💭 {display}"[:DRAFT_MAX]
        await bot.send_message_draft(chat_id=int(chat_id), draft_id=DRAFT_ID, text=display, **thread_kw)

    async def _send_rich(text: str, is_thinking: bool) -> bool:
        """Returns False when the caller should fall back to a plain draft."""
        payload = rich_render.thinking_draft(text) if is_thinking else rich_render.text_draft(text)
        try:
            await bot.send_rich_message_draft(chat_id=int(chat_id), draft_id=DRAFT_ID, rich_message=payload, **thread_kw)
            return True
        except (TelegramBadRequest, TelegramNotFound) as e:
            if rich_render.is_unsupported_error(e):
                rich_render.mark_unavailable(str(e))
            else:
                logger.debug("Rich draft rejected, sending plain draft: %s", e)
            return False

    async def _flush() -> None:
        item = state["latest"]
        if item is None:
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
        if not text:
            return
        state["latest"] = (text, is_thinking)
        since = time.monotonic() - state["last"]
        if since >= DRAFT_THROTTLE:
            await _flush()
        elif state["pending"] is None:
            state["pending"] = asyncio.create_task(_delayed_flush(DRAFT_THROTTLE - since))

    return on_text_chunk
