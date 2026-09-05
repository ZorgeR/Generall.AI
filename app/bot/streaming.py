"""Draft-message streaming of the agent's partial output.

Telegram drafts are an ephemeral preview; the final answer is always sent as
a normal message afterwards. The callback is throttled but always flushes the
last chunk, so the preview never stalls on stale text.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from aiogram import Bot

from bot.config import config

logger = logging.getLogger(__name__)

DRAFT_THROTTLE = 0.3
DRAFT_MAX = 4000

StreamCallback = Callable[[str, bool], Awaitable[None]]


def create_streaming_callback(
    bot: Bot,
    chat_id: int,
    thread_id: int | None = None,
    *,
    enabled: bool | None = None,
) -> StreamCallback | None:
    if not (config.streaming_enabled if enabled is None else enabled):
        return None

    state = {"last": 0.0, "latest": None, "pending": None}

    async def _flush() -> None:
        item = state["latest"]
        if item is None:
            return
        text, is_thinking = item
        display = text[:DRAFT_MAX]
        if is_thinking:
            display = f"💭 {display}"[:DRAFT_MAX]
        state["last"] = time.monotonic()
        try:
            kwargs = {"chat_id": int(chat_id), "draft_id": 1, "text": display}
            if thread_id:
                kwargs["message_thread_id"] = thread_id
            await bot.send_message_draft(**kwargs)
        except Exception as e:  # noqa: BLE001 - drafts are best-effort
            logger.debug("send_message_draft failed: %s", e)

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
