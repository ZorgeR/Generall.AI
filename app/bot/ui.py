"""Small helpers for static UI text sent with legacy Markdown and a plain fallback."""
from __future__ import annotations

import logging
import re

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardMarkup, Message

logger = logging.getLogger(__name__)

LEGACY_MARKDOWN = "Markdown"
_MD_CHARS = re.compile(r"[*_`]")


def strip_markdown(text: str) -> str:
    return _MD_CHARS.sub("", text)


def escape_markdown(text: str) -> str:
    """Escape user-provided text for legacy Markdown v1."""
    return re.sub(r"([_*`\[])", r"\\\1", text)


def _is_not_modified(error: TelegramBadRequest) -> bool:
    return "message is not modified" in str(error).lower()


async def answer_md(message: Message, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> Message:
    try:
        return await message.answer(text, parse_mode=LEGACY_MARKDOWN, reply_markup=reply_markup)
    except TelegramBadRequest:
        return await message.answer(strip_markdown(text), reply_markup=reply_markup)


async def edit_md(message: Message, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> Message | None:
    try:
        return await message.edit_text(text, parse_mode=LEGACY_MARKDOWN, reply_markup=reply_markup)
    except TelegramBadRequest as e:
        if _is_not_modified(e):
            return None
        try:
            return await message.edit_text(strip_markdown(text), reply_markup=reply_markup)
        except TelegramBadRequest as e2:
            if _is_not_modified(e2):
                return None
            logger.warning("edit_md failed: %s", e2)
            return None


async def delete_quietly(message: Message | None) -> None:
    if message is None:
        return
    try:
        await message.delete()
    except Exception:  # noqa: BLE001
        pass
