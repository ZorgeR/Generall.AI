"""ChatSender: the one object tools and handlers use to talk to a chat.

It carries the bot, chat id and forum thread id, so code that sends never
depends on an incoming Telegram update. That is what lets agent reminders and
tool calls send messages without a mock update.

LLM answers go through ``send_markdown``. With ``rich=True`` (the per-user
``rich_messages`` setting, on by default) they are sent as Telegram rich
messages: native GitHub-flavored Markdown with headings, tables, code blocks
and math. When the Bot API server does not support rich messages, or a message
is rejected, the text degrades tier by tier (rich HTML → MarkdownV2 → legacy
Markdown → raw text, see ``bot/rich.py``) and nothing is ever sent twice.
Static UI text keeps using ``send_text`` (legacy Markdown, raw fallback).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.types import BufferedInputFile, FSInputFile, InlineKeyboardMarkup, Message, ReactionTypeEmoji

from bot import rich as rich_render
from bot.ui import LEGACY_MARKDOWN

logger = logging.getLogger(__name__)

MAX_TEXT = 4000
MAX_CAPTION = 1024
SPLIT_NOTICE = "📨 The answer was long, so it was sent in several parts above."


def split_text_intelligently(text: str, max_length: int = MAX_TEXT) -> list[str]:
    """Split text into chunks at paragraph, line, or word boundaries."""
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        if start + max_length >= len(text):
            chunks.append(text[start:])
            break
        end = start + max_length
        cut = text.rfind("\n\n", start, end)
        if cut != -1 and cut > start:
            chunks.append(text[start:cut + 2])
            start = cut + 2
            continue
        cut = text.rfind("\n", start, end)
        if cut != -1 and cut > start:
            chunks.append(text[start:cut + 1])
            start = cut + 1
            continue
        cut = text.rfind(" ", start, end)
        if cut != -1 and cut > start:
            chunks.append(text[start:cut + 1])
            start = cut + 1
            continue
        chunks.append(text[start:end])
        start = end
    return [c for c in chunks if c]


def _caption(caption: str | None) -> str | None:
    if caption is None:
        return None
    return caption if len(caption) <= MAX_CAPTION else caption[: MAX_CAPTION - 1] + "…"


def _input_file(source: str | Path | bytes, filename: str | None):
    if isinstance(source, (bytes, bytearray)):
        return BufferedInputFile(bytes(source), filename=filename or "file")
    return FSInputFile(str(source), filename=filename)


class ChatSender:
    def __init__(
        self,
        bot: Bot,
        chat_id: int,
        thread_id: int | None = None,
        reply_to_message_id: int | None = None,
        rich: bool = False,
    ) -> None:
        self.bot = bot
        self.chat_id = int(chat_id)
        self.thread_id = thread_id
        self.reply_to_message_id = reply_to_message_id
        self.rich = rich  # render LLM answers as rich messages (per-user setting)

    def _kw(self) -> dict:
        kw: dict = {"chat_id": self.chat_id}
        if self.thread_id:
            kw["message_thread_id"] = self.thread_id
        return kw

    # ---- text ------------------------------------------------------------
    async def send_text(
        self,
        text: str,
        *,
        markdown: bool = True,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        text = text if text else "…"
        if markdown:
            try:
                return await self.bot.send_message(text=text, parse_mode=LEGACY_MARKDOWN, reply_markup=reply_markup, **self._kw())
            except TelegramBadRequest as e:
                logger.debug("Markdown send failed, falling back to raw text: %s", e)
        # Raw text on fallback: LLM output full of snake_case and asterisks must reach the
        # user untouched rather than with its underscores stripped.
        return await self.bot.send_message(text=text, reply_markup=reply_markup, **self._kw())

    async def edit_text(self, message: Message, text: str, *, markdown: bool = True) -> Message | None:
        """Best-effort edit: never raises (status edits must not break a turn or a cancel)."""
        text = text if text else "…"
        try:
            if markdown:
                try:
                    return await self.bot.edit_message_text(
                        chat_id=self.chat_id, message_id=message.message_id, text=text, parse_mode=LEGACY_MARKDOWN
                    )
                except TelegramBadRequest as e:
                    if "message is not modified" in str(e).lower():
                        return None
            return await self.bot.edit_message_text(chat_id=self.chat_id, message_id=message.message_id, text=text)
        except TelegramBadRequest as e:
            if "message is not modified" not in str(e).lower():
                logger.warning("edit_text failed: %s", e)
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("edit_text failed: %s", e)
            return None

    async def send_markdown(self, text: str, *, edit: Message | None = None) -> list[Message]:
        """Deliver a (possibly long) Markdown answer.

        Rich mode: the answer is sent as one or more rich messages (32 KB each)
        and the status message ``edit`` is deleted afterwards, because a plain
        message cannot be edited into a rich one. Every piece degrades on its
        own through the tiers in ``bot/rich.py``, so nothing is sent twice.

        Legacy mode: short answers replace ``edit`` in place; long answers are
        sent as new messages in order and ``edit`` becomes a short notice.
        """
        if not text or not text.strip():
            text = "🤖 *No response from the AI.*"
        if not (self.rich and rich_render.is_available()):
            return await self._send_legacy(text, edit)
        sent: list[Message] = []
        for chunk in rich_render.split_markdown(text):
            sent.extend(await self._send_formatted(chunk))
        if edit is not None:
            await self.delete([edit])
        return sent

    async def _send_formatted(self, chunk: str, depth: int = 0) -> list[Message]:
        """Send one ≤32 KB piece with the best tier this server and text allow."""
        if rich_render.is_available():
            try:
                return [await self.bot.send_rich_message(rich_message=rich_render.markdown_message(chunk), **self._kw())]
            except (TelegramBadRequest, TelegramNotFound) as e:
                if rich_render.is_unsupported_error(e):
                    rich_render.mark_unavailable(str(e))
                else:
                    logger.info("Rich Markdown rejected (%s); retrying as rich HTML", e)
                    sent = await self._send_rich_html(chunk, depth)
                    if sent:
                        return sent
        sent = await self._send_markdown_v2(chunk)
        if sent:
            return sent
        return [await self.send_text(piece) for piece in split_text_intelligently(chunk)]

    async def _send_rich_html(self, chunk: str, depth: int) -> list[Message]:
        try:
            payload = rich_render.html_message(chunk)
        except Exception as e:  # noqa: BLE001 - optional dependency / parser failure
            logger.debug("Rich HTML conversion failed: %s", e)
            return []
        if payload is None:
            # Too big for one HTML message: retry the halves separately (each may
            # still pass as plain rich Markdown), bounded so a pathological text ends.
            halves = rich_render.halve_markdown(chunk)
            if len(halves) < 2 or depth >= 6:
                return []
            sent: list[Message] = []
            for half in halves:
                sent.extend(await self._send_formatted(half, depth + 1))
            return sent
        try:
            return [await self.bot.send_rich_message(rich_message=payload, **self._kw())]
        except (TelegramBadRequest, TelegramNotFound) as e:
            if rich_render.is_unsupported_error(e):
                rich_render.mark_unavailable(str(e))
            else:
                logger.warning("Rich HTML rejected too (%s); falling back to MarkdownV2", e)
            return []

    async def _send_markdown_v2(self, chunk: str) -> list[Message]:
        try:
            pieces = rich_render.markdown_v2_chunks(chunk)
        except Exception as e:  # noqa: BLE001 - telegramify-markdown missing or failed
            logger.debug("MarkdownV2 conversion unavailable: %s", e)
            return []
        sent: list[Message] = []
        for piece in pieces:
            try:
                sent.append(await self.bot.send_message(text=piece, parse_mode="MarkdownV2", **self._kw()))
            except TelegramBadRequest as e:
                logger.info("MarkdownV2 rejected (%s)", e)
                if not sent:
                    return []  # nothing delivered yet: the legacy tier takes the whole chunk
                sent.append(await self.bot.send_message(text=rich_render.unescape_markdown_v2(piece), **self._kw()))
        return sent

    async def _send_legacy(self, text: str, edit: Message | None) -> list[Message]:
        if edit is not None and len(text) <= MAX_TEXT:
            edited = await self.edit_text(edit, text)
            return [edited or edit]
        sent: list[Message] = []
        for chunk in split_text_intelligently(text):
            sent.append(await self.send_text(chunk))
        if edit is not None:
            await self.edit_text(edit, SPLIT_NOTICE, markdown=False)
        return sent

    # ---- media -----------------------------------------------------------
    async def send_document(
        self,
        source: str | Path | bytes,
        *,
        filename: str | None = None,
        caption: str | None = None,
    ) -> Message:
        return await self.bot.send_document(document=_input_file(source, filename), caption=_caption(caption), **self._kw())

    async def send_photo(self, source: str | Path | bytes, *, filename: str | None = None, caption: str | None = None) -> Message:
        return await self.bot.send_photo(photo=_input_file(source, filename), caption=_caption(caption), **self._kw())

    async def send_video(self, source: str | Path | bytes, *, filename: str | None = None, caption: str | None = None) -> Message:
        return await self.bot.send_video(
            video=_input_file(source, filename), caption=_caption(caption), supports_streaming=True, **self._kw()
        )

    async def send_voice(self, data: bytes, *, filename: str = "voice.mp3", caption: str | None = None) -> Message:
        return await self.bot.send_voice(voice=BufferedInputFile(data, filename=filename), caption=_caption(caption), **self._kw())

    # ---- misc ------------------------------------------------------------
    async def react(self, emoji: str) -> bool:
        if not self.reply_to_message_id:
            return False
        await self.bot.set_message_reaction(
            chat_id=self.chat_id, message_id=self.reply_to_message_id, reaction=[ReactionTypeEmoji(emoji=emoji)]
        )
        return True

    async def typing(self) -> None:
        try:
            await self.bot.send_chat_action(action=ChatAction.TYPING, **self._kw())
        except Exception:  # noqa: BLE001
            pass

    async def delete(self, messages: Iterable[Message | None]) -> None:
        for m in messages:
            if m is None:
                continue
            try:
                await self.bot.delete_message(chat_id=self.chat_id, message_id=m.message_id)
            except Exception:  # noqa: BLE001
                pass
