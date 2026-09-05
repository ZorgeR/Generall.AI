"""ChatSender: the one object tools and handlers use to talk to a chat.

It carries the bot, chat id and forum thread id, so code that sends never
depends on an incoming Telegram update. That is what lets agent reminders and
tool calls send messages without a mock update.

For now text goes out as legacy Markdown with a plain-text fallback, exactly
like the previous implementation; ``send_markdown`` is the single place the
rich-message renderer will plug into.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import BufferedInputFile, FSInputFile, InlineKeyboardMarkup, Message, ReactionTypeEmoji

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
    ) -> None:
        self.bot = bot
        self.chat_id = int(chat_id)
        self.thread_id = thread_id
        self.reply_to_message_id = reply_to_message_id

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

        Short answers replace ``edit`` in place when given; long answers are
        sent as new messages in order and ``edit`` becomes a short notice.
        Each chunk falls back to plain text on its own, so nothing is sent twice.
        """
        if not text or not text.strip():
            text = "🤖 *No response from the AI.*"
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
