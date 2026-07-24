"""Rich Messages support (Bot API 10.1 / 10.2).

Bot API 10.1 introduced a formatting model that is far richer than
``parse_mode``: ``sendRichMessage`` renders real headings, tables, checkbox
lists, footnotes, LaTeX, collapsible sections and media blocks, and
``sendRichMessageDraft`` streams a partial answer as an ephemeral preview
while it is still being generated - including a native "thinking" block.

The important part for this bot: ``InputRichMessage`` accepts the Markdown
*as written*::

    {"chat_id": 1, "rich_message": {"markdown": "# Title\\n\\n| a | b |\\n..."}}

so the model's answer no longer has to be downgraded to fit MarkdownV2.

python-telegram-bot has no typed methods for this yet (python-telegram-bot
issue #5261, milestone v23), so the calls go through ``Bot.do_api_request``.
Because of that - and because the API is newer than the deployed Bot API
server may be - every helper here reports failure instead of raising, and
the caller falls back to the MarkdownV2 renderer in ``telegram_md``. The
first "unknown method" style rejection latches rich support off for the
process, so an old server costs one failed call, not one per message.

Set ``RICH_MESSAGES_ENABLED=false`` to skip the rich path entirely.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from telegram_md import escape_html, split_markdown

logger = logging.getLogger(__name__)

#: Documented limits of a single rich message.
RICH_MAX_CHARS = 32768
RICH_MAX_BLOCKS = 500
RICH_MAX_MEDIA = 50

#: Drafts are previews; they expire after ~30 seconds unless refreshed.
RICH_DRAFT_TTL = 30

_ENV_FLAG = os.getenv("RICH_MESSAGES_ENABLED", "true").lower() not in ("false", "0", "no")

# None = not probed yet, True = server supports it, False = latched off.
_supported: Optional[bool] = None if _ENV_FLAG else False

_UNSUPPORTED_HINTS = (
    "method not found",
    "unknown method",
    "not supported",
    "unsupported method",
    "method is not available",
)


def rich_enabled() -> bool:
    """Whether the rich path is worth trying."""
    return _supported is not False


def _mark_unsupported(reason: str) -> None:
    global _supported
    if _supported is not False:
        logger.warning("Rich Messages unavailable, falling back to MarkdownV2: %s", reason)
    _supported = False


def _mark_supported() -> None:
    global _supported
    if _supported is None:
        logger.info("Rich Messages (Bot API 10.1+) are available")
    _supported = True


def _is_unsupported_error(error: Exception) -> bool:
    """Distinguish 'this server has no such method' from 'this message is bad'."""
    message = str(error).lower()
    return any(hint in message for hint in _UNSUPPORTED_HINTS)


def build_rich_message(markdown: str, skip_entity_detection: bool = False) -> dict:
    """Build an ``InputRichMessage`` from Markdown source."""
    payload = {"markdown": markdown}
    if skip_entity_detection:
        payload["skip_entity_detection"] = True
    return payload


def build_thinking_message(text: str) -> dict:
    """Build an ``InputRichMessage`` holding a single thinking block.

    Thinking blocks exist only in drafts - they are never part of a stored
    message - which is exactly the bot's "💭 Thinking..." status.
    """
    return {"html": f"<tg-thinking>{escape_html(text)}</tg-thinking>"}


def split_rich(markdown: str) -> list:
    """Split Markdown into pieces that fit one rich message."""
    text = (markdown or "").strip()
    if not text:
        return []
    if len(text) <= RICH_MAX_CHARS:
        return [text]
    return split_markdown(text, RICH_MAX_CHARS)


def _message_type():
    """python-telegram-bot's Message class, or None when PTB is absent."""
    try:
        from telegram import Message
    except ImportError:  # pragma: no cover - only in unit tests
        return None
    return Message


async def _call(bot, method: str, payload: dict, as_message: bool = False):
    """Invoke a Bot API method that python-telegram-bot does not wrap yet."""
    kwargs = {"api_kwargs": payload}
    return_type = _message_type() if as_message else None
    if return_type is not None:
        kwargs["return_type"] = return_type
    return await bot.do_api_request(method, **kwargs)


async def send_rich_message(bot, chat_id, markdown: str, **params) -> Optional[list]:
    """Send ``markdown`` as one or more rich messages.

    Returns the sent messages, or ``None`` when the rich path is unavailable
    and the caller should fall back.
    """
    if not rich_enabled():
        return None

    pieces = split_rich(markdown)
    if not pieces:
        return []

    sent = []
    for index, piece in enumerate(pieces):
        payload = {"chat_id": chat_id, "rich_message": build_rich_message(piece)}
        # Reply/markup parameters belong on the first message only.
        payload.update(params if index == 0 else {
            key: value for key, value in params.items()
            if key in ("message_thread_id", "disable_notification", "protect_content")
        })
        try:
            sent.append(await _call(bot, "sendRichMessage", payload, as_message=True))
        except Exception as error:
            if _is_unsupported_error(error):
                _mark_unsupported(str(error))
            else:
                logger.warning("sendRichMessage failed: %s", error)
            return sent or None
        _mark_supported()
    return sent


async def edit_rich_message(bot, chat_id, message_id: int, markdown: str, **params) -> Optional[list]:
    """Turn an existing message into a rich one, sending any overflow after it.

    Used to replace the "Thinking..." status with the finished answer.
    """
    if not rich_enabled():
        return None

    pieces = split_rich(markdown)
    if not pieces:
        return []

    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "rich_message": build_rich_message(pieces[0]),
    }
    payload.update(params)
    try:
        edited = await _call(bot, "editMessageText", payload, as_message=True)
    except Exception as error:
        if _is_unsupported_error(error):
            _mark_unsupported(str(error))
        else:
            logger.warning("editMessageText with rich_message failed: %s", error)
        return None
    _mark_supported()

    sent = [edited]
    for piece in pieces[1:]:
        follow_up = await send_rich_message(bot, chat_id, piece, **params)
        if not follow_up:
            break
        sent.extend(follow_up)
    return sent


async def send_rich_draft(bot, chat_id, draft_id: int, markdown: str, thinking: bool = False, **params) -> bool:
    """Stream a partial answer as a rich draft preview.

    Drafts work in private chats only and need a non-zero ``draft_id``;
    reusing the same id animates the update in place. The draft is a preview -
    the final answer still has to be sent with ``sendRichMessage``.
    """
    if not rich_enabled():
        return False

    text = (markdown or "").strip()
    if not text:
        return False
    if len(text) > RICH_MAX_CHARS:
        text = text[:RICH_MAX_CHARS]

    payload = {
        "chat_id": int(chat_id),
        "draft_id": draft_id or 1,
        "rich_message": build_thinking_message(text) if thinking else build_rich_message(text),
    }
    payload.update(params)
    try:
        await _call(bot, "sendRichMessageDraft", payload)
    except Exception as error:
        if _is_unsupported_error(error):
            _mark_unsupported(str(error))
        else:
            logger.debug("sendRichMessageDraft failed: %s", error)
        return False
    _mark_supported()
    return True
