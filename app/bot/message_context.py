"""Reply and forward context for the agent (docs/design-v3.md, F1).

Telegram keeps "this replies to that message", "the user quoted this part of it" and
"this was forwarded from there" in message metadata; the agent used to get the user's
own text alone, so a reply like "translate this" had nothing to point at.
``describe_reference`` turns the metadata into a short plain-text preamble that every
content job puts in front of its prompt (``with_reference``).

It is pure and cheap: only the text, caption and media kind of the referenced message are
used (nothing is downloaded), so a reply to a photo names the photo but does not show it.
A reply to one of the bot's own messages says so, so the agent knows which answer is meant.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aiogram.types import (
    Chat,
    Message,
    MessageOriginChannel,
    MessageOriginChat,
    MessageOriginHiddenUser,
    MessageOriginUser,
)

QUOTE_LIMIT = 1500  # characters of the referenced message kept in the preamble
TRUNCATED = " … (truncated)"
ASSISTANT = "you (the assistant)"
END_OF_REPLY = "[End of the replied-to message]"


def describe_reference(message: Message, bot_id: int | None) -> str:
    """The reply / forward preamble of ``message``, or "" when it is neither.

    Plain text, no Markdown::

        [Replying to a message from Alice Smith (the user), sent 2026-10-17 12:00 UTC]
        <its text or caption, up to QUOTE_LIMIT characters>
        [The user quoted this part of it: "..."]
        [End of the replied-to message]
        [Forwarded from Some Channel, signed Editor (2026-10-17 12:00 UTC)]

    ``bot_id`` is ``Bot.id``; without it a message from any bot counts as the assistant's.
    """
    parts = [_describe_reply(message, bot_id), _describe_forward(message.forward_origin)]
    return "\n".join(p for p in parts if p)


def with_reference(prompt: str, message: Message, bot_id: int | None) -> str:
    """``prompt`` with the preamble in front, separated by a blank line; ``prompt`` as is without one."""
    preamble = describe_reference(message, bot_id)
    return f"{preamble}\n\n{prompt}" if preamble else prompt


# ---- replies -------------------------------------------------------------------------------
def _describe_reply(message: Message, bot_id: int | None) -> str:
    reply = message.reply_to_message
    external = message.external_reply
    if reply is not None:
        header = f"[Replying to {_media_kind(reply) or 'a message'} from {_author(reply, message, bot_id)}, sent {_utc(reply.date)}]"
        text = reply.text or reply.caption or ""
    elif external is not None:
        # A reply to a message of another chat: Telegram gives its origin and media, never its text.
        name, _signature = _origin_source(external.origin)
        header = f"[Replying to {_media_kind(external) or 'a message'} from {name}, sent {_utc(external.origin.date)}, in another chat]"
        text = ""
    else:
        return ""
    lines = [header]
    if text.strip():
        lines.append(_cap(text))
    if message.quote is not None and message.quote.text.strip():
        lines.append(f'[The user quoted this part of it: "{_cap(message.quote.text)}"]')
    lines.append(END_OF_REPLY)
    return "\n".join(lines)


def _author(reply: Message, message: Message, bot_id: int | None) -> str:
    user = reply.from_user
    if user is not None:
        if user.id == bot_id or (bot_id is None and user.is_bot):
            return ASSISTANT
        asker = message.from_user
        if asker is not None and asker.id == user.id:
            return f"{user.full_name} (the user)"
        return user.full_name
    if reply.sender_chat is not None:  # a channel post or an anonymous group admin
        return _chat_name(reply.sender_chat)
    return "an unknown sender"


def _media_kind(m: Any) -> str:
    """What the referenced message carries besides text; "" for a plain text message.

    ``m`` is a Message or an ExternalReplyInfo (same media fields). An animation also sets
    ``document`` for backwards compatibility, so it is checked first.
    """
    if m.animation is not None:
        return "an animation"
    if m.photo:
        return "a photo"
    if m.document is not None:
        return f"a document: {m.document.file_name}" if m.document.file_name else "a document"
    if m.voice is not None:
        return "a voice message"
    if m.video is not None:
        return "a video"
    if m.video_note is not None:
        return "a video note"
    if m.sticker is not None:
        return f"a sticker {m.sticker.emoji}" if m.sticker.emoji else "a sticker"
    if m.location is not None:
        return f"a location {m.location.latitude}, {m.location.longitude}"
    if m.audio is not None:
        title = m.audio.title or m.audio.file_name
        return f"an audio file: {title}" if title else "an audio file"
    if m.poll is not None:
        return f"a poll: {m.poll.question}"
    if m.contact is not None:
        name = " ".join(p for p in (m.contact.first_name, m.contact.last_name) if p)
        return f"a contact: {name}" if name else "a contact"
    return ""


# ---- forwards ------------------------------------------------------------------------------
def _describe_forward(origin: Any) -> str:
    if origin is None:
        return ""
    name, signature = _origin_source(origin)
    signed = f", signed {signature}" if signature else ""
    return f"[Forwarded from {name}{signed} ({_utc(origin.date)})]"


def _origin_source(origin: Any) -> tuple[str, str | None]:
    """(name or title, author signature) of a MessageOrigin*."""
    if isinstance(origin, MessageOriginUser):
        return origin.sender_user.full_name, None
    if isinstance(origin, MessageOriginHiddenUser):
        return origin.sender_user_name, None
    if isinstance(origin, MessageOriginChat):
        return _chat_name(origin.sender_chat), origin.author_signature
    if isinstance(origin, MessageOriginChannel):
        return _chat_name(origin.chat), origin.author_signature
    return "an unknown source", None


# ---- small helpers -------------------------------------------------------------------------
def _chat_name(chat: Chat) -> str:
    if chat.title:
        return chat.title
    name = " ".join(p for p in (chat.first_name, chat.last_name) if p)
    return name or (f"@{chat.username}" if chat.username else "an unknown chat")


def _utc(value: datetime | None) -> str:
    if value is None:
        return "an unknown time"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _cap(text: str) -> str:
    text = text.strip()
    if len(text) <= QUOTE_LIMIT:
        return text
    return text[:QUOTE_LIMIT].rstrip() + TRUNCATED
