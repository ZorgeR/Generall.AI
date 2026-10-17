"""Reply and forward context for the agent (docs/design-v3.md, F1).

Telegram keeps "this replies to that message", "the user quoted this part of it" and
"this was forwarded from there" in message metadata; the agent used to get the user's
own text alone, so a reply like "translate this" had nothing to point at.
``describe_reference`` turns the metadata into a short plain-text preamble that every
content job puts in front of its prompt (``with_reference``).

It is pure and cheap: only the text, caption and media kind of the referenced message are
used (nothing is downloaded), so a reply to a photo names the photo but does not show it.
A reply to one of the bot's own messages says so, so the agent knows which answer is meant;
the bot answers with rich messages, whose text lives in ``rich_message.blocks`` rather than
``text``, so those blocks are flattened to plain text (thinking, buttons and the tool-call
dumps of a turn summary left out). In a forum topic Telegram sets ``reply_to_message`` on
every message to the topic's creation service message; that is not a reply and is ignored.
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
    RichBlockBlockQuotation,
    RichBlockButtons,
    RichBlockDetails,
    RichBlockList,
    RichBlockMathematicalExpression,
    RichBlockTable,
    RichBlockThinking,
)

QUOTE_LIMIT = 1500  # characters of the referenced message kept in the preamble
TRUNCATED = " … (truncated)"
ASSISTANT = "you (the assistant)"
END_OF_REPLY = "[End of the replied-to message]"
# Collapsible sections of the bot's own turn summary (bot/status.py): tool-call dumps and the
# thinking transcript, not something the user means when replying to that message.
SUMMARY_DETAILS_PREFIXES = ("Tool calls (", "🔧 Tools (", "💭 Thinking")


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
    if reply is not None and _is_topic_root(message, reply):
        reply = None
    external = message.external_reply
    if reply is not None:
        header = f"[Replying to {_media_kind(reply) or 'a message'} from {_author(reply, message, bot_id)}, sent {_utc(reply.date)}]"
        text = reply.text or reply.caption or ""
        if not text and reply.rich_message is not None:
            text = _rich_text(reply.rich_message.blocks)
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


def _is_topic_root(message: Message, reply: Message) -> bool:
    """True when ``reply`` is the forum topic's creation service message, not a real reply.

    Every message in a forum topic carries it as ``reply_to_message`` (its id is the thread id).
    """
    if reply.forum_topic_created is not None:
        return True
    return bool(message.is_topic_message) and reply.message_id == message.message_thread_id


def _author(reply: Message, message: Message, bot_id: int | None) -> str:
    user = reply.from_user
    if user is not None and user.id == bot_id:
        return ASSISTANT
    if reply.sender_chat is not None:
        # A channel post, a post auto-forwarded into the linked discussion group or an anonymous
        # admin's message: ``from_user`` is then absent or a placeholder account ("Telegram",
        # "GroupAnonymousBot"), the chat title is the real sender.
        return _chat_name(reply.sender_chat)
    if user is None:
        return "an unknown sender"
    if bot_id is None and user.is_bot:
        return ASSISTANT
    asker = message.from_user
    if asker is not None and asker.id == user.id:
        return f"{user.full_name} (the user)"
    return user.full_name


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


# ---- rich messages -------------------------------------------------------------------------
def _rich_text(blocks: list) -> str:
    """The plain text of rich message blocks, one block per line; "" when nothing is readable.

    Headings, paragraphs, lists (one line per item, with its label), code, tables (cells joined
    with " | "), quotations, footers and media captions are kept. Thinking blocks, button rows
    and the collapsible tool-call / thinking sections of a turn summary are dropped; other
    details blocks contribute their summary and content.
    """
    lines: list[str] = []
    for block in blocks:
        if isinstance(block, (RichBlockThinking, RichBlockButtons)):
            continue
        if isinstance(block, RichBlockDetails):
            summary = _flatten(block.summary).strip()
            if summary.startswith(SUMMARY_DETAILS_PREFIXES):
                continue
            lines.extend(line for line in (summary, _rich_text(block.blocks)) if line)
        elif isinstance(block, RichBlockList):
            for item in block.items:
                body = _rich_text(item.blocks)
                label = (item.label or "").strip()
                lines.append(f"{label} {body}".strip() if label else body)
        elif isinstance(block, RichBlockBlockQuotation):
            lines.append(_rich_text(block.blocks))
            lines.append(_flatten(block.credit))
        elif isinstance(block, RichBlockTable):
            for row in block.cells:
                lines.append(" | ".join(_flatten(cell.text).strip() for cell in row))
            lines.append(_flatten(block.caption))
        elif isinstance(block, RichBlockMathematicalExpression):
            lines.append(block.expression)
        else:  # paragraph, heading, pre, footer, quotations: ``text``; media blocks: ``caption``
            lines.append(_flatten(getattr(block, "text", None)))
            caption = getattr(block, "caption", None)
            if caption is not None and getattr(caption, "text", None) is not None:
                lines.append(_flatten(caption.text))
    return "\n".join(line.strip() for line in lines if line and line.strip())


def _flatten(node: Any) -> str:
    """Plain text of a RichTextUnion: a str, a list of nodes or a styled node wrapping ``text``."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_flatten(n) for n in node)
    if getattr(node, "alternative_text", None) is not None:  # custom emoji
        return node.alternative_text
    if getattr(node, "expression", None) is not None:  # inline math
        return node.expression
    if getattr(node, "button", None) is not None:  # an inline button: not text
        return ""
    return _flatten(getattr(node, "text", None))


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
