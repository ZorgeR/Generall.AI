"""Rendering tiers for LLM answers.

Telegram Bot API 10.1 added *rich messages* (``sendRichMessage``): real
GitHub-flavored Markdown with headings, tables, task lists, code blocks and
math, rendered natively by the client. This module holds the pure conversion
helpers used by ``ChatSender.send_markdown`` / ``bot.streaming`` and the
process-wide "does this Bot API server support rich messages?" flag.

Tiers, best first:

1. rich Markdown   ``InputRichMessage(markdown=...)``: Telegram parses the model's Markdown.
2. rich HTML       the same text converted to Telegram Rich HTML by telegramify-markdown,
                   for text Telegram's own parser rejects.
3. MarkdownV2      entity formatting via telegramify-markdown (tables become monospace
                   blocks); used when the API server has no rich support, e.g. a local
                   telegram-bot-api sidecar older than Bot API 10.1.
4. legacy Markdown / raw text: what the bot always did (``ChatSender.send_text``).

The unsupported flag is sticky for the process: a 404 from ``sendRichMessage``
means the Bot API server does not know the method, so later sends skip straight
to tier 3 instead of paying a failed round-trip per message. Users can also turn
rich messages off per chat in ``/settings`` (``rich_messages.enabled``).
"""
from __future__ import annotations

import html as _html
import logging
import re

from dataclasses import dataclass, field
from pathlib import Path

from aiogram.exceptions import TelegramAPIError, TelegramNotFound
from aiogram.types import FSInputFile, InputMediaPhoto, InputMediaVideo, InputRichMessage, InputRichMessageMedia

logger = logging.getLogger(__name__)

RICH_BYTE_LIMIT = 32768  # UTF-8 bytes of Markdown/HTML per rich message (Bot API limit)
RICH_BLOCK_LIMIT = 500  # top-level blocks per rich message
DRAFT_BYTE_LIMIT = 30000  # keep streaming drafts comfortably under the limit
MDV2_MAX = 4096  # UTF-16 units per regular message

_state: dict[str, object] = {"available": True, "reason": None}


# ---- availability flag ------------------------------------------------------
def is_available() -> bool:
    return bool(_state["available"])


def unavailable_reason() -> str | None:
    return _state["reason"]  # type: ignore[return-value]


def mark_unavailable(reason: str) -> None:
    if _state["available"]:
        logger.warning("Rich messages are not supported by this Bot API server, falling back to MarkdownV2: %s", reason)
    _state["available"] = False
    _state["reason"] = reason


def reset() -> None:
    """Forget a previous 'unsupported' verdict (tests, or after upgrading the sidecar)."""
    _state["available"] = True
    _state["reason"] = None


def is_unsupported_error(error: BaseException) -> bool:
    """True when the API server rejected the *method*, not this particular message."""
    if isinstance(error, TelegramNotFound):
        return True
    if isinstance(error, TelegramAPIError):
        msg = str(error).lower()
        return "method" in msg and ("not found" in msg or "unknown" in msg)
    return False


# ---- conversions ------------------------------------------------------------
def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def truncate_bytes(text: str, limit: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", errors="ignore")


def _split_by_bytes(text: str, limit: int) -> list[str]:
    """Paragraph-boundary split keeping every piece under ``limit`` UTF-8 bytes."""
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        piece = para + "\n\n"
        if _byte_len(piece) > limit:  # a single huge paragraph: hard cut
            if current:
                chunks.append(current)
                current = ""
            while piece:
                head = truncate_bytes(piece, limit)
                chunks.append(head)
                piece = piece[len(head):]
            continue
        if current and _byte_len(current) + _byte_len(piece) > limit:
            chunks.append(current)
            current = ""
        current += piece
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def split_markdown(text: str) -> list[str]:
    """Split source Markdown into pieces that each fit one rich message."""
    if _byte_len(text) <= RICH_BYTE_LIMIT:
        return [text]
    try:
        from telegramify_markdown import InputRichMessage as _TmRich
        from telegramify_markdown import split_rich

        parts = split_rich(_TmRich(markdown=text), byte_limit=RICH_BYTE_LIMIT, block_limit=RICH_BLOCK_LIMIT)
        chunks = [p.markdown for p in parts if p.markdown and p.markdown.strip()]
        if chunks:
            return chunks
    except Exception as e:  # noqa: BLE001 - optional dependency / parser failure
        logger.debug("split_rich failed, using paragraph split: %s", e)
    return _split_by_bytes(text, RICH_BYTE_LIMIT)


def halve_markdown(text: str) -> list[str]:
    """Split text in two at the paragraph boundary nearest the middle (for retrying a rejected piece)."""
    mid = len(text) // 2
    cut = text.rfind("\n\n", 0, mid)
    if cut <= 0:
        cut = text.find("\n\n", mid)
    if cut <= 0:
        cut = text.rfind("\n", 0, mid)
    if cut <= 0:
        return [text]
    left, right = text[:cut].strip(), text[cut:].strip()
    return [p for p in (left, right) if p] or [text]


def markdown_message(text: str, media: list[InputRichMessageMedia] | None = None) -> InputRichMessage:
    """Tier 1 payload: Telegram parses the Markdown itself; ``media`` backs its tg:// links."""
    return InputRichMessage(markdown=text, media=media)


def html_message(text: str) -> InputRichMessage | None:
    """Tier 2 payload, or None when the conversion needs more than one message (caller halves and retries)."""
    from telegramify_markdown import telegramify_rich

    items = telegramify_rich(text, mode="html")
    if len(items) != 1:
        return None
    rm = items[0].rich_message
    if rm.html:
        return InputRichMessage(html=rm.html)
    if rm.markdown:
        return InputRichMessage(markdown=rm.markdown)
    return None


def thinking_draft(text: str) -> InputRichMessage:
    """Streaming-draft payload for the model's thinking (the <tg-thinking> block, drafts only)."""
    body = _html.escape(truncate_bytes(text, DRAFT_BYTE_LIMIT - 64))
    return InputRichMessage(html=f"<tg-thinking>{body}</tg-thinking>")


def text_draft(text: str) -> InputRichMessage:
    return InputRichMessage(markdown=truncate_bytes(text, DRAFT_BYTE_LIMIT))


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def markdown_v2_chunks(text: str) -> list[str]:
    """Tier 3: GFM → MarkdownV2 with correct escaping, every piece under the 4096 limit.

    The *source* is split (paragraph boundaries first) and each piece converted
    on its own; a piece whose escaped form is still too long is halved again.
    """
    from telegramify_markdown import markdownify

    out: list[str] = []

    def convert(piece: str, depth: int = 0) -> None:
        converted = markdownify(piece)
        if _utf16_len(converted) <= MDV2_MAX or depth > 10:
            out.append(converted)
            return
        halves = halve_markdown(piece)
        if len(halves) < 2:
            mid = len(piece) // 2
            halves = [piece[:mid], piece[mid:]]
        for half in halves:
            convert(half, depth + 1)

    for piece in _split_by_bytes(text, MDV2_MAX * 3 // 4):
        convert(piece)
    return [p for p in out if p.strip()]


_MDV2_ESCAPE = re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")


def unescape_markdown_v2(text: str) -> str:
    return _MDV2_ESCAPE.sub(r"\1", text)


# ---- inline media -----------------------------------------------------------
# The model embeds pictures with normal Markdown image syntax: ``![caption](images/x.jpg)``
# for a file in the user's workspace or an https URL. In a rich message the file is
# uploaded with the message and referenced as ``tg://photo?id=…`` so it renders inline;
# the other tiers strip the image from the text and send it as a separate photo.
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
MEDIA_FALLBACK_DIRS = ("images", "videos", "downloads", "documents")
MAX_INLINE_MEDIA = 10
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^\s()<>]+)>?(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")


@dataclass
class MediaItem:
    id: str
    kind: str  # "photo" | "video"
    source: str | Path  # https URL or local file
    alt: str = ""

    def input_media(self) -> InputMediaPhoto | InputMediaVideo:
        media = self.source if isinstance(self.source, str) else FSInputFile(str(self.source))
        return InputMediaPhoto(media=media) if self.kind == "photo" else InputMediaVideo(media=media)


@dataclass
class MediaExtraction:
    rich_text: str  # image links rewritten to tg://photo?id=… / tg://video?id=…
    plain_text: str  # image syntax replaced by its caption (for non-rich tiers)
    items: list[MediaItem] = field(default_factory=list)

    def input_media(self) -> list[InputRichMessageMedia] | None:
        if not self.items:
            return None
        return [InputRichMessageMedia(id=item.id, media=item.input_media()) for item in self.items]


def _media_kind(name: str) -> str | None:
    ext = Path(name.split("?", 1)[0]).suffix.lower()
    if ext in PHOTO_EXTENSIONS:
        return "photo"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def _resolve_local(target: str, media_root: Path | None) -> tuple[str, Path] | None:
    if media_root is None:
        return None
    kind = _media_kind(target)
    if kind is None:
        return None
    from agents.paths import resolve_under

    path = resolve_under(media_root, target, fallback_dirs=MEDIA_FALLBACK_DIRS)
    if path is None or not path.is_file():
        return None
    limit = MAX_PHOTO_BYTES if kind == "photo" else MAX_VIDEO_BYTES
    if path.stat().st_size > limit:
        return None
    return kind, path


def extract_media(text: str, media_root: Path | None) -> MediaExtraction:
    """Find Markdown images in ``text`` and turn the ones we can send into media items.

    Unresolvable images (missing file, outside the workspace, unsupported type,
    too big) are replaced by their caption so Telegram never sees a broken link.
    """
    items: list[MediaItem] = []
    rich_parts: list[str] = []
    plain_parts: list[str] = []
    pos = 0
    for m in _IMAGE_RE.finditer(text):
        alt, target = m.group(1).strip(), m.group(2).strip()
        rich_parts.append(text[pos:m.start()])
        plain_parts.append(text[pos:m.start()])
        pos = m.end()
        resolved: tuple[str, str | Path] | None = None
        if target.startswith(("http://", "https://")):
            kind = _media_kind(target)
            if kind:
                resolved = (kind, target)
        elif target.startswith("tg://"):
            rich_parts.append(m.group(0))  # already a Telegram media link: leave it alone
            plain_parts.append(alt)
            continue
        else:
            resolved = _resolve_local(target, media_root)
        if resolved is None or len(items) >= MAX_INLINE_MEDIA:
            fallback = alt or (target if target.startswith("http") else Path(target).name)
            rich_parts.append(fallback)
            plain_parts.append(fallback)
            continue
        kind, source = resolved
        item = MediaItem(id=f"m{len(items) + 1}", kind=kind, source=source, alt=alt)
        items.append(item)
        rich_parts.append(f"![{alt}](tg://{kind}?id={item.id})")
        plain_parts.append(alt)
    rich_parts.append(text[pos:])
    plain_parts.append(text[pos:])
    return MediaExtraction(rich_text="".join(rich_parts), plain_text="".join(plain_parts), items=items)
