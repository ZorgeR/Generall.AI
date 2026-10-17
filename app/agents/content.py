"""Prompt and tool-result content blocks.

A prompt reaching the agent is a string or a list of blocks. Handlers describe media with
small ``media_ref`` blocks (a path, a kind, a title); ``materialize`` turns them into the
base64 ``image`` / ``document`` blocks the Messages API takes, right before the request is
built, so the transcript stores exactly what was sent while turn records and logs keep the
small reference form.

A tool may likewise return a list of blocks (``text`` and ``image``) instead of a string;
``result_preview`` / ``cap_blocks`` give the trace text and the capped content for it.

No imports from the rest of the agent package: ``agents.transcript`` imports the token
estimates from here.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any

MEDIA_REF = "media_ref"
API_MEDIA_TYPES = ("image", "document")

IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
}
PDF_MEDIA_TYPE = "application/pdf"
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # API limit per image
MAX_DOCUMENT_BYTES = 30 * 1024 * 1024  # under the 32 MB request limit

CHARS_PER_TOKEN = 3.2  # same conservative figure as agents.transcript
IMAGE_TOKENS = 1600  # an image at <= ~1.15 MP; the handlers downscale to 1024 px
DOCUMENT_TOKENS_PER_KB = 8  # ~ a PDF page (60-120 KB) as 500-1000 tokens of text and layout
DOCUMENT_MIN_TOKENS = 1500


# ---- building blocks ---------------------------------------------------------
def text_block(text: str) -> dict:
    return {"type": "text", "text": str(text)}


def image_ref(path: str | os.PathLike, caption: str | None = None) -> dict:
    """A picture in the workspace that the agent should see (JPEG/PNG/GIF/WebP; HEIC must be converted first)."""
    return {"type": MEDIA_REF, "kind": "image", "path": str(path), "title": str(caption or "")}


def document_ref(path: str | os.PathLike, title: str | None = None) -> dict:
    """A PDF in the workspace that the agent should read as a document block."""
    return {"type": MEDIA_REF, "kind": "document", "path": str(path), "title": str(title or os.path.basename(str(path)))}


def is_media_ref(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == MEDIA_REF


def is_media_block(block: Any) -> bool:
    """An API ``image`` or ``document`` block (materialized)."""
    return isinstance(block, dict) and block.get("type") in API_MEDIA_TYPES


def normalize(prompt: str | list | None) -> list[dict]:
    """A prompt as a list of blocks: a string becomes one text block; strings inside a list too."""
    if prompt is None:
        return [text_block("")]
    if isinstance(prompt, str):
        return [text_block(prompt)]
    blocks: list[dict] = []
    for item in prompt:
        if isinstance(item, str):
            blocks.append(text_block(item))
        elif isinstance(item, dict):
            blocks.append(dict(item))
    return blocks or [text_block("")]


def with_prefix(blocks: list[dict], prefix: str) -> list[dict]:
    """``prefix`` in front of the first text block (a new one when the prompt starts with media)."""
    out = [dict(b) for b in blocks]
    if out and out[0].get("type") == "text":
        out[0]["text"] = f"{prefix}{out[0].get('text', '')}"
        return out
    return [text_block(prefix.rstrip("\n"))] + out


def media_marker(block: dict) -> str:
    """The text standing in for a media block: ``[image: path]`` / ``[document: title]``."""
    if is_media_ref(block):
        label = block.get("title") or block.get("path") or ""
        return f"[{block.get('kind', 'file')}: {label}]" if block.get("kind") == "document" else f"[image: {block.get('path', '')}]"
    if block.get("type") == "image":
        return "[image]"
    if block.get("type") == "document":
        title = block.get("title")
        return f"[document: {title}]" if title else "[document]"
    return ""


def prompt_text(prompt: str | list | None) -> str:
    """The text view of a prompt: text blocks joined, media as markers (classifier, memory, logs, records)."""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for b in normalize(prompt):
        if b.get("type") == "text":
            if str(b.get("text", "")).strip():
                parts.append(str(b["text"]))
        else:
            marker = media_marker(b)
            if marker:
                parts.append(marker)
    return "\n\n".join(parts)


# ---- materialization -----------------------------------------------------------
def _sniff_image_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _materialize_one(block: dict) -> dict:
    kind = block.get("kind")
    path = str(block.get("path") or "")
    title = str(block.get("title") or os.path.basename(path))
    try:
        size = os.path.getsize(path)
    except OSError:
        return text_block(f"[{kind or 'file'} not found: {path}]")
    if kind == "image":
        if size > MAX_IMAGE_BYTES:
            return text_block(f"[image too large to show ({size // (1024 * 1024)} MB): {path}; use view_image, it downscales]")
        with open(path, "rb") as f:
            data = f.read()
        media_type = IMAGE_MEDIA_TYPES.get(os.path.splitext(path)[1].lower()) or _sniff_image_type(data)
        if media_type is None:
            return text_block(f"[image in an unsupported format: {path}]")
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": _b64(data)}}
    if kind == "document":
        if os.path.splitext(path)[1].lower() != ".pdf":
            return text_block(f"[document: {title} at {path}; not a PDF, read it with read_file or read_document]")
        if size > MAX_DOCUMENT_BYTES:
            return text_block(f"[document too large to attach ({size // (1024 * 1024)} MB): {path}; read it with read_document]")
        with open(path, "rb") as f:
            data = f.read()
        return {"type": "document", "source": {"type": "base64", "media_type": PDF_MEDIA_TYPE, "data": _b64(data)}, "title": title}
    return text_block(f"[unsupported attachment: {path}]")


def materialize(blocks: list[dict]) -> list[dict]:
    """API blocks for ``blocks``: media refs become base64 image/document blocks, the rest is copied.

    Blocking (reads files): call in a worker thread. A file that cannot be embedded becomes a
    text block saying why, so the model still learns the path.
    """
    out: list[dict] = []
    for b in blocks:
        if is_media_ref(b):
            out.append(_materialize_one(b))
        elif isinstance(b, dict):
            out.append(dict(b))
    return out


def strip_media(messages: list[dict]) -> list[dict]:
    """A copy of ``messages`` with every image/document block replaced by its marker (for files
    that must stay small: conversation history, reasoning transcripts)."""
    out: list[dict] = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            out.append(m)
            continue
        blocks: list = []
        for b in content:
            if is_media_block(b):
                blocks.append(text_block(media_marker(b)))
            elif isinstance(b, dict) and b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                inner = [text_block(media_marker(x)) if is_media_block(x) else x for x in b["content"]]
                blocks.append({**b, "content": inner})
            else:
                blocks.append(b)
        out.append({**m, "content": blocks})
    return out


# ---- tool results as blocks ------------------------------------------------------
def result_preview(result: Any) -> str:
    """Trace / log text of a tool result that may be a list of blocks."""
    if not isinstance(result, list):
        return str(result)
    parts: list[str] = []
    for b in result:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text", "")))
        elif isinstance(b, dict):
            parts.append(media_marker(b) or f"[{b.get('type', 'block')}]")
        else:
            parts.append(str(b))
    return "\n".join(p for p in parts if p)


def cap_blocks(blocks: list, max_chars: int | None) -> list:
    """Text blocks cut to ``max_chars`` in total (with a note); media blocks untouched."""
    if not max_chars:
        return list(blocks)
    out: list = []
    budget = max_chars
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            text = str(b.get("text", ""))
            if len(text) > budget:
                cut = text[:max(0, budget)] + f"\n…[truncated {len(text) - max(0, budget)} characters]"
                out.append({**b, "text": cut})
                budget = 0
            else:
                out.append(b)
                budget -= len(text)
        else:
            out.append(b)
    return out


# ---- size estimates ---------------------------------------------------------------
def estimate_block_tokens(block: Any) -> int:
    """Rough token count of one content block; media by kind, never its base64 as text."""
    if not isinstance(block, dict):
        return int(len(json.dumps(block, ensure_ascii=False, default=str)) / CHARS_PER_TOKEN)
    kind = block.get("type")
    if kind == "image":
        return IMAGE_TOKENS
    if kind == "document":
        data = (block.get("source") or {}).get("data") or ""
        kb = len(data) * 3 / 4 / 1024
        return max(DOCUMENT_MIN_TOKENS, int(kb * DOCUMENT_TOKENS_PER_KB))
    if kind == MEDIA_REF:
        return IMAGE_TOKENS if block.get("kind") == "image" else DOCUMENT_MIN_TOKENS
    if kind == "tool_result" and isinstance(block.get("content"), list):
        return sum(estimate_block_tokens(b) for b in block["content"]) + 8
    return int(len(json.dumps(block, ensure_ascii=False, default=str)) / CHARS_PER_TOKEN)
