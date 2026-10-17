"""VisionTools: let the agent look at images and documents in the workspace (docs/design-v3.md, F7).

Pictures and PDFs the user sends reach the agent as real image / document blocks in the
message itself (``bot/handlers/messages.py``). This provider covers everything else: a file
the agent made or downloaded (``view_image``), a second opinion from another model
(``describe_image``), and PDFs or office files it wants to read again or did not get as a
block (``read_document``). ``view_image`` and ``read_document`` return content blocks, so the
model sees the picture or the PDF itself; ``describe_image`` returns text.

Runs in the bot process, never sandboxed: ``tool_integrator.py`` must not name this class or
its methods (``tests/test_vision_tools.py`` guards that). File reads and PIL work run in a
worker thread; paths are resolved with ``agents.paths.resolve_under`` like the other tools.
Registered in ``ChainOfThoughtAgent.__init__`` with ``register_provider`` (awaited).
"""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from typing import Any

from agents import content
from agents.paths import resolve_under

IMAGE_DIRS = ("images", "downloads", "documents")
DOCUMENT_DIRS = ("documents", "downloads", "images")
MAX_VIEW_PX = 2048  # the API downsizes above ~1568 px anyway
MIN_VIEW_PX = 128
EXTRACTED_TEXT_CAP = 60_000  # characters of extracted document text returned by read_document
TRUNCATED_NOTE = "\n…(truncated: use read_file or execute_python for the rest)"


def _media():
    """``bot.media`` imported on first use: the agent package stays importable without the bot."""
    from bot import media

    return media


def cap_text(text: str, limit: int | None = None) -> str:
    limit = EXTRACTED_TEXT_CAP if limit is None else limit
    return text if len(text) <= limit else text[:limit] + TRUNCATED_NOTE


class VisionTools:
    prompt_notes = (
        "Pictures and PDFs the user sends arrive as real image/document blocks in their message: you see "
        "them directly, no description step is needed. The paths named in the message are the files in "
        "your workspace (the full-resolution originals); use them to edit a picture, to show it again "
        "(![caption](images/<file>)) or to read a document later. Use view_image to look at any image "
        "file in the workspace (generated images, downloads, screenshots, photos from earlier messages), "
        "describe_image for a second opinion on a picture from another model, and read_document for PDFs "
        "and office files (docx, xlsx, json, text) in the workspace."
    )

    def __init__(self, user_id: str, sender=None) -> None:
        self.user_id = str(user_id)
        self.sender = sender
        self.base_path = Path("./data") / self.user_id  # like FileOperations / ImageTools

    # ---- contract ----------------------------------------------------------------
    @property
    def tools_schema(self) -> list[dict]:
        return [
            {
                "name": "view_image",
                "description": (
                    "Look at an image file in the workspace: generated images, downloads, screenshots "
                    "(browse_webpage), video frames or photos the user sent earlier. Returns the picture "
                    "itself (a copy downscaled to max_px on the long side) and its dimensions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path inside the workspace, e.g. images/generated_ab12.png or downloads/screenshot_1a2b.png (a bare file name is looked up in images/, downloads/ and documents/)",
                        },
                        "max_px": {
                            "type": "integer",
                            "description": f"Longest side in pixels of the copy you see (default {_default_px()}; up to 1568 for fine detail)",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "describe_image",
                "description": (
                    "Ask another model to describe an image in the workspace, optionally answering a "
                    "question about it: a second opinion when you are unsure what you see. Returns text."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path of the image inside the workspace"},
                        "question": {"type": "string", "description": "What to look for or answer (default: a detailed description)"},
                        "model": {
                            "type": "string",
                            "enum": ["anthropic", "openai"],
                            "description": "Which model describes it (default anthropic)",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "read_document",
                "description": (
                    "Read a document in the workspace. A PDF within the limits (20 MB, 100 pages) is "
                    "returned as a document block you read directly; larger PDFs and docx/xlsx/xls/json/"
                    "text files come back as extracted text (capped; read_file for the rest)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path inside the workspace, e.g. documents/report.pdf"},
                        "question": {"type": "string", "description": "What you need from it (used when the file has to be summarized instead of read)"},
                    },
                    "required": ["path"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str | list:
        args = tool_args or {}
        try:
            if tool_name == "view_image":
                return await self.view_image(str(args.get("path", "")), args.get("max_px"))
            if tool_name == "describe_image":
                return await self.describe_image(str(args.get("path", "")), args.get("question"), args.get("model") or "anthropic")
            if tool_name == "read_document":
                return await self.read_document(str(args.get("path", "")), args.get("question"))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the model gets the reason as a tool result
            return f"Error: {e}"
        return f"Unknown tool: {tool_name}"

    # ---- tools -------------------------------------------------------------------
    async def view_image(self, path: str, max_px: int | None = None) -> list | str:
        """The image block (downscaled copy made in memory) and a text block with the path and size."""
        resolved = self._resolve(path, IMAGE_DIRS)
        if resolved is None:
            return f"Error: '{path}' is not a file in your workspace. List the files first if you are unsure of the name."
        px = _clamp_px(max_px)
        try:
            data, media_type, original, sent = await asyncio.to_thread(_media().image_for_vision, str(resolved), px)
        except Exception as e:  # noqa: BLE001
            return f"Error: '{self._rel(resolved)}' could not be read as an image: {e}"
        label = f"{self._rel(resolved)} · {original[0]}x{original[1]} px"
        if sent != original:
            label += f" (shown at {sent[0]}x{sent[1]})"
        return [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(data).decode("ascii")}},
            content.text_block(label),
        ]

    async def describe_image(self, path: str, question: str | None = None, model: str = "anthropic") -> str:
        """A text description from another model; the image is prepared like an uploaded photo."""
        resolved = self._resolve(path, IMAGE_DIRS)
        if resolved is None:
            return f"Error: '{path}' is not a file in your workspace."
        if model not in ("anthropic", "openai"):
            return f"Error: unknown model '{model}' (use anthropic or openai)."
        question = (question or "").strip() or "Describe what is in this image in detail."
        media = _media()
        prepared = await asyncio.to_thread(self._temp_jpeg, resolved)
        try:
            if model == "openai":
                text = await media.describe_image_openai(question, prepared, user_id=self.user_id)
            else:
                text = await media.describe_image_anthropic(question, prepared)
        finally:
            await asyncio.to_thread(_unlink_quietly, prepared)
        return text or "Error: the model returned no description."

    async def read_document(self, path: str, question: str | None = None) -> list | str:
        """A PDF within the limits as a document block (+ a text block naming it); otherwise text."""
        resolved = self._resolve(path, DOCUMENT_DIRS)
        if resolved is None:
            return f"Error: '{path}' is not a file in your workspace."
        media = _media()
        name = resolved.name
        rel = self._rel(resolved)
        if resolved.suffix.lower() == ".pdf":
            pages = await asyncio.to_thread(media.pdf_page_count, str(resolved))
            if media.pdf_fits_as_block(str(resolved), pages):
                blocks = await asyncio.to_thread(content.materialize, [content.document_ref(resolved, title=name)])
                return [*blocks, content.text_block(f"{name} · {pages} pages")]
            text = await asyncio.to_thread(media.extract_document_text, str(resolved))
            return f"{rel} · {pages} pages (too large to attach as a document; extracted text follows)\n\n{cap_text(text)}"
        try:
            text = await asyncio.to_thread(media.extract_document_text, str(resolved))
        except ValueError:
            # not a document type with a reader: let the describe path have a go (it reports its own errors)
            text = await media.describe_document(
                (question or "").strip() or "Describe the contents of this document in detail.", str(resolved)
            )
            return text
        return f"{rel}\n\n{cap_text(text)}"

    # ---- helpers -----------------------------------------------------------------
    def _resolve(self, path: str, fallback_dirs: tuple[str, ...]) -> Path | None:
        resolved = resolve_under(self.base_path, path, fallback_dirs=fallback_dirs)
        return resolved if resolved is not None and resolved.is_file() else None

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.base_path.resolve()).as_posix()
        except ValueError:
            return path.name

    def _temp_jpeg(self, source: Path) -> str:
        """A JPEG copy within the vision limits, in a temporary file (blocking)."""
        fd, target = tempfile.mkstemp(prefix="describe_", suffix=".jpg")
        os.close(fd)
        try:
            _media().save_vision_copy(str(source), target)
        except Exception:
            _unlink_quietly(target)
            raise
        return target


def _default_px() -> int:
    try:
        from bot.config import config

        return int(config.max_image_resolution_vision)
    except Exception:  # noqa: BLE001
        return 1024


def _clamp_px(value) -> int | None:
    """``max_px`` as given, within sane bounds; None means the configured default."""
    if value is None:
        return None
    try:
        px = int(value)
    except (TypeError, ValueError):
        return None
    return max(MIN_VIEW_PX, min(MAX_VIEW_PX, px))


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
