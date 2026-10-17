"""DocumentTools: ``create_document`` turns Markdown into a file and hands it to the user.

Markdown and HTML are rendered here with python-markdown (tables, fenced code, sane lists,
toc) and a small stylesheet. PDF, DOCX and XLSX need Chromium or heavier libraries, so those
run ``secure_container/scripts/render_document.py`` in the sandbox: the source is written to
``documents/.render_<hex>.<ext>`` in the user's workspace, the container renders it to the
final path in the same directory, and the temp source is deleted. The output lives under
``documents/<name>.<format>``; an existing name gets ``-2``, ``-3`` rather than being replaced.

``execute_tool`` is synchronous (register with ``blocking=True``): it runs in a worker thread,
so the file is sent through the sender by scheduling the coroutine on the loop captured at
construction time. Without a loop (unit tests) the coroutine is simply run to completion.
"""
from __future__ import annotations

import asyncio
import html
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List

from agents import web_tools

FORMATS = ("md", "html", "pdf", "docx", "xlsx")
SANDBOX_FORMATS = {"pdf", "docx", "xlsx"}
RENDER_SCRIPT = "render_document.py"
RENDER_TIMEOUT = 90
PDF_MEM_LIMIT = "1g"
MARKDOWN_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "toc"]
SEND_TIMEOUT = 120
MAX_NAME_LENGTH = 80

CSS = """
body { font-family: Georgia, 'Times New Roman', serif; font-size: 12pt; line-height: 1.5;
       color: #222; max-width: 820px; margin: 2em auto; padding: 0 1.5em; }
h1, h2, h3, h4 { font-family: Helvetica, Arial, sans-serif; color: #111; line-height: 1.25; margin-top: 1.6em; }
h1 { font-size: 2em; border-bottom: 2px solid #ddd; padding-bottom: .3em; }
h2 { font-size: 1.5em; } h3 { font-size: 1.2em; }
p, ul, ol { margin: .7em 0; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; font-size: .95em; }
th, td { border: 1px solid #bbb; padding: .4em .7em; text-align: left; vertical-align: top; }
th { background: #f0f0f0; font-weight: bold; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: 'Courier New', Consolas, monospace; font-size: .92em; background: #f4f4f4;
       padding: .1em .3em; border-radius: 3px; }
pre { background: #f4f4f4; border: 1px solid #ddd; border-radius: 4px; padding: .8em 1em;
      overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #ccc; margin: 1em 0; padding: .2em 1em; color: #555; }
img { max-width: 100%; }
a { color: #1a5fb4; }
""".strip()

_H1_RE = re.compile(r"^#\s+\S", re.MULTILINE)
_FENCED_RE = re.compile(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", re.MULTILINE | re.DOTALL)

# Output names handed out but not written yet: unique_path() must not give the same name to two
# create_document calls of one concurrent tool batch (check-then-create would).
_reserved: set[Path] = set()
_reserved_lock = threading.Lock()


def has_h1(content: str) -> bool:
    """True when the Markdown has a real ``# Heading`` line; ``# comment`` lines inside fenced code do not count."""
    return _H1_RE.search(_FENCED_RE.sub("", content)) is not None


def sanitize_filename(name: str | None, fallback: str = "document") -> str:
    """A safe base name: word characters, dots and dashes; no leading dots or separators; bounded length.

    Slashes become underscores rather than directories: every document lands in ``documents/``."""
    text = re.sub(r"[^\w.-]+", "_", (name or "").strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("._- ")
    text = text[:MAX_NAME_LENGTH].rstrip("._- ")
    return text or fallback


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """``directory/stem.suffix`` or the first ``stem-N.suffix`` (N >= 2) that neither exists nor
    is reserved by another call in flight. The name stays reserved until ``release_path``."""
    with _reserved_lock:
        candidate = directory / f"{stem}{suffix}"
        n = 2
        while candidate.exists() or candidate in _reserved:
            candidate = directory / f"{stem}-{n}{suffix}"
            n += 1
        _reserved.add(candidate)
    return candidate


def release_path(path: Path) -> None:
    with _reserved_lock:
        _reserved.discard(path)


def render_markdown_html(title: str, content: str) -> str:
    """A complete, self-contained HTML document from Markdown (the agent's tables and code kept)."""
    import markdown

    body = markdown.markdown(content, extensions=MARKDOWN_EXTENSIONS, output_format="html5")
    heading = "" if has_h1(content) else f"<h1>{html.escape(title)}</h1>\n"
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title)}</title>\n<style>\n{CSS}\n</style>\n</head>\n<body>\n"
        f"{heading}{body}\n</body>\n</html>\n"
    )


def markdown_with_title(title: str, content: str) -> str:
    return content if has_h1(content) else f"# {title}\n\n{content}"


class DocumentTools:
    """Tool provider: ``create_document``. Synchronous; register with ``blocking=True``."""

    def __init__(self, user_id: str, sender=None) -> None:
        self.user_id = str(user_id)
        self.sender = sender
        self.base_path = Path("./data") / self.user_id
        self.documents_path = self.base_path / "documents"
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "create_document",
                "description": (
                    "Write a document from Markdown and deliver it to the user as a file: md, html, pdf "
                    "(A4, styled), docx (Word) or xlsx (Excel: every pipe table or ```csv block becomes a "
                    "sheet named after the heading above it). Supported Markdown: # ## ### headings, "
                    "paragraphs, - bullets and 1. lists, **bold**, *italic*, `code`, fenced code blocks "
                    "and pipe tables. The file is saved under documents/ and SENT to the user by this tool; "
                    "do not send it again. Use it whenever the user asks for a report, table, letter or "
                    "export as a file rather than a chat message."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "enum": list(FORMATS), "description": "Output format"},
                        "title": {"type": "string", "description": "Document title (also the default file name and the caption)"},
                        "content": {"type": "string", "description": "The document body in Markdown"},
                        "filename": {
                            "type": "string",
                            "description": "File name without directory (optional; the extension is added; an existing name gets -2, -3)",
                        },
                    },
                    "required": ["format", "title", "content"],
                },
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "create_document":
            try:
                return self.create_document(
                    str(tool_args.get("format", "")),
                    str(tool_args.get("title", "") or ""),
                    str(tool_args.get("content", "") or ""),
                    tool_args.get("filename"),
                )
            except Exception as e:  # noqa: BLE001 - the model gets the reason, the turn goes on
                return f"Error: {type(e).__name__}: {e}"
        return f"Unknown tool: {tool_name}"

    # ---- rendering -------------------------------------------------------------------
    def create_document(self, fmt: str, title: str, content: str, filename: str | None = None) -> str:
        fmt = (fmt or "").strip().lower().lstrip(".")
        if fmt not in FORMATS:
            return f"Error: unsupported format '{fmt}'; use one of {', '.join(FORMATS)}"
        if not content.strip():
            return "Error: content is empty"
        title = title.strip() or "Document"
        stem = sanitize_filename(re.sub(rf"\.{fmt}$", "", (filename or "").strip(), flags=re.IGNORECASE) or title)
        web_tools.writable_dir(self.documents_path)
        output = unique_path(self.documents_path, stem, f".{fmt}")
        try:
            if fmt == "md":
                output.write_text(markdown_with_title(title, content), encoding="utf-8", newline="\n")
            elif fmt == "html":
                output.write_text(render_markdown_html(title, content), encoding="utf-8", newline="\n")
            else:
                error = self._render_in_sandbox(fmt, title, content, output)
                if error:
                    return f"Error: {error}"
            if not output.exists():
                return f"Error: the {fmt} renderer produced no file"
        finally:
            release_path(output)

        rel = output.relative_to(self.base_path).as_posix()
        size_kb = max(1, round(output.stat().st_size / 1024))
        return f"Document saved to {rel} ({size_kb} KB){self._deliver(output, title)}"

    def _render_in_sandbox(self, fmt: str, title: str, content: str, output: Path) -> str | None:
        """Run render_document.py on a temp source next to the output; returns an error or None."""
        source_ext = ".html" if fmt == "pdf" else ".md"
        source = self.documents_path / f".render_{uuid.uuid4().hex[:8]}{source_ext}"
        text = render_markdown_html(title, content) if fmt == "pdf" else content
        source.write_text(text, encoding="utf-8", newline="\n")
        args = [
            "--format", fmt,
            "--input", source.relative_to(self.base_path).as_posix(),
            "--output", output.relative_to(self.base_path).as_posix(),
            f"--title={title}",  # one token: a title starting with "-" is not an option
        ]
        try:
            logs = web_tools.get_container_manager().run_script(
                self.user_id, RENDER_SCRIPT, args,
                timeout=RENDER_TIMEOUT, network_enabled=False,
                mem_limit=PDF_MEM_LIMIT if fmt == "pdf" else None,
            )
        except Exception as e:  # noqa: BLE001
            return f"could not start the sandbox renderer: {e}"
        finally:
            try:
                source.unlink()
            except OSError:
                pass
        data = web_tools.last_json_line(logs)
        if data is None:
            tail = "\n".join((logs or "").strip().splitlines()[-5:])
            return f"the sandbox renderer returned no result. Output tail:\n{tail}"
        if data.get("error"):
            return str(data["error"])
        return None

    # ---- delivery --------------------------------------------------------------------
    def _deliver(self, path: Path, title: str) -> str:
        if self.sender is None:
            return " (no chat to send it to)."
        try:
            self._run_async(self.sender.send_document(str(path), filename=path.name, caption=title))
        except Exception as e:  # noqa: BLE001
            return f", but sending it failed: {e}. Use send_file_path_to_user_via_telegram to retry."
        return " and sent to the user."

    def _run_async(self, coro):
        """Run a coroutine from this worker thread on the bot's loop (or inline without one)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise RuntimeError("create_document must run in a worker thread (register with blocking=True)")
        loop = self._loop
        if loop is not None and loop.is_running() and not loop.is_closed():
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=SEND_TIMEOUT)
        return asyncio.run(coro)
