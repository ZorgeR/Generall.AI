"""WebTools: read a web page with a real browser, inside the sandbox.

``download_webpage`` (file_ops) fetches HTML with ``requests``; pages built by JavaScript or
behind consent walls give it nothing. ``browse_webpage`` instead runs
``secure_container/scripts/browse.py`` in a throwaway sandbox container with network on,
where headless Chromium renders the page and trafilatura extracts the readable text. The
browser never runs in the bot process: this module only copies the script in, parses the
JSON line it prints and formats the result.

The ContainerManager is created lazily (one per process) because importing agents must not
need Docker; the one ``secure_container.main`` built at startup is reused when it exists.
Tests replace ``_container_manager`` with a fake whose ``run_script`` returns canned logs.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

_container_manager = None
_manager_lock = threading.Lock()

BROWSE_SCRIPT = "browse.py"
BROWSE_TIMEOUT = 75          # container timeout; the script's own deadline is 60 s
BROWSE_MEM_LIMIT = "1g"      # Chromium does not fit in the 512 MB default
DEFAULT_MAX_CHARS = 20_000
MAX_LINKS_SHOWN = 20
MIN_WAIT_MS, MAX_WAIT_MS = 0, 10_000
MIN_MAX_CHARS, MAX_MAX_CHARS = 500, 100_000


def get_container_manager():
    """The process-wide ContainerManager, created on first use (Docker is needed only then)."""
    global _container_manager
    with _manager_lock:
        if _container_manager is None:
            existing = None
            try:
                from secure_container import main as sandbox_main

                existing = getattr(sandbox_main, "_container_manager", None)
            except Exception:  # noqa: BLE001 - the sandbox package may be unimportable in tests
                existing = None
            if existing is None:
                from secure_container.container_manager import ContainerManager

                existing = ContainerManager()
            _container_manager = existing
        return _container_manager


def last_json_line(logs: str) -> dict | None:
    """The last line of ``logs`` that parses as a JSON object, or None.

    Sandbox output always starts with the entrypoint banner and may carry package-install
    chatter, so the scripts print their result as the final line and the host reads from the end.
    Split on "\n" only: str.splitlines() also breaks on U+2028/U+2029/U+0085, which page text
    may contain, and a result line cut in two never parses.
    """
    for line in reversed((logs or "").split("\n")):
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def writable_dir(path: Path) -> Path:
    """Create ``path`` and let the sandbox's non-root ``runner`` write into it (best effort).

    ``data/<uid>`` itself is made world-writable by ContainerManager.ensure_user_directory; a
    subdirectory the bot created (root, 755) would still refuse the container's screenshot.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass
    return path


def _clamp(value, lo: int, hi: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


class WebTools:
    """Tool provider: ``browse_webpage``. Synchronous; register with ``blocking=True``."""

    def __init__(self, user_id: str, sender=None) -> None:
        self.user_id = str(user_id)
        self.sender = sender
        self.base_path = Path("./data") / self.user_id

    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "browse_webpage",
                "description": (
                    "Open a web page in a real headless browser (JavaScript rendered) and return its "
                    "readable text, title, final URL and the first links. Use it when download_webpage "
                    "returns little or garbled content, for single-page apps, pages behind consent "
                    "banners or anything that needs JavaScript. It runs in an isolated sandbox and takes "
                    "10-60 s, so prefer download_webpage for plain articles. With screenshot=true a PNG "
                    "of the viewport is saved under downloads/ (open it with view_image)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "http(s) URL to open"},
                        "wait_ms": {
                            "type": "integer",
                            "description": "Extra time to let the page settle after loading, in ms (default 1500, max 10000)",
                        },
                        "screenshot": {
                            "type": "boolean",
                            "description": "Also save a screenshot of the visible page (1280x900) under downloads/",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": f"Cap on the returned text (default {DEFAULT_MAX_CHARS}); the page text is not saved",
                        },
                    },
                    "required": ["url"],
                },
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "browse_webpage":
            return self.browse_webpage(
                tool_args.get("url", ""),
                wait_ms=tool_args.get("wait_ms", 1500),
                screenshot=bool(tool_args.get("screenshot", False)),
                max_chars=tool_args.get("max_chars", DEFAULT_MAX_CHARS),
            )
        return f"Unknown tool: {tool_name}"

    def browse_webpage(self, url: str, *, wait_ms=1500, screenshot: bool = False, max_chars=DEFAULT_MAX_CHARS) -> str:
        url = (url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return "Error: browse_webpage needs an http(s) URL with a host"
        wait_ms = _clamp(wait_ms, MIN_WAIT_MS, MAX_WAIT_MS, 1500)
        max_chars = _clamp(max_chars, MIN_MAX_CHARS, MAX_MAX_CHARS, DEFAULT_MAX_CHARS)

        shot_rel = ""
        if screenshot:
            writable_dir(self.base_path / "downloads")
            shot_rel = f"downloads/screenshot_{uuid.uuid4().hex[:8]}.png"

        args = [url, "--screenshot", shot_rel, "--wait-ms", str(wait_ms), "--max-chars", str(max_chars)]
        try:
            logs = get_container_manager().run_script(
                self.user_id, BROWSE_SCRIPT, args,
                timeout=BROWSE_TIMEOUT, network_enabled=True, mem_limit=BROWSE_MEM_LIMIT,
            )
        except Exception as e:  # noqa: BLE001 - Docker trouble is reported, not raised
            return f"Error: could not start the sandbox browser: {e}"
        data = last_json_line(logs)
        if data is None:
            tail = "\n".join((logs or "").strip().splitlines()[-5:])
            return f"Error: the sandbox browser returned no result. Output tail:\n{tail}"
        if data.get("error"):
            return f"Error: {data['error']}"
        return self._format(data, shot_rel, max_chars)

    def _format(self, data: dict, shot_rel: str, max_chars: int) -> str:
        text = (data.get("text") or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... truncated]"
        parts = [f"Title: {data.get('title') or '(none)'}", f"URL: {data.get('final_url') or ''}", "", text or "(no readable text)"]
        links = [link for link in (data.get("links") or []) if isinstance(link, (list, tuple)) and len(link) == 2]
        if links:
            parts += ["", f"Links (first {min(len(links), MAX_LINKS_SHOWN)} of {len(links)}):"]
            for text_label, href in links[:MAX_LINKS_SHOWN]:
                label = re.sub(r"\s+", " ", str(text_label)).strip() or "(no text)"
                parts.append(f"- {label} — {href}")
        if shot_rel:
            if (self.base_path / shot_rel).exists():
                parts += ["", f"Screenshot: {shot_rel} (view it with view_image)"]
            else:
                parts += ["", "Screenshot: not saved (the browser could not write it)"]
        return "\n".join(parts)
