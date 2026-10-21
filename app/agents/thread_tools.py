"""The user's other conversations: ``list_threads`` and ``read_thread``.

Every Telegram thread (forum topic) of the chat, and the main chat outside them, is a separate
conversation with its own transcript (``data/<uid>/transcripts/[topic_<id>_]transcript.json``), so
the agent sees only the one it is in. The memory lines of the ``<context>`` block and the
``memory_search`` hits say which conversation they come from (``bot.topics.thread_label``); these two
tools let the agent look at another one when the user refers to something said there, instead of
guessing. ``conversation_thread`` finds the thread of a semantic-search hit for those memory lines.

Everything here reads ``data/<uid>``, the sandbox's workspace, so files are read through
``agents.paths.regular_files`` (never through a link) and listed without following links. Without a
transcript (Transcript mode off) a thread is read from its ``dialog_history.json``. Synchronous:
registered ``blocking=True``.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.paths import regular_files
from agents.transcript import SUMMARY_TAG, is_context_block, is_tool_result_message, message_text
from bot.topics import MAIN_CHAT, thread_label, topic_store

DEFAULT_READ_MESSAGES = 20
MAX_READ_MESSAGES = 100
MESSAGE_CHARS = 2000  # one message in read_thread; longer ones are cut
READ_BUDGET_CHARS = 16000  # the whole read_thread result: the newest messages are kept
_TRANSCRIPT_RE = re.compile(r"^(?:topic_(\d+)_)?transcript\.json$")
_DIALOG_RE = re.compile(r"^(?:topic_(\d+)_)?dialog_history\.json$")
_CONTEXT_TIME_RE = re.compile(r"Current time in UTC\+0: (\d{4}-\d{2}-\d{2} \d{2}:\d{2})")
_CONVERSATION_TS_RE = re.compile(r"^\d{8}_\d{6}$")
_FOLLOW_UPS_RE = re.compile(r"\s*<follow_ups>.*?</follow_ups>\s*", re.S)
_MAIN_NAMES = {"", "main", "main chat", "the main chat", "none", "0"}


def as_thread_id(value: Any) -> int | None:
    """A stored thread id as an int; None for the main chat or anything that is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip()) or None
    return None


def conversation_thread(folder: Path, timestamp: str, topic: str) -> tuple[bool, int | None]:
    """``(found, thread id)`` of the conversation file a semantic-search hit came from.

    The embeddings keep the conversation's ``timestamp`` and ``topic``, which also name its file
    (``conversation_<timestamp>_<topic>.json``); the thread is in the file. Not found (or two
    files of that second in different threads) → ``(False, None)``: the hit gets no label.
    """
    if not _CONVERSATION_TS_RE.match(str(timestamp or "")):
        return False, None
    candidates: list[tuple[str, dict]] = []
    for path, raw in regular_files(folder, f"conversation_{timestamp}_*.json"):
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        if isinstance(data, dict):
            candidates.append((path.name, data))
    exact = [d for name, d in candidates if name == f"conversation_{timestamp}_{topic}.json" or d.get("topic") == topic]
    threads = {as_thread_id(d.get("thread_id")) for d in (exact or [d for _, d in candidates])}
    if len(threads) != 1:
        return False, None
    return True, threads.pop()


class ThreadTools:
    prompt_notes = (
        "Each Telegram thread (forum topic) of this chat, and the main chat outside the threads, is a "
        "separate conversation with its own history; you see only the current one. A memory line in the "
        "<context> block that comes from another conversation ends with [from ...], and memory_search hits "
        "name their thread. When the user refers to something said in another thread or in the main chat, "
        "look it up with list_threads and read_thread instead of guessing or asking them to repeat it."
    )

    def __init__(self, user_id: str, thread_id: int | None = None, base_dir: str | Path = "data") -> None:
        self.user_id = str(user_id)
        self.thread_id = as_thread_id(thread_id)
        self.base = Path(base_dir) / self.user_id

    @property
    def tools_schema(self) -> list[dict]:
        return [
            {
                "name": "list_threads",
                "description": (
                    "List the user's conversations with you: the main chat and every Telegram thread (forum "
                    "topic) of this chat, with name, id and time of the last message, newest first; the current "
                    "one is marked."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "read_thread",
                "description": (
                    "Read the latest messages of another conversation with the user (a Telegram thread, or the "
                    "main chat outside the threads): what the user and you wrote there, oldest first, without "
                    "tool calls. Use it when the user refers to something said in another thread."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "thread": {
                            "type": "string",
                            "description": "The thread's id, its name (or a unique part of it), or \"main\" for the main chat.",
                        },
                        "max_messages": {
                            "type": "integer",
                            "description": f"How many of the latest messages to read (default {DEFAULT_READ_MESSAGES}).",
                            "minimum": 1,
                            "maximum": MAX_READ_MESSAGES,
                            "default": DEFAULT_READ_MESSAGES,
                        },
                    },
                    "required": ["thread"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        if tool_name == "list_threads":
            return self.list_threads()
        if tool_name == "read_thread":
            return self.read_thread(tool_args.get("thread", ""), tool_args.get("max_messages", DEFAULT_READ_MESSAGES))
        return f"Unknown tool: {tool_name}"

    # ---- list_threads ---------------------------------------------------------------
    def list_threads(self) -> str:
        names = topic_store.all(self.user_id)
        activity = self._activity()
        threads = set(activity) | set(names) | {self.thread_id}
        lines = []
        for thread_id in sorted(threads, key=lambda t: (-activity.get(t, 0.0), t or 0)):
            parts = [thread_label(thread_id, names.get(thread_id))]
            if thread_id in activity:
                parts.append("last message " + datetime.fromtimestamp(activity[thread_id], timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
            else:
                parts.append("no messages kept")
            if thread_id == self.thread_id:
                parts.append("current")
            lines.append("- " + " · ".join(parts))
        return (
            "Conversations with this user, the most recent first:\n" + "\n".join(lines)
            + "\nRead one with read_thread (its id, its name or \"main\")."
        )

    def _activity(self) -> dict[int | None, float]:
        """Thread id → time of the last write of its transcript (or legacy dialog history)."""
        found: dict[int | None, float] = {}
        for folder, pattern in ((self.base / "transcripts", _TRANSCRIPT_RE), (self.base / "short_term_memory", _DIALOG_RE)):
            for name, mtime in _listing(folder):
                m = pattern.match(name)
                if m:
                    key = as_thread_id(m.group(1))
                    found[key] = max(found.get(key, 0.0), mtime)
        return found

    # ---- read_thread ----------------------------------------------------------------
    def read_thread(self, thread: Any, max_messages: Any = DEFAULT_READ_MESSAGES) -> str:
        try:
            limit = max(1, min(int(max_messages or DEFAULT_READ_MESSAGES), MAX_READ_MESSAGES))
        except (TypeError, ValueError):
            limit = DEFAULT_READ_MESSAGES
        names = topic_store.all(self.user_id)
        resolved = self._resolve(thread, names)
        if isinstance(resolved, str):
            return resolved  # an error
        thread_id = resolved
        label = thread_label(thread_id, names.get(thread_id))
        if thread_id == self.thread_id:
            return (
                f"That is the current conversation ({label}): it is already in your context. "
                "Use memory_search for older parts that were summarized away."
            )
        entries = self._entries(thread_id)
        if not entries:
            return f"No messages are kept for {label}."
        total = len(entries)
        kept: list[str] = []
        used = 0
        for entry in reversed(entries[-limit:]):  # newest first, until the budget is spent
            if kept and used + len(entry) > READ_BUDGET_CHARS:
                break
            kept.append(entry)
            used += len(entry) + 2
        kept.reverse()
        return f"{label}: the last {len(kept)} of {total} messages, oldest first.\n\n" + "\n\n".join(kept)

    def _resolve(self, thread: Any, names: dict) -> int | None | str:
        """The thread id ``thread`` names, None for the main chat, or an error text."""
        text = " ".join(str(thread if thread is not None else "").split())
        key = text.lower().lstrip("#")
        if key in _MAIN_NAMES:
            return None
        exact = [t for t, topic in names.items() if topic.name.lower() == key]
        if len(exact) == 1:
            return exact[0]
        bare = re.sub(r"^(?:thread|topic|id)\s*", "", key).lstrip("#")
        if not exact and bare.isdigit():
            return int(bare) or None
        matches = exact or [t for t, topic in names.items() if key in topic.name.lower()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            options = ", ".join(thread_label(t, names[t]) for t in sorted(matches))
            return f"Error: several threads match {json.dumps(text, ensure_ascii=False)}: {options}. Use the id."
        return f"Error: no thread named {json.dumps(text, ensure_ascii=False)}. Use list_threads to see them."

    def _entries(self, thread_id: int | None) -> list[str]:
        """The conversation as text entries, oldest first: from the transcript, else the legacy
        dialog history."""
        prefix = f"topic_{thread_id}_" if thread_id else ""
        messages = _read_json(self.base / "transcripts", f"{prefix}transcript.json")
        messages = messages.get("messages") if isinstance(messages, dict) else None
        if not isinstance(messages, list):
            messages = _read_json(self.base / "short_term_memory", f"{prefix}dialog_history.json")
        entries: list[tuple[str, str, str]] = []  # (role, time, text)
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                continue
            if is_tool_result_message(message):
                continue
            text = message_text(message).strip()
            if message["role"] == "assistant":
                text = _FOLLOW_UPS_RE.sub("\n", text).strip()
            if not text:
                continue
            when = ""
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if is_context_block(block):
                        m = _CONTEXT_TIME_RE.search(block["text"])
                        when = f"{m.group(1)} UTC" if m else ""
            role = message["role"]
            if role == "user" and text.startswith(f"<{SUMMARY_TAG}>"):
                role = "summary"
            if entries and entries[-1][0] == role and not when:
                entries[-1] = (role, entries[-1][1], entries[-1][2] + "\n\n" + text)
            else:
                entries.append((role, when, text))
        speaker = {"user": "User", "assistant": "You", "summary": "Summary of earlier messages"}
        out = []
        for role, when, text in entries:
            if len(text) > MESSAGE_CHARS:
                text = text[:MESSAGE_CHARS].rstrip() + " … (cut)"
            out.append(f"[{when}] {speaker[role]}: {text}" if when else f"{speaker[role]}: {text}")
        return out


def _listing(folder: Path) -> list[tuple[str, float]]:
    """``(name, mtime)`` of the regular files directly in ``folder``; nothing through a link."""
    try:
        if folder.is_symlink() or not folder.is_dir():
            return []
        with os.scandir(folder) as entries:
            return [(e.name, e.stat(follow_symlinks=False).st_mtime) for e in entries if e.is_file(follow_symlinks=False)]
    except OSError:
        return []


def _read_json(folder: Path, name: str) -> Any:
    for _path, raw in regular_files(folder, name):  # the exact name: no wildcard in it
        try:
            return json.loads(raw)
        except ValueError:
            return None
    return None


__all__ = ["MAIN_CHAT", "ThreadTools", "as_thread_id", "conversation_thread"]
