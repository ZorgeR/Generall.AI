"""Names of the user's Telegram forum topics (threads): ``data/<uid>/topics.json``.

Telegram tells a bot a topic's name only in service messages: ``forum_topic_created`` (the
message that opened the topic; every later message of the topic carries it as
``reply_to_message``, see ``bot/message_context._is_topic_root``) and ``forum_topic_edited`` (a
rename). No Bot API call returns it later, so the bot keeps what it saw::

    {"topics": {"42": {"name": "Travel plans", "implicit": false, "updated": "2026-10-20T12:00:00+00:00"}}}

The key is the thread id (``message_thread_id``, the id of the creation message: unique in the
chat, never reused). ``implicit`` is Telegram's ``is_name_implicit``: the name was not chosen by
the user (a private-chat thread named after its first message). The agent's ``<context>`` block
names the current topic from here (``ChainOfThoughtAgent._topic_line``), and ``thread_label`` is how
memory lines, ``memory_search`` and the thread tools (``agents/thread_tools.py``) name a conversation.

Two ways in: ``set`` for the service messages themselves (a rename always wins) and ``learn`` for
the creation message a topic message points at, which only fills a gap: after a rename that
message still carries the OLD name. All methods are synchronous file I/O (callers use
``asyncio.to_thread``), serialized per user with a ``threading.Lock``. ``data/<uid>`` is the
sandbox's workspace, so the file is read like ``profile.md`` (never through a link, at most
``MAX_FILE_BYTES``), written through a fresh tmp file + ``os.replace``, and every name is cleaned
again on read: one line, at most ``MAX_NAME_CHARS``. A missing or corrupt file is never an error.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bot.profile import read_regular_text

logger = logging.getLogger(__name__)

MAX_NAME_CHARS = 128  # Telegram's own limit on a topic name
MAX_TOPICS = 500  # per user; the least recently updated ones are forgotten first
MAX_FILE_BYTES = 256 * 1024


@dataclass(frozen=True)
class Topic:
    thread_id: int
    name: str
    implicit: bool = False


MAIN_CHAT = "the main chat"


def thread_label(thread_id: int | None, topic: Topic | None = None) -> str:
    """How memory lines and the thread tools name a conversation: the main chat (no thread), a
    thread by name and id, or by id alone when its name was never seen."""
    if not thread_id:
        return MAIN_CHAT
    if topic is not None:
        return f"thread {json.dumps(topic.name, ensure_ascii=False)} (id {thread_id})"
    return f"thread {thread_id}"


def clean_name(name) -> str:
    """One line of printable text, at most ``MAX_NAME_CHARS`` characters ("" for nothing usable)."""
    text = "".join(ch if ch.isprintable() else " " for ch in str(name or ""))
    return " ".join(text.split())[:MAX_NAME_CHARS].strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TopicStore:
    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def path(self, user_id: str) -> Path:
        return self.base_dir / str(user_id) / "topics.json"

    def _lock(self, user_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(str(user_id), threading.Lock())

    def get(self, user_id: str, thread_id: int | None) -> Topic | None:
        """The stored topic, or None (no thread, or its name was never seen)."""
        if thread_id is None:
            return None
        with self._lock(user_id):
            entry = self._load(user_id).get(str(thread_id))
        if entry is None:
            return None
        return Topic(int(thread_id), entry["name"], entry["implicit"])

    def all(self, user_id: str) -> dict[int, Topic]:
        """Every stored topic of the user by thread id."""
        with self._lock(user_id):
            topics = self._load(user_id)
        return {int(k): Topic(int(k), e["name"], e["implicit"]) for k, e in topics.items()}

    def label(self, user_id: str, thread_id: int | None) -> str:
        """``thread_label`` with the stored name."""
        return thread_label(thread_id, self.get(user_id, thread_id) if thread_id else None)

    def set(self, user_id: str, thread_id: int | None, name, *, implicit: bool = False) -> bool:
        """Store the name a service message announced (created or renamed); True when it changed."""
        return self._put(user_id, thread_id, name, implicit, overwrite=True)

    def learn(self, user_id: str, thread_id: int | None, name, *, implicit: bool = False) -> bool:
        """Store the name of the topic's creation message, only when none is known yet."""
        return self._put(user_id, thread_id, name, implicit, overwrite=False)

    def _put(self, user_id: str, thread_id: int | None, name, implicit: bool, *, overwrite: bool) -> bool:
        name = clean_name(name)
        if thread_id is None or not name:
            return False
        key = str(int(thread_id))
        with self._lock(user_id):
            topics = self._load(user_id)
            old = topics.get(key)
            if old is not None and (not overwrite or (old["name"] == name and old["implicit"] == bool(implicit))):
                return False
            topics[key] = {"name": name, "implicit": bool(implicit), "updated": _now()}
            if len(topics) > MAX_TOPICS:
                newest = sorted(topics, key=lambda k: topics[k]["updated"], reverse=True)[:MAX_TOPICS]
                topics = {k: topics[k] for k in topics if k in newest}
            self._write(user_id, topics)
        return True

    def _load(self, user_id: str) -> dict[str, dict]:
        path = self.path(user_id)
        try:
            text = read_regular_text(path, max_bytes=MAX_FILE_BYTES)
            data = json.loads(text) if text.strip() else {}
        except (OSError, ValueError) as e:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
            logger.warning("Ignoring unreadable %s: %s", path, e)
            return {}
        raw = data.get("topics") if isinstance(data, dict) else None
        topics: dict[str, dict] = {}
        for key, entry in (raw if isinstance(raw, dict) else {}).items():
            if not str(key).isdigit() or not isinstance(entry, dict):
                continue
            name = clean_name(entry.get("name"))
            if name:
                topics[str(int(key))] = {
                    "name": name, "implicit": entry.get("implicit") is True, "updated": str(entry.get("updated") or ""),
                }
        return topics

    def _write(self, user_id: str, topics: dict[str, dict]) -> None:
        path = self.path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh tmp file (O_EXCL), never a fixed name the sandbox could have made a link; os.replace
        # then swaps the name itself (bot/profile.py does the same).
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".topics.", suffix=".tmp")
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o644)  # readable in the workspace, like profile.md
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"topics": topics}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


topic_store = TopicStore()
