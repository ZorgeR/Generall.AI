"""The user's profile and notes: ``data/<uid>/profile.md``.

One Markdown file per user that the agent maintains with the ``user_profile_*`` tools
(``agents/profile_tools.py``) and the user reads with ``/profile``::

    ## About
    - Name: Anna, lives in Lisbon (Europe/Lisbon)

    ## Preferences
    - Short answers, metric units

    ## Notes
    - [2026-10-17] Preparing a talk on urban beekeeping for November

The three fixed sections always exist in this order; any other ``## Title`` the agent creates is
kept after them in file order. An entry is one ``- `` bullet line; Notes entries carry the day they
were written (``[YYYY-MM-DD] ``), which is how the cap rotates: when the file would exceed
``MAX_BYTES`` the oldest Notes go first, About and Preferences are never dropped.

All methods are synchronous file I/O (callers use ``asyncio.to_thread``), serialized per user with a
``threading.Lock`` and written atomically (a fresh tmp file + ``os.replace``). A missing or corrupt
file is never an error: the store starts fresh and logs. The sandbox can write ``data/<uid>``, so a
``profile.md`` that is a symbolic link (or not a regular file) reads as absent and is never followed
(``read_regular_text``); the next write replaces the link itself. ``fingerprint(text)`` names a
version of the file; the agent's ``<context>`` block repeats the profile only when that changed
(``agents/main.py``).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

FIXED_SECTIONS = ("About", "Preferences", "Notes")
NOTES = "Notes"
MAX_BYTES = 8 * 1024  # the whole file, UTF-8
MAX_ENTRY_CHARS = 600  # one entry; longer text is cut (the agent is told to keep entries short)

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]\s+)")
_DATE_PREFIX_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]\s*")


@dataclass
class Profile:
    """The parsed file: section title -> entries, in file order (fixed sections first)."""

    sections: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered = {title: list(self.sections.get(title, [])) for title in FIXED_SECTIONS}
        for title, entries in self.sections.items():
            if title not in ordered:
                ordered[title] = list(entries)
        self.sections = ordered

    def entries(self, section: str) -> list[str]:
        title = self.section_title(section)
        return list(self.sections.get(title, [])) if title else []

    def section_title(self, section: str) -> str | None:
        """The stored title matching ``section`` (case-insensitive), or None."""
        wanted = _clean_title(section).lower()
        for title in self.sections:
            if title.lower() == wanted:
                return title
        return None

    def is_empty(self) -> bool:
        return not any(self.sections.values())

    def render(self) -> str:
        """The file text: non-empty sections only, so an empty profile renders as ""."""
        parts = []
        for title, entries in self.sections.items():
            if entries:
                parts.append(f"## {title}\n" + "\n".join(f"- {e}" for e in entries))
        return ("\n\n".join(parts) + "\n") if parts else ""

    def size(self) -> int:
        return len(self.render().encode("utf-8"))


@dataclass
class Saved:
    """What ``ProfileStore.add`` did."""

    section: str
    entry: str
    duplicate: bool = False  # the entry was already there; nothing was written
    rotated: list[str] = field(default_factory=list)  # oldest Notes dropped to stay under the cap


def fingerprint(text: str) -> str:
    """12 hex characters naming this version of the profile text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def parse(text: str) -> Profile:
    """Tolerant parse: ``## Title`` starts a section, bullets are entries, other non-empty lines
    inside a section are entries too, anything before the first heading goes to About."""
    sections: dict[str, list[str]] = {}
    current = FIXED_SECTIONS[0]
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            current = _clean_title(heading.group(1)) or current
            sections.setdefault(current, [])
            continue
        if line.startswith("# "):  # a top-level title, not an entry
            continue
        entry = _entry_text(line)
        if entry:
            sections.setdefault(current, []).append(entry)
    return Profile(sections)


def _clean_title(section: str) -> str:
    title = " ".join(str(section or "").replace("#", " ").split())
    for fixed in FIXED_SECTIONS:  # canonical spelling of the fixed sections
        if title.lower() == fixed.lower():
            return fixed
    return title[:40]


def _entry_text(text: str) -> str:
    """One entry: a single line, no bullet marker, cut to MAX_ENTRY_CHARS."""
    entry = _BULLET_RE.sub("", " ".join(str(text or "").split()), count=1).strip()
    if len(entry) > MAX_ENTRY_CHARS:
        entry = entry[: MAX_ENTRY_CHARS - 1].rstrip() + "…"
    return entry


def _strip_date(entry: str) -> str:
    return _DATE_PREFIX_RE.sub("", entry, count=1)


def _find(entries: list[str], needle) -> int | None:
    """Index of the entry ``needle`` names: the exact text (with or without its date and bullet,
    case-insensitive), a 1-based position, or a substring that matches exactly one entry."""
    if isinstance(needle, bool) or needle is None:
        return None
    if isinstance(needle, int):
        return needle - 1 if 1 <= needle <= len(entries) else None
    text = _entry_text(str(needle))
    if not text:
        return None
    lowered = text.lower()
    for i, entry in enumerate(entries):
        if lowered in (entry.lower(), _strip_date(entry).lower()):
            return i
    if text.isdigit() and 1 <= int(text) <= len(entries):
        return int(text) - 1
    matches = [i for i, entry in enumerate(entries) if lowered in entry.lower()]
    return matches[0] if len(matches) == 1 else None


def read_regular_text(path: Path) -> str:
    """The UTF-8 text of ``path``; "" when there is no such file.

    ``data/<uid>`` is the sandbox's workspace, so the file may have been replaced by a symbolic
    link (to another user's profile, to ``/proc/self/environ``) or a FIFO: the last component is
    never followed (``O_NOFOLLOW``) and anything but a regular file raises ``OSError``, so the
    host never reads what the link points to. Bad UTF-8 raises ``UnicodeDecodeError``.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and path.is_symlink():  # no O_NOFOLLOW on this platform
        raise OSError(f"{path} is a symbolic link")
    try:
        fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return ""
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise OSError(f"{path} is not a regular file")
        return f.read().decode("utf-8")


class ProfileStore:
    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def path(self, user_id: str) -> Path:
        return self.base_dir / str(user_id) / "profile.md"

    def _lock(self, user_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(str(user_id), threading.Lock())

    # ---- reading ---------------------------------------------------------------
    def text(self, user_id: str) -> str:
        """The file as it is, "" when there is none (or it cannot be read, or is not a plain file)."""
        path = self.path(user_id)
        try:
            return read_regular_text(path)
        except (OSError, UnicodeDecodeError) as e:
            logger.error("Unreadable profile %s (%s); treating it as empty", path, e)
            return ""

    def load(self, user_id: str) -> Profile:
        return parse(self.text(user_id))

    def fingerprint(self, user_id: str) -> str:
        return fingerprint(self.text(user_id))

    # ---- writing ---------------------------------------------------------------
    def add(self, user_id: str, section: str, text: str, *, today: date | None = None) -> Saved:
        """Append one entry (Notes entries get today's date). An entry already present in the
        section (case-insensitive, date ignored) is not written twice."""
        with self._lock(user_id):
            profile = self.load(user_id)
            title = profile.section_title(section) or _clean_title(section) or FIXED_SECTIONS[0]
            entry = _entry_text(text)
            if not entry:
                raise ValueError("an entry needs some text")
            entries = profile.sections.setdefault(title, [])
            bare = _strip_date(entry).lower()
            if any(_strip_date(e).lower() == bare for e in entries):
                return Saved(title, entry, duplicate=True)
            if title == NOTES and not _DATE_PREFIX_RE.match(entry):
                day = today or datetime.now(timezone.utc).date()
                entry = f"[{day.isoformat()}] {entry}"
            entries.append(entry)
            rotated = self._save(user_id, profile)
            return Saved(title, entry, rotated=rotated)

    def remove(self, user_id: str, section: str, needle) -> bool:
        """Drop the entry ``needle`` names (see ``_find``); False when the section or entry is unknown."""
        with self._lock(user_id):
            profile = self.load(user_id)
            title = profile.section_title(section)
            if not title:
                return False
            entries = profile.sections[title]
            index = _find(entries, needle)
            if index is None:
                return False
            del entries[index]
            self._save(user_id, profile)
            return True

    def replace(self, user_id: str, section: str, old_needle, new_text: str) -> bool:
        """Rewrite the entry ``old_needle`` names in place; a Notes entry keeps its date unless the
        new text brings its own."""
        with self._lock(user_id):
            profile = self.load(user_id)
            title = profile.section_title(section)
            if not title:
                return False
            entries = profile.sections[title]
            index = _find(entries, old_needle)
            entry = _entry_text(new_text)
            if index is None or not entry:
                return False
            if title == NOTES and not _DATE_PREFIX_RE.match(entry):
                old_date = _DATE_PREFIX_RE.match(entries[index])
                day = old_date.group(1) if old_date else datetime.now(timezone.utc).date().isoformat()
                entry = f"[{day}] {entry}"
            entries[index] = entry
            self._save(user_id, profile)
            return True

    def clear(self, user_id: str) -> None:
        with self._lock(user_id):
            self._write(user_id, "")

    def _save(self, user_id: str, profile: Profile) -> list[str]:
        """Write the profile under the cap: the oldest Notes go first; returns what was dropped."""
        rotated: list[str] = []
        notes = profile.sections.get(NOTES, [])
        while profile.size() > MAX_BYTES and notes:
            rotated.append(notes.pop(0))
        if profile.size() > MAX_BYTES:
            logger.warning("Profile of %s is over %d bytes without notes to drop", user_id, MAX_BYTES)
        self._write(user_id, profile.render())
        return rotated

    def _write(self, user_id: str, text: str) -> None:
        path = self.path(user_id)
        if not text:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.error("Could not delete profile %s: %s", path, e)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh tmp file (O_EXCL): a fixed name could be a link the sandbox planted, and writing
        # through it would overwrite whatever it points to. os.replace then swaps the name itself.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".profile.", suffix=".tmp")
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o644)  # mkstemp makes it 0600; keep it readable in the workspace as before
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


profile_store = ProfileStore()
