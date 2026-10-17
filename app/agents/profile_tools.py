"""ProfileTools: the agent-managed user profile and notes.

Four small tools over ``bot.profile.ProfileStore`` (``data/<uid>/profile.md``): read, add, remove,
replace. They run in the bot process (file I/O in a worker thread), never sandboxed, and are
registered always (``ChainOfThoughtAgent.__init__`` → ``register_provider``). The ``profile``
setting only gates the injection: with ``profile.enabled`` off the tools still work, but the
``<profile>`` block is not added to the ``<context>`` of the turn, so the model sees the file only
when it reads it.

``prompt_notes`` is what the model is told about the profile; ``ChainOfThoughtAgent`` appends it to
the static system prompt inside ``<capabilities>``. The two helpers at the end serve the context
block: ``profile_block`` renders ``<profile fingerprint="...">`` and ``last_shown_profile_fingerprint``
finds the version a transcript already showed, so an unchanged profile is not repeated each turn.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from agents.transcript import is_context_block, is_user_turn
from bot.profile import FIXED_SECTIONS, MAX_BYTES, Profile, ProfileStore, Saved, fingerprint, profile_store

_FINGERPRINT_RE = re.compile(r'<profile fingerprint="([0-9a-f]+)">')
_SECTION_HINT = "One of About, Preferences, Notes (case-insensitive), or a new short title for another section."


def profile_block(text: str, fp: str | None = None) -> str:
    """The ``<profile>`` element of a ``<context>`` block for this version of the file."""
    body = (text or "").strip("\n")
    return f'<profile fingerprint="{fp or fingerprint(text)}">\n{body}\n</profile>'


def last_shown_profile_fingerprint(messages: list[dict]) -> str | None:
    """The fingerprint of the last ``<profile>`` block in this transcript's context blocks, or
    None when the profile was never shown (or the turn that showed it was summarized away)."""
    last: str | None = None
    for message in messages or []:
        if not isinstance(message, dict) or not is_user_turn(message):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if is_context_block(block):
                found = _FINGERPRINT_RE.findall(block["text"])
                if found:
                    last = found[-1]
    return last


class ProfileTools:
    prompt_notes = (
        "You keep a profile of the user with the user_profile_* tools (sections About, Preferences, Notes; "
        "the current profile appears in the <profile> element of the <context> block whenever it changed, so "
        "you do not need to read it first). When a durable fact comes up - name, city or timezone, language, "
        "how they like their answers, ongoing projects, preferences - save it with user_profile_add, quietly, "
        "without announcing it. Honour \"remember that ...\" (add) and \"forget that ...\" (user_profile_remove); "
        "when a fact changes, use user_profile_replace instead of adding a second entry. Keep entries short, one "
        "fact per entry, never duplicate what the profile already says, and put dated observations in Notes."
    )

    def __init__(self, user_id: str, store: ProfileStore | None = None) -> None:
        self.user_id = str(user_id)
        self.store = store or profile_store

    @property
    def tools_schema(self) -> list[dict]:
        section = {"type": "string", "description": _SECTION_HINT}
        return [
            {
                "name": "user_profile_read",
                "description": "Read the user's profile (About, Preferences, Notes) as it is stored now.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "user_profile_add",
                "description": (
                    "Add one short entry to a section of the user's profile. Notes entries are dated "
                    f"automatically. The file is capped at {MAX_BYTES // 1024} KB: beyond it the oldest Notes "
                    "are dropped. An entry the section already has is not added twice."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "section": section,
                        "text": {"type": "string", "description": "The entry: one fact, one line, no bullet."},
                    },
                    "required": ["section", "text"],
                },
            },
            {
                "name": "user_profile_remove",
                "description": "Remove one entry from a section of the user's profile ('forget that').",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "section": section,
                        "needle": {
                            "type": "string",
                            "description": "The entry's text, a unique part of it, or its 1-based position in the section.",
                        },
                    },
                    "required": ["section", "needle"],
                },
            },
            {
                "name": "user_profile_replace",
                "description": "Rewrite one entry of the user's profile in place (a fact that changed).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "section": section,
                        "old": {"type": "string", "description": "The current entry: its text, a unique part of it, or its 1-based position."},
                        "new": {"type": "string", "description": "The new text of the entry."},
                    },
                    "required": ["section", "old", "new"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        args = tool_args or {}
        try:
            if tool_name == "user_profile_read":
                return await asyncio.to_thread(self._read)
            if tool_name == "user_profile_add":
                return await asyncio.to_thread(self._add, args.get("section", ""), args.get("text", ""))
            if tool_name == "user_profile_remove":
                return await asyncio.to_thread(self._remove, args.get("section", ""), args.get("needle", ""))
            if tool_name == "user_profile_replace":
                return await asyncio.to_thread(self._replace, args.get("section", ""), args.get("old", ""), args.get("new", ""))
        except Exception as e:  # noqa: BLE001 - a tool result, never an exception for the turn
            return f"Error: {type(e).__name__}: {e}"
        return f"Unknown tool: {tool_name}"

    # ---- the tools (worker thread) ---------------------------------------------
    def _read(self) -> str:
        text = self.store.text(self.user_id).strip()
        return text or "The profile is empty. Sections: " + ", ".join(FIXED_SECTIONS) + "."

    def _add(self, section: str, text: str) -> str:
        if not str(text or "").strip():
            return "Error: 'text' is empty; give the entry to add."
        saved: Saved = self.store.add(self.user_id, section or FIXED_SECTIONS[0], text)
        if saved.duplicate:
            return f"Already in {saved.section}: {saved.entry}\n\n" + self._section_view(saved.section)
        note = ""
        if saved.rotated:
            note = f" The profile was over {MAX_BYTES // 1024} KB, so the {len(saved.rotated)} oldest note(s) were dropped."
        return f"Saved to {saved.section}: {saved.entry}.{note}\n\n" + self._section_view(saved.section)

    def _remove(self, section: str, needle) -> str:
        if self.store.remove(self.user_id, section, needle):
            return f"Removed from {section}: {needle}\n\n" + self._section_view(section)
        return f"Nothing removed: no single entry in {section or '?'} matches {needle!r}.\n\n" + self._section_view(section)

    def _replace(self, section: str, old, new: str) -> str:
        if not str(new or "").strip():
            return "Error: 'new' is empty; give the new text of the entry."
        if self.store.replace(self.user_id, section, old, new):
            return f"Replaced in {section}: {old} -> {new}\n\n" + self._section_view(section)
        return f"Nothing replaced: no single entry in {section or '?'} matches {old!r}.\n\n" + self._section_view(section)

    def _section_view(self, section: str) -> str:
        profile: Profile = self.store.load(self.user_id)
        title = profile.section_title(section)
        entries = profile.entries(title) if title else []
        if not title:
            known = [t for t, e in profile.sections.items() if e] or list(FIXED_SECTIONS)
            return f"Sections: {', '.join(known)}."
        if not entries:
            return f"## {title}\n(empty)"
        return f"## {title}\n" + "\n".join(f"{i}. {e}" for i, e in enumerate(entries, 1))
