"""Per-user settings persisted at ``data/<user_id>/settings.json``.

``DEFAULT_SETTINGS`` is the single source of truth. Loading merges the file
over a deep copy of the defaults, so every category the agent reads always
exists, and unknown keys in an old file are preserved instead of crashing.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any

DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "summarization_history": {"enabled": True, "size": 5},
    "dialog_history": {"enabled": True, "size": 10},
    "reasoning_context": {"enabled": True},
    "short_term_memory": {"enabled": True},
    "critique": {"enabled": False, "max_iteration": 5},
    "judge": {"enabled": False, "max_iteration": 5},
    "tools": {"enabled": True, "max_iteration": 20},
    "semantic_search": {"enabled": True, "max_results": 3},
    "thinking": {"enabled": True},
    "system_prompt": {"type": "generall-ai-v2"},
    "rich_messages": {"enabled": True},
    # One real conversation transcript per chat/topic (tool calls and results included).
    # When on, dialog_history and reasoning_context are ignored; summaries still feed the memory block.
    "transcript": {"enabled": True, "max_context_tokens": 120000, "keep_tool_results_turns": 3, "max_tool_result_chars": 20000},
    # After a turn the status message is shortened into a summary (tool calls, thinking, tokens)
    # kept above the answer; off = delete it as before.
    "trace": {"keep_summary": True},
    # Image generation defaults. The agent may override engine/quality from the request
    # (e.g. the user asks for the best result while preferring "fast"); max_variants is a hard cap.
    "image": {"engine": "auto", "quality": "auto", "size": "2K", "max_variants": 4, "metadata": True},
    # The model may end an answer with a <follow_ups> block; it is removed from the text and
    # shown as buttons under the answer. Off = the instruction is not added to the prompt.
    "follow_ups": {"enabled": True},
    # Quick answers vs. deep work (docs/design-v3.md, F3): auto = the fast-model classifier picks the
    # quick path or the full agent per message; speed = always the fast model; intelligence = always
    # the full agent with deep thinking.
    "mode": {"type": "auto"},
    # IANA zone name of the user ("" = unknown: times are shown in UTC until the user or the agent sets it).
    "timezone": {"name": ""},
    # Agent-managed profile and notes (data/<uid>/profile.md) injected into the context block.
    "profile": {"enabled": True},
    # /settings page state: advanced = every setting is shown; off = only the everyday ones
    # (the presets row and the section summaries still cover the hidden values).
    "ui": {"advanced": False},
}

SYSTEM_PROMPT_TYPES = ("generall-ai-v2", "generall-ai-v1", "perplexity-deep-research", "perplexity-r1")
MODE_TYPES = ("auto", "speed", "intelligence")

# Presets (docs/design-v3.md, F6): one tap sets the few values that decide speed vs. depth.
# {(category, key): value}; "balanced" is the defaults. Applied with apply_preset, recognized with
# current_preset (every key must match exactly; anything else is a custom state with no preset).
SETTINGS_PRESETS: dict[str, dict[tuple[str, str], Any]] = {
    "fast": {("mode", "type"): "speed", ("thinking", "enabled"): False, ("tools", "max_iteration"): 10},
    "balanced": {("mode", "type"): "auto", ("thinking", "enabled"): True, ("tools", "max_iteration"): 20},
    "deep": {("mode", "type"): "intelligence", ("thinking", "enabled"): True, ("tools", "max_iteration"): 40},
}
PRESET_NAMES = tuple(SETTINGS_PRESETS)

SIZE_MIN, SIZE_MAX = 1, 50
CONTEXT_TOKENS_MIN, CONTEXT_TOKENS_MAX = 20000, 400000
ITERATION_MIN, ITERATION_MAX = 1, 300
SEMANTIC_MIN, SEMANTIC_MAX = 1, 20


class UserSettings:
    def __init__(self, user_id: str, base_dir: str = "data") -> None:
        self.user_id = str(user_id)
        self.base_dir = base_dir
        self.settings: dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
        self.load()

    @property
    def path(self) -> str:
        return os.path.join(self.base_dir, self.user_id, "settings.json")

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:  # noqa: BLE001 - a corrupt file must not lock the user out
            return
        if not isinstance(loaded, dict):
            return
        for key, value in loaded.items():
            current = self.settings.get(key)
            if isinstance(value, dict) and isinstance(current, dict):
                current.update(value)
            else:
                self.settings[key] = value

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=2)

    def get(self, category: str, key: str | None = None) -> Any:
        if key is None:
            return self.settings.get(category)
        value = self.settings.get(category)
        if isinstance(value, dict):
            return value.get(key)
        return None

    def set(self, category: str, value: Any, key: str | None = None) -> None:
        if key is None:
            self.settings[category] = value
        else:
            if not isinstance(self.settings.get(category), dict):
                self.settings[category] = {}
            self.settings[category][key] = value
        self.save()

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.settings)

    # ---- validation helpers used by the /settings UI ---------------------
    @staticmethod
    def validate_size(size: int) -> int:
        return max(SIZE_MIN, min(SIZE_MAX, int(size)))

    @staticmethod
    def validate_iteration(iteration: int, _type: str = "") -> int:
        return max(ITERATION_MIN, min(ITERATION_MAX, int(iteration)))

    @staticmethod
    def validate_semantic_max_results(max_results: int) -> int:
        return max(SEMANTIC_MIN, min(SEMANTIC_MAX, int(max_results)))

    @staticmethod
    def validate_context_tokens(tokens: int) -> int:
        return max(CONTEXT_TOKENS_MIN, min(CONTEXT_TOKENS_MAX, int(tokens)))


def current_preset(user_settings: UserSettings) -> str | None:
    """The preset whose every value the settings hold right now, or None (a custom state)."""
    for name, values in SETTINGS_PRESETS.items():
        if all(user_settings.get(category, key) == value for (category, key), value in values.items()):
            return name
    return None


def apply_preset(user_settings: UserSettings, name: str | None) -> bool:
    """Set every value of preset ``name`` (one save); True when a stored value changed.

    Unknown names change nothing. Other settings are left as they are, so a preset is a
    starting point, not a reset.
    """
    values = SETTINGS_PRESETS.get(str(name or "").lower())
    if values is None:
        return False
    changed = False
    for (category, key), value in values.items():
        current = user_settings.settings.get(category)
        if not isinstance(current, dict):
            user_settings.settings[category] = current = {}
        if current.get(key) != value:
            current[key] = value
            changed = True
    if changed:
        user_settings.save()
    return changed
