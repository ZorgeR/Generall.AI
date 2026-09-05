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
}

SYSTEM_PROMPT_TYPES = ("generall-ai-v2", "generall-ai-v1", "perplexity-deep-research", "perplexity-r1")

SIZE_MIN, SIZE_MAX = 1, 50
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
