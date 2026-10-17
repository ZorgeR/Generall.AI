"""ProfileTools: the agent-managed user profile and notes (docs/design-v3.md, F8).

Skeleton registered in ``ChainOfThoughtAgent.__init__`` (``register_provider``); the F8 package
fills in ``user_profile_read`` / ``user_profile_add`` / ``user_profile_remove`` /
``user_profile_replace`` over ``bot/profile.py``. Runs in the bot process, never sandboxed.
"""
from __future__ import annotations

from typing import Any


class ProfileTools:
    prompt_notes = ""  # appended to the static system prompt inside <capabilities> (see ChainOfThoughtAgent)

    def __init__(self, user_id: str) -> None:
        self.user_id = str(user_id)

    @property
    def tools_schema(self) -> list[dict]:
        return []

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        return f"Unknown tool: {tool_name}"
