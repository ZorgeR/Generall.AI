"""VisionTools: let the agent look at images and documents in the workspace (docs/design-v3.md, F7).

Skeleton registered in ``ChainOfThoughtAgent.__init__`` (``register_provider``); the F7 package
fills in ``view_image``, ``describe_image`` and ``read_document``. Runs in the bot process, never
sandboxed: ``tool_integrator.py`` must not name this class or its methods.
"""
from __future__ import annotations

from typing import Any


class VisionTools:
    prompt_notes = ""  # appended to the static system prompt inside <capabilities> (see ChainOfThoughtAgent)

    def __init__(self, user_id: str, sender=None) -> None:
        self.user_id = str(user_id)
        self.sender = sender

    @property
    def tools_schema(self) -> list[dict]:
        return []

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str | list:
        return f"Unknown tool: {tool_name}"
