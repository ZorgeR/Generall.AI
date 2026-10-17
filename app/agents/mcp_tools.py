"""MCPTools: tools of the user's connected MCP servers (docs/design-v3.md, F12).

Skeleton registered in ``ChainOfThoughtAgent.__init__`` (``register_provider``); the F12 package
fills in discovery, the ``mcp_<server>_<tool>`` schemas from the cached tool lists and the calls
over the MCP client. Runs in the bot process, never sandboxed.
"""
from __future__ import annotations

from typing import Any


class MCPTools:
    prompt_notes = ""  # appended to the static system prompt inside <capabilities> (see ChainOfThoughtAgent)

    def __init__(self, user_id: str) -> None:
        self.user_id = str(user_id)

    @property
    def tools_schema(self) -> list[dict]:
        return []

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str | list:
        return f"Unknown tool: {tool_name}"
