"""``run_subagent``: delegate one self-contained task to a fresh agent loop.

The subagent is a second ``AgentAnthropic`` with the same tool providers (or a
named subset), an empty conversation that is never persisted, the parent's
trace (its tool lines render indented) and a slice of the parent's tool-call
budget. It returns its final text as the tool result; files it creates are in
the same workspace. Depth is one: a subagent gets no ``run_subagent`` tool.
Several subagents requested in one assistant message run concurrently through
the normal parallel tool execution.
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone

from agents.trace import TurnBudget
from models import ANTHROPIC_MODEL, ANTHROPIC_MODEL_FAST

logger = logging.getLogger(__name__)

PROVIDER_SLOTS = (
    "file_ops", "search_tools", "code_tools", "terminal_tools", "time_tools",
    "image_tools", "video_tools", "sms_tools", "user_interactions",
)
DEFAULT_MAX_TOOL_CALLS = 10
MAX_TASK_CHARS = 8000

SUBAGENT_SYSTEM = """You are a subagent of a persistent personal assistant. The main agent delegated ONE self-contained task to you and is waiting for your report. The user does not see your work; only the main agent reads your final message.

Rules:
- Use the tools until the task is done, then answer with a complete, self-contained report: facts found, numbers, sources or URLs, and the paths of any files you created in the workspace.
- Do not ask questions; state your assumptions instead.
- Do not send messages to the user yourself unless the task explicitly says so.
- Be concise but keep every detail the main agent needs to use your result.

Current time in UTC+0: {now}
"""


class SubagentTools:
    """Tool provider wired into the main agent only (see ChainOfThoughtAgent.__init__)."""

    def __init__(self, parent, user_settings: dict) -> None:
        self.parent = parent
        self.user_settings = user_settings or {}
        self.update_status = None  # set per turn by ChainOfThoughtAgent
        self.trace = None

    @property
    def tools_schema(self) -> list[dict]:
        return [{
            "name": "run_subagent",
            "description": (
                "Delegate a self-contained sub-task to a subagent that has its own fresh context and the same tools "
                "(web search, page download, code execution, files, ...). Use it for research that needs many searches "
                "or page reads, for independent sub-tasks that can run in parallel (call run_subagent several times in one "
                "message), or to keep bulky intermediate work out of your own context. Give the subagent everything it "
                "needs in the task text: it cannot see this conversation. It returns a text report; files it creates are "
                "in the shared workspace."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Complete, self-contained description of the task and the expected report."},
                    "tools": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional: restrict the subagent to these tool names (default: all tools).",
                    },
                    "model": {
                        "type": "string", "enum": ["main", "fast"],
                        "description": "'main' (default) for judgement-heavy work, 'fast' for cheap bulk reading/extraction.",
                    },
                    "max_tool_calls": {
                        "type": "integer",
                        "description": f"Tool-call allowance for the subagent (default {DEFAULT_MAX_TOOL_CALLS}); it also counts against your own budget.",
                    },
                },
                "required": ["task"],
            },
        }]

    async def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        if tool_name != "run_subagent":
            return f"Unknown tool: {tool_name}"
        return await self.run(tool_args or {})

    def _available_tool_names(self) -> set[str]:
        names: set[str] = set()
        for slot in PROVIDER_SLOTS:
            provider = getattr(self.parent, slot, None)
            if provider is not None:
                names.update(t["name"] for t in provider.tools_schema)
        return names

    def _child(self, model: str, allowed: set[str] | None, budget: TurnBudget):
        from agents.main import AgentAnthropic  # circular import at module level

        child = AgentAnthropic(model=model, user_id=self.parent.user_id)
        for slot in PROVIDER_SLOTS:
            setattr(child, slot, getattr(self.parent, slot, None))
        child.subagents = None  # depth 1: a subagent cannot spawn subagents
        child.allowed_tools = allowed
        child.thinking = getattr(self.parent, "thinking", False)
        child.trace_depth = getattr(self.parent, "trace_depth", 0) + 1
        child.budget = budget
        return child

    async def run(self, args: dict) -> str:
        task = str(args.get("task") or "").strip()
        if not task:
            return "Error: 'task' is required."
        task = task[:MAX_TASK_CHARS]
        model = ANTHROPIC_MODEL_FAST if args.get("model") == "fast" else ANTHROPIC_MODEL

        parent_budget: TurnBudget | None = getattr(self.parent, "budget", None)
        requested = args.get("max_tool_calls") or DEFAULT_MAX_TOOL_CALLS
        try:
            requested = max(1, int(requested))
        except (TypeError, ValueError):
            requested = DEFAULT_MAX_TOOL_CALLS
        remaining = parent_budget.remaining if parent_budget is not None else requested
        if remaining <= 0:
            return "Error: no tool-call budget left for a subagent; answer with what you already have."
        budget = TurnBudget(limit=min(requested, remaining))

        allowed: set[str] | None = None
        if isinstance(args.get("tools"), list) and args["tools"]:
            available = self._available_tool_names()
            allowed = {str(t) for t in args["tools"] if str(t) in available} or None

        settings = copy.deepcopy(self.user_settings)
        settings.setdefault("tools", {})["max_iteration"] = budget.limit
        settings.setdefault("tools", {})["enabled"] = True
        settings.setdefault("critique", {})["enabled"] = False
        settings.setdefault("judge", {})["enabled"] = False

        child = self._child(model, allowed, budget)
        system = SUBAGENT_SYSTEM.format(now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Subagent start (model=%s, budget=%d, tools=%s): %.80s", model, budget.limit, sorted(allowed) if allowed else "all", task)
        try:
            text, _ = await child.generate_response(
                messages=[{"role": "user", "content": task}],
                system_role=system,
                question=task,
                update_status=self.update_status,
                user_settings=settings,
                trace=self.trace,
                budget=budget,
            )
        finally:
            if parent_budget is not None:
                parent_budget.take(budget.used)
        logger.info("Subagent done: %d tool calls, %d chars", budget.used, len(text or ""))
        return (text or "").strip() or "The subagent finished without a text report."
