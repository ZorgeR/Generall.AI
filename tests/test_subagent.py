"""run_subagent: child wiring, budget accounting, tool subsets, depth."""
import importlib
import os

import pytest

from agents.trace import ToolTrace, TurnBudget


@pytest.fixture(scope="module")
def agents_main(tmp_path_factory):
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("agent"))
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    try:
        yield importlib.import_module("agents.main")
    finally:
        os.chdir(cwd)


class Tools:
    def __init__(self, names):
        self.tools_schema = [{"name": n} for n in names]

    async def execute_tool(self, name, args):
        return f"{name} ok"


async def test_subagent_runs_child_with_shared_budget_and_subset(agents_main, monkeypatch):
    from agents.subagent import SubagentTools

    parent = agents_main.AgentAnthropic(user_id="7")
    parent.search_tools = Tools(["search_web", "deep_research"])
    parent.file_ops = Tools(["read_file"])
    parent.budget = TurnBudget(limit=20)
    parent.thinking = True
    sub = SubagentTools(parent, {"tools": {"max_iteration": 20}, "critique": {"enabled": True}, "judge": {"enabled": True}})
    parent.subagents = sub
    sub.trace = ToolTrace()
    seen = {}

    async def fake_generate(self, **kw):
        seen["self"] = self
        seen["kw"] = kw
        kw["budget"].take(4)  # the child used 4 tool calls
        return "REPORT", []

    monkeypatch.setattr(agents_main.AgentAnthropic, "generate_response", fake_generate)

    result = await sub.execute_tool("run_subagent", {"task": "find X", "tools": ["search_web", "nonexistent"], "model": "fast", "max_tool_calls": 6})
    assert result == "REPORT"
    child = seen["self"]
    assert child is not parent and child.subagents is None and child.trace_depth == 1 and child.thinking is True
    assert child.search_tools is parent.search_tools and child.file_ops is parent.file_ops
    assert child.allowed_tools == {"search_web"}
    assert [t["name"] for t in child.get_tools_schema()] == ["search_web"]
    assert child.model == agents_main.ANTHROPIC_MODEL_FAST
    kw = seen["kw"]
    assert kw["budget"].limit == 6 and kw["user_settings"]["tools"]["max_iteration"] == 6
    assert kw["user_settings"]["critique"]["enabled"] is False and kw["user_settings"]["judge"]["enabled"] is False
    assert kw["messages"] == [{"role": "user", "content": "find X"}] and "subagent" in kw["system_role"].lower()
    assert kw["trace"] is sub.trace
    assert parent.budget.used == 4  # charged to the parent


async def test_subagent_budget_is_capped_by_parent_remaining(agents_main, monkeypatch):
    from agents.subagent import SubagentTools

    parent = agents_main.AgentAnthropic(user_id="7")
    parent.budget = TurnBudget(limit=5)
    parent.budget.take(3)
    sub = SubagentTools(parent, {})
    captured = {}

    async def fake_generate(self, **kw):
        captured["limit"] = kw["budget"].limit
        return "ok", []

    monkeypatch.setattr(agents_main.AgentAnthropic, "generate_response", fake_generate)
    assert await sub.run({"task": "t", "max_tool_calls": 50}) == "ok"
    assert captured["limit"] == 2  # only 2 left on the parent
    parent.budget.take(2)
    assert (await sub.run({"task": "t"})).startswith("Error: no tool-call budget left")
    assert (await sub.run({"task": "   "})).startswith("Error: 'task' is required")


def test_parent_tool_schema_includes_run_subagent_and_request_builder(agents_main):
    from agents.subagent import SubagentTools

    parent = agents_main.AgentAnthropic(user_id="7")
    parent.subagents = SubagentTools(parent, {})
    assert [t["name"] for t in parent.get_tools_schema()] == ["run_subagent"]
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "q"}], "_ephemeral": True},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": "follow-up"},
    ]
    out = agents_main.AgentAnthropic._request_messages(msgs, context_index=2, request_context="<context>now</context>")
    assert "_ephemeral" not in out[0] and msgs[0]["_ephemeral"] is True  # original untouched
    assert out[2]["content"] == [{"type": "text", "text": "<context>now</context>"}, {"type": "text", "text": "follow-up"}]
    assert agents_main.AgentAnthropic._request_messages(msgs)[2]["content"] == "follow-up"
