"""AgentAnthropic.run_tool_batch: concurrent execution, ordering, error isolation, trace."""
import asyncio
import importlib
import time
from types import SimpleNamespace

import pytest

from agents.trace import ToolTrace


@pytest.fixture(scope="module")
def agents_main(tmp_path_factory):
    """Import agents.main once with dummy keys and a throwaway cwd (it creates data/ on import)."""
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("agent"))
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    try:
        yield importlib.import_module("agents.main")
    finally:
        os.chdir(cwd)


class SlowTools:
    tools_schema = [{"name": "slow_a"}, {"name": "slow_b"}, {"name": "boom"}, {"name": "soft_error"}]

    async def execute_tool(self, name, args):
        if name == "boom":
            raise RuntimeError("provider crashed")
        await asyncio.sleep(args.get("delay", 0.3))
        if name == "soft_error":
            return "Error: file not found"
        return f"{name} done with {args.get('x')}"


def block(name, **args):
    return SimpleNamespace(type="tool_use", id=f"id-{name}", name=name, input=args)


async def test_batch_runs_concurrently_keeps_order_and_isolates_errors(agents_main):
    agent = agents_main.AgentAnthropic(user_id=None)
    agent.user_interactions = SlowTools()  # awaited provider path
    trace = ToolTrace()
    refreshed = []

    async def refresh(details):
        refreshed.append(details)

    blocks = [block("slow_a", x=1, delay=0.3), block("boom"), block("slow_b", x=2, delay=0.3), block("soft_error")]
    t0 = time.monotonic()
    results = await agent.run_tool_batch(blocks, trace=trace, refresh=refresh)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.55, f"tools ran sequentially ({elapsed:.2f}s)"  # 3 x 0.3s sequential would be 0.9s
    assert [r["tool_use_id"] for r in results] == ["id-slow_a", "id-boom", "id-slow_b", "id-soft_error"]
    assert results[0]["content"] == "slow_a done with 1" and "is_error" not in results[0]
    assert results[1]["is_error"] is True and "provider crashed" in results[1]["content"]
    assert results[3]["content"] == "Error: file not found" and "is_error" not in results[3]  # tool-reported, not a crash
    assert [c.name for c in trace.calls] == ["slow_a", "boom", "slow_b", "soft_error"]
    assert [c.ok for c in trace.calls] == [True, False, True, False]
    assert trace.running == 0 and len(refreshed) == 4


async def test_batch_without_trace_or_refresh(agents_main):
    agent = agents_main.AgentAnthropic(user_id=None)
    agent.user_interactions = SlowTools()
    results = await agent.run_tool_batch([block("slow_a", x=9, delay=0.01)])
    assert results == [{"type": "tool_result", "tool_use_id": "id-slow_a", "content": "slow_a done with 9"}]
