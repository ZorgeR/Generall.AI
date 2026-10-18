"""Client-side toolsets in the agent loop (the browser, ``browser_toolset_20260801``).

``AgentAnthropic`` sends a provider's nameless toolset entry with the tools, keeps
``toolset_name`` on the member tool_use blocks it stores, runs the members of one message in
order through the owning provider's ``execute_member`` (stopping at the first failure and
answering the rest with ``NOT_EXECUTED``) while normal tools stay concurrent, echoes
``toolset_name`` on the members' results only, passes ``browser_state`` blocks through, charges
ONE budget unit and ONE quota event per message with browser calls, and never hands the
browser to a subagent.
"""
import asyncio
import copy
import importlib
import os
import time
from types import SimpleNamespace

import anthropic
import pytest

import models
from agents import transcript as T
from agents.trace import ToolTrace, TurnBudget

TOOLSET = {"type": "browser_toolset_20260801", "configs": {"javascript_exec": {"enabled": False}}}
STATE = {"type": "browser_state", "tabs": [{"tab_id": "tab-1", "title": "Example", "url": "https://example.com/", "active": True}]}


@pytest.fixture
def agents_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    module = importlib.import_module("agents.main")
    monkeypatch.setattr(module, "_thinking_binding_refused", False)
    return module


@pytest.fixture
def quota(agents_main, monkeypatch):
    """Every track_tool_used call, in order."""
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(agents_main.stats_tracker, "track_tool_used", lambda uid, name: events.append((uid, name)))
    return events


# ---- fakes ---------------------------------------------------------------------------------
class FakeBrowser:
    """A toolset provider: records every member call with its start/end time."""

    toolset_name = "browser"

    def __init__(self, outcomes=None, delay=0.05):
        self.outcomes = dict(outcomes or {})  # member -> (content, is_error)
        self.delay = delay
        self.calls: list[tuple[str, dict]] = []
        self.spans: list[tuple[str, float, float]] = []
        self.after_batches = 0
        self.tools_schema = [{"name": "browser_confirm", "description": "d", "input_schema": {"type": "object", "properties": {}}}]

    @property
    def toolset(self):
        return copy.deepcopy(TOOLSET)

    async def execute_member(self, name, tool_input):
        start = time.monotonic()
        self.calls.append((name, tool_input))
        await asyncio.sleep(self.delay)
        self.spans.append((name, start, time.monotonic()))
        return self.outcomes.get(name, ([{"type": "text", "text": f"{name} ok"}, STATE], False))

    async def execute_tool(self, name, args):
        return f"{name} called"

    async def after_batch(self):
        self.after_batches += 1


class SlowTools:
    def __init__(self):
        self.tools_schema = [{"name": "slow_a", "input_schema": {}}, {"name": "slow_b", "input_schema": {}}, {"name": "search_web", "input_schema": {}}]
        self.spans: list[tuple[str, float, float]] = []

    async def execute_tool(self, name, args):
        start = time.monotonic()
        await asyncio.sleep(args.get("delay", 0.3))
        self.spans.append((name, start, time.monotonic()))
        return f"{name} done"


def member(tool_id, name, **args):
    """A browser member call as the SDK returns it (the real ToolUseBlock type)."""
    return anthropic.types.ToolUseBlock(id=tool_id, name=name, input=args, type="tool_use", toolset_name="browser")


def plain(tool_id, name, **args):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args)


def text(t):
    return SimpleNamespace(type="text", text=t)


def reply(*blocks, stop="end_turn"):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop, input_transformations=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class FakeStream:
    def __init__(self, outcome):
        self.outcome = outcome

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return self.outcome


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = SimpleNamespace(stream=self._stream)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return FakeStream(self.script.pop(0))


def settings(**tools):
    from bot.settings import DEFAULT_SETTINGS

    s = copy.deepcopy(DEFAULT_SETTINGS)
    s["tools"].update(tools)
    return s


def make_agent(agents_main, browser=None, tools=None, user_id="7"):
    agent = agents_main.AgentAnthropic(user_id=user_id)
    agent.user_interactions = tools or SlowTools()  # an awaited slot
    agent.register_provider(browser if browser is not None else FakeBrowser())
    return agent


# ---- the request ---------------------------------------------------------------------------
def test_the_toolset_entry_is_sent_with_the_tools(agents_main):
    agent = make_agent(agents_main)
    tools = agent.get_tools_schema()
    assert TOOLSET in tools and "name" not in tools[-1]  # nameless, right after its provider's own tools
    assert [t.get("name") for t in tools] == ["slow_a", "slow_b", "search_web", "browser_confirm", None]


def test_allowed_tools_keep_a_toolset_only_by_its_name(agents_main):
    agent = make_agent(agents_main)
    agent.allowed_tools = {"search_web"}
    assert agent.get_tools_schema() == [{"name": "search_web", "input_schema": {}}]
    agent.allowed_tools = {"search_web", "browser"}
    assert [t.get("name") or t["type"] for t in agent.get_tools_schema()] == ["search_web", "browser_toolset_20260801"]


def test_the_fingerprint_is_stable_with_a_nameless_entry(agents_main):
    agent = make_agent(agents_main)
    one = T.prefix_fingerprint(models.ANTHROPIC_MODEL, "sys", agent.get_tools_schema())
    two = T.prefix_fingerprint(models.ANTHROPIC_MODEL, "sys", agent.get_tools_schema())
    assert one == two
    without = agents_main.AgentAnthropic(user_id="7")
    without.user_interactions = SlowTools()
    without.register_provider(SimpleNamespace(tools_schema=FakeBrowser().tools_schema))  # same tools, no toolset
    assert T.prefix_fingerprint(models.ANTHROPIC_MODEL, "sys", without.get_tools_schema()) != one


async def test_critique_lists_tools_with_a_nameless_entry(agents_main, monkeypatch):
    agent = make_agent(agents_main)
    seen = {}

    def parse(**kw):
        seen["prompt"] = kw["messages"][0]["content"]
        raise RuntimeError("no network in tests")

    fake = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(agents_main, "openai_client", fake)
    assert await agent.critique_response("q", "a") is None
    assert "browser_toolset_20260801" in seen["prompt"]


# ---- run_tool_batch ------------------------------------------------------------------------
async def test_members_run_in_order_and_results_echo_the_toolset(agents_main, quota):
    browser = FakeBrowser()
    agent = make_agent(agents_main, browser)
    trace = ToolTrace()
    blocks = [member("b1", "navigate", url="example.com"), member("b2", "read_page", filter="interactive"), member("b3", "screenshot")]
    results = await agent.run_tool_batch(blocks, trace=trace)
    assert [c[0] for c in browser.calls] == ["navigate", "read_page", "screenshot"]
    assert browser.calls[0][1] == {"url": "example.com"}
    starts_ends = browser.spans
    assert all(starts_ends[i][2] <= starts_ends[i + 1][1] + 1e-6 for i in range(2)), "members overlapped"
    assert [r["tool_use_id"] for r in results] == ["b1", "b2", "b3"]
    assert all(r["toolset_name"] == "browser" and "is_error" not in r for r in results)
    assert results[0]["content"] == [{"type": "text", "text": "navigate ok"}, STATE]  # browser_state passed through
    assert [c.name for c in trace.calls] == ["browser.navigate", "browser.read_page", "browser.screenshot"]
    assert "[browser: 1 tab(s); active tab-1: Example https://example.com/]" in trace.calls[0].result_excerpt
    assert browser.after_batches == 1
    assert quota == [("7", "browser")]  # one quota event for the whole batch


async def test_the_first_failure_halts_the_rest(agents_main, quota):
    browser = FakeBrowser({"left_click": ("Error: ref_3 is stale or not found on the current page.", True)})
    agent = make_agent(agents_main, browser)
    trace = ToolTrace()
    blocks = [member("b1", "navigate", url="x.org"), member("b2", "left_click", target={"type": "ref", "ref": "ref_3"}),
              member("b3", "type", text="hi"), member("b4", "key", text="Enter")]
    results = await agent.run_tool_batch(blocks, trace=trace)
    assert [c[0] for c in browser.calls] == ["navigate", "left_click"]  # nothing after the failure ran
    assert "is_error" not in results[0]
    assert results[1]["is_error"] is True and "stale" in results[1]["content"]
    for r in results[2:]:
        assert r == {"type": "tool_result", "tool_use_id": r["tool_use_id"], "content": agents_main.NOT_EXECUTED,
                     "is_error": True, "toolset_name": "browser"}
    assert agents_main.NOT_EXECUTED == "Not executed: an earlier action in this turn failed."
    assert [c.name for c in trace.calls] == ["browser.navigate", "browser.left_click"] and trace.calls[1].ok is False
    assert browser.after_batches == 1  # the live view still shows where it stopped


async def test_a_member_that_raises_halts_too(agents_main, quota):
    browser = FakeBrowser()

    async def boom(name, tool_input):
        browser.calls.append((name, tool_input))
        raise RuntimeError("socket closed")

    browser.execute_member = boom
    agent = make_agent(agents_main, browser)
    results = await agent.run_tool_batch([member("b1", "navigate", url="x"), member("b2", "screenshot")])
    assert results[0]["is_error"] is True and "socket closed" in results[0]["content"] and results[0]["toolset_name"] == "browser"
    assert results[1]["content"] == agents_main.NOT_EXECUTED and len(browser.calls) == 1


async def test_a_mixed_batch(agents_main, quota):
    """Normal tools run concurrently with each other and with the browser sequence; the results
    come back in block order and only the browser's carry toolset_name."""
    browser = FakeBrowser(delay=0.15)
    tools = SlowTools()
    agent = make_agent(agents_main, browser, tools)
    blocks = [plain("t1", "slow_a", delay=0.3), member("b1", "navigate", url="a.com"), plain("t2", "slow_b", delay=0.3),
              member("b2", "screenshot")]
    t0 = time.monotonic()
    results = await agent.run_tool_batch(blocks)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.55, f"ran sequentially ({elapsed:.2f}s)"  # 0.3 + 0.3 + 0.15 + 0.15 in series is 0.9 s
    assert [r["tool_use_id"] for r in results] == ["t1", "b1", "t2", "b2"]
    assert [("toolset_name" in r) for r in results] == [False, True, False, True]
    assert results[0]["content"] == "slow_a done" and results[2]["content"] == "slow_b done"
    assert [c[0] for c in browser.calls] == ["navigate", "screenshot"]
    assert browser.spans[0][2] <= browser.spans[1][1] + 1e-6
    assert sorted(quota) == [("7", "browser"), ("7", "slow_a"), ("7", "slow_b")]


async def test_a_toolset_without_its_provider_answers_with_errors(agents_main, quota):
    agent = agents_main.AgentAnthropic(user_id="7")
    results = await agent.run_tool_batch([member("b1", "navigate", url="x"), member("b2", "screenshot")])
    assert results[0]["is_error"] is True and "not available" in results[0]["content"]
    assert results[1]["content"] == agents_main.NOT_EXECUTED
    assert all(r["toolset_name"] == "browser" for r in results)


async def test_long_text_is_capped_but_state_and_images_stay(agents_main, quota):
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
    browser = FakeBrowser({"get_page_text": ([{"type": "text", "text": "x" * 5000}, image, STATE], False)})
    agent = make_agent(agents_main, browser)
    agent.max_tool_result_chars = 1000
    (result,) = await agent.run_tool_batch([member("b1", "get_page_text")])
    content = result["content"]
    assert len(content[0]["text"]) < 1100 and "truncated 4000 characters" in content[0]["text"]
    assert content[1] == image and content[2] == STATE


async def test_a_failing_quota_write_does_not_fail_the_browser_batch(agents_main, monkeypatch):
    """track_tool_used writes SQLite ("database is locked"): for a normal tool that becomes its error
    result; for a browser batch it went through gather and failed the turn while the rest ran on."""
    import sqlite3

    def locked(uid, name):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(agents_main.stats_tracker, "track_tool_used", locked)
    browser = FakeBrowser()
    agent = make_agent(agents_main, browser)
    results = await agent.run_tool_batch([member("b1", "navigate", url="https://example.com"),
                                          plain("t1", "slow_a", delay=0.01), member("b2", "screenshot")])
    assert [c[0] for c in browser.calls] == ["navigate", "screenshot"] and browser.after_batches == 1
    assert "is_error" not in results[0] and "is_error" not in results[2]
    assert results[0]["content"] == [{"type": "text", "text": "navigate ok"}, STATE]
    assert results[1]["is_error"] is True  # a normal tool: its own accounting failed inside execute_tool, as before


SECRET = "correct-horse-42!xy"  # 19 characters
CARD = "4111 1111 1111 1111"


async def test_typed_text_is_masked_wherever_people_see_the_call(agents_main, quota):
    """What browser.type / form_input put into a page (a password, a card number) is shown as its
    length in the status, the turn summary and the reasoning file; the browser and the API get it."""
    from bot.agent_runner import reasoning_text
    from bot.status import render_trace

    browser = FakeBrowser()
    agent = make_agent(agents_main, browser)
    agent.client = FakeClient([
        reply(member("b1", "type", text=SECRET), member("b2", "form_input", target={"type": "ref", "ref": "ref_4"}, value=CARD),
              member("b3", "form_input", target={"type": "ref", "ref": "ref_5"}, value=True), stop="tool_use"),
        reply(text("Logged in.")),
    ])
    trace = ToolTrace()
    response, messages = await agent.generate_response(
        messages=[{"role": "user", "content": [{"type": "text", "text": "log in"}]}], system_role="sys", question="log in",
        user_settings=settings(max_iteration=5), trace=trace,
    )
    assert response == "Logged in."
    # the browser and the API request get the real input; the stored tool_use is what was sent
    assert browser.calls[0] == ("type", {"text": SECRET}) and browser.calls[1][1]["value"] == CARD
    sent = [b for m in agent.client.requests[1]["messages"] if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_use"]
    assert sent[0]["input"] == {"text": SECRET} and sent[1]["input"]["value"] == CARD
    stored = [b for m in messages if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_use"]
    assert stored == sent

    # what people see: the legacy status line, the rich summary's args JSON, the reasoning file
    assert [c.args for c in trace.calls] == [
        {"text": "<19 characters>"},
        {"target": {"type": "ref", "ref": "ref_4"}, "value": "<19 characters>"},
        {"target": {"type": "ref", "ref": "ref_5"}, "value": True},  # a checkbox's state is no secret
    ]
    status = render_trace(trace)
    summary = "\n".join(c.args_text for c in trace.calls)
    reasoning = reasoning_text(messages)
    for shown in (status, summary, reasoning):
        assert SECRET not in shown and CARD not in shown
    assert "<19 characters>" in status and "<19 characters>" in summary
    assert "[Tool call: browser.type]" in reasoning and '"text": "<19 characters>"' in reasoning
    # a browser result's state block is rendered, not an empty line
    assert "[Tool result: browser.type]\ntype ok\n[browser: 1 tab(s); active tab-1: Example https://example.com/]" in reasoning


def test_masked_input_only_touches_typed_text():
    from agents.browser_tools import masked_input

    args = {"text": "x", "target": {"type": "ref", "ref": "ref_1"}}
    assert masked_input("type", args) == {"text": "<1 character>", "target": {"type": "ref", "ref": "ref_1"}}
    assert args["text"] == "x"  # a copy: the input itself is never changed
    assert masked_input("form_input", {"value": 1234}) == {"value": "<4 characters>"}
    assert masked_input("key", {"text": "Enter"}) == {"text": "Enter"}
    assert masked_input("navigate", {"url": "https://example.com/"}) == {"url": "https://example.com/"}
    assert masked_input("type", None) == {}


# ---- generate_response ----------------------------------------------------------------------
async def test_the_loop_stores_toolset_blocks_and_charges_one_unit_per_message(agents_main, quota):
    browser = FakeBrowser()
    agent = make_agent(agents_main, browser)
    agent.client = FakeClient([
        reply(text("Opening it."), member("b1", "navigate", url="example.com"), member("b2", "screenshot"),
              plain("t1", "search_web", delay=0.01), stop="tool_use"),
        reply(member("b3", "left_click", target={"type": "coordinate", "x": 10, "y": 20}), stop="tool_use"),
        reply(text("Done.")),
    ])
    budget = TurnBudget(20)
    response, messages = await agent.generate_response(
        messages=[{"role": "user", "content": [{"type": "text", "text": "open example.com"}]}],
        system_role="sys", question="open example.com", user_settings=settings(), budget=budget,
    )
    assert response == "Done."
    first, second, third = agent.client.requests
    assert TOOLSET in first["tools"] and first["tool_choice"] == {"type": "auto"}

    stored_call = messages[1]
    assert stored_call["role"] == "assistant"
    assert stored_call["content"][1] == {"type": "tool_use", "id": "b1", "name": "navigate", "input": {"url": "example.com"}, "toolset_name": "browser"}
    assert stored_call["content"][3] == {"type": "tool_use", "id": "t1", "name": "search_web", "input": {"delay": 0.01}}  # no toolset_name
    stored_results = messages[2]["content"]
    assert [r.get("toolset_name") for r in stored_results] == ["browser", "browser", None]
    assert second["messages"][: 3] == messages[:3]  # what is stored is what the next request sent

    # message 1: one browser batch (1 unit) + one normal tool (1 unit); message 2: one batch (1 unit)
    assert budget.used == 3
    assert sorted(quota) == [("7", "browser"), ("7", "browser"), ("7", "search_web")]  # per batch, not per action
    assert browser.after_batches == 2


async def test_a_spent_budget_stops_at_the_next_message(agents_main, quota):
    browser = FakeBrowser()
    agent = make_agent(agents_main, browser)
    agent.client = FakeClient([
        reply(member("b1", "navigate", url="a"), member("b2", "screenshot"), member("b3", "scroll"), stop="tool_use"),
        reply(text("Here is what I found so far."), member("b4", "left_click"), stop="tool_use"),  # budget gone: dropped
    ])
    budget = TurnBudget(1)
    response, messages = await agent.generate_response(
        messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}], system_role="sys", question="go",
        user_settings=settings(max_iteration=5), budget=budget,
    )
    assert [c[0] for c in browser.calls] == ["navigate", "screenshot", "scroll"]  # three actions, one unit
    assert budget.used == 1 and agent.limit_reached is True
    assert response == "Here is what I found so far."


async def test_the_transcript_keeps_toolset_blocks_across_turns(agents_main, monkeypatch, quota):
    """End to end through ChainOfThoughtAgent: the stored transcript holds the toolset_name of
    the member calls and of their results, and the next turn's request starts with it as stored."""
    from agents import browser_tools

    class Manager:
        def __init__(self):
            self.requests = []

        def browser_call(self, uid, request, *, remember=True, timeout=75):
            self.requests.append((uid, request, remember))
            return {"ok": True, "content": [{"type": "text", "text": "Navigated to https://example.com/"}, STATE],
                    "meta": {"tab_id": "tab-1", "title": "Example", "url": "https://example.com/"}}

        def browser_peek(self, uid):
            return None

        def status(self, uid):
            return {}

    manager = Manager()
    monkeypatch.setattr(browser_tools, "session_manager", lambda: manager)
    monkeypatch.setattr(browser_tools, "shell_sessions_enabled", lambda: True)

    def chain(script):
        agent = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=settings())
        agent.agent.client = FakeClient(script)

        async def complex_(question, history=None):
            return "complex"

        async def no_save(*a, **k):
            return ""

        async def context(question, update_status=None, messages=None):
            return "<context>\nnow\n</context>", "<context>\nnow\n</context>"

        monkeypatch.setattr(agent, "_classify_complexity", complex_)
        monkeypatch.setattr(agent, "_save_conversation", no_save)
        monkeypatch.setattr(agent, "_request_context", context)
        return agent

    first = chain([reply(member("b1", "navigate", url="example.com"), stop="tool_use"), reply(text("Opened."))])
    assert first.browser_tools is not None and first.sandbox_tools is not None
    response, _ = await first.generate_response("open example.com")
    assert response == "Opened."
    request = first.agent.client.requests[0]
    assert any(t.get("type") == "browser_toolset_20260801" for t in request["tools"])
    assert any(t.get("name") == "sandbox_status" for t in request["tools"])
    assert manager.requests[0][1]["action"] == "navigate" and manager.requests[0][2] is True

    stored = T.transcript_store.load("42", None).messages
    calls = [b for m in stored for b in m["content"] if b.get("type") == "tool_use"]
    results = [b for m in stored for b in m["content"] if b.get("type") == "tool_result"]
    assert calls[0]["toolset_name"] == "browser" and results[0]["toolset_name"] == "browser"
    assert results[0]["content"][1] == STATE

    second = chain([reply(text("Still there."))])
    await second.generate_response("and now?")
    sent = second.agent.client.requests[0]["messages"]
    assert sent[: len(stored)] == stored  # append-only: the stored turn went out unchanged


# ---- subagents ----------------------------------------------------------------------------------
async def test_subagents_never_get_the_browser(agents_main, monkeypatch):
    from agents.subagent import SubagentTools

    parent = make_agent(agents_main)
    other = SimpleNamespace(tools_schema=[{"name": "view_image"}, {"type": "weird_nameless_entry"}])
    parent.register_provider(other)
    parent.budget = TurnBudget(limit=10)
    sub = SubagentTools(parent, settings())
    parent.subagents = sub
    names = sub._available_tool_names()  # a nameless schema entry does not crash it
    assert "browser_confirm" not in names and "view_image" in names and "" not in names

    seen = {}

    async def fake_generate(self, **kw):
        seen["child"] = self
        return "REPORT", []

    monkeypatch.setattr(agents_main.AgentAnthropic, "generate_response", fake_generate)
    assert await sub.execute_tool("run_subagent", {"task": "look it up", "tools": ["browser", "browser_confirm", "search_web"]}) == "REPORT"
    child = seen["child"]
    assert [p for p, _ in child.extra_providers] == [other]
    assert child.toolset_provider("browser") is None
    assert child.allowed_tools == {"search_web"}  # browser names are not available to a subagent
    assert not any(t.get("type") == "browser_toolset_20260801" for t in child.get_tools_schema())
