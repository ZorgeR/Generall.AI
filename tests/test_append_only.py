"""Transcript mode keeps history append-only, so Sonnet 5.5's preserved thinking stays valid.

The API binds every thinking block to the system prompt, the tools and every message
before it. These tests drive real turns through ChainOfThoughtAgent with a fake Anthropic
client and check the property the API checks: each request starts with exactly the
messages of the request before it.
"""
import copy
import importlib
import os
from types import SimpleNamespace

import anthropic
import httpx
import pytest

import models
from agents import transcript as T

MEMORY_LINE = "• [2026-09-30] weather: the user lives in Lisbon"
HIT_LINE = "• [2026-09-29] Q: favourite colour? / A: green"


def text(t):
    return SimpleNamespace(type="text", text=t)


def thinking(sig):
    return SimpleNamespace(type="thinking", thinking=f"reasoning {sig}", signature=sig)


def tool_use(tool_id, name="get_time_in_timezone"):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input={"timezone": "Europe/Lisbon"})


def reply(*blocks, stop="end_turn", transformations=None):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop, input_transformations=transformations or [],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class FakeStream:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return self.response


class FakeClient:
    """Records every request (deep-copied when sent) and answers from a script."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = SimpleNamespace(stream=lambda **kw: self._stream(False, kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: self._stream(True, kw)))

    def _stream(self, beta, kwargs):
        self.requests.append({"beta": beta, **copy.deepcopy(kwargs)})
        return FakeStream(self.script.pop(0))


@pytest.fixture
def agents_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    module = importlib.import_module("agents.main")
    monkeypatch.setattr(module, "_thinking_binding_refused", False)
    return module


def make_agent(agents_main, monkeypatch, script, **overrides):
    from bot.settings import DEFAULT_SETTINGS

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings["transcript"]["max_tool_result_chars"] = 1000
    for category, values in overrides.items():
        settings[category].update(values)
    agent = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=settings)
    client = FakeClient(script)
    agent.agent.client = client

    async def complex_(question, history=None):
        return "complex"

    async def no_save(*args, **kwargs):
        return ""

    async def hits(question, k):
        return [HIT_LINE]

    async def run_tool(name, args):
        return "r" * 5000  # bigger than max_tool_result_chars

    monkeypatch.setattr(agent, "_classify_complexity", complex_)
    monkeypatch.setattr(agent, "_save_conversation", no_save)
    monkeypatch.setattr(agent, "_semantic_hits", hits)
    monkeypatch.setattr(agent, "_recent_summaries", lambda limit: [MEMORY_LINE])
    monkeypatch.setattr(agent.agent, "execute_tool", run_tool)
    return agent, client, settings


def assert_append_only(requests):
    for before, after in zip(requests, requests[1:]):
        assert after["system"] == before["system"]
        assert after["tools"] == before["tools"]
        assert after["messages"][: len(before["messages"])] == before["messages"]


def stored():
    return T.transcript_store.load("42", None)


async def test_turns_are_append_only_and_stored_as_sent(agents_main, monkeypatch):
    judged = iter([False, True, True])

    async def judge(question, answer):
        return next(judged)

    agent, client, _ = make_agent(agents_main, monkeypatch, [
        # turn 1: a tool call, a draft the judge rejects, the final answer
        reply(thinking("s1"), text("let me check"), tool_use("tu1"), stop="tool_use"),
        reply(thinking("s2"), text("draft")),
        reply(thinking("s3"), text("answer one")),
        # turn 2
        reply(thinking("s4"), text("answer two")),
    ], judge={"enabled": True, "max_iteration": 1})
    monkeypatch.setattr(agent.agent, "judge_response", judge)

    first, _ = await agent.generate_response("what time is it?")
    second, _ = await agent.generate_response("and tomorrow?")
    assert (first, second) == ("answer one", "answer two")

    requests = client.requests
    assert len(requests) == 4
    assert_append_only(requests)

    # the tool result is cut before the model sees it, not afterwards
    tool_results = [b for m in requests[1]["messages"] for b in m["content"] if b.get("type") == "tool_result"]
    assert tool_results[0]["content"].endswith("[truncated 4000 characters]")

    # the context block is part of the stored turn; turn 2 repeats no memory line from turn 1
    turn1_user = requests[0]["messages"][-1]["content"]
    assert T.is_context_block(turn1_user[0]) and MEMORY_LINE in turn1_user[0]["text"] and HIT_LINE in turn1_user[0]["text"]
    turn2_user = requests[3]["messages"][-1]["content"]
    assert T.is_context_block(turn2_user[0]) and MEMORY_LINE not in turn2_user[0]["text"] and HIT_LINE not in turn2_user[0]["text"]

    # the judge's rejection stayed in the history the later requests were built on
    assert any("automated judge" in T.message_text(m) for m in requests[3]["messages"])

    # what is stored is what was sent, plus the final answer
    transcript = stored()
    assert transcript.messages == requests[3]["messages"] + [{"role": "assistant", "content": [{"type": "text", "text": "answer two"}]}]
    assert transcript.fingerprint

    # the safety net rides along on the beta client
    for request in requests:
        assert request["beta"] and models.THINKING_BINDING_BETA in request["betas"]
        assert request["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}


async def test_changed_tools_strip_old_thinking_once(agents_main, monkeypatch):
    agent, _, _ = make_agent(agents_main, monkeypatch, [
        reply(thinking("s1"), text("let me check"), tool_use("tu1"), stop="tool_use"),
        reply(thinking("s2"), text("answer one")),
    ])
    await agent.generate_response("first")
    assert any(b.get("type") == "thinking" for m in stored().messages for b in m["content"])

    # the image preference is part of the image tool's description: changing it changes the tools
    agent2, client2, _ = make_agent(agents_main, monkeypatch, [
        reply(thinking("s3"), text("answer two")),
        reply(thinking("s4"), text("answer three")),
    ], image={"engine": "fast"})
    await agent2.generate_response("second")
    sent = client2.requests[0]["messages"]
    assert not any(b.get("type") == "thinking" for m in sent for b in m["content"])

    # from then on the new prefix is append-only again
    await agent2.generate_response("third")
    assert_append_only(client2.requests)
    assert client2.requests[1]["messages"][: len(sent)] == sent


async def test_refused_beta_falls_back_once_and_is_remembered(agents_main, monkeypatch):
    agent = agents_main.AgentAnthropic(user_id="7")
    calls = []

    def refuse(**kw):
        calls.append("beta")
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.BadRequestError(
            "thinking.block_binding: Extra inputs are not permitted",
            response=httpx.Response(400, request=request), body=None,
        )

    def plain(**kw):
        calls.append("plain")
        assert "betas" not in kw and "block_binding" not in kw["thinking"]
        return FakeStream(reply(text("ok")))

    agent.client = SimpleNamespace(messages=SimpleNamespace(stream=plain), beta=SimpleNamespace(messages=SimpleNamespace(stream=refuse)))
    kwargs = {"model": models.ANTHROPIC_MODEL, "messages": [], **models.anthropic_request_options(True)}
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    assert calls == ["beta", "plain", "plain"]


async def test_other_bad_requests_are_not_swallowed(agents_main, monkeypatch):
    agent = agents_main.AgentAnthropic(user_id="7")

    def mismatch(**kw):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.BadRequestError(
            "messages.3.content.0: thinking block is bound to a different conversation",
            response=httpx.Response(400, request=request), body=None,
        )

    agent.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=mismatch)))
    kwargs = {"model": models.ANTHROPIC_MODEL, "messages": [], **models.anthropic_request_options(True)}
    with pytest.raises(anthropic.BadRequestError):
        await agent._stream_message(kwargs)
    assert agents_main._thinking_binding_refused is False


def test_thinking_binding_options(monkeypatch):
    kwargs = {"model": "m", **models.anthropic_request_options(False)}
    bound = models.thinking_binding(kwargs)
    assert bound["betas"] == [models.THINKING_BINDING_BETA]
    assert bound["thinking"] == {"type": "adaptive", "display": "omitted", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    assert "block_binding" not in kwargs["thinking"]  # the plain request is left as it was
    assert models.thinking_binding({"model": "haiku"}) is None  # no adaptive thinking, nothing to bind
    monkeypatch.setattr(models, "THINKING_PREFIX_MISMATCH", "off")
    assert models.thinking_binding(kwargs) is None
    monkeypatch.setattr(models, "THINKING_PREFIX_MISMATCH", "error")
    assert models.thinking_binding(kwargs)["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "error"}
