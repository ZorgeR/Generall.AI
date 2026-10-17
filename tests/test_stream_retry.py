"""F2 error recovery inside the agent loop.

``AgentAnthropic._stream_message`` retries a transient API failure (rate limit, overloaded or
5xx provider, connection or timeout) with backoff and a recovery note, strips the stored
prefix's stale thinking blocks once on the 400 that names them (``prefix_len`` marks where the
turn begins; its own blocks stay, and flags ``history_edited``), and leaves every other error
alone. A retry after a mid-stream failure starts the streamed draft over. ``ChainOfThoughtAgent``
shows the notes as a status line (transcript and legacy mode) and drops the stored prefix's
thinking blocks, before the turn is appended, when a turn needed that.
"""
import asyncio
import copy
import importlib
import json
import os
from types import SimpleNamespace

import anthropic
import httpx
import pytest

import models
from agents import transcript as T

_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
MISMATCH = "messages.3.content.0: thinking block signature does not match the conversation prefix"


def api_error(cls, status: int, message: str = "api error", *, body=None, headers=None):
    return cls(message, response=httpx.Response(status, request=_REQUEST, headers=headers or {}), body=body)


def text(t):
    return SimpleNamespace(type="text", text=t)


def thinking(sig):
    return SimpleNamespace(type="thinking", thinking=f"reasoning {sig}", signature=sig)


def tool_use(tool_id, name="get_time_in_timezone"):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input={"timezone": "Europe/Lisbon"})


def with_tool(sig_a, sig_b, answer):
    """A turn whose stored history keeps a thinking block: the final answer is stored text-only,
    so only the assistant message that called a tool carries one."""
    return [reply(thinking(sig_a), text("let me check"), tool_use(f"tu_{sig_a}"), stop="tool_use"), reply(thinking(sig_b), text(answer))]


def reply(*blocks, stop="end_turn"):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop, input_transformations=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class FakeStream:
    """The SDK's stream manager: the request is sent (and fails) on ``__aenter__``."""

    def __init__(self, outcome):
        self.outcome = outcome

    async def __aenter__(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return self.outcome


class EventStream(FakeStream):
    """A stream that yields ``events`` (the SDK yields ``message_start`` first), fails with a
    connection error after ``fail_after`` of them when set, and ends in ``final``."""

    def __init__(self, events, final=None, fail_after=None):
        super().__init__(final)
        self.events, self.fail_after, self.i = list(events), fail_after, 0

    async def __anext__(self):
        if self.fail_after is not None and self.i >= self.fail_after:
            raise anthropic.APIConnectionError(request=_REQUEST)  # the connection drops mid-stream
        if self.i >= len(self.events):
            raise StopAsyncIteration
        event = self.events[self.i]
        self.i += 1
        return event


def ev(kind, **attrs):
    return SimpleNamespace(type=kind, **attrs)


class FakeClient:
    """Answers (or fails) from a script, recording every request as sent."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = SimpleNamespace(stream=lambda **kw: self._stream(False, kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: self._stream(True, kw)))

    def _stream(self, beta, kwargs):
        self.requests.append({"beta": beta, **copy.deepcopy(kwargs)})
        item = self.script.pop(0)
        return item if isinstance(item, FakeStream) else FakeStream(item)


@pytest.fixture
def agents_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    module = importlib.import_module("agents.main")
    monkeypatch.setattr(module, "_thinking_binding_refused", False)
    return module


@pytest.fixture
def no_sleep(monkeypatch):
    """Backoff without waiting: records the delays ``_stream_message`` asked for."""
    delays: list[float] = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


WITH_THINKING = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    {"role": "assistant", "content": [{"type": "thinking", "thinking": "r", "signature": "s1"}, {"type": "text", "text": "hello"}]},
    {"role": "user", "content": [{"type": "text", "text": "again"}]},
]


def make_agent(agents_main, script, messages=None):
    agent = agents_main.AgentAnthropic(user_id="7")
    agent.client = FakeClient(script)
    notes: list[str] = []

    async def on_recovery(note):
        notes.append(note)

    agent.on_recovery = on_recovery
    kwargs = {"model": models.ANTHROPIC_MODEL, "messages": copy.deepcopy(messages or []), **models.anthropic_request_options(True)}
    return agent, kwargs, notes


def has_thinking(request) -> bool:
    return any(b.get("type") in T.THINKING_TYPES for m in request["messages"] for b in m["content"])


def signatures(request) -> list[str]:
    """The thinking blocks of one request as sent, in order, by signature."""
    return [b["signature"] for m in request["messages"] for b in m["content"] if b.get("type") == "thinking"]


# ---- transient failures -----------------------------------------------------------------
async def test_rate_limit_is_retried_with_backoff_then_succeeds(agents_main, no_sleep):
    agent, kwargs, notes = make_agent(agents_main, [
        api_error(anthropic.RateLimitError, 429, "rate limited"),
        api_error(anthropic.RateLimitError, 429, "rate limited"),
        reply(text("ok")),
    ])
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    assert len(agent.client.requests) == 3 and all(r["beta"] for r in agent.client.requests)
    assert no_sleep == [2.0, 5.0]
    assert notes == ["Retrying after a rate limit (1/3)", "Retrying after a rate limit (2/3)"]
    assert agent.history_edited is False


async def test_retry_after_header_wins_up_to_the_cap(agents_main, no_sleep):
    agent, kwargs, _ = make_agent(agents_main, [
        api_error(anthropic.RateLimitError, 429, headers={"retry-after": "7"}),
        api_error(anthropic.RateLimitError, 429, headers={"retry-after": "120"}),
        api_error(anthropic.RateLimitError, 429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        reply(text("ok")),
    ])
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    assert no_sleep == [7.0, agents_main.MAX_RETRY_AFTER, 12.0]  # header, capped header, an HTTP date -> the backoff


async def test_gives_up_after_the_last_delay(agents_main, no_sleep):
    agent, kwargs, notes = make_agent(agents_main, [api_error(anthropic.OverloadedError, 529, "Overloaded")] * 4)
    with pytest.raises(anthropic.OverloadedError):
        await agent._stream_message(kwargs)
    assert len(agent.client.requests) == 4 and no_sleep == [2.0, 5.0, 12.0]
    assert notes == [f"Retrying after a provider error ({i}/3)" for i in (1, 2, 3)]


@pytest.mark.parametrize("error, reason", [
    (api_error(anthropic.InternalServerError, 500, "internal error"), "a provider error"),
    (api_error(anthropic.APIStatusError, 503, "service unavailable"), "a provider error"),
    (api_error(anthropic.APIStatusError, 529, "x", body={"error": {"type": "overloaded_error"}}), "a provider error"),
    (anthropic.APIConnectionError(request=_REQUEST), "a network error"),
    (anthropic.APITimeoutError(request=_REQUEST), "a network error"),
])
async def test_every_transient_kind_is_retried(agents_main, no_sleep, error, reason):
    agent, kwargs, notes = make_agent(agents_main, [error, reply(text("ok"))])
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    assert len(agent.client.requests) == 2 and no_sleep == [2.0]
    assert notes == [f"Retrying after {reason} (1/3)"]


@pytest.mark.parametrize("error", [
    api_error(anthropic.BadRequestError, 400, "messages.0.content: field required"),
    api_error(anthropic.BadRequestError, 400, "prompt is too long: 210000 tokens > 200000 maximum"),
    api_error(anthropic.AuthenticationError, 401, "invalid x-api-key"),
    api_error(anthropic.PermissionDeniedError, 403, "forbidden"),
    api_error(anthropic.NotFoundError, 404, "model not found"),
    api_error(anthropic.RequestTooLargeError, 413, "request too large"),
])
async def test_final_errors_are_not_retried(agents_main, no_sleep, error):
    agent, kwargs, notes = make_agent(agents_main, [error, reply(text("never"))], WITH_THINKING)
    with pytest.raises(type(error)):
        await agent._stream_message(kwargs)
    assert len(agent.client.requests) == 1 and no_sleep == [] and notes == []
    assert agent.history_edited is False


async def test_a_failing_recovery_hook_does_not_stop_the_retry(agents_main, no_sleep):
    agent, kwargs, _ = make_agent(agents_main, [api_error(anthropic.RateLimitError, 429), reply(text("ok"))])

    async def broken(note):
        raise RuntimeError("status edit failed")

    agent.on_recovery = broken
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    agent.on_recovery = None  # no hook at all is fine too
    agent.client = FakeClient([api_error(anthropic.RateLimitError, 429), reply(text("ok"))])
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"


def test_error_classification(agents_main):
    assert agents_main.is_transient_error(api_error(anthropic.RateLimitError, 429))
    assert agents_main.is_transient_error(api_error(anthropic.APIStatusError, 500))
    assert agents_main.is_transient_error(api_error(anthropic.BadRequestError, 400, "Overloaded"))  # message says so
    assert not agents_main.is_transient_error(api_error(anthropic.BadRequestError, 400, "invalid"))
    assert not agents_main.is_transient_error(api_error(anthropic.AuthenticationError, 401))
    assert not agents_main.is_transient_error(RuntimeError("boom"))
    assert agents_main.retry_delay(RuntimeError("no response"), 5.0) == 5.0
    assert agents_main.retry_delay(api_error(anthropic.RateLimitError, 429, headers={"retry-after": "-3"}), 5.0) == 0.0


# ---- stale thinking blocks ---------------------------------------------------------------
async def test_stale_thinking_is_stripped_and_retried_once(agents_main, no_sleep):
    agent, kwargs, notes = make_agent(agents_main, [api_error(anthropic.BadRequestError, 400, MISMATCH), reply(text("ok"))], WITH_THINKING)
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    first, second = agent.client.requests
    assert has_thinking(first) and not has_thinking(second)
    assert [m["content"][-1]["text"] for m in second["messages"]] == ["hi", "hello", "again"]  # only the thinking went
    assert agent.history_edited is True
    assert notes == ["Retrying without the stored thinking blocks"] and no_sleep == []
    assert kwargs["messages"] == WITH_THINKING  # the caller's request was not edited in place


@pytest.mark.parametrize("message", [
    "Invalid signature in thinking block",
    "thinking blocks cannot be modified: prefix mismatch",
    "block_binding: prefix_mismatch at messages.1",
    "redacted_thinking block is bound to a different conversation",
])
def test_mismatch_wordings(agents_main, message):
    assert agents_main.is_thinking_mismatch(api_error(anthropic.BadRequestError, 400, message))


def test_mismatch_is_not_the_beta_refusal_or_another_400(agents_main):
    assert not agents_main.is_thinking_mismatch(api_error(anthropic.BadRequestError, 400, "thinking.block_binding: Extra inputs are not permitted"))
    assert not agents_main.is_thinking_mismatch(api_error(anthropic.BadRequestError, 400, "thinking.budget_tokens: must be >= 1024"))
    assert not agents_main.is_thinking_mismatch(api_error(anthropic.BadRequestError, 400, "prompt is too long"))
    assert not agents_main.is_thinking_mismatch(api_error(anthropic.RateLimitError, 429, "thinking block signature"))


async def test_a_second_mismatch_propagates(agents_main, no_sleep):
    agent, kwargs, _ = make_agent(agents_main, [api_error(anthropic.BadRequestError, 400, MISMATCH)] * 2 + [reply(text("never"))], WITH_THINKING)
    with pytest.raises(anthropic.BadRequestError):
        await agent._stream_message(kwargs)
    assert len(agent.client.requests) == 2 and agent.history_edited is True


async def test_a_mismatch_with_nothing_to_strip_propagates(agents_main, no_sleep):
    plain = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    agent, kwargs, notes = make_agent(agents_main, [api_error(anthropic.BadRequestError, 400, MISMATCH), reply(text("never"))], plain)
    with pytest.raises(anthropic.BadRequestError):
        await agent._stream_message(kwargs)
    assert len(agent.client.requests) == 1 and notes == [] and agent.history_edited is False


async def test_only_the_stored_prefix_loses_its_thinking(agents_main, no_sleep):
    """``prefix_len`` (set by generate_response) marks where this turn begins: a mismatch strips
    the blocks before it and keeps the turn's own, which the API wants on the last tool call."""
    current = [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "r", "signature": "new"}, {"type": "tool_use", "id": "t1", "name": "list_timezones", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "UTC"}]},
    ]
    agent, kwargs, notes = make_agent(agents_main, [api_error(anthropic.BadRequestError, 400, MISMATCH), reply(text("ok"))], WITH_THINKING + current)
    agent.prefix_len = len(WITH_THINKING)
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    first, second = agent.client.requests
    assert signatures(first) == ["s1", "new"] and signatures(second) == ["new"]
    assert second["messages"][len(WITH_THINKING):] == current  # the turn's messages went as they were
    assert agent.history_edited is True and notes == ["Retrying without the stored thinking blocks"]

    # the rest of the turn: the prefix goes without its blocks right away, the turn keeps its own
    agent.client = FakeClient([reply(text("ok"))])
    await agent._stream_message(kwargs)
    assert signatures(agent.client.requests[0]) == ["new"]

    # a prefix with nothing left to strip makes a mismatch final, whatever the turn carries
    agent.history_edited = False
    agent.client = FakeClient([api_error(anthropic.BadRequestError, 400, MISMATCH), reply(text("never"))])
    bare = {**kwargs, "messages": [WITH_THINKING[0], WITH_THINKING[2]] + current}
    agent.prefix_len = 2
    with pytest.raises(anthropic.BadRequestError):
        await agent._stream_message(bare)
    assert len(agent.client.requests) == 1 and agent.history_edited is False


async def test_once_edited_the_rest_of_the_turn_sends_no_thinking(agents_main, no_sleep):
    agent, kwargs, notes = make_agent(agents_main, [reply(text("ok"))], WITH_THINKING)
    agent.history_edited = True  # an earlier call of this turn stripped
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    assert not has_thinking(agent.client.requests[0]) and notes == []


async def test_a_mid_stream_failure_does_not_double_the_draft(agents_main, monkeypatch, no_sleep):
    """The connection drops after "Hello wor" was streamed and the retry streams the whole answer
    again: the draft shown to the user (kept as the partial answer on Stop) starts over with the
    retry instead of carrying the first attempt's text in front of it."""
    from bot.settings import DEFAULT_SETTINGS

    monkeypatch.setattr(agents_main, "streaming_enabled", True)
    answer = "Hello world, here is the answer."
    agent, _, notes = make_agent(agents_main, [
        EventStream([ev("message_start"), ev("thinking", thinking="let me"), ev("text", text="Hello wor")], fail_after=3),
        EventStream([ev("message_start"), ev("thinking", thinking="let me see"), ev("text", text=answer)], final=reply(text(answer))),
    ])
    shown: list[tuple[bool, str]] = []

    async def on_text_chunk(draft, is_thinking=False):
        shown.append((is_thinking, draft))

    response, messages = await agent.generate_response(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}], system_role="sys", question="hi",
        user_settings=copy.deepcopy(DEFAULT_SETTINGS), on_text_chunk=on_text_chunk,
    )
    assert response == answer and notes == ["Retrying after a network error (1/3)"] and no_sleep == [2.0]
    assert [d for thinking_, d in shown if not thinking_] == ["Hello wor", answer]
    assert [d for thinking_, d in shown if thinking_] == ["let me", "let me see"]
    assert agent.prefix_len == 1 and messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": answer}]}


async def test_the_beta_refusal_is_still_handled_inside_one_attempt(agents_main, no_sleep):
    agent, kwargs, notes = make_agent(agents_main, [], WITH_THINKING)
    calls = []

    def refuse(**kw):
        calls.append("beta")
        raise api_error(anthropic.BadRequestError, 400, "thinking.block_binding: Extra inputs are not permitted")

    def plain(**kw):
        calls.append("plain")
        assert "betas" not in kw and has_thinking({"messages": kw["messages"]})
        return FakeStream(reply(text("ok")))

    agent.client = SimpleNamespace(messages=SimpleNamespace(stream=plain), beta=SimpleNamespace(messages=SimpleNamespace(stream=refuse)))
    assert models.anthropic_text(await agent._stream_message(kwargs)) == "ok"
    assert calls == ["beta", "plain"] and notes == [] and agent.history_edited is False
    assert agents_main._thinking_binding_refused is True


# ---- ChainOfThoughtAgent: the status line and the stored transcript ----------------------
def make_chain(agents_main, monkeypatch, script):
    from bot.settings import DEFAULT_SETTINGS

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    agent = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=settings)
    client = FakeClient(script)
    agent.agent.client = client

    async def complex_(question, history=None):
        return "complex"

    async def no_save(*args, **kwargs):
        return ""

    async def context(question, update_status=None, messages=None):
        return "<context>now</context>", "<context>now</context>"

    async def run_tool(name, args):
        return "12:00"

    monkeypatch.setattr(agent, "_classify_complexity", complex_)
    monkeypatch.setattr(agent, "_save_conversation", no_save)
    monkeypatch.setattr(agent, "_request_context", context)
    monkeypatch.setattr(agent.agent, "execute_tool", run_tool)
    return agent, client


def stored():
    return T.transcript_store.load("42", None)


def stored_has_thinking() -> bool:
    return any(b.get("type") in T.THINKING_TYPES for m in stored().messages for b in m["content"])


def stored_signatures() -> list[str]:
    return signatures({"messages": stored().messages})


async def test_recovery_notes_become_a_status_line(agents_main, monkeypatch, no_sleep):
    agent, client = make_chain(agents_main, monkeypatch, [api_error(anthropic.RateLimitError, 429), *with_tool("s1", "s2", "answer")])
    updates = []

    async def update_status(step, details, iteration, critique):
        updates.append((step, details))

    response, _ = await agent.generate_response("hi", update_status)
    assert response == "answer" and len(client.requests) == 3
    assert ("recovering", "Retrying after a rate limit (1/3)") in updates
    assert stored_has_thinking() and agent.agent.history_edited is False  # a plain retry edits nothing


async def test_recovery_notes_reach_the_status_line_in_legacy_mode_too(agents_main, monkeypatch):
    """The hook is wired for the turn, whichever memory mode runs it."""
    agent, _ = make_chain(agents_main, monkeypatch, [])
    agent.user_settings["transcript"]["enabled"] = False
    agent.user_settings["semantic_search"]["enabled"] = False  # no embeddings call
    inner = agent.agent

    async def retried(messages, **kw):
        await inner.on_recovery("Retrying after a provider error (1/3)")
        return "answer", messages + [{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]

    monkeypatch.setattr(inner, "generate_response", retried)
    updates = []

    async def update_status(step, details, iteration, critique):
        updates.append((step, details))

    response, _ = await agent.generate_response("hi", update_status)
    assert response == "answer" and ("recovering", "Retrying after a provider error (1/3)") in updates
    assert not T.transcript_store.exists("42", None)  # legacy mode: no transcript was touched


async def test_a_stripped_request_strips_the_stored_transcript_too(agents_main, monkeypatch, no_sleep):
    agent, _ = make_chain(agents_main, monkeypatch, with_tool("s1", "s2", "one"))
    await agent.generate_response("first")
    assert stored_has_thinking()

    # the same prefix (settings unchanged), so the stored blocks survive the fingerprint check;
    # the API refuses them anyway: the request is retried without the prefix's blocks, and so
    # is the store, while the turn's own tool-calling message keeps its thinking (the API wants
    # it there when the tool results go back)
    again, client = make_chain(agents_main, monkeypatch, [api_error(anthropic.BadRequestError, 400, MISMATCH), *with_tool("s3", "s4", "two")])
    response, _ = await again.generate_response("second")
    assert response == "two"
    assert [signatures(r) for r in client.requests] == [["s1"], [], ["s3"]]  # refused, retried bare, tool results with this turn's block
    assert stored_signatures() == ["s3"] and again.agent.history_edited is False  # the stale s1 is gone, the final answer is text-only
    assert [m["role"] for m in stored().messages] == ["user", "assistant", "user", "assistant"] * 2  # question, tool call, result, answer
    assert T.message_text(stored().messages[-1]) == "two"
    assert stored().messages[: len(client.requests[-1]["messages"])] == client.requests[-1]["messages"]  # stored == sent

    # from then on the history is append-only again
    third, client3 = make_chain(agents_main, monkeypatch, with_tool("s5", "s6", "three"))
    await third.generate_response("third")
    sent_last = client.requests[-1]["messages"]
    assert client3.requests[0]["messages"][: len(sent_last)] == sent_last


async def test_history_edited_alone_strips_the_store(agents_main, monkeypatch):
    """The flag is what counts: an agent that reports an edited history gets its stored thinking dropped."""
    agent, _ = make_chain(agents_main, monkeypatch, with_tool("s1", "s2", "one"))
    await agent.generate_response("first")
    assert stored_has_thinking()

    again, _ = make_chain(agents_main, monkeypatch, [])
    inner = again.agent

    async def edited(messages, **kw):
        inner.history_edited = True
        return "two", messages + [{"role": "assistant", "content": [{"type": "text", "text": "two"}]}]

    monkeypatch.setattr(inner, "generate_response", edited)
    assert (await again.generate_response("second"))[0] == "two"
    assert not stored_has_thinking() and inner.history_edited is False

    untouched, _ = make_chain(agents_main, monkeypatch, with_tool("s3", "s4", "three"))
    await untouched.generate_response("third")
    assert stored_has_thinking()  # a normal turn keeps its blocks


class BindingClient(FakeClient):
    """An enforced account: each thinking block it returns is bound to the request's tool set
    (with the model, system prompt and earlier messages, what preserved thinking compares; the
    tool_choice is not part of it), and a request replaying a block under another tool set gets
    the mismatch 400."""

    def __init__(self, script, bound: dict):
        super().__init__(script)
        self.bound = bound  # signature -> the tool set it was minted under, shared across turns

    def _stream(self, beta, kwargs):
        tools = json.dumps(kwargs.get("tools"), sort_keys=True, default=str)
        if any(self.bound.get(sig, tools) != tools for sig in signatures(kwargs)):
            self.requests.append({"beta": beta, **copy.deepcopy(kwargs)})
            return FakeStream(api_error(anthropic.BadRequestError, 400, MISMATCH))
        stream = super()._stream(beta, kwargs)
        if not isinstance(stream.outcome, BaseException):
            self.bound.update({b.signature: tools for b in stream.outcome.content if b.type == "thinking"})
        return stream


async def test_the_forced_final_call_keeps_the_loops_tools(agents_main, monkeypatch, no_sleep):
    """The budget is spent and the SYSTEM NOTICE call compiles the answer: it sends the loop's
    tool definitions with tool_choice none, so every thinking block of the request (the stored
    ones and this turn's) stays valid, the compiled answer arrives and the store keeps its
    thinking. Without the tools the call was refused twice, the user got the unfinished
    "let me check" and the stored blocks were stripped."""
    bound: dict[str, str] = {}
    first, _ = make_chain(agents_main, monkeypatch, [])
    first.agent.client = BindingClient(with_tool("s1", "s2", "one"), bound)
    assert (await first.generate_response("first"))[0] == "one"

    second, _ = make_chain(agents_main, monkeypatch, [])
    second.user_settings["tools"]["max_iteration"] = 1
    client = BindingClient([reply(thinking("s3"), text("let me check"), tool_use("tu_s3"), stop="tool_use"), reply(thinking("s4"), text("compiled"))], bound)
    second.agent.client = client
    response, _ = await second.generate_response("second")
    assert response == "compiled" and second.agent.limit_reached is True
    loop, final = client.requests  # nothing refused, nothing retried
    assert final["tools"] == loop["tools"] and final["tools"]
    assert loop["tool_choice"] == {"type": "auto"} and final["tool_choice"] == {"type": "none"}
    assert signatures(final) == ["s1", "s3"] and second.agent.history_edited is False
    assert stored_signatures() == ["s1", "s3"]  # the compiled answer is stored text-only
    assert T.message_text(stored().messages[-1]) == "compiled"
