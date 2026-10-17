"""F3: the mode setting (auto / speed / intelligence) and streaming on by default.

``mode.type`` decides, per user, whether the fast-model classifier may route a message to
the quick answer (auto), whether the whole tool loop runs on the fast model with light
thinking (speed) or on the main model with deep thinking whatever the thinking toggle says
(intelligence). Streaming drafts are on unless STREAMING_ENABLED=false, in both places that
read the variable (bot/config.py and agents/main.py).
"""
import copy
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import models
from bot.settings import DEFAULT_SETTINGS

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
DUMMY_KEYS = {name: "x" for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY")}


# ---- streaming default ---------------------------------------------------------
def test_config_streams_unless_told_otherwise(monkeypatch):
    from bot.config import load_config

    monkeypatch.delenv("STREAMING_ENABLED", raising=False)
    assert load_config().streaming_enabled is True
    monkeypatch.setenv("STREAMING_ENABLED", "false")
    assert load_config().streaming_enabled is False
    monkeypatch.setenv("STREAMING_ENABLED", "true")
    assert load_config().streaming_enabled is True


@pytest.mark.parametrize("env_value, expected", [(None, "True True"), ("false", "False False")])
def test_both_readers_of_streaming_enabled_agree(tmp_path, env_value, expected):
    """agents/main.py reads STREAMING_ENABLED at import, independently of bot/config.py: the
    two defaults must agree, so both are read in a fresh interpreter."""
    code = (
        "import bot.config, agents.main\n"
        "print(bot.config.config.streaming_enabled, agents.main.streaming_enabled)"
    )
    env = {**os.environ, **DUMMY_KEYS, "PYTHONPATH": str(APP)}
    env.pop("STREAMING_ENABLED", None)
    if env_value is not None:
        env["STREAMING_ENABLED"] = env_value
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected


def test_env_example_documents_streaming_on():
    lines = [line.strip() for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines()]
    assert any(line.startswith("STREAMING_ENABLED=true") for line in lines)


# ---- model and thinking per mode ---------------------------------------------------
@pytest.fixture
def agents_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, value in DUMMY_KEYS.items():
        os.environ.setdefault(name, value)
    return importlib.import_module("agents.main")


def settings(mode=None, thinking=True):
    s = copy.deepcopy(DEFAULT_SETTINGS)
    if mode is not None:
        s["mode"]["type"] = mode
    s["thinking"]["enabled"] = thinking
    return s


@pytest.mark.parametrize("mode, thinking, expected_mode, expected_model, expected_thinking", [
    ("auto", True, "auto", models.ANTHROPIC_MODEL, True),
    ("auto", False, "auto", models.ANTHROPIC_MODEL, False),
    ("speed", True, "speed", models.ANTHROPIC_MODEL_FAST, False),  # light, whatever the toggle says
    ("intelligence", False, "intelligence", models.ANTHROPIC_MODEL, True),  # deep, whatever the toggle says
    ("warp", False, "auto", models.ANTHROPIC_MODEL, False),  # unknown values behave as auto
    ("Speed", True, "speed", models.ANTHROPIC_MODEL_FAST, False),
])
def test_mode_selects_the_model_and_thinking(agents_main, mode, thinking, expected_mode, expected_model, expected_thinking):
    agent = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=settings(mode, thinking))
    assert agent.mode == expected_mode
    assert agent.agent.model == expected_model
    assert agent.agent.thinking is expected_thinking


def test_request_options_follow_the_mode(agents_main):
    """What the loop sends: no effort/thinking parameters for the fast model (CLAUDE.md), the
    light ceiling with thinking off, deep thinking in intelligence mode."""
    speed = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=settings("speed")).agent
    assert models.request_options_for(speed.model, speed.thinking) == {}
    assert models.max_tokens_for(speed.model, speed.thinking) == models.ANTHROPIC_MAX_TOKENS_FAST

    deep = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=settings("intelligence", thinking=False)).agent
    options = models.request_options_for(deep.model, deep.thinking)
    assert options["output_config"]["effort"] == models.ANTHROPIC_EFFORT
    assert options["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert models.max_tokens_for(deep.model, deep.thinking) == models.ANTHROPIC_MAX_TOKENS

    light = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=settings("auto", thinking=False)).agent
    options = models.request_options_for(light.model, light.thinking)
    assert options["output_config"]["effort"] == models.ANTHROPIC_EFFORT_LIGHT
    assert options["thinking"]["display"] == "omitted"
    assert models.max_tokens_for(light.model, light.thinking) == models.ANTHROPIC_MAX_TOKENS_LIGHT


def test_a_missing_mode_category_is_auto(agents_main):
    s = settings()
    del s["mode"]
    agent = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=s)
    assert agent.mode == "auto" and agent.agent.model == models.ANTHROPIC_MODEL
    assert agents_main.ChainOfThoughtAgent._mode_of(None) == "auto"
    assert agents_main.ChainOfThoughtAgent._mode_of({"mode": {"type": None}}) == "auto"
    assert agents_main.ChainOfThoughtAgent._mode_of({"mode": "speed"}) == "auto"  # a hand-edited non-dict


# ---- the classifier gate ------------------------------------------------------------
def bare_agent(agents_main, mode=None, *, classify):
    """A ChainOfThoughtAgent without __init__ (no clients, no FAISS): only the mode gate."""
    cot = agents_main.ChainOfThoughtAgent.__new__(agents_main.ChainOfThoughtAgent)
    if mode is not None:
        cot.user_settings = {"mode": {"type": mode}}
    cot._classify_complexity = classify
    return cot


async def never(question, history):
    raise AssertionError("the classifier must not run in this mode")


async def simple(question, history):
    return "simple"


@pytest.mark.parametrize("mode", ["speed", "intelligence"])
async def test_speed_and_intelligence_skip_the_classifier(agents_main, mode):
    cot = bare_agent(agents_main, mode, classify=never)
    assert await cot._decide_complexity("hi", []) == "complex"
    assert await cot._decide_complexity("hi", [], has_media=True) == "complex"


async def test_auto_asks_the_classifier_unless_there_is_media(agents_main):
    cot = bare_agent(agents_main, "auto", classify=simple)
    assert await cot._decide_complexity("hi", []) == "simple"
    cot = bare_agent(agents_main, "auto", classify=never)
    assert await cot._decide_complexity("hi", [], has_media=True) == "complex"


async def test_unknown_mode_and_no_settings_behave_as_auto(agents_main):
    assert await bare_agent(agents_main, "warp", classify=simple)._decide_complexity("hi", []) == "simple"
    assert await bare_agent(agents_main, None, classify=simple)._decide_complexity("hi", []) == "simple"


# ---- a whole turn per mode (transcript mode, fake client) -------------------------------
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
    """The loop's ``messages.stream`` and the judge's ``messages.create``, scripted in call order."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []  # the loop's stream requests
        self.created = []  # the judge's create requests
        self.messages = SimpleNamespace(stream=lambda **kw: self._stream(kw), create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: self._stream(kw)))

    def _stream(self, kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return FakeStream(self.script.pop(0))

    async def _create(self, **kwargs):
        self.created.append(copy.deepcopy(kwargs))
        return self.script.pop(0)


def answer(text):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn", input_transformations=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


async def run_turn(agents_main, monkeypatch, user_settings, script=None):
    """One transcript-mode turn on a fake client; returns the loop's single request."""
    client = await run_turn_client(agents_main, monkeypatch, user_settings, script)
    (request,) = client.requests
    return request


async def run_turn_client(agents_main, monkeypatch, user_settings, script=None, classify=never):
    """Runs the turn and returns the fake client (loop requests in ``requests``, judge calls in
    ``created``). ``classify`` stands in for the Haiku classifier; the default fails if it runs."""
    agent = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=user_settings)
    client = FakeClient(script or [answer("done")])
    agent.agent.client = client

    async def no_save(*args, **kwargs):
        return ""

    async def no_hits(question, k):
        return []

    monkeypatch.setattr(agents_main, "streaming_enabled", False)
    monkeypatch.setattr(agents_main, "_thinking_binding_refused", False)
    monkeypatch.setattr(agent, "_classify_complexity", classify)
    monkeypatch.setattr(agent, "_save_conversation", no_save)
    monkeypatch.setattr(agent, "_semantic_hits", no_hits)
    monkeypatch.setattr(agent, "_recent_summaries", lambda limit: [])
    response, _ = await agent.generate_response("what time is it?")
    assert response == "done"
    return client


async def test_speed_mode_runs_the_tool_loop_on_the_fast_model(agents_main, monkeypatch):
    request = await run_turn(agents_main, monkeypatch, settings("speed"))
    assert request["model"] == models.ANTHROPIC_MODEL_FAST
    assert request["tools"], "speed mode keeps the tools (it is not the quick text-only answer)"
    assert "thinking" not in request and "output_config" not in request
    assert request["max_tokens"] == models.ANTHROPIC_MAX_TOKENS_FAST


async def test_intelligence_mode_thinks_deeply_with_the_toggle_off(agents_main, monkeypatch):
    request = await run_turn(agents_main, monkeypatch, settings("intelligence", thinking=False))
    assert request["model"] == models.ANTHROPIC_MODEL
    assert request["thinking"]["display"] == "summarized"
    assert request["output_config"]["effort"] == models.ANTHROPIC_EFFORT
    assert request["max_tokens"] == models.ANTHROPIC_MAX_TOKENS


async def test_speed_mode_judge_asks_the_fast_model_plainly(agents_main, monkeypatch):
    """The completeness check runs on the loop's model. With the fast model that call must
    carry no effort/thinking parameters either, or it fails and judge_response silently
    accepts every answer (its exception path returns True)."""
    user_settings = settings("speed")
    user_settings["judge"]["enabled"] = True
    client = await run_turn_client(agents_main, monkeypatch, user_settings, script=[answer("done"), answer("Yes")])
    (judge_request,) = client.created
    assert judge_request["model"] == models.ANTHROPIC_MODEL_FAST
    assert "thinking" not in judge_request and "output_config" not in judge_request
    assert judge_request["max_tokens"] >= 1 and "Response: done" in judge_request["messages"][0]["content"][0]["text"]
    assert not client.script, "the judge's verdict was consumed: the check ran"


async def complex_(question, history):
    return "complex"


async def test_auto_mode_judge_keeps_light_thinking_on_the_main_model(agents_main, monkeypatch):
    user_settings = settings("auto")
    user_settings["judge"]["enabled"] = True
    client = await run_turn_client(
        agents_main, monkeypatch, user_settings, script=[answer("done"), answer("Yes")], classify=complex_,
    )
    (judge_request,) = client.created
    assert judge_request["model"] == models.ANTHROPIC_MODEL
    assert judge_request["output_config"]["effort"] == models.ANTHROPIC_EFFORT_LIGHT
    assert judge_request["thinking"]["display"] == "omitted"
