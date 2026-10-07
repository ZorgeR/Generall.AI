"""Fast-model calls work whether or not the model thinks by default.

Haiku 4.5 never returned thinking unless asked; a newer fast model may start its
answer with a thinking block, so answers are read by block type, not position.
"""
import copy
import importlib
import os
from types import SimpleNamespace

import pytest

import models


def reply_with_thinking(answer):
    return SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking="", signature="sig"),
        SimpleNamespace(type="text", text=answer),
    ])


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    module = importlib.import_module("agents.main")
    from bot.settings import DEFAULT_SETTINGS

    calls = []

    async def create(**kw):
        calls.append(kw)
        return reply_with_thinking("simple")

    monkeypatch.setattr(module, "anthropic_client", SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setattr(module, "streaming_enabled", False)
    return module.ChainOfThoughtAgent(user_id="42", user_settings=copy.deepcopy(DEFAULT_SETTINGS)), calls


async def test_classifier_reads_past_a_thinking_block(agent):
    chat, calls = agent
    assert await chat._classify_complexity("hi there") == "simple"
    assert calls[0]["model"] == models.ANTHROPIC_MODEL_FAST
    assert calls[0]["max_tokens"] == models.ANTHROPIC_MAX_TOKENS_FAST_SHORT
    assert "thinking" not in calls[0] and "output_config" not in calls[0]  # fast-model calls stay plain


async def test_simple_answer_reads_past_a_thinking_block(agent):
    chat, _ = agent
    text, _ = await chat._simple_response([{"role": "user", "content": "hi"}], "system", "hi")
    assert text == "simple"
