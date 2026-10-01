"""Veo videos add their estimated cost (per generated second) to the turn's trace."""
import importlib
import os
from types import SimpleNamespace

import pytest

import models
from agents.trace import ToolTrace


@pytest.fixture
def video_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.environ.setdefault("GOOGLE_API_KEY", "x")  # the module builds its genai client at import
    module = importlib.import_module("agents.video_tools")
    done = SimpleNamespace(done=True, result=SimpleNamespace(generated_videos=[SimpleNamespace(video="v")]))
    calls = []

    def generate_videos(**kw):
        calls.append(kw)
        return done

    monkeypatch.setattr(module, "genai_client", SimpleNamespace(models=SimpleNamespace(generate_videos=generate_videos)))
    tools = module.VideoTools("7", sender=None)
    tools.trace = ToolTrace()
    return module, tools, calls


async def test_new_video_is_priced_at_the_default_length(video_tools):
    module, tools, calls = video_tools
    video = await tools._generate_and_wait(prompt="a cat", config=module.types.GenerateVideosConfig(resolution="720p"))
    assert video.video == "v" and calls[0]["model"] == models.VEO_MODEL
    assert abs(tools.trace.model_cost(models.VEO_MODEL) - models.VEO_DEFAULT_SECONDS * 0.40) < 1e-9


async def test_extension_and_explicit_duration(video_tools):
    module, tools, _ = video_tools
    await tools._generate_and_wait(video="source", prompt="more", config=module.types.GenerateVideosConfig(number_of_videos=1))
    await tools._generate_and_wait(prompt="short", config=module.types.GenerateVideosConfig(duration_seconds=4))
    bucket = tools.trace.usage_by_model[models.VEO_MODEL]
    assert bucket["api_calls"] == 2
    assert abs(bucket["cost_usd"] - (models.VEO_EXTENSION_SECONDS + 4) * 0.40) < 1e-9


async def test_no_trace_records_nothing(video_tools):
    module, tools, _ = video_tools
    tools.trace = None
    assert (await tools._generate_and_wait(prompt="a cat")).video == "v"
