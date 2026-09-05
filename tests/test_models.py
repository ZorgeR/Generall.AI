"""app/models.py: defaults, environment overrides and the request-option helpers."""
import importlib

import pytest

import models

OVERRIDE_VARS = [
    "ANTHROPIC_MODEL", "ANTHROPIC_EFFORT", "ANTHROPIC_EFFORT_LIGHT", "ANTHROPIC_MAX_TOKENS", "ANTHROPIC_MAX_TOKENS_LIGHT", "ANTHROPIC_MODEL_FAST",
    "OPENAI_MODEL", "VIDEO_FRAMES_MODEL", "OPENAI_REASONING_EFFORT",
    "WHISPER_MODEL", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION",
    "GEMINI_IMAGE_MODEL_FLASH", "GEMINI_IMAGE_MODEL_PRO", "GPT_IMAGE_MODEL", "DALLE_MODEL",
    "VEO_MODEL", "PERPLEXITY_MODEL", "TTS_MODEL",
]


@pytest.fixture
def clean_models(monkeypatch):
    """`models` reloaded with no overrides in the environment; reloaded again afterwards."""
    for name in OVERRIDE_VARS:
        monkeypatch.delenv(name, raising=False)
    yield importlib.reload(models)
    monkeypatch.undo()          # restore the real environment first ...
    importlib.reload(models)    # ... then put the module back the way other tests expect it


def test_defaults(clean_models):
    m = clean_models
    assert m.ANTHROPIC_MODEL == "claude-sonnet-5"
    assert m.ANTHROPIC_EFFORT == "high"
    assert m.ANTHROPIC_EFFORT_LIGHT == "low"
    assert (m.ANTHROPIC_MAX_TOKENS, m.ANTHROPIC_MAX_TOKENS_LIGHT) == (64000, 16000)
    assert m.ANTHROPIC_MODEL_FAST == "claude-haiku-4-5"
    assert m.OPENAI_MODEL == "gpt-5.6-terra"
    assert m.VIDEO_FRAMES_MODEL == "gpt-5.6-luna"
    assert m.OPENAI_REASONING_EFFORT == "high"
    assert m.OPENAI_REASONING_MODELS == {"gpt-5.6-terra", "gpt-5.6-luna"}
    assert m.WHISPER_MODEL == "whisper-1"
    assert m.EMBEDDING_MODEL == "text-embedding-ada-002"
    assert m.EMBEDDING_DIMENSION == 1536
    assert m.GEMINI_IMAGE_MODEL_FLASH == "gemini-3.1-flash-image-preview"
    assert m.GEMINI_IMAGE_MODEL_PRO == "gemini-3-pro-image-preview"
    assert m.GPT_IMAGE_MODEL == "gpt-image-2-2026-04-21"
    assert m.DALLE_MODEL == "dall-e-3"
    assert m.VEO_MODEL == "veo-3.1-generate-preview"
    assert m.PERPLEXITY_MODEL == "sonar"
    assert m.PERPLEXITY_MODELS == ("sonar-reasoning-pro", "sonar-pro", "sonar")
    assert m.PERPLEXITY_MODEL in m.PERPLEXITY_MODELS
    assert m.TTS_MODEL == "eleven_multilingual_v2"


def test_env_override_is_honoured(clean_models, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "xhigh")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-nova")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "3072")
    monkeypatch.setenv("VEO_MODEL", "   ")  # blank means "use the default"
    m = importlib.reload(clean_models)

    assert m.ANTHROPIC_MODEL == "claude-opus-5"
    assert m.anthropic_request_options() == {"output_config": {"effort": "xhigh"}}
    assert m.OPENAI_MODEL == "gpt-5.6-nova"
    assert m.openai_reasoning_options("gpt-5.6-nova") == {"reasoning_effort": "low"}
    assert m.openai_reasoning_options("gpt-5.6-terra") == {}  # no longer a configured model
    assert m.EMBEDDING_DIMENSION == 3072
    assert m.VEO_MODEL == "veo-3.1-generate-preview"
    assert m.ANTHROPIC_MODEL_FAST == "claude-haiku-4-5"  # untouched


def test_anthropic_request_options(clean_models):
    m = clean_models
    high = {"output_config": {"effort": "high"}}
    assert m.anthropic_request_options() == high
    assert m.anthropic_request_options(thinking=None) == high
    # thinking on: full effort, summarized display for the streaming block / reasoning file
    assert m.anthropic_request_options(thinking=True) == {
        **high, "thinking": {"type": "adaptive", "display": "summarized"},
    }
    # thinking off: still adaptive (never "disabled"), lighter effort, nothing displayed
    assert m.anthropic_request_options(thinking=False) == {
        "output_config": {"effort": "low"}, "thinking": {"type": "adaptive", "display": "omitted"},
    }
    assert m.anthropic_request_options(thinking=False, effort="medium")["output_config"] == {"effort": "medium"}
    # Sonnet 5 rejects the old fixed budget and the effort must not be top-level.
    for opts in (m.anthropic_request_options(True), m.anthropic_request_options(False)):
        assert "budget_tokens" not in opts["thinking"] and opts["thinking"]["type"] == "adaptive"
        assert "effort" not in opts
    # helpers hand out fresh dicts, so a caller mutating one cannot leak into the next call
    m.anthropic_request_options()["output_config"]["effort"] = "low"
    assert m.anthropic_request_options() == high


def test_anthropic_max_tokens_leave_room_for_thinking(clean_models):
    m = clean_models
    assert m.anthropic_max_tokens(True) == 64000
    assert m.anthropic_max_tokens(None) == 64000
    assert m.anthropic_max_tokens(False) == 16000
    assert m.ANTHROPIC_MAX_TOKENS_LIGHT <= 21000  # the SDK's non-streaming ceiling (media calls do not stream)


def test_anthropic_text_skips_thinking_blocks(clean_models):
    from types import SimpleNamespace as NS

    message = NS(content=[NS(type="thinking", thinking="hmm"), NS(type="text", text="Yes"), NS(type="text", text=".")])
    assert clean_models.anthropic_text(message) == "Yes."
    assert clean_models.anthropic_text(NS(content=[])) == ""


def test_openai_reasoning_options(clean_models):
    m = clean_models
    assert m.openai_reasoning_options(m.OPENAI_MODEL) == {"reasoning_effort": "high"}
    assert m.openai_reasoning_options(m.VIDEO_FRAMES_MODEL) == {"reasoning_effort": "high"}
    assert m.openai_reasoning_options(m.WHISPER_MODEL) == {}
    assert m.openai_reasoning_options(m.EMBEDDING_MODEL) == {}
    assert m.openai_reasoning_options(m.GPT_IMAGE_MODEL) == {}
    assert m.openai_reasoning_options(m.DALLE_MODEL) == {}


def test_estimate_cost(clean_models):
    m = clean_models
    # 1M input at $2 + 1M cache reads at 10% + 1M cache writes at 125% + 1M output at $10
    assert m.estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000, 1_000_000, 1_000_000) == 2.0 + 0.2 + 2.5 + 10.0
    assert m.estimate_cost("claude-haiku-4-5", 1000, 0) == 0.001
    assert m.estimate_cost("some-unknown-model", 1000, 1000) is None
