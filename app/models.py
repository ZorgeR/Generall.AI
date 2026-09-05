"""Single source of truth for every model / provider setting.

Every consumer (``bot/*`` and ``agents/*``) imports its model name from here
instead of defining its own constant. Each value is read from the environment
once, at import time, so it can be overridden through ``.env`` (``load_dotenv()``
runs in ``main_bot.py`` before anything else is imported); the default is the
model the code was written and tested against.

The request options that belong to a model (Anthropic ``effort``, OpenAI
``reasoning_effort``) live next to it and are handed to callers by the helper
functions at the bottom, so a model change never has to touch a call site.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # harmless when main_bot.py already did it; makes overrides work for bare imports


def _env(name: str, default: str) -> str:
    """Environment override for a model setting; blank values fall back to the default."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


# ---------------------------------------------------------------------------
# Anthropic - agent loop, judge, final compile (agents/main.py); document and
# image description (bot/media.py)
# ---------------------------------------------------------------------------
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-sonnet-5")
# Passed as ``output_config={"effort": ...}`` on every call to ANTHROPIC_MODEL.
# Sonnet 5 accepts low | medium | high | xhigh | max.
ANTHROPIC_EFFORT = _env("ANTHROPIC_EFFORT", "high")

# ---------------------------------------------------------------------------
# Anthropic fast - topic/summary, complexity classifier, "simple" answers
# (agents/main.py). Haiku 4.5 does not accept ``effort``; keep its calls plain.
# ---------------------------------------------------------------------------
ANTHROPIC_MODEL_FAST = _env("ANTHROPIC_MODEL_FAST", "claude-haiku-4-5")

# ---------------------------------------------------------------------------
# OpenAI reasoning models - critique (agents/main.py) and GPT vision on photos
# (bot/media.py); video frame description (bot/media.py)
# ---------------------------------------------------------------------------
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-5.6-terra")
VIDEO_FRAMES_MODEL = _env("VIDEO_FRAMES_MODEL", "gpt-5.6-luna")
# ``reasoning_effort`` for both models above (low | medium | high). Reasoning
# models reject ``temperature`` / ``top_p`` and take ``max_completion_tokens``
# instead of ``max_tokens``; openai_reasoning_options() below adds the effort.
OPENAI_REASONING_EFFORT = _env("OPENAI_REASONING_EFFORT", "high")
OPENAI_REASONING_MODELS = frozenset({OPENAI_MODEL, VIDEO_FRAMES_MODEL})

# ---------------------------------------------------------------------------
# Transcription (bot/media.py) - OpenAI Whisper, no request options
# ---------------------------------------------------------------------------
WHISPER_MODEL = _env("WHISPER_MODEL", "whisper-1")

# ---------------------------------------------------------------------------
# Embeddings (agents/embeddings.py) - the FAISS index is built with this
# dimension, so changing the model needs a matching dimension (and a rebuilt
# index for existing users).
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSION = int(_env("EMBEDDING_DIMENSION", "1536"))

# ---------------------------------------------------------------------------
# Image generation / editing / composition (agents/image_tools.py)
# ---------------------------------------------------------------------------
GEMINI_IMAGE_MODEL_FLASH = _env("GEMINI_IMAGE_MODEL_FLASH", "gemini-3.1-flash-image-preview")  # "Normal" mode
GEMINI_IMAGE_MODEL_PRO = _env("GEMINI_IMAGE_MODEL_PRO", "gemini-3-pro-image-preview")  # "Pro" mode
GPT_IMAGE_MODEL = _env("GPT_IMAGE_MODEL", "gpt-image-2-2026-04-21")  # "GPT" mode
DALLE_MODEL = _env("DALLE_MODEL", "dall-e-3")  # legacy generate_image_dall_e tool

# ---------------------------------------------------------------------------
# Video generation (agents/video_tools.py) - Google Veo, all five tools
# ---------------------------------------------------------------------------
VEO_MODEL = _env("VEO_MODEL", "veo-3.1-generate-preview")

# ---------------------------------------------------------------------------
# Search (agents/search_tools.py) - Perplexity deep_research default; the tool
# schema still offers the full PERPLEXITY_MODELS enum to the agent
# ---------------------------------------------------------------------------
PERPLEXITY_MODEL = _env("PERPLEXITY_MODEL", "sonar")
PERPLEXITY_MODELS = ("sonar-reasoning-pro", "sonar-pro", "sonar")
PERPLEXITY_REASONING_MODEL = "sonar-reasoning-pro"

# ---------------------------------------------------------------------------
# Text to speech (bot/media.py voice replies, agents/user_interactions.py)
# ---------------------------------------------------------------------------
TTS_MODEL = _env("TTS_MODEL", "eleven_multilingual_v2")


# ---------------------------------------------------------------------------
# Request options
# ---------------------------------------------------------------------------
def anthropic_request_options(thinking: bool | None = None) -> dict:
    """kwargs for ``messages.create`` / ``messages.stream`` on ANTHROPIC_MODEL.

    Always sets ``output_config.effort`` (GA, no beta header). ``thinking`` maps
    the user's thinking switch onto the API: ``True`` -> adaptive thinking with a
    summarized display (there is no ``budget_tokens`` any more; Sonnet 5 rejects
    it), ``False`` -> explicitly disabled (Sonnet 5 runs adaptive thinking when
    the parameter is omitted, which would eat small ``max_tokens`` budgets),
    ``None`` -> leave the parameter out. Not for ANTHROPIC_MODEL_FAST.
    """
    options: dict = {"output_config": {"effort": ANTHROPIC_EFFORT}}
    if thinking is True:
        options["thinking"] = {"type": "adaptive", "display": "summarized"}
    elif thinking is False:
        options["thinking"] = {"type": "disabled"}
    return options


def openai_reasoning_options(model: str) -> dict:
    """kwargs for ``chat.completions.create`` / ``.parse`` on an OpenAI model.

    Returns ``{"reasoning_effort": ...}`` for the reasoning models (OPENAI_MODEL,
    VIDEO_FRAMES_MODEL) and nothing for everything else (Whisper, embeddings,
    image models), so callers can splat it unconditionally.
    """
    if model in OPENAI_REASONING_MODELS:
        return {"reasoning_effort": OPENAI_REASONING_EFFORT}
    return {}
