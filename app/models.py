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
# Sonnet 5 accepts low | medium | high | xhigh | max. ANTHROPIC_EFFORT is the
# level for the user's "thinking on" mode; ANTHROPIC_EFFORT_LIGHT for "thinking
# off" and for internal calls (judge, image/document descriptions). Thinking is
# adaptive in both cases: Anthropic's guidance for Sonnet 5 is to lower effort
# rather than disable thinking (disabled thinking can leak tool calls into text).
ANTHROPIC_EFFORT = _env("ANTHROPIC_EFFORT", "high")
ANTHROPIC_EFFORT_LIGHT = _env("ANTHROPIC_EFFORT_LIGHT", "low")
# max_tokens ceilings. There is no separate thinking budget any more: thinking
# tokens count against max_tokens, so the ceiling must leave room for the
# reasoning AND the answer (Sonnet 5 allows up to 128k). The agent loop always
# streams, which is what makes the large value safe: the SDK refuses roughly
# > 21k tokens on non-streaming calls because they could exceed its 10-minute
# request timeout.
ANTHROPIC_MAX_TOKENS = int(_env("ANTHROPIC_MAX_TOKENS", "64000"))
ANTHROPIC_MAX_TOKENS_LIGHT = int(_env("ANTHROPIC_MAX_TOKENS_LIGHT", "16000"))

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
# instead of ``max_tokens``; that cap covers the hidden reasoning AND the
# visible answer, so the calls leave it unset and let the model's own output
# limit apply. openai_reasoning_options() below adds the effort.
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
def anthropic_request_options(thinking: bool | None = None, *, effort: str | None = None) -> dict:
    """kwargs for ``messages.create`` / ``messages.stream`` on ANTHROPIC_MODEL.

    Adaptive thinking is the only thinking mode on Sonnet 5 (a fixed
    ``budget_tokens`` is rejected) and it runs whether the parameter is present
    or omitted, so the user's thinking switch selects *how much*:

    * ``True``  -> effort ANTHROPIC_EFFORT, ``display: "summarized"`` (feeds the
      streaming thinking block and the reasoning file)
    * ``False`` -> effort ANTHROPIC_EFFORT_LIGHT, ``display: "omitted"`` (faster
      and cheaper; the model still reasons briefly when it must)
    * ``None``  -> effort only, thinking left to the API default (adaptive)

    ``effort`` overrides the level. Not for ANTHROPIC_MODEL_FAST: Haiku rejects
    ``effort``.
    """
    if effort is None:
        effort = ANTHROPIC_EFFORT_LIGHT if thinking is False else ANTHROPIC_EFFORT
    options: dict = {"output_config": {"effort": effort}}
    if thinking is True:
        options["thinking"] = {"type": "adaptive", "display": "summarized"}
    elif thinking is False:
        options["thinking"] = {"type": "adaptive", "display": "omitted"}
    return options


def anthropic_max_tokens(thinking: bool | None = None) -> int:
    """The max_tokens ceiling matching :func:`anthropic_request_options`."""
    return ANTHROPIC_MAX_TOKENS_LIGHT if thinking is False else ANTHROPIC_MAX_TOKENS


ANTHROPIC_MAX_TOKENS_FAST = int(_env("ANTHROPIC_MAX_TOKENS_FAST", "16000"))


def request_options_for(model: str, thinking: bool | None = None) -> dict:
    """Request options for any Anthropic model: nothing for the fast model (Haiku rejects
    ``effort`` and adaptive thinking), :func:`anthropic_request_options` otherwise."""
    if model == ANTHROPIC_MODEL_FAST:
        return {}
    return anthropic_request_options(thinking)


def max_tokens_for(model: str, thinking: bool | None = None) -> int:
    if model == ANTHROPIC_MODEL_FAST:
        return ANTHROPIC_MAX_TOKENS_FAST
    return anthropic_max_tokens(thinking)


# ---------------------------------------------------------------------------
# Prompt caching (agents/main.py): the static system prompt gets an explicit
# breakpoint with this TTL; the conversation tail uses the API's automatic
# breakpoint. PROMPT_CACHING=false turns both off (debugging only).
# ---------------------------------------------------------------------------
PROMPT_CACHING = _env("PROMPT_CACHING", "true").strip().lower() in ("1", "true", "yes", "on")
SYSTEM_CACHE_TTL = _env("SYSTEM_CACHE_TTL", "1h")  # "5m" | "1h"


def anthropic_text(message) -> str:
    """The concatenated text blocks of a Messages API response.

    With adaptive thinking the first content block can be a ``thinking`` block,
    so ``message.content[0].text`` is not safe any more.
    """
    return "".join(
        getattr(block, "text", "") or ""
        for block in getattr(message, "content", []) or []
        if getattr(block, "type", None) == "text"
    )


def openai_reasoning_options(model: str) -> dict:
    """kwargs for ``chat.completions.create`` / ``.parse`` on an OpenAI model.

    Returns ``{"reasoning_effort": ...}`` for the reasoning models (OPENAI_MODEL,
    VIDEO_FRAMES_MODEL) and nothing for everything else (Whisper, embeddings,
    image models), so callers can splat it unconditionally.
    """
    if model in OPENAI_REASONING_MODELS:
        return {"reasoning_effort": OPENAI_REASONING_EFFORT}
    return {}


# ---------------------------------------------------------------------------
# Pricing (USD per million tokens: input, output) for the token accounting shown
# in the status summary and the admin /stats view. Cache reads cost 10% of the
# input price, cache writes 125% (5-minute entries; the hourly system block is
# written rarely). Unknown models get no cost estimate, only token counts.
# ---------------------------------------------------------------------------
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def estimate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    """USD estimate for one model's usage, or None when the model is not in MODEL_PRICES."""
    prices = MODEL_PRICES.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    return (
        input_tokens * price_in
        + cache_read_tokens * price_in * CACHE_READ_MULTIPLIER
        + cache_write_tokens * price_in * CACHE_WRITE_MULTIPLIER
        + output_tokens * price_out
    ) / 1_000_000
