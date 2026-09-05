"""Lazily constructed API clients used by the Telegram layer.

Clients are created on first use so importing the package never requires API
keys (the agent package still builds its own clients at import time). Model
names and request options live in the top-level ``models`` module.
"""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def whisper_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY_WHISPER") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key)


@lru_cache(maxsize=1)
def anthropic_client():
    import anthropic

    return anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
