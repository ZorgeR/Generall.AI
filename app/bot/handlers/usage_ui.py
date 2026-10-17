"""``/usage``: the user's own usage (docs/design-v3.md, F4). Router on ``ui`` (auth, no quota)."""
from __future__ import annotations

from aiogram import Router

router = Router(name="usage")
