"""Process-wide runtime objects shared by handlers, jobs and the app."""
from __future__ import annotations

import asyncio

from bot.config import config
from bot.queue import QueueManager

queue = QueueManager(
    max_concurrent_turns=config.max_concurrent_turns,
    turn_timeout=config.turn_timeout_seconds,
)

background_tasks: list[asyncio.Task] = []
