"""Buttons that act on a finished turn (``turn_*`` callbacks).

``router`` (ui: auth only) handles actions that do not run the agent;
``turn_router`` (chat: auth + action limit) handles the ones that start a new turn.
"""
from __future__ import annotations

from aiogram import Router

router = Router(name="turn-actions")
turn_router = Router(name="turn-actions-agent")
