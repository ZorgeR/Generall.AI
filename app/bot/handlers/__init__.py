"""Router assembly. Order matters: the catch-all text handler lives in the last router."""
from __future__ import annotations

from aiogram import Router

from bot.handlers import (
    commands,
    mcp_ui,
    messages,
    profile_ui,
    questions_ui,
    reminders_ui,
    sandbox_ui,
    settings_ui,
    stats_ui,
    turn_actions_ui,
    usage_ui,
)
from bot.handlers.middleware import AuthMiddleware


def build_root_router() -> Router:
    root = Router(name="root")

    # /start and /invite must work for people who are not yet authorized.
    root.include_router(commands.public_router)

    admin = Router(name="admin")
    admin.message.middleware(AuthMiddleware(require_admin=True))
    admin.callback_query.middleware(AuthMiddleware(require_admin=True))
    admin.include_router(commands.admin_router)
    admin.include_router(stats_ui.router)
    root.include_router(admin)

    ui = Router(name="ui")
    ui.message.middleware(AuthMiddleware())
    ui.callback_query.middleware(AuthMiddleware())
    ui.stopped_message_generation.middleware(AuthMiddleware())  # the Stop button on a streaming draft
    ui.include_router(commands.voice_router)
    ui.include_router(settings_ui.router)
    ui.include_router(reminders_ui.router)
    ui.include_router(usage_ui.router)  # /usage
    ui.include_router(profile_ui.router)  # /profile, profile_ buttons
    ui.include_router(mcp_ui.router)  # /mcp, mcp_ buttons
    ui.include_router(sandbox_ui.router)  # /sandbox, sandbox_ buttons (incl. the idle warnings)
    ui.include_router(messages.control_router)
    ui.include_router(questions_ui.router)  # ask_user_choice answers
    ui.include_router(turn_actions_ui.router)  # listen / reasoning file: no agent run, no quota
    root.include_router(ui)

    chat = Router(name="chat")
    chat.message.middleware(AuthMiddleware(check_limits=True))
    # Retry / Continue / follow-up buttons start an agent turn, so they pass the quota check too.
    chat.callback_query.middleware(AuthMiddleware(check_limits=True))
    chat.include_router(turn_actions_ui.turn_router)
    chat.include_router(messages.router)
    root.include_router(chat)

    return root
