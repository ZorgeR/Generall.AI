"""Router assembly. Order matters: the catch-all text handler lives in the last router."""
from __future__ import annotations

from aiogram import Router

from bot.handlers import commands, messages, reminders_ui, settings_ui, stats_ui
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
    ui.include_router(commands.voice_router)
    ui.include_router(settings_ui.router)
    ui.include_router(reminders_ui.router)
    ui.include_router(messages.control_router)
    root.include_router(ui)

    chat = Router(name="chat")
    chat.message.middleware(AuthMiddleware(check_limits=True))
    chat.include_router(messages.router)
    root.include_router(chat)

    return root
