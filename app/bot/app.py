"""Bot and dispatcher assembly plus the polling entry point."""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer

from bot import background, oauth_flow, oauth_web, runtime, widgets
from bot.auth import auth
from bot.config import config
from bot.handlers import build_root_router, sandbox_ui
from bot.handlers.commands import bot_commands
from bot.handlers.mcp_ui import refresh_admin_servers
from bot.handlers.widget_queries import UPDATE_TYPE as WIDGET_QUERY, WidgetQueryMiddleware
from bot.jobs import reminders_loop
from bot.media import configure_ffmpeg, ensure_temp_dirs
from bot.queue import Job
from bot.sender import ChatSender

logger = logging.getLogger(__name__)


def create_bot() -> Bot:
    session = None
    if config.use_local_api:
        server = TelegramAPIServer.from_base(config.local_api_url, is_local=True)
        session = AiohttpSession(api=server)
        logger.info("Using local Telegram Bot API at %s", config.local_api_url)
    return Bot(token=config.bot_token or "", session=session)


async def _on_timeout(bot: Bot, job: Job) -> None:
    try:
        await ChatSender(bot, int(job.user_id)).send_text(
            f"⏱ The task *{job.label}* exceeded the time limit ({config.turn_timeout_seconds // 60} min) and was stopped."
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Could not notify %s about timeout: %s", job.user_id, e)


async def register_commands(bot: Bot) -> None:
    """The command menu behind Telegram's "/" button. An old local Bot API server may refuse
    the call; the bot works without the menu, so a failure is only logged."""
    try:
        await bot.set_my_commands(bot_commands())
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not register the command menu: %s", e)


async def on_startup(bot: Bot) -> None:
    auth.load()
    configure_ffmpeg()
    ensure_temp_dirs()
    runtime.queue.on_timeout = lambda job: _on_timeout(bot, job)
    runtime.background_tasks.append(asyncio.create_task(reminders_loop(bot, runtime.queue), name="reminders-loop"))
    # The admin's MCP servers (MCP_SERVERS_FILE) get their tool cache refreshed once, off the startup path.
    runtime.background_tasks.append(asyncio.create_task(refresh_admin_servers(), name="mcp-admin-refresh"))
    # The callback page of MCP sign-ins (only with PUBLIC_BASE_URL; a bind failure is logged, never fatal).
    runtime.background_tasks.append(asyncio.create_task(oauth_web.serve(), name="oauth-web"))
    await start_sandbox_sessions(bot)
    await register_commands(bot)
    # Does the Bot API server know sendWidget (10.4)? Without it the widget tools are not offered.
    await widgets.probe(bot)
    me = await bot.me()
    oauth_web.set_bot_username(me.username)  # the "Open Telegram" link of the sign-in page
    logger.info("Bot @%s is running (streaming=%s, local_api=%s)", me.username, config.streaming_enabled, config.use_local_api)


async def start_sandbox_sessions(bot: Bot) -> None:
    """Persistent sandbox sessions (``secure_container.sessions``), when the manager exists.

    Containers a previous run left behind are removed before polling starts (a new session could
    otherwise be caught by the cleanup), then the ``sandbox-reaper`` task closes idle sessions and
    warns their users first, every ``sandbox_ui.REAP_INTERVAL`` seconds. No manager: nothing to do.
    Never raises.
    """
    manager = sandbox_ui.session_manager()
    if manager is None:
        return
    await sandbox_ui.cleanup_stale(manager)
    runtime.background_tasks.append(asyncio.create_task(sandbox_ui.reaper_loop(bot, manager), name="sandbox-reaper"))


async def on_shutdown(bot: Bot) -> None:
    logger.info("Shutting down: cancelling background tasks and queued jobs")
    for task in runtime.background_tasks:
        task.cancel()
    if runtime.background_tasks:
        await asyncio.gather(*runtime.background_tasks, return_exceptions=True)
    runtime.background_tasks.clear()
    await oauth_flow.shutdown()  # MCP sign-ins still waiting for the browser: cancelled, nothing announced
    await background.shutdown()  # the agents' background tool calls: cancelled, nothing announced
    await runtime.queue.shutdown()  # running turns drop their unread messages (cancel reason "shutdown")
    await sandbox_ui.shutdown_sessions()  # after the jobs: no turn starts a session behind it


def allowed_updates(dp: Dispatcher) -> list[str]:
    """The update types polling asks for: what the routers handle, plus the widget queries no
    router knows (a Bot API server without widgets ignores the unknown name)."""
    used = dp.resolve_used_update_types()
    return used + [WIDGET_QUERY] if WIDGET_QUERY not in used else used


def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    # Taps inside a widget that ask the bot (Bot API 10.4 widget_query): aiogram does not know the
    # update, so an outer middleware takes it before the "unknown update type" path.
    dp.update.outer_middleware(WidgetQueryMiddleware())
    dp.include_router(build_root_router())
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    return dp


async def run() -> None:
    # Blocking work (Docker waits, HTTP SDKs, ffmpeg) runs in asyncio.to_thread; the default
    # pool is tiny on small hosts, so size it explicitly to keep turns from starving each other.
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=config.thread_pool_size, thread_name_prefix="bot-worker")
    )
    bot = create_bot()
    dp = create_dispatcher()
    if config.drop_pending_updates:
        await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=allowed_updates(dp))
