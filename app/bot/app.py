"""Bot and dispatcher assembly plus the polling entry point."""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer

from bot import runtime
from bot.auth import auth
from bot.config import config
from bot.handlers import build_root_router
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


async def on_startup(bot: Bot) -> None:
    auth.load()
    configure_ffmpeg()
    ensure_temp_dirs()
    runtime.queue.on_timeout = lambda job: _on_timeout(bot, job)
    runtime.background_tasks.append(asyncio.create_task(reminders_loop(bot, runtime.queue), name="reminders-loop"))
    me = await bot.me()
    logger.info("Bot @%s is running (streaming=%s, local_api=%s)", me.username, config.streaming_enabled, config.use_local_api)


async def on_shutdown(bot: Bot) -> None:
    logger.info("Shutting down: cancelling background tasks and queued jobs")
    for task in runtime.background_tasks:
        task.cancel()
    if runtime.background_tasks:
        await asyncio.gather(*runtime.background_tasks, return_exceptions=True)
    runtime.background_tasks.clear()
    await runtime.queue.shutdown()


def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()
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
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
