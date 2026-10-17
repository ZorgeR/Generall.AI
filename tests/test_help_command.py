"""/help, the richer /start and the command menu registered at startup (F15)."""
import asyncio
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.filters import CommandObject
from aiogram.types import Chat, Message

from bot import app, runtime
from bot.handlers import commands

USER = "42"
MENU = ["start", "help", "settings", "usage", "reminders", "profile", "timezone", "voice", "mcp", "cancel"]
HELP_COMMANDS = ["/help", "/settings", "/usage", "/reminders", "/profile", "/timezone", "/voice", "/mcp", "/cancel"]


class FakeAuth:
    def __init__(self, allowed):
        self.allowed = set(allowed)

    def is_authorized(self, user_id):
        return str(user_id) in self.allowed


def message(chat_id: int = 42, text: str = "/help") -> Message:
    return Message(message_id=1, date=datetime.now(timezone.utc), chat=Chat(id=chat_id, type="private"), text=text)


@pytest.fixture
def answers(monkeypatch) -> AsyncMock:
    sent = AsyncMock()
    monkeypatch.setattr(commands, "answer_md", sent)
    return sent


# ---- /help ---------------------------------------------------------------------------------
def test_help_lists_capabilities_commands_and_examples():
    text = commands.help_text(True)
    for name, description in commands.COMMANDS:
        if name != "start":
            assert f"/{name} — {description}" in text
    for command in HELP_COMMANDS:
        assert command in text
    assert 4 <= len(commands.HELP_EXAMPLES) <= 6
    assert all(example in text for example in commands.HELP_EXAMPLES)
    lowered = text.lower()
    for capability in ("voice", "photo", "document", "video", "web", "code", "file", "image", "reminder", "memory"):
        assert capability in lowered, capability


def test_help_for_an_unauthorized_user_explains_the_invite_flow():
    text = commands.help_text(False)
    assert "/invite" in text and "invite" in text.lower()
    assert "/settings" not in text and "/usage" not in text


async def test_help_command_answers_according_to_authorization(monkeypatch, answers):
    monkeypatch.setattr(commands, "auth", FakeAuth({USER}))
    await commands.help_command(message(42))
    await commands.help_command(message(43))
    assert [call.args[1] for call in answers.await_args_list] == [commands.help_text(True), commands.help_text(False)]
    assert all(call.args[0].chat.id == chat for call, chat in zip(answers.await_args_list, (42, 43)))


# ---- /start --------------------------------------------------------------------------------
async def test_start_for_an_authorized_user_adds_examples_and_pointers(monkeypatch, answers):
    monkeypatch.setattr(commands, "auth", FakeAuth({USER}))
    await commands.start_command(message(42, "/start"), bot=None, command=CommandObject(prefix="/", command="start"))
    text = answers.await_args.args[1]
    assert text.startswith(commands.WELCOME_AUTHORIZED)
    assert "Try" in text and len(commands.START_EXAMPLES) == 3
    assert all(example in text for example in commands.START_EXAMPLES)
    assert "/help" in text and "/settings" in text


# ---- the command menu ----------------------------------------------------------------------
def test_bot_commands_menu_lists_the_user_commands_only():
    menu = commands.bot_commands()
    assert [c.command for c in menu] == MENU
    assert all(0 < len(c.description) <= 256 for c in menu)
    assert not {"stats", "listusers"} & {c.command for c in menu}


class StartupBot:
    def __init__(self, error: Exception | None = None):
        self.set_my_commands = AsyncMock(side_effect=error)
        self.me = AsyncMock(return_value=SimpleNamespace(username="generall_bot"))


@pytest.fixture
async def quiet_startup(monkeypatch, tmp_path):
    """on_startup without the auth file, ffmpeg or a live reminders loop; background tasks cancelled after."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app.auth, "load", lambda: None)
    monkeypatch.setattr(app, "configure_ffmpeg", lambda: None)

    async def idle_loop(bot, qm):
        await asyncio.sleep(3600)

    monkeypatch.setattr(app, "reminders_loop", idle_loop)
    yield
    for task in runtime.background_tasks:
        task.cancel()
    await asyncio.gather(*runtime.background_tasks, return_exceptions=True)
    runtime.background_tasks.clear()


async def test_startup_registers_the_command_menu(quiet_startup):
    bot = StartupBot()
    await app.on_startup(bot)
    bot.set_my_commands.assert_awaited_once()
    (menu,) = bot.set_my_commands.await_args.args
    assert [c.command for c in menu] == MENU


async def test_a_refused_command_menu_does_not_stop_startup(quiet_startup, caplog):
    bot = StartupBot(error=RuntimeError("Bad Request: method not found"))
    with caplog.at_level(logging.WARNING, logger="bot.app"):
        await app.on_startup(bot)
    bot.me.assert_awaited_once()  # startup went on past the menu
    assert any("command menu" in record.getMessage() for record in caplog.records)
