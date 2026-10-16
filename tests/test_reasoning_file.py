"""The reasoning file: the turn as a transcript, tool arguments included, sent when its button is pressed."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from aiogram.types import Chat, InputRichBlockButtons, Message

import bot.agent_runner as agent_runner
import bot.streaming as streaming
from bot import rich, turn_actions
from bot.agent_runner import reasoning_text, run_turn
from bot.handlers.turn_actions_ui import on_reasoning

TURN = [
    {"role": "user", "content": "Add a papakha and replace the background"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "The user attached a photo, so this is an edit."},
            {"type": "text", "text": "Editing your photo."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "generate_image",
                "input": {"prompt": "add a papakha", "images": ["images/photo.jpg"], "quality": "high"},
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Edited 1 image(s) with gpt-image-2.5-sunburst"}
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "Here it is."}]},
]


def test_tool_calls_are_rendered_with_their_arguments():
    text = reasoning_text(TURN)
    assert "[Tool call: generate_image]" in text
    # the parameters the call was actually made with, which is the point of the file
    assert '"images"' in text and "images/photo.jpg" in text and '"quality": "high"' in text
    # the result names the call it belongs to
    assert "[Tool result: generate_image]" in text
    assert "gpt-image-2.5-sunburst" in text


def test_every_block_is_rendered_not_just_the_first():
    text = reasoning_text(TURN)
    # a thinking block used to hide the text and tool calls that followed it
    assert "[Assistant thinking]" in text
    assert "Editing your photo." in text and "Here it is." in text
    assert "Add a papakha and replace the background" in text


def test_failed_calls_are_marked_and_results_carry_block_content():
    text = reasoning_text(
        [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "run_command", "input": {"command": "ls"}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "is_error": True,
                        "content": [{"type": "text", "text": "boom"}],
                    }
                ],
            },
        ]
    )
    assert "[Tool result: run_command — error]" in text and "boom" in text


def test_long_values_are_clipped_and_junk_is_survivable():
    text = reasoning_text(
        [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {"p": "y" * 9000}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "z" * 9000}]},
            {"role": "assistant", "content": [None, {"type": "text", "text": "still here"}]},
            {"role": "user", "content": None},
            {},
        ]
    )
    assert "chars total)" in text and len(text) < 12000
    assert "still here" in text


def test_no_messages_is_empty_not_a_stray_separator():
    assert reasoning_text([]) == ""
    assert reasoning_text([{"role": "assistant", "content": []}]) == ""


# ---- on demand: stored with the turn, sent by the 📄 button --------------------------
USER = "42"


class FakeBot:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._id = 100

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_chat_action(self, **kw):
        return True

    async def _record(self, name, kw):
        self.calls.append((name, kw))
        return self._msg()

    async def send_message(self, **kw):
        return await self._record("send_message", kw)

    async def send_rich_message(self, **kw):
        return await self._record("send_rich_message", kw)

    async def edit_message_text(self, **kw):
        return await self._record("edit_message_text", kw)

    async def edit_message_reply_markup(self, **kw):
        return await self._record("edit_message_reply_markup", kw)

    async def send_document(self, **kw):
        return await self._record("send_document", kw)

    def of(self, name):
        return [kw for n, kw in self.calls if n == name]


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    stats = ModuleType("stats")
    stats.stats_tracker = SimpleNamespace(
        track_message_sent=lambda *a, **k: None, track_usage=lambda *a, **k: None, get_user_action_count=lambda *a, **k: 0,
    )
    monkeypatch.setitem(sys.modules, "stats", stats)
    monkeypatch.setattr(streaming, "config", SimpleNamespace(streaming_enabled=False))
    monkeypatch.setattr(turn_actions, "config", SimpleNamespace(elevenlabs_api_key=None, data_dir="data"))

    class Agent:
        def __init__(self, **kw):
            pass

        async def generate_response(self, prompt, update_status, on_text_chunk=None, trace=None):
            return "Here it is.", TURN

    monkeypatch.setattr(agent_runner, "_agents", lambda: SimpleNamespace(ChainOfThoughtAgent=Agent))
    rich.reset()
    yield
    rich.reset()


def reasoning_buttons(bot) -> list[str]:
    summary = bot.of("edit_message_text")[-1]["rich_message"]
    rows = [b for b in summary.blocks if isinstance(b, InputRichBlockButtons)]
    return [btn.callback_data for row in rows for btn in row.buttons if btn.callback_data.startswith("turn_reason_")]


async def test_the_file_is_stored_not_sent_and_the_button_sends_it(env):
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="Add a papakha", reasoning_caption="Agent task reasoning history.")
    assert bot.of("send_document") == []  # nothing is sent after the turn any more
    assert reasoning_buttons(bot) == [f"turn_reason_{result.turn_id}"]
    assert turn_actions.load_reasoning(USER, result.turn_id) == reasoning_text(TURN)

    class Callback:
        data = f"turn_reason_{result.turn_id}"
        message = Message(message_id=5, date=datetime.now(timezone.utc), chat=Chat(id=42, type="private"))
        answers: list = []

        async def answer(self, text=None, **kw):
            self.answers.append(text)

    await on_reasoning(Callback(), bot=bot, user_id=USER)
    doc = bot.of("send_document")[0]
    assert doc["document"].data.decode() == reasoning_text(TURN)
    assert doc["document"].filename == f"reasoning_42_{result.turn_id}.txt" and doc["caption"] == "Agent task reasoning history."


async def test_reasoning_context_off_stores_nothing_and_shows_no_button(env):
    path = Path("data") / USER / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"reasoning_context": {"enabled": False}}))
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi")
    assert turn_actions.load(USER, result.turn_id).has_reasoning is False
    assert turn_actions.load_reasoning(USER, result.turn_id) is None
    assert reasoning_buttons(bot) == [] and bot.of("send_document") == []


async def test_a_failed_save_sends_the_file_right_away_as_before(env, monkeypatch):
    async def broken(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(turn_actions, "save", broken)
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi", reasoning_caption="Agent task reasoning history.")
    assert result.turn_id is None and reasoning_buttons(bot) == []
    doc = bot.of("send_document")[0]
    assert doc["document"].data.decode() == reasoning_text(TURN) and doc["caption"] == "Agent task reasoning history."
