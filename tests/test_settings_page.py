"""bot/handlers/settings_ui.py: the rich /settings page, apply_setting and the keyboard fallback."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.methods import SendMessage
from aiogram.types import (
    Chat,
    InputRichBlockButtons,
    InputRichBlockDetails,
    InputRichBlockParagraph,
    InputRichMessage,
    Message,
    RichTextBold,
)

from bot import rich
from bot.handlers import settings_ui as su
from bot.settings import DEFAULT_SETTINGS, MODE_TYPES, PRESET_NAMES, SYSTEM_PROMPT_TYPES, UserSettings, current_preset
from models import IMAGE_MAX_VARIANTS

_M = SendMessage(chat_id=1, text="x")
USER = 42


def bad_request(msg="Bad Request: can't parse rich message"):
    return TelegramBadRequest(method=_M, message=msg)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # UserSettings writes data/<uid>/settings.json under the cwd
    rich.reset()
    monkeypatch.setitem(su._page_state, "flat", False)
    yield
    rich.reset()


@pytest.fixture
def ui(monkeypatch):
    """Record the keyboard UI (answer_md / edit_md) instead of talking to Telegram."""
    calls = []

    async def answer_md(message, text, reply_markup=None):
        calls.append(("answer_md", text, reply_markup))

    async def edit_md(message, text, reply_markup=None):
        calls.append(("edit_md", text, reply_markup))

    monkeypatch.setattr(su, "answer_md", answer_md)
    monkeypatch.setattr(su, "edit_md", edit_md)
    return calls


def make_message(*, thread_id=None, message_id=7):
    return Message(
        message_id=message_id,
        date=datetime.now(timezone.utc),
        chat=Chat(id=USER, type="supergroup" if thread_id else "private"),
        message_thread_id=thread_id,
        is_topic_message=True if thread_id else None,
    )


def make_callback(data, message=None):
    return SimpleNamespace(data=data, message=message or make_message(), answer=AsyncMock())


def make_bot(*, send_errors=(), edit_errors=()):
    bot = SimpleNamespace()
    bot.send_rich_message = AsyncMock(side_effect=list(send_errors) + [SimpleNamespace(message_id=100)])
    bot.edit_message_text = AsyncMock(side_effect=list(edit_errors) + [SimpleNamespace(message_id=7)])
    return bot


def with_values(values):
    """A UserSettings with raw (unsaved, unvalidated) values: {(category, key): value}."""
    us = UserSettings(str(USER))
    for (category, key), value in values.items():
        us.settings.setdefault(category, {})[key] = value
    return us


def walk(blocks):
    for block in blocks:
        yield block
        if isinstance(block, InputRichBlockDetails):
            yield from walk(block.blocks)


def rows(blocks):
    return [b for b in walk(blocks) if isinstance(b, InputRichBlockButtons)]


def details(blocks):
    return [b for b in blocks if isinstance(b, InputRichBlockDetails)]


def row_of(blocks, name):
    """The button row that follows the paragraph whose bold name is ``name``."""
    flat = list(walk(blocks))
    for i, block in enumerate(flat):
        if isinstance(block, InputRichBlockParagraph) and isinstance(block.text, list) and block.text[0].text == name:
            return flat[i + 1]
    raise AssertionError(f"no setting named {name!r}")


def rows_of(blocks, name):
    """Every button row that follows the paragraph whose bold name is ``name`` (a choice may take several)."""
    flat = list(walk(blocks))
    for i, block in enumerate(flat):
        if isinstance(block, InputRichBlockParagraph) and isinstance(block.text, list) and block.text[0].text == name:
            rows = []
            for nxt in flat[i + 1:]:
                if not isinstance(nxt, InputRichBlockButtons):
                    break
                rows.append(nxt)
            return rows
    raise AssertionError(f"no setting named {name!r}")


def choice_buttons(blocks, name):
    return [b for row in rows_of(blocks, name) for b in row.buttons]


def names_on(blocks):
    """The bold names of the settings (and the Preset paragraph) in page order."""
    return [b.text[0].text for b in walk(blocks) if isinstance(b, InputRichBlockParagraph) and isinstance(b.text, list)]


SECTION_CONTROLS = [c for s in su.SECTIONS for c in s.controls]
BASIC = [c for c in SECTION_CONTROLS if not c.advanced]
ADVANCED = [c for c in SECTION_CONTROLS if c.advanced]
SHOW_ALL = {("ui", "advanced"): True}

EDGE_SETTINGS = [
    {},  # defaults
    {  # everything off, every number at its minimum
        ("transcript", "enabled"): False, ("transcript", "max_context_tokens"): 20000, ("transcript", "keep_tool_results_turns"): 0,
        ("summarization_history", "enabled"): False, ("summarization_history", "size"): 1,
        ("dialog_history", "size"): 1, ("tools", "max_iteration"): 1, ("critique", "max_iteration"): 1,
        ("semantic_search", "max_results"): 1, ("image", "max_variants"): 1, ("image", "metadata"): False,
        ("follow_ups", "enabled"): False, ("rich_messages", "enabled"): False, ("trace", "keep_summary"): False,
        ("mode", "type"): "speed", ("thinking", "enabled"): False,
        ("browser", "enabled"): False, ("browser", "stream"): False, ("browser", "remember"): False,
    },
    {  # every number at its maximum, last choices, every setting shown
        ("transcript", "max_context_tokens"): 400000, ("transcript", "keep_tool_results_turns"): 50,
        ("summarization_history", "size"): 50, ("dialog_history", "size"): 50, ("tools", "max_iteration"): 300,
        ("judge", "max_iteration"): 300, ("semantic_search", "max_results"): 20, ("image", "max_variants"): IMAGE_MAX_VARIANTS,
        ("image", "engine"): "story", ("image", "quality"): "max", ("image", "size"): "4K",
        ("system_prompt", "type"): SYSTEM_PROMPT_TYPES[-1], ("mode", "type"): MODE_TYPES[-1], **SHOW_ALL,
    },
    {  # hand-edited garbage
        ("tools", "max_iteration"): "lots", ("transcript", "max_context_tokens"): None, ("summarization_history", "size"): 4,
        ("image", "engine"): "dall-e-" + "x" * 80, ("image", "quality"): "", ("system_prompt", "type"): "custom",
        ("image", "max_variants"): 99, ("mode", "type"): "warp", ("ui", "advanced"): "yes",
    },
]


# ---- rendering ---------------------------------------------------------------
def test_page_layout_for_defaults():
    blocks = su.settings_page_blocks(UserSettings(str(USER)))
    assert blocks[0].text.text == "⚙️ Settings"
    assert blocks[1].text.text == su.PAGE_HINT
    assert blocks[2].text[0].text == "Preset" and isinstance(blocks[3], InputRichBlockButtons)
    sections = details(blocks)
    assert [d.summary.split(" · ")[0] for d in sections] == ["🧠 Memory", "🤖 Agent", "💬 Replies", "🎨 Images", "🌐 Browser"]
    assert sections[0].summary == "🧠 Memory · transcript ✅ 120k · summaries ✅ 5"
    assert sections[1].summary == "🤖 Agent · auto · thinking ✅ · tools ✅ 20 · generall-ai-v2"
    assert sections[2].summary == "💬 Replies · rich ✅ · summary ✅ · follow-ups ✅"
    assert sections[3].summary == "🎨 Images · auto · auto · 2K · ×4"
    assert sections[4].summary == "🌐 Browser · on · screenshots ✅ · logins ✅"
    assert not any(d.is_open for d in sections)  # a fresh /settings: all closed
    # Every setting is a bold paragraph followed by its button row(s): a choice may take several.
    us = UserSettings(str(USER))
    for section in sections:
        groups, current = [], None
        for block in section.blocks:
            if isinstance(block, InputRichBlockParagraph):
                current = [block]
                groups.append(current)
            else:
                assert isinstance(block, InputRichBlockButtons) and current is not None
                current.append(block)
        assert all(len(g) >= 2 and isinstance(g[0].text[0], RichTextBold) for g in groups)
    # The presets row plus each everyday setting's rows; Advanced on shows every setting.
    def row_count(controls, settings):
        return sum(len(su.control_rows(c, settings)) for c in controls)

    assert len(rows(blocks)) == 1 + row_count(BASIC, us)
    assert len(rows(su.settings_page_blocks(with_values(SHOW_ALL)))) == 1 + row_count(SECTION_CONTROLS, us)
    assert len(su.CONTROLS) == len(SECTION_CONTROLS) + 1  # + the Advanced toggle itself


def test_only_the_open_section_is_open():
    blocks = su.settings_page_blocks(UserSettings(str(USER)), open_section="replies")
    assert [bool(d.is_open) for d in details(blocks)] == [False, False, True, False, False]


def test_flat_layout_has_the_same_content_without_details():
    us = UserSettings(str(USER))
    flat = su.settings_page_blocks(us, sections=False)
    assert not any(isinstance(b, InputRichBlockDetails) for b in flat)
    headings = [b.text.text for b in flat if isinstance(b, InputRichBlockParagraph) and isinstance(b.text, RichTextBold)]
    assert headings[1:] == [d.summary for d in details(su.settings_page_blocks(us))]
    assert [r.model_dump() for r in rows(flat)] == [r.model_dump() for r in rows(su.settings_page_blocks(us))]


@pytest.mark.parametrize("values", EDGE_SETTINGS)
@pytest.mark.parametrize("sections", [True, False])
def test_every_button_fits_parses_and_applies(values, sections):
    us = with_values(values)
    blocks = su.settings_page_blocks(us, open_section="memory", sections=sections)
    page = InputRichMessage(blocks=blocks)
    assert len(page.model_dump_json(exclude_none=True).encode()) < rich.RICH_BYTE_LIMIT // 2
    assert len(list(walk(blocks))) < rich.RICH_BLOCK_LIMIT
    for row in rows(blocks):
        assert 1 <= len(row.buttons) <= su.ROW_BUTTON_LIMIT
        for button in row.buttons:
            assert button.text
            if button.callback_data is None:
                assert button.disabled is not None  # value labels and the −/+ at either end
                continue
            data = button.callback_data
            assert data.startswith(su.PAGE_PREFIX) and len(data.encode()) <= su.CALLBACK_DATA_LIMIT
            token, action, value = su.parse_page_callback(data)
            assert (token, action) in su.CONTROLS or (token, action) == (su.PRESET_TOKEN, "set")
            if token != "system_prompt" and value is not None:
                assert "_" not in value
            # Every button really changes something, each applied to the rendered state: apply_setting
            # saves, so drop the file a previous button wrote or this would see that button's change.
            # The one exception is the active preset, which is marked and a no-op.
            Path(us.path).unlink(missing_ok=True)
            fresh = with_values(values)
            assert fresh.as_dict() == us.as_dict()
            expected = value != current_preset(fresh) if token == su.PRESET_TOKEN else True
            assert su.apply_setting(fresh, token, action, value) is expected, data


def test_toggle_button():
    us = with_values({("follow_ups", "enabled"): True, ("thinking", "enabled"): False})
    blocks = su.settings_page_blocks(us)
    on = row_of(blocks, "🧭 Suggested next questions").buttons
    off = row_of(blocks, "🤔 Deep reasoning").buttons
    assert [(b.text, b.style, b.callback_data) for b in on] == [("✅ On", "success", "setp_followups_toggle")]
    assert [(b.text, b.style, b.callback_data) for b in off] == [("⬜ Off", None, "setp_thinking_toggle")]


def test_number_row_steps_and_ends():
    us = with_values({("tools", "max_iteration"): 1, ("judge", "max_iteration"): 300, ("transcript", "max_context_tokens"): 120000, **SHOW_ALL})
    blocks = su.settings_page_blocks(us)
    minus, label, plus = row_of(blocks, "Tool calls per answer").buttons
    assert minus.disabled is not None and minus.callback_data is None and minus.text == "−"
    assert label.text == "1" and label.disabled is not None
    assert plus.callback_data == "setp_tools_iteration_2"
    minus, label, plus = row_of(blocks, "Check rounds").buttons
    assert minus.callback_data == "setp_judge_iteration_200" and label.text == "300" and plus.disabled is not None
    minus, label, plus = row_of(blocks, "Memory window").buttons
    assert (minus.callback_data, label.text, plus.callback_data) == (
        "setp_transcript_ctx_100000", "120k", "setp_transcript_ctx_150000")


def test_step_values_snap_off_list_values_without_skipping():
    choices = (1, 5, 10, 20)
    assert su.step_values(choices, 5) == (1, 10)
    assert su.step_values(choices, 1) == (None, 5)
    assert su.step_values(choices, 20) == (10, None)
    assert su.step_values(choices, 4) == (1, 5)  # off the list: nearest choice in each direction
    assert su.step_values(choices, 7) == (5, 10)
    assert su.step_values(choices, 0) == (None, 1)
    assert su.step_values(choices, 99) == (20, None)


def test_number_row_for_an_off_list_value_shows_the_real_value():
    blocks = su.settings_page_blocks(with_values({("summarization_history", "size"): 4, ("transcript", "max_context_tokens"): 400000, **SHOW_ALL}))
    minus, label, plus = row_of(blocks, "Summaries kept").buttons
    assert (minus.callback_data, label.text, plus.callback_data) == ("setp_summarization_size_1", "4", "setp_summarization_size_5")
    minus, label, plus = row_of(blocks, "Memory window").buttons
    assert (minus.callback_data, label.text, plus.disabled is not None) == ("setp_transcript_ctx_300000", "400k", True)


def test_choices_are_one_button_each_with_the_current_one_checked():
    blocks = su.settings_page_blocks(with_values({("image", "quality"): "max", ("system_prompt", "type"): SYSTEM_PROMPT_TYPES[-1], **SHOW_ALL}))
    quality = choice_buttons(blocks, "✨ Image quality")
    assert [b.text for b in quality] == ["auto", "low", "medium", "high", "xhigh", "✅ max"]
    assert quality[-1].disabled is not None and quality[-1].callback_data is None  # the current one is not tappable
    assert [b.callback_data for b in quality[:-1]] == [f"setp_image_quality_{q}" for q in ("auto", "low", "medium", "high", "xhigh")]
    assert [len(r.buttons) for r in rows_of(blocks, "✨ Image quality")] == [3, 3]  # six choices: 3 + 3

    engine = choice_buttons(blocks, "🖌 Image engine")
    assert [b.text for b in engine] == ["✅ auto", "best", "fast", "story"]  # every engine visible, one row
    assert len(rows_of(blocks, "🖌 Image engine")) == 1
    assert [b.callback_data for b in engine[1:]] == ["setp_image_engine_best", "setp_image_engine_fast", "setp_image_engine_story"]

    prompts = choice_buttons(blocks, "🧩 Personality prompt")
    assert [b.callback_data for b in prompts if b.callback_data] == [f"setp_sysprompt_{t}" for t in SYSTEM_PROMPT_TYPES[:-1]]
    assert prompts[-1].text == f"✅ {su.CHOICE_LABELS[('system_prompt', 'type')][SYSTEM_PROMPT_TYPES[-1]]}"


def test_an_unknown_stored_choice_leaves_every_button_tappable():
    blocks = su.settings_page_blocks(with_values({("image", "size"): "8K"}))
    sizes = choice_buttons(blocks, "📐 Image size")
    assert [b.text for b in sizes] == ["1K", "2K", "4K"] and all(b.callback_data for b in sizes)


def test_split_rows_keeps_rows_narrow_and_even():
    assert su.split_rows(["auto", "best", "fast", "story"]) == [4]
    assert su.split_rows(["auto", "low", "medium", "high", "xhigh", "max"]) == [3, 3]
    assert su.split_rows(["a" * 30, "b"]) == [1, 1]  # a long label gets its own row
    assert su.split_rows(["x"] * 9) == [3, 3, 3]
    assert su.split_rows([]) == []


MODE_NAME = "🎛 Quick answers vs. deep work"


def test_mode_choices_set_each_mode():
    us = UserSettings(str(USER))
    for mode in MODE_TYPES:
        buttons = choice_buttons(su.settings_page_blocks(us), MODE_NAME)
        assert [b.text.removeprefix("✅ ") for b in buttons] == list(MODE_TYPES)
        (checked,) = [b for b in buttons if b.callback_data is None]
        assert checked.text == f"✅ {us.get('mode', 'type')}"
        (button,) = [b for b in buttons if b.callback_data and b.callback_data.endswith(f"_{mode}")] or [None]
        if button is None:  # already the current mode
            continue
        token, action, value = su.parse_page_callback(button.callback_data)
        assert (token, action, value) == ("mode", "set", mode)
        assert su.apply_setting(us, token, action, value) is True
        assert UserSettings(str(USER)).get("mode", "type") == mode
    assert us.get("mode", "type") == MODE_TYPES[-1]
    assert su.MODE_HINT in su.CONTROLS[("mode", "set")].hint
    assert "intelligence" in su.section_summary(su.SECTIONS[1], with_values({("mode", "type"): "intelligence"}))


def test_presets_row_marks_the_active_preset():
    row = su.settings_page_blocks(UserSettings(str(USER)))[3]
    assert [(b.text, b.style, b.callback_data) for b in row.buttons] == [
        ("⚡ Fast", None, "setp_preset_fast"),
        ("✅ ⚖️ Balanced", "primary", "setp_preset_balanced"),
        ("🧠 Deep", None, "setp_preset_deep"),
        ("⚙️ Advanced ⬜", None, "setp_ui_advanced"),
    ]
    custom = su.settings_page_blocks(with_values({("tools", "max_iteration"): 11}))[3]
    assert [b.text for b in custom.buttons[:3]] == [su.PRESET_LABELS[name] for name in PRESET_NAMES]
    assert not any(b.style for b in custom.buttons)
    fast = su.settings_page_blocks(with_values({("mode", "type"): "speed", ("thinking", "enabled"): False, ("tools", "max_iteration"): 10, **SHOW_ALL}))[3]
    assert [(b.text, b.style) for b in fast.buttons] == [
        ("✅ ⚡ Fast", "primary"), ("⚖️ Balanced", None), ("🧠 Deep", None), ("⚙️ Advanced ✅", None)]


def test_advanced_toggle_hides_the_power_settings():
    basic_page = su.settings_page_blocks(UserSettings(str(USER)))
    full_page = su.settings_page_blocks(with_values(SHOW_ALL))
    assert names_on(basic_page) == ["Preset"] + [c.name for c in BASIC]
    assert names_on(full_page) == ["Preset"] + [c.name for c in SECTION_CONTROLS]
    with pytest.raises(AssertionError):
        row_of(basic_page, "🧩 Personality prompt")
    assert row_of(full_page, "🧩 Personality prompt")
    # the hidden ones are exactly the power settings of the design
    assert {(c.category, c.key) for c in ADVANCED} == {
        ("transcript", "max_context_tokens"), ("transcript", "keep_tool_results_turns"),
        ("summarization_history", "size"), ("dialog_history", "enabled"), ("dialog_history", "size"),
        ("reasoning_context", "enabled"), ("short_term_memory", "enabled"), ("semantic_search", "max_results"),
        ("critique", "enabled"), ("critique", "max_iteration"), ("judge", "enabled"), ("judge", "max_iteration"),
        ("system_prompt", "type"), ("image", "max_variants"), ("image", "metadata"),
    }
    # a hidden setting still counts in its section's summary, and the flat layout hides it too
    assert "generall-ai-v2" in details(basic_page)[1].summary
    assert names_on(su.settings_page_blocks(UserSettings(str(USER)), sections=False)) == names_on(basic_page)


BROWSER_NAMES = ["🌐 Interactive browser", "📸 Show screenshots while browsing", "🔑 Remember logins"]


def test_browser_section_has_three_everyday_toggles():
    blocks = su.settings_page_blocks(UserSettings(str(USER)), open_section="browser")
    browser = details(blocks)[4]
    assert browser.is_open and browser.summary.startswith("🌐 Browser")
    names = [b.text[0].text for b in browser.blocks if isinstance(b, InputRichBlockParagraph)]
    assert names == BROWSER_NAMES
    assert all(not c.advanced for c in su.SECTIONS[4].controls)
    datas = [row_of(blocks, name).buttons[0].callback_data for name in BROWSER_NAMES]
    assert datas == ["setp_browser_toggle", "setp_browser_stream", "setp_browser_remember"]
    assert all(row_of(blocks, name).buttons[0].text == "✅ On" for name in BROWSER_NAMES)
    hints = " ".join(c.hint for c in su.SECTIONS[4].controls)
    assert "/sandbox" in hints and "close" in hints and "forget" in hints
    off = with_values({("browser", "enabled"): False, ("browser", "stream"): False, ("browser", "remember"): False})
    assert su.section_summary(su.SECTIONS[4], off) == "🌐 Browser · off · screenshots ❌ · logins ❌"
    assert all(row_of(su.settings_page_blocks(off), name).buttons[0].text == "⬜ Off" for name in BROWSER_NAMES)


@pytest.mark.parametrize("action, key", [("toggle", "enabled"), ("stream", "stream"), ("remember", "remember")])
def test_apply_setting_browser_toggles(action, key):
    us = UserSettings(str(USER))
    assert su.apply_setting(us, "browser", action) is True
    assert UserSettings(str(USER)).get("browser", key) is False
    assert su.apply_setting(us, "browser", action, "ignored") is True
    assert UserSettings(str(USER)).get("browser", key) is True
    others = {"enabled", "stream", "remember"} - {key}
    assert all(UserSettings(str(USER)).get("browser", k) is True for k in others)


def test_parse_page_callback():
    assert su.parse_page_callback("setp_tools_iteration_30") == ("tools", "iteration", "30")
    assert su.parse_page_callback("setp_followups_toggle") == ("followups", "toggle", None)
    assert su.parse_page_callback("setp_sysprompt_perplexity-deep-research") == ("system_prompt", "set", "perplexity-deep-research")
    assert su.parse_page_callback("setp_preset_fast") == ("preset", "set", "fast")
    assert su.parse_page_callback("setp_ui_advanced") == ("ui", "advanced", None)
    assert su.parse_page_callback("setp_mode_set_speed") == ("mode", "set", "speed")
    assert su.parse_page_callback("settings_tools_toggle") is None


# ---- apply_setting -----------------------------------------------------------
def test_apply_setting_toggles_and_persists():
    us = UserSettings(str(USER))
    assert su.apply_setting(us, "followups", "toggle") is True
    assert su.apply_setting(us, "trace", "toggle", "ignored") is True
    assert su.apply_setting(us, "image", "metadata") is True
    again = UserSettings(str(USER))
    assert again.get("follow_ups", "enabled") is False
    assert again.get("trace", "keep_summary") is False
    assert again.get("image", "metadata") is False


@pytest.mark.parametrize("token, action, value, category, key, expected", [
    ("summarization", "size", "999", "summarization_history", "size", 50),
    ("dialog", "size", "0", "dialog_history", "size", 1),
    ("tools", "iteration", "0", "tools", "max_iteration", 1),
    ("critique", "iteration", "1000", "critique", "max_iteration", 300),
    ("semantic", "max", "50", "semantic_search", "max_results", 20),
    ("transcript", "ctx", "1", "transcript", "max_context_tokens", 20000),
    ("transcript", "keep", "99", "transcript", "keep_tool_results_turns", 50),
    ("transcript", "keep", "0", "transcript", "keep_tool_results_turns", 0),
    ("image", "variants", "9", "image", "max_variants", IMAGE_MAX_VARIANTS),
    ("image", "variants", "0", "image", "max_variants", 1),
])
def test_apply_setting_clamps(token, action, value, category, key, expected):
    us = UserSettings(str(USER))
    us.set(category, 2, key)  # start away from both ends, so every clamp is a change
    assert su.apply_setting(us, token, action, value) is True
    assert UserSettings(str(USER)).get(category, key) == expected


@pytest.mark.parametrize("token, action, value", [
    ("image", "engine", "dalle"),
    ("image", "quality", "ultra"),
    ("image", "size", "8K"),
    ("system_prompt", "set", "evil"),
    ("tools", "iteration", "abc"),
    ("tools", "iteration", None),
    ("tools", "explode", "1"),
    ("nope", "toggle", None),
    (None, None, None),
    ("tools", "iteration", "20"),  # the current value: nothing changes
    ("image", "engine", "auto"),
    ("mode", "set", "warp"),
    ("mode", "set", "auto"),
    ("preset", "set", "balanced"),  # the active preset
    ("preset", "set", "turbo"),
    ("preset", "toggle", "fast"),
])
def test_apply_setting_rejects_unknown_and_unchanged(token, action, value):
    us = UserSettings(str(USER))
    before = us.as_dict()
    assert su.apply_setting(us, token, action, value) is False
    assert us.as_dict() == before


def test_apply_setting_choices():
    us = UserSettings(str(USER))
    assert su.apply_setting(us, "image", "engine", "fast") is True
    assert su.apply_setting(us, "system_prompt", "set", "perplexity-r1") is True
    assert su.apply_setting(us, "mode", "set", "speed") is True
    assert su.apply_setting(us, "ui", "advanced") is True
    again = UserSettings(str(USER))
    assert again.get("image", "engine") == "fast"
    assert again.get("system_prompt", "type") == "perplexity-r1"
    assert again.get("mode", "type") == "speed"
    assert again.get("ui", "advanced") is True


def test_apply_setting_applies_a_preset():
    us = UserSettings(str(USER))
    assert su.apply_setting(us, "preset", "set", "deep") is True
    again = UserSettings(str(USER))
    assert (again.get("mode", "type"), again.get("thinking", "enabled"), again.get("tools", "max_iteration")) == ("intelligence", True, 40)
    assert current_preset(again) == "deep"


def test_every_control_addresses_a_default_setting():
    for (token, action), control in su.CONTROLS.items():
        assert control.key in DEFAULT_SETTINGS[control.category], (token, action)
        if control is su.ADVANCED:
            continue  # page state in the first row, in no section
        assert su.SECTION_OF_TOKEN[token] in {s.id for s in su.SECTIONS}
    assert su.SECTION_OF_TOKEN[su.PRESET_TOKEN] == "agent"


# ---- /settings ---------------------------------------------------------------
async def test_settings_command_sends_one_rich_page(ui):
    bot = make_bot()
    await su.settings_command(make_message(), bot)
    assert ui == []
    bot.send_rich_message.assert_awaited_once()
    kw = bot.send_rich_message.await_args.kwargs
    assert kw["chat_id"] == USER and "message_thread_id" not in kw
    page = kw["rich_message"]
    assert isinstance(page, InputRichMessage) and page.markdown is None and page.html is None
    assert len(details(page.blocks)) == 5 and not any(d.is_open for d in details(page.blocks))


async def test_settings_command_in_a_forum_topic(ui):
    bot = make_bot()
    await su.settings_command(make_message(thread_id=5), bot)
    assert bot.send_rich_message.await_args.kwargs["message_thread_id"] == 5


async def test_settings_command_retries_flat_and_remembers_it(ui):
    bot = make_bot(send_errors=[bad_request()])
    await su.settings_command(make_message(), bot)
    assert ui == []
    first, second = (c.kwargs["rich_message"] for c in bot.send_rich_message.await_args_list)
    assert details(first.blocks) and not details(second.blocks)
    assert su._page_state["flat"] is True and rich.buttons_available()
    # The next page starts flat: one call, no sections.
    bot = make_bot()
    await su.settings_command(make_message(), bot)
    bot.send_rich_message.assert_awaited_once()
    assert not details(bot.send_rich_message.await_args.kwargs["rich_message"].blocks)


async def test_settings_command_falls_back_to_the_keyboards(ui):
    bot = make_bot(send_errors=[bad_request(), bad_request("Bad Request: unknown block type")])
    await su.settings_command(make_message(), bot)
    assert bot.send_rich_message.await_count == 2
    assert not rich.buttons_available() and rich.is_available()
    assert su._page_state["flat"] is False  # the sections were not what failed
    assert [c[0] for c in ui] == ["answer_md"] and ui[0][1].startswith("*Settings Overview*")
    # Buttons are off for the process now: straight to the keyboards.
    bot = make_bot()
    await su.settings_command(make_message(), bot)
    bot.send_rich_message.assert_not_awaited()
    assert [c[0] for c in ui] == ["answer_md", "answer_md"]


async def test_settings_command_without_rich_support(ui):
    bot = make_bot(send_errors=[TelegramNotFound(method=_M, message="Not Found")])
    await su.settings_command(make_message(), bot)
    assert not rich.is_available()
    assert [c[0] for c in ui] == ["answer_md"]


async def test_settings_command_rich_messages_off_uses_the_keyboards(ui):
    UserSettings(str(USER)).set("rich_messages", False, "enabled")
    bot = make_bot()
    await su.settings_command(make_message(), bot)
    bot.send_rich_message.assert_not_awaited()
    text, keyboard = ui[0][1], ui[0][2]
    assert "🧭 *Follow-ups*: ✅" in text and "🎛 *Mode*: auto" in text
    datas = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "settings_followups" in datas and "settings_summarization" in datas and "settings_mode" in datas
    assert [b.callback_data for b in keyboard.inline_keyboard[0]] == [f"settings_preset_{n}" for n in PRESET_NAMES]
    assert [b.text for b in keyboard.inline_keyboard[0]] == ["⚡ Fast", "✅ ⚖️ Balanced", "🧠 Deep"]


# ---- page buttons ------------------------------------------------------------
async def test_page_button_applies_and_edits_in_place(ui):
    bot = make_bot()
    cb = make_callback("setp_tools_iteration_30", make_message(message_id=55))
    await su.settings_page_button(cb, bot)
    cb.answer.assert_awaited_once()
    assert UserSettings(str(USER)).get("tools", "max_iteration") == 30
    kw = bot.edit_message_text.await_args.kwargs
    assert (kw["chat_id"], kw["message_id"]) == (USER, 55)
    sections = details(kw["rich_message"].blocks)
    assert [bool(d.is_open) for d in sections] == [False, True, False, False, False]
    assert "tools ✅ 30" in sections[1].summary
    assert ui == []


@pytest.mark.parametrize("data, open_index, check", [
    ("setp_followups_toggle", 2, lambda s: s.get("follow_ups", "enabled") is False),
    ("setp_sysprompt_perplexity-r1", 1, lambda s: s.get("system_prompt", "type") == "perplexity-r1"),
    ("setp_image_size_4K", 3, lambda s: s.get("image", "size") == "4K"),
    ("setp_summarization_size_10", 0, lambda s: s.get("summarization_history", "size") == 10),
    ("setp_browser_remember", 4, lambda s: s.get("browser", "remember") is False),
])
async def test_page_button_opens_the_changed_section(ui, data, open_index, check):
    bot = make_bot()
    await su.settings_page_button(make_callback(data), bot)
    assert check(UserSettings(str(USER)))
    sections = details(bot.edit_message_text.await_args.kwargs["rich_message"].blocks)
    assert [i for i, d in enumerate(sections) if d.is_open] == [open_index]


async def test_page_preset_button_applies_the_preset_and_opens_the_agent_section(ui):
    bot = make_bot()
    await su.settings_page_button(make_callback("setp_preset_fast"), bot)
    s = UserSettings(str(USER))
    assert (s.get("mode", "type"), s.get("thinking", "enabled"), s.get("tools", "max_iteration")) == ("speed", False, 10)
    blocks = bot.edit_message_text.await_args.kwargs["rich_message"].blocks
    assert [(b.text, b.style) for b in blocks[3].buttons[:3]] == [("✅ ⚡ Fast", "primary"), ("⚖️ Balanced", None), ("🧠 Deep", None)]
    assert [bool(d.is_open) for d in details(blocks)] == [False, True, False, False, False]
    assert "speed · thinking ❌ · tools ✅ 10" in details(blocks)[1].summary
    assert ui == []


async def test_page_advanced_button_shows_every_setting(ui):
    bot = make_bot()
    await su.settings_page_button(make_callback("setp_ui_advanced"), bot)
    assert UserSettings(str(USER)).get("ui", "advanced") is True
    blocks = bot.edit_message_text.await_args.kwargs["rich_message"].blocks
    assert blocks[3].buttons[3].text == "⚙️ Advanced ✅"
    assert names_on(blocks) == ["Preset"] + [c.name for c in SECTION_CONTROLS]
    assert not any(d.is_open for d in details(blocks))
    await su.settings_page_button(make_callback("setp_ui_advanced"), bot)
    assert UserSettings(str(USER)).get("ui", "advanced") is False
    blocks = bot.edit_message_text.await_args.kwargs["rich_message"].blocks
    assert blocks[3].buttons[3].text == "⚙️ Advanced ⬜" and names_on(blocks) == ["Preset"] + [c.name for c in BASIC]


async def test_page_button_not_modified_is_success(ui):
    bot = make_bot(edit_errors=[bad_request("Bad Request: message is not modified")])
    await su.settings_page_button(make_callback("setp_image_engine_dalle"), bot)  # rejected value: same page
    bot.edit_message_text.assert_awaited_once()
    assert ui == []


async def test_page_button_edit_failure_sends_the_keyboards(ui):
    bot = make_bot(edit_errors=[bad_request("Bad Request: message can't be edited")] * 2)
    await su.settings_page_button(make_callback("setp_thinking_toggle"), bot)
    assert bot.edit_message_text.await_count == 2  # sections, then flat
    assert UserSettings(str(USER)).get("thinking", "enabled") is False  # the change itself is kept
    assert [c[0] for c in ui] == ["answer_md"] and "🤔 *Thinking Status*: ❌" in ui[0][1]
    assert rich.buttons_available() and su._page_state["flat"] is False  # an edit failure is about that message


async def test_page_button_ignores_inaccessible_messages(ui):
    bot = make_bot()
    cb = SimpleNamespace(data="setp_thinking_toggle", message=None, answer=AsyncMock())
    await su.settings_page_button(cb, bot)
    cb.answer.assert_awaited_once()
    bot.edit_message_text.assert_not_awaited()
    assert UserSettings(str(USER)).get("thinking", "enabled") is True


# ---- keyboards (old settings messages keep working) ---------------------------
async def test_keyboard_toggle(ui):
    await su.settings_button(make_callback("settings_thinking_toggle"))
    assert UserSettings(str(USER)).get("thinking", "enabled") is False
    assert ui[-1][0] == "edit_md" and ui[-1][1].startswith("*Thinking Status Settings*") and "Disabled" in ui[-1][1]


async def test_keyboard_numbers(ui):
    await su.settings_button(make_callback("settings_tools_iteration_7"))
    assert UserSettings(str(USER)).get("tools", "max_iteration") == 7
    assert ui[-1][1].startswith("*Tools Settings*")
    await su.settings_button(make_callback("settings_tools_iteration"))  # no value: the choice list
    assert ui[-1][1].startswith("*Select Tools Max Iterations*")
    assert UserSettings(str(USER)).get("tools", "max_iteration") == 7
    await su.settings_button(make_callback("settings_summarization_size_999"))
    assert UserSettings(str(USER)).get("summarization_history", "size") == 50
    await su.settings_button(make_callback("settings_transcript_ctx"))
    assert ui[-1][1].startswith("*Max context (tokens)*")


async def test_keyboard_image_choices(ui):
    await su.settings_button(make_callback("settings_image_size"))
    assert ui[-1][1].startswith("*Size*")  # image size is a choice list, not the history-size menu
    await su.settings_button(make_callback("settings_image_size_4K"))
    assert UserSettings(str(USER)).get("image", "size") == "4K" and ui[-1][1].startswith("*Image Settings*")
    await su.settings_button(make_callback("settings_image_engine_dalle"))
    assert UserSettings(str(USER)).get("image", "engine") == "auto" and ui[-1][1].startswith("*Image Settings*")


async def test_keyboard_system_prompt(ui):
    await su.settings_button(make_callback("settings_system_prompt"))
    assert ui[-1][1].startswith("*System Prompt Settings*")
    await su.settings_button(make_callback("settings_system_prompt_set_generall-ai-v1"))
    assert UserSettings(str(USER)).get("system_prompt", "type") == "generall-ai-v1"
    assert "Current Type: *generall-ai-v1*" in ui[-1][1]
    await su.settings_button(make_callback("settings_system_prompt_set_bogus"))
    assert UserSettings(str(USER)).get("system_prompt", "type") == "generall-ai-v1"


async def test_keyboard_followups(ui):
    await su.settings_button(make_callback("settings_followups"))
    text, keyboard = ui[-1][1], ui[-1][2]
    assert text.startswith("*Follow-up Suggestions*") and "buttons under answers" in text
    assert keyboard.inline_keyboard[0][0].callback_data == "settings_followups_toggle"
    await su.settings_button(make_callback("settings_followups_toggle"))
    assert UserSettings(str(USER)).get("follow_ups", "enabled") is False
    assert "Status: Disabled" in ui[-1][1]


async def test_keyboard_browser_menu(ui):
    await su.settings_button(make_callback("settings_main"))
    text, keyboard = ui[-1][1], ui[-1][2]
    assert "🌐 *Browser*: ✅ | Screenshots: ✅ | Logins: ✅" in text
    assert "settings_browser" in [b.callback_data for row in keyboard.inline_keyboard for b in row]
    await su.settings_button(make_callback("settings_browser"))
    text, keyboard = ui[-1][1], ui[-1][2]
    assert text.startswith("*Browser Settings*") and "/sandbox" in text
    datas = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert datas == ["settings_browser_toggle", "settings_browser_stream", "settings_browser_remember", "settings_main"]
    assert [row[0].text for row in keyboard.inline_keyboard[:3]] == [
        "✅ Interactive browser", "✅ Screenshots while browsing", "✅ Remember logins"]
    for data, key in (("settings_browser_stream", "stream"), ("settings_browser_remember", "remember"), ("settings_browser_toggle", "enabled")):
        await su.settings_button(make_callback(data))
        assert UserSettings(str(USER)).get("browser", key) is False
        assert ui[-1][1].startswith("*Browser Settings*")
    text, keyboard = ui[-1][1], ui[-1][2]
    assert "Interactive browser: Disabled" in text and "Remember logins: Disabled" in text
    assert keyboard.inline_keyboard[0][0].text == "❌ Interactive browser"
    await su.settings_button(make_callback("settings_browser_explode"))  # unknown action: just the menu again
    assert UserSettings(str(USER)).get("browser") == {"enabled": False, "stream": False, "remember": False}


async def test_keyboard_main_and_unknown(ui):
    await su.settings_button(make_callback("settings_main"))
    assert ui[-1][1].startswith("*Settings Overview*")
    count = len(ui)
    await su.settings_button(make_callback("settings_bogus_toggle"))
    assert len(ui) == count


async def test_keyboard_mode_menu(ui):
    await su.settings_button(make_callback("settings_mode"))
    text, keyboard = ui[-1][1], ui[-1][2]
    assert text.startswith("*Mode Settings*") and "Current: *auto*" in text
    datas = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert datas == [f"settings_mode_set_{m}" for m in MODE_TYPES] + ["settings_main"]
    assert keyboard.inline_keyboard[0][0].text == "✅ Auto"
    await su.settings_button(make_callback("settings_mode_set_speed"))
    assert UserSettings(str(USER)).get("mode", "type") == "speed"
    assert "Current: *speed*" in ui[-1][1] and ui[-1][2].inline_keyboard[1][0].text == "✅ Speed"
    await su.settings_button(make_callback("settings_mode_set"))  # no value: the menu is the choice list
    assert ui[-1][1].startswith("*Mode Settings*")
    await su.settings_button(make_callback("settings_mode_set_warp"))
    assert UserSettings(str(USER)).get("mode", "type") == "speed"


async def test_keyboard_presets(ui):
    await su.settings_button(make_callback("settings_preset_deep"))
    s = UserSettings(str(USER))
    assert (s.get("mode", "type"), s.get("thinking", "enabled"), s.get("tools", "max_iteration")) == ("intelligence", True, 40)
    kind, text, keyboard = ui[-1]
    assert kind == "edit_md" and text.startswith("*Settings Overview*") and "🎛 *Mode*: intelligence" in text
    assert [b.text for b in keyboard.inline_keyboard[0]] == ["⚡ Fast", "⚖️ Balanced", "✅ 🧠 Deep"]
    count = len(ui)
    await su.settings_button(make_callback("settings_preset_turbo"))
    assert UserSettings(str(USER)).as_dict() == s.as_dict() and len(ui) == count + 1  # still the overview
