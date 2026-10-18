"""``/settings``: one rich page with buttons inside it, or the ``settings_*`` inline keyboards.

Both UIs change settings through :func:`apply_setting`, so they share every clamp and check.

Settings page (``rich_messages`` on, and the Bot API server takes rich messages with buttons):
``/settings`` sends ONE rich message. A first row holds the presets "⚡ Fast · ⚖️ Balanced ·
🧠 Deep" (``bot.settings.SETTINGS_PRESETS``; the one matching the current values is marked)
and the "⚙️ Advanced" toggle (``ui.advanced``): off, the power settings (``Control.advanced``:
transcript sizes, dialog/reasoning/summaries, critique, judge, semantic max, system prompt,
image variants/metadata) are hidden and only the section summaries show their values. Five
collapsible sections (Memory, Agent, Replies, Images, Browser) hold every setting under a friendly
name as a bold paragraph followed by a row of buttons inside the message
(``InputRichBlockButtons``): a toggle is one "✅ On" / "⬜ Off" button, a number is
"−  value  +", a choice is one "<current> ▸" button that cycles. A tap applies the change and
edits the page in place with that setting's section open; there is no submenu navigation.
callback_data is ``setp_<token>_<action>[_<value>]`` with the same tokens, actions and values
as the keyboards below, plus ``setp_sysprompt_<type>`` (types contain hyphens, so it is
parsed by prefix), ``setp_preset_<name>`` and ``setp_ui_advanced``. A server that refuses the
collapsible sections gets the same content flat (sections as bold paragraphs) and later pages
start flat; when that is refused too (and the error is not about the chat itself),
``rich.mark_buttons_unavailable`` sends everyone to the keyboards.

Keyboards (rich messages off, or the page was refused): ported from the python-telegram-bot
handlers in ``main_bot.py``. Every user-visible text, button label and ``callback_data``
string is preserved, so settings messages already sitting in chats keep working;
authorization is handled by ``AuthMiddleware``.

Keyboard callback_data scheme: ``settings_<token>[_<action>[_<value>]]`` where
``<token>`` is a short name (summarization, dialog, reasoning, memory,
critique, judge, tools, semantic, thinking, mode, rich, transcript, trace, image, followups,
browser, main), parsed with a plain ``split("_")``. ``settings_system_prompt`` and
``settings_system_prompt_set_<type>`` are special-cased because of the
underscore in their token; ``settings_preset_<name>`` applies a preset and shows the overview.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    DisabledButton,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputRichBlockButtons,
    InputRichBlockDetails,
    InputRichBlockParagraph,
    InputRichMessage,
    Message,
    RichMessageButton,
    RichTextBold,
    RichTextItalic,
)

from bot import rich
from bot.settings import (
    DEFAULT_SETTINGS,
    MODE_TYPES,
    PRESET_NAMES,
    SYSTEM_PROMPT_TYPES,
    UserSettings,
    apply_preset,
    current_preset,
)
from models import IMAGE_BACKENDS, IMAGE_MAX_VARIANTS, IMAGE_QUALITIES, IMAGE_SIZES
from bot.ui import answer_md, edit_md, is_not_modified

logger = logging.getLogger(__name__)

router = Router(name="settings")

SYSTEM_PROMPT_SET_PREFIX = "settings_system_prompt_set_"
PRESET_SET_PREFIX = "settings_preset_"
PAGE_PREFIX = "setp_"
PAGE_SYSTEM_PROMPT_PREFIX = "setp_sysprompt_"
PAGE_PRESET_PREFIX = "setp_preset_"
PRESET_TOKEN = "preset"  # callback token of the presets row; not a Control (it sets several values)
PAGE_HINT = "Tap a button to change a setting; this page updates in place. Advanced shows every setting."
CALLBACK_DATA_LIMIT = 64  # bytes, Bot API
ROW_BUTTON_LIMIT = 8  # InputRichBlockButtons holds 1-8 buttons

PRESET_LABELS: dict[str, str] = {"fast": "⚡ Fast", "balanced": "⚖️ Balanced", "deep": "🧠 Deep"}
PRESET_HINT = "⚡ Fast: quickest model · ⚖️ Balanced: picks per message · 🧠 Deep: full reasoning every time"
MODE_HINT = "Auto: picks per message · Speed: fastest model · Intelligence: full reasoning every time"
MODE_DESCRIPTIONS: dict[str, str] = {
    "auto": "a quick model answers simple messages, the full agent takes the rest",
    "speed": "the fastest model runs every message, tools included, with light thinking",
    "intelligence": "the main model with deep reasoning on every message",
}

SIZE_CHOICES = (1, 5, 10, 20, 30, 50)
ITERATION_CHOICES = (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 80, 100, 150, 200, 300)
SEMANTIC_MAX_RESULTS_CHOICES = (1, 3, 5, 7, 10, 15, 20)
CONTEXT_TOKEN_CHOICES = (50000, 100000, 120000, 150000, 200000, 300000)
KEEP_TOOL_RESULTS_CHOICES = (1, 2, 3, 5, 10)

# Button text of a choice on the settings page, when the stored value is not readable as is.
CHOICE_LABELS: dict[tuple[str, str], dict[str, str]] = {
    ("system_prompt", "type"): {
        "generall-ai-v2": "GenerALL v2",
        "generall-ai-v1": "GenerALL v1",
        "perplexity-deep-research": "Deep research",
        "perplexity-r1": "Perplexity R1",
    },
}
CHOICE_ROW_BUTTONS = 4  # at most this many choice buttons side by side ...
CHOICE_ROW_CHARS = 42  # ... and buttons share a row's width, so the longest label × buttons stays under this

SYSTEM_PROMPT_DESCRIPTIONS: dict[str, str] = {
    "generall-ai-v2": "Generall.AI v2 system prompt.",
    "generall-ai-v1": "Generall.AI v1 system prompt.",
    "perplexity-deep-research": "Perplexity Deep Research system prompt.",
    "perplexity-r1": "Perplexity R1 system prompt.",
}


# ---------------------------------------------------------------------------
# Keyboard helpers
# ---------------------------------------------------------------------------
def _mark(enabled: bool) -> str:
    return "✅" if enabled else "❌"


def _status(enabled: bool) -> str:
    return "Enabled" if enabled else "Disabled"


def _toggle_button(enabled: bool, callback_data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=f"{_mark(enabled)} Enabled", callback_data=callback_data)


def _back_button(callback_data: str = "settings_main") -> InlineKeyboardButton:
    return InlineKeyboardButton(text="⬅️ Back", callback_data=callback_data)


def _choice_rows(values: tuple[int, ...], callback_prefix: str, per_row: int = 3) -> list[list[InlineKeyboardButton]]:
    """Lay out numeric choice buttons ``per_row`` per row; callback is ``<prefix><value>``."""
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for value in values:
        row.append(InlineKeyboardButton(text=str(value), callback_data=f"{callback_prefix}{value}"))
        if len(row) == per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Non-numeric settings callback value: %r", value)
        return None


def _clamp(low: int, high: int) -> Callable[[int], int]:
    return lambda number: max(low, min(high, int(number)))


def _fmt_k(tokens: int) -> str:
    return f"{tokens // 1000}k" if tokens % 1000 == 0 else f"{tokens / 1000:.1f}k"


def _clip(value: Any, limit: int = 24) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(user_settings: UserSettings, category: str, key: str) -> int:
    """A numeric setting as int; a hand-edited non-number reads as the default."""
    try:
        return int(user_settings.get(category, key))
    except (TypeError, ValueError):
        return int(DEFAULT_SETTINGS[category][key])


# ---------------------------------------------------------------------------
# What can be changed, and how (shared by the page and the keyboards)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Control:
    """One changeable value, addressed the way callbacks address it: ``token`` + ``action``.

    ``kind`` is ``toggle`` (flip a bool), ``number`` (any integer, clamped by ``clamp`` as the
    keyboards always did; ``choices`` are only the −/+ steps of the page) or ``choice`` (one of
    ``choices``: the page shows every choice as its own button, the current one checked). ``name``/``hint``
    are the page's paragraph (friendly words, not the
    JSON key). ``advanced`` controls are on the page only while ``ui.advanced`` is on.
    """

    token: str
    action: str
    category: str  # key in DEFAULT_SETTINGS
    key: str
    kind: str
    name: str
    hint: str = ""
    choices: tuple = ()
    clamp: Callable[[int], int] | None = None
    label: Callable[[int], str] = str
    advanced: bool = False


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    controls: tuple[Control, ...]
    state: Callable[[UserSettings], str]  # compact state shown after the title


def _memory_state(s: UserSettings) -> str:
    return (
        f"transcript {_mark(s.get('transcript', 'enabled'))} {_fmt_k(_number(s, 'transcript', 'max_context_tokens'))}"
        f" · summaries {_mark(s.get('summarization_history', 'enabled'))} {_number(s, 'summarization_history', 'size')}"
    )


def _agent_state(s: UserSettings) -> str:
    return (
        f"{_clip(s.get('mode', 'type'))}"
        f" · thinking {_mark(s.get('thinking', 'enabled'))}"
        f" · tools {_mark(s.get('tools', 'enabled'))} {_number(s, 'tools', 'max_iteration')}"
        f" · {_clip(s.get('system_prompt', 'type'))}"
    )


def _replies_state(s: UserSettings) -> str:
    return (
        f"rich {_mark(s.get('rich_messages', 'enabled'))}"
        f" · summary {_mark(s.get('trace', 'keep_summary'))}"
        f" · follow-ups {_mark(s.get('follow_ups', 'enabled'))}"
    )


def _images_state(s: UserSettings) -> str:
    return (
        f"{_clip(s.get('image', 'engine'))} · {_clip(s.get('image', 'quality'))} · {_clip(s.get('image', 'size'))}"
        f" · ×{_number(s, 'image', 'max_variants')}"
    )


def _browser_state(s: UserSettings) -> str:
    return (
        f"{'on' if s.get('browser', 'enabled') else 'off'}"
        f" · screenshots {_mark(s.get('browser', 'stream'))}"
        f" · logins {_mark(s.get('browser', 'remember'))}"
    )


SECTIONS: tuple[Section, ...] = (
    Section("memory", "🧠 Memory", (
        Control("transcript", "toggle", "transcript", "enabled", "toggle",
                "🧵 Remember this chat in full", "the real conversation, tool calls included"),
        Control("transcript", "ctx", "transcript", "max_context_tokens", "number",
                "Memory window", "tokens kept before the oldest turns are summarized",
                CONTEXT_TOKEN_CHOICES, UserSettings.validate_context_tokens, _fmt_k, advanced=True),
        Control("transcript", "keep", "transcript", "keep_tool_results_turns", "number",
                "Keep tool results (turns)", "turns whose tool output stays in full",
                KEEP_TOOL_RESULTS_CHOICES, _clamp(0, 50), advanced=True),
        Control("summarization", "toggle", "summarization_history", "enabled", "toggle",
                "📚 Recent summaries in memory", "short summaries of recent conversations"),
        Control("summarization", "size", "summarization_history", "size", "number",
                "Summaries kept", "how many recent ones", SIZE_CHOICES, UserSettings.validate_size, advanced=True),
        Control("dialog", "toggle", "dialog_history", "enabled", "toggle",
                "💬 Dialog history", "question/answer window, used when the full chat is off", advanced=True),
        Control("dialog", "size", "dialog_history", "size", "number",
                "Dialog size", "question/answer pairs kept", SIZE_CHOICES, UserSettings.validate_size, advanced=True),
        Control("reasoning", "toggle", "reasoning_context", "enabled", "toggle",
                "🧠 Reasoning context", "last turn's reasoning (full chat off) and the reasoning file", advanced=True),
        Control("memory", "toggle", "short_term_memory", "enabled", "toggle",
                "💭 Short-term memory", "master switch for summaries, dialog and reasoning", advanced=True),
        Control("semantic", "toggle", "semantic_search", "enabled", "toggle",
                "🔍 Recall similar past chats", "related conversations found by meaning"),
        Control("semantic", "max", "semantic_search", "max_results", "number",
                "Recalled chats per message", "how many related conversations are added",
                SEMANTIC_MAX_RESULTS_CHOICES, UserSettings.validate_semantic_max_results, advanced=True),
    ), _memory_state),
    Section("agent", "🤖 Agent", (
        Control("mode", "set", "mode", "type", "choice",
                "🎛 Quick answers vs. deep work", MODE_HINT, MODE_TYPES),
        Control("thinking", "toggle", "thinking", "enabled", "toggle",
                "🤔 Deep reasoning", "think longer before answering; off = faster, lighter answers"),
        Control("tools", "toggle", "tools", "enabled", "toggle",
                "🛠️ Tools", "files, web, code, images and more"),
        Control("tools", "iteration", "tools", "max_iteration", "number",
                "Tool calls per answer", "the most tool calls one message may take",
                ITERATION_CHOICES, UserSettings.validate_iteration),
        Control("critique", "toggle", "critique", "enabled", "toggle",
                "🎯 Second opinion rewrite (GPT)", "a second model reviews each answer", advanced=True),
        Control("critique", "iteration", "critique", "max_iteration", "number",
                "Rewrite rounds", "max rewrites", ITERATION_CHOICES, UserSettings.validate_iteration, advanced=True),
        Control("judge", "toggle", "judge", "enabled", "toggle",
                "⚖️ Completeness check", "re-prompts until the answer is complete", advanced=True),
        Control("judge", "iteration", "judge", "max_iteration", "number",
                "Check rounds", "max re-prompts", ITERATION_CHOICES, UserSettings.validate_iteration, advanced=True),
        Control("system_prompt", "set", "system_prompt", "type", "choice",
                "🧩 Personality prompt", "the system prompt the assistant follows", SYSTEM_PROMPT_TYPES, advanced=True),
    ), _agent_state),
    Section("replies", "💬 Replies", (
        Control("rich", "toggle", "rich_messages", "enabled", "toggle",
                "✨ Rich formatting", "native formatting and this page; off = classic Markdown and menus"),
        Control("trace", "toggle", "trace", "keep_summary", "toggle",
                "🧾 Show work summary", "tool calls, thinking and tokens above each answer"),
        Control("followups", "toggle", "follow_ups", "enabled", "toggle",
                "🧭 Suggested next questions", "as buttons under answers"),
    ), _replies_state),
    Section("images", "🎨 Images", (
        Control("image", "engine", "image", "engine", "choice",
                "🖌 Image engine", "auto/best strongest, fast quicker, story for narratives", tuple(sorted(IMAGE_BACKENDS))),
        Control("image", "quality", "image", "quality", "choice",
                "✨ Image quality", "effort per image; auto lets the provider decide", tuple(IMAGE_QUALITIES)),
        Control("image", "size", "image", "size", "choice",
                "📐 Image size", "resolution tier", tuple(IMAGE_SIZES)),
        Control("image", "variants", "image", "max_variants", "number",
                "🔢 Images per request", "the most alternatives one request may produce",
                tuple(range(1, IMAGE_MAX_VARIANTS + 1)), _clamp(1, IMAGE_MAX_VARIANTS), advanced=True),
        Control("image", "metadata", "image", "metadata", "toggle",
                "🏷 Caption details", "model, quality and size in each caption", advanced=True),
    ), _images_state),
    Section("browser", "🌐 Browser", (
        Control("browser", "toggle", "browser", "enabled", "toggle",
                "🌐 Interactive browser", "I open sites, click, type and log in for you; /sandbox closes it"),
        Control("browser", "stream", "browser", "stream", "toggle",
                "📸 Show screenshots while browsing", "a live view of the page as I work"),
        Control("browser", "remember", "browser", "remember", "toggle",
                "🔑 Remember logins", "keep cookies between sessions; /sandbox forgets them"),
    ), _browser_state),
)

# The "⚙️ Advanced" toggle of the first row: a Control (apply_setting, the keyboards' parser),
# but page state rather than a setting of a section.
ADVANCED = Control("ui", "advanced", "ui", "advanced", "toggle", "⚙️ Advanced", "show every setting")

CONTROLS: dict[tuple[str, str], Control] = {(c.token, c.action): c for s in SECTIONS for c in s.controls}
CONTROLS[(ADVANCED.token, ADVANCED.action)] = ADVANCED
SECTION_OF_TOKEN: dict[str, str] = {c.token: s.id for s in SECTIONS for c in s.controls}
SECTION_OF_TOKEN[PRESET_TOKEN] = "agent"  # a preset changes the mode, thinking and tool calls


def visible_controls(section: Section, user_settings: UserSettings) -> list[Control]:
    """The controls of ``section`` on the page: all of them with ``ui.advanced`` on, else the basic ones."""
    advanced = bool(user_settings.get("ui", "advanced"))
    return [c for c in section.controls if advanced or not c.advanced]


def apply_setting(user_settings: UserSettings, category: str | None, action: str | None, value: Any = None) -> bool:
    """Apply one change the way a callback addresses it; True when a stored value changed.

    ``category`` is the callback token (``summarization``, ``tools``, ``image``, ``followups``,
    ``system_prompt``, ``mode``, ``ui`` ...), not the JSON key. Toggles flip and ignore
    ``value``; numbers are clamped like the keyboards always did (sizes 1..50, iterations
    1..300, search results 1..20, context 20k..400k tokens, kept tool results 0..50, variants
    1..IMAGE_MAX_VARIANTS); choices must be one of the allowed values. ``("preset", "set",
    <name>)`` applies a whole preset (``bot.settings.apply_preset``). Unknown tokens/actions,
    non-numeric numbers and unknown choices change nothing and return False, as does setting
    the current value (or the active preset) again.
    """
    if category == PRESET_TOKEN:
        return apply_preset(user_settings, value) if action == "set" else False
    control = CONTROLS.get((category or "", action or ""))
    if control is None:
        return False
    current = user_settings.get(control.category, control.key)
    if control.kind == "toggle":
        user_settings.set(control.category, not current, control.key)
        return True
    if control.kind == "number":
        number = _to_int(value)
        if number is None or control.clamp is None:
            return False
        new = control.clamp(number)
    else:
        if value not in control.choices:
            logger.warning("Unknown %s %s value: %r", control.token, control.action, value)
            return False
        new = value
    if new == current:
        return False
    user_settings.set(control.category, new, control.key)
    return True


# ---------------------------------------------------------------------------
# Main overview (shared by /settings and the "settings_main" back button)
# ---------------------------------------------------------------------------
def _overview(user_settings: UserSettings) -> tuple[str, InlineKeyboardMarkup]:
    s = user_settings
    status_text = (
        "*Settings Overview*\n\n"
        "📚 *Summarization*: "
        f"{_mark(s.get('summarization_history', 'enabled'))} | "
        f"Size: {s.get('summarization_history', 'size')}\n"
        "💬 *Dialog History*: "
        f"{_mark(s.get('dialog_history', 'enabled'))} | "
        f"Size: {s.get('dialog_history', 'size')}\n"
        "🧠 *Reasoning Context*: "
        f"{_mark(s.get('reasoning_context', 'enabled'))}\n"
        "💭 *Short Term Memory*: "
        f"{_mark(s.get('short_term_memory', 'enabled'))}\n"
        "🎯 *Critique*: "
        f"{_mark(s.get('critique', 'enabled'))} | "
        f"Max: {s.get('critique', 'max_iteration')}\n"
        "⚖️ *Judge*: "
        f"{_mark(s.get('judge', 'enabled'))} | "
        f"Max: {s.get('judge', 'max_iteration')}\n"
        "🛠️ *Tools*: "
        f"{_mark(s.get('tools', 'enabled'))} | "
        f"Max: {s.get('tools', 'max_iteration')}\n"
        "🔍 *Semantic Search*: "
        f"{_mark(s.get('semantic_search', 'enabled'))} | "
        f"Max Results: {s.get('semantic_search', 'max_results')}\n"
        "🤔 *Thinking Status*: "
        f"{_mark(s.get('thinking', 'enabled'))}\n"
        "🎛 *Mode*: "
        f"{s.get('mode', 'type')}\n"
        "✨ *Rich Messages*: "
        f"{_mark(s.get('rich_messages', 'enabled'))}\n"
        "🧵 *Transcript*: "
        f"{_mark(s.get('transcript', 'enabled'))} | "
        f"Context: {int(s.get('transcript', 'max_context_tokens') or 0) // 1000}k\n"
        "🧾 *Turn Summary*: "
        f"{_mark(s.get('trace', 'keep_summary'))}\n"
        "🧭 *Follow-ups*: "
        f"{_mark(s.get('follow_ups', 'enabled'))}\n"
        "🎨 *Images*: "
        f"{s.get('image', 'engine')} | "
        f"Quality: {s.get('image', 'quality')} | "
        f"{s.get('image', 'size')}\n"
        "🧩 *System Prompt*: "
        f"{s.get('system_prompt', 'type')}\n"
        "🌐 *Browser*: "
        f"{_mark(s.get('browser', 'enabled'))} | "
        f"Screenshots: {_mark(s.get('browser', 'stream'))} | "
        f"Logins: {_mark(s.get('browser', 'remember'))}\n\n"
        "Select a setting to configure:"
    )

    active = current_preset(s)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{'✅ ' if name == active else ''}{PRESET_LABELS[name]}",
                    callback_data=f"{PRESET_SET_PREFIX}{name}",
                )
                for name in PRESET_NAMES
            ],
            [
                InlineKeyboardButton(text="📚 Summarization", callback_data="settings_summarization"),
                InlineKeyboardButton(text="💬 Dialog", callback_data="settings_dialog"),
            ],
            [
                InlineKeyboardButton(text="🧠 Reasoning", callback_data="settings_reasoning"),
                InlineKeyboardButton(text="💭 Memory", callback_data="settings_memory"),
            ],
            [
                InlineKeyboardButton(text="🎯 Critique", callback_data="settings_critique"),
                InlineKeyboardButton(text="⚖️ Judge", callback_data="settings_judge"),
            ],
            [
                InlineKeyboardButton(text="🛠️ Tools", callback_data="settings_tools"),
                InlineKeyboardButton(text="🔍 Semantic", callback_data="settings_semantic"),
            ],
            [
                InlineKeyboardButton(text="🤔 Thinking", callback_data="settings_thinking"),
                InlineKeyboardButton(text="🧩 System Prompt", callback_data="settings_system_prompt"),
            ],
            [
                InlineKeyboardButton(text="✨ Rich Messages", callback_data="settings_rich"),
                InlineKeyboardButton(text="🧵 Transcript", callback_data="settings_transcript"),
            ],
            [
                InlineKeyboardButton(text="🧾 Turn Summary", callback_data="settings_trace"),
                InlineKeyboardButton(text="🎨 Images", callback_data="settings_image"),
            ],
            [
                InlineKeyboardButton(text="🧭 Follow-ups", callback_data="settings_followups"),
                InlineKeyboardButton(text="🎛 Mode", callback_data="settings_mode"),
            ],
            [InlineKeyboardButton(text="🌐 Browser", callback_data="settings_browser")],
        ]
    )
    return status_text, keyboard


# ---------------------------------------------------------------------------
# Rich settings page
# ---------------------------------------------------------------------------
# The server refused the collapsible sections once (and took the flat page): later pages start flat.
_page_state: dict[str, bool] = {"flat": False}


def page_callback(control: Control, value: Any = None) -> str:
    """callback_data of a page button: ``setp_<token>_<action>[_<value>]`` / ``setp_sysprompt_<type>``."""
    if control.token == "system_prompt":
        return f"{PAGE_SYSTEM_PROMPT_PREFIX}{value}"
    data = f"{PAGE_PREFIX}{control.token}_{control.action}"
    return data if value is None else f"{data}_{value}"


def preset_callback(name: str) -> str:
    """callback_data of a preset button: ``setp_preset_<name>``."""
    return f"{PAGE_PRESET_PREFIX}{name}"


def parse_page_callback(data: str) -> tuple[str, str | None, str | None] | None:
    """``(token, action, value)`` for :func:`apply_setting`, or None when ``data`` is not a page callback."""
    if data.startswith(PAGE_SYSTEM_PROMPT_PREFIX):
        return "system_prompt", "set", data[len(PAGE_SYSTEM_PROMPT_PREFIX):]
    if data.startswith(PAGE_PRESET_PREFIX):
        return PRESET_TOKEN, "set", data[len(PAGE_PRESET_PREFIX):]
    if not data.startswith(PAGE_PREFIX):
        return None
    parts = data[len(PAGE_PREFIX):].split("_")
    return parts[0], (parts[1] if len(parts) > 1 else None), (parts[2] if len(parts) > 2 else None)


def step_values(choices: Sequence[int], current: int) -> tuple[int | None, int | None]:
    """The choices just below and just above ``current``; None at either end of the list.

    A value that is not in the list (an older choice list, a hand-edited file) snaps onto the
    nearest choice in the tapped direction, so a tap always changes the value and never skips one.
    """
    lower = max((c for c in choices if c < current), default=None)
    higher = min((c for c in choices if c > current), default=None)
    return lower, higher


def choice_label(control: Control, value: Any) -> str:
    return CHOICE_LABELS.get((control.category, control.key), {}).get(str(value), _clip(value))


def split_rows(labels: Sequence[str], max_buttons: int = CHOICE_ROW_BUTTONS, max_chars: int = CHOICE_ROW_CHARS) -> list[int]:
    """How many buttons go on each row: as few rows as fit, then spread evenly (six choices become
    3 + 3, not 4 + 2), keeping the order. Buttons in a row share its width, so a row fits when its
    longest label (with the "✅ " of the checked one) times its button count stays under ``max_chars``."""
    count = len(labels)
    for rows in range(1, count + 1):
        base, extra = divmod(count, rows)
        sizes = [base + (1 if i < extra else 0) for i in range(rows)]
        start, fits = 0, True
        for size in sizes:
            widest = max(len(label) for label in labels[start:start + size]) + 2
            if size > max_buttons or (size > 1 and widest * size > max_chars):
                fits = False
                break
            start += size
        if fits:
            return sizes
    return [1] * count

def _disabled(text: str) -> RichMessageButton:
    return RichMessageButton(text=text, disabled=DisabledButton())


def _step_button(text: str, control: Control, target: int | None) -> RichMessageButton:
    if target is None:
        return _disabled(text)
    return RichMessageButton(text=text, callback_data=page_callback(control, target))


def control_rows(control: Control, user_settings: UserSettings) -> list[InputRichBlockButtons]:
    """The button row(s) of one setting, showing its current value.

    A toggle is one button, a number is "−  value  +", a choice is one button per choice (several
    rows when they do not fit one), the current one checked and not tappable, so every option is
    visible and one tap picks it. A stored value that is not in the list (an older choice, a
    hand-edited file) leaves every button tappable.
    """
    current = user_settings.get(control.category, control.key)
    if control.kind == "toggle":
        button = RichMessageButton(
            text="✅ On" if current else "⬜ Off",
            style="success" if current else None,
            callback_data=page_callback(control),
        )
        return [InputRichBlockButtons(buttons=[button])]
    if control.kind == "number":
        value = _number(user_settings, control.category, control.key)
        lower, higher = step_values(control.choices, value)
        return [InputRichBlockButtons(buttons=[
            _step_button("−", control, lower),
            _disabled(control.label(value)),
            _step_button("+", control, higher),
        ])]
    labels = [choice_label(control, choice) for choice in control.choices]
    buttons = [
        _disabled(f"✅ {label}") if choice == current else RichMessageButton(text=label, callback_data=page_callback(control, choice))
        for choice, label in zip(control.choices, labels)
    ]
    rows, start = [], 0
    for size in split_rows(labels):
        rows.append(InputRichBlockButtons(buttons=buttons[start:start + size]))
        start += size
    return rows


def _control_paragraph(control: Control) -> InputRichBlockParagraph:
    text: list = [RichTextBold(text=control.name)]
    if control.hint:
        text.append(f" — {control.hint}")
    return InputRichBlockParagraph(text=text)


def section_summary(section: Section, user_settings: UserSettings) -> str:
    return f"{section.title} · {section.state(user_settings)}"


def preset_row(user_settings: UserSettings) -> InputRichBlockButtons:
    """The first row of the page: the presets (the active one checked and primary) and the Advanced toggle."""
    active = current_preset(user_settings)
    buttons = [
        RichMessageButton(
            text=f"✅ {PRESET_LABELS[name]}" if name == active else PRESET_LABELS[name],
            style="primary" if name == active else None,
            callback_data=preset_callback(name),
        )
        for name in PRESET_NAMES
    ]
    advanced = bool(user_settings.get(ADVANCED.category, ADVANCED.key))
    buttons.append(RichMessageButton(
        text=f"{ADVANCED.name} {'✅' if advanced else '⬜'}",
        callback_data=page_callback(ADVANCED),
    ))
    return InputRichBlockButtons(buttons=buttons)


def settings_page_blocks(user_settings: UserSettings, *, open_section: str | None = None, sections: bool = True) -> list:
    """Blocks of the settings page.

    ``sections``: every section is a collapsible block, only ``open_section`` (a ``Section.id``)
    open; False gives the flat layout (same content, each section headed by a bold paragraph).
    Advanced controls are left out while ``ui.advanced`` is off (:func:`visible_controls`).
    """
    blocks: list = [
        InputRichBlockParagraph(text=RichTextBold(text="⚙️ Settings")),
        InputRichBlockParagraph(text=RichTextItalic(text=PAGE_HINT)),
        InputRichBlockParagraph(text=[RichTextBold(text="Preset"), f" — {PRESET_HINT}"]),
        preset_row(user_settings),
    ]
    for section in SECTIONS:
        inner: list = []
        for control in visible_controls(section, user_settings):
            inner.append(_control_paragraph(control))
            inner.extend(control_rows(control, user_settings))
        summary = section_summary(section, user_settings)
        if sections:
            is_open = True if section.id == open_section else None
            blocks.append(InputRichBlockDetails(summary=summary, is_open=is_open, blocks=inner))
        else:
            blocks.append(InputRichBlockParagraph(text=RichTextBold(text=summary)))
            blocks.extend(inner)
    return blocks


async def _show_page(
    deliver: Callable[[InputRichMessage], Awaitable[Any]],
    user_settings: UserSettings,
    open_section: str | None,
    *,
    fresh: bool,
) -> bool:
    """Send or edit (``deliver``) the page; True when it is on screen. Never raises.

    The collapsible layout goes first unless it was refused before; a refused one is retried
    flat once. For a ``fresh`` /settings message a refused flat page means the buttons are the
    problem, so they are switched off for the process, unless the error is about the chat
    (``rich.is_chat_error``: topic closed, no rights, ...). An edit of a page that was
    delivered before fails for reasons of its own (message too old, deleted) and only falls back.
    """
    layouts = (False,) if _page_state["flat"] else (True, False)
    for sectioned in layouts:
        try:
            page = InputRichMessage(blocks=settings_page_blocks(user_settings, open_section=open_section, sections=sectioned))
            await deliver(page)
        except TelegramAPIError as e:
            if is_not_modified(e):
                return True
            if rich.is_unsupported_error(e):
                rich.mark_unavailable(str(e))
                return False
            if not isinstance(e, TelegramBadRequest):
                logger.warning("Settings page failed: %s", e)
                return False
            if sectioned:
                logger.warning("Settings page with sections refused (%s); retrying flat", e)
                continue
            if fresh and not rich.is_chat_error(e):
                rich.mark_buttons_unavailable(str(e))
            else:
                logger.warning("Settings page refused for this message: %s", e)
            return False
        except Exception as e:  # noqa: BLE001 - network hiccup or a bad payload: the keyboards still work
            logger.warning("Settings page failed: %s", e)
            return False
        if not sectioned and len(layouts) > 1:
            _page_state["flat"] = True
        return True
    return False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
@router.message(F.text, Command("settings"))
async def settings_command(message: Message, bot: Bot) -> None:
    """Handle the /settings command: the rich page when possible, else the keyboard overview."""
    user_id = str(message.chat.id)
    user_settings = UserSettings(user_id)
    if user_settings.get("rich_messages", "enabled") and rich.is_available() and rich.buttons_available():
        kwargs: dict[str, Any] = {"chat_id": message.chat.id}
        if message.is_topic_message and message.message_thread_id:
            kwargs["message_thread_id"] = message.message_thread_id

        async def send(page: InputRichMessage) -> None:
            await bot.send_rich_message(rich_message=page, **kwargs)

        if await _show_page(send, user_settings, None, fresh=True):
            return
    status_text, keyboard = _overview(user_settings)
    await answer_md(message, status_text, keyboard)


async def _answer(callback: CallbackQuery) -> None:
    """Stop the button's spinner; best-effort, so a query that is too old never loses the tap."""
    try:
        await callback.answer()
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not answer the settings button: %s", e)


@router.callback_query(F.data.startswith(PAGE_PREFIX))
async def settings_page_button(callback: CallbackQuery, bot: Bot) -> None:
    """A button inside the rich settings page: apply it and redraw the page in place."""
    await _answer(callback)
    message = callback.message
    if not isinstance(message, Message):
        return

    user_id = str(message.chat.id)
    user_settings = UserSettings(user_id)
    data = callback.data or ""
    token, action, value = parse_page_callback(data) or ("", None, None)
    changed = apply_setting(user_settings, token, action, value)
    logger.debug("Settings page callback from %s: %s (changed: %s)", user_id, data, changed)

    if rich.is_available():
        async def edit(page: InputRichMessage) -> None:
            await bot.edit_message_text(chat_id=message.chat.id, message_id=message.message_id, rich_message=page)

        if await _show_page(edit, user_settings, SECTION_OF_TOKEN.get(token), fresh=False):
            return
    logger.warning("Settings page of %s could not be updated; sending the classic menu", user_id)
    status_text, keyboard = _overview(user_settings)
    await answer_md(message, status_text, keyboard)


@router.callback_query(F.data.startswith("settings_"))
async def settings_button(callback: CallbackQuery) -> None:
    """Handle settings menu button presses."""
    await _answer(callback)
    message = callback.message
    if not isinstance(message, Message):
        return

    user_id = str(message.chat.id)
    user_settings = UserSettings(user_id)
    data = callback.data or ""
    logger.debug("Settings callback from %s: %s", user_id, data)

    # Special cases first: the system_prompt token contains an underscore.
    if data == "settings_system_prompt":
        await show_system_prompt_menu(message, user_settings)
        return
    if data.startswith(SYSTEM_PROMPT_SET_PREFIX):
        prompt_type = data[len(SYSTEM_PROMPT_SET_PREFIX):]
        if prompt_type in SYSTEM_PROMPT_TYPES:
            logger.info("Setting system prompt type for %s to: %s", user_id, prompt_type)
        apply_setting(user_settings, "system_prompt", "set", prompt_type)  # warns about an unknown type
        await show_system_prompt_menu(message, user_settings)
        return
    if data.startswith(PRESET_SET_PREFIX):
        name = data[len(PRESET_SET_PREFIX):]
        if not apply_setting(user_settings, PRESET_TOKEN, "set", name):
            logger.debug("Preset %s of %s: nothing to change", name, user_id)
        status_text, keyboard = _overview(user_settings)
        await edit_md(message, status_text, keyboard)
        return

    # Regular settings navigation: settings_<token>[_<action>[_<value>]]
    parts = data.split("_")
    category = parts[1] if len(parts) > 1 else None
    action = parts[2] if len(parts) > 2 else None
    value = parts[3] if len(parts) > 3 else None
    logger.debug("Parsed settings callback - category: %s, action: %s, value: %s", category, action, value)

    if category == "main":
        status_text, keyboard = _overview(user_settings)
        await edit_md(message, status_text, keyboard)
        return

    menu = _LEGACY_MENUS.get(category or "")
    if menu is None:
        logger.debug("Ignoring unknown settings callback: %s", data)
        return

    # A number or choice without a usable value opens its choice list; everything else is a
    # change (or a no-op for an unknown action) followed by the category's menu.
    control = CONTROLS.get((category or "", action or ""))
    if control is not None and (
        (control.kind == "number" and _to_int(value) is None) or (control.kind == "choice" and not value)
    ):
        await _show_choice_menu(message, control, user_settings)
        return
    apply_setting(user_settings, category, action, value)
    await menu(message, user_settings)


async def _show_choice_menu(message: Message, control: Control, user_settings: UserSettings) -> None:
    """The keyboards' choice list for one number / choice (e.g. ``settings_tools_iteration``)."""
    if control.token == "image":
        await show_image_choice_menu(message, control.action)
    elif control.token == "transcript":
        await show_transcript_choice_menu(message, control.action)
    elif control.token == "semantic":
        await show_semantic_max_results_menu(message)
    elif control.token == "mode":
        await show_mode_menu(message, user_settings)  # its menu is the choice list
    elif control.action == "size":
        await show_size_input_menu(message, control.token)
    elif control.action == "iteration":
        await show_iteration_input_menu(message, control.token)


# ---------------------------------------------------------------------------
# Sub-menus
# ---------------------------------------------------------------------------
async def show_summarization_menu(message: Message, user_settings: UserSettings) -> None:
    """Show summarization history settings menu."""
    enabled = user_settings.get("summarization_history", "enabled")
    size = user_settings.get("summarization_history", "size")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_summarization_toggle")],
            [InlineKeyboardButton(text=f"📊 History Size: {size}", callback_data="settings_summarization_size")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Summarization History Settings*\n"
        f"Status: {_status(enabled)}\n"
        f"History Size: {size} entries",
        keyboard,
    )


async def show_dialog_menu(message: Message, user_settings: UserSettings) -> None:
    """Show dialog history settings menu."""
    enabled = user_settings.get("dialog_history", "enabled")
    size = user_settings.get("dialog_history", "size")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_dialog_toggle")],
            [InlineKeyboardButton(text=f"📊 History Size: {size}", callback_data="settings_dialog_size")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Dialog History Settings*\n"
        f"Status: {_status(enabled)}\n"
        f"History Size: {size} entries",
        keyboard,
    )


async def show_reasoning_menu(message: Message, user_settings: UserSettings) -> None:
    """Show reasoning context settings menu."""
    enabled = user_settings.get("reasoning_context", "enabled")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_reasoning_toggle")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Reasoning Context Settings*\n"
        f"Status: {_status(enabled)}",
        keyboard,
    )


async def show_memory_menu(message: Message, user_settings: UserSettings) -> None:
    """Show short term memory settings menu."""
    enabled = user_settings.get("short_term_memory", "enabled")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_memory_toggle")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Short Term Memory Settings*\n"
        f"Status: {_status(enabled)}",
        keyboard,
    )


async def show_size_input_menu(message: Message, category: str) -> None:
    """Show size input menu (summarization / dialog history size)."""
    rows = _choice_rows(SIZE_CHOICES, f"settings_{category}_size_")
    rows.append([_back_button(f"settings_{category}")])
    await edit_md(
        message,
        f"*Select {category.title()} History Size*\n"
        "Choose the number of entries to keep:",
        InlineKeyboardMarkup(inline_keyboard=rows),
    )


async def show_critique_menu(message: Message, user_settings: UserSettings) -> None:
    """Show critique settings menu."""
    enabled = user_settings.get("critique", "enabled")
    max_iteration = user_settings.get("critique", "max_iteration")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_critique_toggle")],
            [InlineKeyboardButton(text=f"🔄 Max Iterations: {max_iteration}", callback_data="settings_critique_iteration")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Critique Settings*\n"
        f"Status: {_status(enabled)}\n"
        f"Max Iterations: {max_iteration}",
        keyboard,
    )


async def show_judge_menu(message: Message, user_settings: UserSettings) -> None:
    """Show judge settings menu."""
    enabled = user_settings.get("judge", "enabled")
    max_iteration = user_settings.get("judge", "max_iteration")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_judge_toggle")],
            [InlineKeyboardButton(text=f"🔄 Max Iterations: {max_iteration}", callback_data="settings_judge_iteration")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Judge Settings*\n"
        f"Status: {_status(enabled)}\n"
        f"Max Iterations: {max_iteration}",
        keyboard,
    )


async def show_tools_menu(message: Message, user_settings: UserSettings) -> None:
    """Show tools settings menu."""
    enabled = user_settings.get("tools", "enabled")
    max_iteration = user_settings.get("tools", "max_iteration")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_tools_toggle")],
            [InlineKeyboardButton(text=f"🔄 Max Iterations: {max_iteration}", callback_data="settings_tools_iteration")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Tools Settings*\n"
        f"Status: {_status(enabled)}\n"
        f"Max Iterations: {max_iteration}",
        keyboard,
    )


async def show_iteration_input_menu(message: Message, category: str) -> None:
    """Show iteration input menu (critique / judge / tools max iterations)."""
    rows = _choice_rows(ITERATION_CHOICES, f"settings_{category}_iteration_")
    rows.append([_back_button(f"settings_{category}")])
    await edit_md(
        message,
        f"*Select {category.title()} Max Iterations*\n"
        "Choose the maximum number of iterations:",
        InlineKeyboardMarkup(inline_keyboard=rows),
    )


async def show_semantic_menu(message: Message, user_settings: UserSettings) -> None:
    """Show semantic search settings menu."""
    semantic_enabled = user_settings.get("semantic_search", "enabled")
    max_results = user_settings.get("semantic_search", "max_results")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(semantic_enabled, "settings_semantic_toggle")],
            [InlineKeyboardButton(text=f"🔍 Max Results: {max_results}", callback_data="settings_semantic_max")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Semantic Search Settings*\n\n"
        f"Status: {_status(semantic_enabled)}\n"
        f"Max Results: {max_results}\n\n"
        "This feature enables semantic search over your conversation history\n"
        "to find relevant past interactions.",
        keyboard,
    )


async def show_semantic_max_results_menu(message: Message) -> None:
    """Show menu for setting max results in semantic search."""
    rows = _choice_rows(SEMANTIC_MAX_RESULTS_CHOICES, "settings_semantic_max_")
    rows.append([_back_button("settings_semantic")])
    await edit_md(
        message,
        "*Select Maximum Results*\n\n"
        "Choose the maximum number of past conversations\n"
        "to include in semantic search:",
        InlineKeyboardMarkup(inline_keyboard=rows),
    )


async def show_thinking_menu(message: Message, user_settings: UserSettings) -> None:
    """Show thinking status settings menu."""
    enabled = user_settings.get("thinking", "enabled")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_thinking_toggle")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Thinking Status Settings*\n\n"
        f"Status: {_status(enabled)}\n\n"
        "This setting controls whether the bot shows detailed\n"
        "thinking steps during processing.",
        keyboard,
    )


async def show_mode_menu(message: Message, user_settings: UserSettings) -> None:
    """Quick answers vs. deep work: one button per mode (``settings_mode_set_<value>``)."""
    current = user_settings.get("mode", "type")
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(
            text=f"{'✅' if mode == current else '○'} {mode.title()}",
            callback_data=f"settings_mode_set_{mode}",
        )]
        for mode in MODE_TYPES
    ]
    rows.append([_back_button()])
    lines = "\n".join(f"• *{mode.title()}*: {MODE_DESCRIPTIONS[mode]}" for mode in MODE_TYPES)
    await edit_md(
        message,
        "*Mode Settings*\n\n"
        f"Current: *{current}*\n\n"
        f"{lines}\n\n"
        "Quick answers vs. deep work:",
        InlineKeyboardMarkup(inline_keyboard=rows),
    )


async def show_rich_menu(message: Message, user_settings: UserSettings) -> None:
    """Show rich messages settings menu."""
    enabled = user_settings.get("rich_messages", "enabled")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_rich_toggle")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Rich Messages Settings*\n\n"
        f"Status: {_status(enabled)}\n\n"
        "Answers are sent as rich Telegram messages with real\n"
        "Markdown: headings, tables, code blocks, task lists and math.\n"
        "Turn this off if your Telegram app shows them as\n"
        "\"unsupported message\" (old client); answers then use\n"
        "classic Markdown formatting.",
        keyboard,
    )


async def show_image_menu(message: Message, user_settings: UserSettings) -> None:
    """Show image generation settings menu."""
    s = user_settings
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"🖌 Engine: {s.get('image', 'engine')}", callback_data="settings_image_engine")],
            [InlineKeyboardButton(text=f"✨ Quality: {s.get('image', 'quality')}", callback_data="settings_image_quality")],
            [InlineKeyboardButton(text=f"📐 Size: {s.get('image', 'size')}", callback_data="settings_image_size")],
            [InlineKeyboardButton(text=f"🔢 Max variants: {s.get('image', 'max_variants')}", callback_data="settings_image_variants")],
            [_toggle_button(s.get("image", "metadata"), "settings_image_metadata")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Image Settings*\n\n"
        "Your preferred engine and quality for generated images. The\n"
        "assistant follows them, but may pick a better engine when you\n"
        "ask for the best result, or a faster one for a quick draft.\n\n"
        "`auto`/`best` use the most capable model, `fast` is quicker at\n"
        "the same price with lower quality, `story` illustrates a\n"
        "narrative. The toggle adds the model, quality and size to the\n"
        "caption of every delivered image.",
        keyboard,
    )


async def show_image_choice_menu(message: Message, kind: str) -> None:
    """Choice list for one image setting."""
    if kind == "variants":
        rows = _choice_rows(tuple(range(1, IMAGE_MAX_VARIANTS + 1)), "settings_image_variants_")
        title = "*Max variants*\n\nUpper limit on how many alternatives one request may produce:"
    else:
        values = {"engine": sorted(IMAGE_BACKENDS), "quality": list(IMAGE_QUALITIES), "size": list(IMAGE_SIZES)}[kind]
        rows = [[InlineKeyboardButton(text=v, callback_data=f"settings_image_{kind}_{v}")] for v in values]
        title = {
            "engine": "*Engine*\n\nauto and best use the most capable model; fast is quicker\nat the same price with lower quality; story illustrates a narrative:",
            "quality": "*Quality*\n\nHow much effort to spend. auto lets the provider decide\nfrom the prompt:",
            "size": "*Size*\n\nResolution tier; 4K is slower and is clamped to the\nprovider's limits:",
        }[kind]
    rows.append([_back_button("settings_image")])
    await edit_md(message, title, InlineKeyboardMarkup(inline_keyboard=rows))


async def show_trace_menu(message: Message, user_settings: UserSettings) -> None:
    """Show the turn-summary (status message) settings menu."""
    enabled = user_settings.get("trace", "keep_summary")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_trace_toggle")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Turn Summary Settings*\n\n"
        f"Status: {_status(enabled)}\n\n"
        "While the bot works, the status message shows the tool\n"
        "calls as they run. When the answer is ready the status is\n"
        "shortened into a summary kept above the answer, with\n"
        "expandable tool calls, the model's thinking and the token\n"
        "usage. Off: the status message is deleted instead.",
        keyboard,
    )


async def show_transcript_menu(message: Message, user_settings: UserSettings) -> None:
    """Show transcript (conversation memory) settings menu."""
    enabled = user_settings.get("transcript", "enabled")
    ctx = int(user_settings.get("transcript", "max_context_tokens") or 0)
    keep = user_settings.get("transcript", "keep_tool_results_turns")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_transcript_toggle")],
            [InlineKeyboardButton(text=f"🧮 Max context: {ctx // 1000}k tokens", callback_data="settings_transcript_ctx")],
            [InlineKeyboardButton(text=f"🧹 Keep tool results: {keep} turns", callback_data="settings_transcript_keep")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Transcript Settings*\n\n"
        f"Status: {_status(enabled)}\n\n"
        "The bot keeps one real conversation transcript per chat\n"
        "(or forum topic), including tool calls and their results,\n"
        "so it remembers what it looked up and did. Older tool results\n"
        "are cleared and the oldest turns summarized when the transcript\n"
        "grows past the max context. With Transcript off the classic\n"
        "dialog history / reasoning context memory is used instead.",
        keyboard,
    )


async def show_transcript_choice_menu(message: Message, kind: str) -> None:
    if kind == "ctx":
        rows = [[InlineKeyboardButton(text=f"{v // 1000}k", callback_data=f"settings_transcript_ctx_{v}") for v in CONTEXT_TOKEN_CHOICES[:3]],
                [InlineKeyboardButton(text=f"{v // 1000}k", callback_data=f"settings_transcript_ctx_{v}") for v in CONTEXT_TOKEN_CHOICES[3:]]]
        title = "*Max context (tokens)*\n\nHow large the transcript may grow before it is pruned:"
    else:
        rows = _choice_rows(KEEP_TOOL_RESULTS_CHOICES, "settings_transcript_keep_")
        title = "*Keep tool results*\n\nTool results of the last N turns are kept in full;\nolder ones are cleared to save context:"
    rows.append([_back_button("settings_transcript")])
    await edit_md(message, title, InlineKeyboardMarkup(inline_keyboard=rows))


async def show_system_prompt_menu(message: Message, user_settings: UserSettings) -> None:
    """Show system prompt settings menu."""
    current_type = user_settings.get("system_prompt", "type")
    logger.debug("Showing system prompt menu, current type: %s", current_type)

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=f"{'✅' if current_type == prompt_type else '○'} {prompt_type}",
                callback_data=f"{SYSTEM_PROMPT_SET_PREFIX}{prompt_type}",
            )
        ]
        for prompt_type in SYSTEM_PROMPT_TYPES
    ]
    rows.append([_back_button()])

    description = SYSTEM_PROMPT_DESCRIPTIONS.get(current_type, "No description available")
    await edit_md(
        message,
        "*System Prompt Settings*\n\n"
        f"Current Type: *{current_type}*\n\n"
        f"Description: _{description}_\n\n"
        "Choose a system prompt type:",
        InlineKeyboardMarkup(inline_keyboard=rows),
    )


async def show_followups_menu(message: Message, user_settings: UserSettings) -> None:
    """Show the follow-up suggestions settings menu."""
    enabled = user_settings.get("follow_ups", "enabled")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [_toggle_button(enabled, "settings_followups_toggle")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Follow-up Suggestions*\n\n"
        f"Status: {_status(enabled)}\n\n"
        "Suggested next requests appear as buttons under answers.",
        keyboard,
    )


async def show_browser_menu(message: Message, user_settings: UserSettings) -> None:
    """Show the interactive browser settings menu (three toggles)."""
    enabled = user_settings.get("browser", "enabled")
    stream = user_settings.get("browser", "stream")
    remember = user_settings.get("browser", "remember")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"{_mark(enabled)} Interactive browser", callback_data="settings_browser_toggle")],
            [InlineKeyboardButton(text=f"{_mark(stream)} Screenshots while browsing", callback_data="settings_browser_stream")],
            [InlineKeyboardButton(text=f"{_mark(remember)} Remember logins", callback_data="settings_browser_remember")],
            [_back_button()],
        ]
    )
    await edit_md(
        message,
        "*Browser Settings*\n\n"
        f"Interactive browser: {_status(enabled)}\n"
        f"Screenshots while browsing: {_status(stream)}\n"
        f"Remember logins: {_status(remember)}\n\n"
        "The assistant can open websites in its own browser, click,\n"
        "type and log in for you. It asks you first before buying,\n"
        "paying, sending messages or deleting anything. Screenshots\n"
        "show the page in the chat while it works; remembered logins\n"
        "keep you signed in between sessions. /sandbox closes the\n"
        "browser or forgets the logins.",
        keyboard,
    )


# Keyboard token -> the menu shown after a change (or when the token itself is tapped).
_LEGACY_MENUS: dict[str, Callable[[Message, UserSettings], Awaitable[None]]] = {
    "summarization": show_summarization_menu,
    "dialog": show_dialog_menu,
    "reasoning": show_reasoning_menu,
    "memory": show_memory_menu,
    "critique": show_critique_menu,
    "judge": show_judge_menu,
    "tools": show_tools_menu,
    "semantic": show_semantic_menu,
    "thinking": show_thinking_menu,
    "mode": show_mode_menu,
    "rich": show_rich_menu,
    "image": show_image_menu,
    "trace": show_trace_menu,
    "transcript": show_transcript_menu,
    "followups": show_followups_menu,
    "browser": show_browser_menu,
}
