"""``/settings`` command and the ``settings_*`` inline-keyboard family.

Ported from the python-telegram-bot handlers in ``main_bot.py``. Every
user-visible text, button label and ``callback_data`` string is preserved so
the UI behaves the same; authorization is handled by ``AuthMiddleware``.

callback_data scheme: ``settings_<token>[_<action>[_<value>]]`` where
``<token>`` is a short name (summarization, dialog, reasoning, memory,
critique, judge, tools, semantic, thinking, rich, transcript, trace, main), parsed with a plain
``split("_")``. ``settings_system_prompt`` and
``settings_system_prompt_set_<type>`` are special-cased because of the
underscore in their token.
"""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.settings import SYSTEM_PROMPT_TYPES, UserSettings
from bot.ui import answer_md, edit_md

logger = logging.getLogger(__name__)

router = Router(name="settings")

SYSTEM_PROMPT_SET_PREFIX = "settings_system_prompt_set_"

SIZE_CHOICES = (1, 5, 10, 20, 30, 50)
ITERATION_CHOICES = (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 80, 100, 150, 200, 300)
SEMANTIC_MAX_RESULTS_CHOICES = (1, 3, 5, 7, 10, 15, 20)
CONTEXT_TOKEN_CHOICES = (50000, 100000, 120000, 150000, 200000, 300000)
KEEP_TOOL_RESULTS_CHOICES = (1, 2, 3, 5, 10)

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


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Non-numeric settings callback value: %r", value)
        return None


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
        "✨ *Rich Messages*: "
        f"{_mark(s.get('rich_messages', 'enabled'))}\n"
        "🧵 *Transcript*: "
        f"{_mark(s.get('transcript', 'enabled'))} | "
        f"Context: {int(s.get('transcript', 'max_context_tokens') or 0) // 1000}k\n"
        "🧾 *Turn Summary*: "
        f"{_mark(s.get('trace', 'keep_summary'))}\n"
        "🧩 *System Prompt*: "
        f"{s.get('system_prompt', 'type')}\n\n"
        "Select a setting to configure:"
    )

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
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
            ],
        ]
    )
    return status_text, keyboard


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
@router.message(F.text, Command("settings"))
async def settings_command(message: Message) -> None:
    """Handle the /settings command."""
    user_id = str(message.chat.id)
    user_settings = UserSettings(user_id)
    status_text, keyboard = _overview(user_settings)
    await answer_md(message, status_text, keyboard)


@router.callback_query(F.data.startswith("settings_"))
async def settings_button(callback: CallbackQuery) -> None:
    """Handle settings menu button presses."""
    await callback.answer()
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
            user_settings.set("system_prompt", prompt_type, "type")
        else:
            logger.warning("Unknown system prompt type requested by %s: %r", user_id, prompt_type)
        await show_system_prompt_menu(message, user_settings)
        return

    # Regular settings navigation: settings_<token>[_<action>[_<value>]]
    parts = data.split("_")
    category = parts[1] if len(parts) > 1 else None
    action = parts[2] if len(parts) > 2 else None
    value = parts[3] if len(parts) > 3 else None
    logger.debug("Parsed settings callback - category: %s, action: %s, value: %s", category, action, value)

    if category == "summarization":
        if action == "toggle":
            current = user_settings.get("summarization_history", "enabled")
            user_settings.set("summarization_history", not current, "enabled")
            await show_summarization_menu(message, user_settings)
        elif action == "size":
            number = _to_int(value)
            if number is not None:
                size = user_settings.validate_size(number)
                user_settings.set("summarization_history", size, "size")
                await show_summarization_menu(message, user_settings)
            else:
                await show_size_input_menu(message, "summarization")
        else:
            await show_summarization_menu(message, user_settings)

    elif category == "dialog":
        if action == "toggle":
            current = user_settings.get("dialog_history", "enabled")
            user_settings.set("dialog_history", not current, "enabled")
            await show_dialog_menu(message, user_settings)
        elif action == "size":
            number = _to_int(value)
            if number is not None:
                size = user_settings.validate_size(number)
                user_settings.set("dialog_history", size, "size")
                await show_dialog_menu(message, user_settings)
            else:
                await show_size_input_menu(message, "dialog")
        else:
            await show_dialog_menu(message, user_settings)

    elif category == "reasoning":
        if action == "toggle":
            current = user_settings.get("reasoning_context", "enabled")
            user_settings.set("reasoning_context", not current, "enabled")
        await show_reasoning_menu(message, user_settings)

    elif category == "memory":
        if action == "toggle":
            current = user_settings.get("short_term_memory", "enabled")
            user_settings.set("short_term_memory", not current, "enabled")
        await show_memory_menu(message, user_settings)

    elif category == "critique":
        if action == "toggle":
            current = user_settings.get("critique", "enabled")
            user_settings.set("critique", not current, "enabled")
            await show_critique_menu(message, user_settings)
        elif action == "iteration":
            number = _to_int(value)
            if number is not None:
                iteration = user_settings.validate_iteration(number, "critique")
                user_settings.set("critique", iteration, "max_iteration")
                await show_critique_menu(message, user_settings)
            else:
                await show_iteration_input_menu(message, "critique")
        else:
            await show_critique_menu(message, user_settings)

    elif category == "judge":
        if action == "toggle":
            current = user_settings.get("judge", "enabled")
            user_settings.set("judge", not current, "enabled")
            await show_judge_menu(message, user_settings)
        elif action == "iteration":
            number = _to_int(value)
            if number is not None:
                iteration = user_settings.validate_iteration(number, "judge")
                user_settings.set("judge", iteration, "max_iteration")
                await show_judge_menu(message, user_settings)
            else:
                await show_iteration_input_menu(message, "judge")
        else:
            await show_judge_menu(message, user_settings)

    elif category == "tools":
        if action == "toggle":
            current = user_settings.get("tools", "enabled")
            user_settings.set("tools", not current, "enabled")
            await show_tools_menu(message, user_settings)
        elif action == "iteration":
            number = _to_int(value)
            if number is not None:
                iteration = user_settings.validate_iteration(number, "tools")
                user_settings.set("tools", iteration, "max_iteration")
                await show_tools_menu(message, user_settings)
            else:
                await show_iteration_input_menu(message, "tools")
        else:
            await show_tools_menu(message, user_settings)

    elif category == "main":
        status_text, keyboard = _overview(user_settings)
        await edit_md(message, status_text, keyboard)

    elif category == "semantic":
        if action == "toggle":
            current = user_settings.get("semantic_search", "enabled")
            user_settings.set("semantic_search", not current, "enabled")
            await show_semantic_menu(message, user_settings)
        elif action == "max":
            number = _to_int(value)
            if number is not None:
                max_results = user_settings.validate_semantic_max_results(number)
                user_settings.set("semantic_search", max_results, "max_results")
                await show_semantic_menu(message, user_settings)
            else:
                await show_semantic_max_results_menu(message)
        else:
            await show_semantic_menu(message, user_settings)

    elif category == "thinking":
        if action == "toggle":
            current = user_settings.get("thinking", "enabled")
            user_settings.set("thinking", not current, "enabled")
        await show_thinking_menu(message, user_settings)

    elif category == "rich":
        if action == "toggle":
            current = user_settings.get("rich_messages", "enabled")
            user_settings.set("rich_messages", not current, "enabled")
        await show_rich_menu(message, user_settings)

    elif category == "trace":
        if action == "toggle":
            current = user_settings.get("trace", "keep_summary")
            user_settings.set("trace", not current, "keep_summary")
        await show_trace_menu(message, user_settings)

    elif category == "transcript":
        if action == "toggle":
            current = user_settings.get("transcript", "enabled")
            user_settings.set("transcript", not current, "enabled")
            await show_transcript_menu(message, user_settings)
        elif action == "ctx":
            number = _to_int(value)
            if number is not None:
                user_settings.set("transcript", user_settings.validate_context_tokens(number), "max_context_tokens")
                await show_transcript_menu(message, user_settings)
            else:
                await show_transcript_choice_menu(message, "ctx")
        elif action == "keep":
            number = _to_int(value)
            if number is not None:
                user_settings.set("transcript", max(0, min(50, number)), "keep_tool_results_turns")
                await show_transcript_menu(message, user_settings)
            else:
                await show_transcript_choice_menu(message, "keep")
        else:
            await show_transcript_menu(message, user_settings)

    else:
        logger.debug("Ignoring unknown settings callback: %s", data)


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
