"""The status message of a turn: live progress, then a compact summary.

While the agent works, the status shows the step, the tool calls as they run and
the token usage. In rich mode it is a rich message edited in place with
expandable blocks; when the turn ends it is edited into a shortened summary
(no "Thinking…" header, a collapsible list of tool calls with their arguments
and results, the model's summarized thinking, tokens and cost) instead of being
deleted. Every rich edit falls back to the legacy-Markdown text version, so an
old Bot API server or client still gets a readable status.

A turn that runs as a queue job gets a Stop button while it runs: a button row
at the end of the rich progress blocks, or an inline keyboard on the text
version (and on rich edits once button rows were refused, see
``rich.buttons_available``). The summary and the stopped/error notices carry no
Stop button, so it disappears when the turn ends. Instead the summary can carry
the turn's action buttons (Retry, Listen, … from ``bot/turn_actions.py``): a row
at its end, with the same inline-keyboard fallbacks.
"""
from __future__ import annotations

import asyncio
import logging
import re

from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputRichBlockButtons,
    InputRichBlockDetails,
    InputRichBlockList,
    InputRichBlockListItem,
    InputRichBlockParagraph,
    InputRichBlockPreformatted,
    InputRichMessage,
    RichTextBold,
    RichTextCode,
    RichTextItalic,
    RichMessageButton,
)

from bot import rich as rich_render
from bot.ui import escape_markdown, is_not_modified

logger = logging.getLogger(__name__)

TRACE_LINES = 10  # most recent tool calls shown in the plain-text status
LIVE_CALLS = 15  # in the rich progress view
SUMMARY_CALLS = 40  # expandable entries in the final summary
SUMMARY_BYTES = 28000  # keep the summary well under the 32 KB rich-message limit
STOP_LABEL = "🛑 Stop"
_ENTITY_UNSAFE = re.compile(r"[_*`\[]")
_HEADER_MARKS = re.compile(r"[*_]")


def entity_safe(text: str) -> str:
    """Text placed INSIDE a legacy-Markdown entity (*bold* / _italic_).

    Legacy Markdown does not allow backslash escapes inside an entity, so the special
    characters are replaced instead of escaped (``run_command`` -> ``run-command``).
    """
    return _ENTITY_UNSAFE.sub("-", str(text))


def plain_header(header: str) -> str:
    """'💭 *Thinking...*' -> '💭 Thinking...' for rich blocks."""
    return _HEADER_MARKS.sub("", header or "").strip()


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 10 else f"{int(round(seconds))}s"


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _icon(call) -> str:
    return "⏳" if call.running else ("✅" if call.ok else "❌")


# ---- text versions (legacy Markdown, also the fallback) ----------------------
def usage_text(trace) -> str:
    """Token usage of the turn so far, with the cached share and the estimated cost."""
    if not trace or not trace.api_calls:
        return ""
    prompt = trace.input_tokens + trace.cache_read_tokens + trace.cache_write_tokens
    text = f"🧮 {trace.api_calls} call{'s' if trace.api_calls != 1 else ''} · in {_fmt_tokens(prompt)}"
    if trace.cache_read_tokens:
        text += f" ({int(round(trace.cache_hit_ratio * 100))}% cached)"
    text += f" · out {_fmt_tokens(trace.output_tokens)}"
    cost = trace.cost_usd
    if cost is not None:
        text += f" · ≈${cost:.3f}" if cost < 0.1 else f" · ≈${cost:.2f}"
    return text


def render_trace(trace, limit: int = TRACE_LINES) -> str:
    """The tool-call list shown under the status header (legacy Markdown)."""
    from agents.trace import describe_args

    if not trace or not trace.calls:
        return ""
    lines = [f"🔧 *Tools* ({trace.total}, {_fmt_seconds(trace.elapsed)})"]
    calls = trace.calls[-limit:]
    if len(trace.calls) > limit:
        lines.append(f"… {len(trace.calls) - limit} earlier")
    for call in calls:
        indent = "  " * call.depth
        args = describe_args(call.args)
        line = f"{indent}{_icon(call)} `{call.name}`"
        if args:
            line += f" {escape_markdown(args)}"
        if not call.running:
            line += f" · {_fmt_seconds(call.duration)}"
        lines.append(line)
    return "\n".join(lines)


def progress_text(header: str, quota: str, step, details, iteration, critique, trace) -> str:
    text = (
        f"{header}\n- - - - \n{quota}"
        f"📝 *Step:* _{entity_safe(step)}_\n"
        f"📋 *Details:* _{entity_safe(details)}_\n"
        f"🔄 *Iterations:* _{iteration}_\n"
        f"🎯 *Critiques:* _{critique}_"
    )
    trace_text = render_trace(trace)
    if trace_text:
        text += "\n\n" + trace_text
    usage = usage_text(trace)
    if usage:
        text += "\n" + usage
    return text


def _headline(trace) -> str:
    if not trace or not trace.calls:
        return f"✅ Done in {_fmt_seconds(trace.elapsed)}" if trace else "✅ Done"
    text = f"🔧 {trace.total} tool call{'s' if trace.total != 1 else ''} in {_fmt_seconds(trace.elapsed)}"
    if trace.errors:
        text += f" · {trace.errors} failed"
    return text


def _names(trace) -> str:
    return ", ".join(f"{name} ×{n}" if n > 1 else name for name, n in trace.counts_by_name())


def trace_summary(trace) -> str:
    """One line (plus usage) kept above the answer once the turn is over (legacy Markdown)."""
    text = f"*{_headline(trace)}*"
    if trace.calls:
        text += f": {escape_markdown(_names(trace))}"
    usage = usage_text(trace)
    return f"{text}\n{usage}" if usage else text


# ---- the Stop button -----------------------------------------------------------
def stop_button_row(callback_data: str) -> InputRichBlockButtons:
    """The Stop button as a rich block (a <tg-button-row>), last block of the live status."""
    return InputRichBlockButtons(buttons=[RichMessageButton(text=STOP_LABEL, style="danger", callback_data=callback_data)])


def stop_keyboard(callback_data: str) -> InlineKeyboardMarkup:
    """The same button as an inline keyboard: text status, or rich edits once button rows were refused."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=STOP_LABEL, style="danger", callback_data=callback_data)]]
    )


# ---- action buttons of a finished turn -------------------------------------------
# An action is (label, callback_data, style or None); see bot.turn_actions.record_actions.
def action_row(actions: list) -> InputRichBlockButtons:
    """Callback buttons as one rich button row (1-8 buttons), e.g. at the end of the summary."""
    return InputRichBlockButtons(buttons=[RichMessageButton(text=text, style=style, callback_data=data) for text, data, style in actions])


def inline_buttons(actions: list) -> list[InlineKeyboardButton]:
    """The same buttons as one inline-keyboard row."""
    return [InlineKeyboardButton(text=text, style=style, callback_data=data) for text, data, style in actions]


def actions_keyboard(actions: list) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[inline_buttons(actions)])


# ---- rich versions -----------------------------------------------------------
def _call_rich_line(call) -> list:
    from agents.trace import describe_args

    parts: list = [("  " * call.depth) + _icon(call) + " ", RichTextCode(text=call.name)]
    args = describe_args(call.args)
    if args:
        parts.append(f" {args}")
    if not call.running:
        parts.append(f" · {_fmt_seconds(call.duration)}")
    return parts


def progress_blocks(header: str, quota: str, step, details, iteration, critique, trace, stop_callback: str | None = None) -> list:
    """Live status blocks; with ``stop_callback`` they end with the Stop button row."""
    blocks: list = [InputRichBlockParagraph(text=[RichTextBold(text=plain_header(header))])]
    meta = f"{quota.strip()} · " if quota and quota.strip() else ""
    blocks.append(InputRichBlockParagraph(text=[RichTextItalic(text=f"{meta}{step} · {details}")]))
    blocks.append(InputRichBlockParagraph(text=f"Iterations: {iteration} · Critiques: {critique}"))
    if trace and trace.calls:
        calls = trace.calls[-LIVE_CALLS:]
        items = [InputRichBlockListItem(blocks=[InputRichBlockParagraph(text=_call_rich_line(c))]) for c in calls]
        summary = f"🔧 Tools ({trace.total}, {_fmt_seconds(trace.elapsed)})"
        if len(trace.calls) > LIVE_CALLS:
            summary += f", last {LIVE_CALLS}"
        blocks.append(InputRichBlockDetails(summary=summary, is_open=True, blocks=[InputRichBlockList(items=items)]))
    usage = usage_text(trace)
    if usage:
        blocks.append(InputRichBlockParagraph(text=usage))
    if stop_callback:
        blocks.append(stop_button_row(stop_callback))
    return blocks


def _summary_blocks(trace, args_chars: int, result_chars: int, max_calls: int) -> list:
    blocks: list = [InputRichBlockParagraph(text=[RichTextBold(text=_headline(trace))] + ([f": {_names(trace)}"] if trace.calls else []))]
    if trace.calls:
        entries = []
        for call in trace.calls[:max_calls]:
            inner: list = []
            args = call.args_text
            if args:
                inner.append(InputRichBlockPreformatted(text=args[:args_chars], language="json"))
            if call.result_excerpt:
                inner.append(InputRichBlockPreformatted(text=call.result_excerpt[:result_chars]))
            if not inner:
                inner.append(InputRichBlockParagraph(text="(no output)"))
            title = f"{'  ' * call.depth}{_icon(call)} {call.name} · {_fmt_seconds(call.duration)}"
            entries.append(InputRichBlockDetails(summary=title, blocks=inner))
        if len(trace.calls) > max_calls:
            entries.append(InputRichBlockParagraph(text=f"… {len(trace.calls) - max_calls} more"))
        blocks.append(InputRichBlockDetails(summary=f"Tool calls ({trace.total})", blocks=entries))
    thinking = trace.thinking_text
    if thinking:
        blocks.append(InputRichBlockDetails(summary="💭 Thinking", blocks=[InputRichBlockParagraph(text=[RichTextItalic(text=thinking)])]))
    usage = usage_text(trace)
    if usage:
        blocks.append(InputRichBlockParagraph(text=usage))
    return blocks


def _approx_bytes(blocks: list) -> int:
    return sum(len(b.model_dump_json(exclude_none=True).encode("utf-8")) for b in blocks)


def summary_blocks(trace) -> list:
    """Final status: shortened, with expandable tool calls and thinking; sized under the limit."""
    from agents.trace import ARGS_CHARS, RESULT_CHARS

    for args_chars, result_chars, max_calls in ((ARGS_CHARS, RESULT_CHARS, SUMMARY_CALLS), (200, 300, SUMMARY_CALLS), (120, 160, 20)):
        blocks = _summary_blocks(trace, args_chars, result_chars, max_calls)
        if _approx_bytes(blocks) <= SUMMARY_BYTES:
            return blocks
    return [InputRichBlockParagraph(text=[RichTextBold(text=_headline(trace))]), InputRichBlockParagraph(text=usage_text(trace) or "")]


# ---- the message -------------------------------------------------------------
class StatusMessage:
    """Wraps the status message of one turn; rich edits with a plain-text fallback.

    Progress updates are coalesced: while one edit is in flight the next one is
    dropped, because the update after it carries newer state anyway. That keeps a
    batch of parallel tool calls from firing a burst of edits at the same message.

    ``stop_callback`` (callback data, e.g. ``queue_stop_<job token>``) adds the Stop
    button to every progress edit; ``finish()`` and ``set_text()`` remove it.
    """

    MAX_PROGRESS_FAILURES = 2  # stop paying for rejected progress edits; finish() still tries

    def __init__(self, sender, message, *, rich: bool, header: str, stop_callback: str | None = None) -> None:
        self.sender = sender
        self.message = message
        self.header = header
        self.stop_callback = stop_callback
        self.rich = bool(rich and message is not None and rich_render.is_available())
        self._lock = asyncio.Lock()
        self._progress_failures = 0

    async def _rich_edit(self, blocks: list, reply_markup: InlineKeyboardMarkup | None) -> None:
        await self.sender.bot.edit_message_text(
            chat_id=self.sender.chat_id,
            message_id=self.message.message_id,
            rich_message=InputRichMessage(blocks=blocks),
            reply_markup=reply_markup,
        )

    async def _edit_rich(
        self,
        blocks: list,
        *,
        progress: bool,
        reply_markup: InlineKeyboardMarkup | None = None,
        buttons_fallback: InlineKeyboardMarkup | None = None,
    ) -> bool:
        """True when the message now shows ``blocks``. Never raises.

        ``buttons_fallback``: ``blocks`` ends with a button row. If the edit is refused it
        is retried once without the row and with this inline keyboard instead; when that
        works, button rows are given up on for the process.
        """
        if not self.rich or self.message is None:
            return False
        try:
            await self._rich_edit(blocks, reply_markup)
            self._progress_failures = 0
            return True
        except TelegramNotFound as e:
            # The server does not know sendRichMessage/rich edits at all.
            rich_render.mark_unavailable(str(e))
            self.rich = False
        except TelegramBadRequest as e:
            if is_not_modified(e):
                return True
            if buttons_fallback is not None and await self._edit_without_buttons(blocks[:-1], buttons_fallback, e):
                return True
            if progress:
                # This particular progress payload was refused; the summary is a
                # different message, so do not give up on rich for the whole turn.
                self._progress_failures += 1
                logger.warning("Rich status update rejected (%s); plain text for this update", e)
            else:
                logger.warning("Rich status summary rejected (%s); falling back to text", e)
                self.rich = False
        except Exception as e:  # noqa: BLE001 - network hiccup: keep rich mode, skip this update
            logger.warning("Rich status edit failed: %s", e)
        return False

    async def _edit_without_buttons(self, blocks: list, keyboard: InlineKeyboardMarkup, error: Exception) -> bool:
        """The edit with a button row was refused: the same blocks without it, the inline keyboard instead."""
        try:
            await self._rich_edit(blocks, keyboard)
        except TelegramBadRequest as e:
            if not is_not_modified(e):
                logger.debug("Rich status refused without the button row too: %s", e)
                return False
        except Exception as e:  # noqa: BLE001 - the original error decides what happens next
            logger.debug("Rich status retry without the button row failed: %s", e)
            return False
        # Only the button row was the problem: later edits use the inline keyboard right away.
        rich_render.mark_buttons_unavailable(str(error))
        self._progress_failures = 0
        return True

    async def update(self, *, step, details, iteration, critique, trace, quota: str = "") -> None:
        if self.message is None or self._lock.locked():
            return  # an edit is already in flight; the next update carries newer state
        async with self._lock:
            keyboard = stop_keyboard(self.stop_callback) if self.stop_callback else None
            if self.rich and self._progress_failures < self.MAX_PROGRESS_FAILURES:
                row = keyboard is not None and rich_render.buttons_available()
                blocks = progress_blocks(
                    self.header, quota, step, details, iteration, critique, trace,
                    stop_callback=self.stop_callback if row else None,
                )
                if row:
                    edited = await self._edit_rich(blocks, progress=True, buttons_fallback=keyboard)
                else:
                    edited = await self._edit_rich(blocks, progress=True, reply_markup=keyboard)
                if edited:
                    return
            await self.sender.edit_text(
                self.message, progress_text(self.header, quota, step, details, iteration, critique, trace),
                fallback="strip", reply_markup=keyboard,
            )

    async def finish(self, trace, *, keep: bool, actions: list | None = None) -> bool:
        """End of a successful turn: shorten into the summary (no Stop button), or delete.

        ``actions`` ((label, callback_data, style) tuples) go on the summary: a button row
        at its end; when the row is refused, the rich summary without it plus an inline
        keyboard (and rows are given up on for the process); when rich fails entirely, the
        text summary with the inline keyboard. Returns True when the actions were placed,
        False when there were none, the status was deleted (``keep`` off) or no edit
        worked, so the caller puts them under the answer instead.
        """
        if self.message is None:
            return False
        async with self._lock:
            if not keep:
                await self.sender.delete([self.message])
                return False
            keyboard = actions_keyboard(actions) if actions else None
            # Always worth one rich attempt: the summary differs from every progress payload.
            if self.rich:
                blocks = summary_blocks(trace)
                if keyboard is not None and rich_render.buttons_available():
                    edited = await self._edit_rich(blocks + [action_row(actions)], progress=False, buttons_fallback=keyboard)
                else:
                    edited = await self._edit_rich(blocks, progress=False, reply_markup=keyboard)
                if edited:
                    return keyboard is not None
            edited_text = await self.sender.edit_text(self.message, trace_summary(trace), fallback="strip", reply_markup=keyboard)
            return keyboard is not None and edited_text is not None

    async def set_text(self, text: str) -> None:
        """Stopped / error notices (no Stop button)."""
        if self.message is None:
            return
        async with self._lock:
            if self.rich and await self._edit_rich([InputRichBlockParagraph(text=text)], progress=False):
                return
            await self.sender.edit_text(self.message, text, markdown=False)
