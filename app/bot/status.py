"""The status message of a turn: live progress, then a compact summary.

While the agent works, the status shows the step, the tool calls as they run and
the token usage. In rich mode it is a rich message edited in place with
expandable blocks; when the turn ends it is edited into a shortened summary
(no "Thinking…" header, a collapsible list of tool calls with their arguments
and results, the model's summarized thinking, tokens and cost) instead of being
deleted. Every rich edit falls back to the legacy-Markdown text version, so an
old Bot API server or client still gets a readable status.
"""
from __future__ import annotations

import logging
import re

from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.types import (
    InputRichBlockDetails,
    InputRichBlockList,
    InputRichBlockListItem,
    InputRichBlockParagraph,
    InputRichBlockPreformatted,
    InputRichMessage,
    RichTextBold,
    RichTextCode,
    RichTextItalic,
)

from bot import rich as rich_render
from bot.ui import escape_markdown

logger = logging.getLogger(__name__)

TRACE_LINES = 10  # most recent tool calls shown in the plain-text status
LIVE_CALLS = 15  # in the rich progress view
SUMMARY_CALLS = 40  # expandable entries in the final summary
SUMMARY_BYTES = 28000  # keep the summary well under the 32 KB rich-message limit
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


def progress_blocks(header: str, quota: str, step, details, iteration, critique, trace) -> list:
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
    """Wraps the status message of one turn; rich edits with a plain-text fallback."""

    def __init__(self, sender, message, *, rich: bool, header: str) -> None:
        self.sender = sender
        self.message = message
        self.header = header
        self.rich = bool(rich and message is not None and rich_render.is_available())

    async def _edit_rich(self, blocks: list) -> bool:
        if not self.rich or self.message is None:
            return False
        try:
            await self.sender.bot.edit_message_text(
                chat_id=self.sender.chat_id, message_id=self.message.message_id, rich_message=InputRichMessage(blocks=blocks)
            )
            return True
        except TelegramNotFound as e:
            rich_render.mark_unavailable(str(e))
            self.rich = False
        except TelegramBadRequest as e:
            if "not modified" in str(e).lower():
                return True
            logger.info("Rich status edit rejected (%s); using the plain status from now on", e)
            self.rich = False
        except Exception as e:  # noqa: BLE001 - network hiccup: keep rich mode, skip this update
            logger.warning("Rich status edit failed: %s", e)
        return False

    async def update(self, *, step, details, iteration, critique, trace, quota: str = "") -> None:
        if self.message is None:
            return
        if self.rich and await self._edit_rich(progress_blocks(self.header, quota, step, details, iteration, critique, trace)):
            return
        await self.sender.edit_text(self.message, progress_text(self.header, quota, step, details, iteration, critique, trace), fallback="strip")

    async def finish(self, trace, *, keep: bool) -> None:
        """End of a successful turn: shorten into the summary, or delete."""
        if self.message is None:
            return
        if not keep:
            await self.sender.delete([self.message])
            return
        if self.rich and await self._edit_rich(summary_blocks(trace)):
            return
        await self.sender.edit_text(self.message, trace_summary(trace), fallback="strip")

    async def set_text(self, text: str) -> None:
        """Stopped / error notices."""
        if self.message is None:
            return
        if self.rich and await self._edit_rich([InputRichBlockParagraph(text=text)]):
            return
        await self.sender.edit_text(self.message, text, markdown=False)
