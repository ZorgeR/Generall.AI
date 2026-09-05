"""Run one agent turn for a user and deliver the answer.

This replaces the old ``get_answer`` / ``send_response_to_user`` /
``send_reasoning_file`` trio and the copy of it that lived in the reminder
job. Everything user-facing about a turn (status message, streaming draft,
voice reply, text reply, reasoning file, error and cancel notices) happens
here, so every entry point behaves the same.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from aiogram import Bot
from aiogram.types import Message

from bot.limits import usage_line
from bot.media import synthesize_speech, user_dir
from bot.queue import JobContext
from bot.sender import ChatSender
from bot.settings import UserSettings
from bot.streaming import create_streaming_callback
from bot.ui import escape_markdown
from models import ANTHROPIC_MODEL

logger = logging.getLogger(__name__)

THINKING = "💭 *Thinking...*"
TRACE_LINES = 10  # most recent tool calls shown in the status message


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 10 else f"{int(round(seconds))}s"


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def usage_text(trace) -> str:
    """Token usage of the turn so far, with the share served from the prompt cache."""
    if not trace or not trace.api_calls:
        return ""
    prompt = trace.input_tokens + trace.cache_read_tokens + trace.cache_write_tokens
    text = f"🧮 {trace.api_calls} call{'s' if trace.api_calls != 1 else ''} · in {_fmt_tokens(prompt)}"
    if trace.cache_read_tokens:
        text += f" ({int(round(trace.cache_hit_ratio * 100))}% cached)"
    return text + f" · out {_fmt_tokens(trace.output_tokens)}"


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
        icon = "⏳" if call.running else ("✅" if call.ok else "❌")
        indent = "  " * call.depth
        args = describe_args(call.args)
        line = f"{indent}{icon} `{call.name}`"
        if args:
            line += f" {escape_markdown(args)}"
        if not call.running:
            line += f" · {_fmt_seconds(call.duration)}"
        lines.append(line)
    return "\n".join(lines)


def trace_summary(trace) -> str:
    """One line kept above the answer once the turn is over."""
    parts = ", ".join(f"{name} ×{n}" if n > 1 else name for name, n in trace.counts_by_name())
    text = f"🔧 *{trace.total} tool call{'s' if trace.total != 1 else ''} in {_fmt_seconds(trace.elapsed)}*"
    if trace.errors:
        text += f" · {trace.errors} failed"
    text = f"{text}: {escape_markdown(parts)}"
    usage = usage_text(trace)
    return f"{text}\n{usage}" if usage else text


@dataclass
class TurnResult:
    response: str
    messages: list


def _agents():
    """Import the agent package lazily: it must be imported after the sandbox patcher ran."""
    import agents.main as agents

    return agents


def reasoning_text(messages: list) -> str:
    out: list[str] = []
    for msg in messages:
        try:
            content = msg.get("content", [])
            if isinstance(content, str):
                out.append(content)
            elif isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and first.get("type") == "text":
                    out.append(first.get("text", ""))
                elif isinstance(first, dict) and first.get("type") == "tool_use":
                    out.append(f"[Tool: {first.get('name', 'unknown')}]")
                elif isinstance(first, dict) and first.get("type") == "tool_result":
                    out.append(f"[Tool Result: {str(first.get('content', ''))[:200]}]")
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
    return "\n\n========\n\n".join(out) + "\n\n========\n\n" if out else ""


async def send_reasoning_file(sender: ChatSender, messages: list, settings: UserSettings, caption: str = "Reasoning history.") -> None:
    if not settings.get("reasoning_context", "enabled") or not messages:
        return
    text = reasoning_text(messages)
    if not text:
        return
    try:
        await sender.send_document(text.encode("utf-8"), filename=f"reasoning_{sender.chat_id}_{uuid.uuid4().hex[:8]}.txt", caption=caption)
    except Exception as e:  # noqa: BLE001
        logger.error("Error sending reasoning file: %s", e)


async def run_turn(
    *,
    bot: Bot,
    user_id: str,
    chat_id: int,
    prompt: str,
    thread_id: int | None = None,
    reply_to_message_id: int | None = None,
    ctx: JobContext | None = None,
    limit: int | None = None,
    status: Message | None = None,
    header: str = THINKING,
    speak: bool = False,
    reasoning_caption: str = "Reasoning history.",
) -> TurnResult | None:
    """Run the agent on ``prompt`` and deliver the result. Returns None after reporting an error."""
    from stats import stats_tracker

    settings = UserSettings(user_id)
    settings.save()  # persist any defaults added since the file was written
    user_settings: dict[str, Any] = settings.as_dict()
    rich_enabled = bool(settings.get("rich_messages", "enabled"))
    sender = ChatSender(bot, chat_id, thread_id, reply_to_message_id, rich=rich_enabled, media_root=user_dir(user_id))

    await sender.typing()
    if status is None:
        status = await sender.send_text(header)

    from agents.trace import ToolTrace

    trace = ToolTrace()

    async def update_status(step: str, details: str, iteration: Any, critique: Any) -> None:
        if step == "saving":
            iteration, critique = "final", "end"
        if ctx is not None:
            ctx.set_progress(f"{step}: {details}"[:120])
        text = (
            f"{header}\n- - - - \n{usage_line(user_id, limit)}"
            f"📝 *Step:* _{str(step).replace('_', '-')}_\n"
            f"📋 *Details:* _{escape_markdown(str(details))}_\n"
            f"🔄 *Iterations:* _{iteration}_\n"
            f"🎯 *Critiques:* _{critique}_"
        )
        trace_text = render_trace(trace)
        if trace_text:
            text += "\n\n" + trace_text
        usage = usage_text(trace)
        if usage:
            text += "\n" + usage
        await sender.edit_text(status, text)

    on_text_chunk = create_streaming_callback(bot, chat_id, thread_id, rich=rich_enabled)

    try:
        agents = _agents()
        agent = agents.ChainOfThoughtAgent(
            model_type="anthropic",
            model=ANTHROPIC_MODEL,
            user_id=user_id,
            sender=sender,
            user_settings=user_settings,
            message_thread_id=thread_id,
        )
        response, messages = await agent.generate_response(prompt, update_status, on_text_chunk=on_text_chunk, trace=trace)
        stats_tracker.track_message_sent(user_id)

        if speak and response:
            await sender.edit_text(status, "🎙️ *Generating audio...*")
            from voice import VoiceManager

            voice_id = VoiceManager().get_user_voice(user_id)
            audio = await synthesize_speech(response, voice_id)
            if audio:
                try:
                    await sender.send_voice(audio)
                except Exception as e:  # noqa: BLE001
                    logger.error("Error sending voice reply: %s", e)

        if rich_enabled and trace.calls:
            # Keep the status message as a compact record of the tool calls above the answer
            # (a rich answer cannot replace it anyway); without tool calls it is deleted as before.
            await sender.send_markdown(response)
            await sender.edit_text(status, trace_summary(trace))
        else:
            await sender.send_markdown(response, edit=status)
        await send_reasoning_file(sender, messages, settings, caption=reasoning_caption)
        return TurnResult(response=response, messages=messages)
    except asyncio.CancelledError:
        reason = ctx.cancel_reason if ctx else None
        await sender.edit_text(status, "🛑 Stopped." if reason == "user" else "🛑 Cancelled.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        trace_id = str(uuid.uuid4())
        logger.exception("Turn failed for user %s (trace %s): %s", user_id, trace_id, e)
        await sender.edit_text(status, f"❌ An error occurred. Trace ID: {trace_id}", markdown=False)
        return None
