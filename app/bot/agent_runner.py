"""Run one agent turn for a user and deliver the answer.

This replaces the old ``get_answer`` / ``send_response_to_user`` /
``send_reasoning_file`` trio and the copy of it that lived in the reminder
job. Everything user-facing about a turn (status message, streaming draft,
voice reply, text reply, reasoning file, error and cancel notices) happens
here, so every entry point behaves the same.

A turn that runs as a queue job (``ctx`` given) can be stopped where the user is
looking: the live status carries a Stop button (``queue_stop_<job token>``) and
the streaming drafts use the job id as ``draft_id`` with Telegram's own Stop
button. Both end in ``QueueManager.cancel_current`` for exactly this job; the
answer streamed so far is then kept as a real message, marked as partial. Once
the agent has returned, Stop no longer applies (``ctx.stoppable``), and once the
status has become the answer or the summary, a late /cancel leaves it as it is.

A successful turn ends with buttons (``bot/turn_actions.py``): the turn is saved
as a record, its actions (Continue after the tool budget ran out, Retry, Listen,
Reasoning) go at the end of the rich summary, or under the last answer message
when there is no summary, and the model's ``<follow_ups>`` suggestions become one
button per row under the answer. The reasoning file is sent only when its button
is pressed; if the record cannot be saved it is sent right away, as before, and
the turn gets no buttons.

A failed turn is not a dead end either: the status becomes "❌ <plain-language
reason> (trace <id>)" (``friendly_error``: rate limit, overloaded provider, network,
request too large, Telegram refusal, or a generic line; the id is in the log) with a
🔁 Retry button, backed by a turn record that carries the prompt and ``error``.
The tokens spent before the failure are still accounted for.

While the agent works, the turn has an inbox (``bot.inbox``): what the user sends in the
same chat and topic is handed to the agent between two steps instead of waiting for a
turn of its own, and tool calls that run long continue in the background
(``bot.background``). Both go to ``ChainOfThoughtAgent`` as ``inbox=`` / ``background=``
(only when its constructor takes them). The inbox closes as soon as the agent returned,
failed or was stopped; what it did not consume becomes the next turns, in order (dropped
on /cancel and at shutdown). The messages it did consume are part of the turn record
("[Sent while I was working]"), so Retry replays them. A scheduled task (or a notification
turn) that read one and still answers the quiet marker stays quiet and hands the messages it read
back to the queue: each gets a turn of its own on its prepared prompt, in arrival order.

A call moved to the background keeps spending after the turn's usage was recorded (a video
generation, a subagent, a slow image): from then on every addition to the turn's trace is
recorded as it arrives (``record_late_usage``). A failed turn cancels the calls it moved to the
background (silently), so Retry is the only re-run; a successful or stopped turn keeps them.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import anthropic
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from bot import background, inbox, turn_actions
from bot.limits import usage_line
from bot.media import synthesize_speech, user_dir
from bot.queue import STOP_CALLBACK_PREFIX, JobContext
from bot.sender import ChatSender
from bot.settings import UserSettings
from bot.ui import delete_quietly
from bot.status import StatusMessage, entity_safe, render_trace, stop_keyboard, trace_summary, usage_text  # noqa: F401 (re-exported)
from bot.streaming import create_streaming_callback
from models import ANTHROPIC_MODEL

logger = logging.getLogger(__name__)

THINKING = "💭 *Thinking...*"
QUIET_MARKER = "<no_update/>"  # a scheduled task answers with this alone when there is nothing to say
STOPPED_NOTE = "⏹ Stopped — partial answer, not saved to memory."

# Plain-language reasons for a failed turn (friendly_error). The technical detail is in the log.
ERROR_RATE_LIMIT = "The AI provider is rate-limiting us right now. Try again in a minute."
ERROR_OVERLOADED = "The AI provider is overloaded. Try again shortly."
ERROR_NETWORK = "Network problem reaching the AI provider."
ERROR_TOO_LARGE = "The request was too large (very long conversation or file). Try a shorter message, a smaller file or a new topic."
ERROR_TELEGRAM = "Telegram refused the message: {}"
ERROR_GENERIC = "Something went wrong."
_TOO_LARGE_HINTS = ("too large", "prompt is too long", "max_tokens", "exceed")
_TELEGRAM_DETAIL_CHARS = 120


def friendly_error(error: BaseException, trace_id: str | None = None) -> str:
    """What the user is told about a failed turn, with " (trace <id>)" appended when given.

    Anthropic SDK errors are told apart by class and status: a rate limit, an overloaded or
    failing provider (5xx), a connection or timeout problem, a request the API found too
    large; a Telegram refusal quotes Telegram's short description; everything else is the
    generic line. The trace id is what the log entry carries, so support can find it.
    """
    text = str(error).lower()
    if isinstance(error, anthropic.RateLimitError):
        reason = ERROR_RATE_LIMIT
    elif isinstance(error, anthropic.APIConnectionError):  # APITimeoutError included
        reason = ERROR_NETWORK
    elif isinstance(error, anthropic.APIStatusError) and (
        (isinstance(error.status_code, int) and error.status_code >= 500)
        or getattr(error, "type", None) == "overloaded_error" or "overloaded" in text
    ):
        reason = ERROR_OVERLOADED
    elif isinstance(error, anthropic.APIStatusError) and (
        error.status_code == 413 or (isinstance(error, anthropic.BadRequestError) and any(hint in text for hint in _TOO_LARGE_HINTS))
    ):
        reason = ERROR_TOO_LARGE
    elif isinstance(error, TelegramAPIError):
        detail = " ".join(str(getattr(error, "message", None) or error).split())
        if len(detail) > _TELEGRAM_DETAIL_CHARS:
            detail = detail[: _TELEGRAM_DETAIL_CHARS - 1].rstrip() + "…"
        reason = ERROR_TELEGRAM.format(detail)
    else:
        reason = ERROR_GENERIC
    return f"{reason} (trace {trace_id})" if trace_id else reason


@dataclass
class TurnResult:
    response: str  # the displayed answer: the follow-ups block is removed
    messages: list
    turn_id: str | None = None  # the saved turn record (bot.turn_actions), None when saving failed
    quiet: bool = False  # the answer was the quiet marker: nothing was sent (scheduled tasks)


def _agents():
    """Import the agent package lazily: it must be imported after the sandbox patcher ran."""
    import agents.main as agents

    return agents


def takes_live_turn(agent_cls) -> bool:
    """The agent class takes ``inbox=`` and ``background=`` (or any keyword arguments)."""
    try:
        params = inspect.signature(agent_cls).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "inbox" in params and "background" in params


def with_drained(prompt_text: str, prompt_blocks: list | None, box: inbox.TurnInbox | None) -> tuple[str, list | None]:
    """The turn record's prompt plus what the user sent while the agent worked (Retry replays both)."""
    texts = box.drained_user_texts if box is not None else []
    if not texts:
        return prompt_text, prompt_blocks
    text = f"{prompt_text}\n\n{inbox.SENT_WHILE_WORKING}\n" + "\n\n".join(texts)
    if not prompt_blocks and not box.drained_media:
        return text, None
    from agents.content import text_block

    return text, [*(prompt_blocks or [text_block(prompt_text)]), *box.drained_record_blocks]


def _record_blocks(prompt) -> list | None:
    """The prompt as stored blocks: media refs as they are; an image/document block that came
    materialized (a background task's screenshot) as its marker, so records stay small."""
    if not isinstance(prompt, list):
        return None
    from agents import content as prompt_content

    out = []
    for b in prompt_content.normalize(prompt):
        out.append(prompt_content.text_block(prompt_content.media_marker(b)) if prompt_content.is_media_block(b) else dict(b))
    return out


ARGS_CHARS = 2000  # per tool call, in the reasoning file
RESULT_CHARS = 2000  # per tool result
THINKING_CHARS = 4000  # per thinking block


def _clip(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit]}… ({len(text)} chars total)"


def _result_text(content: Any) -> str:
    """A tool result is a string or a list of content blocks (text, images as ``[image]``, a
    browser result's ``browser_state`` as its one-line summary)."""
    if isinstance(content, list):
        from agents.content import result_preview  # no import-time side effects

        return result_preview(content)
    return str(content)


def _call_name(block: dict) -> str:
    """``browser.type`` for a toolset member, the plain tool name otherwise (as in the trace)."""
    name = str(block.get("name", "unknown"))
    family = block.get("toolset_name")
    return f"{family}.{name}" if family else name


def _call_args(block: dict) -> Any:
    """A tool call's arguments for the file; what a browser member typed into a page is masked."""
    args = block.get("input")
    if block.get("toolset_name"):
        from agents.browser_tools import masked_input  # no import-time side effects

        args = masked_input(block.get("name", ""), args)
    return args


def reasoning_text(messages: list) -> str:
    """The turn as a readable transcript: prompts, thinking, tool calls WITH their arguments, results.

    Every block of every message is rendered, not just the first one: with thinking on, an
    assistant message starts with a ``thinking`` block and its tool calls follow it, so reading
    only ``content[0]`` hid the calls and the parameters they were made with.
    """
    names: dict[str, str] = {}  # tool_use_id -> tool name, so results can name their call
    for msg in messages:
        for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names[str(block.get("id"))] = _call_name(block)

    out: list[str] = []
    for msg in messages:
        try:
            role = str(msg.get("role", "")).capitalize() or "Message"
            content = msg.get("content", [])
            if isinstance(content, str):
                out.append(f"[{role}]\n{content}" if role else content)
                continue
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text" and str(block.get("text", "")).strip():
                    out.append(f"[{role}]\n{block['text']}")
                elif kind == "thinking" and str(block.get("thinking", "")).strip():
                    out.append(f"[{role} thinking]\n{_clip(block['thinking'], THINKING_CHARS)}")
                elif kind == "tool_use":
                    args = _call_args(block)
                    try:
                        rendered = json.dumps(args, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError):
                        rendered = str(args)
                    out.append(f"[Tool call: {_call_name(block)}]\n{_clip(rendered, ARGS_CHARS)}")
                elif kind == "tool_result":
                    name = names.get(str(block.get("tool_use_id")), "unknown")
                    status = " — error" if block.get("is_error") else ""
                    body = _clip(_result_text(block.get("content", "")), RESULT_CHARS)
                    out.append(f"[Tool result: {name}{status}]\n{body}")
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
    return "\n\n========\n\n".join(out) + "\n\n========\n\n" if out else ""


def record_usage(stats_tracker, user_id: str, trace) -> None:
    """Token accounting: one usage row per model used in this turn (never raises)."""
    try:
        for model, u in (trace.usage_by_model or {}).items():
            stats_tracker.track_usage(
                user_id,
                model=model,
                api_calls=u["api_calls"],
                input_tokens=u["input_tokens"],
                output_tokens=u["output_tokens"],
                cache_read_tokens=u["cache_read_tokens"],
                cache_write_tokens=u["cache_write_tokens"],
                tool_calls=trace.total,
                duration_s=trace.elapsed,
                cost_usd=trace.model_cost(model),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record token usage for %s: %s", user_id, e)


def record_late_usage(stats_tracker, user_id: str, trace) -> None:
    """After ``record_usage``: usage that still reaches ``trace`` (a call the turn moved to the
    background: Veo, a subagent's model calls, a slow image) is recorded as it arrives, one usage
    row per addition with exactly that usage, so nothing is lost and nothing is counted twice."""

    def late(model: str, counts: dict, cost_usd: float | None) -> None:
        try:
            if cost_usd is None:
                from models import estimate_cost

                cost_usd = estimate_cost(model, counts["input_tokens"], counts["output_tokens"],
                                         counts["cache_read_tokens"], counts["cache_write_tokens"])
            stats_tracker.track_usage(
                user_id, model=model, api_calls=counts["api_calls"], input_tokens=counts["input_tokens"],
                output_tokens=counts["output_tokens"], cache_read_tokens=counts["cache_read_tokens"],
                cache_write_tokens=counts["cache_write_tokens"], tool_calls=0, duration_s=0.0, cost_usd=cost_usd,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not record late token usage for %s: %s", user_id, e)

    trace.on_late_usage = late


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


def _close_drafts(on_text_chunk) -> None:
    """No draft may reappear after the answer (or the stop notice) went out."""
    close = getattr(on_text_chunk, "close", None)
    if close is not None:
        close()


async def save_turn(user_id: str, record: turn_actions.TurnRecord, messages: list, settings: UserSettings) -> turn_actions.TurnRecord | None:
    """Store the turn for its buttons, with the reasoning transcript when that setting is on.

    Returns None (after logging) when it could not be saved: the caller then sends the
    reasoning file right away and shows no buttons, since nothing could answer them.
    """
    reasoning = reasoning_text(messages) if settings.get("reasoning_context", "enabled") and messages else ""
    try:
        await turn_actions.save(user_id, record, reasoning or None)
    except Exception as e:  # noqa: BLE001
        logger.error("Could not save turn %s for %s (no turn buttons): %s", record.id, user_id, e)
        return None
    return record


async def send_partial_answer(sender: ChatSender, on_text_chunk) -> None:
    """After a user stop: keep the answer streamed so far as a real message (the draft vanishes).

    Best-effort and never raises an ``Exception``: the caller is re-raising the
    ``CancelledError``, and a failed send must not replace it.
    """
    partial_text = getattr(on_text_chunk, "partial_text", None)
    if partial_text is None:
        return
    try:
        _close_drafts(on_text_chunk)
        partial = (partial_text() or "").strip()
        if partial:
            await sender.send_markdown(f"{partial}\n\n{STOPPED_NOTE}")
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not send the partial answer to %s: %s", sender.chat_id, e)


async def run_turn(
    *,
    bot: Bot,
    user_id: str,
    chat_id: int,
    prompt: str | list,
    thread_id: int | None = None,
    reply_to_message_id: int | None = None,
    ctx: JobContext | None = None,
    limit: int | None = None,
    status: Message | None = None,
    header: str = THINKING,
    speak: bool = False,
    reasoning_caption: str = "Reasoning history.",
    quiet_marker: str | None = None,
    quiet_notice: str | None = None,
) -> TurnResult | None:
    """Run the agent on ``prompt`` (a string, or a list of content blocks built with
    ``agents.content``: text and media refs) and deliver the result. Returns None after reporting an error.

    With ``ctx`` the status gets a Stop button for ``ctx.job`` and the drafts use
    ``ctx.job.id``; without it (no queue job) there is no Stop button. A successful
    turn is saved (``bot.turn_actions``) and gets its action and follow-up buttons.

    ``quiet_marker`` (scheduled tasks): an answer consisting of exactly that marker means the
    agent has nothing worth saying; nothing is sent, the status message is removed (or, with
    ``quiet_notice``, edited into that line: a turn the user started by a button expects some
    reply) and the result carries ``quiet=True``; user messages the turn read meanwhile are
    queued again as turns of their own (``inbox.requeue``). The marker is stored on the turn
    record, so Retry runs with it again. Such a turn never shows the browser's live view (it
    would stay behind after a quiet run).
    """
    from stats import stats_tracker
    from agents import content as prompt_content  # no import-time side effects; after the patcher ran

    prompt_text = prompt_content.prompt_text(prompt)
    prompt_blocks = _record_blocks(prompt)
    settings = UserSettings(user_id)
    settings.save()  # persist any defaults added since the file was written
    user_settings: dict[str, Any] = settings.as_dict()
    rich_enabled = bool(settings.get("rich_messages", "enabled"))
    sender = ChatSender(bot, chat_id, thread_id, reply_to_message_id, rich=rich_enabled, media_root=user_dir(user_id))

    job = ctx.job if ctx is not None else None
    # The Stop button names this job, so a late press can never stop the next queued turn.
    stop_callback = f"{STOP_CALLBACK_PREFIX}{job.token}" if job is not None else None

    await sender.typing()
    if status is None:
        status = await sender.send_text(header, reply_markup=stop_keyboard(stop_callback) if stop_callback else None)

    from agents.trace import ToolTrace

    trace = ToolTrace()
    status_msg = StatusMessage(sender, status, rich=rich_enabled, header=header, stop_callback=stop_callback)
    keep_summary = bool(settings.get("trace", "keep_summary"))

    async def update_status(step: str, details: str, iteration: Any, critique: Any) -> None:
        if step == "saving":
            iteration, critique = "final", "end"
        if ctx is not None:
            ctx.set_progress(f"{step}: {details}"[:120])
        await status_msg.update(step=step, details=details, iteration=iteration, critique=critique, trace=trace, quota=usage_line(user_id, limit))

    # Drafts carry the job id, so Telegram's Stop button on a draft names the job to cancel.
    on_text_chunk = create_streaming_callback(
        bot, chat_id, thread_id, rich=rich_enabled,
        draft_id=job.id if job is not None else None, can_stop=job is not None,
    )

    agent_done = False
    # Set once the status message is the answer (legacy) or the turn summary (rich): a cancel
    # or error after that (e.g. /cancel while the buttons go on) must not overwrite it.
    status_final = False
    box: inbox.TurnInbox | None = None  # the turn's inbox, while the agent runs
    host: background.BackgroundHost | None = None  # the user's background host, bound to this turn
    host_turn = 0

    def close_inbox() -> None:
        # Leftovers become the next turns; at shutdown the queue is closing, so they are dropped.
        inbox.finish(box, discard=ctx is not None and ctx.cancel_reason == "shutdown")

    agent_settings = user_settings
    if quiet_marker:
        # A scheduled task runs unattended, and a quiet run must leave nothing in the chat: no
        # live view of the browser (browser.stream off for this turn only; it changes neither the
        # tools nor the system prompt, so the transcript's fingerprint stays).
        browser = user_settings.get("browser") if isinstance(user_settings.get("browser"), dict) else {}
        agent_settings = {**user_settings, "browser": {**browser, "stream": False}}
    try:
        agents = _agents()
        live: dict[str, Any] = {}
        if takes_live_turn(agents.ChainOfThoughtAgent):
            # Messages of this chat and topic join this turn from now on (bot/handlers/messages.py).
            box = inbox.open_inbox(user_id, thread_id, base=user_dir(user_id))
            host = background.host_for(user_id, bot=bot, chat_id=chat_id, thread_id=thread_id)
            host_turn = host.turn
            live = {"inbox": box, "background": host}
        agent = agents.ChainOfThoughtAgent(
            model_type="anthropic",
            model=ANTHROPIC_MODEL,
            user_id=user_id,
            sender=sender,
            user_settings=agent_settings,
            message_thread_id=thread_id,
            **live,
        )
        response, messages = await agent.generate_response(prompt, update_status, on_text_chunk=on_text_chunk, trace=trace)
        close_inbox()  # what the agent did not read becomes the next turns, queued behind this one
        _close_drafts(on_text_chunk)
        # The answer is complete (and already in memory): Stop no longer applies, so a late
        # press cannot cut the delivery short or resend the streamed part as "not saved".
        agent_done = True
        status_msg.stop_callback = None
        if ctx is not None:
            ctx.stoppable = False
        stats_tracker.track_message_sent(user_id)
        record_usage(stats_tracker, user_id, trace)
        record_late_usage(stats_tracker, user_id, trace)  # calls still running in the background

        # A trailing <follow_ups> block becomes buttons; it is never part of the visible answer.
        display, suggestions = turn_actions.parse_follow_ups(response or "")
        if not (user_settings.get("follow_ups") or {}).get("enabled", True):
            suggestions = []
        record_prompt, record_blocks = with_drained(prompt_text, prompt_blocks, box)
        if quiet_marker and display.strip() == quiet_marker:
            # Nothing to tell the user: no answer, no voice, no buttons; the status goes away too
            # (or says so, when the user pressed a button and waits for a reply).
            status_final = True
            if box is not None and box.consumed_user_items:
                # The user wrote while the task ran and the agent read it: those messages wait for
                # an answer the quiet run does not give. Each becomes a turn of its own on its
                # prepared prompt, in arrival order, ahead of anything sent later.
                inbox.requeue(box.consumed_user_items)
            if quiet_notice:
                await status_msg.set_text(quiet_notice)
            else:
                await delete_quietly(status)
            return TurnResult(response="", messages=messages, turn_id=None, quiet=True)
        limit_reached = getattr(getattr(agent, "agent", None), "limit_reached", False) is True
        record = await save_turn(user_id, turn_actions.TurnRecord(
            id=turn_actions.new_turn_id(), prompt=record_prompt, prompt_blocks=record_blocks, response=display, speak=speak, thread_id=thread_id,
            limit_reached=limit_reached, follow_ups=suggestions, reasoning_caption=reasoning_caption, quiet_marker=quiet_marker,
        ), messages, settings)

        if speak and display:
            await status_msg.update(step="audio", details="Generating audio", iteration="final", critique="end", trace=trace)
            from voice import VoiceManager

            voice_id = VoiceManager().get_user_voice(user_id)
            audio = await synthesize_speech(display, voice_id)
            if audio:
                try:
                    await sender.send_voice(audio)
                except Exception as e:  # noqa: BLE001
                    logger.error("Error sending voice reply: %s", e)

        actions = turn_actions.record_actions(record) if record is not None else []
        placed = False
        if rich_enabled:
            # The answer is a new rich message; the status is shortened into the turn summary
            # (tool calls, thinking, tokens, and the action buttons) above it, or deleted when
            # the user turned that off.
            sent = await sender.send_markdown(display)
            placed = await status_msg.finish(trace, keep=keep_summary, actions=actions)
        else:
            sent = await sender.send_markdown(display, edit=status)
        status_final = True
        if record is None:
            await send_reasoning_file(sender, messages, settings, caption=reasoning_caption)
        else:
            # Follow-ups (and the actions, unless the summary carries them) under the last answer message.
            markup = turn_actions.answer_keyboard(record, None if placed else actions)
            if markup is not None and sent:
                await turn_actions.attach_keyboard(bot, chat_id, sent[-1], markup)
        return TurnResult(response=display, messages=messages, turn_id=record.id if record is not None else None)
    except asyncio.CancelledError:
        close_inbox()  # a Stop or the deadline: the unread messages still get their turns
        if not agent_done:
            record_late_usage(stats_tracker, user_id, trace)  # calls moved to the background keep running
        reason = ctx.cancel_reason if ctx else None
        if reason == "user" and not agent_done:
            await send_partial_answer(sender, on_text_chunk)
        else:
            _close_drafts(on_text_chunk)
        if not status_final:
            await status_msg.set_text("🛑 Stopped." if reason == "user" else "🛑 Cancelled.")
        raise
    except Exception as e:  # noqa: BLE001
        close_inbox()
        _close_drafts(on_text_chunk)
        trace_id = str(uuid.uuid4())
        logger.exception("Turn failed for user %s (trace %s): %s", user_id, trace_id, e)
        if not agent_done:
            record_usage(stats_tracker, user_id, trace)  # the tokens spent before the failure
            record_late_usage(stats_tracker, user_id, trace)
        if status_final:
            return None  # the answer (or the summary) is already there; nothing to replace it with
        if host is not None:
            # Retry runs the turn again, background calls included: the ones this turn started end
            # here, silently, so they do not run twice.
            host.cancel_turn(host_turn)
        reason = friendly_error(e, trace_id)
        # The failed turn is saved too (prompt and reason, no answer, no reasoning file), so the
        # status can carry a Retry button; when even that fails, the text goes out without buttons.
        # Messages the agent read before it failed are part of it: Retry runs them again.
        record_prompt, record_blocks = with_drained(prompt_text, prompt_blocks, box)
        record = await save_turn(user_id, turn_actions.TurnRecord(
            id=turn_actions.new_turn_id(), prompt=record_prompt, prompt_blocks=record_blocks, response="", speak=speak,
            thread_id=thread_id, reasoning_caption=reasoning_caption, error=reason, quiet_marker=quiet_marker,
        ), [], settings)
        actions = turn_actions.record_actions(record) if record is not None else []
        await status_msg.fail(f"❌ {reason}", actions=actions)
        return None
    finally:
        close_inbox()  # idempotent: already closed on every path above
