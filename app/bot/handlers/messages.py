"""Inbound content handlers: text, voice, video, audio, photos/albums, documents.

Handlers validate quickly, enqueue a job on the user's queue and return.
If the user already has a job running they get a busy notice immediately;
the new message is processed as its own turn once the queue reaches it.

Unless that job is a turn of the same chat and forum topic whose agent is still working
(``inject_target``: its inbox is open and no earlier message of that topic is still
queued): then the message joins that turn instead (``bot.inbox``). It gets a 👀 reaction,
no busy notice, and is prepared in the background (``prepare_<kind>``: the download,
conversion and transcription a job would do, without the status messages) while the agent
goes on; the agent reads it at its next tool boundary (✍). Every kind of message is split
that way: ``prepare_<kind>`` builds a ``bot.inbox.Prepared`` (the prompt, the voice reply
flag, the error to show), and the ``_run_<kind>`` job wraps it in status messages and runs
the turn. A message the turn did not read is queued as a turn of its own on the prompt it
already prepared (``_run_prepared``), in arrival order. A plain text sent while an
``ask_user_choice`` question waits in that chat and topic answers the question instead.

``control_router`` (auth only) stops work: ``/cancel`` and the busy notice's
button stop the running job and clear the queue (the messages waiting in the running
turn's inbox and the user's background tasks too); the Stop button in a turn's
status (``queue_stop_<job token>``) and Telegram's Stop button on its streaming
draft (a ``stopped_message_generation`` update) stop only that job (unread messages
then become the next turns; background tasks keep running).

Every job puts the reply / forward preamble (``bot.message_context``) in front of
its prompt, so a reply to an earlier message or a forwarded post reaches the agent
with what it refers to.

Photos and PDFs are not described beforehand: the image job and the document job
build a block prompt (``agents.content``: one text block, then an ``image_ref`` per
picture or a ``document_ref`` for a PDF within the limits) and the agent sees the
media itself. Text, docx, xlsx and json documents go in as
extracted text; a PDF too large to attach falls back to the description path.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Awaitable, Callable

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, MessageGenerationStopped

from bot import background, inbox, media, questions
from bot.agent_runner import THINKING, run_turn
from bot.inbox import Prepared
from bot.message_context import with_reference
from bot.queue import STOP_CALLBACK_PREFIX, Job, JobContext, SubmitResult, job_id_from_token, next_seq
from bot.runtime import queue
from bot.sender import ChatSender
from bot.ui import answer_md, delete_quietly
from image_utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

router = Router(name="messages")
control_router = Router(name="queue-control")

MEDIA_GROUP_TIMEOUT = 10.0
CANCEL_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🛑 Stop current task", callback_data="queue_cancel")]])


TEXT_PROMPT_CAP = 30_000  # characters of a text document put in the prompt
EXTRACTED_TEXT_CAP = 60_000  # characters of extracted docx/xlsx/json text put in the prompt
IMAGE_EMBED_HINT = "The images are saved in your workspace; to show one inline in your answer, write ![caption](images/<file name>)."


def _stats():
    from stats import stats_tracker

    return stats_tracker


def _content():
    """``agents.content`` on first use (bot/ imports nothing from agents at module level)."""
    from agents import content

    return content


def _prompt_with_reference(text: str, media_blocks: list, message: Message, bot_id: int | None) -> str | list:
    """The job's prompt: the preamble + ``text`` as one string when there is no media (exactly
    ``with_reference``), else that string as the first text block followed by the media refs."""
    text = with_reference(text, message, bot_id)
    if not media_blocks:
        return text
    return [_content().text_block(text), *media_blocks]


def _thread_id(message: Message) -> int | None:
    return getattr(message, "message_thread_id", None) if getattr(message, "is_topic_message", None) else None


def _bot_id(bot: Bot) -> int | None:
    """``Bot.id`` (from the token); None for the bot stand-ins of the tests."""
    return getattr(bot, "id", None)


def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _md_safe(text: str) -> str:
    """Make free text safe inside legacy-Markdown bold/italic spans."""
    return text.replace("_", "-").replace("*", "").replace("`", "'")


async def notify_busy(message: Message, result: SubmitResult) -> None:
    current = result.current
    if current is not None:
        label = _md_safe(current.label)
        if current.started_at is None:
            text = f"⏳ I'm about to start on your previous request (*{label}*).\nI'll handle this message right after it finishes."
        else:
            step = _md_safe(current.progress or "starting")
            text = (
                f"⏳ I'm still working on your previous request (*{label}*, "
                f"{_format_elapsed(current.elapsed)} elapsed, step: _{step}_).\n"
                "I'll handle this message right after it finishes."
            )
    else:
        text = "⏳ Your previous messages are still queued. I'll handle this one in turn."
    ahead = result.position - 1
    if ahead > 0:
        text += f"\n📬 {ahead} more message(s) are queued ahead of it."
    text += "\nSend /cancel to stop the current task and clear the queue."
    try:
        await message.answer(text, parse_mode="Markdown", reply_markup=CANCEL_KEYBOARD)
    except Exception:  # noqa: BLE001
        await message.answer(text.replace("*", "").replace("_", ""), reply_markup=CANCEL_KEYBOARD)


async def submit(message: Message, job: Job) -> None:
    result = await queue.submit(job)
    if result.was_busy:
        await notify_busy(message, result)


# ---------------------------------------------------------------------------
# joining the running turn (bot.inbox)
# ---------------------------------------------------------------------------
Progress = Callable[[str | None, str | None], Awaitable[None]]  # (status text, job progress)


async def _no_progress(status_text: str | None = None, step: str | None = None) -> None:
    """Preparing for the running turn: no status message, no job to report to."""
    return None


def _job_progress(sender: ChatSender, status: Message | None, ctx: JobContext | None) -> Progress:
    """Preparing as a queued job: the status message and the job's progress line follow along."""

    async def progress(status_text: str | None = None, step: str | None = None) -> None:
        if step and ctx is not None:
            ctx.set_progress(step)
        if status_text and status is not None:
            await sender.edit_text(status, status_text)

    return progress


def inject_target(user_id: str, thread_id: int | None) -> inbox.TurnInbox | None:
    """The running turn a new message of this topic joins, or None (queue it as today).

    The turn's inbox must be open for this topic and no earlier message (or button press) of
    the user in this topic may still be queued: that one is answered first, and this one after it.
    """
    target = inbox.current(user_id)
    if target is None or not target.accepts(thread_id):
        return None
    pending = getattr(queue, "has_pending_user_jobs", None)
    if pending is not None and pending(user_id, thread_id):
        return None
    return target


async def _prepare_for_inbox(
    prepare: Callable[[], Awaitable[Prepared]], *, kind: str, label: str, error_text: str, echo: ChatSender | None,
) -> Prepared:
    """Run a ``prepare_<kind>`` for the running turn; never raises (a failure is a short note the agent reads)."""
    try:
        prepared = await prepare()
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Could not prepare a %s for the running turn: %s", kind, e)
        return Prepared(prompt=None, kind=kind, label=label, error_text=error_text, reason=inbox.failure_reason(e))
    if prepared.ok and prepared.echo and echo is not None:
        try:
            await echo.send_text(prepared.echo)  # e.g. the transcription of a voice message
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not show what was prepared: %s", e)
    return prepared


def inject(
    target: inbox.TurnInbox, bot: Bot, user_id: str, limit: int | None, *, chat_id: int, thread_id: int | None,
    message_id: int | None, sent_at: Any = None, label: str, kind: str, error_text: str,
    prepare: Callable[[], Awaitable[Prepared]] | None = None, prepared: Prepared | None = None, seq: int | None = None,
) -> inbox.InboxItem:
    """Deliver a message to the running turn: ``prepared`` as it is, or ``prepare`` run in the background.

    No suspension point between ``inject_target`` and this call, so nothing can slip in between.
    A leftover (the turn ended before reading it) is queued as a turn of its own on the same
    prepared prompt (``_run_prepared``), in arrival order.
    """
    if prepared is not None:
        future = inbox.ready_future(prepared)
    else:
        echo = ChatSender(bot, chat_id, thread_id, message_id)
        future = asyncio.create_task(
            _prepare_for_inbox(prepare, kind=kind, label=label, error_text=error_text, echo=echo), name=f"prepare-{user_id}-{label}",
        )
    item = inbox.InboxItem(
        kind=inbox.USER, future=future, thread_id=thread_id, label=label, bot=bot, chat_id=chat_id, message_id=message_id,
        sent_at=sent_at, **({"seq": seq} if seq is not None else {}),
    )
    item.requeue = partial(_requeue, bot, user_id, chat_id, thread_id, message_id, limit, item)
    target.deliver(item)
    return item


def _requeue(bot: Bot, user_id: str, chat_id: int, thread_id: int | None, reply_to: int | None, limit: int | None, item: inbox.InboxItem) -> None:
    queue.submit_nowait(Job(
        user_id=user_id, label=item.label, seq=item.seq, thread_id=thread_id, from_user=True,
        run=partial(_run_prepared, bot, user_id, chat_id, thread_id, reply_to, limit, item.future),
    ))


async def _run_prepared(
    bot: Bot, user_id: str, chat_id: int, thread_id: int | None, reply_to: int | None, limit: int | None,
    future: asyncio.Future, ctx: JobContext,
) -> None:
    """A message the running turn did not read, as a turn of its own (its preparation is reused)."""
    if future.cancelled():
        return
    ctx.set_progress("preparing")
    prepared: Prepared = await future  # usually done long ago; a cancel of this job cancels it too
    if not prepared.ok:
        await ChatSender(bot, chat_id, thread_id, reply_to).send_text(prepared.error_text or "❌ Could not process this message.", markdown=False)
        return
    await run_turn(
        bot=bot, user_id=user_id, chat_id=chat_id, prompt=prepared.prompt, thread_id=thread_id,
        reply_to_message_id=reply_to, ctx=ctx, limit=limit, speak=prepared.speak,
    )


async def accept(
    message: Message, bot: Bot, user_id: str, limit: int | None, *, label: str, kind: str, error_text: str,
    prepare: Callable[[], Awaitable[Prepared]], run: Callable[[JobContext], Awaitable[Any]], seq: int | None = None,
) -> None:
    """A content message: into the running turn of its topic (👀), else a queued job (busy notice as before)."""
    thread_id = _thread_id(message)
    target = inject_target(user_id, thread_id)
    if target is not None:
        item = inject(
            target, bot, user_id, limit, chat_id=message.chat.id, thread_id=thread_id, message_id=message.message_id,
            sent_at=message.date, label=label, kind=kind, error_text=error_text, prepare=prepare, seq=seq,
        )
        await inbox.mark_seen(item)
        return
    job = Job(user_id=user_id, label=label, run=run, thread_id=thread_id, from_user=True, **({"seq": seq} if seq is not None else {}))
    await submit(message, job)


# ---------------------------------------------------------------------------
# /cancel
# ---------------------------------------------------------------------------
async def _cancel_for(user_id: str) -> str:
    # Messages waiting in the running turn's inbox are dropped with the queue (not requeued when
    # the turn stops), and the user's background tasks end without a notification.
    dropped_unread = inbox.discard(user_id)
    stopped_tasks = background.cancel_all(user_id)
    cancelled, dropped = await queue.cancel(user_id, reason="user")
    dropped += dropped_unread
    if not cancelled and not dropped and not stopped_tasks:
        return "Nothing is running right now."
    parts = []
    if cancelled:
        parts.append("stopped the current task")
    if dropped:
        parts.append(f"dropped {dropped} queued message(s)")
    if stopped_tasks:
        parts.append(f"cancelled {stopped_tasks} background task(s)")
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return "🛑 " + text.capitalize() + "."


@control_router.message(F.text, Command("cancel"))
async def cancel_command(message: Message, user_id: str) -> None:
    await message.answer(await _cancel_for(user_id))


@control_router.callback_query(F.data == "queue_cancel")
async def cancel_button(callback: CallbackQuery, user_id: str) -> None:
    text = await _cancel_for(user_id)
    await callback.answer(text, show_alert=False)
    if isinstance(callback.message, Message):
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Stop buttons of one turn: its status message and its streaming draft
# ---------------------------------------------------------------------------
STOP_STOPPING_TEXT = "🛑 Stopping…"
STOP_TOO_LATE_TEXT = "Too late to stop: the answer is already being sent."
STOP_FINISHED_TEXT = "This task has already finished."


def _only_button(markup: InlineKeyboardMarkup | None, data: str | None) -> bool:
    """True when ``markup`` is just the pressed button (a leftover Stop keyboard)."""
    buttons = [b for row in markup.inline_keyboard for b in row] if markup is not None else []
    return bool(buttons) and all(b.callback_data == data for b in buttons)


@control_router.callback_query(F.data.startswith(STOP_CALLBACK_PREFIX))
async def stop_button(callback: CallbackQuery, user_id: str) -> None:
    """The status message's Stop: cancel that turn only; queued messages still run."""
    job_id = job_id_from_token((callback.data or "")[len(STOP_CALLBACK_PREFIX):])
    cancelled = job_id is not None and await queue.cancel_current(user_id, job_id, reason="user")
    current = queue.status(user_id).current
    running = job_id is not None and current is not None and current.id == job_id
    if cancelled:
        notice = STOP_STOPPING_TEXT
    else:
        # Still running but refused: the agent is done and the answer is being delivered.
        notice = STOP_TOO_LATE_TEXT if running else STOP_FINISHED_TEXT
    try:
        await callback.answer(notice)
    except Exception as e:  # noqa: BLE001 - the query may have expired
        logger.debug("Could not answer the stop button: %s", e)
    if (
        not cancelled
        and job_id is not None
        and not running  # a still-running turn clears its own button
        and isinstance(callback.message, Message)
        and _only_button(callback.message.reply_markup, callback.data)
    ):
        # A status left behind by a turn that ended without its final edit (e.g. a restart).
        # Any other keyboard (the turn buttons on a legacy answer) is left alone.
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass


@control_router.stopped_message_generation()
async def draft_stopped(event: MessageGenerationStopped, user_id: str) -> None:
    """Telegram's Stop button on a streaming draft; the draft id is the job id (see run_turn)."""
    cancelled = await queue.cancel_current(user_id, job_id=event.draft_id, reason="user")
    logger.info("Draft %s stopped by %s: %s", event.draft_id, user_id, "turn cancelled" if cancelled else "no matching turn")


# ---------------------------------------------------------------------------
# location → time zone; the agent never sees locations
# ---------------------------------------------------------------------------
LOCATION_NO_FINDER = (
    "I can't tell the time zone from a location on this server.\n"
    "Please type it: /timezone Europe/Berlin (any city name works too)."
)


@control_router.message(F.location)
async def on_location(message: Message, user_id: str) -> None:
    """A shared location sets ``timezone.name`` (timezonefinder, optional; else the zone name is asked for)."""
    from bot import timezones
    from bot.handlers.commands import timezone_set_text
    from bot.settings import UserSettings

    location = message.location
    name = await asyncio.to_thread(timezones.zone_from_location, location.latitude, location.longitude)
    if name is None:
        await answer_md(message, LOCATION_NO_FINDER)
        return
    UserSettings(user_id).set("timezone", name, "name")
    await answer_md(message, timezone_set_text(name))


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------
TEXT_KIND = "text message"


async def prepare_text(bot: Bot, message: Message, user_id: str) -> Prepared:
    return Prepared(prompt=with_reference(message.text or "", message, _bot_id(bot)), kind=TEXT_KIND, label=TEXT_KIND)


async def _run_text(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    await run_turn(
        bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=with_reference(message.text or "", message, _bot_id(bot)),
        thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
    )


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    logger.info("Text message from %s", user_id)
    _stats().track_message_received(user_id, "text")
    thread_id = _thread_id(message)
    if questions.waiting_in(message.chat.id, thread_id):
        # An ask_user_choice question waits here: the text is its answer, not a new request.
        if await questions.answer_in_text(bot, message.chat.id, thread_id, message.text or ""):
            return
    await accept(
        message, bot, user_id, limit, label=TEXT_KIND, kind=TEXT_KIND, error_text="❌ Error processing text message",
        prepare=partial(prepare_text, bot, message, user_id), run=partial(_run_text, bot, message, user_id, limit),
    )


# ---------------------------------------------------------------------------
# voice
# ---------------------------------------------------------------------------
VOICE_KIND = "voice message"
VOICE_ERROR = "❌ Error processing voice message"


async def prepare_voice(bot: Bot, message: Message, user_id: str) -> Prepared:
    """Download, convert and transcribe; only the transcription reaches the agent (nothing is kept)."""
    media.ensure_temp_dirs()
    temp_ogg = os.path.join(media.TEMP_AUDIO, f"voice_{uuid.uuid4()}.oga")
    temp_mp3 = os.path.join(media.TEMP_AUDIO, f"voice_{uuid.uuid4()}.mp3")
    try:
        await bot.download(message.voice.file_id, destination=temp_ogg)
        await media.convert_audio(temp_ogg, temp_mp3, "ogg")
        transcription = await media.transcribe_audio(temp_mp3)
    finally:
        media.remove_files([temp_ogg, temp_mp3])
    if not transcription:
        return Prepared(
            prompt=None, kind=VOICE_KIND, label=VOICE_KIND, error_text="❌ Failed to transcribe audio",
            reason="it could not be transcribed",
        )
    shown = transcription[:512] + "..." if len(transcription) > 1500 else transcription
    bot_id = _bot_id(bot)
    return Prepared(
        prompt=with_reference(transcription, message, bot_id),
        inject=with_reference(f"(voice message, transcribed) {transcription}", message, bot_id),
        kind=VOICE_KIND, label=VOICE_KIND, speak=True, echo=f"🎙️ *Transcription:*\n{shown}",
    )


async def _run_voice(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    status = await sender.send_text("🎙️ *Transcribing audio...*")
    try:
        ctx.set_progress("transcribing voice")
        prepared = await prepare_voice(bot, message, user_id)
        if not prepared.ok:
            await sender.edit_text(status, prepared.error_text or VOICE_ERROR)
            return
        await sender.edit_text(status, prepared.echo or "🎙️")
        await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit, speak=True,
        )
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing voice message: %s", e)
        await sender.edit_text(status, VOICE_ERROR)


@router.message(F.voice)
async def on_voice(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "voice")
    await accept(
        message, bot, user_id, limit, label=VOICE_KIND, kind=VOICE_KIND, error_text=VOICE_ERROR,
        prepare=partial(prepare_voice, bot, message, user_id), run=partial(_run_voice, bot, message, user_id, limit),
    )


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
VIDEO_KIND = "video"
VIDEO_ERROR = "❌ Error processing video message"
VIDEO_MISSING = "❌ Could not process video message."


async def prepare_video(bot: Bot, message: Message, user_id: str, progress: Progress = _no_progress) -> Prepared:
    """Save the video, transcribe its audio and describe four frames (``gpt-6-luna``)."""
    is_video_note = message.video_note is not None
    video = message.video or message.video_note or message.document
    if not video:
        return Prepared(prompt=None, kind=VIDEO_KIND, label=VIDEO_KIND, error_text=VIDEO_MISSING, reason="the file is missing")
    caption = message.caption or ""
    media.ensure_temp_dirs()
    temp_video = os.path.join(media.TEMP_AUDIO, f"video_{uuid.uuid4()}.mp4")
    temp_mp3 = os.path.join(media.TEMP_AUDIO, f"video_audio_{uuid.uuid4()}.mp3")
    screenshots: list[str] = []
    try:
        await progress(None, "downloading video")
        await bot.download(video.file_id, destination=temp_video)
        saved_video = str(media.user_dir(user_id, "videos") / f"video_{uuid.uuid4()}.mp4")
        await asyncio.to_thread(shutil.copy, temp_video, saved_video)

        await progress("🎬 *Extracting audio and analyzing frames...*", "transcribing video")

        async def extract_audio() -> str | None:
            try:
                await media.convert_audio(temp_video, temp_mp3, "mp4")
            except Exception as e:  # noqa: BLE001
                logger.warning("No audio track or conversion failed: %s", e)
                return None
            return await media.transcribe_audio(temp_mp3)

        transcription, screenshots = await asyncio.gather(extract_audio(), media.extract_video_screenshots(temp_video))

        visual = ""
        saved_frames: list[str] = []
        if screenshots:
            images_dir = media.user_dir(user_id, "images")
            for sp in screenshots:
                dest = str(images_dir / f"video_frame_{uuid.uuid4()}.jpg")
                await asyncio.to_thread(shutil.copy, sp, dest)
                saved_frames.append(dest)
            await progress("🖼️ *Analyzing video frames...*", "describing frames")
            visual = await media.describe_video_screenshots(screenshots, transcription=(transcription or "") if is_video_note else "", caption=caption, user_id=user_id)

        if not (transcription or visual or saved_frames):
            return Prepared(
                prompt=None, kind=VIDEO_KIND, label=VIDEO_KIND, error_text="❌ Could not extract any content from video",
                reason="no audio or picture could be extracted",
            )

        display = "🎬 *Video analysis:*\n"
        if transcription:
            shown = transcription[:512] + "..." if len(transcription) > 1500 else transcription
            display += f"🎙️ *Audio:* {shown}\n"
        if visual:
            display += f"🖼️ *Visual:* {visual[:200]}..."
        if caption:
            display += f"\n📝 *Caption:* {caption}"
        if len(display) > 4000:
            display = display[:2048] + "..."

        parts = [f"User sent a video file. Saved to: {saved_video}"]
        if caption:
            parts.append(f"User caption: {caption}")
        if transcription:
            parts.append(f"Audio transcription from video: {transcription}")
        if visual:
            parts.append(f"Visual description of video: {visual}")
        if saved_frames:
            lines = []
            for i, sp in enumerate(saved_frames):
                label = media.FRAME_LABELS[i] if i < len(media.FRAME_LABELS) else f"frame {i + 1}"
                lines.append(f"  - Frame at {label}: {sp}")
            parts.append("Video screenshots saved:\n" + "\n".join(lines))
        prompt = with_reference("\n\n".join(parts), message, _bot_id(bot))
        return Prepared(prompt=prompt, kind=VIDEO_KIND, label=VIDEO_KIND, speak=True, echo=display)
    finally:
        media.remove_files([temp_video, temp_mp3, *screenshots])


async def _run_video(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    if not (message.video or message.video_note or message.document):
        await sender.send_text(VIDEO_MISSING, markdown=False)
        return
    status = await sender.send_text("🎬 *Processing video...*")
    try:
        prepared = await prepare_video(bot, message, user_id, _job_progress(sender, status, ctx))
        if not prepared.ok:
            await sender.edit_text(status, prepared.error_text or VIDEO_ERROR)
            return
        await sender.edit_text(status, prepared.echo or "🎬")
        await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit, speak=True,
        )
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing video message: %s", e)
        await sender.edit_text(status, VIDEO_ERROR)


async def _accept_video(message: Message, bot: Bot, user_id: str, limit: int | None, label: str) -> None:
    await accept(
        message, bot, user_id, limit, label=label, kind=VIDEO_KIND, error_text=VIDEO_ERROR,
        prepare=partial(prepare_video, bot, message, user_id), run=partial(_run_video, bot, message, user_id, limit),
    )


@router.message(F.video | F.video_note)
async def on_video(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "video")
    await _accept_video(message, bot, user_id, limit, "video message")


# ---------------------------------------------------------------------------
# audio files
# ---------------------------------------------------------------------------
AUDIO_KIND = "audio file"
AUDIO_ERROR = "❌ Error processing audio file"
AUDIO_MISSING = "❌ Could not process audio file."


async def prepare_audio(bot: Bot, message: Message, user_id: str) -> Prepared:
    """Save the file under ``audio/<original name>``; the prompt carries its metadata and path (no transcription)."""
    audio = message.audio
    if not audio:
        return Prepared(prompt=None, kind=AUDIO_KIND, label=AUDIO_KIND, error_text=AUDIO_MISSING, reason="the file is missing")
    caption = message.caption or ""
    file_name = os.path.basename(audio.file_name or f"audio_{uuid.uuid4()}.mp3")
    audio_path = str(media.user_dir(user_id, "audio") / file_name)
    await bot.download(audio.file_id, destination=audio_path)
    duration = f"{audio.duration // 60}m{audio.duration % 60}s" if audio.duration else "unknown"
    prompt = f"User sent an audio file.\nFile: {file_name}\nPath: {audio_path}\nDuration: {duration}"
    if audio.performer:
        prompt += f"\nPerformer: {audio.performer}"
    if audio.title:
        prompt += f"\nTitle: {audio.title}"
    if caption:
        prompt += f"\nUser message: {caption}"
    return Prepared(prompt=with_reference(prompt, message, _bot_id(bot)), kind=AUDIO_KIND, label=AUDIO_KIND)


async def _run_audio(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    if not message.audio:
        await sender.send_text(AUDIO_MISSING, markdown=False)
        return
    status = await sender.send_text("🎵 *Saving audio file...*")
    try:
        ctx.set_progress("saving audio")
        prepared = await prepare_audio(bot, message, user_id)
        await sender.edit_text(status, "🎵 *Audio saved, processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🎵 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing audio file: %s", e)
        await sender.edit_text(status, AUDIO_ERROR)


@router.message(F.audio)
async def on_audio(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "audio")
    await accept(
        message, bot, user_id, limit, label=AUDIO_KIND, kind=AUDIO_KIND, error_text=AUDIO_ERROR,
        prepare=partial(prepare_audio, bot, message, user_id), run=partial(_run_audio, bot, message, user_id, limit),
    )


# ---------------------------------------------------------------------------
# photos and albums
# ---------------------------------------------------------------------------
IMAGES_ERROR = "❌ Could not process the image(s)."


def _images_kind(count: int) -> str:
    return "photo" if count == 1 else f"photo album ({count} images)"


async def prepare_images(
    bot: Bot, message: Message, user_id: str, refs: list[Any], caption: str | None, progress: Progress = _no_progress,
) -> Prepared:
    """Photos (one message or a flushed album) go to the agent as image blocks, not descriptions.

    Each picture is kept twice under ``images/``: ``image_<uuid>.jpg`` at full resolution (the
    path the prompt names, for edits and re-sends) and ``image_<uuid>_v.jpg``, downscaled to
    ``MAX_IMAGE_RESOLUTION_VISION`` and under the API size limit, which is what the ``image_ref``
    points at and what the transcript stores. A picture that cannot be prepared is named as such.
    """
    media.ensure_temp_dirs()
    temp_files: list[str] = []
    lines: list[str] = []
    media_blocks: list[dict] = []
    kind = _images_kind(len(refs))
    try:
        images_dir = media.user_dir(user_id, "images")
        for i, ref in enumerate(refs, 1):
            try:
                await progress(None, f"preparing image {i}/{len(refs)}")
                ext = os.path.splitext(getattr(ref, "file_name", "") or "")[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    ext = ".jpg"
                downloaded = os.path.join(media.TEMP_PHOTOS, f"photo_{uuid.uuid4()}{ext}")
                temp_files.append(downloaded)
                await bot.download(ref.file_id, destination=downloaded)
                stem = f"image_{uuid.uuid4()}"
                permanent = str(images_dir / f"{stem}.jpg")
                await asyncio.to_thread(media.prepare_downloaded_image_for_vision, downloaded, permanent)
                vision_copy = str(images_dir / f"{stem}_v.jpg")
                await asyncio.to_thread(media.save_vision_copy, permanent, vision_copy)
                lines.append(f"Image {i}: {permanent}")
                media_blocks.append(_content().image_ref(vision_copy, caption=f"Image {i}"))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("Error processing photo %d: %s", i, e)
                lines.append(f"Image {i}: could not be processed ({type(e).__name__})")
    finally:
        media.remove_files(temp_files)

    if not media_blocks:
        return Prepared(prompt=None, kind=kind, label=kind, error_text=IMAGES_ERROR, reason="no picture could be read")
    n = len(refs)
    if caption is None:
        head = f"User sent {n} photo(s)."
    else:
        head = f"{caption}\n\nUser attached {n} image(s) to this message."
    text = f"{head} They are shown below as images and saved in your workspace:\n" + "\n".join(lines) + f"\n\n{IMAGE_EMBED_HINT}"
    return Prepared(prompt=_prompt_with_reference(text, media_blocks, message, _bot_id(bot)), kind=kind, label=kind)


async def _run_images(
    bot: Bot, message: Message, user_id: str, limit: int | None, refs: list[Any], caption: str | None,
    status: Message | None, ctx: JobContext,
) -> None:
    """The queued form of ``prepare_images``: a status message while the pictures are prepared, then the turn."""
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    if status is not None:
        await sender.edit_text(status, "🖼️ *Preparing images...*")
    else:
        status = await sender.send_text("🖼️ *Preparing images...*")
    try:
        prepared = await prepare_images(bot, message, user_id, refs, caption, _job_progress(sender, None, ctx))
        if not prepared.ok:
            await sender.edit_text(status, prepared.error_text or IMAGES_ERROR, markdown=False)
            return
        await sender.edit_text(status, "🤖 *Processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🤖 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        trace_id = str(uuid.uuid4())
        logger.exception("Error processing images (trace %s): %s", trace_id, e)
        await sender.edit_text(status, f"❌ An error occurred while processing the images. Trace ID: {trace_id}", markdown=False)


async def _accept_images(
    message: Message, bot: Bot, user_id: str, limit: int | None, refs: list[Any], caption: str | None, *,
    label: str, status: Message | None = None, seq: int | None = None,
) -> None:
    await accept(
        message, bot, user_id, limit, label=label, kind=_images_kind(len(refs)), error_text=IMAGES_ERROR,
        prepare=partial(prepare_images, bot, message, user_id, list(refs), caption),
        run=partial(_run_images, bot, message, user_id, limit, list(refs), caption, status), seq=seq,
    )


@dataclass
class _Album:
    message: Message
    user_id: str
    limit: int | None
    refs: list[Any] = field(default_factory=list)
    caption: str | None = None
    waiting: Message | None = None
    flush_task: asyncio.Task | None = None
    seq: int = field(default_factory=next_seq)  # the first picture's arrival: the album keeps that place


_albums: dict[tuple[int, str], _Album] = {}


async def _flush_album(bot: Bot, key: tuple[int, str]) -> None:
    try:
        await asyncio.sleep(MEDIA_GROUP_TIMEOUT)
    except asyncio.CancelledError:
        return
    album = _albums.pop(key, None)
    if album is None or not album.refs:
        return
    _stats().track_media_group_processed(album.user_id, len(album.refs))
    label = f"photo album ({len(album.refs)} images)"
    if inject_target(album.user_id, _thread_id(album.message)) is not None and album.waiting is not None:
        # The album joins the running turn: the "waiting for images" line has nothing more to say.
        await delete_quietly(album.waiting)
        album.waiting = None
    await _accept_images(
        album.message, bot, album.user_id, album.limit, album.refs, album.caption, label=label, status=album.waiting, seq=album.seq,
    )


async def collect_album(message: Message, bot: Bot, user_id: str, limit: int | None, ref: Any, caption: str | None) -> None:
    key = (message.chat.id, str(message.media_group_id))
    album = _albums.get(key)
    if album is None:
        album = _Album(message=message, user_id=user_id, limit=limit)
        _albums[key] = album
        album.waiting = await answer_md(message, "🖼️ *Image media group received... Waiting for images...*")
    if caption and album.caption is None:
        album.caption = caption
    album.refs.append(ref)
    if album.waiting is not None:
        try:
            await album.waiting.edit_text(f"🖼️ *Image {len(album.refs)} received... Waiting for other images...*", parse_mode="Markdown")
        except Exception:  # noqa: BLE001
            pass
    if album.flush_task is not None:
        album.flush_task.cancel()
    album.flush_task = asyncio.create_task(_flush_album(bot, key))


@router.message(F.photo)
async def on_photo(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "photo")
    ref = message.photo[-1]
    if message.media_group_id:
        await collect_album(message, bot, user_id, limit, ref, message.caption)
        return
    await _accept_images(message, bot, user_id, limit, [ref], message.caption, label="photo")


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def _cap_document_text(text: str, limit: int, hint: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…(truncated: {hint})"


async def _describe_fallback(
    progress: Progress, user_id: str, ext: str, doc_type: str, file_name: str, saved: str, caption: str | None, reason: str,
) -> str:
    """The pre-F7 path: a model's analysis of the document goes in the prompt (a PDF over the
    block limits, text that could not be extracted or is too long to include)."""
    default = "Analyze this document and describe its contents in detail."
    question = default if caption is None else f"{default} When you are done, answer the following question: {caption}"
    await progress(f"🤖 *Analyzing {doc_type} content...*", None)
    description = await media.describe_document(question, saved)
    _stats().track_describe_used(user_id, media.describe_type_for(ext))
    return (
        f"{caption or default}\n\nUser attached a {doc_type} document to this message.\nFile: {file_name}\nSaved to: {saved}\n"
        f"{reason}\n\nHere is the analysis of document contents:\n\n{description}"
    )


async def _document_prompt(
    progress: Progress, user_id: str, ext: str, doc_type: str, file_name: str, saved: str, caption: str | None,
) -> tuple[str, list]:
    """(prompt text, media blocks) for a saved document, by extension: a PDF within the limits
    as a ``document_ref``; text files and extracted docx/xlsx/json text in the prompt (capped);
    everything else through ``_describe_fallback``."""
    user_line = f"\nUser message: {caption}" if caption else ""
    if ext == ".pdf":
        try:
            pages = await asyncio.to_thread(media.pdf_page_count, saved)
        except Exception as e:  # noqa: BLE001 - not a PDF pypdf can read
            logger.warning("Could not read %s with pypdf: %s", saved, e)
            pages = None
        if pages is not None and media.pdf_fits_as_block(saved, pages):
            text = (
                f"User sent a PDF: {file_name} ({pages} pages), saved at {saved}.{user_line}\n"
                "Read it from the attached document block; read_document gives it to you again later."
            )
            return text, [_content().document_ref(saved, title=file_name)]
        if pages is None:
            reason = "The PDF could not be attached directly (it could not be parsed as a PDF)."
        else:
            size_mb = os.path.getsize(saved) / (1024 * 1024)
            reason = (
                f"The PDF was too large to attach directly ({pages} pages, {size_mb:.1f} MB; the limits are "
                f"{media.PDF_MAX_PAGES} pages and {media.PDF_MAX_BYTES // (1024 * 1024)} MB), so it was analyzed instead."
            )
        return await _describe_fallback(progress, user_id, ext, doc_type, file_name, saved, caption, reason), []
    if ext in media.TEXT_EXTENSIONS:
        content = await asyncio.to_thread(media.extract_document_text, saved)
        shown = _cap_document_text(content, TEXT_PROMPT_CAP, "use read_file for the rest")
        text = (
            f"User sent a {doc_type} file: {file_name} ({len(content)} characters), saved at {saved}.{user_line}\n\n"
            f"File content:\n<file_content>\n{shown}\n</file_content>"
        )
        return text, []
    if ext in media.OFFICE_EXTENSIONS:
        try:
            extracted = await asyncio.to_thread(media.extract_document_text, saved)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not extract text from %s: %s", saved, e)
            extracted = None
        if extracted is not None and len(extracted) <= EXTRACTED_TEXT_CAP:
            text = (
                f"User sent a {doc_type} document: {file_name}, saved at {saved}.{user_line}\n\n"
                f"Extracted text:\n<document_text>\n{extracted}\n</document_text>"
            )
            return text, []
        reason = (
            "The document's text could not be extracted, so it was analyzed instead." if extracted is None
            else f"The document's text ({len(extracted)} characters) is too long to include, so it was analyzed instead; read_file / execute_python give you the details."
        )
        return await _describe_fallback(progress, user_id, ext, doc_type, file_name, saved, caption, reason), []
    return await _describe_fallback(progress, user_id, ext, doc_type, file_name, saved, caption, ""), []


def _document_names(message: Message) -> tuple[str, str, str]:
    """(file name, extension, type) of a document: the name lowercased (re-uploads overwrite)."""
    document = message.document
    file_name = os.path.basename((document.file_name or f"document_{uuid.uuid4()}").lower())
    ext = os.path.splitext(file_name)[1].lower()
    return file_name, ext, ext.replace(".", "").upper()


async def prepare_document(bot: Bot, message: Message, user_id: str, progress: Progress = _no_progress) -> Prepared:
    """Save the document under ``documents/<lowercased name>`` and build its block prompt (``_document_prompt``)."""
    file_name, ext, doc_type = _document_names(message)
    await progress(None, "reading document")
    saved = str(media.user_dir(user_id, "documents") / file_name)
    await bot.download(message.document.file_id, destination=saved)
    text, media_blocks = await _document_prompt(progress, user_id, ext, doc_type, file_name, saved, message.caption)
    kind = f"{doc_type} document"
    return Prepared(prompt=_prompt_with_reference(text, media_blocks, message, _bot_id(bot)), kind=kind, label=kind)


async def _run_document(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    """A document goes to the agent as a block prompt (``prepare_document``) after a status message."""
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    _, _, doc_type = _document_names(message)
    status = await sender.send_text(f"📄 *Processing {doc_type} document...*")
    try:
        prepared = await prepare_document(bot, message, user_id, _job_progress(sender, status, ctx))
        await sender.edit_text(status, "🤖 *Processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prepared.prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🤖 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        trace_id = str(uuid.uuid4())
        logger.exception("Error processing document (trace %s): %s", trace_id, e)
        await sender.edit_text(status, f"❌ An error occurred while processing the document. Trace ID: {trace_id}", markdown=False)


@router.message(F.document)
async def on_document(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    file_name = (message.document.file_name or "").lower()
    ext = os.path.splitext(file_name)[1].lower()
    if ext in media.VIDEO_EXTENSIONS:
        _stats().track_message_received(user_id, "video")
        await _accept_video(message, bot, user_id, limit, "video file")
        return
    if ext in IMAGE_EXTENSIONS:
        _stats().track_message_received(user_id, "photo")
        if message.media_group_id:
            await collect_album(message, bot, user_id, limit, message.document, message.caption)
            return
        await _accept_images(message, bot, user_id, limit, [message.document], message.caption, label="image file")
        return
    if ext not in media.DOCUMENT_EXTENSIONS:
        supported = ", ".join(e.replace(".", "").upper() for e in media.DOCUMENT_EXTENSIONS)
        images = "/".join(e.replace(".", "").upper() for e in IMAGE_EXTENSIONS)
        await message.answer(f"❌ Only {supported} documents and {images} images are supported.")
        return
    _stats().track_message_received(user_id, "document")
    kind = f"{ext.replace('.', '').upper()} document"
    await accept(
        message, bot, user_id, limit, label=kind, kind=kind, error_text="❌ An error occurred while processing the document.",
        prepare=partial(prepare_document, bot, message, user_id), run=partial(_run_document, bot, message, user_id, limit),
    )
