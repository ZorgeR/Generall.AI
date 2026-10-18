import asyncio
import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv

from bot import timezones
from bot.settings import UserSettings
from models import TTS_MODEL
from reminders_store import WEEKDAY_NAMES, clean_weekdays, first_allowed_trigger, reminders_store
from voice import VoiceManager

load_dotenv()

logger = logging.getLogger(__name__)

# Define valid reaction emojis
VALID_REACTIONS = [
    "👍", "👎", "❤", "🔥", "🥰", "👏", "🎉", "🤩", "🤔", "🤯", "🤬", "😱", "🤮",
    "💩", "🥱", "🥴", "😭", "😂", "🤣", "🌚", "🌭", "💯", "🤙", "🤝", "🎃", "🎄",
    "💋", "🎯", "🏆", "⚡", "🌟", "💔", "🖕", "💘", "🎵", "🤓", "👻", "👨‍💻", "👀",
    "🦄", "🦅", "🦋", "🧨", "🎸", "🌶", "⚔", "🛡", "🧲", "🎲", "🎳", "🎯", "🧩",
    "🎨", "🎭", "🎪", "🎫", "🎟", "🎪", "🎭", "🎨", "🎯", "🎲", "🎳", "🎯", "✅"
]

APPROVED_REACTIONS = [
    "👍", "👎", "❤", "🔥", "🥰", "👏", "🎉", "🤩", "🤔", "🤯","😱", "😂", "⚡", "🏆", "💯"
]

# Appended to the static system prompt inside <capabilities> (ChainOfThoughtAgent._provider_notes).
# Static text only: per-turn facts (the current local time, the zone itself) are in the <context> block.
PROMPT_NOTES = (
    "Time zone and scheduling: the <context> block shows the current UTC time and, once known, the user's "
    "local time and zone. When it says the time zone is unknown, infer it from the user's city or country if "
    "they mentioned one, otherwise ask once (briefly, together with your answer), and save it with "
    "set_user_timezone as soon as you know it; never ask again once it is set. Schedule reminders in the "
    "user's local time (schedule_reminder reads reminder_time in the user's zone by default) and confirm "
    "times to the user in their local time. For recurring monitoring jobs (\"check X every morning and tell "
    "me only if...\") use a periodic agent reminder (reminder_type agent, weekdays when only some days "
    "matter); when such a scheduled task runs and finds nothing new or worth telling the user, reply with "
    "exactly <no_update/> and nothing else, so the user is not disturbed."
)


class UserInteractions:
    prompt_notes = PROMPT_NOTES

    def __init__(self, user_id: str, sender):
        """
        Args:
            user_id: chat id of the user as a string
            sender: bot.sender.ChatSender bound to the user's chat (and forum thread)
        """
        self.user_id = user_id
        self.sender = sender
        # the zone set_user_timezone accepted this turn, recorded before its save finishes, so a
        # schedule_reminder of the same (concurrent) tool batch uses it
        self._zone_override: str | None = None
        self.voice_manager = VoiceManager()
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        self.tools_schema = [
            {
                "name": "send_user_telegram_message",
                "description": "Send an intermediate message to the user via Telegram while processing their request. Use this to keep the user informed about progress, thoughts, or status updates.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the user"
                        }
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "send_voice_message",
                "description": "If you want to send a voice message to the user, use this tool. Text will be converted to speech and sent as a voice message to the user. It's useful if you want to support the user or say something important, or if he himself asked to answer by voice.",
                "input_schema": {

                    "type": "object",
                    "properties": {

                        "text": {
                            "type": "string",
                            "description": "The text to convert to speech and send as a voice message to the user"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "set_message_reaction",
                "description": "Set a reaction emoji on the user's message. Use this to provide quick feedback or acknowledgment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "emoji": {
                            "type": "string",
                            "description": "The emoji reaction to set. Allowed values: " + ", ".join(APPROVED_REACTIONS),
                            "enum": [
                                "👍", "👎", "❤", "🔥", "🥰", "👏", "🎉", "🤩", "🤔", "🤯","😱", "😂", "⚡", "🏆", "💯"
                            ]
                        }
                    },
                    "required": ["emoji"]
                }
            },
            {
                "name": "set_user_timezone",
                "description": "Save the user's time zone so that times are shown and reminders are scheduled in their local time. Call it once when the user tells you their city, country or zone, or when the context says the time zone is unknown and you inferred it. Accepts an IANA name (Europe/Berlin, America/New_York) or a big city / country name (Berlin, Kyiv, Japan). Returns the user's local time.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA time zone name such as 'Europe/Berlin', or a city/country name such as 'Berlin'"
                        }
                    },
                    "required": ["timezone"]
                }
            },
            {
                "name": "schedule_reminder",
                "description": (
                    "Schedule a reminder for the user or a task for yourself, one-time or periodic. "
                    "'user' reminders: the text is sent to the user at that time (with snooze buttons). "
                    "'agent' tasks: at that time you run again with the text as your task and the user is away, so "
                    "do the work with your tools and report in your answer; when a scheduled task finds nothing new or "
                    "worth telling the user, answer exactly <no_update/> and nothing else (nothing is sent). "
                    "Use periodic agent tasks for monitoring jobs (\"every weekday morning check X and tell me only if ...\"). "
                    "reminder_time is read in the user's local time zone (see the context block) unless it carries a Z or "
                    "UTC offset or you pass timezone; tell the user the time in their local time."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reminder_text": {
                            "type": "string",
                            "description": "The reminder message for the user, or for an agent task the full task description (what to check, what to compare against, when to stay quiet)"
                        },
                        "reminder_time": {
                            "type": "string",
                            "description": "When to fire, 'YYYY-MM-DD HH:MM[:SS]' in the user's local time zone (or with a trailing Z / +HH:MM offset); for periodic reminders the first time"
                        },
                        "timezone": {
                            "type": "string",
                            "description": "IANA zone reminder_time is given in. Default: the user's saved time zone, else UTC"
                        },
                        "reminder_type": {
                            "type": "string",
                            "description": "'user' sends the text to the user; 'agent' runs you with the text as a task",
                            "enum": ["user", "agent"]
                        },
                        "is_periodic": {
                            "type": "boolean",
                            "description": "Whether this is a periodic reminder or not.",
                            "default": False
                        },
                        "period_type": {
                            "type": "string",
                            "description": "Type of period for periodic reminders, repeat every: 'hourly', 'daily', 'weekly', 'monthly'. Daily/weekly/monthly repeat at the same local time (DST-safe)",
                            "enum": ["hourly", "daily", "weekly", "monthly"]
                        },
                        "period_interval": {
                            "type": "integer",
                            "description": "Interval for the period (e.g., 4 for every 4 hours if period_type is 'hourly')",
                            "minimum": 1
                        },
                        "weekdays": {
                            "type": "array",
                            "description": "Daily periodic reminders only: the weekdays that fire, 0 = Monday .. 6 = Sunday (e.g. [0,1,2,3,4] for weekdays); other days are skipped",
                            "items": {"type": "integer", "minimum": 0, "maximum": 6},
                            "minItems": 1,
                            "maxItems": 7
                        }
                    },
                    "required": ["reminder_text", "reminder_time", "reminder_type"]
                }
            },
            {
                "name": "send_file_content_to_user_via_telegram",
                "description": "Send a file to the user via Telegram using file text content or base64-encoded data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_content": {
                            "type": "string",
                            "description": "The content of the file, either as plain text or base64-encoded string"
                        },
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to be sent"
                        },
                        "is_base64": {
                            "type": "boolean",
                            "description": "Whether the file_content is base64 encoded",
                            "default": False
                        }
                    },
                    "required": ["file_content", "filename"]
                }
            },
            {
                "name": "ask_user_choice",
                "description": "Ask the user to pick one option by tapping a button in Telegram and wait for the answer. Use it when you genuinely need the user's decision before continuing: to confirm an irreversible or outward-facing action the user did not explicitly ask for (sending an SMS, deleting files, uploading or publishing something, spending money), or to choose between a few clearly different ways to proceed. Do not use it for questions you can settle yourself, for open-ended questions (ask those in your reply), or when the user is unlikely to be present (scheduled agent tasks). It blocks until the user taps an option or the timeout passes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question shown above the buttons, short and self-contained (say what each choice will do)"
                        },
                        "options": {
                            "type": "array",
                            "description": "2-8 short button labels, e.g. [\"Send it\", \"Cancel\"]",
                            "items": {"type": "string", "maxLength": 64},
                            "minItems": 2,
                            "maxItems": 8
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "How long to wait for a tap, 30-900 seconds (default 300)",
                            "minimum": 30,
                            "maximum": 900,
                            "default": 300
                        }
                    },
                    "required": ["question", "options"]
                }
            }
        ]

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool by name with given arguments"""
        if tool_name == "send_user_telegram_message":
            return await self._send_user_message(tool_args["message"])
        elif tool_name == "send_voice_message":
            return await self._send_voice_message(tool_args["text"])
        elif tool_name == "set_message_reaction":
            return await self._set_message_reaction(tool_args["emoji"])
        elif tool_name == "set_user_timezone":
            return await self._set_user_timezone(tool_args.get("timezone"))
        elif tool_name == "schedule_reminder":
            # Safely get optional parameters with defaults
            is_periodic = tool_args.get("is_periodic", False)
            period_type = tool_args.get("period_type", None)
            period_interval = tool_args.get("period_interval", None)

            if is_periodic and (period_type is None or period_interval is None):
                return "Error: period_type and period_interval are required for periodic reminders"

            return await self._schedule_reminder(
                tool_args["reminder_text"],
                tool_args["reminder_time"],
                tool_args["reminder_type"],
                is_periodic,
                period_type,
                period_interval,
                timezone_name=tool_args.get("timezone"),
                weekdays=tool_args.get("weekdays"),
            )
        elif tool_name == "send_file_content_to_user_via_telegram":
            return await self._send_file_to_user(
                tool_args["file_content"],
                tool_args["filename"],
                tool_args.get("is_base64", False)
            )
        elif tool_name == "ask_user_choice":
            return await self._ask_user_choice(
                tool_args.get("question"),
                tool_args.get("options"),
                tool_args.get("timeout_seconds")
            )
        return f"Unknown tool: {tool_name}"

    async def _ask_user_choice(self, question: Any, options: Any, timeout_seconds: Any = None) -> str:
        """Ask a multiple-choice question with buttons and wait for the tap (see bot/questions.py)"""
        from bot import questions

        if self.sender is None or getattr(self.sender, "bot", None) is None:
            return "Error: there is no Telegram chat to ask in."
        try:
            question = questions.clean_question(question)
            choices = questions.clean_options(options)
            timeout = questions.clamp_timeout(timeout_seconds)
        except ValueError as e:
            return f"Error: {e}"
        try:
            # Cancellation (/cancel, turn deadline) propagates: ask() marks the message and re-raises.
            answer = await questions.ask(self.sender, question, choices, timeout)
        except Exception as e:
            logger.warning("ask_user_choice failed for %s: %s", self.user_id, e)
            return f"Error: could not ask the user: {e}"
        if answer is None:
            return f"No answer within {timeout} seconds. Continue without it: take the safe default or ask in your final reply."
        if getattr(answer, "tapped", True) is False:
            return questions.typed_result(answer)  # the user typed a reply instead of tapping
        return f'User chose: "{answer}"'

    async def _send_user_message(self, message: str) -> str:
        """Send a message to the user via Telegram"""
        try:
            logger.info("Sending intermediate message to user %s", self.user_id)
            await self.sender.send_markdown(message)
            return "Successfully sent message to user."
        except Exception as e:
            return f"Error sending message: {str(e)}"

    async def _send_voice_message(self, text: str) -> str:
        """Convert text to speech and send it as a voice message to the user"""
        try:
            voice_id = self.voice_manager.get_user_voice(self.user_id)
            api_key = self.elevenlabs_api_key

            def _tts() -> bytes:
                from elevenlabs.client import ElevenLabs

                client = ElevenLabs(api_key=api_key)
                stream = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=TTS_MODEL
                )
                return b"".join(stream)

            audio = await asyncio.to_thread(_tts)
            await self.sender.send_voice(audio, caption="Voice message")
            return "Requested voice message has been successfully sent to user."
        except Exception as e:
            return f"Error sending requested voice message: {str(e)}"

    async def _set_message_reaction(self, emoji: str) -> str:
        """Set a reaction emoji on the user's message"""
        try:
            if await self.sender.react(emoji):
                return f"Successfully set reaction {emoji} on user's message."
            return "No user message to react to in this context (for example a scheduled task)."
        except Exception as e:
            return f"Error setting reaction {emoji}: {str(e)}"

    def _user_zone(self):
        """The zone set_user_timezone accepted this turn (even while its save still runs), else the
        user's saved zone (``timezone.name``), read fresh so a zone set by /timezone counts."""
        if self._zone_override:
            return timezones.zone_named(self._zone_override)
        try:
            return timezones.zone_of(UserSettings(self.user_id))
        except Exception as e:  # noqa: BLE001 - a settings read must not break a tool call
            logger.warning("Could not read the time zone of %s: %s", self.user_id, e)
            return None

    async def _set_user_timezone(self, name: Any) -> str:
        """Validate and save the user's time zone; the answer carries the local time."""
        zone_name = timezones.valid_zone(name)
        if zone_name is None:
            return (
                f"Error: unknown time zone {str(name or '')!r}. Use an IANA name such as Europe/Berlin or "
                "America/New_York, or a big city / country name."
            )

        def _save() -> None:
            UserSettings(self.user_id).set("timezone", zone_name, "name")

        # Before the first await: the tool calls of one message run concurrently, and a
        # schedule_reminder next to this call must not read the settings before the save lands.
        previous, self._zone_override = self._zone_override, zone_name
        try:
            await asyncio.to_thread(_save)
        except Exception as e:  # noqa: BLE001
            self._zone_override = previous
            return f"Error saving the time zone: {e}"
        zone = timezones.zone_named(zone_name)
        return f"Time zone saved: {zone_name}. The user's local time is now {timezones.format_local(datetime.now(timezone.utc), zone)}."

    async def _schedule_reminder(
        self,
        reminder_text: str,
        reminder_time: str,
        reminder_type: str = "user",
        is_periodic: bool = False,
        period_type: str = None,
        period_interval: int = None,
        timezone_name: str | None = None,
        weekdays: Any = None,
    ) -> str:
        """Schedule a reminder for the user or a task for the agent.

        ``reminder_time`` is local time in ``timezone_name`` (default: the user's zone, else UTC)
        unless it carries its own offset; the record stores UTC plus the zone it was scheduled in,
        so the scheduler can step daily periods in local wall-clock time and apply ``weekdays``."""
        try:
            if timezone_name:
                zone_label = timezones.valid_zone(timezone_name)
                if zone_label is None:
                    return f"Error: unknown time zone {timezone_name!r}. Use an IANA name such as Europe/Berlin."
                zone = timezones.zone_named(zone_label)
            else:
                # Yield once first: every call of the tool batch is already scheduled, so a
                # set_user_timezone that comes after this call in the same message records its
                # zone (synchronously, before its own first await) before we read it.
                await asyncio.sleep(0)
                zone = self._user_zone()
            try:
                reminder_datetime = timezones.to_utc(reminder_time, zone)
            except ValueError:
                return "Error: Invalid datetime format. Please use YYYY-MM-DD HH:MM[:SS] (local time, or with a Z / +HH:MM offset)"
            current_time = datetime.now(timezone.utc)

            try:
                days = clean_weekdays(weekdays)
            except ValueError as e:
                return f"Error: {e}"
            if days is not None and not (is_periodic and period_type == "daily"):
                return "Error: weekdays only apply to periodic daily reminders (is_periodic true, period_type 'daily')"
            if days is not None:
                reminder_datetime = first_allowed_trigger(reminder_datetime, days, zone)

            if reminder_datetime <= current_time:
                return "Error: Reminder time must be in the future"

            reminder_data = {
                "user_id": self.user_id,
                "text": reminder_text,
                "time": reminder_datetime.isoformat(),
                "timezone": timezones.zone_name(zone) if zone is not None else "",
                "type": reminder_type,
                "status": "pending",
                "created_at": current_time.isoformat(),
                "is_periodic": bool(is_periodic),
                "period_type": period_type if is_periodic else None,
                "period_interval": int(period_interval) if is_periodic else None,
                "weekdays": days,
                "last_triggered": None,
                "next_trigger": reminder_datetime.isoformat() if is_periodic else None,
                "enabled": True,
            }
            await reminders_store.add(self.user_id, reminder_data)

            formatted_time = timezones.format_local(reminder_datetime, zone)
            periodic_info = ""
            if is_periodic:
                periodic_info = f"\nThis is a periodic reminder that will repeat every {period_interval} {period_type}"
                if days is not None:
                    periodic_info += " on " + ", ".join(WEEKDAY_NAMES[d] for d in days)

            if reminder_type == "user":
                await self._send_user_message(f"✅ Reminder scheduled for {formatted_time}:{periodic_info}\n{reminder_text}")
            else:
                await self._send_user_message(f"✅ Agent task scheduled for {formatted_time}:{periodic_info}\n{reminder_text}")

            return f"Successfully scheduled {reminder_type} reminder for {formatted_time}{periodic_info}"

        except Exception as e:
            return f"Error scheduling reminder: {str(e)}"

    async def _send_file_to_user(self, file_content: str, filename: str, is_base64: bool = False) -> str:
        """Send a file to the user via Telegram"""
        try:
            if is_base64:
                try:
                    data = base64.b64decode(file_content)
                except Exception as e:
                    return f"Error decoding base64 content: {str(e)}"
            else:
                data = file_content.encode("utf-8")

            await self.sender.send_document(data, filename=os.path.basename(filename) or "file.txt")
            return f"Successfully sent file '{filename}' to user."
        except Exception as e:
            return f"Error sending file: {str(e)}"
