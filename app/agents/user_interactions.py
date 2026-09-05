import asyncio
import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv

from models import TTS_MODEL
from reminders_store import reminders_store
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

class UserInteractions:
    def __init__(self, user_id: str, sender):
        """
        Args:
            user_id: chat id of the user as a string
            sender: bot.sender.ChatSender bound to the user's chat (and forum thread)
        """
        self.user_id = user_id
        self.sender = sender
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
                "name": "schedule_reminder",
                "description": "Schedule a reminder for the user or agent. Can be one-time or periodic. For user reminders, a message will be sent to the user at the specified time. For agent reminders, the agent will process the reminder at the specified time and take appropriate action.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reminder_text": {
                            "type": "string",
                            "description": "The text of the reminder message"
                        },
                        "reminder_time": {
                            "type": "string",
                            "description": "The time for the reminder in ISO format (YYYY-MM-DD HH:MM:SS) in UTC+0 timezone, or time when it start frirst time for periodic reminders"
                        },
                        "reminder_type": {
                            "type": "string",
                            "description": "Type of reminder: 'user' for user notifications, 'agent' for agent tasks",
                            "enum": ["user", "agent"]
                        },
                        "is_periodic": {
                            "type": "boolean",
                            "description": "Whether this is a periodic reminder or not.",
                            "default": False
                        },
                        "period_type": {
                            "type": "string",
                            "description": "Type of period for periodic reminders, repeat every: 'hourly', 'daily', 'weekly', 'monthly'",
                            "enum": ["hourly", "daily", "weekly", "monthly"]
                        },
                        "period_interval": {
                            "type": "integer",
                            "description": "Interval for the period (e.g., 4 for every 4 hours if period_type is 'hourly')",
                            "minimum": 1
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
                period_interval
            )
        elif tool_name == "send_file_content_to_user_via_telegram":
            return await self._send_file_to_user(
                tool_args["file_content"],
                tool_args["filename"],
                tool_args.get("is_base64", False)
            )
        return f"Unknown tool: {tool_name}"

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

    async def _schedule_reminder(self, reminder_text: str, reminder_time: str, reminder_type: str = "user", is_periodic: bool = False, period_type: str = None, period_interval: int = None) -> str:
        """Schedule a reminder for the user or agent"""
        try:
            reminder_datetime = datetime.fromisoformat(reminder_time)
            if reminder_datetime.tzinfo is None:
                reminder_datetime = reminder_datetime.replace(tzinfo=timezone.utc)
            current_time = datetime.now(timezone.utc)

            if reminder_datetime <= current_time:
                return "Error: Reminder time must be in the future"

            reminder_data = {
                "user_id": self.user_id,
                "text": reminder_text,
                "time": reminder_datetime.isoformat(),
                "type": reminder_type,
                "status": "pending",
                "created_at": current_time.isoformat(),
                "is_periodic": bool(is_periodic),
                "period_type": period_type if is_periodic else None,
                "period_interval": int(period_interval) if is_periodic else None,
                "last_triggered": None,
                "next_trigger": reminder_datetime.isoformat() if is_periodic else None,
                "enabled": True,
            }
            await reminders_store.add(self.user_id, reminder_data)

            formatted_time = reminder_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
            periodic_info = ""
            if is_periodic:
                periodic_info = f"\nThis is a periodic reminder that will repeat every {period_interval} {period_type}"

            if reminder_type == "user":
                await self._send_user_message(f"✅ Reminder scheduled for {formatted_time}:{periodic_info}\n{reminder_text}")
            else:
                await self._send_user_message(f"✅ Agent task scheduled for {formatted_time}:{periodic_info}\n{reminder_text}")

            return f"Successfully scheduled {reminder_type} reminder for {formatted_time}{periodic_info}"

        except ValueError:
            return "Error: Invalid datetime format. Please use YYYY-MM-DD HH:MM:SS format in UTC timezone"
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
