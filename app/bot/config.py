"""Runtime configuration read from the environment.

``load_dotenv()`` must run before this module is imported (main_bot.py does
that). Values are never validated here so the module can be imported in tests;
``validate()`` is called once at startup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_list(name: str) -> frozenset[str]:
    value = os.getenv(name) or ""
    return frozenset(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Config:
    bot_token: str | None
    chat_ids: frozenset[str]
    admin_id: str | None
    allow_all_users: bool
    invite_limit: int
    use_local_api: bool
    local_api_url: str
    streaming_enabled: bool
    drop_pending_updates: bool
    max_concurrent_turns: int
    turn_timeout_seconds: int
    thread_pool_size: int
    elevenlabs_api_key: str | None
    max_image_resolution_vision: int
    data_dir: str = "data"

    def validate(self) -> list[str]:
        """Return a list of fatal configuration problems (empty when fine)."""
        problems = []
        if not self.bot_token:
            problems.append("TELEGRAM_BOT_TOKEN is not set")
        if not self.chat_ids and not self.allow_all_users and not self.admin_id:
            problems.append(
                "No one can talk to the bot: set TELEGRAM_CHAT_ID, TELEGRAM_ADMIN_ID "
                "or TELEGRAM_ALLOWED_ALL_USERS=true"
            )
        return problems


def load_config() -> Config:
    return Config(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        chat_ids=_env_list("TELEGRAM_CHAT_ID"),
        admin_id=(os.getenv("TELEGRAM_ADMIN_ID") or "").strip() or None,
        allow_all_users=_env_bool("TELEGRAM_ALLOWED_ALL_USERS", False),
        invite_limit=_env_int("INVITE_LIMIT", 3),
        use_local_api=_env_bool("TELEGRAM_USE_LOCAL_API", False),
        local_api_url=(os.getenv("TELEGRAM_LOCAL_API_URL") or "http://localhost:8081").rstrip("/"),
        streaming_enabled=_env_bool("STREAMING_ENABLED", False),
        drop_pending_updates=_env_bool("DROP_PENDING_UPDATES", True),
        max_concurrent_turns=max(1, _env_int("MAX_CONCURRENT_TURNS", 8)),
        turn_timeout_seconds=max(60, _env_int("TURN_TIMEOUT_SECONDS", 1800)),
        thread_pool_size=max(8, _env_int("THREAD_POOL_SIZE", 32)),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY") or None,
        max_image_resolution_vision=_env_int("MAX_IMAGE_RESOLUTION_VISION", 1024) or 1024,
    )


config = load_config()
