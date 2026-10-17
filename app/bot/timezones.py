"""Time zones: the user's zone and local-time helpers (docs/design-v3.md, F5).

The user's zone is the setting ``timezone.name`` (an IANA name; "" = unknown, and times are
shown in UTC until it is set). It is set by ``/timezone`` (``bot/handlers/commands.py``), by a
shared location (``bot/handlers/messages.py``, through ``zone_from_location``) or by the agent's
``set_user_timezone`` tool (``agents/user_interactions.py``). Everything that shows or parses a
time for the user goes through this module: the ``<context>`` block, ``schedule_reminder``, the
scheduler and the ``/reminders`` UI.

``valid_zone`` accepts the exact IANA name, any capitalization of it, and a few dozen big-city
and country aliases ("Moscow", "New York", "Kyiv"). ``timezonefinder`` (zone from coordinates)
is optional and imported lazily; without it ``zone_from_location`` is None and the bot asks for
the zone name instead. This module imports nothing from the app, so ``reminders_store`` and the
agents may use it freely.
"""
from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

logger = logging.getLogger(__name__)

# Picker buttons of /timezone: the zones most users of the bot are in, two per row.
COMMON_ZONES: tuple[str, ...] = (
    "Europe/London", "Europe/Berlin",
    "Europe/Kyiv", "Europe/Moscow",
    "Europe/Istanbul", "Asia/Dubai",
    "Asia/Tbilisi", "Asia/Yerevan",
    "Asia/Almaty", "Asia/Tashkent",
    "Asia/Bangkok", "Asia/Tokyo",
    "America/New_York", "America/Los_Angeles",
    "Australia/Sydney", "UTC",
)

# City / country aliases (lower case, spaces removed) -> IANA name. The agent and the user say
# "Berlin" far more often than "Europe/Berlin".
ALIASES: dict[str, str] = {
    "moscow": "Europe/Moscow", "russia": "Europe/Moscow",
    "berlin": "Europe/Berlin", "germany": "Europe/Berlin",
    "london": "Europe/London", "uk": "Europe/London", "unitedkingdom": "Europe/London",
    "newyork": "America/New_York", "nyc": "America/New_York",
    "kyiv": "Europe/Kyiv", "kiev": "Europe/Kyiv", "ukraine": "Europe/Kyiv",
    "istanbul": "Europe/Istanbul", "turkey": "Europe/Istanbul",
    "dubai": "Asia/Dubai", "uae": "Asia/Dubai",
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo",
    "tbilisi": "Asia/Tbilisi", "georgia": "Asia/Tbilisi",
    "yerevan": "Asia/Yerevan", "armenia": "Asia/Yerevan",
    "almaty": "Asia/Almaty", "kazakhstan": "Asia/Almaty",
    "tashkent": "Asia/Tashkent", "uzbekistan": "Asia/Tashkent",
    "bangkok": "Asia/Bangkok", "thailand": "Asia/Bangkok",
    "singapore": "Asia/Singapore",
    "losangeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "paris": "Europe/Paris", "france": "Europe/Paris",
    "madrid": "Europe/Madrid", "spain": "Europe/Madrid",
    "rome": "Europe/Rome", "italy": "Europe/Rome",
    "warsaw": "Europe/Warsaw", "poland": "Europe/Warsaw",
    "prague": "Europe/Prague", "czechia": "Europe/Prague",
    "vienna": "Europe/Vienna", "austria": "Europe/Vienna",
    "amsterdam": "Europe/Amsterdam", "netherlands": "Europe/Amsterdam",
    "lisbon": "Europe/Lisbon", "portugal": "Europe/Lisbon",
    "belgrade": "Europe/Belgrade", "serbia": "Europe/Belgrade",
    "minsk": "Europe/Minsk", "belarus": "Europe/Minsk",
    "riga": "Europe/Riga", "latvia": "Europe/Riga",
    "vilnius": "Europe/Vilnius", "lithuania": "Europe/Vilnius",
    "tallinn": "Europe/Tallinn", "estonia": "Europe/Tallinn",
    "helsinki": "Europe/Helsinki", "finland": "Europe/Helsinki",
    "stockholm": "Europe/Stockholm", "sweden": "Europe/Stockholm",
    "sydney": "Australia/Sydney",
    "utc": "UTC",
}


def _key(text: str) -> str:
    return "".join(str(text).split()).replace("_", "").replace("-", "").lower()


@functools.lru_cache(maxsize=1)
def _zones_by_key() -> dict[str, str]:
    """Every IANA name under a case- and separator-insensitive key ("europe/newyork")."""
    return {_key(name): name for name in sorted(available_timezones())}


def valid_zone(name: Any) -> str | None:
    """The canonical IANA name for ``name`` (exact, any capitalization, or an alias), else None.

    The alias table is consulted first: a tz database that ships the IANA backward links
    (Debian's does) lists "Japan", "Singapore", "Turkey", "Poland", ... as zones of their own, and
    those must resolve to the canonical "Asia/Tokyo", not be stored as the deprecated link."""
    text = str(name or "").strip()
    if not text:
        return None
    found = ALIASES.get(_key(text))
    if found is None:
        found = _zones_by_key().get(_key(text))
    if found is None:
        return None
    try:
        ZoneInfo(found)
    except (ZoneInfoNotFoundError, ValueError):  # the tz database on this host lacks it
        return None
    return found


def zone_named(name: Any) -> ZoneInfo | None:
    """``ZoneInfo`` for a stored name; None for "" or anything unknown."""
    canonical = valid_zone(name)
    return ZoneInfo(canonical) if canonical else None


def zone_of(settings: Any) -> ZoneInfo | None:
    """The user's zone from a settings dict or a ``UserSettings``; None when unknown."""
    if settings is None:
        return None
    if isinstance(settings, dict):
        category = settings.get("timezone") or {}
        name = category.get("name") if isinstance(category, dict) else None
    else:
        name = settings.get("timezone", "name")
    return zone_named(name)


def zone_name(zone: ZoneInfo | None) -> str:
    return getattr(zone, "key", None) or "UTC"


def now_local(zone: ZoneInfo | None) -> datetime:
    return datetime.now(zone or timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def offset_label(moment: datetime) -> str:
    """``UTC+2`` / ``UTC-3:30`` / ``UTC+0`` for the offset ``moment`` carries."""
    offset = moment.utcoffset()
    total = int(offset.total_seconds()) if offset is not None else 0
    sign = "-" if total < 0 else "+"
    hours, rest = divmod(abs(total), 3600)
    minutes = rest // 60
    return f"UTC{sign}{hours}" + (f":{minutes:02d}" if minutes else "")


def to_local(moment: datetime, zone: ZoneInfo | None) -> datetime:
    """``moment`` (aware, or naive = UTC) in ``zone`` (None = UTC)."""
    return _as_utc(moment).astimezone(zone or timezone.utc)


def format_local(moment: datetime, zone: ZoneInfo | None, with_zone: bool = True) -> str:
    """``2026-10-17 14:05 (Europe/Berlin, UTC+2)``; ``2026-10-17 14:05 (UTC)`` when the zone is unknown.

    ``with_zone=False`` leaves the zone name out: ``2026-10-17 14:05 (UTC+2)``."""
    local = to_local(moment, zone)
    stamp = local.strftime("%Y-%m-%d %H:%M")
    if zone is None:
        return f"{stamp} (UTC)"
    if not with_zone:
        return f"{stamp} ({offset_label(local)})"
    return f"{stamp} ({zone_name(zone)}, {offset_label(local)})"


def to_utc(local_text: str, zone: ZoneInfo | None) -> datetime:
    """Aware UTC datetime for ``YYYY-MM-DD HH:MM[:SS]`` read as local time in ``zone`` (None = UTC).

    A text with its own offset (a trailing ``Z`` or ``±HH:MM``) is taken as given. Raises
    ``ValueError`` for anything ``datetime.fromisoformat`` does not understand."""
    text = str(local_text or "").strip()
    if not text:
        raise ValueError("empty time")
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone or timezone.utc)
    return parsed.astimezone(timezone.utc)


@functools.lru_cache(maxsize=1)
def _finder() -> Any:
    """The ``timezonefinder`` instance (loads its polygons once), or None when not installed."""
    try:
        from timezonefinder import TimezoneFinder  # optional dependency
    except ImportError:
        return None
    try:
        return TimezoneFinder()
    except Exception as e:  # noqa: BLE001 - a broken install must not break the handler
        logger.warning("timezonefinder is installed but could not start: %s", e)
        return None


def zone_from_location(latitude: float, longitude: float) -> str | None:
    """IANA zone at the coordinates via ``timezonefinder``; None without it (or over the ocean).

    Blocking (the first call loads the polygon data): callers run it in a worker thread."""
    finder = _finder()
    if finder is None:
        return None
    try:
        name = finder.timezone_at(lat=float(latitude), lng=float(longitude))
    except Exception as e:  # noqa: BLE001
        logger.warning("timezonefinder failed for %s,%s: %s", latitude, longitude, e)
        return None
    return valid_zone(name) if name else None
