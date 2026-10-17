"""Reminder persistence shared by the agent tool, the scheduler and the /reminders UI.

All read-modify-write cycles go through ``update()`` which holds a per-user
``asyncio.Lock`` so the scheduler, the UI and the agent tool cannot clobber
each other's writes.

Reminder record shape (the ``processing_since`` field is used while an agent reminder is
queued/running; ``timezone``, ``weekdays`` and the ``last_result`` / ``quiet_runs`` fields came
with the v3 wave)::

    {id, user_id, text, time (ISO UTC), type: "user"|"agent", status: "pending"|"processing"|"completed"|"failed",
     created_at, is_periodic, period_type, period_interval, last_triggered, next_trigger,
     [enabled], [completed_at], [agent_response], [processing_since],
     [timezone: IANA name the reminder was scheduled in], [weekdays: [0..6], Monday = 0, daily reminders only],
     [last_result: "quiet"|"answered"], [quiet_runs: int], [snoozed_from: id]}

Times are stored in UTC. ``calculate_next_trigger`` steps daily/weekly/monthly periods in the
reminder's own zone (wall-clock time, so "every day at 9" stays at 9 across a DST change) and
skips the days a ``weekdays`` list leaves out; hourly periods are absolute.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

PERIOD_TYPES = ("hourly", "daily", "weekly", "monthly")
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def period_delta(period_type: str, interval: int) -> timedelta:
    interval = int(interval)
    if period_type == "hourly":
        return timedelta(hours=interval)
    if period_type == "daily":
        return timedelta(days=interval)
    if period_type == "weekly":
        return timedelta(weeks=interval)
    if period_type == "monthly":
        return timedelta(days=30 * interval)
    raise ValueError(f"Unknown period type: {period_type}")


def zone_or_utc(tz: Any) -> ZoneInfo | timezone:
    """``ZoneInfo`` for a stored zone name (or a ``ZoneInfo`` as is); UTC for "" / None / unknown."""
    if isinstance(tz, ZoneInfo):
        return tz
    if not tz:
        return timezone.utc
    try:
        return ZoneInfo(str(tz))
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("Unknown reminder time zone %r, using UTC", tz)
        return timezone.utc


def clean_weekdays(weekdays: Any) -> list[int] | None:
    """Sorted, de-duplicated weekday numbers (0 = Monday .. 6 = Sunday); None when nothing was given.

    Raises ``ValueError`` for anything that is not an int in 0..6 or for an empty list."""
    if weekdays is None:
        return None
    if isinstance(weekdays, (str, bytes)) or not hasattr(weekdays, "__iter__"):
        raise ValueError("weekdays must be a list of integers 0 (Monday) .. 6 (Sunday)")
    days: set[int] = set()
    for item in weekdays:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 6:
            raise ValueError("weekdays must be a list of integers 0 (Monday) .. 6 (Sunday)")
        days.add(item)
    if not days:
        raise ValueError("weekdays must name at least one day")
    return sorted(days)


def _allowed_day(moment: datetime, weekdays: list[int] | None, tz: ZoneInfo | timezone) -> bool:
    return not weekdays or moment.astimezone(tz).weekday() in weekdays


def first_allowed_trigger(first: datetime, weekdays: list[int] | None, tz: Any = None) -> datetime:
    """``first`` itself, or the same local time on the next day in ``weekdays`` (in zone ``tz``)."""
    zone = zone_or_utc(tz)
    moment = first
    for _ in range(8):
        if _allowed_day(moment, weekdays, zone):
            return moment
        moment = (moment.astimezone(zone) + timedelta(days=1)).astimezone(timezone.utc)
    return first  # unreachable with a valid list; never loop forever on a corrupt record


def calculate_next_trigger(
    reminder_time: str, period_type: str, period_interval: int, weekdays: Any = None, tz: Any = None,
) -> datetime:
    """Next trigger strictly in the future, stepping from ``reminder_time`` by the period.

    Daily, weekly and monthly steps are wall-clock steps in zone ``tz`` (an IANA name or ``ZoneInfo``;
    None = UTC), so a local 09:00 stays 09:00 across a DST change; hourly steps are absolute. With a
    ``weekdays`` list (daily reminders) days outside the list are skipped."""
    now = datetime.now(timezone.utc)
    last = parse_time(reminder_time)
    step = period_delta(period_type, period_interval)
    if step <= timedelta(0):
        raise ValueError("period_interval must be positive")
    zone = zone_or_utc(tz)
    days = clean_weekdays(weekdays)
    if period_type == "hourly":
        zone_for_step: ZoneInfo | timezone = timezone.utc
    else:
        zone_for_step = zone
    local = last.astimezone(zone_for_step)
    for _ in range(100_000):  # bounded: a reminder far in the past still terminates
        local = local + step  # aware + timedelta = wall-clock arithmetic in zone_for_step
        candidate = local.astimezone(timezone.utc)
        if candidate > now and _allowed_day(candidate, days, zone):
            return candidate
    raise ValueError("could not find the next trigger time")


def parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def new_reminder_id(existing: list[dict]) -> str:
    taken = {str(r.get("id")) for r in existing}
    while True:
        candidate = uuid.uuid4().hex[:8]
        if candidate not in taken:
            return candidate


class RemindersStore:
    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)
        self._locks: dict[str, asyncio.Lock] = {}

    def path(self, user_id: str) -> Path:
        return self.base_dir / str(user_id) / "reminders" / "reminders.json"

    def _lock(self, user_id: str) -> asyncio.Lock:
        return self._locks.setdefault(str(user_id), asyncio.Lock())

    def user_ids(self) -> Iterator[str]:
        if not self.base_dir.exists():
            return iter(())
        return (p.name for p in self.base_dir.iterdir() if p.is_dir() and (p / "reminders" / "reminders.json").exists())

    def _read(self, user_id: str) -> list[dict]:
        path = self.path(user_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:  # noqa: BLE001
            logger.error("Error reading reminders for %s: %s", user_id, e)
            return []

    def _write(self, user_id: str, reminders: list[dict]) -> None:
        path = self.path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(reminders, f, indent=2, ensure_ascii=False)
        tmp.replace(path)

    async def load(self, user_id: str) -> list[dict]:
        async with self._lock(user_id):
            return self._read(user_id)

    async def save(self, user_id: str, reminders: list[dict]) -> None:
        async with self._lock(user_id):
            self._write(user_id, reminders)

    async def update(self, user_id: str, mutate: Callable[[list[dict]], Any]) -> Any:
        """Load, apply ``mutate(reminders)`` in place, persist if anything changed, return its result."""
        async with self._lock(user_id):
            reminders = self._read(user_id)
            before = json.dumps(reminders, sort_keys=True, ensure_ascii=False)
            result = mutate(reminders)
            if json.dumps(reminders, sort_keys=True, ensure_ascii=False) != before:
                self._write(user_id, reminders)
            return result

    async def add(self, user_id: str, reminder: dict) -> dict:
        def _add(reminders: list[dict]) -> dict:
            reminder.setdefault("id", new_reminder_id(reminders))
            reminders.append(reminder)
            return reminder

        return await self.update(user_id, _add)


reminders_store = RemindersStore()
