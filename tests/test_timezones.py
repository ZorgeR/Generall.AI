"""bot/timezones.py: zone lookup (IANA, aliases), local formatting, local -> UTC parsing, location lookup."""
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from bot import timezones as tz
from bot.settings import UserSettings

BERLIN = ZoneInfo("Europe/Berlin")
SUMMER = datetime(2026, 10, 17, 12, 5, tzinfo=timezone.utc)  # CEST: UTC+2
WINTER = datetime(2026, 12, 1, 12, 5, tzinfo=timezone.utc)  # CET: UTC+1


# ---- valid_zone ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, expected", [
    ("Europe/Berlin", "Europe/Berlin"),
    ("europe/berlin", "Europe/Berlin"),  # any capitalization
    ("EUROPE/BERLIN", "Europe/Berlin"),
    ("America/New York", "America/New_York"),  # a space for the underscore
    ("Berlin", "Europe/Berlin"),  # city aliases
    ("moscow", "Europe/Moscow"),
    ("New York", "America/New_York"),
    (" Kyiv ", "Europe/Kyiv"),
    ("Tokyo", "Asia/Tokyo"),
    ("Los Angeles", "America/Los_Angeles"),
    ("UTC", "UTC"),
    ("utc", "UTC"),
])
def test_valid_zone_accepts_iana_names_any_case_and_aliases(name, expected):
    assert tz.valid_zone(name) == expected


@pytest.mark.parametrize("name", ["Mars", "", None, "Europe", "Berlin/Europe", 42, "   "])
def test_valid_zone_rejects_unknown_names(name):
    assert tz.valid_zone(name) is None


def test_every_alias_and_common_zone_is_a_real_zone():
    for alias, name in tz.ALIASES.items():
        assert tz.valid_zone(name) == name, (alias, name)
        assert tz.valid_zone(alias) == name, alias
    assert 12 <= len(tz.COMMON_ZONES) <= 16
    assert len(set(tz.COMMON_ZONES)) == len(tz.COMMON_ZONES)
    assert all(tz.valid_zone(name) == name for name in tz.COMMON_ZONES)
    for city in ("Moscow", "Berlin", "London", "New York", "Kyiv", "Istanbul", "Dubai", "Tokyo", "Tbilisi", "Yerevan",
                 "Almaty", "Tashkent", "Bangkok", "Singapore", "Los Angeles", "Chicago", "Paris", "Madrid", "Rome",
                 "Warsaw", "Prague", "Vienna", "Amsterdam", "Lisbon", "Belgrade", "Minsk", "Riga", "Vilnius", "Tallinn",
                 "Helsinki", "Stockholm", "Sydney"):
        assert tz.valid_zone(city), city


def test_aliases_win_over_legacy_backward_links(monkeypatch):
    """Debian's tzdata lists the IANA backward links ("Japan", "Singapore", "Turkey", "Poland") as
    zones; the alias table must still give the canonical name, and exact names keep working."""
    table = dict(tz._zones_by_key())
    table.update({"japan": "Japan", "singapore": "Singapore", "turkey": "Turkey", "poland": "Poland"})
    monkeypatch.setattr(tz, "_zones_by_key", lambda: table)
    assert tz.valid_zone("Japan") == "Asia/Tokyo"
    assert tz.valid_zone("singapore") == "Asia/Singapore"
    assert tz.valid_zone("Turkey") == "Europe/Istanbul"
    assert tz.valid_zone("POLAND") == "Europe/Warsaw"
    assert tz.valid_zone("Asia/Tokyo") == "Asia/Tokyo"  # the exact-IANA path is untouched
    assert tz.valid_zone("america/new york") == "America/New_York"


# ---- zone_of ------------------------------------------------------------------------------
def test_zone_of_reads_dicts_and_user_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tz.zone_of({"timezone": {"name": "Europe/Berlin"}}) == BERLIN
    assert tz.zone_of({"timezone": {"name": ""}}) is None
    assert tz.zone_of({}) is None
    assert tz.zone_of({"timezone": "garbage"}) is None
    assert tz.zone_of(None) is None
    settings = UserSettings("42")
    assert tz.zone_of(settings) is None  # the default: unknown
    settings.set("timezone", "asia/tokyo", "name")  # stored as typed; still resolves
    assert tz.zone_of(settings) == ZoneInfo("Asia/Tokyo")
    assert tz.zone_of(UserSettings("42")) == ZoneInfo("Asia/Tokyo")  # persisted


def test_zone_named_and_zone_name():
    assert tz.zone_named("Berlin") == BERLIN
    assert tz.zone_named("") is None and tz.zone_named(None) is None
    assert tz.zone_name(BERLIN) == "Europe/Berlin" and tz.zone_name(None) == "UTC"


# ---- format_local / offsets ---------------------------------------------------------------
def test_format_local_shows_the_zone_and_its_current_offset():
    assert tz.format_local(SUMMER, BERLIN) == "2026-10-17 14:05 (Europe/Berlin, UTC+2)"
    assert tz.format_local(WINTER, BERLIN) == "2026-12-01 13:05 (Europe/Berlin, UTC+1)"
    assert tz.format_local(SUMMER, BERLIN, with_zone=False) == "2026-10-17 14:05 (UTC+2)"
    assert tz.format_local(SUMMER, None) == "2026-10-17 12:05 (UTC)"
    assert tz.format_local(SUMMER.replace(tzinfo=None), BERLIN) == "2026-10-17 14:05 (Europe/Berlin, UTC+2)"  # naive = UTC
    assert tz.format_local(SUMMER, ZoneInfo("Asia/Kolkata")) == "2026-10-17 17:35 (Asia/Kolkata, UTC+5:30)"
    assert tz.format_local(SUMMER, ZoneInfo("America/Sao_Paulo")) == "2026-10-17 09:05 (America/Sao_Paulo, UTC-3)"
    assert tz.format_local(SUMMER, ZoneInfo("UTC")) == "2026-10-17 12:05 (UTC, UTC+0)"


def test_now_local_and_to_local():
    assert tz.now_local(None).tzinfo == timezone.utc
    assert tz.now_local(BERLIN).tzinfo == BERLIN
    assert tz.to_local(SUMMER, BERLIN).hour == 14 and tz.to_local(SUMMER, None).hour == 12


# ---- to_utc -------------------------------------------------------------------------------
def test_to_utc_reads_local_time_in_the_zone():
    assert tz.to_utc("2026-10-17 14:05", BERLIN) == SUMMER
    assert tz.to_utc("2026-12-01 13:05:00", BERLIN) == WINTER
    assert tz.to_utc("2026-10-17T14:05", BERLIN) == SUMMER  # the T separator too
    assert tz.to_utc("2026-10-17 12:05", None) == SUMMER  # unknown zone = UTC


def test_to_utc_keeps_an_explicit_offset():
    assert tz.to_utc("2026-10-17 12:05Z", BERLIN) == SUMMER
    assert tz.to_utc("2026-10-17 12:05:00z", BERLIN) == SUMMER
    assert tz.to_utc("2026-10-17 17:05+05:00", BERLIN) == SUMMER
    assert tz.to_utc("2026-10-17T09:05-03:00", None) == SUMMER
    assert tz.to_utc("2026-10-17 12:05", BERLIN).tzinfo == timezone.utc


@pytest.mark.parametrize("text", ["", None, "tomorrow", "2026-13-01 10:00", "14:05"])
def test_to_utc_rejects_bad_text(text):
    with pytest.raises(ValueError):
        tz.to_utc(text, BERLIN)


# ---- zone_from_location -------------------------------------------------------------------
def test_zone_from_location_uses_timezonefinder_when_available(monkeypatch):
    calls = []

    def timezone_at(*, lat, lng):
        calls.append((lat, lng))
        return "Asia/Tbilisi"

    monkeypatch.setattr(tz, "_finder", lambda: SimpleNamespace(timezone_at=timezone_at))
    assert tz.zone_from_location(41.7, 44.8) == "Asia/Tbilisi"
    assert calls == [(41.7, 44.8)]


def test_zone_from_location_without_the_library_or_over_the_ocean(monkeypatch):
    monkeypatch.setattr(tz, "_finder", lambda: None)
    assert tz.zone_from_location(41.7, 44.8) is None  # timezonefinder not installed: the bot asks for the name
    monkeypatch.setattr(tz, "_finder", lambda: SimpleNamespace(timezone_at=lambda *, lat, lng: None))
    assert tz.zone_from_location(0.0, -30.0) is None
    monkeypatch.setattr(tz, "_finder", lambda: SimpleNamespace(timezone_at=lambda *, lat, lng: "Not/AZone"))
    assert tz.zone_from_location(1.0, 1.0) is None  # an unknown answer is not saved


def test_finder_is_none_when_the_optional_package_is_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "timezonefinder":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    tz._finder.cache_clear()
    try:
        assert tz._finder() is None
    finally:
        tz._finder.cache_clear()
