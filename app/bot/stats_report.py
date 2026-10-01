"""Data and text of the admin ``/stats`` views.

``collect()`` gathers everything one view needs from ``stats.stats_tracker``
(synchronous SQLite: call it through ``asyncio.to_thread``). ``render_main`` and
``render_user`` turn the report into legacy Markdown: bold headings over short
monospace blocks with Unicode bars, a daily sparkline and deltas against the
previous period, every block line at most ``LINE_WIDTH`` characters so it does
not wrap on a phone. The same report feeds ``bot.stats_chart.render_dashboard``.

Nothing here talks to Telegram: display names are passed in as a ``{user_id: name}``
dict, so the module is pure and testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

DAYS = 30
LINE_WIDTH = 32  # monospace characters per line that fit a phone screen without wrapping
MESSAGE_TYPES = ("text", "voice", "photo", "video", "audio", "document")
TOP_TOOLS = 8
TOP_SPENDERS = 5  # rows in the text view
CHART_SPENDERS = 8  # bars in the chart
TOP_MODELS = 4

_EIGHTHS = " ▏▎▍▌▋▊▉"
_SPARKS = "▁▂▃▄▅▆▇█"


@dataclass
class StatsReport:
    days: int
    user_id: str | None  # None = all users
    generated_at: datetime
    current: dict  # stats_tracker.get_window_totals for the period
    previous: dict  # the same for the period before it (deltas)
    all_time: dict
    breakdown: dict  # get_aggregated_stats / get_user_stats for the period (types, tools, describes)
    usage: dict  # get_usage for the period (per-model split)
    daily: list[dict] = field(default_factory=list)
    hourly: list[int] = field(default_factory=list)
    top_spenders: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total_users(self) -> int:
        return self.all_time.get("active_users", 0)

    @property
    def active_today(self) -> int:
        return self.daily[-1]["active_users"] if self.daily else 0

    def user_ids(self) -> list[str]:
        """Users whose display names the views show."""
        return [uid for uid, _ in self.top_spenders]


def collect(user_id: str | None = None, days: int = DAYS) -> StatsReport:
    """Everything one /stats view needs. Blocking (SQLite): run it in a worker thread."""
    from stats import stats_tracker as t  # lazy: importing stats creates data/stats.db

    uid = str(user_id) if user_id is not None else None
    return StatsReport(
        days=days,
        user_id=uid,
        generated_at=datetime.now(timezone.utc),
        current=t.get_window_totals(days, user_id=uid),
        previous=t.get_window_totals(days, offset_days=days, user_id=uid),
        all_time=t.get_window_totals(None, user_id=uid),
        breakdown=t.get_user_stats(uid, days=days) if uid else t.get_aggregated_stats(days=days),
        usage=t.get_usage(user_id=uid, days=days),
        daily=t.get_daily_series(days, user_id=uid),
        hourly=t.get_hourly_activity(days, user_id=uid),
        top_spenders=[] if uid else t.get_users_ranked_by_cost(days=days, limit=CHART_SPENDERS),
    )


# ---- number formatting ------------------------------------------------------
def fmt_int(n: float) -> str:
    return f"{int(n or 0):,}"


def fmt_compact(n: float) -> str:
    """1,234 → 1.2k, 96,320,000 → 96.3M (three significant figures at most)."""
    n = float(n or 0)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            v = n / div
            digits = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2
            text = f"{v:.{digits}f}"
            return (text.rstrip("0").rstrip(".") if "." in text else text) + suffix
    return f"{n:.0f}"


def fmt_money(x: float) -> str:
    x = float(x or 0)
    if 0 < x < 0.01:
        return "<$0.01"
    return f"${x:,.0f}" if x >= 10_000 else f"${x:,.2f}"


def fmt_delta(current: float, previous: float) -> str:
    """Signed change against the previous period: ▲12%, ▼3%, =, new, or '' when both are zero."""
    if previous <= 0:
        return "new" if current > 0 else ""
    change = (current - previous) / previous
    if abs(change) < 0.005:
        return "="
    arrow = "▲" if change > 0 else "▼"
    if change >= 9.995:
        return f"{arrow}{current / previous:.0f}×"
    return f"{arrow}{abs(change) * 100:.0f}%"


def fmt_ago(ts: str | None, now: datetime) -> str:
    if not ts:
        return "never"
    try:
        then = datetime.fromisoformat(ts)
    except ValueError:
        return "unknown"
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    seconds = max(0, int((now - then).total_seconds()))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return "just now"


def fmt_day(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d:%b} {d.day}"


# ---- text graphics ----------------------------------------------------------
def hbar(value: float, maximum: float, width: int) -> str:
    """Horizontal bar in eighth blocks, ``width`` cells for ``maximum``; any non-zero value shows."""
    if value <= 0 or maximum <= 0:
        return ""
    eighths = max(1, min(width * 8, round(value / maximum * width * 8)))
    full, rest = divmod(eighths, 8)
    return "█" * full + (_EIGHTHS[rest] if rest else "")


def sparkline(values: list[float]) -> str:
    """▁ for zero, ▂…█ for the rest scaled to the peak, so idle days stand out from quiet ones."""
    peak = max(values, default=0)
    if peak <= 0:
        return _SPARKS[0] * len(values)
    return "".join(_SPARKS[0] if v <= 0 else _SPARKS[max(1, min(7, round(v / peak * 7)))] for v in values)


def meter(used: int, limit: int, width: int = 10) -> str:
    filled = min(width, round(width * used / limit)) if limit > 0 else 0
    return "█" * filled + "░" * (width - filled)


def plain(text: str) -> str:
    """Drop characters outside the Basic Multilingual Plane (emoji): double width in monospace, no glyph in charts."""
    return "".join(ch for ch in str(text) if ord(ch) < 0x10000).strip() or "?"


def pct(part: float, whole: float) -> str:
    if not whole or not part:
        return ""
    share = 100 * part / whole
    return "<1%" if share < 1 else f"{round(share)}%"


def trim(text: str, width: int) -> str:
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


def table(rows: list[list[str]], align: str) -> list[str]:
    """Columns padded to their widest cell; ``align`` is one 'l'/'r' per column."""
    if not rows:
        return []
    widths = [max(len(r[i]) for r in rows) for i in range(len(align))]
    return [
        " ".join(c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(row, widths, align)).rstrip()
        for row in rows
    ]


def code_block(lines: list[str]) -> str:
    # Backticks would close the block; nothing inside a pre block is parsed otherwise.
    body = "\n".join(line.replace("`", "'") for line in lines)
    return f"```\n{body}\n```"


def _md(text: str) -> str:
    from bot.ui import escape_markdown

    return escape_markdown(str(text))


def _bar_rows(items: list[tuple[str, float, str]], label_width: int, bar_width: int) -> list[list[str]]:
    """(label, value, value text) → table rows with a bar scaled to the largest value."""
    peak = max((v for _, v, _ in items), default=0)
    return [[trim(label, label_width), shown, hbar(value, peak, bar_width)] for label, value, shown in items]


# ---- sections ---------------------------------------------------------------
def _period_section(r: StatsReport) -> list[str]:
    cur, prev = r.current, r.previous
    rows = [
        ("Messages in", fmt_int(cur["received"]), fmt_delta(cur["received"], prev["received"])),
        ("Replies", fmt_int(cur["sent"]), fmt_delta(cur["sent"], prev["sent"])),
        ("Tool calls", fmt_int(cur["tools"]), fmt_delta(cur["tools"], prev["tools"])),
    ]
    if r.user_id is None:
        rows.append(("Active users", fmt_int(cur["active_users"]), fmt_delta(cur["active_users"], prev["active_users"])))
    rows.append(("Est. cost", fmt_money(cur["cost_usd"]), fmt_delta(cur["cost_usd"], prev["cost_usd"])))
    return [f"📅 *Last {r.days} days* · change vs previous {r.days}", code_block(table([list(x) for x in rows], "lrr"))]


def _activity_section(r: StatsReport) -> list[str]:
    values = [d["received_total"] for d in r.daily]
    if not values:
        return []
    peak = max(values)
    head = "📈 *Messages per day*"
    if peak:
        head += f" · peak {fmt_int(peak)} on {fmt_day(r.daily[values.index(peak)]['date'])}"
    spark = sparkline(values)
    first, last = fmt_day(r.daily[0]["date"]), fmt_day(r.daily[-1]["date"])
    axis = first + last.rjust(max(1, len(spark) - len(first)))
    return [head, code_block([spark, axis])]


def _messages_section(r: StatsReport) -> list[str]:
    received = r.breakdown.get("messages_received", {})
    total = received.get("total", 0)
    if not total:
        return [f"💬 *Messages in* · none in {r.days} days"]
    items = sorted(((t, received.get(t, 0)) for t in MESSAGE_TYPES if received.get(t, 0)), key=lambda x: -x[1])
    peak = items[0][1]
    rows = [[t, hbar(n, peak, 10), fmt_int(n), pct(n, total)] for t, n in items]
    lines = [f"💬 *Messages in* · {fmt_int(total)}", code_block(table(rows, "llrr"))]
    extras = []
    if r.breakdown.get("media_groups_processed"):
        extras.append(f"{fmt_int(r.breakdown['media_groups_processed'])} albums")
    if r.breakdown.get("describe_total"):
        extras.append(f"{fmt_int(r.breakdown['describe_total'])} media descriptions")
    if extras:
        lines.append("_" + " · ".join(extras) + "_")
    return lines


def _tools_section(r: StatsReport) -> list[str]:
    tools = r.breakdown.get("tools_used", {})
    total = r.breakdown.get("tools_total", 0)
    if not total:
        return []
    ranked = sorted(tools.items(), key=lambda x: -x[1])
    shown, rest = ranked[:TOP_TOOLS], ranked[TOP_TOOLS:]
    rows = [[trim(name, 15), hbar(n, shown[0][1], 8), fmt_int(n)] for name, n in shown]
    if rest:
        rows.append([f"+{len(rest)} more", "", fmt_int(sum(n for _, n in rest))])
    return [f"🛠 *Tools* · {fmt_int(total)} calls", code_block(table(rows, "llr"))]


def _tokens_section(r: StatsReport) -> list[str]:
    u = r.usage
    if not u or not u.get("api_calls"):
        return ["🧮 *Tokens* · _no usage recorded_"]
    prompt = u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"]
    cached = round(100 * u["cache_read_tokens"] / prompt) if prompt else 0
    head = (f"🧮 *Tokens* · in {fmt_compact(prompt)} ({cached}% cached) · out {fmt_compact(u['output_tokens'])}"
            f" · {fmt_int(u['api_calls'])} API calls")
    models = sorted(u.get("models", {}).items(), key=lambda x: -x[1]["cost_usd"])
    if not models:
        return [head]
    shown = models[:TOP_MODELS]
    items = [(name, m["cost_usd"], fmt_money(m["cost_usd"])) for name, m in shown]
    if len(models) > TOP_MODELS:
        other = sum(m["cost_usd"] for _, m in models[TOP_MODELS:])
        items.append((f"+{len(models) - TOP_MODELS} more", other, fmt_money(other)))
    return [head, code_block(table(_bar_rows(items, 17, 6), "lrl"))]


def _spenders_section(r: StatsReport, names: dict[str, str]) -> list[str]:
    spenders = [(uid, cost) for uid, cost in r.top_spenders if cost > 0]
    if not spenders:
        return []
    total = r.current.get("cost_usd") or sum(c for _, c in spenders)
    peak = spenders[0][1]
    rows = [
        [trim(plain(names.get(uid, uid)), 11), fmt_money(cost), pct(cost, total), hbar(cost, peak, 6)]
        for uid, cost in spenders[:TOP_SPENDERS]
    ]
    return [f"💸 *Top spenders* · {r.days} days", code_block(table(rows, "lrrl"))]


def _all_time_section(r: StatsReport) -> list[str]:
    a = r.all_time
    parts = [f"{fmt_int(a['received'])} in", f"{fmt_int(a['sent'])} replies", f"{fmt_int(a['tools'])} tool calls"]
    if a.get("cost_usd"):
        parts.append(fmt_money(a["cost_usd"]))
    return ["🗂 *All time* · " + " · ".join(parts)]


def _join(sections: list[list[str]]) -> str:
    return "\n\n".join("\n".join(s) for s in sections if s)


# ---- views ------------------------------------------------------------------
def render_main(r: StatsReport, names: dict[str, str] | None = None) -> str:
    """Aggregated view (all users), legacy Markdown."""
    names = names or {}
    header = [
        "📊 *Usage · all users*",
        f"{fmt_int(r.total_users)} users · {fmt_int(r.current['active_users'])} active in {r.days} days"
        f" · {fmt_int(r.active_today)} today",
    ]
    return _join([
        header, _period_section(r), _activity_section(r), _messages_section(r), _tools_section(r),
        _tokens_section(r), _spenders_section(r, names), _all_time_section(r),
    ])


def render_user(r: StatsReport, display_name: str, *, blocked: bool, limit: int | None, used: int,
                default_limit: int) -> str:
    """Per-user view, legacy Markdown. ``limit``: None = no row (default applies), 0 = unlimited."""
    status = "🚫 *Blocked*" if blocked else "✅ Active"
    header = [
        f"👤 *{_md(display_name)}* · `{r.user_id}`",
        f"{status} · last seen {fmt_ago(r.all_time.get('last_seen'), r.generated_at)}",
    ]
    if limit == 0:
        header.append("Quota: ♾ unlimited")
    else:
        cap = limit if limit else default_limit
        note = "" if limit else " (default)"
        if used >= cap:
            note += " · ⛔ limit reached"
        header.append(f"Quota `{meter(used, cap)}` {fmt_int(used)}/{fmt_int(cap)}{note}")
    return _join([
        header, _period_section(r), _activity_section(r), _messages_section(r), _tools_section(r),
        _tokens_section(r), _all_time_section(r),
    ])
