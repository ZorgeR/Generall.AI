"""PNG dashboard for the admin ``/stats`` views.

One image per view: a header, a row of stat tiles with the change against the
previous period, messages per day stacked by type, estimated cost per day, the
top tools, and either the top spenders (all users) or messages by hour of day
(one user). The text view (``bot.stats_report``) carries every number shown
here, so it doubles as the chart's table view.

Rendering is synchronous and CPU-bound: call ``render_dashboard`` through
``asyncio.to_thread``. Only matplotlib's object API is used (``Figure`` with an
Agg canvas, never ``pyplot``), so renders in concurrent worker threads share no
global state. matplotlib is imported lazily: the ``bot`` package keeps no
import-time side effects.
"""
from __future__ import annotations

import io
import warnings
from datetime import date

from bot.stats_report import StatsReport, fmt_compact, fmt_delta, fmt_int, fmt_money, plain, trim

# Light chart surface and ink (the dataviz reference palette); categorical slots 1-3
# for the three main message types, a recessive gray for the folded tail.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
OTHER = "#b5b4ad"
UP_GOOD = "#006300"
DOWN_BAD = "#d03b3b"

STACKED_TYPES = ("text", "voice", "photo")  # everything else is folded into "other"
TOP_BARS = 8
HEADROOM = 1.18  # room above the tallest column for its label

warnings.filterwarnings("ignore", message=r"Glyph \d+ .*missing from")  # emoji in user names


def _t(text: str) -> str:
    """Escape '$' so matplotlib never switches a label into mathtext."""
    return str(text).replace("$", r"\$")


def _style(ax, *, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0, pad=4)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)


def _title(ax, title: str, note: str = "") -> None:
    ax.set_title(_t(title), loc="left", fontsize=10, fontweight="bold", color=INK, pad=14)
    if note:
        ax.text(0, 1.02, _t(note), transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")


def _empty(ax, message: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color=MUTED)


def _day_ticks(ax, days: list[str]) -> None:
    n = len(days)
    ticks = sorted(i for i in range(n - 1, -1, -7))
    ax.set_xticks(ticks)
    labels = []
    for i in ticks:
        d = date.fromisoformat(days[i])
        labels.append(f"{d:%b} {d.day}")
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, n - 0.4)


def _label_peak(ax, values: list[float], text: str) -> None:
    if not values or max(values) <= 0:
        return
    i = values.index(max(values))
    ax.annotate(_t(text), (i, values[i]), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color=INK_2)


def _tiles(fig, r: StatsReport, *, top: float, left: float, right: float) -> None:
    """Stat tiles in figure coordinates: label, value, change against the previous period."""
    cur, prev = r.current, r.previous
    tiles = [("Messages in", fmt_int(cur["received"]), cur["received"], prev["received"], True)]
    if r.user_id is None:
        tiles.append(("Active users", fmt_int(cur["active_users"]), cur["active_users"], prev["active_users"], True))
    else:
        tiles.append(("Replies", fmt_int(cur["sent"]), cur["sent"], prev["sent"], True))
    tiles.append(("Tool calls", fmt_int(cur["tools"]), cur["tools"], prev["tools"], True))
    tiles.append(("Est. cost", fmt_money(cur["cost_usd"]), cur["cost_usd"], prev["cost_usd"], False))
    step = (right - left) / len(tiles)
    for i, (label, value, now, before, up_is_good) in enumerate(tiles):
        x = left + i * step
        fig.text(x, top, label, fontsize=8, color=INK_2, va="top")
        fig.text(x, top - 0.022, _t(value), fontsize=19, fontweight="bold", color=INK, va="top")
        delta = fmt_delta(now, before)
        if delta:
            color = INK_2
            if up_is_good and delta[0] in "▲▼":
                color = UP_GOOD if delta[0] == "▲" else DOWN_BAD
            fig.text(x, top - 0.064, delta, fontsize=8, fontweight="bold", color=color, va="top")


def _messages_per_day(ax, r: StatsReport) -> None:
    days = [d["date"] for d in r.daily]
    totals = [d["received_total"] for d in r.daily]
    _style(ax)
    _title(ax, "Messages per day", "by type · UTC days")  # totals live in the tiles (rolling window)
    if not any(totals):
        _empty(ax, "No messages in this period")
        return
    x = list(range(len(days)))
    bottom = [0] * len(days)
    layers = [(t, [d["received"].get(t, 0) for d in r.daily], color) for t, color in zip(STACKED_TYPES, SERIES)]
    layers.append(("other", [d["received_total"] - sum(d["received"].get(t, 0) for t in STACKED_TYPES) for d in r.daily], OTHER))
    for name, values, color in layers:
        if not any(values):
            continue
        ax.bar(x, values, bottom=bottom, width=0.62, color=color, edgecolor=SURFACE, linewidth=1.0, label=name)
        bottom = [b + v for b, v in zip(bottom, values)]
    _label_peak(ax, totals, fmt_int(max(totals)))
    _day_ticks(ax, days)
    ax.yaxis.set_major_formatter(_formatter(lambda v: fmt_compact(v)))
    ax.yaxis.get_major_locator().set_params(nbins=4, integer=True)
    ax.set_ylim(0, max(totals) * HEADROOM)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=4, frameon=False, fontsize=7.5,
              handlelength=0.9, handleheight=0.9, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.3,
              labelcolor=INK_2)


def _cost_per_day(ax, r: StatsReport) -> None:
    days = [d["date"] for d in r.daily]
    costs = [d["cost_usd"] for d in r.daily]
    _style(ax)
    _title(ax, "Estimated cost per day", "USD · UTC days")
    if not any(costs):
        _empty(ax, "No token usage recorded in this period")
        return
    ax.bar(range(len(days)), costs, width=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
    _label_peak(ax, costs, fmt_money(max(costs)))
    _day_ticks(ax, days)
    ax.yaxis.set_major_formatter(_formatter(lambda v: _t(f"${v:,.0f}" if max(costs) >= 5 else f"${v:,.2f}")))
    ax.yaxis.get_major_locator().set_params(nbins=4)
    ax.set_ylim(0, max(costs) * HEADROOM)


def _ranked_bars(ax, title: str, note: str, items: list[tuple[str, float, str]], empty: str) -> None:
    """Horizontal bars, largest on top, label above each bar and value at its tip (no value axis).

    The slot count is fixed at ``TOP_BARS`` so both bottom panels share one bar thickness.
    """
    _style(ax, grid_axis=None)
    _title(ax, title, note)
    ax.set_xticks([])
    ax.set_yticks([])
    if not items:
        _empty(ax, empty)
        return
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(0.8)
    values = [v for _, v, _ in items]
    peak = max(values) or 1
    ax.barh(range(len(items)), values, height=0.42, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
    for i, (label, value, shown) in enumerate(items):
        ax.text(peak * 0.012, i - 0.26, _t(trim(plain(label), 34)), va="bottom", fontsize=7.5, color=INK_2)
        ax.text(value + peak * 0.02, i, _t(shown), va="center", fontsize=7, color=INK_2)
    ax.set_xlim(0, peak * 1.22)
    ax.set_ylim(TOP_BARS - 0.6, -1.0)


def _hourly(ax, r: StatsReport) -> None:
    _style(ax)
    _title(ax, "Messages by hour", "UTC")
    if not any(r.hourly):
        _empty(ax, "No messages in this period")
        return
    ax.bar(range(24), r.hourly, width=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["00", "06", "12", "18", "23"])
    ax.set_xlim(-0.6, 23.6)
    ax.yaxis.get_major_locator().set_params(nbins=4, integer=True)
    ax.set_ylim(0, max(r.hourly) * HEADROOM)


def _formatter(fn):
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: fn(v))


def render_dashboard(r: StatsReport, names: dict[str, str] | None = None, *, title: str = "All users") -> bytes:
    """The dashboard as PNG bytes. Blocking: run it in a worker thread."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    names = names or {}
    fig = Figure(figsize=(8, 10), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    left, right = 0.085, 0.97

    fig.text(left, 0.975, _t(plain(title)), fontsize=15, fontweight="bold", color=INK, va="top")
    first = date.fromisoformat(r.daily[0]["date"]) if r.daily else None
    last = date.fromisoformat(r.daily[-1]["date"]) if r.daily else None
    span = f" · {first:%b} {first.day} – {last:%b} {last.day}, {last.year} (UTC)" if first and last else ""
    fig.text(left, 0.945, _t(f"Last {r.days} days{span} · arrows compare with the previous {r.days} days"),
             fontsize=8, color=MUTED, va="top")
    _tiles(fig, r, top=0.905, left=left, right=right)
    fig.add_artist(Line2D([left, right], [0.815, 0.815], color=GRID, linewidth=0.8))

    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.6, 2.5], hspace=0.55, wspace=0.12,
                          left=left, right=right, top=0.74, bottom=0.025)
    _messages_per_day(fig.add_subplot(gs[0, :]), r)
    _cost_per_day(fig.add_subplot(gs[1, :]), r)

    tools = sorted(r.breakdown.get("tools_used", {}).items(), key=lambda x: -x[1])[:TOP_BARS]
    _ranked_bars(fig.add_subplot(gs[2, 0]), "Top tools", f"{fmt_int(r.breakdown.get('tools_total', 0))} calls",
                 [(name, n, fmt_int(n)) for name, n in tools], "No tool calls in this period")
    if r.user_id is None:
        spenders = [(uid, c) for uid, c in r.top_spenders if c > 0][:TOP_BARS]
        _ranked_bars(fig.add_subplot(gs[2, 1]), "Top spenders", "estimated cost",
                     [(names.get(uid, uid), c, fmt_money(c)) for uid, c in spenders], "No token usage recorded")
    else:
        _hourly(fig.add_subplot(gs[2, 1]), r)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    return buf.getvalue()
