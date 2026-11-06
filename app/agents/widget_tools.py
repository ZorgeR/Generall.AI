"""WidgetTools: Telegram widgets (Bot API 10.4), interactive mini-HTML cards inside a chat message.

``send_widget`` sends one (the HTML inline or from a workspace file), checked first against the
server's limits (32 KB of HTML, 4 KB after zlib level 9) and for features every client removes
(``bot.widgets.lint``); ``update_widget`` changes its state (``chat_data`` / ``public_data``), which
redraws it in place without a new message; ``list_widgets`` names the ones sent; ``widget_guide``
returns the authoring guide (``WIDGET_GUIDE``), read once before writing the first widget, so the
static prompt carries only the short ``PROMPT_NOTES``. Taps that ask the bot
(``TelegramWidget.request``) come back as agent turns: ``bot/handlers/widget_queries.py``.

Listed only while the Bot API server supports widgets (``widgets_available``: the startup probe, or a 404 since).
Runs in the bot process (one Bot API call per tool; registry and file I/O in worker threads),
never sandboxed: ``tool_integrator.py`` must never name this class or its methods
(``tests/test_widgets.py`` guards it).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any

from bot import widgets

logger = logging.getLogger(__name__)

def widgets_available() -> bool:
    """Whether ChainOfThoughtAgent lists the widget tools: not when the Bot API server is known to lack
    them (``bot.widgets.probe`` at startup, or a 404 since). A tool the agent cannot use would still be
    offered to the user and called."""
    return widgets.available()


HTML_DIR = "widgets"  # workspace folder of the sent HTML (data/<uid>/widgets/<name>-<hex>.html)

PROMPT_NOTES = (
    "Widgets: send_widget puts a live mini-HTML card into the chat (a dashboard, chart, calculator, converter, "
    "timer, tracker, checklist, quiz or small game) that runs inside the Telegram app with no network. Offer one "
    "when an interactive or visual answer serves the user better than text, and whenever they ask for a widget, "
    "mini app or dashboard. Call widget_guide before writing your first widget in a conversation and follow it: "
    "clients silently remove everything outside the widget profile. A widget's HTML never changes after sending; "
    "keep its data in chat_data / public_data (JSON) and change it with update_widget, which redraws it in place "
    "without a new message (a periodic agent task can keep a dashboard fresh this way and answer <no_update/>). "
    "With allow_queries, a tap that calls TelegramWidget.request(data) reaches you as a turn starting with "
    "[Widget request]: do what it asks, update the widget, and reply with exactly <no_update/> when the widget "
    "change is the whole answer."
)

WIDGET_GUIDE = """\
TELEGRAM WIDGETS: THE AUTHORING GUIDE (widget profile 1, SDK v1)

A widget is ONE self-contained HTML document (or fragment) with inline <style> and <script>, sent in a
message and drawn inside the chat bubble on every Telegram app. Each app runs its JavaScript in a sandbox with
NO network and silently removes everything outside the profile below: an out-of-profile feature is a bug you
will never see. The HTML never changes after it is sent; only its state does (below).

SIZE
- At most 32768 bytes of HTML and 4096 bytes after zlib level 9 (send_widget measures and refuses more; aim
  for 3800). No comments, no dead code, short class names; build repeated markup in JS; data: images and long
  unique texts compress badly.

THE BUBBLE
- 200-600 px wide (phones about 320-430: design for 320), as tall as the content, at most 600 px; the rest is
  clipped. NOTHING scrolls inside a widget (overflow auto/scroll becomes hidden): fit content with tabs,
  pages or "show more" buttons. Lay out with %, flex, grid, vw; never vh (removed).
- html has no background: the bubble's colour shows through (a background on html/body paints the bubble).
  body has the app's text padding (about 8px 12px) as its margin; body{margin:0} goes edge to edge.
- Defaults: the app's message font (system-ui, about 15px) in --tg-theme-text-color; an unstyled <button>
  looks like the app's own button. Your CSS wins over the defaults.
- Theme: build every colour on these CSS variables (each '#rrggbb') so light and dark both work:
  --tg-theme-bg-color, --tg-theme-text-color, --tg-theme-hint-color, --tg-theme-link-color,
  --tg-theme-button-color, --tg-theme-button-text-color, --tg-theme-secondary-bg-color,
  --tg-theme-accent-text-color, --tg-theme-destructive-text-color, --tg-theme-section-bg-color,
  --tg-theme-section-separator-color, --tg-theme-subtitle-text-color. prefers-color-scheme and
  light-dark() follow the chat's theme. Status colours of your own (green/red) are fine.
- Fonts: system-ui, sans-serif, serif, monospace, ui-monospace, ui-rounded only (named fonts are removed).
- Without a caption below it, the app draws the message time over the bottom corner: keep the bottom clear,
  e.g. body{margin:0;padding:12px 14px max(12px,env(safe-area-inset-bottom))}.

WHAT WORKS
- HTML: div, span, p, h1-h6, ul/ol/li, dl, table, details/summary, figure, section, header, b, i, em, strong,
  small, code, pre, mark, time, a (https links), img (data: PNG/JPEG/GIF/WebP only), canvas, button.
- Inline SVG: svg, g, path, rect, circle, ellipse, line, polyline, polygon, text, tspan, defs, symbol, use,
  linearGradient, radialGradient, stop. Great for charts, gauges, icons and sparklines.
- CSS: flex, grid, positioning (fixed = inside the bubble), box model, borders, radius, box-shadow, linear /
  radial / conic gradients, 2D transforms, transitions, @keyframes, custom properties, calc/min/max/clamp,
  rgb/hsl/hex/light-dark colours, ::before/::after, :active, :not/:is/:where, structural pseudo-classes,
  @media on the width, prefers-color-scheme, prefers-reduced-motion, @supports, one-line text-overflow:
  ellipsis, font-variant-numeric, touch-action, pointer-events, user-select.
- JS: ES2023 in an interpreter (no JIT: keep per-frame work small), Promises, async/await, timers,
  requestAnimationFrame, the DOM (createElement, querySelector, innerHTML, textContent, classList, style,
  dataset, addEventListener, replaceChildren), getBoundingClientRect (a slow round trip: cache it), canvas 2D
  (paths, arc, roundRect, fillText, gradients; scale by devicePixelRatio), el.animate, Intl and toLocale*String
  in the user's locale, crypto.getRandomValues, localStorage / sessionStorage, structuredClone.
- Element ids are NOT global variables: use document.getElementById('x').

OUT (removed or throwing; never use)
- Form controls: input, select, textarea, form, label, option; also progress, meter, dialog, iframe, audio,
  video, object, embed. There is no keyboard and no text input at all: use buttons (steppers, chips,
  presets, +/- buttons, horizontal sliders built from divs).
- Network and outside files: fetch, XMLHttpRequest, WebSocket; <script src>, <link>, @import, @font-face, any
  http(s) image or url(); no libraries (no Chart.js, no D3): draw charts with SVG or canvas yourself.
- CSS: vh units, position: sticky, overflow scrolling, 3D transforms and perspective, text-shadow, clip-path,
  mask, mix-blend-mode, filters other than blur/brightness/drop-shadow, oklch/lab/color-mix, background-clip:
  text, line-clamp, :has(), :focus, :checked, CSS nesting, @container, @layer, @property.
- SVG: clipPath, mask, pattern, marker, filter primitives, SMIL animation, foreignObject.
- Canvas: shadows, globalCompositeOperation, getImageData/putImageData, toDataURL, WebGL.
- JS: alert/confirm/prompt, window.open, location changes, clipboard API, focus(), touch events (use pointer
  events), key, wheel and scroll events, Worker, Blob/FileReader, customElements, Telegram.WebApp.

INPUT
- Taps (click) and pointer events (pointerdown/move/up/cancel). A double tap is the widget's (two clicks,
  then dblclick).
- Vertical drags ALWAYS scroll the chat: nothing can be dragged up or down (no vertical sliders, wheel
  pickers, drawing, map panning). Use horizontal controls or buttons.
- Holds and horizontal drags are claimed with CSS touch-action: auto (default) leaves them to the chat
  (message menu, swipe-to-reply; the widget gets pointercancel); pan-y gives the widget horizontal drags
  (sliders, carousels); none gives it holds and horizontal drags (game buttons, a canvas). Keep controls 20 px
  away from the left edge (the system back gesture).

window.TelegramWidget (undefined outside Telegram: always check it)
- colorScheme ('light'|'dark'), themeParams.{bg_color,text_color,...}, onEvent(type, fn) / offEvent with
  'themeChanged', 'activated' / 'deactivated' (on/off screen), 'chatDataChanged', 'publicDataChanged'.
- openLink('https://...') (the user confirms the URL), openTelegramLink('https://t.me/...'), copyText(text,
  1-256 chars), request(data): each needs a REAL TAP, called synchronously inside a click handler (not after
  await, a timer or a promise), one per tap.
- HapticFeedback.impactOccurred('light'|'medium'|'heavy'), .notificationOccurred('success'|'error'|'warning'),
  .selectionChanged().
- chatData / publicData: the widget's state as strings (or null): JSON.parse them in try/catch.
  viewer: { id, first_name, ... } of the person looking (null where the widget may not ask the bot).
  isForward (a forwarded copy: chatData is null; it shows publicData as it was when forwarded).
- localStorage: kept for this message on this device, 4 KB (catch QuotaExceededError): this person's own
  choices (best score, last tab, ticked items). It can be wiped: never the only copy of something important.

LIMITS
- CPU 300 ms per second, one task at most 500 ms; 8 MB memory; 2000 DOM nodes; 400 CSS rules; 64 timers;
  4000 canvas commands per frame. Past a limit the widget stops on its last frame.
- At most 16 elements at once with opacity < 1, a blurred box-shadow, a filter or backdrop-filter (2 with
  backdrop-filter), animated ones and ::before/::after included: past it the widget STOPS. Mute text with
  colour (var(--tg-theme-hint-color), rgba()), never with opacity; a box-shadow without blur is free.
- Draw the first frame synchronously; animate with CSS or requestAnimationFrame, changing only what moved.

STATE AND TAPS IN THIS BOT
- The widget's state is two texts of at most 1024 bytes each that only you write: chat_data (this message in
  this chat; never forwarded: personal details) and public_data (also shown by forwards, frozen as of the
  forward). Pass them to send_widget as compact JSON (short keys, rounded numbers) and change them with
  update_widget(widget, chat_data / public_data): silent, no new message; the widget gets chatDataChanged /
  publicDataChanged and redraws. So write ONE draw() that renders everything from the state, call it at load
  and on both events. Put changing data (prices, scores, items, progress) in the state, not in the HTML.
- A widget cannot fetch anything. For live data, refresh the state yourself: on request (below), or from a
  periodic agent task (schedule_reminder, reminder_type agent: fetch, update_widget, answer <no_update/>).
- Asking you: send it with allow_queries true; in a click handler call TelegramWidget.request('refresh')
  (data 1-64 bytes, e.g. 'refresh', 'tab:2', 'add:milk', 'answer:3'). The bot answers at once with
  result '{"ok":true}' (or '{"ok":false,"error":"..."}'): that only means you received it. You then run as a
  turn starting with [Widget request] and change the state with update_widget; the widget redraws when the
  change event arrives (seconds to a minute later). So show a pending line ("Asking the assistant...") after
  request() resolves and clear it in draw() when the new state comes; show e.code on failure ('NOT_ALLOWED',
  'BUSY', 'RATE_LIMITED', 'TIMEOUT', 'FAILED'). Never call request() outside a tap handler, never in a loop.
- Each request costs a full turn of yours: use requests for what needs you (fresh data, a real decision, a
  new question); keep local interaction local (tabs, toggles, counters, games run entirely in JS).
- Give send_widget a short caption only when it helps (it is the preview text in the chat list and what old
  apps show instead of the widget), a name you can refer to later, and keep its id (w1, w2, ...).

EXAMPLE (a pattern, not a template): a dashboard drawn from chatData with a Refresh tap
<style>
body{margin:0;padding:12px 14px max(12px,env(safe-area-inset-bottom))}
h3{margin:0 0 6px;font-size:16px}
.r{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--tg-theme-section-separator-color)}
.v{font-weight:600;font-variant-numeric:tabular-nums}.up{color:#2ea043}.dn{color:var(--tg-theme-destructive-text-color)}
.h{color:var(--tg-theme-hint-color);font-size:13px;margin-top:8px}button{width:100%;margin-top:10px}
</style>
<h3 id="t"></h3><div id="rows"></div><button id="b">Refresh</button><div class="h" id="h"></div>
<script>
const TW=window.TelegramWidget,$=id=>document.getElementById(id);
function st(){try{return JSON.parse(TW&&TW.chatData)||{}}catch(e){return {}}}
function draw(){const s=st();$('t').textContent=s.t||'Prices';
$('rows').replaceChildren(...(s.r||[]).map(([k,v,d])=>{const e=document.createElement('div');e.className='r';
e.innerHTML='<span></span><span></span>';e.children[0].textContent=k;const c=e.children[1];
c.className='v '+(d>0?'up':d<0?'dn':'');c.textContent=v+(d?(d>0?' +':' ')+d+'%':'');return e}));
$('h').textContent=s.at?'Updated '+s.at:''}
$('b').onclick=()=>{if(!TW)return;$('h').textContent='Asking the assistant...';
TW.request('refresh').catch(e=>{$('h').textContent='Not sent: '+e.code})};
if(TW)TW.onEvent('chatDataChanged',draw);draw();
</script>
with chat_data '{"t":"Crypto","r":[["BTC","$97,120",1.2],["ETH","$3,410",-0.4]],"at":"14:05"}' and
allow_queries true; on [Widget request] "refresh": fetch the prices, update_widget with the new chat_data,
answer <no_update/>.
"""

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower()).strip("-")[:40] or "widget"


class WidgetTools:
    """The widget tools of one turn (``user_id`` = the chat id; ``sender`` = its ``ChatSender``)."""

    prompt_notes = PROMPT_NOTES

    def __init__(self, user_id: str, sender=None):
        self.user_id = str(user_id)
        self.sender = sender
        self.base = Path("./data") / self.user_id

    @property
    def tools_schema(self) -> list[dict]:
        state_doc = (
            "Compact JSON text, at most 1024 bytes, read by the widget as TelegramWidget.{field} "
            "(redraw on '{event}')."
        )
        return [
            {
                "name": "send_widget",
                "description": (
                    "Send an interactive Telegram widget into the chat: a self-contained mini-HTML page (inline CSS and "
                    "JS) drawn inside a message, like a tiny mini app: dashboards, charts, calculators, converters, "
                    "timers, trackers, checklists, quizzes, small games. Call widget_guide first (once per conversation): "
                    "apps silently remove everything outside the widget profile (no network, no text inputs, no outside "
                    "files, at most 4 KB compressed). The widget cannot fetch data: put data in chat_data / public_data "
                    "and change it later with update_widget. With allow_queries, a tap that calls "
                    "TelegramWidget.request(data) reaches you as a [Widget request] turn. Checks the size and the "
                    "common out-of-profile features before sending; returns the widget id (w1, w2, ...)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "html": {"type": "string", "description": "The widget: one HTML document or fragment with inline <style> and <script>. Give html or html_file."},
                        "html_file": {"type": "string", "description": "Instead of html: a workspace file holding the widget's HTML (e.g. one you built or minified with code, or widgets/<file>.html of an earlier widget)."},
                        "name": {"type": "string", "description": "A short name to refer to it later (e.g. 'BTC dashboard')."},
                        "caption": {"type": "string", "description": "Optional text under the widget: the preview text in the chat list and what older Telegram apps show instead. Omit it for a clean card."},
                        "chat_data": {"type": "string", "description": state_doc.format(field="chatData", event="chatDataChanged") + " The chat state: stays with this message, never forwarded."},
                        "public_data": {"type": "string", "description": state_doc.format(field="publicData", event="publicDataChanged") + " The shared state: forwards show it as of the forward."},
                        "allow_queries": {"type": "boolean", "description": "Let taps ask you: TelegramWidget.request(data) in a click handler becomes a [Widget request] turn for you. Default false."},
                        "width": {"type": "integer", "description": "Fixed width in CSS px (200-600); 0 or omitted = the bubble's width (recommended)."},
                        "height": {"type": "integer", "description": "Fixed height in CSS px (50-600); 0 or omitted = the content's height (recommended)."},
                        "ignore_checks": {"type": "boolean", "description": "Send although the pre-send check reported problems. Only when you are sure a reported problem is a false positive."},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "update_widget",
                "description": (
                    "Change the state of a widget you sent: silent (no new message, no notification); the widget gets "
                    "chatDataChanged / publicDataChanged and redraws in place. Each value is the WHOLE new text (at most "
                    "1024 bytes), not a patch; omit a field to keep it, '' clears it. The HTML never changes: for a "
                    "new layout send a new widget."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "widget": {"type": "string", "description": "The widget id (w3), its name, or its message id."},
                        "chat_data": {"type": "string", "description": "The new chat state (JSON text)."},
                        "public_data": {"type": "string", "description": "The new shared state (JSON text)."},
                    },
                    "required": ["widget"],
                },
            },
            {
                "name": "list_widgets",
                "description": "The widgets you sent in this chat, newest first: id, name, message, topic, whether taps can ask you, their state as you last set it, and the saved HTML file.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "widget_guide",
                "description": "The authoring guide for Telegram widgets: what the apps support, size and layout, input, the TelegramWidget API, limits, and how state and taps work with this bot. Read it before writing your first widget in a conversation.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        args = tool_args or {}
        if tool_name == "send_widget":
            return await self._send(args)
        if tool_name == "update_widget":
            return await self._update(args)
        if tool_name == "list_widgets":
            return await asyncio.to_thread(self._list)
        if tool_name == "widget_guide":
            return WIDGET_GUIDE
        return f"Error: unknown widget tool {tool_name}"

    # ---- send_widget ---------------------------------------------------------------------------
    async def _send(self, args: dict[str, Any]) -> str:
        if widgets.unsupported():
            return widgets.UNSUPPORTED_TEXT
        if self.sender is None:
            return "Error: there is no chat to send the widget to."
        html, html_file = args.get("html"), args.get("html_file")
        if html and html_file:
            return "Error: give html or html_file, not both."
        if html_file:
            html, error = await asyncio.to_thread(self._read_html_file, str(html_file))
            if error:
                return error
        error = widgets.check_html(html)
        if error:
            return error
        chat_data, public_data = args.get("chat_data"), args.get("public_data")
        error = widgets.check_state("chat_data", chat_data) or widgets.check_state("public_data", public_data)
        if error:
            return error
        sizes = {}
        for axis in ("width", "height"):
            try:
                value = int(args.get(axis) or 0)
            except (TypeError, ValueError):
                return f"Error: {axis} must be a number of CSS px (0 = auto)."
            sizes[axis] = min(max(value, 0), widgets.MAX_SIZE_ANCHOR)
        name = widgets.clean_name(args.get("name"))
        caption = str(args.get("caption") or "").strip()[: widgets.MAX_CAPTION] or None
        allow_queries = bool(args.get("allow_queries"))
        problems, notes = widgets.lint(html, allow_queries=allow_queries)
        if problems and not args.get("ignore_checks"):
            return (
                "Error: the widget was not sent: it uses what Telegram apps remove or block, so it would be broken:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\nFix these (widget_guide has the rules) and call send_widget again."
            )
        size, packed = widgets.measure(html)
        method = widgets.SendWidget(
            chat_id=self.sender.chat_id,
            message_thread_id=self.sender.thread_id or None,
            html=html,
            width=sizes["width"] or None,
            height=sizes["height"] or None,
            caption=caption,
            chat_data=widgets.state_value(chat_data),
            public_data=widgets.state_value(public_data),
            allow_queries=True if allow_queries else None,
        )
        try:
            message = await self.sender.bot(method)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the agent reads the reason
            return self._send_error(e, packed)
        saved = html_file if html_file else await asyncio.to_thread(self._save_html, name, html)
        info = widgets.sent_widget_info(message)
        record = await asyncio.to_thread(
            widgets.add, self.user_id, name=name, message_id=message.message_id, thread_id=self.sender.thread_id,
            allow_queries=allow_queries, chat_data=widgets.state_value(chat_data),
            public_data=widgets.state_value(public_data), caption=caption, html_file=saved,
            html_bytes=size, packed_bytes=packed, telegram_id=str(info["id"]) if info.get("id") is not None else None,
        )
        lines = [
            f'Widget {record["id"]} "{name}" sent (message {message.message_id}; {size} bytes of HTML, '
            f"{packed} of {widgets.MAX_PACKED_BYTES} bytes compressed)."
        ]
        if saved:
            lines.append(f"Its HTML is saved as {saved}.")
        lines.append(
            "Taps that call request() reach you as [Widget request] turns." if allow_queries
            else "Its taps cannot ask you (allow_queries off)."
        )
        if notes:
            lines.append("Notes (out of the widget profile, removed by the apps; fix them in the next version):")
            lines += [f"- {n}" for n in notes]
        if problems:
            lines.append("Sent despite the reported problems: " + "; ".join(problems))
        return "\n".join(lines)

    def _send_error(self, e: Exception, packed: int) -> str:
        text = str(getattr(e, "message", "") or e)
        if widgets.is_unsupported_error(e):
            widgets.mark_unsupported(text)
            return widgets.UNSUPPORTED_TEXT
        upper = text.upper()
        hint = ""
        if "WIDGET_DATA" in upper:
            hint = " chat_data / public_data must be UTF-8 text of at most 1024 bytes."
        elif "TOO_LARGE" in upper:
            hint = f" The server counts more than {widgets.MAX_PACKED_BYTES} bytes compressed (here {packed}): make it smaller."
        elif "NOT_ALLOWED" in upper:
            hint = " Widgets can be sent to private chats and groups only (not channels)."
        elif "SEND_PHOTOS_FORBIDDEN" in upper or "not enough rights" in text.lower():
            hint = " The bot may not send media in this chat."
        elif "HTML_INVALID" in upper:
            hint = " The HTML must be plain UTF-8 text."
        logger.warning("sendWidget refused for %s: %s", self.user_id, text)
        return f"Error: Telegram refused the widget: {text}.{hint}"

    def _read_html_file(self, path: str) -> tuple[str | None, str | None]:
        """(the HTML, None) of a workspace file, never through a link; (None, error) otherwise."""
        from agents.paths import resolve_under

        target = resolve_under(self.base, path, fallback_dirs=(HTML_DIR, "downloads", "documents"))
        if target is None:
            return None, f"Error: html_file not found in your workspace: {path}"
        try:
            fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    return None, f"Error: html_file is not a regular file: {path}"
                raw = f.read(widgets.MAX_HTML_BYTES + 1)
        except OSError as e:
            return None, f"Error: could not read html_file {path}: {e.strerror or e}"
        if len(raw) > widgets.MAX_HTML_BYTES:
            return None, f"Error: html_file is over {widgets.MAX_HTML_BYTES} bytes; a widget must be smaller."
        try:
            return raw.decode("utf-8"), None
        except UnicodeDecodeError:
            return None, "Error: html_file is not UTF-8 text."

    def _save_html(self, name: str, html: str) -> str | None:
        """A copy of the sent HTML in the workspace (``widgets/<name>-<hex>.html``), for later edits;
        None when it cannot be written (logged). Link-safe: the folder is made real by
        ``ensure_real_dir`` and the file created ``O_EXCL | O_NOFOLLOW`` relative to it."""
        from agents.paths import ensure_real_dir

        try:
            self.base.mkdir(parents=True, exist_ok=True)
            folder = ensure_real_dir(self.base / HTML_DIR, 0o777)
            dir_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for _ in range(5):
                    file_name = f"{_slug(name)}-{secrets.token_hex(3)}.html"
                    try:
                        fd = os.open(file_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
                    except FileExistsError:
                        continue
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(html)
                    return f"{HTML_DIR}/{file_name}"
            finally:
                os.close(dir_fd)
        except OSError as e:
            logger.warning("Could not save the widget HTML of %s: %s", self.user_id, e)
        return None

    # ---- update_widget -------------------------------------------------------------------------
    async def _update(self, args: dict[str, Any]) -> str:
        if widgets.unsupported():
            return widgets.UNSUPPORTED_TEXT
        chat_data, public_data = args.get("chat_data"), args.get("public_data")
        if chat_data is None and public_data is None:
            return "Error: give chat_data and/or public_data (the whole new text of each)."
        error = widgets.check_state("chat_data", chat_data) or widgets.check_state("public_data", public_data)
        if error:
            return error
        ref = args.get("widget")
        record = await asyncio.to_thread(widgets.find, self.user_id, ref)
        if record is None:
            return f"Error: no widget {ref!r} in this chat. list_widgets shows the ones you sent."
        if record.get("gone"):
            return f'Error: the message of widget {record["id"]} "{record["name"]}" is gone (deleted). Send a new widget.'
        bot = getattr(self.sender, "bot", None)
        if bot is None:
            return "Error: there is no bot to edit the widget with."
        fields = {}
        if chat_data is not None:
            fields["chat_data"] = widgets.state_value(chat_data)
        if public_data is not None:
            fields["public_data"] = widgets.state_value(public_data)
        unchanged = False
        try:
            await bot(widgets.EditMessageWidgetData(chat_id=int(self.user_id), message_id=record["message_id"], **fields))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            text = str(getattr(e, "message", "") or e)
            if widgets.is_not_modified(e):
                unchanged = True
            elif widgets.is_unsupported_error(e):
                widgets.mark_unsupported(text)
                return widgets.UNSUPPORTED_TEXT
            elif widgets.is_message_gone(e):
                await asyncio.to_thread(widgets.update, self.user_id, record["id"], gone=True)
                return f'Error: the message of widget {record["id"]} "{record["name"]}" is gone (deleted). Send a new widget.'
            else:
                logger.warning("editMessageWidgetData refused for %s: %s", self.user_id, text)
                return f"Error: Telegram refused the update: {text}"
        await asyncio.to_thread(widgets.update, self.user_id, record["id"], **fields)
        what = " and ".join(k for k in ("chat_data", "public_data") if k in fields)
        if unchanged:
            return f'Widget {record["id"]} "{record["name"]}" already shows this {what}: nothing changed.'
        return f'Widget {record["id"]} "{record["name"]}" updated ({what}); it redraws in place.'

    # ---- list_widgets --------------------------------------------------------------------------
    def _list(self) -> str:
        records = widgets.load(self.user_id)
        if not records:
            return "No widgets sent in this chat yet."
        lines = []
        for w in reversed(records):
            parts = [w["id"], f'"{w.get("name")}"', f'message {w["message_id"]}']
            if w.get("thread_id"):
                parts.append(f'topic {w["thread_id"]}')
            parts.append("taps ask you" if w.get("allow_queries") else "no requests")
            if w.get("taps"):
                parts.append(f'{w["taps"]} request(s)')
            if w.get("gone"):
                parts.append("message deleted")
            parts.append(f'updated {str(w.get("updated", ""))[:16].replace("T", " ")} UTC')
            if w.get("html_file"):
                parts.append(f'HTML {w["html_file"]}')
            line = " · ".join(parts)
            for key in ("chat_data", "public_data"):
                value = w.get(key)
                if value:
                    preview = value if len(value) <= 300 else value[:300] + "…"
                    line += f"\n  {key}: {preview}"
            lines.append(line)
        return "\n".join(lines)
