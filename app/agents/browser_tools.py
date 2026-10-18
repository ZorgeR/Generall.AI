"""The agent's persistent sandbox: its status, and an interactive browser of its own.

Two providers, both registered in ``ChainOfThoughtAgent.__init__`` (awaited) and only when the
process has a sandbox session manager (``secure_container.sessions.get_manager()``):

* ``SandboxTools`` (``sandbox_status``): registered whenever a manager exists and commands run in
  sessions (``sessions.SESSIONS_ENABLED``), whatever the browser setting says, because its
  ``prompt_notes`` carry the facts the agent needs for the persistent shell session (idle
  timeout, what persists, packages) even without a browser. Subagents inherit it (read-only).
* ``BrowserTools``: Anthropic's client-side browser toolset (``browser_toolset_20260801``, one
  nameless ``tools`` entry; the model's calls are ``tool_use`` blocks named after the member and
  carrying ``toolset_name: "browser"``) plus two normal tools, ``browser_confirm`` and
  ``browser_show_screenshot``. Registered only while the user's ``browser.enabled`` setting is on.
  Subagents never get it (``SubagentTools`` drops every provider with a ``toolset``).

Member calls are executed by ``execute_member``: ``AgentAnthropic.run_tool_batch`` runs the
browser calls of one assistant message in order, stops at the first failure, and calls
``after_batch`` once the sequence is done (the live view). Each call goes to the per-user
browser container through ``SessionManager.browser_call`` (blocking: a worker thread); a
``CapacityError`` (no memory for a browser right now) becomes an error result with its text.
Until a member call of this turn succeeded, the request carries ``force_state``: the server
reports the tab list only when it changed since its last report, which may have gone to another
conversation (forum topics share one browser) or to a cancelled turn.

Consequential actions (paying, sending, deleting, account changes) are refused by the browser
server until the user allowed one through ``browser_confirm``: the allowance is one action,
for ``ALLOWANCE_SECONDS``, and only for this turn (a new turn means a new instance), so a
permission the agent did not use cannot leak into a later, different request. Best effort:
the server recognises such actions by the element's name.

The live view: with ``browser.stream`` on, after each browser batch the current viewport is sent
to the chat as ONE photo per turn ("🌐 title / url / ▶️ last action"), edited in place on later
batches at most every ``LIVE_VIEW_INTERVAL`` seconds. Failures are logged, never fatal.

No import-time side effects. The sessions module is looked up in ``sys.modules`` only: the
manager exists only once ``secure_container`` was initialised (``ContainerManager`` imports it),
so an agent built without the sandbox (tests, Docker down) never imports Docker code.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import logging
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

SESSIONS_MODULE = "secure_container.sessions"

TOOLSET_TYPE = "browser_toolset_20260801"
TOOLSET_NAME = "browser"
# Members we do not offer (also off by default in the API; said explicitly so a default change
# cannot switch them on): no JavaScript execution, file uploads or console/network reads.
DISABLED_MEMBERS = ("javascript_exec", "file_upload", "read_console", "read_network")
NOT_EXECUTED = "Not executed: an earlier action in this turn failed."
RESULT_BLOCK_TYPES = ("text", "image", "browser_state")

BROWSER_CALL_TIMEOUT = 75.0   # seconds per member call (navigation 30 s + settle, wait up to 30 s)
CONFIRM_TIMEOUT = 300         # seconds the user has to answer browser_confirm
ALLOWANCE_SECONDS = 300       # how long an allowed consequential action stays allowed (one action)
LIVE_VIEW_INTERVAL = 2.5      # seconds between two live-view updates
LIVE_VIEW_TIMEOUT = 20.0      # a live-view update that takes longer is given up
ALLOW_OPTION = "✅ Allow"
CANCEL_OPTION = "❌ Cancel"
MAX_SUMMARY = 500
MAX_CAPTION = 1000
MIN_CROP_SIDE = 640           # a small crop (a captcha) is scaled up to this long side, at most 3x

# Fallbacks for the sessions module's constants (spec defaults), used when the module is absent.
DEFAULT_IDLE_SECONDS = 1800
DEFAULT_SESSION_MEM_MB = 768
DEFAULT_WARN_BEFORE_SECONDS = 300  # the user is warned this long before an idle session closes


# ---- the session manager ---------------------------------------------------------
def sessions_module():
    """``secure_container.sessions`` when it was imported (the bot imports it at startup), else None."""
    return sys.modules.get(SESSIONS_MODULE)


def session_manager():
    """The process-wide SessionManager, or None (never initialised, Docker down, tests)."""
    getter = getattr(sessions_module(), "get_manager", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception as e:  # noqa: BLE001 - no manager is a normal state, never an error for the turn
        logger.warning("Sandbox session manager unavailable: %s", e)
        return None


def sessions_constant(name: str, default: int) -> int:
    value = getattr(sessions_module(), name, None)
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def idle_minutes() -> int:
    return max(1, round(sessions_constant("SESSION_IDLE_SECONDS", DEFAULT_IDLE_SECONDS) / 60))


def warn_minutes() -> int:
    return max(1, round(sessions_constant("SESSION_WARN_BEFORE_SECONDS", DEFAULT_WARN_BEFORE_SECONDS) / 60))


def is_capacity_error(error: BaseException) -> bool:
    """The manager's ``CapacityError`` (matched by class name, so the sessions module stays optional)."""
    return any(cls.__name__ == "CapacityError" for cls in type(error).__mro__)


def browser_settings(user_settings: Any) -> dict:
    """The user's ``browser`` category with defaults (enabled, stream, remember: all on)."""
    category = user_settings.get("browser") if isinstance(user_settings, dict) else None
    category = category if isinstance(category, dict) else {}
    return {key: bool(category.get(key, True)) for key in ("enabled", "stream", "remember")}


def browser_enabled(user_settings: Any) -> bool:
    return browser_settings(user_settings)["enabled"]


def shell_sessions_enabled() -> bool:
    """``sessions.SESSIONS_ENABLED``: sandboxed commands run in the persistent session (SANDBOX_SESSIONS).
    Off, they run in one-off containers and the sandbox facts in ``SandboxTools`` would be wrong."""
    return bool(getattr(sessions_module(), "SESSIONS_ENABLED", True))


# ---- small helpers ----------------------------------------------------------------
# Member inputs that carry what the model typed into the page (passwords, one-time codes):
# member -> input key. Shown to people (status line, turn summary, reasoning file) as their
# length only; the request and the stored tool_use block keep the real value.
TYPED_INPUTS = {"type": "text", "form_input": "value"}


def masked_input(name: str, args: Any) -> dict:
    """A copy of a member's input for display, typed text and form values replaced by
    ``<N characters>`` (a checkbox's true/false stays)."""
    shown = dict(args) if isinstance(args, dict) else {}
    key = TYPED_INPUTS.get(str(name))
    value = shown.get(key) if key else None
    if value is not None and not isinstance(value, bool):
        length = len(str(value))
        shown[key] = f"<{length} character{'' if length == 1 else 's'}>"
    return shown


def describe_action(name: str, args: Any) -> str:
    """One line for the live view. Typed text and form values are not repeated (passwords)."""
    args = args if isinstance(args, dict) else {}
    target = args.get("target") if isinstance(args.get("target"), dict) else {}
    where = ""
    if target.get("type") == "ref" or "ref" in target:
        where = str(target.get("ref") or "")
    elif "x" in target and "y" in target:
        where = f"({target.get('x')}, {target.get('y')})"
    if name == "navigate":
        detail = str(args.get("url") or "")
    elif name == "type":
        detail = f"{len(str(args.get('text') or ''))} characters"
    elif name == "key":
        detail = str(args.get("text") or "")
    elif name == "scroll":
        detail = " ".join(p for p in (str(args.get("scroll_direction") or ""), where) if p)
    elif name == "find":
        detail = str(args.get("query") or "")
    elif name in ("switch_tab", "close_tab"):
        detail = str(args.get("tab_id") or "")
    else:
        detail = where
    line = f"{name} {detail}".strip()
    return line if len(line) <= 120 else line[:119] + "…"


def clean_result_content(content: Any) -> list | str:
    """The server's result content limited to what a toolset tool_result may hold (text, image,
    browser_state), empty text dropped; "Done." when nothing is left."""
    if isinstance(content, str):
        return content if content.strip() else "Done."
    if not isinstance(content, list):
        return "Done."
    kept = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in RESULT_BLOCK_TYPES:
            continue
        if block["type"] == "text" and not str(block.get("text") or "").strip():
            continue
        kept.append(block)
    return kept or "Done."


def _as_utc(value: Any) -> datetime | None:
    """``closes_at`` as given by ``SessionManager.status`` (aware datetime, ISO text or epoch seconds)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _human_bytes(size: Any) -> str:
    try:
        size = float(size)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def crop_jpeg(data: bytes, region: list[int]) -> bytes:
    """``region`` [x0, y0, x1, y1] of a screenshot as JPEG, scaled up when small (blocking: PIL)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        width, height = image.size
        x0, y0, x1, y1 = (int(v) for v in region)
        x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
        y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError(f"the region is outside the {width}x{height} view or too small")
        part = image.crop((x0, y0, x1, y1))
        long_side = max(part.size)
        if long_side < MIN_CROP_SIDE:
            scale = min(3.0, MIN_CROP_SIDE / long_side)
            part = part.resize((round(part.size[0] * scale), round(part.size[1] * scale)), Image.LANCZOS)
        out = io.BytesIO()
        part.save(out, format="JPEG", quality=88)
        return out.getvalue()


def _region(value: Any) -> list[int] | None:
    if value is None or value == []:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("region must be [x0, y0, x1, y1] in viewport pixels")
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError("region must be four integers [x0, y0, x1, y1]") from None


# ---- the sandbox status -------------------------------------------------------------
SANDBOX_NOTES = """Sandbox: run_command, run_shell_script, execute_python, install_package, create_directory and delete_file run in your own persistent Linux sandbox session (user "runner"), started on demand. Each command runs in a fresh bash shell (pipes, &&, globs and redirects work; cd and exported variables do not carry over to the next call), but files, installed packages and background processes stay while the session runs.
/home/runner/workspace is the user's persistent workspace, the same files list_files and read_file see; work and save there. Everything else (files in /tmp or the home directory, running processes, packages installed by hand) is lost when the session closes. It closes after {idle} minutes without commands (about {mem} MB of memory); the user is warned {warn} minutes before it closes and can keep it open, but anything you need to keep must be in the workspace.
A single command has a time limit: start long jobs in the background (nohup <command> > job.log 2>&1 &) and read the log later. The sandbox has internet access; private and local networks are blocked.
install_package (manager apt or pip) installs into the running session and records the package in installed_packages.txt (apt) or installed_python_packages.txt (pip) in the workspace; the packages listed there are reinstalled whenever a new session starts (remove a line to stop that). sandbox_status shows whether the session and the browser are running, when they close, and the persisted packages."""


class SandboxTools:
    """``sandbox_status`` and the static facts about the persistent sandbox (``prompt_notes``)."""

    def __init__(self, user_id: str, user_settings: dict | None = None, manager=None) -> None:
        self.user_id = str(user_id)
        self.user_settings = user_settings or {}
        self._manager = manager

    @property
    def manager(self):
        return self._manager if self._manager is not None else session_manager()

    @property
    def prompt_notes(self) -> str:
        return SANDBOX_NOTES.format(
            idle=idle_minutes(), warn=warn_minutes(), mem=sessions_constant("SESSION_MEM_MB", DEFAULT_SESSION_MEM_MB)
        )

    @property
    def tools_schema(self) -> list[dict]:
        return [{
            "name": "sandbox_status",
            "description": (
                "Show the state of your persistent sandbox: whether the shell session and the browser are "
                "running and when they close if idle, their memory, the open browser tabs, whether browser "
                "logins are stored, and the packages reinstalled at session start."
            ),
            "input_schema": {"type": "object", "properties": {}},
        }]

    async def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        if tool_name != "sandbox_status":
            return f"Unknown tool: {tool_name}"
        manager = self.manager
        if manager is None:
            return "Error: the sandbox session manager is not available on this server."
        try:
            status = await asyncio.to_thread(manager.status, self.user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("sandbox status failed for %s: %s", self.user_id, e)
            return f"Error: could not read the sandbox status: {e}"
        return format_status(status if isinstance(status, dict) else {}, self.user_settings)


def _closing_line(part: dict, zone, now: datetime) -> str:
    closes = _as_utc(part.get("closes_at"))
    left = part.get("idle_seconds_left")
    if closes is None and isinstance(left, (int, float)):
        closes = now + timedelta(seconds=max(0.0, float(left)))
    if closes is None:
        return ""
    from bot import timezones

    minutes = max(0, round((closes - now).total_seconds() / 60))
    return f"closes at {timezones.format_local(closes, zone)} if idle (in {minutes} min)"


def format_status(status: dict, user_settings: dict | None = None, now: datetime | None = None) -> str:
    """The text ``sandbox_status`` returns, closing times in the user's zone (UTC when unknown)."""
    from bot import timezones

    zone = timezones.zone_of(user_settings or {})
    now = now or datetime.now(timezone.utc)
    lines: list[str] = []

    shell = status.get("shell") if isinstance(status.get("shell"), dict) else {}
    if shell.get("running"):
        parts = ["running"]
        closing = _closing_line(shell, zone, now)
        if closing:
            parts.append(closing)
        used, limit = shell.get("mem_mb"), shell.get("mem_limit_mb")
        if used is not None and limit:
            parts.append(f"memory {used} MB used of {limit} MB")
        elif limit:
            parts.append(f"memory limit {limit} MB")
        lines.append("Shell session: " + ", ".join(parts) + ".")
    else:
        lines.append("Shell session: not running (it starts with the next command).")

    browser = status.get("browser") if isinstance(status.get("browser"), dict) else {}
    if browser.get("running"):
        parts = ["running"]
        closing = _closing_line(browser, zone, now)
        if closing:
            parts.append(closing)
        tabs = [t for t in (browser.get("tabs") or []) if isinstance(t, dict)]
        lines.append("Browser: " + ", ".join(parts) + f"; {len(tabs)} tab(s)" + (":" if tabs else "."))
        for tab in tabs:
            mark = " (active)" if tab.get("active") else ""
            title = str(tab.get("title") or "").strip() or "(untitled)"
            lines.append(f"  - {tab.get('tab_id', '?')}{mark}: {title[:120]} | {str(tab.get('url') or '')[:300]}")
    else:
        lines.append("Browser: not running (it opens with the next browser action).")

    remember = browser_settings(user_settings)["remember"]
    profile = status.get("profile_bytes")
    if not remember:
        lines.append("Browser logins: not remembered (the user turned that off; each browser starts without cookies).")
    elif isinstance(profile, (int, float)) and profile > 0:
        lines.append(f"Browser logins: remembered ({_human_bytes(profile)} profile).")
    else:
        lines.append("Browser logins: none stored yet.")

    packages = status.get("packages") if isinstance(status.get("packages"), dict) else {}
    apt = [str(p) for p in packages.get("apt") or []]
    pip = [str(p) for p in packages.get("pip") or []]
    if apt or pip:
        listed = "; ".join(part for part in (
            f"apt: {', '.join(apt)}" if apt else "", f"pip: {', '.join(pip)}" if pip else ""
        ) if part)
        lines.append(f"Packages reinstalled when a session starts: {listed} "
                     "(installed_packages.txt / installed_python_packages.txt in the workspace).")
    else:
        lines.append("Packages reinstalled when a session starts: none.")

    idle = status.get("idle_timeout_seconds") or sessions_constant("SESSION_IDLE_SECONDS", DEFAULT_IDLE_SECONDS)
    try:
        idle_text = f"{max(1, round(float(idle) / 60))} min"
    except (TypeError, ValueError):
        idle_text = f"{idle_minutes()} min"
    lines.append(f"Idle timeout: {idle_text} (the user is warned {warn_minutes()} minutes before closing).")
    if status.get("warned"):
        lines.append("The user was warned that the sandbox closes soon (they can keep it open from that message).")
    if zone is None:
        lines.append("Times in UTC (the user's time zone is unknown).")
    return "\n".join(lines)


# ---- the browser -----------------------------------------------------------------------
BROWSER_NOTES = """Browser: the browser tools (navigate, read_page, find, left_click, type, form_input, key, scroll, screenshot, zoom, new_tab, ...) drive a real Chromium of your own that stays open between turns and closes after {idle} minutes without use. Use it for interactive work: logging in, filling forms, clicking through sites, pages that need JavaScript. To just read a page, browse_webpage or download_webpage is faster.
Work with element references: read_page (filter "interactive" for controls) or find give refs like ref_7; act on refs, and take a screenshot when the layout matters or a ref does not work. Refs go stale after navigation: read the page again. Several browser actions in one message run in order and stop at the first one that fails.
Before a consequential action (paying or buying, placing an order, subscribing, donating or transferring money, sending a message, post, comment or email, deleting data, changing a password, email or other account or security setting) call browser_confirm with a one-line summary and do it only after the user allowed it; the browser refuses such clicks until then, so repeat the action after the user allowed it. Accepting terms or cookie banners, logging in and registering accounts need no confirmation.
When you hit a captcha or need something only the user knows (a username, a one-time code, a choice), show it with browser_show_screenshot (crop the region), end your turn and ask; the browser stays open, so the next turn continues where you stopped.
Logins and cookies are remembered between browser sessions unless the user turned that off. When you register a new account, generate a strong password, tell it to the user once in your reply and never store it in files or the profile. Never ask the user to type the password of an existing account unless they offer it. Downloads land in downloads/ of the workspace."""


class BrowserTools:
    """The browser toolset plus ``browser_confirm`` and ``browser_show_screenshot`` (see module doc)."""

    toolset_name = TOOLSET_NAME

    def __init__(self, user_id: str, sender=None, user_settings: dict | None = None, manager=None) -> None:
        self.user_id = str(user_id)
        self.sender = sender
        self.user_settings = user_settings or {}
        self._manager = manager
        self._clock = time.monotonic
        self._allow_until = 0.0          # a consequential action is allowed until then (one action)
        self._state_reported = False     # a member result of this turn carried the tab list
        self._last_action = ""
        self._last_meta: dict = {}
        self._live_message = None        # the turn's live-view photo
        self._live_at: float | None = None
        self._live_key = ""              # hash of the last picture + caption shown
        self._live_lock = asyncio.Lock()

    # ---- the provider contract -------------------------------------------------
    @property
    def manager(self):
        return self._manager if self._manager is not None else session_manager()

    @property
    def settings(self) -> dict:
        return browser_settings(self.user_settings)

    @property
    def toolset(self) -> dict:
        """The ``tools`` entry of the toolset (no name); a fresh copy for every request."""
        return copy.deepcopy({
            "type": TOOLSET_TYPE,
            "configs": {member: {"enabled": False} for member in DISABLED_MEMBERS},
        })

    @property
    def prompt_notes(self) -> str:
        return BROWSER_NOTES.format(idle=idle_minutes())

    @property
    def tools_schema(self) -> list[dict]:
        return [
            {
                "name": "browser_confirm",
                "description": (
                    "Ask the user to allow ONE consequential browser action before you do it: paying or buying, "
                    "placing an order, subscribing, donating or transferring money, sending a message/post/comment/"
                    "email, deleting data, changing a password, email or other account/security setting. The user "
                    "sees the current browser view with your summary and taps Allow or Cancel (you wait up to 5 "
                    "minutes). When allowed, repeat the action right away: the permission covers one consequential "
                    "action in the next 5 minutes of this turn. Not needed for accepting terms or cookie banners, "
                    "logging in or registering."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "One line saying exactly what will happen, e.g. 'Place the order for 2 tickets (48 EUR) on example.com'.",
                        },
                    },
                    "required": ["summary"],
                },
            },
            {
                "name": "browser_show_screenshot",
                "description": (
                    "Send the current browser view, or a region of it, to the user's chat as a photo with your "
                    "caption. Use it when the user must see something to answer you (a captcha, a choice only they "
                    "can make, a result worth showing); then end your turn and ask. The browser stays open."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string", "description": "What the user should look at or answer."},
                        "region": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "Optional [x0, y0, x1, y1] in viewport pixels (as in screenshots) to crop, e.g. the captcha.",
                        },
                    },
                    "required": ["caption"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        args = tool_args if isinstance(tool_args, dict) else {}
        if tool_name == "browser_confirm":
            return await self.confirm(args.get("summary"))
        if tool_name == "browser_show_screenshot":
            return await self.show_screenshot(args.get("caption"), args.get("region"))
        return f"Unknown tool: {tool_name}"

    # ---- toolset members ------------------------------------------------------------
    def allowance_active(self) -> bool:
        return self._clock() < self._allow_until

    async def execute_member(self, name: str, tool_input: Any) -> tuple[list | str, bool]:
        """Run one browser member call: ``(content, is_error)``. Never raises (cancellation aside)."""
        manager = self.manager
        if manager is None:
            return "Error: the browser is not available on this server right now.", True
        if name in DISABLED_MEMBERS:
            return f"Error: {name} is not enabled in this environment.", True
        args = dict(tool_input) if isinstance(tool_input, dict) else {}
        request = {
            "id": uuid.uuid4().hex[:12],
            "action": name,
            "input": args,
            "allow_sensitive": self.allowance_active(),
            # The server leaves browser_state out while the tabs are unchanged since its last
            # report, but that report may have gone to another conversation (forum topics share one
            # browser) or to a cancelled turn: until this turn has seen one, ask for it.
            "force_state": not self._state_reported,
        }
        self._last_action = describe_action(name, args)
        try:
            response = await asyncio.to_thread(
                manager.browser_call, self.user_id, request,
                remember=self.settings["remember"], timeout=BROWSER_CALL_TIMEOUT,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if is_capacity_error(e):
                return str(e) or "Error: no browser can start right now; try again in a few minutes.", True
            logger.warning("browser %s failed for %s: %s", name, self.user_id, e)
            return f"Error: the browser action failed: {e}", True
        if not isinstance(response, dict):
            return "Error: the browser gave no answer.", True
        if isinstance(response.get("meta"), dict):
            self._last_meta = response["meta"]
        if response.get("ok"):
            if response.get("sensitive_done"):
                self._allow_until = 0.0  # the one allowed consequential action happened
            self._state_reported = True
            return clean_result_content(response.get("content")), False
        error = str(response.get("error") or "").strip() or "Error: the browser action failed."
        if not error.lower().startswith("error"):
            error = f"Error: {error}"
        return error, True

    # ---- browser_confirm --------------------------------------------------------------
    async def _peek(self) -> dict | None:
        manager = self.manager
        if manager is None:
            return None
        try:
            peek = await asyncio.to_thread(manager.browser_peek, self.user_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.info("browser peek failed for %s: %s", self.user_id, e)
            return None
        if not isinstance(peek, dict) or not peek.get("ok") or not peek.get("image_b64"):
            return None
        return peek

    @staticmethod
    def _peek_image(peek: dict) -> bytes:
        return base64.b64decode(peek["image_b64"])

    async def confirm(self, summary: Any) -> str:
        from bot import questions

        if self.sender is None or getattr(self.sender, "bot", None) is None:
            return "Error: there is no Telegram chat to ask in."
        summary = " ".join(str(summary or "").split())
        if not summary:
            return "Error: 'summary' must say which action needs the user's permission."
        if len(summary) > MAX_SUMMARY:
            summary = summary[: MAX_SUMMARY - 1] + "…"
        peek = await self._peek()
        if peek is not None:
            try:
                await self.sender.send_photo(self._peek_image(peek), filename="browser.jpg", caption=f"🛡 {summary}")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the question still goes out without the picture
                logger.warning("browser_confirm screenshot failed for %s: %s", self.user_id, e)
        try:
            # Cancellation (/cancel, Stop, the turn deadline) propagates: ask() marks the question.
            answer = await questions.ask(
                self.sender, questions.clean_question(f"Allow this action in the browser?\n\n{summary}"),
                [ALLOW_OPTION, CANCEL_OPTION], CONFIRM_TIMEOUT,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("browser_confirm failed for %s: %s", self.user_id, e)
            return f"Error: could not ask the user: {e}"
        if answer is None:
            return (f"No answer within {CONFIRM_TIMEOUT // 60} minutes: do not perform the action. Tell the user in "
                    f"your reply what is waiting for their permission: {summary}")
        if answer == ALLOW_OPTION:
            self._allow_until = self._clock() + ALLOWANCE_SECONDS
            return (f"The user allowed: {summary}. Do it now: repeat the action (the permission covers one "
                    f"consequential action within {ALLOWANCE_SECONDS // 60} minutes).")
        self._allow_until = 0.0
        return f"The user declined: {summary}. Do not perform it; ask what they want instead if unclear."

    # ---- browser_show_screenshot ----------------------------------------------------------
    async def show_screenshot(self, caption: Any, region: Any = None) -> str:
        if self.sender is None or getattr(self.sender, "bot", None) is None:
            return "Error: there is no Telegram chat to send the screenshot to."
        try:
            box = _region(region)
        except ValueError as e:
            return f"Error: {e}"
        peek = await self._peek()
        if peek is None:
            return "Error: the browser is not running; open a page first."
        data = self._peek_image(peek)
        if box is not None:
            try:
                data = await asyncio.to_thread(crop_jpeg, data, box)
            except Exception as e:  # noqa: BLE001
                return f"Error: could not crop the screenshot: {e}"
        text = " ".join(str(caption or "").split()) or "🌐 Browser"
        if len(text) > MAX_CAPTION:
            text = text[: MAX_CAPTION - 1] + "…"
        try:
            await self.sender.send_photo(data, filename="browser.jpg", caption=text)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("browser_show_screenshot failed for %s: %s", self.user_id, e)
            return f"Error: could not send the screenshot: {e}"
        return "Sent."

    # ---- the live view ---------------------------------------------------------------------
    def live_caption(self, meta: dict | None = None) -> str:
        meta = meta if isinstance(meta, dict) else self._last_meta
        title = " ".join(str(meta.get("title") or "").split())[:200] or "(untitled)"
        url = str(meta.get("url") or "")[:400]
        lines = [f"🌐 {title}"]
        if url:
            lines.append(url)
        if self._last_action:
            lines.append(f"▶️ {self._last_action}")
        return "\n".join(lines)[:MAX_CAPTION]

    async def after_batch(self) -> None:
        """Show the current view in the chat after a browser batch (``browser.stream``). Never raises
        (cancellation aside): one photo per turn, edited in place, at most every LIVE_VIEW_INTERVAL s."""
        if not self.settings["stream"] or self.sender is None or getattr(self.sender, "bot", None) is None:
            return
        if self._live_lock.locked():
            return  # an update is in flight; the next batch shows the newer state
        async with self._live_lock:
            now = self._clock()
            if self._live_at is not None and now - self._live_at < LIVE_VIEW_INTERVAL:
                return
            self._live_at = now
            try:
                await asyncio.wait_for(self._update_live_view(), LIVE_VIEW_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the live view is a convenience
                logger.info("browser live view failed for %s: %s", self.user_id, e)

    async def _update_live_view(self) -> None:
        peek = await self._peek()
        if peek is None:
            return
        data = self._peek_image(peek)
        caption = self.live_caption(peek.get("meta") if isinstance(peek.get("meta"), dict) else None)
        key = hashlib.sha1(data + caption.encode("utf-8")).hexdigest()
        if key == self._live_key:
            return  # nothing changed since the last picture
        if self._live_message is None:
            self._live_message = await self.sender.send_photo(data, filename="browser.jpg", caption=caption, silent=True)
        else:
            await self.sender.edit_photo(self._live_message, data, filename="browser.jpg", caption=caption)
        self._live_key = key
