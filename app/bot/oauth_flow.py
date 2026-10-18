"""OAuth sign-in for MCP servers: pending logins, the SDK's token storage, the interactive sign-in task.

The mcp SDK (``mcp.client.auth.OAuthClientProvider``, an httpx2 ``Auth``) does the protocol: a 401 from
the server → protected resource metadata → authorization server metadata → dynamic client registration
→ PKCE authorization URL (it generates the ``state``) → token exchange → retry, and refreshes an expired
access token before a request. This module gives it what it asks for:

* ``OAuthStorage``: the SDK's ``TokenStorage`` for one user's sign-in to one server, over
  ``bot.mcp_store.read_oauth`` / ``write_oauth`` (encrypted, in worker threads), or in memory for the probe
  of ``/mcp add``. The SDK never restores an expiry for stored tokens, and refreshes at the origin's
  ``/token`` when it has discovered no metadata yet; so the storage keeps ``expires_at`` and the discovered
  metadata next to the tokens, hands an expired token to the SDK with an empty access token (which makes it
  refresh first) and puts the metadata back into the provider's context. One refresh at a time per user and
  server (``_refresh_lock``): a second connection waits and reads what the first stored, so a provider that
  rotates refresh tokens never sees the old one twice.
* ``OAuthConnection``: storage + handlers, turned into a provider by ``agents.mcp_tools.open_transport``.
  Without handlers it is non-interactive: reaching the authorization step raises ``NeedsSignIn`` (tool calls,
  refreshes, the probe), so a tool call never starts a login.
* pending logins: ``register`` (from the redirect handler, keyed by the SDK's ``state``, bound to the user
  and server, one per user and server: a new one replaces the old), ``resolve`` (the callback endpoint
  ``bot.oauth_web`` or a pasted loopback URL; one-shot), ``wait_for_code`` (the callback handler), expiry
  ``LOGIN_TIMEOUT``.
* ``start_sign_in``: the background task behind ``/mcp add`` (when the probe asked for a sign-in) and the
  ``mcp_signin_<name>`` button: it edits its status message into "🔐 <name> needs you to sign in." with a URL
  button, waits for the callback, lets the SDK finish, discovers the tools, stores them and edits the message
  into "✅ <name> connected (N tools)" (or "⌛ Sign-in expired …", "❌ …"). "✅" only once the store took the
  tools and the cleared "sign in needed" flag: a server removed or replaced meanwhile ends as
  ``REMOVED_TEXT`` with nothing stored.
* ``sign_out`` / ``remove_server`` (the ``/mcp`` buttons and ``/mcp remove``): cancel a sign-in of that user
  to that server still running (``cancel_sign_in``) and take the refresh lock around the store call, so a
  refresh in flight finishes first and the sign-out (or removal) is what stays.

The redirect URI is the public callback (``PUBLIC_BASE_URL`` + ``/oauth/callback``) while ``bot.oauth_web``
serves it, else ``LOOPBACK_REDIRECT``: the user copies the address of the browser's error page back into the
chat (``mcp_ui.pasted_callback``). Never logged or shown: codes, states, tokens, client secrets (the SDK's
own error logging is cut down to the exception class by ``install_log_filter``). Nothing at import time:
the mcp package is imported inside the functions that need it.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, urlsplit

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import mcp_store
from bot.ui import LEGACY_MARKDOWN, edit_md, escape_markdown, is_not_modified, strip_markdown

logger = logging.getLogger(__name__)

LOGIN_TIMEOUT = 600  # seconds a sign-in link stays valid
LOOPBACK_REDIRECT = "http://localhost:8765/oauth/callback"  # RFC 8252 loopback; the copy-paste fallback
CALLBACK_PATH = "/oauth/callback"
CLIENT_NAME = "GALL.AI"
REFRESH_MARGIN = 60  # an access token this close to its expiry is refreshed first
SIGN_IN_ERROR = "Error: MCP server {name} needs you to sign in again (use /mcp)."
MAX_REASON_CHARS = 300

PUBLIC_HINT = "Tap the button and sign in; I finish the connection by myself. The link works for 10 minutes."
FALLBACK_HINT = ("After you sign in, your browser will show an error page. Copy the whole address from the "
                 "address bar and send it here. The link works for 10 minutes.")
EXPIRED_TEXT = "⌛ Sign-in expired. Try again with /mcp."
REPLACED_TEXT = "⌛ This sign-in was replaced by a newer one."
REMOVED_TEXT = "❌ The server was removed while you were signing in."
SIGNED_OUT_TEXT = "🚪 You signed out, so this sign-in was cancelled."


# ---- errors --------------------------------------------------------------------------------
class NeedsSignIn(Exception):
    """A non-interactive connection reached the step that needs the user in a browser."""


class SignInRequired(Exception):
    """Discovery or a call found that the server needs a sign-in. ``storage`` is the probe's in-memory
    storage (its client registration is reused by the sign-in), None otherwise; ``stale``: stored tokens
    existed but no longer work (refused, refresh failed), as opposed to nothing stored at all."""

    def __init__(self, storage: "OAuthStorage | None" = None, *, stale: bool = False) -> None:
        super().__init__(mcp_store.SIGN_IN_NEEDED)
        self.storage = storage
        self.stale = stale


class UnsafeEndpoint(ValueError):
    """A request of a user's connection to plain http or a non-public address (refused, never sent)."""


class LoginError(Exception):
    """The interactive part of a sign-in ended without a code."""


class LoginExpired(LoginError):
    pass


class LoginReplaced(LoginError):
    pass


class LoginRefused(LoginError):
    def __init__(self, error: str, description: str | None = None) -> None:
        self.error = str(error or "error")[:100]
        self.description = str(description or "")[:200] or None
        super().__init__(self.error)


class SignInLinkRefused(Exception):
    """Telegram refused the message with the sign-in button."""


class ServerGone(Exception):
    """The server was removed (or replaced by a record for another URL) while the sign-in ran: the store
    refused to keep its tokens or tools."""


# ---- loop-bound registries (cleared when the running loop changes: every test has its own) ---------
_loop: Any = None
_pending: dict[str, "PendingLogin"] = {}
_tasks: dict[tuple[str, str], asyncio.Task] = {}
_replaced: set[asyncio.Task] = set()
_stopped: dict[asyncio.Task, str] = {}  # sign-ins cancelled by a removal or sign-out → their final text
_refresh_locks: dict[tuple, asyncio.Lock] = {}


def _clear() -> None:
    _pending.clear()
    _tasks.clear()
    _replaced.clear()
    _stopped.clear()
    _refresh_locks.clear()


def _check_loop() -> asyncio.AbstractEventLoop:
    global _loop
    loop = asyncio.get_running_loop()
    if _loop is not loop:
        _loop = loop
        _clear()
    return loop


def reset() -> None:
    """Forget every pending login, task and lock (tests)."""
    global _loop
    _loop = None
    _clear()


def lock_key(user_id: str, server: dict) -> tuple[str, str, bool]:
    """The refresh lock of one user's sign-in to one server."""
    return str(user_id), str(server["name"]), bool(server.get("admin"))


def _refresh_lock(key: tuple) -> asyncio.Lock:
    _check_loop()
    return _refresh_locks.setdefault(key, asyncio.Lock())


# ---- pending logins ------------------------------------------------------------------------
@dataclass
class CodeResult:
    code: str
    state: str
    iss: str | None = None


@dataclass(eq=False)
class PendingLogin:
    state: str
    user_id: str
    server: str
    future: asyncio.Future
    chat_id: int | None = None
    thread_id: int | None = None
    message_id: int | None = None
    created: float = field(default_factory=time.monotonic)
    code: str | None = None  # kept only to scrub it from an error text

    def remaining(self) -> float:
        return LOGIN_TIMEOUT - (time.monotonic() - self.created)

    def expired(self) -> bool:
        return self.remaining() <= 0


def _settle(login: PendingLogin, *, result: CodeResult | None = None, error: BaseException | None = None) -> None:
    if _pending.get(login.state) is login:
        del _pending[login.state]
    if login.future.done():
        return
    if error is not None:
        login.future.set_exception(error)
    elif result is not None:
        login.future.set_result(result)
    else:
        login.future.cancel()


def _purge() -> None:
    for login in [l for l in _pending.values() if l.expired()]:
        _settle(login, error=LoginExpired())


def register(state: str, user_id: str, server: str, *, chat_id: int | None = None, thread_id: int | None = None,
             message_id: int | None = None) -> PendingLogin:
    """A pending login for the SDK-generated ``state``; replaces the user's pending login to the same server."""
    loop = _check_loop()
    _purge()
    for other in [l for l in _pending.values() if l.user_id == str(user_id) and l.server == server]:
        _settle(other, error=LoginReplaced())
    future = loop.create_future()
    future.add_done_callback(lambda f: f.cancelled() or f.exception())  # nobody may be waiting any more
    login = PendingLogin(state=state, user_id=str(user_id), server=server, future=future, chat_id=chat_id,
                         thread_id=thread_id, message_id=message_id)
    _pending[state] = login
    return login


def resolve(state: str | None, *, code: str | None = None, error: str | None = None,
            error_description: str | None = None, iss: str | None = None, user_id: str | None = None) -> PendingLogin | None:
    """Hand the callback's code (or error) to the waiting sign-in; the login, or None when ``state`` is
    unknown, expired, already used or (with ``user_id``) another user's. One-shot."""
    _check_loop()
    _purge()
    login = _pending.get(str(state or ""))
    if login is None or (user_id is not None and login.user_id != str(user_id)):
        return None
    if error:
        _settle(login, error=LoginRefused(error, error_description))
    elif not code:
        _settle(login, error=LoginRefused("no_code", "the sign-in page sent no code"))
    else:
        login.code = code
        _settle(login, result=CodeResult(code=code, state=login.state, iss=iss))
    return login


def has_pending(user_id: str) -> bool:
    if _loop is None:
        return False
    try:
        if asyncio.get_running_loop() is not _loop:
            return False
    except RuntimeError:
        return False
    _purge()
    return any(l.user_id == str(user_id) for l in _pending.values())


def drop(login: PendingLogin | None) -> None:
    if login is not None:
        _settle(login)


async def wait_for_code(login: PendingLogin):
    """The SDK's callback handler: the code once the user came back, LoginExpired after LOGIN_TIMEOUT."""
    from mcp.client.auth import AuthorizationCodeResult

    try:
        result = await asyncio.wait_for(asyncio.shield(login.future), timeout=max(0.0, login.remaining()))
    except asyncio.TimeoutError:
        drop(login)
        raise LoginExpired() from None
    return AuthorizationCodeResult(code=result.code, state=result.state, iss=result.iss)


def parse_callback_url(text: str) -> dict[str, str | None]:
    """``code``, ``state``, ``error``, ``error_description``, ``iss`` from a callback URL (pasted or requested)."""
    query = parse_qs(urlsplit(str(text or "").strip()).query)
    return {key: (query.get(key) or [None])[0] for key in ("code", "state", "error", "error_description", "iss")}


def is_pasted_callback(text: str | None) -> bool:
    return str(text or "").strip().lower().startswith(LOOPBACK_REDIRECT)


# ---- redirect URI, client metadata, request guard, log filter --------------------------------------
def redirect_uri() -> str:
    """The public callback while ``bot.oauth_web`` serves it, else the loopback (copy-paste) one."""
    from bot import oauth_web

    return oauth_web.callback_url() or LOOPBACK_REDIRECT


def _same_url(a: Any, b: str) -> bool:
    return str(a).rstrip("/") == str(b).rstrip("/")


def client_metadata(redirect: str):
    """What the bot registers as (dynamic client registration): a public client (no secret), code + refresh."""
    from mcp.shared.auth import OAuthClientMetadata

    loopback = redirect.startswith("http://localhost") or redirect.startswith("http://127.0.0.1")
    return OAuthClientMetadata(
        client_name=CLIENT_NAME,
        redirect_uris=[redirect],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        application_type="native" if loopback else "web",
    )


def _origin(scheme: Any, host: Any, port: Any) -> tuple[str, str, int]:
    scheme = str(scheme or "").lower()
    return scheme, str(host or "").lower(), int(port or (443 if scheme == "https" else 80))


def url_origin(url: str) -> tuple[str, str, int] | None:
    """(scheme, host, port with the default filled in) of ``url``, None when it has none."""
    try:
        parts = urlsplit(str(url or ""))
        return _origin(parts.scheme, parts.hostname, parts.port) if parts.hostname else None
    except ValueError:
        return None


def request_guard(server_url: str, *, trust_origin: bool = False) -> Callable[[Any], Awaitable[None]]:
    """An httpx2 request hook for a non-admin user's connections: every request (the MCP endpoint and the
    metadata, registration and token endpoints the OAuth flow discovers, redirects) must be https to a
    public address (``mcp_store.check_public_host``, in a worker thread, once per host and connection).
    Fails closed: a refused request raises ``UnsafeEndpoint`` and is never sent. A user's server: its own
    host was checked before connecting (``MCPTools._check_host``) and still needs https. ``trust_origin``
    (an admin's shared server used by a non-admin user): requests to the configured server's own origin
    (scheme, host and port) pass unchecked, since the admin may run it on plain http or a private address;
    every other origin, the same host on another port or scheme included, needs https and a public address
    like any user's endpoint, so the server cannot point the sign-in at the bot's network."""
    own = url_origin(server_url) if trust_origin else None
    trusted = set() if trust_origin else {(urlsplit(str(server_url or "")).hostname or "").lower()}

    async def guard(request: Any) -> None:
        url = request.url
        host = str(url.host or "").lower()
        if own is not None and _origin(url.scheme, host, url.port) == own:
            return
        if url.scheme != "https":
            raise UnsafeEndpoint(f"Refused a plain-http request to {host}: MCP servers and their sign-in endpoints must use https.")
        if host in trusted:
            return
        literal = f"[{host}]" if ":" in host else host  # an IPv6 literal
        try:
            await asyncio.to_thread(mcp_store.check_public_host, f"https://{literal}:{url.port or 443}/")
        except ValueError:
            raise UnsafeEndpoint(f"Refused a request to {host}: {mcp_store.NOT_PUBLIC}") from None
        trusted.add(host)

    return guard


class _WithholdDetails(logging.Filter):
    """The SDK logs "OAuth flow error" with the exception, whose text can carry a state or a server's
    response body; keep the line, drop everything but the exception class. Our own expected ends of a
    flow (a non-interactive connection reaching the login, an expired or refused login) are not errors:
    the caller reports them, the SDK's line is dropped."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[1] is not None:
            if isinstance(record.exc_info[1], (NeedsSignIn, LoginError)):
                return False
            name = type(record.exc_info[1]).__name__
            record.msg = f"{record.getMessage()} ({name}; details withheld)"
            record.args = None
            record.exc_info = None
            record.exc_text = None
        return True


_LOG_FILTER = _WithholdDetails()


def install_log_filter() -> None:
    for name in ("mcp.client.auth.oauth2", "mcp.client.auth.utils", "mcp.client.auth"):
        logger_ = logging.getLogger(name)
        if _LOG_FILTER not in logger_.filters:
            logger_.addFilter(_LOG_FILTER)


# ---- token storage -------------------------------------------------------------------------
class OAuthStorage:
    """``mcp.client.auth.TokenStorage`` of one user's sign-in to one server (see the module docstring).

    ``persist=False`` keeps everything in memory (the probe of ``/mcp add``). ``for_sign_in``: a stored
    client registered for another redirect URI is ignored (registered again), since the authorization
    request would name the current one; refreshes keep using it."""

    def __init__(self, user_id: str, server: dict, *, persist: bool = True, redirect: str | None = None,
                 for_sign_in: bool = False) -> None:
        self.user_id = str(user_id)
        self.server = {"name": server["name"], "url": server["url"], "admin": bool(server.get("admin"))}
        self.persist = persist
        self.redirect_uri = redirect or redirect_uri()
        self.for_sign_in = for_sign_in
        self.tokens: dict | None = None
        self.expires_at: float | None = None
        self.metadata: dict | None = None
        self.client: dict | None = None
        self.needs_signin = False
        self.provider: Any = None
        self.stale_access: str | None = None  # an access token the server refused: refreshed before reuse
        self.refreshed = False
        self.dropped = False  # a write the store refused: the server was removed or replaced meanwhile
        self._loaded = not persist
        self._held: asyncio.Lock | None = None

    # -- loading and state --
    async def load(self) -> None:
        if self.persist:
            data = await asyncio.to_thread(mcp_store.read_oauth, self.user_id, self.server)
            self.tokens, self.expires_at, self.metadata = data["tokens"], data["expires_at"], data["metadata"]
            self.client, self.needs_signin = data["client"], data["needs_signin"]
        self._loaded = True
        self._restore_metadata()

    async def ensure_loaded(self) -> None:
        if not self._loaded:
            await self.load()

    def has_tokens(self) -> bool:
        return bool(self.tokens and (self.tokens.get("access_token") or self.tokens.get("refresh_token")))

    def can_force_refresh(self) -> bool:
        return bool(self.tokens and self.tokens.get("refresh_token") and self.client
                    and self.stale_access is None and not self.refreshed)

    def force_refresh(self) -> None:
        """The server refused the stored access token: refresh it on the next connection."""
        self.stale_access = str((self.tokens or {}).get("access_token") or "")

    def _needs_refresh(self) -> bool:
        if not self.tokens:
            return False
        access = self.tokens.get("access_token")
        if not access or (self.stale_access is not None and access == self.stale_access):
            return True
        return self.expires_at is not None and time.time() >= self.expires_at - REFRESH_MARGIN

    def snapshot(self) -> dict:
        """The decrypted pieces in ``mcp_store.write_oauth`` / ``seal_oauth`` form."""
        return {"tokens": self._token_payload() if self.tokens else None, "client": self.client}

    def _token_payload(self) -> dict:
        return {"tokens": self.tokens, "expires_at": self.expires_at, "metadata": self.metadata}

    # -- the provider's discovered metadata --
    def attach(self, provider: Any) -> None:
        self.provider = provider
        if self._loaded:
            self._restore_metadata()

    def _restore_metadata(self) -> None:
        context = getattr(self.provider, "context", None)
        if context is None or not self.metadata:
            return
        try:
            from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata

            if context.oauth_metadata is None and self.metadata.get("oauth"):
                context.oauth_metadata = OAuthMetadata.model_validate(self.metadata["oauth"])
            if context.protected_resource_metadata is None and self.metadata.get("prm"):
                context.protected_resource_metadata = ProtectedResourceMetadata.model_validate(self.metadata["prm"])
            if not context.auth_server_url and self.metadata.get("auth_server_url"):
                context.auth_server_url = str(self.metadata["auth_server_url"])
        except Exception as e:  # noqa: BLE001 - the SDK discovers it again on a 401
            logger.debug("Stored OAuth metadata of %s not restored: %s", self.server["name"], type(e).__name__)

    def _capture_metadata(self) -> dict | None:
        context = getattr(self.provider, "context", None)
        if context is None:
            return None
        out: dict[str, Any] = {}
        if context.oauth_metadata is not None:
            out["oauth"] = context.oauth_metadata.model_dump(mode="json", exclude_none=True)
        if context.protected_resource_metadata is not None:
            out["prm"] = context.protected_resource_metadata.model_dump(mode="json", exclude_none=True)
        if context.auth_server_url:
            out["auth_server_url"] = str(context.auth_server_url)
        return out or None

    # -- the refresh lock --
    async def _acquire(self) -> None:
        if self._held is None:  # re-entrant for this storage: the SDK may load the tokens twice
            lock = _refresh_lock(lock_key(self.user_id, self.server))
            await lock.acquire()
            self._held = lock

    def close(self) -> None:
        """Release the refresh lock (after the connection, whatever happened)."""
        if self._held is not None:
            lock, self._held = self._held, None
            if lock.locked():
                lock.release()

    # -- TokenStorage --
    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken

        await self.ensure_loaded()
        if self.tokens and self._needs_refresh():
            await self._acquire()
            if self.persist:
                await self.load()  # the connection that held the lock before may have refreshed already
            if not self.tokens or not self._needs_refresh():
                self.close()
        if not self.tokens:
            return None
        try:
            token = OAuthToken.model_validate(self.tokens)
        except Exception:  # noqa: BLE001 - a stored value that is not a token: signed out
            return None
        if self._needs_refresh():
            # The SDK refreshes only a token it thinks has expired and restores no expiry from storage.
            token = token.model_copy(update={"access_token": ""})
        return token

    async def set_tokens(self, tokens: Any) -> None:
        try:
            self.tokens = tokens.model_dump(mode="json", exclude_none=True)
            self.expires_at = time.time() + int(tokens.expires_in) if tokens.expires_in else None
            self.metadata = self._capture_metadata() or self.metadata
            self.stale_access, self.refreshed, self.needs_signin = None, True, False
            if self.persist:
                self._written(await asyncio.to_thread(mcp_store.write_oauth, self.user_id, self.server,
                                                      tokens=self._token_payload(), needs_signin=False))
        finally:
            self.close()

    def _written(self, ok: bool) -> None:
        """The store refused the write (the record is gone, no longer OAuth or has another URL): this
        connection goes on with what it holds in memory, nothing is stored, the sign-in reports it."""
        if not ok:
            self.dropped = True
            logger.info("MCP sign-in of %s to %s not stored: the server was removed or changed meanwhile",
                        self.user_id, self.server["name"])

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull

        await self.ensure_loaded()
        if not self.client:
            return None
        try:
            info = OAuthClientInformationFull.model_validate(self.client)
        except Exception:  # noqa: BLE001
            return None
        if self.for_sign_in and info.redirect_uris and not any(_same_url(u, self.redirect_uri) for u in info.redirect_uris):
            return None  # registered for another redirect URI (PUBLIC_BASE_URL changed): register again
        return info

    async def set_client_info(self, client_info: Any) -> None:
        self.client = client_info.model_dump(mode="json", exclude_none=True)
        if self.persist:
            self._written(await asyncio.to_thread(mcp_store.write_oauth, self.user_id, self.server, client=self.client))


# ---- connections ---------------------------------------------------------------------------
async def _needs_sign_in(*_: Any) -> Any:
    raise NeedsSignIn()


@dataclass
class OAuthConnection:
    """What ``agents.mcp_tools.open_transport`` turns into the httpx2 ``auth`` of one connection."""

    storage: OAuthStorage
    redirect_handler: Callable[[str], Awaitable[None]] | None = None
    callback_handler: Callable[[], Awaitable[Any]] | None = None

    def provider(self):
        from mcp.client.auth import OAuthClientProvider

        install_log_filter()
        provider = OAuthClientProvider(
            server_url=self.storage.server["url"],
            client_metadata=client_metadata(self.storage.redirect_uri),
            storage=self.storage,
            redirect_handler=self.redirect_handler or _needs_sign_in,
            callback_handler=self.callback_handler or _needs_sign_in,
        )
        self.storage.attach(provider)
        return provider


# ---- the interactive sign-in ----------------------------------------------------------------------
def _is_admin(user_id: str) -> bool:
    from bot.auth import auth

    return auth.is_admin(user_id)


def start_sign_in(user_id: str, server: dict, status: Message) -> asyncio.Task:
    """Run the sign-in to ``server`` in the background, reporting in ``status`` (edited in place).
    A sign-in of the same user to the same server still running is cancelled ("replaced")."""
    _check_loop()
    key = (str(user_id), server["name"])
    old = _tasks.get(key)
    if old is not None and not old.done():
        _replaced.add(old)
        old.cancel()
    task = asyncio.create_task(_sign_in(str(user_id), dict(server), status), name=f"mcp-sign-in-{server['name']}")
    _tasks[key] = task

    def forget(t: asyncio.Task) -> None:
        if _tasks.get(key) is t:
            del _tasks[key]
        _replaced.discard(t)
        _stopped.pop(t, None)

    task.add_done_callback(forget)
    return task


def cancel_sign_in(user_id: str, name: str, text: str) -> bool:
    """Cancel the sign-in of ``user_id`` to ``name`` still running (its message ends as ``text``) and its
    pending login (the link then shows the expired page). True when one was running."""
    _check_loop()
    user_id, name = str(user_id), str(name)
    task = _tasks.get((user_id, name))
    running = task is not None and not task.done()
    if running:
        _stopped[task] = text
        task.cancel()
    for login in [l for l in _pending.values() if l.user_id == user_id and l.server == name]:
        drop(login)
    return running


async def sign_out(user_id: str, server: dict) -> bool:
    """Sign out of ``server`` (``mcp_store.clear_oauth``: the tokens go, the registration stays). A sign-in
    still running is cancelled, and a refresh of this sign-in in flight is waited for (its refresh lock),
    which would otherwise store fresh tokens right after the sign-out. False when nothing was stored."""
    cancel_sign_in(user_id, server["name"], SIGNED_OUT_TEXT)
    async with _refresh_lock(lock_key(user_id, server)):
        return await asyncio.to_thread(mcp_store.clear_oauth, user_id, server)


async def remove_server(user_id: str, name: str) -> bool:
    """Remove the user's server ``name`` with its sign-in (``mcp_store.remove_user_server``), cancelling a
    sign-in to it still running and waiting for a refresh in flight, like ``sign_out``. False when there
    was no such server."""
    name = str(name or "").strip().lower()
    cancel_sign_in(user_id, name, REMOVED_TEXT)
    async with _refresh_lock(lock_key(user_id, {"name": name, "admin": False})):
        return await asyncio.to_thread(mcp_store.remove_user_server, user_id, name)


async def discover_with_sign_in(user_id: str, server: dict, connection: OAuthConnection) -> list[dict]:
    """The tools, discovered through the interactive connection (lazy agents import; tests replace it)."""
    from agents.mcp_tools import DISCOVER_TIMEOUT, MCPTools  # bot/ never imports agents at module level

    return await MCPTools(user_id, servers=[]).discover(server, timeout=LOGIN_TIMEOUT + DISCOVER_TIMEOUT, oauth=connection)


def sign_in_text(name: str, *, fallback: bool) -> str:
    return f"🔐 *{escape_markdown(name)}* needs you to sign in.\n\n{FALLBACK_HINT if fallback else PUBLIC_HINT}"


def sign_in_keyboard(name: str, url: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"🔐 Sign in to {name}", url=url)]])


async def _show_sign_in(status: Message, name: str, url: str, *, fallback: bool) -> None:
    text, markup = sign_in_text(name, fallback=fallback), sign_in_keyboard(name, url)
    try:
        await status.edit_text(text, parse_mode=LEGACY_MARKDOWN, reply_markup=markup)
        return
    except TelegramBadRequest as e:
        if is_not_modified(e):
            return
    try:
        await status.answer(strip_markdown(text), reply_markup=markup)
    except TelegramBadRequest:
        raise SignInLinkRefused("Telegram refused the message with the sign-in button") from None


def _safe_reason(exc: BaseException, login: PendingLogin | None) -> str:
    """The reason for the user: already without URL queries (MCPUnreachable), minus the code and state."""
    reason = str(exc) or type(exc).__name__
    if login is not None:
        for value in (login.code, login.state):
            if value:
                reason = reason.replace(value, "…")
    return reason[:MAX_REASON_CHARS]


def connected_text(name: str, count: int) -> str:
    return (f"✅ *{escape_markdown(name)}* connected ({count} tool{'s' if count != 1 else ''}). "
            "I can use them from your next message on.")


ADMIN_CACHE_NOTE = "Its tools reach me once the admin has signed in to it too (or run /mcp refresh)."


async def _save_tools(user_id: str, server: dict, tools: list[dict]) -> str:
    """Store the discovered tools; a note for the user when they cannot use them yet ("" otherwise).
    ServerGone when the user's server was removed or now points at another URL."""
    if not server.get("admin"):
        stored = await asyncio.to_thread(mcp_store.update_user_tools, user_id, server["name"], tools, None,
                                         auth=mcp_store.AUTH_OAUTH, url=server["url"])
        if stored is None:
            raise ServerGone()
        return ""
    current = await asyncio.to_thread(mcp_store.load_admin)
    if _is_admin(user_id):  # the admin's sign-in fills the shared tool cache of an admin server
        merged = []
        for s in current:
            if s["name"] == server["name"]:
                s = {**mcp_store.with_tools(s, tools), "auth": mcp_store.AUTH_OAUTH}
            merged.append(s)
        await asyncio.to_thread(mcp_store.save_admin_cache, merged)
        return ""
    shared = mcp_store.find(current, server["name"])
    return "" if shared is not None and shared.get("tools") else ADMIN_CACHE_NOTE


async def _sign_in(user_id: str, server: dict, status: Message) -> None:
    name = server["name"]
    storage = OAuthStorage(user_id, server, for_sign_in=True)
    box: dict[str, PendingLogin] = {}
    fallback = storage.redirect_uri == LOOPBACK_REDIRECT
    trusted = _is_admin(user_id)
    own_origin = url_origin(server["url"]) if server.get("admin") else None  # an admin server's own origin may be http

    async def redirect_handler(authorization_url: str) -> None:
        storage.close()  # a failed refresh may have left the refresh lock with us: never hold it while the user signs in
        state = (parse_qs(urlsplit(authorization_url).query).get("state") or [""])[0]
        if not state:
            raise LoginRefused("invalid_request", "the authorization URL has no state")
        if (not trusted and urlsplit(authorization_url).scheme != "https"
                and (own_origin is None or url_origin(authorization_url) != own_origin)):
            raise UnsafeEndpoint("The sign-in page must use https.")
        chat = getattr(status, "chat", None)
        box["login"] = register(state, user_id, name, chat_id=getattr(chat, "id", None),
                                thread_id=getattr(status, "message_thread_id", None),
                                message_id=getattr(status, "message_id", None))
        logger.info("MCP sign-in to %s started for %s", name, user_id)
        await _show_sign_in(status, name, authorization_url, fallback=fallback)

    async def callback_handler():
        return await wait_for_code(box["login"])

    connection = OAuthConnection(storage, redirect_handler, callback_handler)
    try:
        tools = await discover_with_sign_in(user_id, server, connection)
        if storage.dropped:  # the store refused the tokens or the registration: removed or replaced meanwhile
            raise ServerGone()
        note = await _save_tools(user_id, server, tools)
        # Signed in now, whether it took a login or the stored tokens were accepted (then nothing else
        # cleared the flag); False: the server went away since the tools were saved.
        if not await asyncio.to_thread(mcp_store.write_oauth, user_id, storage.server, needs_signin=False):
            raise ServerGone()
        logger.info("MCP sign-in to %s finished for %s: %d tools", name, user_id, len(tools))
        await edit_md(status, connected_text(name, len(tools)) + (f"\n\n{note}" if note else ""))
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task in _replaced:
            await edit_md(status, REPLACED_TEXT)
        elif task in _stopped:
            await edit_md(status, _stopped[task])
        raise
    except ServerGone:
        logger.info("MCP sign-in to %s for %s ended: the server was removed or changed meanwhile, nothing stored", name, user_id)
        await edit_md(status, REMOVED_TEXT)
    except LoginExpired:
        logger.info("MCP sign-in to %s expired for %s", name, user_id)
        await edit_md(status, EXPIRED_TEXT)
    except LoginReplaced:
        await edit_md(status, REPLACED_TEXT)
    except LoginRefused as e:
        logger.info("MCP sign-in to %s refused for %s: %s", name, user_id, e.error)
        detail = f" ({escape_markdown(e.description)})" if e.description else ""
        await edit_md(status, f"❌ Sign-in to *{escape_markdown(name)}* was not completed: {escape_markdown(e.error)}{detail}")
    except Exception as e:  # noqa: BLE001 - every reason becomes a message without secrets
        reason = _safe_reason(e, box.get("login"))
        logger.warning("MCP sign-in to %s failed for %s: %s", name, user_id, reason)
        await edit_md(status, f"❌ Sign-in to *{escape_markdown(name)}* failed: {escape_markdown(reason)}")
    finally:
        drop(box.get("login"))
        storage.close()


async def shutdown() -> None:
    """Cancel the running sign-ins (bot shutdown); nothing is announced."""
    tasks = [t for t in list(_tasks.values()) if not t.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    for login in list(_pending.values()):
        drop(login)
