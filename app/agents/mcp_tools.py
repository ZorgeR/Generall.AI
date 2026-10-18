"""MCPTools: the tools of the user's connected MCP servers as agent tools.

Servers and their cached tool lists come from ``bot.mcp_store`` (the user's ``data/<uid>/mcp.json``
plus the admin's servers). Nothing touches the network at turn start: ``tools_schema`` is built once
per instance from the cache, every enabled server's tools as ``mcp_<server>_<tool>`` (sanitized to
``[a-zA-Z0-9_-]{1,64}``, a numeric suffix on a collision) with the description prefixed
``[MCP <server>] `` and the input schema passed through (an object schema; anything else becomes an
empty object schema). The tool list is part of the transcript prefix fingerprint, so adding, removing
or toggling a server strips the stored thinking of that user once; that is expected.

Every call opens a fresh connection: the transport named by ``server["transport"]`` (streamable HTTP
or SSE, with the stored headers), a ``ClientSession``, ``initialize()``, then ``list_tools()`` for
``discover`` (used by ``/mcp add`` and refresh) or ``call_tool()`` for ``execute_tool``, closed again.
Results: text content becomes the joined text; an image becomes a base64 ``image`` block (the Messages
API types only), so the result is a list of blocks when there is an image; an embedded resource its
text or ``[resource <uri>]``; ``isError`` → ``Error: <text>``. ``execute_tool`` never raises: an
unreachable server is ``Error: MCP server <name> unreachable: <reason>``, recorded in the store (best
effort) so ``/mcp`` shows it. The reason goes to the user, the store and the transcript, so every URL
in it is shown as ``mcp_store.display_url`` shows it: httpx embeds the full request URL, query string
and all, in its status errors, and a key passed in the query must not leak that way.

OAuth (``bot.oauth_flow``): a server with ``"auth": "oauth"`` connects with an
``mcp.client.auth.OAuthClientProvider`` as the httpx2 ``auth``, over the user's stored (encrypted) tokens
and without interactive handlers: the SDK refreshes an expired access token by itself; when the refresh
fails or nothing is stored, the call answers ``Error: MCP server <name> needs you to sign in again (use
/mcp).``, the record says "sign in needed" and ``/mcp`` offers the button (one retry with a forced refresh
first when the server refused a token that still looked valid). A tool call never starts a login. A
server without credentials is discovered with a provider over an empty in-memory storage (the probe of
``/mcp add`` and refresh): a server that answers 401 raises ``oauth_flow.SignInRequired`` instead of an
error. ``discover(..., oauth=connection)`` is the interactive sign-in (``oauth_flow.start_sign_in``). The
provider and the request guard reach ``open_transport`` through the ``_CONNECTION`` context variable, so
the transport keeps its ``(server, timeout)`` signature. Every request of a non-admin user's connection
(redirects and the OAuth endpoints the SDK discovers included) passes ``oauth_flow.request_guard``: https
to a public address, or it is never sent.

The mcp package is imported lazily (it costs almost a second) inside ``open_transport`` and
``client_session``, which tests replace with fakes (``HTTP_TRANSPORT`` puts an ``httpx2.MockTransport``
under the real ones). mcp 1.x and 2.x differ in attribute names
(``inputSchema`` / ``input_schema``, ``mimeType`` / ``mime_type``, ``isError`` / ``is_error``) and in
how the streamable HTTP client takes headers; ``_field`` and ``open_transport`` cover both.
Runs in the bot process, never sandboxed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import Any

from agents import content
from bot import mcp_store, oauth_flow

logger = logging.getLogger(__name__)

TOOL_PREFIX = "mcp_"
MAX_TOOL_NAME = 64
_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]+")
DEFAULT_INPUT_SCHEMA: dict = {"type": "object", "properties": {}}
DISCOVER_TIMEOUT = 30
CALL_TIMEOUT = 60
MAX_DESCRIPTION_CHARS = 2000
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}  # what the Messages API accepts
MAX_REASON_CHARS = 300

PROMPT_NOTES = (
    "Tools whose names start with mcp_ are the user's own connected services (MCP servers set up with /mcp); "
    "the part after mcp_ names the server, then the tool, so the server name tells you which service a tool "
    "belongs to. Use them like any other tool. If such a tool answers that the server is unreachable or needs a "
    "new token, say so and point the user to /mcp."
)


class MCPUnreachable(Exception):
    """Discovery or a call could not reach the server (or it refused); str() is the plain reason."""


# ---- names and schemas ------------------------------------------------------------------
def sanitize_tool_name(server_name: str, tool_name: str) -> str:
    """``mcp_<server>_<tool>`` with every other character replaced by ``_``, cut to 64."""
    raw = f"{TOOL_PREFIX}{server_name}_{tool_name}"
    safe = _UNSAFE.sub("_", raw).strip("_") or TOOL_PREFIX.rstrip("_")
    if not safe.startswith(TOOL_PREFIX):
        safe = TOOL_PREFIX + safe
    return safe[:MAX_TOOL_NAME]


def unique_tool_name(base: str, taken: set[str]) -> str:
    """``base``, or ``base_2``, ``base_3`` ... (the base shortened so the suffix still fits 64)."""
    if base not in taken:
        return base
    n = 2
    while True:
        suffix = f"_{n}"
        candidate = base[: MAX_TOOL_NAME - len(suffix)] + suffix
        if candidate not in taken:
            return candidate
        n += 1


def input_schema_of(schema: Any) -> dict:
    """A JSON-schema object for the API: the server's when it is one, else an empty object schema."""
    if not isinstance(schema, dict):
        return dict(DEFAULT_INPUT_SCHEMA)
    kind = schema.get("type", "object")
    if kind != "object":
        return dict(DEFAULT_INPUT_SCHEMA)
    out = dict(schema)
    out["type"] = "object"
    if not isinstance(out.get("properties"), dict):
        out["properties"] = {}
    return out


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """The first attribute (or key) present among ``names``: mcp 2.x snake_case or 1.x camelCase."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return default


def convert_tool(tool: Any) -> dict:
    """One ``mcp.types.Tool`` (or a dict) as the cached ``{name, description, input_schema}``."""
    return {
        "name": str(_field(tool, "name", default="") or ""),
        "description": str(_field(tool, "description", default="") or ""),
        "input_schema": input_schema_of(_field(tool, "input_schema", "inputSchema")),
    }


# ---- results -----------------------------------------------------------------------------
def convert_content(items: Any) -> list[dict]:
    """MCP content items as Messages API blocks (``text`` and base64 ``image`` only)."""
    blocks: list[dict] = []
    for item in items or []:
        kind = _field(item, "type", default="")
        if kind == "text":
            blocks.append({"type": "text", "text": str(_field(item, "text", default="") or "")})
        elif kind == "image":
            mime = str(_field(item, "mime_type", "mimeType", default="") or "").lower()
            data = _field(item, "data", default="")
            if mime in SUPPORTED_IMAGE_TYPES and data and len(str(data)) > content.MAX_IMAGE_BYTES:
                # The API refuses the whole request for one image over its limit (the base64 data).
                megabytes = len(str(data)) * 3 / 4 / (1024 * 1024)
                blocks.append({"type": "text", "text": f"[image {mime} omitted: too large to show ({megabytes:.1f} MB)]"})
            elif mime in SUPPORTED_IMAGE_TYPES and data:
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": str(data)}})
            else:
                blocks.append({"type": "text", "text": f"[image {mime or 'of unknown type'} omitted: not displayable]"})
        elif kind == "resource":
            resource = _field(item, "resource")
            text = _field(resource, "text")
            uri = _field(resource, "uri", default="")
            blocks.append({"type": "text", "text": str(text) if text is not None else f"[resource {uri}]"})
        elif kind == "resource_link":
            uri = _field(item, "uri", default="")
            name = _field(item, "name", default="")
            blocks.append({"type": "text", "text": f"[resource {uri}{f' ({name})' if name else ''}]"})
        elif kind == "audio":
            blocks.append({"type": "text", "text": f"[audio {_field(item, 'mime_type', 'mimeType', default='')} omitted]"})
        elif kind:
            blocks.append({"type": "text", "text": f"[{kind} content omitted]"})
    return blocks


def _joined_text(blocks: list[dict]) -> str:
    return "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text").strip()


def convert_result(result: Any) -> str | list[dict]:
    """A ``CallToolResult`` as the tool result: an error string, the joined text, or blocks with images."""
    blocks = convert_content(_field(result, "content", default=[]))
    if _field(result, "is_error", "isError", default=False):
        text = _joined_text(blocks) or "the server reported an error without details"
        return text if text.lower().startswith("error") else f"Error: {text}"
    if not blocks:
        structured = _field(result, "structured_content", "structuredContent")
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False, default=str)
        return "(the tool returned no content)"
    if any(b.get("type") == "image" for b in blocks):
        return blocks
    return _joined_text(blocks) or "(the tool returned empty text)"


_URL_RE = re.compile(r"https?://[^\s'\"<>]+")
_HTTPX_HINT_RE = re.compile(r"\s*For more information check:.*$", re.S)  # httpx appends an MDN link to status errors


def redact_urls(message: str) -> str:
    """Every URL in ``message`` reduced to scheme, host and path (``mcp_store.display_url``): no user
    info, query or fragment, so a key in the query string never reaches the user or the transcript.
    Punctuation right after a URL (``..., url.`` or ``(url)``) stays as it was."""

    def one(match: re.Match) -> str:
        url = match.group(0).rstrip(".,;:)]")
        return mcp_store.display_url(url) + match.group(0)[len(url):]

    return _URL_RE.sub(one, message)


def error_reason(exc: BaseException) -> str:
    """A short plain reason: the first leaf of an ExceptionGroup, the class name when the message is
    empty, URLs without their query string."""
    seen = 0
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions and seen < 10:  # anyio task groups wrap failures
        exc = exc.exceptions[0]
        seen += 1
    if isinstance(exc, asyncio.TimeoutError):
        return "timed out"
    message = redact_urls(_HTTPX_HINT_RE.sub("", str(exc))).strip()
    name = type(exc).__name__
    reason = message if message and message.lower() != name.lower() else name
    if message and name not in ("MCPError", "McpError", "MCPUnreachable", "RuntimeError", "ValueError", "Exception"):
        reason = f"{name}: {message}"
    return reason[:MAX_REASON_CHARS]


def find_exception(exc: BaseException | None, kind: type | tuple, depth: int = 0) -> BaseException | None:
    """The first ``kind`` in ``exc``, its exception-group leaves and its causes (anyio wraps a failure
    of the transport's task group, the SDK's auth flow raises from inside httpx2)."""
    if exc is None or depth > 12:
        return None
    if isinstance(exc, kind):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            found = find_exception(inner, kind, depth + 1)
            if found is not None:
                return found
    return find_exception(exc.__cause__ or exc.__context__, kind, depth + 1)


# ---- connections (replaced by fakes in tests) ---------------------------------------------------
@dataclass
class Connection:
    """Per-connection options for ``open_transport``: the OAuth connection and the request guard."""

    oauth: Any = None  # bot.oauth_flow.OAuthConnection
    guard: Any = None  # an async httpx2 request hook


_CONNECTION: ContextVar[Connection | None] = ContextVar("mcp_connection", default=None)
HTTP_TRANSPORT: Any = None  # tests: an httpx2 transport (MockTransport) under every client built here


def _http_client(headers: dict | None = None, timeout: Any = None, auth: Any = None, *, hooks: list | None = None):
    """``create_mcp_http_client`` (the SDK's timeouts) plus the request hooks; matches the SDK's
    ``McpHttpClientFactory`` so ``sse_client`` can use it too."""
    if HTTP_TRANSPORT is not None:
        import httpx2

        client = httpx2.AsyncClient(headers=headers, timeout=timeout or httpx2.Timeout(30.0, read=300.0), auth=auth,
                                    transport=HTTP_TRANSPORT)
    else:
        from mcp.client.streamable_http import create_mcp_http_client

        client = create_mcp_http_client(headers=headers, timeout=timeout, auth=auth)
    if hooks:
        client.event_hooks = {"request": list(hooks), "response": []}
    return client


@asynccontextmanager
async def open_transport(server: dict, timeout: float):
    """The transport of ``server`` as an async context manager yielding ``(read_stream, write_stream)``;
    the OAuth provider and the request guard of ``_CONNECTION`` (set by ``open_session``) are applied."""
    url = str(server.get("url") or "")
    headers = {str(k): str(v) for k, v in (server.get("headers") or {}).items()} or None
    options = _CONNECTION.get()
    auth = options.oauth.provider() if options is not None and options.oauth is not None else None
    hooks = [options.guard] if options is not None and options.guard is not None else None
    if str(server.get("transport") or "http") == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(url, headers=headers, timeout=timeout, auth=auth,
                              httpx_client_factory=partial(_http_client, hooks=hooks)) as streams:
            yield streams[0], streams[1]
        return
    try:
        from mcp.client.streamable_http import streamable_http_client  # mcp 2.x
    except ImportError:  # pragma: no cover - mcp 1.x
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(url, headers=headers, auth=auth) as streams:
            yield streams[0], streams[1]
        return
    async with _http_client(headers=headers, auth=auth, hooks=hooks) as http_client:
        async with streamable_http_client(url, http_client=http_client) as streams:
            yield streams[0], streams[1]


def client_session(read, write, timeout: float):
    """An ``mcp.ClientSession`` over the streams (an async context manager with initialize/list_tools/call_tool)."""
    from mcp import ClientSession

    return ClientSession(read, write, read_timeout_seconds=timeout)


@asynccontextmanager
async def open_session(server: dict, timeout: float, connection: Connection | None = None):
    token = _CONNECTION.set(connection)
    try:
        async with open_transport(server, timeout) as (read, write):
            async with client_session(read, write, timeout) as session:
                await session.initialize()
                yield session
    finally:
        _CONNECTION.reset(token)


async def _list_tools(session, cursor: str | None):
    """One ``tools/list`` page: mcp 2.x takes ``params=``, 1.x took ``cursor=``."""
    if not cursor:
        return await session.list_tools()
    try:
        from mcp.types import PaginatedRequestParams

        return await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
    except (ImportError, TypeError):
        return await session.list_tools(cursor=cursor)


# ---- the provider --------------------------------------------------------------------------
class MCPTools:
    """Async tool provider; ``servers`` overrides the store (tests, previews)."""

    def __init__(self, user_id: str, servers: list[dict] | None = None) -> None:
        self.user_id = str(user_id)
        self._servers: list[dict] | None = list(servers) if servers is not None else None
        self._schema: list[dict] | None = None
        self._map: dict[str, tuple[dict, str]] = {}  # tool name -> (server record, remote tool name)

    # -- the cached tool list --
    def _load(self) -> None:
        if self._schema is not None:
            return
        if self._servers is None:
            try:
                self._servers = mcp_store.all_servers(self.user_id)
            except Exception as e:  # noqa: BLE001 - a broken file must not break the turn
                logger.error("Could not load the MCP servers of %s: %s", self.user_id, e)
                self._servers = []
        schema: list[dict] = []
        taken: set[str] = set()
        for server in self._servers:
            if not server.get("enabled", True):
                continue
            for tool in server.get("tools") or []:
                remote = str(tool.get("name") or "")
                if not remote:
                    continue
                name = unique_tool_name(sanitize_tool_name(server["name"], remote), taken)
                taken.add(name)
                self._map[name] = (server, remote)
                description = str(tool.get("description") or "").strip()[:MAX_DESCRIPTION_CHARS]
                schema.append({
                    "name": name,
                    "description": f"[MCP {server['name']}] {description}".rstrip(),
                    "input_schema": input_schema_of(tool.get("input_schema")),
                })
        self._schema = schema

    @property
    def tools_schema(self) -> list[dict]:
        self._load()
        return list(self._schema or [])

    @property
    def prompt_notes(self) -> str:
        """Only when the user has MCP tools: nobody else's prompt mentions them."""
        return PROMPT_NOTES if self.tools_schema else ""

    def resolve(self, tool_name: str) -> tuple[dict, str] | None:
        self._load()
        return self._map.get(tool_name)

    # -- connection policy --
    def _trusted(self, server: dict) -> bool:
        """The admin's servers (shared or their own) skip the public-address checks."""
        if server.get("admin"):
            return True
        from bot.auth import auth  # bot/auth has no import-time side effects

        return auth.is_admin(self.user_id)

    def _connection(self, server: dict, oauth: Any = None) -> Connection:
        guard = None if self._trusted(server) else oauth_flow.request_guard(str(server.get("url") or ""))
        return Connection(oauth=oauth, guard=guard)

    @staticmethod
    def uses_oauth(server: dict) -> bool:
        return server.get("auth") == mcp_store.AUTH_OAUTH

    @staticmethod
    def has_credentials(server: dict) -> bool:
        """Stored headers (a bearer token, an API key header): such a server is never probed for OAuth."""
        return bool(server.get("headers"))

    async def _with_stored_sign_in(self, server: dict, run, timeout: float):
        """``run(connection)`` over the user's stored sign-in, non-interactive. SignInRequired when nothing
        usable is stored or the server still asks for a sign-in after one forced refresh (a token that
        looked valid but was refused, or one without a known expiry)."""
        storage = oauth_flow.OAuthStorage(self.user_id, server)
        try:
            await storage.load()
            if not storage.has_tokens():
                raise oauth_flow.SignInRequired()
            for attempt in range(2):
                try:
                    result = await asyncio.wait_for(run(oauth_flow.OAuthConnection(storage)), timeout)
                except asyncio.CancelledError:
                    raise
                except BaseException as e:  # noqa: BLE001
                    if find_exception(e, oauth_flow.NeedsSignIn) is None:
                        raise
                    if attempt == 0 and storage.can_force_refresh():
                        storage.force_refresh()
                        continue
                    raise oauth_flow.SignInRequired(stale=True) from None
                if storage.needs_signin and not storage.refreshed:  # it works again: drop the stale flag
                    await asyncio.to_thread(mcp_store.write_oauth, self.user_id, server, needs_signin=False)
                return result
        finally:
            storage.close()

    async def _mark_signin_needed(self, server: dict) -> None:
        try:
            await asyncio.to_thread(mcp_store.mark_signin_needed, self.user_id, server)
            if not server.get("admin"):
                server["error"] = mcp_store.SIGN_IN_NEEDED
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not mark the sign-in of %s: %s", server.get("name"), e)

    # -- discovery --
    async def discover(self, server: dict, timeout: float = DISCOVER_TIMEOUT, *, oauth: Any = None) -> list[dict]:
        """Connect once and list the tools as ``{name, description, input_schema}``.

        MCPUnreachable on failure; ``oauth_flow.SignInRequired`` when the server needs a sign-in (an OAuth
        server whose stored sign-in is missing or no longer works, or a server without credentials that
        answered 401: then ``.storage`` holds the probe's client registration). ``oauth``: the interactive
        connection of a sign-in; its LoginError (expired, refused, replaced) is raised as it is."""
        try:
            if oauth is not None:
                return await asyncio.wait_for(self._discover(server, timeout, oauth), timeout)
            if self.uses_oauth(server):
                try:
                    return await self._with_stored_sign_in(server, partial(self._discover, server, timeout), timeout)
                except oauth_flow.SignInRequired as e:
                    if e.stale:
                        await self._mark_signin_needed(server)
                    raise
            if not self.has_credentials(server):
                probe = oauth_flow.OAuthStorage(self.user_id, server, persist=False)
                try:
                    return await asyncio.wait_for(self._discover(server, timeout, oauth_flow.OAuthConnection(probe)), timeout)
                except asyncio.CancelledError:
                    raise
                except BaseException as e:  # noqa: BLE001
                    if find_exception(e, oauth_flow.NeedsSignIn) is not None:
                        raise oauth_flow.SignInRequired(probe) from None
                    raise
            return await asyncio.wait_for(self._discover(server, timeout, None), timeout)
        except (asyncio.CancelledError, oauth_flow.SignInRequired):
            raise
        except BaseException as e:  # noqa: BLE001 - anyio raises ExceptionGroups; every reason becomes one message
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            login_error = find_exception(e, oauth_flow.LoginError)
            if login_error is not None:
                raise login_error from None
            raise MCPUnreachable(error_reason(e)) from None

    async def _check_host(self, server: dict) -> None:
        """A user's server must be on a public address, checked after DNS before every connection:
        the bot runs on the host network, and the record may come straight from ``mcp.json``, which
        the sandbox can write. The admin's servers (shared or their own) are trusted."""
        if self._trusted(server):
            return
        await asyncio.to_thread(mcp_store.check_public_host, str(server.get("url") or ""))

    async def _discover(self, server: dict, timeout: float, oauth: Any = None) -> list[dict]:
        await self._check_host(server)
        tools: list[dict] = []
        async with open_session(server, timeout, self._connection(server, oauth)) as session:
            cursor = None
            for _ in range(20):  # pages; a server with more than ~1000 tools is not meant for a chat bot
                result = await _list_tools(session, cursor)
                tools.extend(convert_tool(t) for t in (_field(result, "tools", default=[]) or []))
                cursor = _field(result, "next_cursor", "nextCursor")
                if not cursor:
                    break
        return [t for t in tools if t["name"]]

    # -- calls --
    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str | list:
        resolved = self.resolve(tool_name)
        if resolved is None:
            return f"Unknown tool: {tool_name}"
        server, remote = resolved
        try:
            if self.uses_oauth(server):
                run = partial(self._call, server, remote, tool_args or {}, CALL_TIMEOUT)
                result = await self._with_stored_sign_in(server, lambda connection: run(oauth=connection), CALL_TIMEOUT)
            else:
                result = await asyncio.wait_for(self._call(server, remote, tool_args or {}, CALL_TIMEOUT), CALL_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except oauth_flow.SignInRequired as e:
            logger.info("MCP call %s needs a sign-in to %s for %s", tool_name, server.get("name"), self.user_id)
            if e.stale:  # nothing stored at all already reads as "sign in needed"
                await self._mark_signin_needed(server)
            return oauth_flow.SIGN_IN_ERROR.format(name=server["name"])
        except BaseException as e:  # noqa: BLE001 - a tool must never take the turn down
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            reason = error_reason(e)
            logger.warning("MCP call %s (%s) failed for %s: %s", tool_name, server.get("url"), self.user_id, reason)
            await self._remember_error(server, reason)
            return f"Error: MCP server {server['name']} unreachable: {reason}"
        if server.get("error"):
            await self._remember_error(server, None)  # it works again
        return result

    async def _call(self, server: dict, remote: str, args: dict, timeout: float, oauth: Any = None) -> str | list:
        await self._check_host(server)
        async with open_session(server, timeout, self._connection(server, oauth)) as session:
            result = await session.call_tool(remote, dict(args))
        return convert_result(result)

    async def _remember_error(self, server: dict, reason: str | None) -> None:
        try:
            await asyncio.to_thread(mcp_store.record_error, self.user_id, server, reason)
            server["error"] = reason
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not record the MCP error of %s: %s", server.get("name"), e)
