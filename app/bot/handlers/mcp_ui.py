"""``/mcp`` and the ``mcp_`` buttons. Router on ``ui`` (auth, no quota).

The user connects MCP servers whose tools the agent then gets as ``mcp_<server>_<tool>``
(``agents/mcp_tools.py``); the records live in ``bot.mcp_store``. One legacy-Markdown page
(``bot.ui.answer_md`` / ``edit_md``, plain fallback), every user-derived string escaped (server names
are ``[a-z0-9_-]`` and go raw only inside code spans, where legacy Markdown honours no escapes) and URLs
shown without user info, query or token::

    /mcp                         the list: name, transport, ✅/⬜, tool count, last error;
                                 admin servers marked 🏢 and read-only
    /mcp add <name> <url> [token] [sse]
                                 validate, connect once to list the tools (the count or the error is
                                 reported, the server is saved either way so a refresh can fix it);
                                 the token is kept as a bearer header and the command message is
                                 deleted (best effort) before anything else, even when the name or
                                 URL is rejected, so the token does not stay in the chat
    /mcp remove <name>           /mcp refresh [name]          /mcp help

OAuth: ``/mcp add <name> <url>`` without a token probes the server with the SDK's OAuth provider; when it
asks for a sign-in (``oauth_flow.SignInRequired``) the record is saved at once as ``auth: "oauth"``, "sign
in needed" (with the probe's client registration, encrypted), and ``oauth_flow.start_sign_in`` runs in
the background: it edits the "Connecting…" message into the 🔐 URL button and later into "✅ <name>
connected (N tools)". A token given explicitly keeps the Bearer header (no OAuth).

Buttons under the list, per user server: toggle (``mcp_toggle_<name>``), refresh
(``mcp_refresh_<name>``), remove (``mcp_remove_<name>`` → a confirmation with ``mcp_rm_<name>`` /
``mcp_back``; removing a server deletes its sign-in with the record and cancels a sign-in to it still
running), plus "🔄 Refresh all" (``mcp_refreshall``). Per OAuth server (the user's or an admin's, each user
signs in with their own account): "🔐 Sign in to <name>" (``mcp_signin_<name>``, the same background
sign-in) while the sign-in is missing or broken, "🚪 Sign out of <name>" (``mcp_signout_<name>``: the
tokens are deleted, the server stays) while signed in. Removal and sign-out go through
``oauth_flow.remove_server`` / ``sign_out``, which wait for a token refresh in flight so it cannot store
the tokens again afterwards. Names are ``[a-z0-9_-]`` so the data is split as ``mcp_<action>_<name>``; actions
carry no underscore. Store calls run in worker threads; discovery goes through ``discover_tools`` (lazy
``agents`` import, replaced in tests). The admin sees the admin servers like everyone else; only the
admin's ``/mcp refresh`` re-discovers them too, as does ``refresh_admin_servers`` at startup
(``bot/app.py``, with the admin's own sign-ins: an OAuth admin server's tool cache is filled once the
admin has signed in to it).

``pasted_callback``: the copy-paste fallback (no public callback endpoint). A plain text starting with
``oauth_flow.LOOPBACK_REDIRECT`` while the chat has a pending sign-in resolves it (its ``state`` must be
that user's) and the message, which holds the code, is deleted. Without a pending sign-in the text goes
on to the agent like any other.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import mcp_store, oauth_flow
from bot.auth import auth
from bot.ui import answer_md, delete_quietly, edit_md, escape_markdown

logger = logging.getLogger(__name__)

router = Router(name="mcp")

PREFIX = "mcp_"
REFRESH_ALL = "mcp_refreshall"
BACK = "mcp_back"
HEADER = "🔌 *MCP servers*"
EMPTY = "No MCP servers yet."
HELP = (
    "🔌 *MCP servers*\n\n"
    "Connect a service that speaks the Model Context Protocol and its tools become mine:\n\n"
    "`/mcp add <name> <url> [token] [sse]` — connect a server (name: lowercase letters, digits, `-` or `_`; "
    "https URL; the token becomes a Bearer header; add `sse` for a legacy SSE endpoint)\n"
    "`/mcp remove <name>` — forget a server\n"
    "`/mcp refresh [name]` — rediscover the tools\n"
    "`/mcp` — the list with toggle, refresh and remove buttons\n\n"
    "Example: `/mcp add github https://api.githubcopilot.com/mcp/ ghp_xxx`\n\n"
    "Services with an account (Notion, Linear, Sentry, …): add them without a token, e.g. "
    "`/mcp add notion https://mcp.notion.com/mcp`, and tap 🔐 Sign in. The sign-in is stored encrypted and "
    "renewed by itself; `/mcp` shows it with Sign in and Sign out buttons.\n\n"
    "_Changing the server set resets my saved reasoning for this chat once; the conversation itself stays._"
)


# ---- discovery (lazy agents import) ------------------------------------------------------
async def discover_tools(user_id: str, server: dict) -> list[dict]:
    """The server's tools via ``MCPTools.discover``; raises with a plain reason on failure."""
    from agents.mcp_tools import MCPTools  # bot/ never imports agents at module level

    return await MCPTools(user_id, servers=[]).discover(server)


async def refresh_one(user_id: str, server: dict) -> dict:
    """Discover and store; the updated record (the error inside when the server could not be reached,
    "sign in needed" and ``auth: "oauth"`` when it asks for a sign-in: never an interactive one here)."""
    auth_flag = None
    try:
        tools = await discover_tools(user_id, server)
        error = None
    except oauth_flow.SignInRequired:
        tools, error, auth_flag = None, mcp_store.SIGN_IN_NEEDED, mcp_store.AUTH_OAUTH
    except Exception as e:  # noqa: BLE001 - every reason is a message for the user
        tools, error = None, str(e) or type(e).__name__
    if server.get("admin"):
        updated = mcp_store.with_tools(server, tools, error)
        if auth_flag:
            updated["auth"] = auth_flag
        return updated
    updated = await asyncio.to_thread(mcp_store.update_user_tools, user_id, server["name"], tools, error,
                                      auth=auth_flag, url=server.get("url"))
    return updated or mcp_store.with_tools(server, tools, error)


async def refresh_admin_servers() -> None:
    """Re-discover every enabled admin server and rewrite the cache (startup task, admin /mcp refresh).
    An OAuth admin server is discovered with the admin's own sign-in (none yet: "sign in needed")."""
    try:
        servers = await asyncio.to_thread(mcp_store.load_admin)
        if not servers:
            return
        owner = getattr(auth, "admin_id", None) or "admin"
        refreshed = await asyncio.gather(*(refresh_one(owner, s) for s in servers if s.get("enabled", True)))
        by_name = {s["name"]: s for s in refreshed}
        merged = [by_name.get(s["name"], s) for s in servers]
        await asyncio.to_thread(mcp_store.save_admin_cache, merged)
        failed = [s["name"] for s in merged if s.get("error")]
        logger.info("MCP admin servers refreshed: %d, %d tools, failed: %s",
                    len(merged), sum(len(s.get("tools") or []) for s in merged), ", ".join(failed) or "none")
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - the bot runs without the admin tools
        logger.warning("Refreshing the MCP admin servers failed: %s", e)


# ---- rendering ------------------------------------------------------------------------------
def describe(server: dict) -> str:
    """One server as Markdown lines: name, transport, state, tool count; URL and headers; the last error."""
    name = escape_markdown(server["name"])
    state = "✅" if server.get("enabled", True) else "⬜"
    count = len(server.get("tools") or [])
    tools = f"{count} tool{'s' if count != 1 else ''}"
    first = f"{'🏢 ' if server.get('admin') else ''}*{name}* · {server.get('transport', 'http')} · {state} · {tools}"
    if server.get("admin"):
        first += " · read-only"
    detail = escape_markdown(mcp_store.display_url(server.get("url", "")))
    headers = mcp_store.describe_headers(server)
    if headers:
        detail += f" · {headers}"
    lines = [first, f"    {detail}"]
    error = server.get("error")
    if error and not (server.get("auth") == mcp_store.AUTH_OAUTH and error == mcp_store.SIGN_IN_NEEDED):
        lines.append(f"    ⚠️ {escape_markdown(str(error))}")  # (the 🔐 state above already says "sign in needed")
    return "\n".join(lines)


def render(servers: list[dict]) -> str:
    if not servers:
        return f"{HEADER}\n\n{EMPTY}\n\nAdd one with `/mcp add <name> <url> [token]` · `/mcp help`"
    return "\n\n".join([HEADER, *(describe(s) for s in servers), "`/mcp help` for the commands"])


def signed_in(server: dict) -> bool:
    return server.get("oauth_state") == mcp_store.STATE_SIGNED_IN


def oauth_row(server: dict) -> list[InlineKeyboardButton]:
    """"🔐 Sign in to <name>" while the sign-in is missing or broken, else "🚪 Sign out of <name>"."""
    name = server["name"]
    if signed_in(server):
        return [InlineKeyboardButton(text=f"🚪 Sign out of {name}", callback_data=f"mcp_signout_{name}")]
    return [InlineKeyboardButton(text=f"🔐 Sign in to {name}", callback_data=f"mcp_signin_{name}")]


def keyboard(servers: list[dict]) -> InlineKeyboardMarkup | None:
    rows = []
    own = 0
    for server in servers:
        is_oauth = server.get("auth") == mcp_store.AUTH_OAUTH
        if server.get("admin"):
            if is_oauth:  # each user signs in to a shared OAuth server with their own account
                rows.append(oauth_row(server))
            continue
        name = server["name"]
        own += 1
        rows.append([
            InlineKeyboardButton(text=f"{'✅' if server.get('enabled', True) else '⬜'} {name}", callback_data=f"mcp_toggle_{name}"),
            InlineKeyboardButton(text="🔄", callback_data=f"mcp_refresh_{name}"),
            InlineKeyboardButton(text="🗑", callback_data=f"mcp_remove_{name}"),
        ])
        if is_oauth:
            rows.append(oauth_row(server))
    if own:
        rows.append([InlineKeyboardButton(text="🔄 Refresh all", callback_data=REFRESH_ALL)])
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def confirm_keyboard(name: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🗑 Yes, remove", callback_data=f"mcp_rm_{name}"),
        InlineKeyboardButton(text="Cancel", callback_data=BACK),
    ]])


async def servers_of(user_id: str) -> list[dict]:
    """The user's servers and the admin's, OAuth ones with this user's ``oauth_state``."""
    return await asyncio.to_thread(partial(mcp_store.all_servers, user_id, with_state=True))


async def show_list(message: Message, user_id: str, *, edit: bool) -> None:
    servers = await servers_of(user_id)
    if edit:
        await edit_md(message, render(servers), keyboard(servers))
    else:
        await answer_md(message, render(servers), keyboard(servers))


def added_text(server: dict) -> str:
    raw = server["name"]  # [a-z0-9_-]: safe inside a code span, where escapes would show as backslashes
    name = escape_markdown(raw)
    if server.get("error"):
        return (f"⚠️ Saved *{name}*, but it could not be reached: {escape_markdown(server['error'])}\n\n"
                f"Check the URL and token (`/mcp add {raw} <url> [token]` replaces it) or try `/mcp refresh {raw}` later.")
    count = len(server.get("tools") or [])
    if not count:
        return f"✅ Connected *{name}*, but it offers no tools."
    return f"✅ Connected *{name}*: {count} tool{'s' if count != 1 else ''}. I can use them from your next message on."


def refreshed_text(servers: list[dict]) -> str:
    lines = []
    for s in servers:
        name = escape_markdown(s["name"])
        if s.get("error"):
            lines.append(f"⚠️ *{name}*: {escape_markdown(s['error'])}")
        else:
            count = len(s.get("tools") or [])
            lines.append(f"✅ *{name}*: {count} tool{'s' if count != 1 else ''}")
    return "\n".join(lines) if lines else "Nothing to refresh."


# ---- /mcp ------------------------------------------------------------------------------------
def parse_add(args: list[str]) -> tuple[str, str, str | None, str]:
    """``add <name> <url> [token] [sse|http]`` → (name, url, token, transport); ValueError when short."""
    if len(args) < 2:
        raise ValueError("Usage: `/mcp add <name> <url> [token] [sse]`")
    name, url, token, transport = args[0], args[1], None, "http"
    for extra in args[2:4]:
        if extra.lower() in mcp_store.TRANSPORTS:
            transport = extra.lower()
        elif token is None:
            token = extra
    return name, url, token, transport


@router.message(F.text, Command("mcp"))
async def mcp_command(message: Message, command: CommandObject, user_id: str) -> None:
    args = (command.args or "").split()
    action = args[0].lower() if args else ""
    if not action:
        await show_list(message, user_id, edit=False)
    elif action == "help":
        await answer_md(message, HELP)
    elif action == "add":
        await add_command(message, user_id, args[1:])
    elif action == "remove":
        await remove_command(message, user_id, args[1:])
    elif action == "refresh":
        await refresh_command(message, user_id, args[1:])
    else:
        await answer_md(message, f"Unknown subcommand \"{escape_markdown(action)}\".\n\n{HELP}")


async def add_command(message: Message, user_id: str, args: list[str]) -> None:
    try:
        name, url, token, transport = parse_add(args)
    except ValueError as e:
        await answer_md(message, f"❌ {e}")  # our own usage line, with its code span
        return
    if token:
        await delete_quietly(message)  # the token must not stay in the chat, whatever the validation says
    try:
        server = mcp_store.new_server(name, url, token, transport, admin=auth.is_admin(user_id))
    except ValueError as e:
        await answer_md(message, f"❌ {escape_markdown(str(e))}")
        return
    admin_names = {s["name"] for s in await asyncio.to_thread(mcp_store.load_admin)}
    if server["name"] in admin_names:
        await answer_md(message, f"❌ *{escape_markdown(server['name'])}* is a shared server; pick another name.")
        return
    status = await answer_md(message, f"🔌 Connecting to *{escape_markdown(server['name'])}*…")
    try:
        tools = await discover_tools(user_id, server)
        server = mcp_store.with_tools(server, tools)
    except oauth_flow.SignInRequired as e:
        await begin_sign_in(user_id, server, status, e.storage)
        return
    except Exception as e:  # noqa: BLE001
        server = mcp_store.with_tools(server, None, str(e) or type(e).__name__)
    await asyncio.to_thread(mcp_store.add_user_server, user_id, server)
    await edit_md(status, added_text(server))


async def begin_sign_in(user_id: str, server: dict, status: Message, probe: "oauth_flow.OAuthStorage | None") -> None:
    """The server asked for a sign-in: save it now (OAuth, "sign in needed", the probe's client registration,
    any older record of that name and its sign-in replaced), then sign in in the background, reporting in
    ``status``. A timeout leaves the record for "🔐 Sign in" in /mcp."""
    record = {**mcp_store.with_tools(server, None, mcp_store.SIGN_IN_NEEDED), "auth": mcp_store.AUTH_OAUTH}
    secrets = probe.snapshot() if probe is not None else {}
    record["oauth"] = await asyncio.to_thread(mcp_store.seal_oauth, user_id, record, secrets)
    await asyncio.to_thread(mcp_store.add_user_server, user_id, record)
    oauth_flow.start_sign_in(user_id, record, status)


async def remove_command(message: Message, user_id: str, args: list[str]) -> None:
    if not args:
        await answer_md(message, "Usage: `/mcp remove <name>`")
        return
    name = args[0].lower()
    removed = await oauth_flow.remove_server(user_id, name)  # a sign-in or refresh in flight cannot bring it back
    if removed:
        await answer_md(message, f"🗑 Removed *{escape_markdown(name)}*.")
    else:
        await answer_md(message, f"❌ No server named *{escape_markdown(name)}* (shared servers cannot be removed here).")


async def refresh_command(message: Message, user_id: str, args: list[str]) -> None:
    servers = await servers_of(user_id)
    is_admin = auth.is_admin(user_id)
    if args:
        name = args[0].lower()
        servers = [s for s in servers if s["name"] == name]
        if not servers:
            await answer_md(message, f"❌ No server named *{escape_markdown(name)}*.")
            return
    targets = [s for s in servers if is_admin or not s.get("admin")]
    if not targets:
        await answer_md(message, "Nothing to refresh.")
        return
    status = await answer_md(message, "🔄 Refreshing…")
    refreshed = await asyncio.gather(*(refresh_one(user_id, s) for s in targets))
    admin_refreshed = [s for s in refreshed if s.get("admin")]
    if admin_refreshed:
        current = await asyncio.to_thread(mcp_store.load_admin)
        by_name = {s["name"]: s for s in admin_refreshed}
        await asyncio.to_thread(mcp_store.save_admin_cache, [by_name.get(s["name"], s) for s in current])
    await edit_md(status, refreshed_text(list(refreshed)))


# ---- buttons --------------------------------------------------------------------------------
@router.callback_query(F.data.startswith(PREFIX))
async def mcp_button(callback: CallbackQuery, user_id: str) -> None:
    try:
        await callback.answer()
    except Exception as e:  # noqa: BLE001 - an old query; the action still happens
        logger.debug("Could not answer the mcp button: %s", e)
    if not isinstance(callback.message, Message):
        return
    message = callback.message
    action, _, name = (callback.data or "")[len(PREFIX):].partition("_")
    if action == "refreshall":
        servers = [s for s in await servers_of(user_id) if not s.get("admin")]
        await edit_md(message, "🔄 Refreshing…")
        await asyncio.gather(*(refresh_one(user_id, s) for s in servers))
        await show_list(message, user_id, edit=True)
    elif action == "back":
        await show_list(message, user_id, edit=True)
    elif action == "toggle" and name:
        state = await asyncio.to_thread(mcp_store.toggle_user_server, user_id, name)
        if state is None:
            logger.info("Toggle of an unknown MCP server %r by %s", name, user_id)
        await show_list(message, user_id, edit=True)
    elif action == "refresh" and name:
        server = mcp_store.find([s for s in await servers_of(user_id) if not s.get("admin")], name)
        if server is not None:
            await edit_md(message, f"🔄 Refreshing *{escape_markdown(name)}*…")
            await refresh_one(user_id, server)
        await show_list(message, user_id, edit=True)
    elif action == "remove" and name:
        await edit_md(message, f"🗑 Remove *{escape_markdown(name)}* and its tools?", confirm_keyboard(name))
    elif action == "rm" and name:
        await oauth_flow.remove_server(user_id, name)  # its sign-in goes with the record; a running one is cancelled
        await show_list(message, user_id, edit=True)
    elif action == "signin" and name:
        server = mcp_store.find(await servers_of(user_id), name)
        if server is None or server.get("auth") != mcp_store.AUTH_OAUTH:
            await show_list(message, user_id, edit=True)
            return
        status = await answer_md(message, f"🔌 Connecting to *{escape_markdown(name)}*…")
        oauth_flow.start_sign_in(user_id, server, status)
    elif action == "signout" and name:
        server = mcp_store.find(await servers_of(user_id), name)
        if server is not None and server.get("auth") == mcp_store.AUTH_OAUTH:
            await oauth_flow.sign_out(user_id, server)  # waits for a refresh in flight, cancels a running sign-in
            logger.info("MCP sign-out of %s for %s", name, user_id)
        await show_list(message, user_id, edit=True)
    else:
        logger.warning("Unknown mcp callback from %s: %r", user_id, callback.data)


# ---- the copy-paste fallback of a sign-in ------------------------------------------------------
PASTE_EXPIRED = "⌛ This sign-in link has expired or was already used. Start again with /mcp."


async def _pasted_while_pending(message: Message) -> bool:
    """Filter: a pasted loopback callback URL while the chat has a pending sign-in. Async on purpose:
    aiogram runs a plain function filter in a worker thread, and the pending logins belong to the loop."""
    return (message.chat is not None and oauth_flow.is_pasted_callback(message.text)
            and oauth_flow.has_pending(str(message.chat.id)))


@router.message(F.text, _pasted_while_pending)
async def pasted_callback(message: Message, user_id: str) -> None:
    params = oauth_flow.parse_callback_url(message.text or "")
    await delete_quietly(message)  # it holds the authorization code
    login = oauth_flow.resolve(params["state"], code=params["code"], error=params["error"],
                               error_description=params["error_description"], iss=params["iss"], user_id=user_id)
    if login is None:
        await answer_md(message, PASTE_EXPIRED)
        return
    logger.info("Pasted OAuth callback for %s of %s", login.server, user_id)
    if params["code"] and not params["error"]:
        await answer_md(message, f"🔐 Got it: finishing the sign-in to *{escape_markdown(login.server)}*…")
