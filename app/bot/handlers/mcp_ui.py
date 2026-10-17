"""``/mcp`` and the ``mcp_`` buttons (docs/design-v3.md, F12). Router on ``ui`` (auth, no quota).

The user connects MCP servers whose tools the agent then gets as ``mcp_<server>_<tool>``
(``agents/mcp_tools.py``); the records live in ``bot.mcp_store``. One legacy-Markdown page
(``bot.ui.answer_md`` / ``edit_md``, plain fallback), every user-derived string escaped and URLs shown
without user info, query or token::

    /mcp                         the list: name, transport, ✅/⬜, tool count, last error;
                                 admin servers marked 🏢 and read-only
    /mcp add <name> <url> [token] [sse]
                                 validate, connect once to list the tools (the count or the error is
                                 reported, the server is saved either way so a refresh can fix it);
                                 the token is kept as a bearer header and the command message is
                                 deleted (best effort) so it does not stay in the chat
    /mcp remove <name>           /mcp refresh [name]          /mcp help

Buttons under the list, per user server: toggle (``mcp_toggle_<name>``), refresh
(``mcp_refresh_<name>``), remove (``mcp_remove_<name>`` → a confirmation with ``mcp_rm_<name>`` /
``mcp_back``), plus "🔄 Refresh all" (``mcp_refreshall``). Names are ``[a-z0-9_-]`` so the data is
split as ``mcp_<action>_<name>``; actions carry no underscore. Store calls run in worker threads;
discovery goes through ``discover_tools`` (lazy ``agents`` import, replaced in tests). The admin
sees the admin servers like everyone else; only the admin's ``/mcp refresh`` re-discovers them too,
as does ``refresh_admin_servers`` at startup (``bot/app.py``).
"""
from __future__ import annotations

import asyncio
import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import mcp_store
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
    "`/mcp add <name> <url> [token] [sse]` — connect a server (name: lowercase letters, digits, - or _; "
    "https URL; the token becomes a Bearer header; add `sse` for a legacy SSE endpoint)\n"
    "`/mcp remove <name>` — forget a server\n"
    "`/mcp refresh [name]` — rediscover the tools\n"
    "`/mcp` — the list with toggle, refresh and remove buttons\n\n"
    "Example: `/mcp add github https://api.githubcopilot.com/mcp/ ghp_xxx`\n\n"
    "_Changing the server set resets my saved reasoning for this chat once; the conversation itself stays._"
)


# ---- discovery (lazy agents import) ------------------------------------------------------
async def discover_tools(user_id: str, server: dict) -> list[dict]:
    """The server's tools via ``MCPTools.discover``; raises with a plain reason on failure."""
    from agents.mcp_tools import MCPTools  # bot/ never imports agents at module level

    return await MCPTools(user_id, servers=[]).discover(server)


async def refresh_one(user_id: str, server: dict) -> dict:
    """Discover and store; the updated record (the error inside when the server could not be reached)."""
    try:
        tools = await discover_tools(user_id, server)
        error = None
    except Exception as e:  # noqa: BLE001 - every reason is a message for the user
        tools, error = None, str(e) or type(e).__name__
    if server.get("admin"):
        return mcp_store.with_tools(server, tools, error)
    updated = await asyncio.to_thread(mcp_store.update_user_tools, user_id, server["name"], tools, error)
    return updated or mcp_store.with_tools(server, tools, error)


async def refresh_admin_servers() -> None:
    """Re-discover every enabled admin server and rewrite the cache (startup task, admin /mcp refresh)."""
    try:
        servers = await asyncio.to_thread(mcp_store.load_admin)
        if not servers:
            return
        refreshed = await asyncio.gather(*(refresh_one("admin", s) for s in servers if s.get("enabled", True)))
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
    if server.get("error"):
        lines.append(f"    ⚠️ {escape_markdown(str(server['error']))}")
    return "\n".join(lines)


def render(servers: list[dict]) -> str:
    if not servers:
        return f"{HEADER}\n\n{EMPTY}\n\nAdd one with `/mcp add <name> <url> [token]` · `/mcp help`"
    return "\n\n".join([HEADER, *(describe(s) for s in servers), "`/mcp help` for the commands"])


def keyboard(servers: list[dict]) -> InlineKeyboardMarkup | None:
    rows = []
    for server in servers:
        if server.get("admin"):
            continue
        name = server["name"]
        rows.append([
            InlineKeyboardButton(text=f"{'✅' if server.get('enabled', True) else '⬜'} {name}", callback_data=f"mcp_toggle_{name}"),
            InlineKeyboardButton(text="🔄", callback_data=f"mcp_refresh_{name}"),
            InlineKeyboardButton(text="🗑", callback_data=f"mcp_remove_{name}"),
        ])
    if rows:
        rows.append([InlineKeyboardButton(text="🔄 Refresh all", callback_data=REFRESH_ALL)])
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def confirm_keyboard(name: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🗑 Yes, remove", callback_data=f"mcp_rm_{name}"),
        InlineKeyboardButton(text="Cancel", callback_data=BACK),
    ]])


async def servers_of(user_id: str) -> list[dict]:
    return await asyncio.to_thread(mcp_store.all_servers, user_id)


async def show_list(message: Message, user_id: str, *, edit: bool) -> None:
    servers = await servers_of(user_id)
    if edit:
        await edit_md(message, render(servers), keyboard(servers))
    else:
        await answer_md(message, render(servers), keyboard(servers))


def added_text(server: dict) -> str:
    name = escape_markdown(server["name"])
    if server.get("error"):
        return (f"⚠️ Saved *{name}*, but it could not be reached: {escape_markdown(server['error'])}\n\n"
                f"Check the URL and token (`/mcp add {name} <url> [token]` replaces it) or try `/mcp refresh {name}` later.")
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
        await answer_md(message, f"Unknown subcommand `{escape_markdown(action)}`.\n\n{HELP}")


async def add_command(message: Message, user_id: str, args: list[str]) -> None:
    try:
        name, url, token, transport = parse_add(args)
        server = mcp_store.new_server(name, url, token, transport, admin=auth.is_admin(user_id))
    except ValueError as e:
        await answer_md(message, f"❌ {escape_markdown(str(e))}")
        return
    if token:
        await delete_quietly(message)  # the token must not stay in the chat
    admin_names = {s["name"] for s in await asyncio.to_thread(mcp_store.load_admin)}
    if server["name"] in admin_names:
        await answer_md(message, f"❌ *{escape_markdown(server['name'])}* is a shared server; pick another name.")
        return
    status = await answer_md(message, f"🔌 Connecting to *{escape_markdown(server['name'])}*…")
    try:
        tools = await discover_tools(user_id, server)
        server = mcp_store.with_tools(server, tools)
    except Exception as e:  # noqa: BLE001
        server = mcp_store.with_tools(server, None, str(e) or type(e).__name__)
    await asyncio.to_thread(mcp_store.add_user_server, user_id, server)
    await edit_md(status, added_text(server))


async def remove_command(message: Message, user_id: str, args: list[str]) -> None:
    if not args:
        await answer_md(message, "Usage: `/mcp remove <name>`")
        return
    name = args[0].lower()
    removed = await asyncio.to_thread(mcp_store.remove_user_server, user_id, name)
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
        await asyncio.to_thread(mcp_store.remove_user_server, user_id, name)
        await show_list(message, user_id, edit=True)
    else:
        logger.warning("Unknown mcp callback from %s: %r", user_id, callback.data)
