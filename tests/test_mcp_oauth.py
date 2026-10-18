"""OAuth sign-in for MCP servers: bot/oauth_flow.py with the real mcp SDK provider against a fake OAuth +
MCP server (tests/mcp_oauth_fakes.py, an httpx2 MockTransport), the encrypted storage in bot/mcp_store.py,
the /mcp UI paths (add, sign in, sign out, the pasted callback) and tool calls with refresh. No network."""
import asyncio
import json
import logging
import os
import stat
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import httpx2
import pytest
from aiogram.filters import CommandObject
from aiogram.types import Chat, Message
from cryptography.fernet import Fernet

import agents.mcp_tools as mt
from bot import mcp_store, oauth_flow, oauth_web
from bot.handlers import mcp_ui
from mcp_oauth_fakes import AUTH, MCP_URL, FakeOAuthServer

USER = "42"
OTHER = "43"
ADMIN = "1"
SECRET_MARKERS = ("SECRETCODE", "SECRETACCESS", "SECRETREFRESH")


class FakeAuth:
    admin_id = ADMIN

    def is_admin(self, user_id):
        return str(user_id) == ADMIN


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Files under tmp_path, a fresh token key, no DNS, the fake server under every MCP client."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(mcp_store.ENV_ADMIN_FILE, str(tmp_path / "data" / "mcp_servers.json"))
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, Fernet.generate_key().decode())
    monkeypatch.delenv(mcp_store.ENV_TOKEN_KEY_OLD, raising=False)
    mcp_store.reset_cipher()
    oauth_flow.reset()
    oauth_web.reset()

    def unresolvable(host, port):
        raise OSError("Name or service not known")

    monkeypatch.setattr(mcp_store, "_resolve", unresolvable)
    fake = FakeOAuthServer()
    monkeypatch.setattr(mt, "HTTP_TRANSPORT", httpx2.MockTransport(fake.handler))
    fake_auth = FakeAuth()
    monkeypatch.setattr(mcp_ui, "auth", fake_auth)
    import bot.auth

    monkeypatch.setattr(bot.auth, "auth", fake_auth)
    yield fake
    oauth_flow.reset()
    oauth_web.reset()
    mcp_store.reset_cipher()


def make_message(text="/mcp", chat_id=int(USER)):
    msg = Message(message_id=7, date=datetime.now(timezone.utc), chat=Chat(id=chat_id, type="private"), text=text)
    status = Message(message_id=8, date=datetime.now(timezone.utc), chat=Chat(id=chat_id, type="private"), text="status")
    for m in (msg, status):
        object.__setattr__(m, "edit_text", AsyncMock(return_value=m))
        object.__setattr__(m, "delete", AsyncMock(return_value=True))
        object.__setattr__(m, "answer", AsyncMock(return_value=status))
    return msg, status


async def mcp(text, user_id=USER):
    message, status = make_message(f"/mcp {text}".strip(), chat_id=int(user_id))
    await mcp_ui.mcp_command(message, CommandObject(prefix="/", command="mcp", args=text or None), user_id=user_id)
    return message, status


def sign_in_url(status) -> str | None:
    """The URL of the 🔐 button the sign-in put on ``status`` (None while it is not there yet)."""
    for call in status.edit_text.await_args_list:
        markup = call.kwargs.get("reply_markup")
        if markup is not None:
            for row in markup.inline_keyboard:
                for button in row:
                    if button.url:
                        return button.url
    return None


async def wait_for_button(status, timeout=5.0) -> str:
    for _ in range(int(timeout / 0.01)):
        url = sign_in_url(status)
        if url:
            return url
        await asyncio.sleep(0.01)
    raise AssertionError(f"no sign-in button; edits: {status.edit_text.await_args_list}")


def last_edit(status) -> str:
    return status.edit_text.await_args_list[-1].args[0]


def sign_in_task(user_id, name) -> asyncio.Task:
    return oauth_flow._tasks[(user_id, name)]


@pytest.fixture
async def endpoint(world, monkeypatch):
    """The real callback endpoint on 127.0.0.1 (an OS-chosen port), PUBLIC_BASE_URL https://gall.example.com."""
    settings = oauth_web.settings_from_env({"PUBLIC_BASE_URL": "https://gall.example.com", "OAUTH_LISTEN": "127.0.0.1:0"})
    server = await oauth_web.start(settings)
    oauth_web.set_bot_username("gall_test_bot")
    yield SimpleNamespace(server=server, base=f"http://127.0.0.1:{server.port}")
    await server.close()


async def browser_get(url: str):
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, allow_redirects=False) as response:
            return response.status, await response.text(), dict(response.headers)


def local(endpoint, redirect_target: str) -> str:
    """The provider's redirect to https://gall.example.com/... sent to the local endpoint instead."""
    parts = urlsplit(redirect_target)
    assert f"{parts.scheme}://{parts.netloc}" == "https://gall.example.com"
    return f"{endpoint.base}{parts.path}?{parts.query}"


# ---- the whole flow --------------------------------------------------------------------------------
async def test_add_signs_in_through_the_callback_endpoint_and_saves_the_tools(world, endpoint, caplog):
    caplog.set_level(logging.DEBUG)
    message, status = await mcp(f"add notion {MCP_URL}")
    message.delete.assert_not_awaited()  # no token in it
    assert status.edit_text.await_count == 0 or not last_edit(status).startswith("✅")
    url = await wait_for_button(status)
    text = status.edit_text.await_args_list[-1].args[0]
    assert text.startswith("🔐 *notion* needs you to sign in.") and oauth_flow.PUBLIC_HINT in text
    button = status.edit_text.await_args_list[-1].kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.text == "🔐 Sign in to notion" and button.url.startswith(f"{AUTH}/authorize?")
    saved = mcp_store.load_user(USER)[0]  # saved at once, so /mcp can offer "Sign in" after a timeout
    assert saved["auth"] == "oauth" and saved["error"] == mcp_store.SIGN_IN_NEEDED and saved["tools"] == []

    code_status, page, headers = await browser_get(local(endpoint, world.redirect_target(url)))
    assert code_status == 200 and "Signed in" in page and "tg://resolve?domain=gall_test_bot" in page
    assert headers["Cache-Control"] == "no-store" and headers["Referrer-Policy"] == "no-referrer"
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    assert last_edit(status) == "✅ *notion* connected (2 tools). I can use them from your next message on."
    assert status.edit_text.await_args_list[-1].kwargs.get("reply_markup") is None  # the button is gone

    saved = mcp_store.load_user(USER)[0]
    assert saved["auth"] == "oauth" and saved["error"] is None and [t["name"] for t in saved["tools"]] == ["search", "fetch"]
    assert world.registrations == 1  # the probe's registration was reused by the sign-in
    raw = mcp_store.user_path(USER).read_text()
    assert not any(marker in raw for marker in SECRET_MARKERS)  # encrypted at rest
    stored = mcp_store.read_oauth(USER, saved)
    assert stored["tokens"]["access_token"].endswith("SECRETACCESS") and stored["client"]["client_id"] == "client-1"
    assert stored["metadata"]["oauth"]["token_endpoint"] == f"{AUTH}/token"
    client = world.clients["client-1"]
    assert client["redirect_uris"] == ["https://gall.example.com/oauth/callback"] and client["token_endpoint_auth_method"] == "none"
    assert client["client_name"] == "GALL.AI" and client["grant_types"] == ["authorization_code", "refresh_token"]

    # the agent can call the tool right away, with the stored token
    provider = mt.MCPTools(USER)
    assert [t["name"] for t in provider.tools_schema] == ["mcp_notion_search", "mcp_notion_fetch"]
    assert await provider.execute_tool("mcp_notion_search", {"q": "x"}) == "search ok"
    assert world.calls == [("search", {"q": "x"})]
    logged = "\n".join(r.getMessage() for r in caplog.records)
    state = dict(p.split("=", 1) for p in urlsplit(url).query.split("&"))["state"]
    assert state not in logged and not any(marker in logged for marker in SECRET_MARKERS)


async def test_the_sign_in_expires_and_leaves_a_sign_in_button_in_the_list(world, endpoint, monkeypatch):
    monkeypatch.setattr(oauth_flow, "LOGIN_TIMEOUT", 0.3)
    _, status = await mcp(f"add notion {MCP_URL}")
    await wait_for_button(status)
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    assert last_edit(status) == oauth_flow.EXPIRED_TEXT == "⌛ Sign-in expired. Try again with /mcp."
    assert not oauth_flow.has_pending(USER) and oauth_flow._pending == {}
    message, _ = await mcp("")
    text, kw = message.answer.await_args_list[-1].args[0], message.answer.await_args_list[-1].kwargs
    assert "*notion* · http · ✅ · 0 tools" in text and "🔐 sign in needed" in text
    assert "⚠️ sign in needed" not in text  # said once, by the 🔐 state
    rows = [[b.callback_data for b in row] for row in kw["reply_markup"].inline_keyboard]
    assert rows == [["mcp_toggle_notion", "mcp_refresh_notion", "mcp_remove_notion"], ["mcp_signin_notion"], ["mcp_refreshall"]]


async def test_a_refused_sign_in_reaches_the_flow_and_the_page_says_so(world, endpoint):
    _, status = await mcp(f"add notion {MCP_URL}")
    url = await wait_for_button(status)
    state = dict(p.split("=", 1) for p in urlsplit(url).query.split("&"))["state"]
    code, page, _ = await browser_get(f"{endpoint.base}/oauth/callback?error=access_denied&error_description=User+said+no&state={state}")
    assert code == 200 and "Sign-in not completed" in page
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    assert last_edit(status) == "❌ Sign-in to *notion* was not completed: access\\_denied (User said no)"
    code, page, _ = await browser_get(f"{endpoint.base}/oauth/callback?code=x&state={state}")  # one-shot
    assert code == 400 and "expired" in page
    assert mcp_store.load_user(USER)[0]["error"] == mcp_store.SIGN_IN_NEEDED


async def test_a_token_error_is_reported_without_the_code_or_state(world, endpoint, caplog):
    caplog.set_level(logging.DEBUG)
    _, status = await mcp(f"add notion {MCP_URL}")
    url = await wait_for_button(status)
    target = world.redirect_target(url)
    world.codes.clear()  # the provider forgot the code: the token exchange fails
    await browser_get(local(endpoint, target))
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    text = last_edit(status)
    assert text.startswith("❌ Sign-in to *notion* failed: OAuthTokenError") and "invalid\\_grant" in text
    state = dict(p.split("=", 1) for p in urlsplit(url).query.split("&"))["state"]
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for secret in (state, "SECRETCODE"):
        assert secret not in text and secret not in logged
    assert "OAuth flow error (OAuthTokenError; details withheld)" in logged  # the SDK's own log line, cut down


async def test_copy_paste_fallback_without_a_public_endpoint(world, caplog):
    caplog.set_level(logging.DEBUG)
    _, status = await mcp(f"add notion {MCP_URL}")
    url = await wait_for_button(status)
    assert oauth_flow.FALLBACK_HINT in last_edit(status)
    assert world.clients["client-1"]["redirect_uris"] == [oauth_flow.LOOPBACK_REDIRECT]
    assert world.clients["client-1"]["application_type"] == "native"
    pasted = world.redirect_target(url)
    assert pasted.startswith("http://localhost:8765/oauth/callback?code=")
    paste, _ = make_message(pasted)
    assert await mcp_ui._pasted_while_pending(paste) is True
    other_chat, _ = make_message(pasted, chat_id=int(OTHER))
    assert await mcp_ui._pasted_while_pending(other_chat) is False  # another chat has nothing pending
    await mcp_ui.pasted_callback(paste, user_id=USER)
    paste.delete.assert_awaited_once()  # it held the code
    assert paste.answer.await_args_list[-1].args[0] == "🔐 Got it: finishing the sign-in to *notion*…"
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    assert last_edit(status).startswith("✅ *notion* connected (2 tools)")
    assert await mcp_ui._pasted_while_pending(paste) is False  # nothing pending any more: the agent gets such a text
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "SECRETCODE" not in logged


async def test_a_pasted_url_of_another_user_or_a_used_one_resolves_nothing(world):
    _, status = await mcp(f"add notion {MCP_URL}")
    url = await wait_for_button(status)
    pasted = world.redirect_target(url)
    stranger, _ = make_message(pasted, chat_id=int(OTHER))
    await mcp_ui.pasted_callback(stranger, user_id=OTHER)  # (the filter would not even let it in)
    assert stranger.answer.await_args_list[-1].args[0] == mcp_ui.PASTE_EXPIRED
    assert oauth_flow.has_pending(USER)  # still waiting for its own user
    mine, _ = make_message(pasted)
    await mcp_ui.pasted_callback(mine, user_id=USER)
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    again, _ = make_message(pasted)
    await mcp_ui.pasted_callback(again, user_id=USER)
    again.delete.assert_awaited_once()
    assert again.answer.await_args_list[-1].args[0] == mcp_ui.PASTE_EXPIRED


async def test_a_new_sign_in_replaces_the_pending_one(world):
    _, status = await mcp(f"add notion {MCP_URL}")
    first_url = await wait_for_button(status)
    first_task = sign_in_task(USER, "notion")
    list_message, second_status = make_message()
    await mcp_ui.mcp_button(SimpleNamespace(data="mcp_signin_notion", message=list_message, answer=AsyncMock()), user_id=USER)
    second_url = await wait_for_button(second_status)
    await asyncio.wait_for(asyncio.gather(first_task, return_exceptions=True), 5)
    assert last_edit(status) == oauth_flow.REPLACED_TEXT
    assert oauth_flow.resolve(dict(p.split("=", 1) for p in urlsplit(first_url).query.split("&"))["state"], code="x") is None
    paste, _ = make_message(world.redirect_target(second_url))
    await mcp_ui.pasted_callback(paste, user_id=USER)
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    assert last_edit(second_status).startswith("✅ *notion* connected (2 tools)")


# ---- tool calls: refresh and "sign in again" ------------------------------------------------------
async def signed_in(world, name="notion", user_id=USER):
    """Add and sign in through the copy-paste path; the stored record."""
    _, status = await mcp(f"add {name} {MCP_URL}", user_id=user_id)
    url = await wait_for_button(status)
    paste, _ = make_message(world.redirect_target(url), chat_id=int(user_id))
    await mcp_ui.pasted_callback(paste, user_id=user_id)
    await asyncio.wait_for(sign_in_task(user_id, name), 5)
    assert last_edit(status).startswith(f"✅ *{name}* connected")
    return mcp_store.find(mcp_store.load_user(user_id), name)


def expire_stored_token(user_id, record):
    stored = mcp_store.read_oauth(user_id, record)
    payload = {"tokens": stored["tokens"], "expires_at": 1000.0, "metadata": stored["metadata"]}
    mcp_store.write_oauth(user_id, record, tokens=payload)


async def test_an_expired_access_token_is_refreshed_at_the_stored_token_endpoint(world):
    record = await signed_in(world)
    old = mcp_store.read_oauth(USER, record)["tokens"]
    world.expire_all_access_tokens()
    expire_stored_token(USER, record)
    world.requests.clear()
    assert await mt.MCPTools(USER).execute_tool("mcp_notion_fetch", {}) == "fetch ok"
    assert world.refreshes == 1
    assert ("POST", f"{AUTH}/token") in world.requests and ("POST", "https://mcp.example.com/token") not in world.requests
    assert world.requests[0] == ("POST", f"{AUTH}/token")  # refreshed before the first request, no 401 round trip
    new = mcp_store.read_oauth(USER, record)
    assert new["tokens"]["access_token"] != old["access_token"] and new["tokens"]["refresh_token"] != old["refresh_token"]
    assert new["expires_at"] > 1000.0 and not new["needs_signin"]


async def test_a_refused_token_without_an_expiry_is_refreshed_once(world):
    world.expires_in = None  # the provider sends no expires_in: only the server's 401 tells
    record = await signed_in(world)
    world.expire_all_access_tokens()
    assert await mt.MCPTools(USER).execute_tool("mcp_notion_fetch", {}) == "fetch ok"
    assert world.refreshes == 1 and mcp_store.read_oauth(USER, record)["tokens"]["access_token"] in world.access


async def test_a_failed_refresh_asks_the_user_to_sign_in_again(world, caplog):
    record = await signed_in(world)
    world.expire_all_access_tokens()
    world.refresh_fails = True
    expire_stored_token(USER, record)
    result = await mt.MCPTools(USER).execute_tool("mcp_notion_fetch", {})
    assert result == "Error: MCP server notion needs you to sign in again (use /mcp)."
    stored = mcp_store.find(mcp_store.load_user(USER), "notion")
    assert stored["error"] == mcp_store.SIGN_IN_NEEDED and stored["oauth"]["needs_signin"] is True
    message, _ = await mcp("")
    kw = message.answer.await_args_list[-1].kwargs
    assert ["mcp_signin_notion"] in [[b.callback_data for b in row] for row in kw["reply_markup"].inline_keyboard]
    assert world.calls == []  # never reached the tool
    assert not any(m in r.getMessage() for r in caplog.records for m in SECRET_MARKERS)


async def test_no_stored_sign_in_answers_at_once_without_any_request(world):
    record = {**mcp_store.new_server("notion", MCP_URL), "auth": "oauth", "tools": [{"name": "fetch"}]}
    mcp_store.add_user_server(USER, record)
    assert await mt.MCPTools(USER).execute_tool("mcp_notion_fetch", {}) == oauth_flow.SIGN_IN_ERROR.format(name="notion")
    assert world.requests == []


async def test_concurrent_calls_refresh_once(world):
    record = await signed_in(world)
    world.expire_all_access_tokens()
    expire_stored_token(USER, record)
    provider = mt.MCPTools(USER)
    results = await asyncio.gather(*(provider.execute_tool("mcp_notion_fetch", {}) for _ in range(3)))
    assert results == ["fetch ok"] * 3
    assert world.refreshes == 1  # the others waited and used the refreshed token (the old refresh token is spent)


async def test_refresh_command_uses_the_stored_sign_in_and_never_starts_a_login(world):
    record = await signed_in(world)
    world.tools.append({"name": "create", "description": "Create a page", "inputSchema": {"type": "object"}})
    message, status = await mcp("refresh notion")
    assert last_edit(status) == "✅ *notion*: 3 tools"
    mcp_store.clear_oauth(USER, record)
    message, status = await mcp("refresh notion")
    assert last_edit(status) == "⚠️ *notion*: sign in needed"
    assert sign_in_url(status) is None and oauth_flow._pending == {}


# ---- sign in again, sign out, remove -----------------------------------------------------------------
async def test_sign_out_keeps_the_server_and_sign_in_again_brings_the_tokens_back(world):
    await signed_in(world)
    message, _ = await mcp("")
    text, kw = message.answer.await_args_list[-1].args[0], message.answer.await_args_list[-1].kwargs
    assert "🔐 signed in" in text
    rows = [[b.callback_data for b in row] for row in kw["reply_markup"].inline_keyboard]
    assert ["mcp_signout_notion"] in rows
    list_message, _ = make_message()
    await mcp_ui.mcp_button(SimpleNamespace(data="mcp_signout_notion", message=list_message, answer=AsyncMock()), user_id=USER)
    record = mcp_store.find(mcp_store.load_user(USER), "notion")
    assert record is not None and record["enabled"] is True and record["oauth"]["tokens"] is None
    assert mcp_store.read_oauth(USER, record)["tokens"] is None and mcp_store.read_oauth(USER, record)["client"]  # client kept
    redrawn = list_message.edit_text.await_args_list[-1]
    assert "🔐 sign in needed" in redrawn.args[0]
    assert ["mcp_signin_notion"] in [[b.callback_data for b in row] for row in redrawn.kwargs["reply_markup"].inline_keyboard]

    list_message, status = make_message()
    await mcp_ui.mcp_button(SimpleNamespace(data="mcp_signin_notion", message=list_message, answer=AsyncMock()), user_id=USER)
    url = await wait_for_button(status)
    paste, _ = make_message(world.redirect_target(url))
    await mcp_ui.pasted_callback(paste, user_id=USER)
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    assert last_edit(status).startswith("✅ *notion* connected (2 tools)")
    assert world.registrations == 1  # the kept registration was used again
    assert await mt.MCPTools(USER).execute_tool("mcp_notion_search", {}) == "search ok"


async def test_remove_deletes_the_sign_in_with_the_record(world):
    await signed_in(world)
    message, _ = make_message()
    await mcp_ui.mcp_button(SimpleNamespace(data="mcp_rm_notion", message=message, answer=AsyncMock()), user_id=USER)
    raw = json.loads(mcp_store.user_path(USER).read_text())
    assert raw["servers"] == [] and "oauth_admin" not in raw


async def test_re_adding_a_name_never_sends_the_old_servers_token_to_the_new_url(world, monkeypatch):
    await signed_in(world)
    old_tokens = set(world.access)
    seen = []
    original = world.handler

    def spy(request):
        seen.append((request.url.host, request.headers.get("authorization")))
        return original(request)

    monkeypatch.setattr(mt, "HTTP_TRANSPORT", httpx2.MockTransport(spy))
    await mcp("add notion https://other.example.com/mcp")
    assert all(not (auth or "").endswith(tuple(old_tokens)) for host, auth in seen if host == "other.example.com")
    record = mcp_store.find(mcp_store.load_user(USER), "notion")
    assert record["url"] == "https://other.example.com/mcp" and "oauth" not in record  # a plain server there


# ---- admin servers: everyone signs in with their own account --------------------------------------
def admin_file(tmp_path, *servers):
    path = tmp_path / "data" / "mcp_servers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"servers": list(servers)}))


async def test_admin_oauth_server_per_user_tokens_and_the_admins_sign_in_fills_the_cache(world, tmp_path):
    admin_file(tmp_path, {"name": "corp", "url": MCP_URL})
    await mcp_ui.refresh_admin_servers()  # no sign-in yet: the startup refresh records it
    (corp,) = mcp_store.load_admin()
    assert corp["auth"] == "oauth" and corp["error"] == mcp_store.SIGN_IN_NEEDED and corp["tools"] == []
    assert not (tmp_path / "data" / "admin").exists()

    for user_id in (ADMIN, USER):
        list_message, status = make_message(chat_id=int(user_id))
        await mcp_ui.mcp_button(SimpleNamespace(data="mcp_signin_corp", message=list_message, answer=AsyncMock()), user_id=user_id)
        url = await wait_for_button(status)
        paste, _ = make_message(world.redirect_target(url), chat_id=int(user_id))
        await mcp_ui.pasted_callback(paste, user_id=user_id)
        await asyncio.wait_for(sign_in_task(user_id, "corp"), 5)
        assert last_edit(status).startswith("✅ *corp* connected (2 tools)")
        raw = json.loads(mcp_store.user_path(user_id).read_text())
        assert raw["servers"] == [] and set(raw["oauth_admin"]) == {"corp"}
    (corp,) = mcp_store.load_admin()
    assert [t["name"] for t in corp["tools"]] == ["search", "fetch"] and corp["error"] is None  # filled by the admin

    provider = mt.MCPTools(USER)
    assert await provider.execute_tool("mcp_corp_search", {}) == "search ok"
    admin_token = mcp_store.read_oauth(ADMIN, corp)["tokens"]["access_token"]
    user_token = mcp_store.read_oauth(USER, corp)["tokens"]["access_token"]
    assert admin_token != user_token  # each with their own account
    message, _ = await mcp("")
    text, kw = message.answer.await_args_list[-1].args[0], message.answer.await_args_list[-1].kwargs
    assert "🏢 *corp*" in text and "🔐 signed in" in text
    assert [[b.callback_data for b in row] for row in kw["reply_markup"].inline_keyboard] == [["mcp_signout_corp"]]

    world.expire_all_access_tokens()
    world.refresh_fails = True
    expire_stored_token(USER, corp)
    assert (await provider.execute_tool("mcp_corp_search", {})).endswith("needs you to sign in again (use /mcp).")
    assert mcp_store.load_admin()[0]["error"] is None  # the shared cache is not the user's state
    assert json.loads(mcp_store.user_path(USER).read_text())["oauth_admin"]["corp"]["needs_signin"] is True
    nobody, _ = await mcp("", user_id=OTHER)
    assert "🔐 sign in needed" in nobody.answer.await_args_list[-1].args[0]


# ---- the request guard ----------------------------------------------------------------------------
async def test_oauth_endpoints_on_private_addresses_are_refused_for_users(world, monkeypatch):
    def resolve(host, port):
        if host == "auth.example.com":
            return ["10.0.0.7"]
        raise OSError("Name or service not known")

    monkeypatch.setattr(mcp_store, "_resolve", resolve)
    _, status = await mcp(f"add notion {MCP_URL}")
    text = last_edit(status)
    assert text.startswith("⚠️ Saved *notion*, but it could not be reached") and "Refused a request to auth.example.com" in text
    assert not any(host == f"{AUTH}/register" for _, host in world.requests)  # never sent
    assert sign_in_url(status) is None

    world.requests.clear()
    _, status = await mcp(f"add notion {MCP_URL}", user_id=ADMIN)  # the admin is trusted
    assert await wait_for_button(status)


async def test_plain_http_oauth_endpoints_are_refused_for_users(monkeypatch, world):
    fake = FakeOAuthServer(auth_host="http://auth.example.com")
    monkeypatch.setattr(mt, "HTTP_TRANSPORT", httpx2.MockTransport(fake.handler))
    _, status = await mcp(f"add notion {MCP_URL}")
    assert "Refused a plain-http request to auth.example.com" in last_edit(status)
    assert ("GET", "http://auth.example.com/.well-known/oauth-authorization-server") not in fake.requests


async def test_registration_refused_is_a_readable_error(world):
    world.registration_status = 400
    _, status = await mcp(f"add notion {MCP_URL}")
    text = last_edit(status)
    assert text.startswith("⚠️ Saved *notion*") and "OAuthRegistrationError" in text and "registration closed" in text


# ---- the key ----------------------------------------------------------------------------------------
TOKENS = {"tokens": {"access_token": "at-SECRETACCESS", "token_type": "Bearer", "refresh_token": "rt-SECRETREFRESH"},
          "expires_at": None, "metadata": None}
CLIENT = {"client_id": "client-9", "redirect_uris": [oauth_flow.LOOPBACK_REDIRECT]}


def stored_server(user_id=USER, name="notion", url=MCP_URL):
    record = {**mcp_store.new_server(name, url), "auth": "oauth"}
    mcp_store.add_user_server(user_id, record)
    return mcp_store.find(mcp_store.load_user(user_id), name)


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(mcp_store.ENV_ADMIN_FILE, str(tmp_path / "data" / "mcp_servers.json"))
    monkeypatch.delenv(mcp_store.ENV_TOKEN_KEY, raising=False)
    monkeypatch.delenv(mcp_store.ENV_TOKEN_KEY_OLD, raising=False)
    mcp_store.reset_cipher()
    yield tmp_path
    mcp_store.reset_cipher()


def test_the_env_key_encrypts_and_no_key_file_is_made(store, monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, key.decode())
    record = stored_server()
    mcp_store.write_oauth(USER, record, tokens=TOKENS, client=CLIENT)
    entry = mcp_store.find(mcp_store.load_user(USER), "notion")["oauth"]
    assert json.loads(Fernet(key).decrypt(entry["tokens"].encode()))["data"] == TOKENS
    assert not mcp_store.key_path().exists()
    assert mcp_store.read_oauth(USER, record)["tokens"] == TOKENS["tokens"]


def test_without_an_env_key_one_is_generated_once_private_and_reused(store, caplog):
    caplog.set_level(logging.WARNING)
    record = stored_server()
    mcp_store.write_oauth(USER, record, tokens=TOKENS)
    path = mcp_store.key_path()
    key = path.read_bytes().strip()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and Fernet(key)  # a valid Fernet key, owner-only
    assert not list(path.parent.glob(".mcp_token_key.*"))  # the temp file is gone
    warning = [r.getMessage() for r in caplog.records if "MCP_TOKEN_KEY is not set" in r.getMessage()]
    assert len(warning) == 1 and key.decode() not in warning[0] and "generated" in warning[0]
    mcp_store.reset_cipher()  # a restart: the same key, the same sign-in
    assert mcp_store.read_oauth(USER, record)["tokens"] == TOKENS["tokens"]
    assert path.read_bytes().strip() == key


def test_an_existing_key_file_is_never_overwritten(store, monkeypatch):
    path = mcp_store.key_path()
    path.parent.mkdir(parents=True)
    existing = Fernet.generate_key()
    path.write_bytes(existing + b"\n")
    record = stored_server()
    mcp_store.write_oauth(USER, record, tokens=TOKENS)
    assert path.read_bytes() == existing + b"\n"
    entry = mcp_store.find(mcp_store.load_user(USER), "notion")["oauth"]
    assert Fernet(existing).decrypt(entry["tokens"].encode())
    # the race: another start linked its key first; ours is dropped and theirs wins
    path.unlink()
    monkeypatch.setattr(os, "link", lambda src, dst: (_ for _ in ()).throw(FileExistsError()))
    path.write_bytes(existing)
    assert mcp_store._create_key_file() == existing.decode()


def test_rotation_with_the_old_keys_and_reencryption_on_the_next_write(store, monkeypatch):
    old, new = Fernet.generate_key(), Fernet.generate_key()
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, old.decode())
    record = stored_server()
    mcp_store.write_oauth(USER, record, tokens=TOKENS, client=CLIENT)
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, new.decode())
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY_OLD, f"not-a-key, {old.decode()}")
    mcp_store.reset_cipher()
    assert mcp_store.read_oauth(USER, record)["tokens"] == TOKENS["tokens"]  # the old key still decrypts
    mcp_store.write_oauth(USER, record, needs_signin=False)  # any write moves both values to the new key
    entry = mcp_store.find(mcp_store.load_user(USER), "notion")["oauth"]
    for field in ("tokens", "client"):
        assert Fernet(new).decrypt(entry[field].encode())


def test_a_key_file_from_before_the_env_key_still_decrypts(store, monkeypatch):
    record = stored_server()
    mcp_store.write_oauth(USER, record, tokens=TOKENS)  # sealed with a generated key file
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, Fernet.generate_key().decode())
    mcp_store.reset_cipher()
    assert mcp_store.read_oauth(USER, record)["tokens"] == TOKENS["tokens"]


def test_an_invalid_env_key_is_logged_and_the_key_file_used(store, monkeypatch, caplog):
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, "too-short")
    record = stored_server()
    mcp_store.write_oauth(USER, record, tokens=TOKENS)
    assert any("MCP_TOKEN_KEY is not a valid Fernet key" in r.getMessage() for r in caplog.records)
    assert "too-short" not in caplog.text.replace("MCP_TOKEN_KEY is not a valid", "")
    assert mcp_store.key_path().exists() and mcp_store.read_oauth(USER, record)["tokens"] == TOKENS["tokens"]


def test_an_undecryptable_sign_in_reads_as_signed_out(store, monkeypatch, caplog):
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, Fernet.generate_key().decode())
    record = stored_server()
    mcp_store.write_oauth(USER, record, tokens=TOKENS, client=CLIENT)
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, Fernet.generate_key().decode())  # the key was lost
    mcp_store.reset_cipher()
    stored = mcp_store.read_oauth(USER, record)
    assert stored["tokens"] is None and stored["client"] is None
    (shown,) = mcp_store.all_servers(USER, with_state=True)
    assert shown["oauth_state"] == mcp_store.STATE_SIGN_IN_NEEDED and mcp_store.describe_headers(shown) == "🔐 sign in needed"
    mcp_store.read_oauth(USER, record)
    lines = [r.getMessage() for r in caplog.records if "cannot be decrypted" in r.getMessage()]
    assert len(lines) == 2 and "(tokens)" in lines[0] and "(client)" in lines[1]  # once per value and process


# ---- the storage ----------------------------------------------------------------------------------
def test_storage_per_user_and_server_and_the_admin_servers_in_the_users_file(store, monkeypatch):
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, Fernet.generate_key().decode())
    a, b = stored_server(name="a"), stored_server(name="b", url="https://b.example.com/mcp")
    other = stored_server(user_id=OTHER, name="a")
    corp = {**mcp_store.normalize({"name": "corp", "url": "https://corp.example.com/mcp", "auth": "oauth"}, admin=True)}
    for user_id, server, token in ((USER, a, "A"), (USER, b, "B"), (OTHER, other, "O"), (USER, corp, "C")):
        mcp_store.write_oauth(user_id, server, tokens={**TOKENS, "tokens": {"access_token": token}})
    assert [mcp_store.read_oauth(u, s)["tokens"]["access_token"] for u, s in ((USER, a), (USER, b), (OTHER, other), (USER, corp))] == ["A", "B", "O", "C"]
    assert mcp_store.read_oauth(OTHER, corp)["tokens"] is None
    raw = json.loads(mcp_store.user_path(USER).read_text())
    assert set(raw["oauth_admin"]) == {"corp"} and raw["servers"][0]["oauth"]["tokens"]
    # every other change of the file keeps the admin-server sign-ins
    mcp_store.toggle_user_server(USER, "a")
    mcp_store.update_user_tools(USER, "b", [{"name": "t"}])
    mcp_store.record_error(USER, a, "boom")
    mcp_store.add_user_server(USER, mcp_store.new_server("c", "https://c.example.com/mcp"))
    mcp_store.remove_user_server(USER, "c")
    mcp_store.save_user(USER, mcp_store.load_user(USER))
    assert mcp_store.read_oauth(USER, corp)["tokens"]["access_token"] == "C"
    assert mcp_store.read_oauth(USER, a)["tokens"]["access_token"] == "A"
    assert mcp_store.clear_oauth(USER, corp, keep_client=False) is True
    assert "oauth_admin" not in json.loads(mcp_store.user_path(USER).read_text())


def test_a_sign_in_moved_to_another_record_url_or_user_reads_as_signed_out(store, monkeypatch):
    """mcp.json is in the user's sandbox workspace: copying an encrypted token to a record the sandbox
    controls must not make the bot send it there."""
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, Fernet.generate_key().decode())
    good = stored_server(name="notion")
    evil = stored_server(name="evil", url="https://evil.example.com/mcp")
    mcp_store.write_oauth(USER, good, tokens=TOKENS, client=CLIENT)
    path = mcp_store.user_path(USER)
    raw = json.loads(path.read_text())
    raw["servers"][1]["oauth"] = raw["servers"][0]["oauth"]  # moved to another server
    path.write_text(json.dumps(raw))
    assert mcp_store.read_oauth(USER, evil)["tokens"] is None
    raw["servers"][0]["url"] = "https://evil.example.com/mcp"  # or the URL changed under it
    path.write_text(json.dumps(raw))
    assert mcp_store.read_oauth(USER, mcp_store.find(mcp_store.load_user(USER), "notion"))["tokens"] is None
    raw["servers"][0]["url"] = MCP_URL
    other = mcp_store.user_path(OTHER)
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps(raw))  # or copied into another user's file
    assert mcp_store.read_oauth(OTHER, good)["tokens"] is None
    path.write_text(json.dumps(raw))
    assert mcp_store.read_oauth(USER, good)["tokens"] == TOKENS["tokens"]  # in place it still works


def test_tampered_records_never_raise(store, monkeypatch):
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, Fernet.generate_key().decode())
    path = mcp_store.user_path(USER)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "servers": [
            {"name": "a", "url": MCP_URL, "auth": "oauth", "oauth": {"tokens": 42, "client": ["x"], "account": {"x": 1}}},
            {"name": "b", "url": MCP_URL, "auth": "oauth", "oauth": {"tokens": "not-fernet", "client": "gAAAAA-garbage"}},
            {"name": "c", "url": MCP_URL, "auth": "oauth", "oauth": "nope"},
        ],
        "oauth_admin": {"Bad Name": {"tokens": "x"}, "corp": [1, 2], "ok": {"tokens": "zzz", "needs_signin": "yes"}},
    }))
    servers = mcp_store.all_servers(USER, with_state=True)
    assert [s["oauth_state"] for s in servers] == [mcp_store.STATE_SIGN_IN_NEEDED] * 3
    for server in servers:
        assert mcp_store.read_oauth(USER, server)["tokens"] is None
    assert mcp_store.read_oauth(USER, {"name": "ok", "url": MCP_URL, "admin": True})["needs_signin"] is False
    path.write_text("[not json")
    assert mcp_store.all_servers(USER, with_state=True) == [] and mcp_store.read_oauth(USER, {"name": "a", "url": MCP_URL})["tokens"] is None


def test_the_users_file_is_never_read_or_written_through_a_link(store, tmp_path):
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"servers": [{"name": "stolen", "url": MCP_URL, "headers": {"Authorization": "Bearer x"}}]}))
    path = mcp_store.user_path(USER)
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    assert mcp_store.load_user(USER) == []  # another user's file behind a link is not read
    mcp_store.add_user_server(USER, mcp_store.new_server("mine", MCP_URL))
    assert json.loads(target.read_text())["servers"][0]["name"] == "stolen"  # the link's target is untouched
    assert not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [s["name"] for s in mcp_store.load_user(USER)] == ["mine"]


async def test_storage_hands_the_sdk_an_expired_token_with_an_empty_access_token(store, monkeypatch):
    monkeypatch.setenv(mcp_store.ENV_TOKEN_KEY, Fernet.generate_key().decode())
    oauth_flow.reset()
    record = stored_server()
    mcp_store.write_oauth(USER, record, tokens={**TOKENS, "expires_at": 1000.0}, client=CLIENT)
    storage = oauth_flow.OAuthStorage(USER, record, redirect="https://gall.example.com/oauth/callback")
    token = await storage.get_tokens()
    assert token.access_token == "" and token.refresh_token == "rt-SECRETREFRESH"  # the SDK refreshes first
    assert storage._held is not None  # a refresh is coming: one at a time
    storage.close()
    assert (await storage.get_client_info()).client_id == "client-9"  # refreshes keep the registration
    signing_in = oauth_flow.OAuthStorage(USER, record, redirect="https://gall.example.com/oauth/callback", for_sign_in=True)
    assert await signing_in.get_client_info() is None  # registered for another redirect URI: register again
    mcp_store.write_oauth(USER, record, tokens={**TOKENS, "expires_at": 4e9})
    fresh = oauth_flow.OAuthStorage(USER, record)
    assert (await fresh.get_tokens()).access_token == "at-SECRETACCESS" and fresh._held is None


async def test_a_users_sign_in_to_a_shared_server_before_the_admins_says_when_the_tools_come(world, tmp_path):
    admin_file(tmp_path, {"name": "corp", "url": MCP_URL, "auth": "oauth"})
    list_message, status = make_message()
    await mcp_ui.mcp_button(SimpleNamespace(data="mcp_signin_corp", message=list_message, answer=AsyncMock()), user_id=USER)
    url = await wait_for_button(status)
    paste, _ = make_message(world.redirect_target(url))
    await mcp_ui.pasted_callback(paste, user_id=USER)
    await asyncio.wait_for(sign_in_task(USER, "corp"), 5)
    assert last_edit(status).endswith(oauth_flow.ADMIN_CACHE_NOTE)
    assert mcp_store.load_admin()[0]["tools"] == []  # a user's discovery never writes the shared cache


async def test_shutdown_cancels_waiting_sign_ins_quietly(world):
    _, status = await mcp(f"add notion {MCP_URL}")
    await wait_for_button(status)
    edits, task = status.edit_text.await_count, sign_in_task(USER, "notion")
    await oauth_flow.shutdown()
    assert task.cancelled() and oauth_flow._pending == {}
    assert status.edit_text.await_count == edits  # nothing announced


async def test_the_sse_transport_gets_the_provider_and_the_guard(world, monkeypatch):
    """sse_client takes ``auth`` and builds its client through our factory (the request hooks)."""
    import mcp.client.sse as sse

    seen = {}

    class Stop(Exception):
        pass

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_sse_client(url, headers=None, timeout=5.0, auth=None, httpx_client_factory=None, **kw):
        client = httpx_client_factory(headers=headers, auth=auth, timeout=httpx2.Timeout(timeout))
        seen.update(url=url, auth=auth, hooks=client.event_hooks["request"])
        await client.aclose()
        raise Stop()
        yield  # noqa: unreachable

    monkeypatch.setattr(sse, "sse_client", fake_sse_client)
    record = {**mcp_store.new_server("legacy", MCP_URL, transport="sse"), "auth": "oauth", "tools": [{"name": "t"}]}
    mcp_store.add_user_server(USER, record)
    mcp_store.write_oauth(USER, mcp_store.load_user(USER)[0], tokens={"tokens": {"access_token": "a", "token_type": "Bearer"},
                                                                       "expires_at": None, "metadata": None})
    result = await mt.MCPTools(USER).execute_tool("mcp_legacy_t", {})
    assert result.startswith("Error: MCP server legacy unreachable: Stop")
    from mcp.client.auth import OAuthClientProvider

    assert isinstance(seen["auth"], OAuthClientProvider) and len(seen["hooks"]) == 1


async def test_signing_in_again_after_a_failed_refresh_never_holds_the_refresh_lock_while_waiting(world):
    record = await signed_in(world)
    world.expire_all_access_tokens()
    world.refresh_fails = True
    expire_stored_token(USER, record)
    list_message, status = make_message()
    await mcp_ui.mcp_button(SimpleNamespace(data="mcp_signin_notion", message=list_message, answer=AsyncMock()), user_id=USER)
    url = await wait_for_button(status)  # the refresh failed, the user is asked
    assert world.refreshes == 1
    assert not oauth_flow._refresh_locks[(USER, "notion", False)].locked()  # tool calls meanwhile do not wait 10 minutes
    call = await asyncio.wait_for(mt.MCPTools(USER).execute_tool("mcp_notion_fetch", {}), 5)
    assert call.endswith("needs you to sign in again (use /mcp).")
    paste, _ = make_message(world.redirect_target(url))
    await mcp_ui.pasted_callback(paste, user_id=USER)
    await asyncio.wait_for(sign_in_task(USER, "notion"), 5)
    assert last_edit(status).startswith("✅ *notion* connected (2 tools)")
    assert await mt.MCPTools(USER).execute_tool("mcp_notion_fetch", {}) == "fetch ok"
    stored = mcp_store.find(mcp_store.load_user(USER), "notion")
    assert stored["error"] is None and "needs_signin" not in stored["oauth"]
