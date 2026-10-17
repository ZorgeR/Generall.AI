"""/mcp and the mcp_ buttons (F12): bot/handlers/mcp_ui.py with a fake Message/CallbackQuery and a fake discovery."""
import asyncio
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandObject
from aiogram.methods import AnswerCallbackQuery
from aiogram.types import Chat, Message

from bot import mcp_store
from bot.handlers import mcp_ui

USER = "42"
ADMIN = "1"
URL = "https://api.example.com/mcp"


class FakeAuth:
    def is_admin(self, user_id):
        return str(user_id) == ADMIN


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Files under tmp_path, the admin file beside them, a recording fake discovery; returns its recorder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(mcp_store.ENV_ADMIN_FILE, str(tmp_path / "data" / "mcp_servers.json"))
    monkeypatch.setattr(mcp_ui, "auth", FakeAuth())
    state = SimpleNamespace(calls=[], tools={"default": [{"name": "search", "description": "d", "input_schema": {"type": "object", "properties": {}}}]},
                            error=None)

    async def fake_discover(user_id, server):
        state.calls.append((user_id, server["name"], server["url"], dict(server.get("headers") or {}), server["transport"]))
        if state.error:
            raise RuntimeError(state.error)
        return list(state.tools.get(server["name"], state.tools["default"]))

    monkeypatch.setattr(mcp_ui, "discover_tools", fake_discover)
    return state


def make_message(text="/mcp"):
    msg = Message(message_id=7, date=datetime.now(timezone.utc), chat=Chat(id=int(USER), type="private"), text=text)
    object.__setattr__(msg, "answer", AsyncMock(return_value=msg))
    object.__setattr__(msg, "edit_text", AsyncMock(return_value=msg))
    object.__setattr__(msg, "delete", AsyncMock(return_value=True))
    return msg


def make_callback(data, message=None, *, answer=None):
    return SimpleNamespace(data=data, message=message if message is not None else make_message(), answer=answer or AsyncMock())


def command(args: str):
    return CommandObject(prefix="/", command="mcp", args=args or None)


async def mcp(text: str, user_id=USER, message=None):
    message = message or make_message(f"/mcp {text}".strip())
    await mcp_ui.mcp_command(message, command(text), user_id=user_id)
    return message


def answered(message, index=-1):
    call = message.answer.await_args_list[index]
    return call.args[0], call.kwargs


def edited(message, index=-1):
    call = message.edit_text.await_args_list[index]
    return call.args[0], call.kwargs


def buttons(markup):
    return [[b.callback_data for b in row] for row in markup.inline_keyboard] if markup else []


def admin_file(tmp_path, *servers):
    path = tmp_path / "data" / "mcp_servers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"servers": list(servers)}))
    return path


# ---- /mcp add -------------------------------------------------------------------------------
async def test_add_validates_discovers_saves_and_hides_the_token(wired):
    message = await mcp(f"add GitHub {URL} ghp_secret")
    assert wired.calls == [(USER, "github", URL, {"Authorization": "Bearer ghp_secret"}, "http")]
    message.delete.assert_awaited_once()  # the token must not stay in the chat
    text, kw = edited(message)
    assert text.startswith("✅ Connected *github*: 1 tool") and kw["parse_mode"] == "Markdown"
    assert "ghp_secret" not in text and "ghp_secret" not in answered(message)[0]
    saved = mcp_store.load_user(USER)
    assert saved[0]["name"] == "github" and saved[0]["headers"] == {"Authorization": "Bearer ghp_secret"}
    assert saved[0]["tools"][0]["name"] == "search" and saved[0]["updated"] and saved[0]["error"] is None


async def test_add_without_a_token_keeps_the_message_and_accepts_the_sse_flag(wired):
    message = await mcp(f"add legacy {URL} sse")
    message.delete.assert_not_awaited()
    assert wired.calls[0][3] == {} and wired.calls[0][4] == "sse"
    assert mcp_store.load_user(USER)[0]["transport"] == "sse"
    await mcp(f"add other {URL} tok sse")  # token and flag in either order
    await mcp(f"add third {URL} sse tok")
    assert [c[3:] for c in wired.calls[1:]] == [({"Authorization": "Bearer tok"}, "sse")] * 2


async def test_add_rejects_bad_names_urls_and_missing_arguments(wired):
    for args, expected in ((f"add Bad.Name {URL}", "name must be"), ("add x http://example.com/mcp", "Only https"),
                           ("add x", "Usage"), ("add", "Usage")):
        message = await mcp(args)
        text, _ = answered(message)
        assert text.startswith("❌") and expected in text, (args, text)
        message.delete.assert_not_awaited()
    assert wired.calls == [] and mcp_store.load_user(USER) == []


async def test_add_deletes_a_message_with_a_token_even_when_validation_rejects_it(wired):
    for args, expected in ((f"add Bad.Name {URL} ghp_secret", "name must be"), ("add x http://example.com/mcp ghp_secret", "Only https")):
        message = await mcp(args)
        message.delete.assert_awaited_once()  # the token must not stay in the chat, rejected or not
        text, _ = answered(message)
        assert text.startswith("❌") and expected in text and "ghp_secret" not in text, (args, text)
    assert wired.calls == [] and mcp_store.load_user(USER) == []


async def test_added_text_keeps_raw_names_inside_code_spans(wired):
    wired.error = "connection refused"
    message = await mcp(f"add my_srv {URL}")
    text, _ = edited(message)
    assert text.startswith("⚠️ Saved *my\\_srv*")  # escaped in the bold entity
    assert "`/mcp add my_srv <url> [token]`" in text and "`/mcp refresh my_srv`" in text  # raw in code spans
    assert "my\\_srv" not in text.replace("*my\\_srv*", "")  # the escaped spelling only inside the bold entity
    wired.error = None
    text, _ = edited(await mcp(f"add my_srv {URL}"))
    assert text.startswith("✅ Connected *my\\_srv*: 1 tool")


async def test_admin_may_add_plain_http_but_nobody_takes_a_shared_name(wired, tmp_path):
    message = await mcp("add intern http://intranet.corp/mcp", user_id=ADMIN)
    assert edited(message)[0].startswith("✅") and mcp_store.load_user(ADMIN)[0]["url"] == "http://intranet.corp/mcp"
    admin_file(tmp_path, {"name": "corp", "url": URL})
    message = await mcp(f"add corp {URL}")
    assert "shared server" in answered(message)[0] and mcp_store.load_user(USER) == []


async def test_add_reports_a_discovery_error_but_saves_the_server(wired):
    wired.error = "401 Unauthorized"
    message = await mcp(f"add github {URL} badtoken")
    text, _ = edited(message)
    assert text.startswith("⚠️ Saved *github*") and "401 Unauthorized" in text and "/mcp refresh github" in text
    saved = mcp_store.load_user(USER)[0]
    assert saved["error"] == "401 Unauthorized" and saved["tools"] == []
    wired.error = None
    await mcp(f"add github {URL} goodtoken")  # re-adding replaces the record
    saved = mcp_store.load_user(USER)
    assert len(saved) == 1 and saved[0]["error"] is None and saved[0]["headers"] == {"Authorization": "Bearer goodtoken"}


# ---- /mcp list, help, remove, refresh -----------------------------------------------------------
async def test_list_shows_user_and_admin_servers_with_buttons_for_the_users_only(wired, tmp_path):
    mcp_store.add_user_server(USER, mcp_store.with_tools(mcp_store.new_server("github", URL, "tok"), [{"name": "a"}, {"name": "b"}]))
    mcp_store.add_user_server(USER, mcp_store.with_tools(mcp_store.new_server("broken", "https://b.example.com/x?key=sec"), None, "boom_err"))
    mcp_store.toggle_user_server(USER, "broken")
    admin_file(tmp_path, {"name": "corp", "url": "https://corp.example.com/mcp"})
    mcp_store.save_admin_cache([{"name": "corp", "tools": [{"name": "t"}], "updated": "x", "error": None}])

    message = await mcp("")
    text, kw = answered(message)
    assert text.startswith(mcp_ui.HEADER) and kw["parse_mode"] == "Markdown"
    assert "*github* · http · ✅ · 2 tools" in text and "token set" in text and "tok" not in text.replace("token", "")
    assert "*broken* · http · ⬜ · 0 tools" in text and "https://b.example.com/x" in text and "key=sec" not in text
    assert "⚠️ boom\\_err" in text  # escaped for legacy Markdown
    assert "🏢 *corp* · http · ✅ · 1 tool · read-only" in text
    assert buttons(kw["reply_markup"]) == [
        ["mcp_toggle_github", "mcp_refresh_github", "mcp_remove_github"],
        ["mcp_toggle_broken", "mcp_refresh_broken", "mcp_remove_broken"],
        ["mcp_refreshall"],
    ]
    labels = [b.text for row in kw["reply_markup"].inline_keyboard for b in row]
    assert labels[0] == "✅ github" and labels[3] == "⬜ broken"


async def test_empty_list_and_help_and_unknown_subcommand(wired):
    text, kw = answered(await mcp(""))
    assert mcp_ui.EMPTY in text and kw["reply_markup"] is None
    text, _ = answered(await mcp("help"))
    assert text == mcp_ui.HELP and "/mcp add <name> <url> [token] [sse]" in text
    text, _ = answered(await mcp("fro`b*nicate"))  # user text: escaped, never inside a code span
    assert text.startswith('Unknown subcommand "fro\\`b\\*nicate".') and mcp_ui.HELP in text


def test_help_is_balanced_legacy_markdown():
    """Telegram rejects an odd number of _ or * outside code spans ("can't find end of the entity") and
    answer_md would then strip every mark; HELP goes out as written, so its marks must pair up."""
    outside_code = re.sub(r"`[^`\n]*`", "", mcp_ui.HELP)
    assert "`" not in outside_code
    assert outside_code.count("_") % 2 == 0 and outside_code.count("*") % 2 == 0
    assert "[" not in outside_code  # a bare [ starts a link entity in legacy Markdown


async def test_remove_command(wired, tmp_path):
    mcp_store.add_user_server(USER, mcp_store.new_server("github", URL))
    admin_file(tmp_path, {"name": "corp", "url": URL})
    assert answered(await mcp("remove GITHUB"))[0] == "🗑 Removed *github*."
    assert mcp_store.load_user(USER) == []
    assert "No server named *github*" in answered(await mcp("remove github"))[0]
    assert "No server named *corp*" in answered(await mcp("remove corp"))[0]  # shared servers stay
    assert "Usage" in answered(await mcp("remove"))[0]


async def test_refresh_command_updates_user_servers_and_admin_ones_only_for_the_admin(wired, tmp_path):
    mcp_store.add_user_server(USER, mcp_store.new_server("a", URL))
    mcp_store.add_user_server(USER, mcp_store.new_server("b", URL))
    admin_file(tmp_path, {"name": "corp", "url": URL})
    wired.tools["b"] = [{"name": "x"}, {"name": "y"}]
    message = await mcp("refresh")
    assert sorted(c[1] for c in wired.calls) == ["a", "b"]  # not corp: a plain user cannot refresh shared servers
    text, _ = edited(message)
    assert "✅ *a*: 1 tool" in text and "✅ *b*: 2 tools" in text
    assert len(mcp_store.load_user(USER)[1]["tools"]) == 2

    wired.calls.clear()
    message = await mcp("refresh b")
    assert [c[1] for c in wired.calls] == ["b"] and edited(message)[0] == "✅ *b*: 2 tools"
    assert "No server named *zzz*" in answered(await mcp("refresh zzz"))[0]

    wired.calls.clear()
    wired.tools["corp"] = [{"name": "lookup"}]
    message = await mcp("refresh corp", user_id=ADMIN)
    assert [c[1] for c in wired.calls] == ["corp"] and edited(message)[0] == "✅ *corp*: 1 tool"
    assert mcp_store.load_admin()[0]["tools"][0]["name"] == "lookup"
    assert json.loads((tmp_path / "data" / "mcp_servers.cache.json").read_text())["servers"][0]["name"] == "corp"

    wired.error = "connection refused"
    message = await mcp("refresh a")
    assert edited(message)[0] == "⚠️ *a*: connection refused" and mcp_store.load_user(USER)[0]["error"] == "connection refused"


# ---- buttons -------------------------------------------------------------------------------
async def test_toggle_button_flips_the_server_and_redraws_the_list(wired):
    mcp_store.add_user_server(USER, mcp_store.new_server("github", URL))
    message = make_message()
    callback = make_callback("mcp_toggle_github", message)
    await mcp_ui.mcp_button(callback, user_id=USER)
    callback.answer.assert_awaited_once()
    assert mcp_store.load_user(USER)[0]["enabled"] is False
    text, kw = edited(message)
    assert "*github* · http · ⬜" in text and buttons(kw["reply_markup"])[0][0] == "mcp_toggle_github"
    assert message.answer.await_count == 0
    await mcp_ui.mcp_button(make_callback("mcp_toggle_github", message), user_id=USER)
    assert mcp_store.load_user(USER)[0]["enabled"] is True


async def test_remove_button_asks_first_then_removes_or_goes_back(wired):
    mcp_store.add_user_server(USER, mcp_store.new_server("github", URL))
    message = make_message()
    await mcp_ui.mcp_button(make_callback("mcp_remove_github", message), user_id=USER)
    text, kw = edited(message)
    assert "Remove *github*" in text and buttons(kw["reply_markup"]) == [["mcp_rm_github", "mcp_back"]]
    assert mcp_store.load_user(USER) != []
    await mcp_ui.mcp_button(make_callback("mcp_back", message), user_id=USER)
    assert edited(message)[0].startswith(mcp_ui.HEADER) and "*github*" in edited(message)[0]
    await mcp_ui.mcp_button(make_callback("mcp_rm_github", message), user_id=USER)
    assert mcp_store.load_user(USER) == [] and mcp_ui.EMPTY in edited(message)[0]


async def test_refresh_buttons_rediscover_one_or_all(wired, tmp_path):
    mcp_store.add_user_server(USER, mcp_store.new_server("a", URL))
    mcp_store.add_user_server(USER, mcp_store.new_server("b", URL))
    admin_file(tmp_path, {"name": "corp", "url": URL})
    message = make_message()
    await mcp_ui.mcp_button(make_callback("mcp_refresh_a", message), user_id=USER)
    assert [c[1] for c in wired.calls] == ["a"]
    assert edited(message, 0)[0] == "🔄 Refreshing *a*…" and edited(message)[0].startswith(mcp_ui.HEADER)
    assert "*a* · http · ✅ · 1 tool" in edited(message)[0]
    wired.calls.clear()
    await mcp_ui.mcp_button(make_callback("mcp_refreshall", message), user_id=USER)
    assert sorted(c[1] for c in wired.calls) == ["a", "b"]  # never the shared ones
    assert edited(message)[0].startswith(mcp_ui.HEADER)
    wired.calls.clear()
    await mcp_ui.mcp_button(make_callback("mcp_refresh_corp", message), user_id=USER)  # not the user's: just redraw
    assert wired.calls == [] and edited(message)[0].startswith(mcp_ui.HEADER)


async def test_buttons_tolerate_old_queries_unknown_data_and_missing_messages(wired, caplog):
    too_old = TelegramBadRequest(method=AnswerCallbackQuery(callback_query_id="1"), message="Bad Request: query is too old")
    message = make_message()
    await mcp_ui.mcp_button(make_callback("mcp_back", message, answer=AsyncMock(side_effect=too_old)), user_id=USER)
    message.edit_text.assert_awaited_once()  # the action still happened
    await mcp_ui.mcp_button(make_callback("mcp_back", SimpleNamespace(chat=None)), user_id=USER)
    message2 = make_message()
    await mcp_ui.mcp_button(make_callback("mcp_bogus_x", message2), user_id=USER)
    await mcp_ui.mcp_button(make_callback("mcp_toggle_", message2), user_id=USER)
    assert message2.edit_text.await_count == 0


async def test_toggle_of_an_unknown_server_just_redraws(wired):
    message = make_message()
    await mcp_ui.mcp_button(make_callback("mcp_toggle_ghost", message), user_id=USER)
    assert mcp_ui.EMPTY in edited(message)[0]


# ---- the startup refresh of the admin servers ------------------------------------------------------
async def test_refresh_admin_servers_fills_the_cache_and_keeps_disabled_entries(wired, tmp_path, monkeypatch):
    admin_file(tmp_path, {"name": "corp", "url": URL}, {"name": "off", "url": URL, "enabled": False}, {"name": "down", "url": URL})
    wired.tools["corp"] = [{"name": "lookup"}, {"name": "search"}]

    async def flaky(user_id, server):
        if server["name"] == "down":
            raise RuntimeError("no route to host")
        return wired.tools[server["name"]]

    monkeypatch.setattr(mcp_ui, "discover_tools", flaky)
    await mcp_ui.refresh_admin_servers()
    servers = {s["name"]: s for s in mcp_store.load_admin()}
    assert len(servers["corp"]["tools"]) == 2 and servers["corp"]["error"] is None
    assert servers["down"]["tools"] == [] and servers["down"]["error"] == "no route to host"
    assert servers["off"]["tools"] == [] and servers["off"]["updated"] is None  # disabled: not contacted


async def test_refresh_admin_servers_is_a_no_op_without_a_file_and_never_raises(wired, monkeypatch, caplog):
    await mcp_ui.refresh_admin_servers()
    assert not mcp_store.admin_cache_path().exists()

    def boom():
        raise OSError("disk on fire")

    monkeypatch.setattr(mcp_store, "load_admin", boom)
    await mcp_ui.refresh_admin_servers()
    assert any("Refreshing the MCP admin servers failed" in r.getMessage() for r in caplog.records)


def test_module_imports_nothing_heavy_at_import_time():
    """bot/ never imports agents or stats at module level; the mcp client is reached lazily through agents."""
    import ast

    tree = ast.parse(open(mcp_ui.__file__, encoding="utf-8").read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    modules = [a.name for n in top if isinstance(n, ast.Import) for a in n.names] + [n.module for n in top if isinstance(n, ast.ImportFrom)]
    assert not any(m == "stats" or m.startswith("agents") or m.startswith("mcp") for m in modules), modules
