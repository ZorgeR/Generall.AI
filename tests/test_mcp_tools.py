"""MCP client (F12): agents/mcp_tools.py over fake transports and an in-memory server, bot/mcp_store.py."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import agents.mcp_tools as mt
from bot import mcp_store

USER = "42"
REAL_RESOLVE = mcp_store._resolve


@pytest.fixture(autouse=True)
def no_dns(monkeypatch):
    """The connect-time host check resolves the server's name: here no name resolves (no network in tests)."""

    def unresolvable(host, port):
        raise OSError("Name or service not known")

    monkeypatch.setattr(mcp_store, "_resolve", unresolvable)


def server(name="demo", *, tools=None, enabled=True, admin=False, **over):
    record = {"name": name, "url": f"https://{name}.example.com/mcp", "transport": "http", "headers": {},
              "enabled": enabled, "tools": tools or [], "updated": None, "error": None}
    if admin:
        record["admin"] = True
    record.update(over)
    return record


# ---- fakes --------------------------------------------------------------------------------
class FakeSession:
    """Stands in for mcp.ClientSession: records the calls, answers from ``tools`` / ``results``."""

    def __init__(self, tools=(), results=None, *, fail_on=None):
        self.tools = list(tools)
        self.results = results or {}
        self.fail_on = fail_on  # "initialize" | "list_tools" | "call_tool" -> raise there
        self.calls: list[tuple] = []
        self.entered = self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *exc):
        self.exited += 1
        return False

    async def initialize(self):
        self.calls.append(("initialize",))
        if self.fail_on == "initialize":
            raise ConnectionError("connection refused")

    async def list_tools(self, **kw):
        self.calls.append(("list_tools", kw))
        if self.fail_on == "list_tools":
            raise RuntimeError("tools/list failed")
        return SimpleNamespace(tools=self.tools, next_cursor=None)

    async def call_tool(self, name, arguments=None, **kw):
        self.calls.append(("call_tool", name, arguments))
        if self.fail_on == "call_tool":
            raise RuntimeError("boom")
        return self.results[name]


@pytest.fixture
def wired(monkeypatch):
    """A fake transport + session in place of the mcp client; returns the recorder."""
    state = SimpleNamespace(session=FakeSession(), transports=[], read_timeouts=[])

    @asynccontextmanager
    async def fake_transport(srv, timeout):
        state.transports.append((srv["name"], srv["transport"], dict(srv.get("headers") or {}), timeout))
        yield "read", "write"

    def fake_session(read, write, timeout):
        assert (read, write) == ("read", "write")
        state.read_timeouts.append(timeout)
        return state.session

    monkeypatch.setattr(mt, "open_transport", fake_transport)
    monkeypatch.setattr(mt, "client_session", fake_session)
    return state


def text(t):
    return SimpleNamespace(type="text", text=t)


def image(data="QUJD", mime="image/png"):
    return SimpleNamespace(type="image", data=data, mime_type=mime)


def result(*content, is_error=False, structured=None):
    return SimpleNamespace(content=list(content), is_error=is_error, structured_content=structured)


# ---- names and schemas --------------------------------------------------------------------
def test_tool_names_are_prefixed_sanitized_and_cut_to_64():
    assert mt.sanitize_tool_name("github", "list_issues") == "mcp_github_list_issues"
    assert mt.sanitize_tool_name("my-srv", "search/web pages!") == "mcp_my-srv_search_web_pages"
    long = mt.sanitize_tool_name("a" * 40, "b" * 40)
    assert len(long) == 64 and long.startswith("mcp_" + "a" * 40 + "_")
    assert mt.sanitize_tool_name("x", "日本語") == "mcp_x"


def test_collisions_get_a_numeric_suffix_within_64_characters():
    taken = {"mcp_x_y"}
    assert mt.unique_tool_name("mcp_x_y", taken) == "mcp_x_y_2"
    taken.add("mcp_x_y_2")
    assert mt.unique_tool_name("mcp_x_y", taken) == "mcp_x_y_3"
    base = "m" * 64
    assert mt.unique_tool_name(base, {base}) == "m" * 62 + "_2"


def test_input_schema_passes_objects_through_and_defaults_otherwise():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    assert mt.input_schema_of(schema) == schema
    assert mt.input_schema_of({"properties": {"a": {}}}) == {"properties": {"a": {}}, "type": "object"}
    for bad in (None, "x", [], {"type": "string"}):
        assert mt.input_schema_of(bad) == {"type": "object", "properties": {}}
    assert mt.input_schema_of(None) is not mt.DEFAULT_INPUT_SCHEMA  # a copy, never the shared constant


def test_tools_schema_comes_from_the_cache_without_any_network(wired):
    """Two servers, a disabled one, a collision across servers; nothing opens a transport."""
    servers = [
        server("github", tools=[{"name": "search", "description": "Search code", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}},
                                {"name": "sea rch", "description": ""}]),  # sanitizes to the same name
        server("off", enabled=False, tools=[{"name": "hidden", "description": "never"}]),
        server("corp", admin=True, tools=[{"name": "lookup", "description": "Company lookup", "input_schema": "nope"}]),
    ]
    tools = mt.MCPTools(USER, servers=servers)
    schema = tools.tools_schema
    assert [t["name"] for t in schema] == ["mcp_github_search", "mcp_github_sea_rch", "mcp_corp_lookup"]
    assert schema[0]["description"] == "[MCP github] Search code"
    assert schema[0]["input_schema"]["properties"] == {"q": {"type": "string"}}
    assert schema[1]["description"] == "[MCP github]" and schema[1]["input_schema"] == {"type": "object", "properties": {}}
    assert schema[2]["input_schema"] == {"type": "object", "properties": {}}
    assert tools.tools_schema is not schema and tools.tools_schema == schema  # cached, copied out
    assert tools.resolve("mcp_github_sea_rch") == (servers[0], "sea rch")
    assert tools.resolve("mcp_off_hidden") is None
    assert wired.transports == []
    assert tools.prompt_notes == mt.PROMPT_NOTES
    assert mt.MCPTools(USER, servers=[]).prompt_notes == ""


def test_a_true_collision_across_servers_keeps_both_tools():
    servers = [server("a_b", tools=[{"name": "c"}]), server("a", tools=[{"name": "b_c"}])]
    names = [t["name"] for t in mt.MCPTools(USER, servers=servers).tools_schema]
    assert names == ["mcp_a_b_c", "mcp_a_b_c_2"]


def test_tools_schema_reads_the_store_when_no_servers_are_given(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(mcp_store.ENV_ADMIN_FILE, str(tmp_path / "admin.json"))
    mcp_store.save_user(USER, [server("mine", tools=[{"name": "t1"}])])
    (tmp_path / "admin.json").write_text(json.dumps({"servers": [{"name": "shared", "url": "https://s.example.com/mcp"}]}))
    mcp_store.save_admin_cache([{"name": "shared", "tools": [{"name": "t2", "description": "d"}], "updated": "x", "error": None}])
    assert [t["name"] for t in mt.MCPTools(USER).tools_schema] == ["mcp_mine_t1", "mcp_shared_t2"]
    assert mt.MCPTools("nobody").tools_schema[0]["name"] == "mcp_shared_t2"


# ---- discovery ----------------------------------------------------------------------------
async def test_discover_converts_the_tool_list(wired):
    wired.session.tools = [
        SimpleNamespace(name="search", description="Search", input_schema={"type": "object", "properties": {"q": {}}}),
        SimpleNamespace(name="legacy", description=None, inputSchema={"type": "object"}),  # mcp 1.x spelling
        SimpleNamespace(name="odd", description="x", input_schema=["not", "a", "schema"]),
    ]
    srv = server("demo", transport="sse", headers={"Authorization": "Bearer t"})
    tools = await mt.MCPTools(USER, servers=[]).discover(srv, timeout=7)
    assert tools == [
        {"name": "search", "description": "Search", "input_schema": {"type": "object", "properties": {"q": {}}}},
        {"name": "legacy", "description": "", "input_schema": {"type": "object", "properties": {}}},
        {"name": "odd", "description": "x", "input_schema": {"type": "object", "properties": {}}},
    ]
    assert wired.transports == [("demo", "sse", {"Authorization": "Bearer t"}, 7)]
    assert wired.read_timeouts == [7]
    assert wired.session.calls[0] == ("initialize",) and wired.session.entered == wired.session.exited == 1


async def test_discover_turns_every_failure_into_a_plain_reason(wired, monkeypatch):
    wired.session.fail_on = "initialize"
    with pytest.raises(mt.MCPUnreachable, match="ConnectionError: connection refused"):
        await mt.MCPTools(USER, servers=[]).discover(server())

    @asynccontextmanager
    async def dead_transport(srv, timeout):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [OSError("All connection attempts failed")])
        yield  # noqa: unreachable - makes this a generator

    monkeypatch.setattr(mt, "open_transport", dead_transport)
    with pytest.raises(mt.MCPUnreachable, match="OSError: All connection attempts failed"):
        await mt.MCPTools(USER, servers=[]).discover(server())


async def test_discover_times_out(wired, monkeypatch):
    async def hang(**kw):
        await asyncio.sleep(5)

    monkeypatch.setattr(wired.session, "list_tools", hang)
    with pytest.raises(mt.MCPUnreachable, match="timed out"):
        await mt.MCPTools(USER, servers=[]).discover(server(), timeout=0.05)


# ---- calls --------------------------------------------------------------------------------
@pytest.fixture
def tools(wired):
    srv = server("demo", tools=[{"name": "echo"}, {"name": "pic"}, {"name": "res"}, {"name": "bad"}], headers={"Authorization": "Bearer t"})
    return mt.MCPTools(USER, servers=[srv]), srv


async def test_execute_tool_joins_text_content(tools, wired):
    provider, srv = tools
    wired.session.results["echo"] = result(text("one"), text("two"))
    assert await provider.execute_tool("mcp_demo_echo", {"text": "x"}) == "one\ntwo"
    assert wired.session.calls[-1] == ("call_tool", "echo", {"text": "x"})
    assert wired.transports == [("demo", "http", {"Authorization": "Bearer t"}, mt.CALL_TIMEOUT)]
    assert wired.session.entered == wired.session.exited == 1  # closed again after the call


async def test_execute_tool_returns_blocks_when_there_is_an_image(tools, wired):
    provider, _ = tools
    wired.session.results["pic"] = result(text("here"), image(), image(mime="image/bmp"))
    blocks = await provider.execute_tool("mcp_demo_pic", {})
    assert blocks == [
        {"type": "text", "text": "here"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
        {"type": "text", "text": "[image image/bmp omitted: not displayable]"},
    ]


def test_an_image_over_the_api_limit_becomes_a_note(monkeypatch):
    """The API refuses the whole request when one image's base64 data is over its limit."""
    from agents import content

    monkeypatch.setattr(content, "MAX_IMAGE_BYTES", 8)
    items = [SimpleNamespace(type="image", mime_type="image/png", data="QUJDREVG"),  # 8 chars: fits
             SimpleNamespace(type="image", mime_type="image/png", data="QUJDREVGRw==")]  # 12 chars: too large
    fits, note = mt.convert_content(items)
    assert fits == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJDREVG"}}
    assert note["type"] == "text" and note["text"].startswith("[image image/png omitted: too large to show (")


async def test_execute_tool_converts_resources_and_structured_content(tools, wired):
    provider, _ = tools
    wired.session.results["res"] = result(
        SimpleNamespace(type="resource", resource=SimpleNamespace(uri="file:///a.txt", text="file text")),
        SimpleNamespace(type="resource", resource=SimpleNamespace(uri="file:///b.bin", blob="AAAA")),
        SimpleNamespace(type="resource_link", uri="https://x/y", name="y"),
        SimpleNamespace(type="audio", mime_type="audio/wav", data="..."),
    )
    assert await provider.execute_tool("mcp_demo_res", {}) == "file text\n[resource file:///b.bin]\n[resource https://x/y (y)]\n[audio audio/wav omitted]"
    wired.session.results["res"] = result(structured={"a": 1})
    assert await provider.execute_tool("mcp_demo_res", {}) == '{"a": 1}'
    wired.session.results["res"] = result()
    assert await provider.execute_tool("mcp_demo_res", {}) == "(the tool returned no content)"


async def test_is_error_results_become_error_strings(tools, wired):
    provider, _ = tools
    wired.session.results["bad"] = result(text("rate limited"), is_error=True)
    assert await provider.execute_tool("mcp_demo_bad", {}) == "Error: rate limited"
    wired.session.results["bad"] = SimpleNamespace(content=[text("Error executing tool bad")], isError=True)  # mcp 1.x spelling
    assert await provider.execute_tool("mcp_demo_bad", {}) == "Error executing tool bad"
    wired.session.results["bad"] = result(is_error=True)
    assert (await provider.execute_tool("mcp_demo_bad", {})).startswith("Error: the server reported an error")


async def test_connection_errors_never_raise_and_are_recorded(tools, wired, monkeypatch):
    provider, srv = tools
    recorded = []
    monkeypatch.setattr(mcp_store, "record_error", lambda uid, s, err: recorded.append((uid, s["name"], err)))
    wired.session.fail_on = "call_tool"
    assert await provider.execute_tool("mcp_demo_echo", {}) == "Error: MCP server demo unreachable: boom"
    assert recorded == [(USER, "demo", "boom")] and srv["error"] == "boom"
    wired.session.fail_on = None
    wired.session.results["echo"] = result(text("ok"))
    assert await provider.execute_tool("mcp_demo_echo", {}) == "ok"
    assert recorded[-1] == (USER, "demo", None) and srv["error"] is None  # cleared once it works again


async def test_a_failing_store_write_does_not_hide_the_result(tools, wired, monkeypatch):
    provider, _ = tools

    def broken(*a):
        raise OSError("disk full")

    monkeypatch.setattr(mcp_store, "record_error", broken)
    wired.session.fail_on = "initialize"
    assert (await provider.execute_tool("mcp_demo_echo", {})).startswith("Error: MCP server demo unreachable: ConnectionError")


async def test_a_hung_server_times_out(tools, wired, monkeypatch):
    provider, _ = tools

    async def hang(*a, **kw):
        await asyncio.sleep(5)

    monkeypatch.setattr(wired.session, "call_tool", hang)
    monkeypatch.setattr(mt, "CALL_TIMEOUT", 0.05)
    monkeypatch.setattr(mcp_store, "record_error", lambda *a: None)
    assert await provider.execute_tool("mcp_demo_echo", {}) == "Error: MCP server demo unreachable: timed out"


async def test_unknown_tools_are_reported(tools):
    provider, _ = tools
    assert await provider.execute_tool("mcp_other_x", {}) == "Unknown tool: mcp_other_x"


def test_error_reason_unwraps_groups_and_names_empty_exceptions():
    assert mt.error_reason(ExceptionGroup("g", [ExceptionGroup("h", [ValueError("inner")])])) == "inner"
    assert mt.error_reason(OSError()) == "OSError"
    assert mt.error_reason(asyncio.TimeoutError()) == "timed out"
    assert len(mt.error_reason(RuntimeError("x" * 1000))) == mt.MAX_REASON_CHARS


def http_status_error(url: str, status: int = 401):
    """The real httpx error of a refused request: its message embeds the full URL, query string included."""
    import httpx

    response = httpx.Response(status, request=httpx.Request("GET", url))
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return e
    raise AssertionError("raise_for_status accepted the status")


def test_error_reason_hides_query_strings_and_the_httpx_hint():
    """An SSE server refusing with 401 names the URL; a key passed in its query must not come back."""
    exc = http_status_error("http://127.0.0.1:9/sse?api_key=URLSECRET")
    reason = mt.error_reason(exc)
    assert reason == "HTTPStatusError: Client error '401 Unauthorized' for url 'http://127.0.0.1:9/sse'"
    assert mt.error_reason(ExceptionGroup("g", [exc])) == reason
    redacted = mt.redact_urls("a https://u:pw@h.example.com:8443/x?k=s#f, 'http://localhost:3000/mcp?t=1' and https://plain.example.com/p.")
    assert redacted == "a https://h.example.com:8443/x, 'http://localhost:3000/mcp' and https://plain.example.com/p."


async def test_url_secrets_never_reach_the_user_the_store_or_the_model(tools, wired, monkeypatch):
    provider, srv = tools
    recorded = []
    monkeypatch.setattr(mcp_store, "record_error", lambda uid, s, err: recorded.append(err))

    async def refused():
        raise http_status_error("http://127.0.0.1:9/sse?api_key=URLSECRET")

    monkeypatch.setattr(wired.session, "initialize", refused)
    with pytest.raises(mt.MCPUnreachable) as info:
        await provider.discover(srv)
    assert "URLSECRET" not in str(info.value) and "401 Unauthorized" in str(info.value)
    result = await provider.execute_tool("mcp_demo_echo", {})
    assert result.startswith("Error: MCP server demo unreachable: HTTPStatusError") and "URLSECRET" not in result
    assert recorded == [srv["error"]] and "URLSECRET" not in srv["error"] and "127.0.0.1:9/sse" in srv["error"]


# ---- the real client over an in-memory server ---------------------------------------------------
@pytest.fixture
def memory_server(monkeypatch):
    """An mcp 2.x MCPServer with three tools, reached through the real ClientSession over memory streams."""
    from mcp.server.lowlevel import Server
    from mcp.server.mcpserver import MCPServer
    from mcp.shared.memory import create_client_server_memory_streams
    import mcp.types as types

    srv = MCPServer("demo")
    lowlevel = next(v for v in vars(srv).values() if isinstance(v, Server))

    @srv.tool(description="Echo the text back")
    def echo(text: str) -> str:
        return f"echo: {text}"

    @srv.tool(description="A picture")
    def picture() -> types.ImageContent:
        return types.ImageContent(type="image", data="QUJD", mimeType="image/png")

    @srv.tool(description="Fails")
    def broken() -> str:
        raise ValueError("kaput")

    @asynccontextmanager
    async def memory_transport(server_record, timeout):
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            task = asyncio.create_task(lowlevel.run(server_streams[0], server_streams[1], lowlevel.create_initialization_options()))
            try:
                yield client_streams[0], client_streams[1]
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(mt, "open_transport", memory_transport)
    return srv


async def test_real_session_discovers_and_calls_tools(memory_server):
    record = server("demo")
    tools = await mt.MCPTools(USER, servers=[]).discover(record)
    assert [(t["name"], t["description"]) for t in tools] == [("echo", "Echo the text back"), ("picture", "A picture"), ("broken", "Fails")]
    assert tools[0]["input_schema"]["properties"]["text"]["type"] == "string"
    record["tools"] = tools
    provider = mt.MCPTools(USER, servers=[record])
    assert [t["name"] for t in provider.tools_schema] == ["mcp_demo_echo", "mcp_demo_picture", "mcp_demo_broken"]
    assert await provider.execute_tool("mcp_demo_echo", {"text": "hi"}) == "echo: hi"
    blocks = await provider.execute_tool("mcp_demo_picture", {})
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/png"
    assert (await provider.execute_tool("mcp_demo_broken", {})).startswith("Error")  # isError from the server
    assert "text" in await provider.execute_tool("mcp_demo_echo", {})  # argument validation error, as text


# ---- the store ----------------------------------------------------------------------------
@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(mcp_store.ENV_ADMIN_FILE, str(tmp_path / "data" / "mcp_servers.json"))
    return tmp_path


def test_names_are_lowercased_and_validated(store):
    assert mcp_store.validate_name("GitHub") == "github"
    assert mcp_store.validate_name("a-b_c9") == "a-b_c9"
    for bad in ("", "-x", "_x", "a b", "a.b", "x" * 33, "ünïcode"):
        with pytest.raises(ValueError):
            mcp_store.validate_name(bad)


def test_users_need_https_to_a_public_host_the_admin_does_not(store):
    """The bot runs on the host network: a user's "localhost" would be the server's own services."""
    assert mcp_store.validate_url("https://api.example.com/mcp") == "https://api.example.com/mcp"
    assert mcp_store.validate_url("https://93.184.215.14/mcp") == "https://93.184.215.14/mcp"
    for admin_only in ("http://internal.corp/mcp", "http://localhost:3000/mcp", "http://127.0.0.1/mcp", "https://10.0.0.5/mcp"):
        assert mcp_store.validate_url(admin_only, admin=True) == admin_only
    for plain in ("http://example.com/mcp", "http://localhost:3000/mcp", "http://127.0.0.1:8081/"):
        with pytest.raises(ValueError, match="Only https"):
            mcp_store.validate_url(plain)
    for local in ("https://localhost/mcp", "https://LOCALHOST./mcp", "https://api.localhost/mcp", "https://127.0.0.1:8081/",
                  "https://[::1]/mcp", "https://10.0.0.5/mcp", "https://192.168.1.2/", "https://172.16.0.1/",
                  "https://169.254.169.254/latest", "https://100.64.0.1/", "https://0.0.0.0/", "https://[::ffff:127.0.0.1]/",
                  "https://[fe80::1]/", "https://[fd00::1]/", "https://224.0.0.1/"):
        with pytest.raises(ValueError, match="public address"):
            mcp_store.validate_url(local)
    for bad in ("ftp://x/y", "example.com/mcp", "", "https://"):
        with pytest.raises(ValueError):
            mcp_store.validate_url(bad)
    with pytest.raises(ValueError):
        mcp_store.validate_transport("websocket")


def test_check_public_host_resolves_the_name(store, monkeypatch):
    answers = {"public.example": ["93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c"],
               "rebound.example": ["127.0.0.1"], "mixed.example": ["93.184.215.14", "10.1.2.3"]}

    def resolve(host, port):
        if host not in answers:
            raise OSError("Name or service not known")
        return answers[host]

    monkeypatch.setattr(mcp_store, "_resolve", resolve)
    mcp_store.check_public_host("https://public.example/mcp")
    mcp_store.check_public_host("https://unknown.example/mcp")  # does not resolve: the connection fails on its own
    for url in ("https://rebound.example/mcp", "https://mixed.example:8443/mcp", "http://localhost:3000/", "https://10.0.0.5/"):
        with pytest.raises(ValueError, match="public address"):
            mcp_store.check_public_host(url)
    monkeypatch.setattr(mcp_store, "_resolve", REAL_RESOLVE)
    with pytest.raises(ValueError, match="public address"):  # numeric spellings resolve without DNS
        mcp_store.check_public_host("https://2130706433/")


async def test_a_users_server_is_checked_before_every_connection(wired, monkeypatch):
    """Discovery and calls refuse a user's server whose name resolves to a local address, even one
    written straight into mcp.json; the admin's servers (shared or their own) are not checked."""
    from bot.auth import auth

    resolved: list[str] = []

    def resolve(host, port):
        resolved.append(host)
        return ["127.0.0.1"]

    monkeypatch.setattr(mcp_store, "_resolve", resolve)
    recorded: list = []
    monkeypatch.setattr(mcp_store, "record_error", lambda uid, s, err: recorded.append(err))
    record = server("evil", tools=[{"name": "probe", "description": "", "input_schema": {}}])
    with pytest.raises(mt.MCPUnreachable, match="public address"):
        await mt.MCPTools(USER, servers=[]).discover(record)
    result = await mt.MCPTools(USER, servers=[record]).execute_tool("mcp_evil_probe", {})
    assert result.startswith("Error: MCP server evil unreachable: The server must be on a public address")
    assert wired.transports == [] and resolved == ["evil.example.com", "evil.example.com"]  # never connected
    assert recorded == [result.split("unreachable: ", 1)[1]]  # shown by /mcp like any connection error

    shared = server("corp", admin=True, tools=[{"name": "probe", "description": "", "input_schema": {}}])
    assert await mt.MCPTools(USER, servers=[]).discover(shared) == []
    monkeypatch.setattr(auth, "admin_id", USER)  # the admin's own servers are trusted too
    assert await mt.MCPTools(USER, servers=[]).discover(record) == []
    assert resolved == ["evil.example.com", "evil.example.com"] and len(wired.transports) == 2


def test_new_server_stores_the_token_as_a_bearer_header_and_never_renders_it(store):
    record = mcp_store.new_server("GitHub", "https://x.example.com/mcp", "ghp_secret", "sse")
    assert record["name"] == "github" and record["transport"] == "sse" and record["enabled"] is True
    assert record["headers"] == {"Authorization": "Bearer ghp_secret"}
    assert mcp_store.describe_headers(record) == "token set"
    assert mcp_store.describe_headers({"headers": {"X-Api-Key": "k", "Authorization": "Bearer t"}}) == "token set, 1 more header"
    assert mcp_store.describe_headers({"headers": {"X-Api-Key": "k"}}) == "1 header"
    assert mcp_store.describe_headers({}) == ""
    assert mcp_store.new_server("x", "https://x.example.com/mcp")["headers"] == {}
    assert mcp_store.display_url("https://user:pw@host.com:8443/mcp?token=abc#f") == "https://host.com:8443/mcp"


def test_user_round_trip_add_replace_toggle_tools_remove(store):
    a = mcp_store.new_server("a", "https://a.example.com/mcp", "t1")
    b = mcp_store.new_server("b", "https://b.example.com/mcp")
    mcp_store.add_user_server(USER, a)
    mcp_store.add_user_server(USER, b)
    assert [s["name"] for s in mcp_store.load_user(USER)] == ["a", "b"]
    assert (store / "data" / USER / "mcp.json").exists() and not list((store / "data" / USER).glob("*.tmp"))

    replaced = mcp_store.new_server("a", "https://a2.example.com/mcp", "t2")
    mcp_store.add_user_server(USER, replaced)
    servers = mcp_store.load_user(USER)
    assert [s["name"] for s in servers] == ["a", "b"] and servers[0]["url"] == "https://a2.example.com/mcp"
    assert servers[0]["headers"] == {"Authorization": "Bearer t2"}

    assert mcp_store.toggle_user_server(USER, "a") is False
    assert mcp_store.toggle_user_server(USER, "a") is True
    assert mcp_store.toggle_user_server(USER, "zzz") is None

    updated = mcp_store.update_user_tools(USER, "b", [{"name": "t", "description": "d", "input_schema": {"type": "object"}}])
    assert updated["tools"][0]["name"] == "t" and updated["updated"] and updated["error"] is None
    failed = mcp_store.update_user_tools(USER, "b", None, "401 Unauthorized")
    assert failed["tools"][0]["name"] == "t" and failed["error"] == "401 Unauthorized"  # tools kept on failure
    assert mcp_store.update_user_tools(USER, "zzz", []) is None

    assert mcp_store.remove_user_server(USER, "a") is True
    assert mcp_store.remove_user_server(USER, "a") is False
    assert [s["name"] for s in mcp_store.load_user(USER)] == ["b"]
    assert mcp_store.load_user("someone-else") == []


def test_corrupt_or_odd_files_are_tolerated(store, caplog):
    path = mcp_store.user_path(USER)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert mcp_store.load_user(USER) == []
    path.write_text(json.dumps({"servers": [
        {"name": "ok", "url": "https://ok.example.com/mcp", "tools": "nope", "headers": None, "error": "e" * 500},
        {"name": "Bad Name", "url": "https://x.example.com/mcp"},
        {"name": "plainhttp", "url": "http://x.example.com/mcp"},  # stored http survives: the policy applies when adding
        {"name": "nohost", "url": "https:///mcp"},
        {"name": "ok", "url": "https://dup.example.com/mcp"},
        "garbage",
    ]}))
    servers = mcp_store.load_user(USER)
    assert [s["name"] for s in servers] == ["ok", "plainhttp"]
    assert servers[0]["tools"] == [] and servers[0]["headers"] == {} and len(servers[0]["error"]) == mcp_store.MAX_ERROR_CHARS
    assert servers[0]["transport"] == "http" and servers[0]["enabled"] is True


def test_admin_servers_merge_their_cache_and_are_read_only_records(store):
    admin = store / "data" / "mcp_servers.json"
    admin.parent.mkdir(parents=True)
    admin.write_text(json.dumps({"servers": [
        {"name": "corp", "url": "http://intranet/mcp", "headers": {"X-Key": "k"}},  # http is fine for the admin
        {"name": "off", "url": "https://off.example.com/mcp", "enabled": False},
    ]}))
    before = admin.read_text()
    servers = mcp_store.load_admin()
    assert [(s["name"], s["admin"], s["enabled"], s["tools"]) for s in servers] == [("corp", True, True, []), ("off", True, False, [])]

    servers[0] = mcp_store.with_tools(servers[0], [{"name": "lookup", "description": "d", "input_schema": {"type": "object"}}])
    servers[1] = mcp_store.with_tools(servers[1], None, "unreachable")
    mcp_store.save_admin_cache(servers)
    assert mcp_store.admin_cache_path() == store / "data" / "mcp_servers.cache.json"
    assert admin.read_text() == before  # the hand-edited file is never rewritten
    cached = json.loads(mcp_store.admin_cache_path().read_text())["servers"]
    assert [sorted(c) for c in cached] == [["error", "name", "tools", "updated"]] * 2

    merged = mcp_store.load_admin()
    assert merged[0]["tools"][0]["name"] == "lookup" and merged[0]["error"] is None and merged[0]["updated"]
    assert merged[1]["error"] == "unreachable"
    assert [s["name"] for s in mcp_store.all_servers(USER)] == ["corp", "off"]
    mcp_store.add_user_server(USER, mcp_store.new_server("mine", "https://m.example.com/mcp"))
    assert [s["name"] for s in mcp_store.all_servers(USER)] == ["mine", "corp", "off"]
    assert "admin" not in json.loads(mcp_store.user_path(USER).read_text())["servers"][0]


def test_missing_admin_file_means_no_admin_servers(store):
    assert mcp_store.load_admin() == [] and mcp_store.all_servers(USER) == []


def test_record_error_writes_user_and_admin_sides_only_when_changed(store):
    mcp_store.add_user_server(USER, mcp_store.new_server("mine", "https://m.example.com/mcp"))
    mine = mcp_store.load_user(USER)[0]
    mcp_store.record_error(USER, mine, "boom")
    assert mcp_store.load_user(USER)[0]["error"] == "boom"
    mtime = mcp_store.user_path(USER).stat().st_mtime_ns
    mcp_store.record_error(USER, mine, "boom")  # same error: no write
    assert mcp_store.user_path(USER).stat().st_mtime_ns == mtime
    mcp_store.record_error(USER, mine, None)
    assert mcp_store.load_user(USER)[0]["error"] is None
    mcp_store.record_error(USER, {"name": "ghost"}, "x")  # unknown server: nothing happens

    admin = store / "data" / "mcp_servers.json"
    admin.parent.mkdir(parents=True, exist_ok=True)
    admin.write_text(json.dumps({"servers": [{"name": "corp", "url": "https://c.example.com/mcp"}]}))
    corp = mcp_store.load_admin()[0]
    mcp_store.record_error(USER, corp, "down")
    assert mcp_store.load_admin()[0]["error"] == "down"
    mcp_store.record_error(USER, corp, None)
    assert mcp_store.load_admin()[0]["error"] is None


def test_record_error_never_raises(store, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(mcp_store, "_write", boom)
    mcp_store.record_error(USER, {"name": "x", "admin": True}, "e")  # the admin side writes → swallowed


def test_tool_integrator_never_names_the_mcp_tools():
    """The sandbox patcher rebinds methods by name; MCPTools must stay out of it (see tests/test_image_tools.py)."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app" / "secure_container" / "tool_integrator.py").read_text(encoding="utf-8")
    assert "MCPTools" not in source and "mcp_tools" not in source
