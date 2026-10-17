"""browse_webpage (sandboxed Chromium), last_json_line, download_webpage extraction, browse.py."""
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import web_tools
from agents.web_tools import WebTools, last_json_line

BANNER = (
    "Secure container environment initialized\nAvailable tools:\n- Python Python 3.12.0\n"
    "Executing command: python /home/runner/workspace/temp_tool_ab12cd34.py https://x\n"
    "Working directory: /home/runner/workspace\n-----------------------------------\n"
)
PAGE = {
    "title": "Example Domain",
    "final_url": "https://example.com/final",
    "text": "This domain is for use in illustrative examples.\n\n| a | b |\n| 1 | 2 |",
    "links": [[f"Link {i}", f"https://example.com/{i}"] for i in range(25)],
    "screenshot": None,
    "error": None,
}


class FakeManager:
    """Stands in for ContainerManager: records run_script calls, returns canned container logs."""

    def __init__(self, data=PAGE, logs=None, error=None, write_screenshot=True):
        self.data, self.logs, self.error, self.write_screenshot = data, logs, error, write_screenshot
        self.calls = []

    def run_script(self, user_id, script, args, timeout=90, network_enabled=False, mem_limit=None):
        self.calls.append(SimpleNamespace(user_id=user_id, script=script, args=list(args), timeout=timeout,
                                          network_enabled=network_enabled, mem_limit=mem_limit))
        if self.error:
            raise self.error
        if self.logs is not None:
            return self.logs
        data = dict(self.data)
        shot = args[args.index("--screenshot") + 1] if "--screenshot" in args else ""
        if shot and self.write_screenshot:
            path = Path("data") / user_id / shot
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG fake")
            data["screenshot"] = shot
        return BANNER + json.dumps(data) + "\n"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "7").mkdir(parents=True)
    monkeypatch.setattr(web_tools, "_container_manager", None)
    return tmp_path


def use_manager(monkeypatch, manager):
    monkeypatch.setattr(web_tools, "_container_manager", manager)
    return manager


# ---- last_json_line --------------------------------------------------------------------
def test_last_json_line_skips_the_banner_and_takes_the_last_object():
    logs = BANNER + '{"first": 1}\nsome print\n{"second": 2}\n\n'
    assert last_json_line(logs) == {"second": 2}
    assert last_json_line(BANNER + "[1, 2]\n") is None  # arrays are not a script result
    assert last_json_line(BANNER + "{not json}\n") is None
    assert last_json_line("") is None and last_json_line(None) is None
    assert last_json_line('{"a": 1}\nError (exit code 1):\ntrace') == {"a": 1}


def test_last_json_line_survives_unicode_line_separators_in_the_text():
    """Page text may contain U+2028/U+2029/U+0085; str.splitlines() would cut the result line there."""
    page = dict(PAGE, text="first line\u2028second\u2029third\u0085fourth", links=[])
    for line in (json.dumps(page), json.dumps(page, ensure_ascii=False)):
        data = last_json_line(BANNER + line + "\n")
        assert data is not None and data["text"] == page["text"]
    assert last_json_line(BANNER + '{"a": "x\u2028y"}\r\n') == {"a": "x\u2028y"}  # a trailing CR is stripped


# ---- the tool ---------------------------------------------------------------------------
def test_browse_runs_the_script_with_network_and_memory_and_formats_the_page(workspace, monkeypatch):
    manager = use_manager(monkeypatch, FakeManager())
    tools = WebTools("7", None)

    result = tools.execute_tool("browse_webpage", {"url": "https://example.com"})

    call = manager.calls[0]
    assert call.user_id == "7" and call.script == web_tools.BROWSE_SCRIPT == "browse.py"
    assert call.network_enabled is True and call.mem_limit == "1g" and call.timeout == 75
    assert call.args == ["https://example.com", "--screenshot", "", "--wait-ms", "1500", "--max-chars", "20000"]
    assert result.startswith("Title: Example Domain\nURL: https://example.com/final\n\nThis domain")
    assert "| a | b |" in result
    assert "Links (first 20 of 25):" in result and "- Link 0 — https://example.com/0" in result
    assert "Link 19" in result and "Link 20" not in result
    assert "Screenshot" not in result


def test_browse_screenshot_lands_in_downloads_and_is_offered_to_view_image(workspace, monkeypatch):
    manager = use_manager(monkeypatch, FakeManager())
    result = WebTools("7").execute_tool("browse_webpage", {"url": "https://example.com", "screenshot": True, "wait_ms": 3000})

    shot = manager.calls[0].args[manager.calls[0].args.index("--screenshot") + 1]
    assert shot.startswith("downloads/screenshot_") and shot.endswith(".png")
    assert (workspace / "data" / "7" / shot).exists()
    assert f"Screenshot: {shot} (view it with view_image)" in result
    assert manager.calls[0].args[manager.calls[0].args.index("--wait-ms") + 1] == "3000"
    assert (workspace / "data" / "7" / "downloads").is_dir()  # created before the container ran


def test_browse_reports_a_screenshot_the_browser_could_not_write(workspace, monkeypatch):
    use_manager(monkeypatch, FakeManager(write_screenshot=False))
    result = WebTools("7").execute_tool("browse_webpage", {"url": "https://example.com", "screenshot": True})
    assert "Screenshot: not saved" in result


def test_browse_caps_text_and_clamps_arguments(workspace, monkeypatch):
    long_page = dict(PAGE, text="x" * 5000, links=[])
    manager = use_manager(monkeypatch, FakeManager(data=long_page))
    result = WebTools("7").execute_tool("browse_webpage", {"url": "https://e.com", "max_chars": 1000, "wait_ms": 99999})
    assert "x" * 1000 + "\n[... truncated]" in result and "x" * 1001 not in result
    args = manager.calls[0].args
    assert args[args.index("--max-chars") + 1] == "1000" and args[args.index("--wait-ms") + 1] == "10000"
    # nonsense values fall back to the defaults instead of failing the call
    WebTools("7").execute_tool("browse_webpage", {"url": "https://e.com", "max_chars": "lots", "wait_ms": None})
    args = manager.calls[1].args
    assert args[args.index("--max-chars") + 1] == "20000" and args[args.index("--wait-ms") + 1] == "1500"
    assert "Links" not in result


def test_browse_refuses_non_http_urls_before_starting_a_container(workspace, monkeypatch):
    manager = use_manager(monkeypatch, FakeManager())
    tools = WebTools("7")
    for bad in ("file:///etc/passwd", "ftp://x.y/z", "example.com", "", None, "javascript:alert(1)"):
        assert tools.execute_tool("browse_webpage", {"url": bad}).startswith("Error:"), bad
    assert manager.calls == []


def test_browse_errors_are_strings_never_exceptions(workspace, monkeypatch):
    use_manager(monkeypatch, FakeManager(data=dict(PAGE, error="Timed out after 60 s while loading the page")))
    assert WebTools("7").execute_tool("browse_webpage", {"url": "https://e.com"}) == "Error: Timed out after 60 s while loading the page"

    use_manager(monkeypatch, FakeManager(logs=BANNER + "Traceback (most recent call last):\n  boom\n"))
    result = WebTools("7").execute_tool("browse_webpage", {"url": "https://e.com"})
    assert result.startswith("Error: the sandbox browser returned no result") and "boom" in result

    use_manager(monkeypatch, FakeManager(error=RuntimeError("Cannot connect to the Docker daemon")))
    result = WebTools("7").execute_tool("browse_webpage", {"url": "https://e.com"})
    assert result.startswith("Error: could not start the sandbox browser") and "Docker" in result


def test_schema_and_dispatch(workspace):
    tools = WebTools("7", sender=None)
    assert [t["name"] for t in tools.tools_schema] == ["browse_webpage"]
    schema = tools.tools_schema[0]["input_schema"]
    assert schema["required"] == ["url"] and set(schema["properties"]) == {"url", "wait_ms", "screenshot", "max_chars"}
    assert tools.execute_tool("nope", {}) == "Unknown tool: nope"


async def test_browse_runs_in_a_worker_thread_through_the_registry(workspace, monkeypatch):
    """Registered blocking=True, the agent runs the sync execute_tool via asyncio.to_thread."""
    use_manager(monkeypatch, FakeManager())
    result = await asyncio.to_thread(WebTools("7").execute_tool, "browse_webpage", {"url": "https://e.com"})
    assert result.startswith("Title: Example Domain")


def test_get_container_manager_reuses_the_startup_instance(monkeypatch):
    monkeypatch.setattr(web_tools, "_container_manager", None)
    sandbox_main = importlib.import_module("secure_container.main")
    sentinel = object()
    monkeypatch.setattr(sandbox_main, "_container_manager", sentinel, raising=False)
    assert web_tools.get_container_manager() is sentinel
    assert web_tools.get_container_manager() is sentinel  # cached


# ---- the sandbox patcher must leave the new providers alone ---------------------------------
def test_the_sandbox_patcher_never_names_the_new_providers():
    """A patched method gets the patcher's signature; these tools must keep their own (see
    test_image_tools for how image editing once broke that way)."""
    from agents.document_tools import DocumentTools
    from secure_container.secure_tool_wrapper import SecureToolWrapper

    source = (Path(__file__).resolve().parents[1] / "app" / "secure_container" / "tool_integrator.py").read_text()
    for name in ("WebTools", "DocumentTools", "web_tools", "document_tools", "browse_webpage", "create_document"):
        assert name not in source, name
    for cls in (WebTools, DocumentTools):
        public = {n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n))}
        assert not public & SecureToolWrapper.SECURE_TOOL_METHODS, cls.__name__
    for tool in ("browse_webpage", "create_document"):
        assert tool not in SecureToolWrapper.SECURE_TOOL_METHODS


def test_main_registers_both_providers_as_blocking():
    source = (Path(__file__).resolve().parents[1] / "app" / "agents" / "main.py").read_text()
    assert "self.agent.register_provider(self.web_tools, blocking=True)" in source
    assert "self.agent.register_provider(self.document_tools, blocking=True)" in source


# ---- download_webpage extraction ---------------------------------------------------------------
ARTICLE = " ".join(f"Sentence number {i} of the article body explains the subject in some detail." for i in range(12))
HTML = f"""<!DOCTYPE html><html><head><title>Sample</title><script>var x = 1;</script>
<style>body {{ color: red }}</style></head><body>
<nav><a href="/">Home</a> | <a href="/about">About</a> | <a href="/login">Login</a></nav>
<header><h1>Sample article</h1></header>
<main><article>
<p>{ARTICLE}</p>
<p>Second paragraph with a table below.</p>
<table><tr><th>Region</th><th>Sales</th></tr><tr><td>North</td><td>1200</td></tr><tr><td>South</td><td>800</td></tr></table>
</article></main>
<footer>Copyright 2026 Sample Inc. Privacy policy. Cookie settings.</footer>
</body></html>"""


@pytest.fixture
def file_ops_module(monkeypatch):
    module = importlib.import_module("agents.file_ops")
    monkeypatch.setattr(module, "_TRAFILATURA", module._UNSET)
    return module


def test_extract_readable_text_keeps_the_article_and_tables_and_drops_boilerplate(file_ops_module):
    text = file_ops_module.extract_readable_text(HTML)
    assert "Sentence number 0 of the article body" in text and "Sentence number 11" in text
    assert "North" in text and "1200" in text  # include_tables
    assert "var x = 1" not in text
    assert "Cookie settings" not in text and "Login" not in text  # navigation and footer are gone


def test_extract_readable_text_falls_back_to_the_page_text_without_trafilatura(file_ops_module, monkeypatch):
    monkeypatch.setattr(file_ops_module, "_TRAFILATURA", None)
    text = file_ops_module.extract_readable_text(HTML)
    assert "Sentence number 0" in text and "North" in text
    assert "Home" in text and "Cookie settings" in text  # the whole page, boilerplate included
    assert "var x = 1" not in text and "color: red" not in text  # scripts and styles still removed


def test_extract_readable_text_distrusts_a_short_extraction(file_ops_module):
    short = "<html><body><nav>Menu Home About</nav><p>Tiny.</p></body></html>"
    text = file_ops_module.extract_readable_text(short)
    assert "Tiny." in text and "Menu Home About" in text  # below MIN_EXTRACTED_CHARS: whole page

    class Broken:
        @staticmethod
        def extract(html, **kw):
            raise RuntimeError("lxml exploded")

    file_ops_module._TRAFILATURA = Broken()
    assert "Sentence number 0" in file_ops_module.extract_readable_text(HTML)


def test_download_webpage_uses_the_extractor_and_keeps_save_to_file(file_ops_module, tmp_path, monkeypatch):
    FileOperations = file_ops_module.FileOperations
    ops = FileOperations.__new__(FileOperations)
    ops.user_id, ops.sender = "7", None
    ops.base_path = tmp_path / "data" / "7"
    ops.downloads_path = ops.base_path / "downloads"
    ops.downloads_path.mkdir(parents=True)
    monkeypatch.setattr(file_ops_module.requests, "get",
                        lambda url, **kw: SimpleNamespace(text=HTML, raise_for_status=lambda: None))

    preview = ops.download_webpage("https://example.com/post", max_chars=120)
    assert preview.endswith("...") and len(preview) == 123 and "Login" not in preview

    saved = ops.download_webpage("https://example.com/post", save_to_file=True, filename="post.txt", max_chars=50)
    assert saved.startswith("Webpage content saved to downloads/post.txt")
    content = (ops.downloads_path / "post.txt").read_text(encoding="utf-8")
    assert "Sentence number 11" in content and "1200" in content and "Cookie settings" not in content

    monkeypatch.setattr(file_ops_module.requests, "get", lambda url, **kw: (_ for _ in ()).throw(OSError("dns")))
    assert ops.download_webpage("https://nope").startswith("Error downloading webpage: dns")


@pytest.fixture
def ops_in_data(file_ops_module, tmp_path, monkeypatch):
    """A real FileOperations for user 111 under tmp_path/data (cwd), next to user 222's files."""
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "data" / "222"
    victim.mkdir(parents=True)
    (victim / "settings.json").write_text('{"keep": true}', encoding="utf-8")
    ops = file_ops_module.FileOperations("111")
    (ops.base_path / "a.txt").write_text("hello", encoding="utf-8")
    page = SimpleNamespace(text=HTML, raise_for_status=lambda: None,
                           iter_content=lambda chunk_size: iter([b"payload"]))
    monkeypatch.setattr(file_ops_module.requests, "get", lambda url, **kw: page)
    return SimpleNamespace(ops=ops, data=tmp_path / "data", victim=victim, outside=tmp_path / "outside.txt")


def files_outside_user_111(data: Path) -> set[str]:
    return {str(p.relative_to(data)) for p in data.rglob("*") if p.is_file() and not p.is_relative_to(data / "111")}


def test_download_webpage_and_zip_names_never_leave_the_workspace(ops_in_data):
    """The model's filename / archive_name keep only their base name: `..` or an absolute path lands in downloads/."""
    env, ops = ops_in_data, ops_in_data.ops
    before = files_outside_user_111(env.data)

    saved = ops.download_webpage("https://example.com/x", save_to_file=True, filename="../../222/settings.json")
    assert saved.startswith("Webpage content saved to downloads/settings.json")
    assert (env.victim / "settings.json").read_text(encoding="utf-8") == '{"keep": true}'
    saved = ops.download_webpage("https://example.com/x", save_to_file=True, filename=str(env.outside))
    assert saved.startswith("Webpage content saved to downloads/outside.txt") and not env.outside.exists()
    assert ops.download_webpage("https://example.com/x", save_to_file=True, filename="..").startswith("Error")

    zipped = ops.create_zip_archive(["a.txt"], archive_name="../../222/evil")
    assert "Path: downloads/evil.zip" in zipped and (ops.base_path / "downloads" / "evil.zip").is_file()
    assert "Path: downloads/abs.zip" in ops.create_zip_archive(["a.txt"], archive_name=str(env.data / "222" / "abs"))
    assert ops.download_file("https://example.com/f", filename="../../222/f.bin").endswith("downloads/f.bin")

    assert files_outside_user_111(env.data) == before  # nothing written next to another user's files


def test_a_planted_link_in_downloads_is_refused_before_anything_is_written(ops_in_data):
    """downloads/ is writable from the sandbox: a link there pointing outside must not be written through."""
    env, ops = ops_in_data, ops_in_data.ops
    env.outside.write_text("original", encoding="utf-8")
    for name in ("page.txt", "f.bin", "bundle.zip"):
        os.symlink(env.outside, ops.base_path / "downloads" / name)

    assert ops.download_webpage("https://example.com/x", save_to_file=True, filename="page.txt").startswith("Error")
    assert ops.download_file("https://example.com/f", filename="f.bin").startswith("Error")
    assert ops.create_zip_archive(["a.txt"], archive_name="bundle").startswith("Error")
    assert env.outside.read_text(encoding="utf-8") == "original"


# ---- browse.py (the sandbox side) ------------------------------------------------------------------
@pytest.fixture
def browse_module(monkeypatch):
    module = importlib.import_module("secure_container.scripts.browse")
    monkeypatch.setattr(module, "_printed", False)
    return module


def test_browse_script_argument_parsing_and_url_validation(browse_module):
    args = browse_module.parse_args(["https://e.com"])
    assert (args.url, args.screenshot, args.wait_ms, args.max_chars) == ("https://e.com", "", 1500, 60000)
    args = browse_module.parse_args(["https://e.com", "--screenshot", "downloads/s.png", "--wait-ms", "0", "--max-chars", "10"])
    assert (args.screenshot, args.wait_ms, args.max_chars) == ("downloads/s.png", 0, 10)
    assert browse_module.validate_url("https://e.com/x?y=1") is None
    assert browse_module.validate_url("http://e.com") is None
    for bad in ("file:///etc/passwd", "chrome://settings", "e.com", "https://", "javascript:1"):
        assert browse_module.validate_url(bad), bad


def test_browse_script_prints_one_json_line_on_a_bad_url(browse_module, capsys):
    assert browse_module.main(["ftp://e.com/x"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["error"] == "Only http and https URLs can be opened" and data["final_url"] == "ftp://e.com/x"
    assert set(data) == {"title", "final_url", "text", "links", "screenshot", "error"}


def test_browse_script_reports_a_missing_browser_as_json(browse_module, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "playwright", None)  # "import playwright.sync_api" now raises ImportError
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    assert browse_module.main(["https://e.com"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    data = json.loads(out[-1])
    assert data["error"] and ("playwright" in data["error"].lower() or "import" in data["error"].lower())
    assert data["text"] == "" and data["links"] == []


def test_browse_script_emits_one_ascii_json_line(browse_module, capsys):
    """Non-ASCII, line separators included, is escaped so the host can split the logs on newlines."""
    browse_module.emit(browse_module.result(title="Пример", final_url="https://e.com", text="a\u2028b\u2029c"))
    browse_module.emit(browse_module.result(error="second call is ignored"))
    out = capsys.readouterr().out
    assert out.isascii() and out.count("\n") == 1
    data = json.loads(out)
    assert data["title"] == "Пример" and data["text"] == "a\u2028b\u2029c" and data["error"] is None


def test_browse_script_extract_text_prefers_trafilatura_then_the_body(browse_module):
    assert browse_module.extract_text(HTML, "fallback body").startswith("Sentence number 0") or "Sentence number 0" in browse_module.extract_text(HTML, "fallback body")
    assert browse_module.extract_text("<html><body></body></html>", "fallback body") == "fallback body"


class FakePage:
    """Just enough of a Playwright page for browse.browse(); the screenshot and title may fail."""

    def __init__(self, screenshot_error=None, title_error=None):
        self.screenshot_error, self.title_error = screenshot_error, title_error
        self.url = "https://e.com/final"

    def on(self, *a):
        pass

    def goto(self, url, **kw):
        pass

    def wait_for_load_state(self, *a, **kw):
        pass

    def wait_for_timeout(self, ms):
        pass

    def content(self):
        return "<html><body></body></html>"

    def inner_text(self, selector, **kw):
        return "Text the browser already read"

    def evaluate(self, script, arg):
        return [["Home", "https://e.com/"]]

    def screenshot(self, path, **kw):
        if self.screenshot_error:
            raise self.screenshot_error
        Path(path).write_bytes(b"\x89PNG")

    def title(self):
        if self.title_error:
            raise self.title_error
        return "Fake title"


def use_fake_playwright(monkeypatch, page):
    import types

    browser = SimpleNamespace(new_context=lambda **kw: SimpleNamespace(new_page=lambda: page), close=lambda: None)

    class Playwright:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(launch=lambda **kw: browser))

        def __exit__(self, *exc):
            return False

    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = Playwright
    package = types.ModuleType("playwright")
    package.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)


def _browse(browse_module, capsys, *args):
    assert browse_module.main(["https://e.com", *args]) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_browse_script_takes_the_screenshot_when_asked(browse_module, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    use_fake_playwright(monkeypatch, FakePage())
    data = _browse(browse_module, capsys, "--screenshot", "downloads/s.png", "--wait-ms", "0")
    assert data["error"] is None and data["screenshot"] == "downloads/s.png" and (tmp_path / "downloads" / "s.png").exists()
    assert data["title"] == "Fake title" and "already read" in data["text"] and data["links"] == [["Home", "https://e.com/"]]


def test_a_failed_screenshot_keeps_the_page_that_was_read(browse_module, monkeypatch, tmp_path, capsys):
    """A screenshot timing out on web fonts used to replace the whole result with the error."""
    monkeypatch.chdir(tmp_path)
    use_fake_playwright(monkeypatch, FakePage(
        screenshot_error=TimeoutError("page.screenshot: Timeout 10000ms exceeded. waiting for fonts to load"),
        title_error=RuntimeError("Execution context was destroyed"),
    ))
    data = _browse(browse_module, capsys, "--screenshot", "downloads/s.png", "--wait-ms", "0")
    assert data["error"] is None and data["screenshot"] is None and data["title"] is None
    assert "already read" in data["text"] and data["final_url"] == "https://e.com/final" and data["links"]
    # the host then reports the text with "Screenshot: not saved"
    formatted = WebTools("7")._format(data, "downloads/s.png", 20_000)
    assert "already read" in formatted and "Screenshot: not saved" in formatted
