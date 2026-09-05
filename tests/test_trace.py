import time

from agents.trace import ToolCall, ToolTrace, describe_args
from bot.agent_runner import entity_safe, render_trace, trace_summary


def test_describe_args_prefers_informative_keys_and_truncates():
    assert describe_args({"page": 2, "query": "aiogram rich messages"}) == "aiogram rich messages"
    assert describe_args({"filename": "notes.txt", "content": "x" * 500}) == "notes.txt"
    assert describe_args({"content": "line one\nline   two"}) == "line one line two"
    long = describe_args({"url": "https://example.com/" + "a" * 200})
    assert len(long) == 60 and long.endswith("…")
    assert describe_args({}) == "" and describe_args(None) == "" and describe_args({"n": 3}) == ""


def test_trace_records_calls_in_order_with_status():
    trace = ToolTrace()
    a = trace.start("search_web", {"query": "q"})
    b = trace.start("run_command", {"command": "ls /nope"})
    assert trace.running == 2 and trace.total == 2
    a.done("10 results", ok=True)
    b.done("Error (exit code 2): no such file", ok=False)
    assert trace.running == 0 and trace.errors == 1
    assert a.preview == "10 results" and b.preview.startswith("Error")
    assert trace.counts_by_name() == [("search_web", 1), ("run_command", 1)]
    trace.start("search_web", {"query": "again"}).done("ok", ok=True)
    assert trace.counts_by_name() == [("search_web", 2), ("run_command", 1)]


def test_render_trace_shows_icons_args_and_durations():
    trace = ToolTrace()
    trace.start("search_web", {"query": "snake_case term"}).done("ok", ok=True)
    trace.start("run_command", {"command": "ls"}).done("Error", ok=False)
    trace.start("download_webpage", {"url": "https://x.y"})
    text = render_trace(trace)
    lines = text.splitlines()
    assert lines[0].startswith("🔧 *Tools* (3, ")
    assert lines[1].startswith("✅ `search_web` snake\\_case term · ")  # markdown-escaped args
    assert lines[2].startswith("❌ `run_command` ls · ")
    assert lines[3] == "⏳ `download_webpage` https://x.y"  # running: no duration yet
    assert render_trace(ToolTrace()) == "" and render_trace(None) == ""


def test_render_trace_keeps_only_the_most_recent_lines():
    trace = ToolTrace()
    for i in range(14):
        trace.start("t", {"query": f"call {i}"}).done("ok", ok=True)
    text = render_trace(trace, limit=10)
    assert "… 4 earlier" in text
    assert "call 13" in text and "call 3" not in text


def test_trace_summary_line():
    trace = ToolTrace()
    for name in ("search_web", "search_web", "execute_python"):
        trace.start(name, {}).done("ok", ok=True)
    trace.start("run_command", {}).done("Error", ok=False)
    text = trace_summary(trace)
    assert text.startswith("*🔧 4 tool calls in ")
    assert "· 1 failed*" in text
    assert text.endswith(": search\\_web ×2, execute\\_python, run\\_command")
    single = ToolTrace()
    single.start("x", {}).done("ok", ok=True)
    assert "1 tool call in" in trace_summary(single)


def test_entity_safe_never_escapes_inside_legacy_entities():
    # Legacy Markdown forbids backslash escapes inside *bold* / _italic_: the characters are replaced.
    assert entity_safe("Processing tool results run_command") == "Processing tool results run-command"
    assert entity_safe("a*b`c[d") == "a-b-c-d"
    assert "\\" not in entity_safe("run_command")


def test_rendered_trace_lines_keep_escapes_outside_entities():
    trace = ToolTrace()
    trace.start("run_command", {"command": "for ip in 1.1.1.1 8.8.8.8; do echo \"=== $ip\"; done"}).done("ok", ok=True)
    line = render_trace(trace).splitlines()[1]
    assert line.startswith("✅ `run_command` for ip in 1.1.1.1 8.8.8.8; do echo \"=== $ip\"; done · ")


def test_trace_keeps_per_model_usage_and_result_excerpts():
    trace = ToolTrace()
    call = trace.start("read_file", {"filename": "a.txt"})
    call.done("x" * 2000, ok=True)
    assert len(call.result_excerpt) == 800 and call.result_excerpt.endswith("…")
    assert '"filename": "a.txt"' in call.args_text
    trace.add_usage({"input_tokens": 10, "output_tokens": 5}, model="claude-sonnet-5")
    trace.add_usage({"input_tokens": 20, "output_tokens": 5}, model="claude-haiku-4-5")
    assert trace.usage_by_model["claude-haiku-4-5"]["input_tokens"] == 20 and trace.input_tokens == 30
    assert trace.cost_usd is not None and trace.cost_usd > 0
    trace.add_thinking("  ")
    trace.add_thinking("plan")
    assert trace.thinking_text == "plan"
