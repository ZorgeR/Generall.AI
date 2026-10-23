"""transcript helpers with a live turn's mixed message (tool results + injected user message)."""
from agents import transcript as T


def turn(i, mixed=False):
    content = [{"type": "tool_result", "tool_use_id": f"c{i}", "content": f"result {i}"}]
    if mixed:
        content.append({"type": "text", "text": f'<user_message sent_during_turn="true">\nmore {i}\n</user_message>'})
    return [
        {"role": "user", "content": [{"type": "text", "text": f"question {i}"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": f"c{i}", "name": "x", "input": {}}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": f"answer {i}"}]},
    ]


def test_a_mixed_message_is_not_a_turn_start():
    messages = turn(1) + turn(2, mixed=True) + turn(3)
    starts = [i for i, m in enumerate(messages) if T.is_user_turn(m)]
    assert starts == [0, 4, 8]
    old, rest = T.split_for_summary(messages)
    assert rest[0]["content"][0]["type"] == "text"  # never cuts between a tool_use and its result


def test_old_mixed_results_are_cleared_but_the_user_text_stays():
    messages = turn(1, mixed=True) + turn(2) + turn(3)
    assert T.clear_old_tool_results(messages, keep_turns=2) == 1
    content = messages[2]["content"]
    assert content[0]["content"] == T.CLEARED_MARKER and content[1]["text"].startswith("<user_message")
    view = T.text_only_view(messages)
    assert any("more 1" in m["content"] for m in view if m["role"] == "user")


PIXEL = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}


def note(task_id, result="a very long build log"):
    return {"type": "text", "text": (
        f'<task_notification id="{task_id}" tool="run_command" status="completed" elapsed="95s">\n'
        f"<result>\n{result}\n</result>\n</task_notification>")}


def test_old_task_notifications_are_cleared_like_tool_results():
    """A finished background task's result is cleared with the tool results of its turn: behind
    tool results and in a "background task finished" turn alike; the opening tag (id, tool, status)
    stays, the pictures are left to clear_old_media; the last turns are never touched."""
    behind = turn(1)
    behind[2]["content"] += [note("t3"), PIXEL]
    notified = [
        {"role": "user", "content": [{"type": "text", "text": "<context>\nnow\n</context>"},
                                     {"type": "text", "text": "[Background task finished] A task you started earlier finished."},
                                     note("t4")]},
        {"role": "assistant", "content": [{"type": "text", "text": "t4 is done"}]},
    ]
    recent = turn(3)
    recent[2]["content"].append(note("t5"))
    messages = behind + notified + recent + turn(4)

    assert T.clear_old_tool_results(messages, keep_turns=2) == 3  # c1's result, t3, t4
    cleared = '<task_notification id="{}" tool="run_command" status="completed" elapsed="95s">\n' + T.CLEARED_MARKER + "\n</task_notification>"
    assert behind[2]["content"][0]["content"] == T.CLEARED_MARKER
    assert behind[2]["content"][1] == {"type": "text", "text": cleared.format("t3")} and behind[2]["content"][2] == PIXEL
    assert notified[0]["content"][1]["text"].startswith("[Background task finished]")
    assert notified[0]["content"][2] == {"type": "text", "text": cleared.format("t4")}
    assert recent[2]["content"][1] == note("t5")  # within the last two turns
    assert T.clear_old_tool_results(messages, keep_turns=2) == 0  # idempotent
    assert T.clear_old_media(messages, keep_turns=2) == 1 and behind[2]["content"][2]["text"] == T.IMAGE_CLEARED_MARKER


def notice(task_id):
    return {"role": "system", "content": f'The background task {task_id} (run_command) completed after 95 s. Read its result with get_task_result (task_id "{task_id}").'}


def test_a_system_notice_is_no_turn_start_and_no_text_turn():
    """The bot's notice (a background task ended) sits between the tool results and the next
    assistant message: a prune never starts a kept part with it, the text-only views (the quick
    path, summaries) leave it out, and the size estimate still counts it."""
    first = turn(1)
    first.insert(3, notice("t1"))
    messages = first + turn(2) + turn(3)
    assert [i for i, m in enumerate(messages) if T.is_user_turn(m)] == [0, 5, 9]
    old, rest = T.split_for_summary(messages)
    assert rest[0]["role"] == "user" and rest[0]["content"][0]["text"] == "question 2"
    view = T.text_only_view(messages)
    assert all(m["role"] in ("user", "assistant") for m in view) and "get_task_result" not in str(view)
    assert T.estimate_tokens(messages) > T.estimate_tokens([m for m in messages if m["role"] != "system"])
    assert T.clear_old_tool_results(messages, keep_turns=1) == 2 and messages[3] == notice("t1")  # facts only: kept
    assert T.strip_thinking(messages) == 0 and messages[3] == notice("t1")


def test_sanitize_keeps_a_system_notice_in_place():
    messages = turn(1)
    messages.insert(3, notice("t1"))
    out = T.sanitize_turn(messages, "answer 1")
    assert out[3] == notice("t1") and out[-1]["content"][0]["text"] == "answer 1"
