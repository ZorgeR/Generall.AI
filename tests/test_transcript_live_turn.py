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
