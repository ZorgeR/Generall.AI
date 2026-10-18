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
