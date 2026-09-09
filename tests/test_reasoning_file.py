"""The reasoning file: the turn as a transcript, tool arguments included."""
from bot.agent_runner import reasoning_text

TURN = [
    {"role": "user", "content": "Add a papakha and replace the background"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "The user attached a photo, so this is an edit."},
            {"type": "text", "text": "Editing your photo."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "generate_image",
                "input": {"prompt": "add a papakha", "images": ["images/photo.jpg"], "quality": "high"},
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Edited 1 image(s) with gpt-image-2.5-sunburst"}
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "Here it is."}]},
]


def test_tool_calls_are_rendered_with_their_arguments():
    text = reasoning_text(TURN)
    assert "[Tool call: generate_image]" in text
    # the parameters the call was actually made with, which is the point of the file
    assert '"images"' in text and "images/photo.jpg" in text and '"quality": "high"' in text
    # the result names the call it belongs to
    assert "[Tool result: generate_image]" in text
    assert "gpt-image-2.5-sunburst" in text


def test_every_block_is_rendered_not_just_the_first():
    text = reasoning_text(TURN)
    # a thinking block used to hide the text and tool calls that followed it
    assert "[Assistant thinking]" in text
    assert "Editing your photo." in text and "Here it is." in text
    assert "Add a papakha and replace the background" in text


def test_failed_calls_are_marked_and_results_carry_block_content():
    text = reasoning_text(
        [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "run_command", "input": {"command": "ls"}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "is_error": True,
                        "content": [{"type": "text", "text": "boom"}],
                    }
                ],
            },
        ]
    )
    assert "[Tool result: run_command — error]" in text and "boom" in text


def test_long_values_are_clipped_and_junk_is_survivable():
    text = reasoning_text(
        [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {"p": "y" * 9000}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "z" * 9000}]},
            {"role": "assistant", "content": [None, {"type": "text", "text": "still here"}]},
            {"role": "user", "content": None},
            {},
        ]
    )
    assert "chars total)" in text and len(text) < 12000
    assert "still here" in text


def test_no_messages_is_empty_not_a_stray_separator():
    assert reasoning_text([]) == ""
    assert reasoning_text([{"role": "assistant", "content": []}]) == ""
