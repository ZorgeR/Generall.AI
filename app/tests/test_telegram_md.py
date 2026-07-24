"""Tests for the Markdown -> Telegram formatting converter.

Run with::

    cd app && python -m unittest discover tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram_md import (  # noqa: E402
    HTML,
    MARKDOWN_V2,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    render,
    render_plain,
    split_markdown,
)


class InlineTests(unittest.TestCase):
    def test_bold_and_italic(self):
        self.assertEqual(render("**bold** and *italic*"), "*bold* and _italic_")
        self.assertEqual(render("**bold**", HTML), "<b>bold</b>")

    def test_underline_strike_spoiler(self):
        self.assertEqual(render("__under__"), "__under__")
        self.assertEqual(render("~~gone~~"), "~gone~")
        self.assertEqual(render("||secret||"), "||secret||")
        self.assertEqual(render("~~gone~~", HTML), "<s>gone</s>")
        self.assertEqual(
            render("||secret||", HTML), '<span class="tg-spoiler">secret</span>'
        )

    def test_nested_entities(self):
        self.assertEqual(render("**bold with *italic* inside**"), "*bold with _italic_ inside*")
        self.assertEqual(
            render("**bold with *italic* inside**", HTML), "<b>bold with <i>italic</i> inside</b>"
        )

    def test_special_characters_are_escaped(self):
        self.assertEqual(render("2 + 2 = 4 (really!)"), "2 \\+ 2 \\= 4 \\(really\\!\\)")
        self.assertEqual(render("a < b & c > d", HTML), "a &lt; b &amp; c &gt; d")

    def test_underscores_in_identifiers_are_literal(self):
        # The classic crash of the legacy Markdown mode.
        self.assertEqual(render("open file_name_here.py"), "open file\\_name\\_here\\.py")

    def test_multiplication_is_not_italic(self):
        self.assertEqual(render("3 * 4 * 5"), "3 \\* 4 \\* 5")

    def test_inline_code(self):
        self.assertEqual(render("use `rm -rf` now"), "use `rm -rf` now")
        self.assertEqual(render("use `a<b` now", HTML), "use <code>a&lt;b</code> now")

    def test_links(self):
        self.assertEqual(render("[docs](https://example.com/a_b)"), "[docs](https://example.com/a_b)")
        self.assertEqual(
            render("[docs](https://example.com)", HTML), '<a href="https://example.com">docs</a>'
        )

    def test_images_become_links(self):
        self.assertEqual(render("![cat](https://e.com/c.png)"), "[🖼 cat](https://e.com/c.png)")

    def test_autolink(self):
        self.assertEqual(render("<https://e.com>"), "[https://e\\.com](https://e.com)")

    def test_display_math_becomes_code(self):
        self.assertEqual(render("energy $$E = mc^2$$ here"), "energy `E = mc^2` here")

    def test_escaped_markers_stay_literal(self):
        self.assertEqual(render("literal \\*stars\\*"), "literal \\*stars\\*")


class BlockTests(unittest.TestCase):
    def test_headings(self):
        self.assertEqual(render("## Section"), "*__Section__*")
        self.assertEqual(render("### Sub"), "*Sub*")
        self.assertEqual(render("## Section", HTML), "<b><u>Section</u></b>")

    def test_code_block_keeps_language(self):
        source = "```python\nprint('hi')\n```"
        self.assertEqual(render(source), "```python\nprint('hi')\n```")
        self.assertEqual(
            render(source, HTML), '<pre><code class="language-python">print(\'hi\')</code></pre>'
        )

    def test_code_block_content_is_not_over_escaped(self):
        rendered = render("```\na_b = c.d\n```")
        self.assertEqual(rendered, "```\na_b = c.d\n```")

    def test_unordered_list_uses_bullets(self):
        rendered = render("- one\n- two")
        self.assertEqual(rendered, "• one\n• two")

    def test_nested_list_is_indented(self):
        rendered = render("- one\n  - deep\n- two")
        self.assertEqual(rendered, "• one\n  ◦ deep\n• two")

    def test_ordered_list_keeps_numbers(self):
        self.assertEqual(render("1. first\n2. second"), "*1\\.* first\n*2\\.* second")

    def test_task_list(self):
        self.assertEqual(render("- [x] done\n- [ ] todo"), "☑ done\n☐ todo")

    def test_blockquote(self):
        self.assertEqual(render("> quoted line"), ">quoted line")
        self.assertEqual(render("> quoted line", HTML), "<blockquote>quoted line</blockquote>")

    def test_long_blockquote_is_expandable(self):
        source = "\n".join("> line %d" % i for i in range(12))
        self.assertTrue(render(source).startswith("**>"))
        self.assertTrue(render(source).endswith("||"))
        self.assertIn("<blockquote expandable>", render(source, HTML))

    def test_horizontal_rule(self):
        self.assertEqual(render("---"), "──────────")

    def test_table_becomes_monospace_grid(self):
        source = "| Name | Size |\n| --- | --- |\n| a | 1 |\n| b | 22 |"
        rendered = render(source)
        self.assertTrue(rendered.startswith("```\n"))
        self.assertIn("Name  Size", rendered)
        self.assertIn("b     22", rendered)

    def test_wide_table_falls_back_to_records(self):
        header = "| Feature | Description of the feature in great detail | Notes |"
        separator = "| --- | --- | --- |"
        row = "| Alpha | " + "x" * 60 + " | fine |"
        rendered = render("\n".join([header, separator, row]))
        self.assertNotIn("```", rendered)
        self.assertIn("*Alpha*", rendered)
        self.assertIn("Notes", rendered)

    def test_paragraphs_are_separated(self):
        self.assertEqual(render("first\n\nsecond"), "first\n\nsecond")

    def test_mixed_document(self):
        source = (
            "# Title\n\n"
            "Intro with **bold**.\n\n"
            "- item `code`\n"
            "- item [link](https://e.com)\n\n"
            "```js\nlet a = 1;\n```\n"
        )
        rendered = render(source)
        self.assertIn("*__Title__*", rendered)
        self.assertIn("• item `code`", rendered)
        self.assertIn("```js\nlet a = 1;\n```", rendered)
        # No unescaped Markdown syntax leaked into the plain text runs.
        self.assertNotIn("**", rendered)


class PlainTests(unittest.TestCase):
    def test_plain_strips_markup(self):
        self.assertEqual(render_plain("**bold** and `code`"), "bold and code")

    def test_plain_keeps_structure(self):
        self.assertEqual(render_plain("## Title\n\n- a\n- b"), "Title\n\n• a\n• b")

    def test_plain_keeps_link_targets(self):
        self.assertEqual(
            render_plain("see [docs](https://e.com)"), "see docs (https://e.com)"
        )


class SplitTests(unittest.TestCase):
    def test_short_text_is_one_piece(self):
        self.assertEqual(split_markdown("hello"), ["hello"])

    def test_empty_text_yields_nothing(self):
        self.assertEqual(split_markdown("   "), [])

    def test_long_text_is_split_and_fits(self):
        source = "\n\n".join("Paragraph %d. %s" % (i, "word " * 80) for i in range(60))
        pieces = split_markdown(source)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            for flavour in (MARKDOWN_V2, HTML):
                self.assertLessEqual(len(render(piece, flavour)), TELEGRAM_MAX_MESSAGE_LENGTH)

    def test_split_code_block_is_reopened(self):
        body = "\n".join("line_%d = %d" % (i, i) for i in range(900))
        source = "intro\n\n```python\n%s\n```\n" % body
        pieces = split_markdown(source)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertEqual(piece.count("```") % 2, 0, "unbalanced fence in piece")
            rendered = render(piece)
            self.assertLessEqual(len(rendered), TELEGRAM_MAX_MESSAGE_LENGTH)
        code_pieces = [p for p in pieces if "```python" in p]
        self.assertGreaterEqual(len(code_pieces), 2)

    def test_entities_never_span_pieces(self):
        source = "\n\n".join("**bold %d** and text %s" % (i, "z" * 200) for i in range(60))
        for piece in split_markdown(source):
            rendered = render(piece)
            self.assertEqual(rendered.count("*") % 2, 0)

    def test_no_content_is_lost(self):
        source = "\n\n".join("Paragraph %d unique-marker-%d" % (i, i) for i in range(200))
        joined = " ".join(split_markdown(source))
        for i in range(200):
            self.assertIn("unique-marker-%d" % i, joined)


class RobustnessTests(unittest.TestCase):
    def test_unbalanced_markers_do_not_break(self):
        for source in ("**oops", "a * b", "`unclosed", "||half", "[link](", "~~~", "> "):
            rendered = render(source)
            self.assertIsInstance(rendered, str)

    def test_none_and_empty(self):
        self.assertEqual(render(None), "")
        self.assertEqual(render(""), "")

    def test_deeply_nested_input_terminates(self):
        source = "*" * 40 + "text" + "*" * 40
        self.assertIsInstance(render(source), str)


if __name__ == "__main__":
    unittest.main()
