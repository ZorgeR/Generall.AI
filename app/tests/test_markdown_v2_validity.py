"""Validate that rendered output is legal Telegram MarkdownV2.

Telegram rejects a message when an entity is left open or when one of the
reserved characters appears unescaped in plain text. The validator below
applies those two rules to the renderer's output, and is then run over a
corpus of awkward inputs plus a deterministic fuzz sweep.
"""

import os
import random
import sys
import unittest
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram_md import HTML, MARKDOWN_V2, render, split_markdown  # noqa: E402

RESERVED = set("[]()~>#+-=|{}.!")


class MarkdownV2Error(AssertionError):
    pass


def _strip_quote_markers(text: str) -> str:
    """Remove blockquote prefixes/suffixes, which are line-level markup."""
    lines = []
    expandable = False
    for line in text.split("\n"):
        if line.startswith("**>"):
            expandable = True
            line = line[3:]
        elif line.startswith(">"):
            line = line[1:]
        if expandable and line.endswith("||"):
            # Closing marker of an expandable blockquote.
            line = line[:-2]
            expandable = False
        lines.append(line)
    return "\n".join(lines)


def assert_valid_markdown_v2(text: str) -> None:
    """Raise unless ``text`` parses as MarkdownV2 with all entities closed."""
    source = _strip_quote_markers(text)
    stack = []
    index = 0
    length = len(source)

    while index < length:
        char = source[index]

        if char == "\\":
            if index + 1 >= length:
                raise MarkdownV2Error("trailing backslash")
            index += 2
            continue

        if source.startswith("```", index):
            end = source.find("```", index + 3)
            if end == -1:
                raise MarkdownV2Error("unclosed pre block")
            index = end + 3
            continue

        if char == "`":
            cursor = index + 1
            while cursor < length:
                if source[cursor] == "\\":
                    cursor += 2
                    continue
                if source[cursor] == "`":
                    break
                cursor += 1
            if cursor >= length:
                raise MarkdownV2Error("unclosed code span")
            index = cursor + 1
            continue

        for marker, kind in (("||", "spoiler"), ("__", "underline")):
            if source.startswith(marker, index):
                if stack and stack[-1] == kind:
                    stack.pop()
                else:
                    stack.append(kind)
                index += len(marker)
                break
        else:
            if char in "*_~":
                kind = {"*": "bold", "_": "italic", "~": "strike"}[char]
                if stack and stack[-1] == kind:
                    stack.pop()
                else:
                    stack.append(kind)
                index += 1
                continue

            if char == "[":
                close = _find_unescaped(source, "]", index + 1)
                if close == -1 or not source.startswith("(", close + 1):
                    raise MarkdownV2Error("malformed link label")
                paren = _find_unescaped(source, ")", close + 2)
                if paren == -1:
                    raise MarkdownV2Error("malformed link target")
                # The label may itself contain entities - validate it separately.
                assert_valid_markdown_v2(source[index + 1 : close])
                index = paren + 1
                continue

            if char in RESERVED:
                raise MarkdownV2Error(
                    "unescaped reserved character %r at %d in %r" % (char, index, text)
                )

            index += 1

    if stack:
        raise MarkdownV2Error("unclosed entities %s in %r" % (stack, text))


def _find_unescaped(text: str, needle: str, start: int) -> int:
    index = start
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == needle:
            return index
        index += 1
    return -1


CORPUS = [
    "",
    "plain text",
    "**bold** *italic* __under__ ~~strike~~ ||spoiler||",
    "nested **bold *italic* end**",
    "file_name.py and other_file.py",
    "5 * 3 = 15 (approximately -2%)",
    "a [link](https://e.com/x_y-z?q=1&w=2) inline",
    "![pic](https://e.com/a.png)",
    "`code_with_underscores` and ``double`` ticks",
    "```python\nx = a_b['c'] # note *this*\n```",
    "# H1\n## H2\n### H3\n#### H4",
    "- one\n- two\n  - nested\n    - deeper",
    "1. first\n2) second",
    "- [ ] todo\n- [x] done",
    "> quote line\n> second line",
    "\n".join("> long quote line %d" % i for i in range(20)),
    "| a | b |\n| --- | --- |\n| 1 | 2 |",
    "| Long header one | Long header two | Long header three |\n| --- | --- | --- |\n| %s | y | z |" % ("x" * 50),
    "---",
    "Math $$E = mc^2$$ inline",
    "unbalanced **bold start",
    "unbalanced `code start",
    "|| spoiler with spaces ||",
    "~~~\nfenced with tildes\n~~~",
    "Mixed: **bold `code` and [link](https://e.com)** done",
    "text with \\*escaped\\* markers",
    "emoji 🤖 and ünïcödé, «quotes», 中文",
    "> quote with **bold** and `code`\n\nafter quote",
    "- item with **bold** and [link](https://e.com/a_b)\n- item with `code`",
    "line1\n\n\n\nline2",
    "__underline with _italic_ inside__",
    "***bold italic***",
    "a_b_c_d_e",
    "100% of $5.00 -> +2 [ok]",
    "```\nno language\n```",
    "```js\nconst re = /a`b/;\n```",
    "<https://example.com/auto_link>",
    "<b>not html</b> & <i>tags</i>",
]


class ValidityTests(unittest.TestCase):
    def test_corpus_renders_to_valid_markdown_v2(self):
        for source in CORPUS:
            with self.subTest(source=source[:40]):
                assert_valid_markdown_v2(render(source, MARKDOWN_V2))

    def test_split_pieces_are_individually_valid(self):
        big = "\n\n".join(CORPUS * 12)
        pieces = split_markdown(big)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            assert_valid_markdown_v2(render(piece, MARKDOWN_V2))

    def test_fuzz(self):
        rng = random.Random(20260724)
        atoms = [
            "**", "*", "_", "__", "~~", "~", "||", "`", "```", "[", "]", "(", ")",
            "#", "##", "> ", "- ", "1. ", "|", "---", "\n", "\n\n", " ", "text",
            "a_b", "https://e.com", "!", ".", "\\", "$$x$$", "- [ ] ", "🤖",
        ]
        for _ in range(3000):
            source = "".join(rng.choice(atoms) for _ in range(rng.randint(1, 25)))
            rendered = render(source, MARKDOWN_V2)
            try:
                assert_valid_markdown_v2(rendered)
            except MarkdownV2Error as error:
                self.fail("input %r -> %r: %s" % (source, rendered, error))

    def test_validator_rejects_bad_markup(self):
        for bad in ("*unclosed", "text with (parens)", "a - b", "[label]"):
            with self.assertRaises(MarkdownV2Error):
                assert_valid_markdown_v2(bad)


class RegressionTests(unittest.TestCase):
    """Cases found by the fuzzer - each one used to produce invalid markup."""

    def test_nested_quotes_are_flattened(self):
        # Telegram has no nested blockquotes; a second '>' would be reserved.
        rendered = render("> outer\n> > inner", MARKDOWN_V2)
        self.assertNotIn(">>", rendered)
        assert_valid_markdown_v2(rendered)

    def test_touching_markers_are_separated(self):
        # '_italic___underline__' would be read greedily and cross entities.
        rendered = render("*`*______", MARKDOWN_V2)
        self.assertIn("\r", rendered)
        assert_valid_markdown_v2(rendered)

    def test_nested_links_keep_only_the_outer_target(self):
        rendered = render("[outer [inner](https://in.example) end](https://out.example)", MARKDOWN_V2)
        self.assertEqual(rendered.count("]("), 1)
        self.assertIn("https://out.example", rendered)
        assert_valid_markdown_v2(rendered)
        self.assertEqual(
            render("[a [b](https://in.example)](https://out.example)", HTML).count("<a "), 1
        )


class HtmlValidityTests(unittest.TestCase):
    """The HTML fallback must be well-formed and use only Telegram's tags."""

    ALLOWED = {"b", "i", "u", "s", "a", "code", "pre", "span", "blockquote", "tg-spoiler"}

    def _check(self, markup: str) -> None:
        stack = []

        class Parser(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag not in HtmlValidityTests.ALLOWED:
                    raise AssertionError("unexpected tag <%s> in %r" % (tag, markup))
                stack.append(tag)

            def handle_endtag(self, tag):
                if not stack or stack[-1] != tag:
                    raise AssertionError("mismatched </%s> in %r" % (tag, markup))
                stack.pop()

        parser = Parser(convert_charrefs=True)
        parser.feed(markup)
        parser.close()
        if stack:
            raise AssertionError("unclosed tags %s in %r" % (stack, markup))

    def test_corpus_is_well_formed(self):
        for source in CORPUS:
            with self.subTest(source=source[:40]):
                self._check(render(source, HTML))

    def test_angle_brackets_in_text_are_escaped(self):
        rendered = render("compare a < b and <script>alert(1)</script>", HTML)
        self.assertNotIn("<script>", rendered)
        self._check(rendered)

    def test_fuzz(self):
        rng = random.Random(101)
        atoms = ["**", "*", "_", "~~", "||", "`", "```", "[", "](x)", "<", ">", "&",
                 "# ", "> ", "- ", "\n", "\n\n", "text", "a_b", "🤖", "|", "---"]
        for _ in range(2000):
            source = "".join(rng.choice(atoms) for _ in range(rng.randint(1, 20)))
            self._check(render(source, HTML))


if __name__ == "__main__":
    unittest.main()
