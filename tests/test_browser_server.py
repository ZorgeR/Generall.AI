"""The sandbox browser server (secure_container/scripts/browser_server.py).

Unit tests cover the pure helpers (keys, URL policy, private addresses, the sensitive-action guard,
content/state assembly and validation, read_page formatting, find ranking). Integration tests start
the real server as a subprocess against a local HTTP server and drive it over its unix socket; they
are skipped when Playwright or a Chromium binary is missing.
"""
from __future__ import annotations

import ast
import base64
import glob
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import importlib.util

import pytest

# Loaded by path: the script is standalone (it runs inside the sandbox container), and importing
# it through the secure_container package would pull in docker and the container manager.
SCRIPT = Path(__file__).resolve().parents[1] / "app" / "secure_container" / "scripts" / "browser_server.py"
_spec = importlib.util.spec_from_file_location("browser_server_under_test", SCRIPT)
bs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bs  # dataclasses look the module up while the class is created
_spec.loader.exec_module(bs)
LS, PS = chr(0x2028), chr(0x2029)


# =============================================================================================
# Unit tests: pure helpers
# =============================================================================================

def test_module_is_standalone_stdlib_plus_playwright_and_trafilatura():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    foreign = {r for r in roots if r not in sys.stdlib_module_names} - {"playwright", "trafilatura"}
    assert not foreign, f"the sandbox script may only use stdlib + playwright + trafilatura: {foreign}"


def test_module_imports_without_playwright():
    code = (
        "import sys, importlib.util\n"
        "sys.modules['playwright'] = None\n"
        "sys.modules['playwright.async_api'] = None\n"
        f"spec = importlib.util.spec_from_file_location('bsx', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['bsx'] = m; spec.loader.exec_module(m)\n"
        "print(m.map_key_name('enter'), m.normalize_navigation_url('example.com')[0])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["Enter", "https://example.com"]


@pytest.mark.parametrize("name, expected", [
    ("enter", "Enter"), ("Return", "Enter"), ("esc", "Escape"), ("Escape", "Escape"), ("tab", "Tab"),
    ("backspace", "Backspace"), ("delete", "Delete"), ("up", "ArrowUp"), ("ArrowDown", "ArrowDown"),
    ("left", "ArrowLeft"), ("right", "ArrowRight"), ("home", "Home"), ("end", "End"),
    ("pageup", "PageUp"), ("Page_Down", "PageDown"), ("space", "Space"), ("f1", "F1"), ("F12", "F12"),
    ("ctrl", "Control"), ("control", "Control"), ("alt", "Alt"), ("option", "Alt"), ("shift", "Shift"),
    ("super", "Meta"), ("cmd", "Meta"), ("meta", "Meta"), ("a", "a"), ("A", "A"), ("KeyA", "KeyA"),
])
def test_key_names_map_to_playwright(name, expected):
    assert bs.map_key_name(name) == expected


def test_key_sequences_and_chords():
    assert bs.parse_key_sequence("Enter") == [["Enter"]]
    assert bs.parse_key_sequence("ctrl+a") == [["Control", "a"]]
    assert bs.parse_key_sequence("Backspace Backspace") == [["Backspace"], ["Backspace"]]
    assert bs.parse_key_sequence("ctrl+shift+Tab  esc") == [["Control", "Shift", "Tab"], ["Escape"]]
    assert bs.parse_key_sequence("ctrl++") == [["Control", "+"]]
    assert bs.parse_key_sequence("+") == [["+"]]
    assert bs.chord_to_playwright(["Control", "a"]) == "Control+a"
    for bad in ("", "   ", "ctrl+", None, 5):
        with pytest.raises(bs.ActionError):
            bs.parse_key_sequence(bad)


def test_click_modifiers():
    assert bs.parse_modifiers(None) == []
    assert bs.parse_modifiers("shift") == ["Shift"]
    assert bs.parse_modifiers("ctrl+shift") == ["Control", "Shift"]
    assert bs.parse_modifiers("cmd") == ["Meta"]
    with pytest.raises(bs.ActionError):
        bs.parse_modifiers("ctrl+x")
    with pytest.raises(bs.ActionError):
        bs.parse_modifiers(["shift"])


@pytest.mark.parametrize("raw, expected", [
    ("example.com", "https://example.com"),
    ("example.com/docs?q=1", "https://example.com/docs?q=1"),
    ("http://example.com/a", "http://example.com/a"),
    ("HTTPS://Example.com", "HTTPS://Example.com"),
    ("localhost:8080/x", "https://localhost:8080/x"),
    ("  www.example.org  ", "https://www.example.org"),
])
def test_navigation_urls_are_normalized(raw, expected):
    assert bs.normalize_navigation_url(raw) == (expected, None)


@pytest.mark.parametrize("raw", [
    "file:///etc/passwd", "javascript:alert(1)", "data:text/html,<b>hi</b>", "ftp://example.com/x",
    "chrome://settings", "about:blank", "mailto:a@example.com", "view-source:https://example.com",
    "ws://example.com/socket", "blob:https://example.com/1", "foo://bar",
])
def test_non_http_schemes_are_refused(raw):
    assert bs.normalize_navigation_url(raw) == (None, bs.NAV_SCHEME_ERROR)
    assert bs.NAV_SCHEME_ERROR == "Error: Navigation refused. Only http and https URLs are allowed."


def test_navigation_url_errors():
    assert bs.normalize_navigation_url("")[1].startswith("Error:")
    assert bs.normalize_navigation_url("https://")[1] == "Error: The URL has no host."
    assert bs.normalize_navigation_url("exa\nmple.com")[1].startswith("Error:")


@pytest.mark.parametrize("address", [
    "127.0.0.1", "127.8.9.10", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.1",
    "169.254.169.254", "100.64.0.1", "0.0.0.0", "224.0.0.1", "239.255.255.250", "240.0.0.1",
    "255.255.255.255", "192.0.2.1", "::1", "::", "fe80::1", "fc00::1", "fd12:3456::1",
    "::ffff:127.0.0.1", "::ffff:10.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::1", "ff02::1",
    "fe80::1%eth0", "[::1]",
])
def test_private_addresses_are_detected(address):
    assert bs.is_private_address(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111",
                                     "::ffff:8.8.8.8", "64:ff9b::808:808", "not-an-ip"])
def test_public_addresses_pass(address):
    assert not bs.is_private_address(address)


def test_literal_hosts():
    assert bs.host_is_literal_private("localhost") is True
    assert bs.host_is_literal_private("LOCALHOST.") is True
    assert bs.host_is_literal_private("app.localhost") is True
    assert bs.host_is_literal_private("[::1]") is True
    assert bs.host_is_literal_private("10.0.0.5") is True
    assert bs.host_is_literal_private("8.8.8.8") is False
    assert bs.host_is_literal_private("example.com") is None


async def test_host_policy_resolves_caches_and_checks_schemes():
    calls = []

    async def resolver(host):
        calls.append(host)
        return {"evil.test": ["93.184.216.34", "10.0.0.1"], "good.test": ["93.184.216.34"]}[host]

    policy = bs.HostPolicy(resolver=resolver)
    assert await policy.is_blocked("evil.test") is True        # any private answer blocks
    assert await policy.is_blocked("good.test") is False
    assert await policy.is_blocked("GOOD.test.") is False
    assert calls == ["evil.test", "good.test"]                  # cached (60 s)
    assert await policy.is_blocked("127.0.0.1") is True          # literals never hit DNS
    assert await policy.is_blocked("unknown.test") is False      # unresolvable: the browser fails anyway
    assert await policy.check_url("https://good.test/x") is None
    assert await policy.check_url("https://evil.test/x") == "evil.test"
    assert await policy.check_url("ws://127.0.0.1:9000/s") == "127.0.0.1"
    assert await policy.check_url("wss://[::1]/s") == "::1"
    assert await policy.check_url("file:///etc/passwd") == "file:"
    assert await policy.check_url("chrome://settings") == "chrome:"
    assert await policy.check_url("data:text/plain,hi") is None
    assert await policy.check_url("blob:https://good.test/1") is None
    assert await bs.HostPolicy(allow_private=True).is_blocked("127.0.0.1") is False


@pytest.mark.parametrize("name", [
    "Place order", "Place your order", "Buy now", "Buy", "Pay", "Pay $10.00", "Purchase", "Checkout",
    "Proceed to checkout", "Check out", "Order now", "Confirm order", "Complete purchase", "Subscribe",
    "Donate", "Transfer", "Top up", "Top-up balance", "Send", "Send message", "Post", "Publish",
    "Tweet", "Reply", "Comment", "Delete", "Delete account", "Remove account", "Deactivate",
    "Close my account", "Change password", "Change email", "Оплатить", "Оплата картой", "Купить",
    "Оформить заказ", "Заказать", "Перевести", "Подписаться", "Опубликовать",
    "Ответить", "Удалить", "Сменить пароль", "Send code and pay",
    # missed before (review): orders, payments, account security, Russian money words
    "Submit order", "Submit payment", "Make payment", "Make a payment", "Update password",
    "Update email", "Disable 2FA", "Turn off two-factor authentication", "Remove passkey",
    "Изменить пароль", "Изменить email", "Изменить e-mail", "Пополнить", "Пополнить баланс",
    "Приобрести", "Пожертвовать", "Перевод", "Перевод на карту", "Подтвердить перевод",
    "Отправить сообщение", "Отправить комментарий", "Отключить двухфакторную аутентификацию",
])
def test_sensitive_names(name):
    assert bs.matches_sensitive(name), name


@pytest.mark.parametrize("name", [
    "", "Sign up", "Register", "Create account", "Log in", "Sign in", "Accept", "Accept all cookies",
    "I agree", "Agree", "Continue", "Next", "Search", "Send code", "Resend verification code",
    "Send me a login link", "Отправить код", "Мои заказы", "Payment methods", "Comments (3)", "Posts",
    "Settings", "Unsubscribe", "Add to cart", "Next step",
    # resending a one-time code is a login step (review: these were refused)
    "Send again", "Resend", "Отправить повторно", "Отправить ещё раз",
    # a bare "Отправить" is the generic Submit of Russian forms (registration, login, contact
    # details): by its name alone it is NOT consequential any more; in a message context it is
    # (test_bare_otpravit_is_sensitive_only_in_a_message_context)
    "Отправить",
    # "Перевод" meaning translation
    "Перевод страницы", "Перевод на английский", "Переводы",
    "Set password", "Reset password",
])
def test_not_sensitive_names(name):
    assert not bs.matches_sensitive(name), name


def test_bare_otpravit_is_sensitive_only_in_a_message_context():
    submit = {"role": "button", "name": "Отправить", "actionable": True}
    registration = dict(submit, ctxFields=["Email email", "Пароль password", "Имя name"], ctxPassword=True)
    contact = dict(submit, ctxFields=["Имя", "Телефон phone", "E-mail"])
    message_box = dict(submit, ctxFields=["Сообщение message"], ctxMultiline=True)
    chat = dict(submit, ctxFields=["Напишите сообщение"])
    comment = dict(submit, ctxFields=["Your comment"], ctxMultiline=True)
    feedback = dict(submit, ctxFields=["Ваш вопрос"], ctxMultiline=True)
    for desc in (submit, registration, contact):
        assert bs.sensitive_label(desc, "click") is None, desc
        assert bs.sensitive_label(desc, "space") is None, desc
    for desc in (message_box, chat, comment, feedback):
        assert bs.sensitive_label(desc, "click") == 'button "Отправить"', desc
    # Enter in a field: the form's "Отправить" counts only in a message form
    field = {"role": "textbox", "name": "Email", "textEntry": True, "submit": ["Отправить"]}
    assert bs.sensitive_label(dict(field, ctxFields=["Email", "Пароль"], ctxPassword=True), "enter") is None
    assert bs.sensitive_label(dict(field, ctxFields=["Имя", "Сообщение"], ctxMultiline=True), "enter") \
        == 'button "Отправить"'
    # English "Send" stays consequential by its name alone
    assert bs.sensitive_label({"role": "button", "name": "Send", "actionable": True}, "click") == 'button "Send"'


def test_secret_question_answer_is_not_a_composer():
    # review: "\bответ" made Enter in "Ответ на секретный вопрос" (a login step) a message send
    field = {"role": "textbox", "name": "Ответ на секретный вопрос", "textEntry": True}
    assert bs.sensitive_label(field, "enter") is None
    assert bs.sensitive_label(dict(field, name="Ответить", multiline=True), "enter")
    assert bs.sensitive_label(dict(field, name="Chat", multiline=True), "enter")


def test_clicking_into_a_text_field_is_never_consequential():
    # review: a "Post code" registration field and a "Write a reply" box were refused, and an
    # allowed click into the reply box spent the one-shot allowance on focusing it
    assert bs.sensitive_label({"role": "textbox", "name": "Post code", "textEntry": True}, "click") is None
    reply = {"role": "textbox", "name": "Write a reply", "textEntry": True, "multiline": True}
    assert bs.sensitive_label(reply, "click") is None
    assert bs.sensitive_label(reply, "space") is None          # typing a space
    assert bs.sensitive_label(reply, "enter") == 'Enter in the text field "Write a reply"'
    assert bs.sensitive_label(reply, "shift_enter") is None    # a new line
    single = {"role": "textbox", "name": "Qty", "textEntry": True, "submit": ["Place order"]}
    assert bs.sensitive_label(single, "shift_enter") == 'button "Place order"'


def test_space_on_a_control_is_a_click():
    assert bs.sensitive_label({"role": "button", "name": "Place order"}, "space") == 'button "Place order"'
    assert bs.sensitive_label({"role": "checkbox", "name": "I agree"}, "space") is None
    assert bs.sensitive_label(None, "space") is None


def test_activation_kinds():
    assert bs.activation_kind(["Enter"]) == "enter"
    assert bs.activation_kind(["NumpadEnter"]) == "enter"
    assert bs.activation_kind(["Control", "Enter"]) == "enter"
    assert bs.activation_kind(["Shift", "Enter"]) == "shift_enter"
    assert bs.activation_kind(["Space"]) == "space"
    assert bs.activation_kind(["Tab"]) is None
    assert bs.activation_kind(["Control", "a"]) is None


def test_frames_the_guard_cannot_look_into():
    opaque = {"role": "iframe", "name": "", "actionable": False, "opaqueFrame": True,
              "frame": {"url": "https://js.stripe.com/v3/elements-inner-card.html", "title": "", "ownerName": ""}}
    assert bs.sensitive_label(opaque, "click") == 'frame (looks like a payment form)'
    assert bs.sensitive_label(opaque, "enter") and bs.sensitive_label(opaque, "space")
    named = dict(opaque, name="Secure payment input frame", frame={"url": "https://x.example/f", "title": ""})
    assert bs.sensitive_label(named, "click") == 'frame "Secure payment input frame" (looks like a payment form)'
    video = dict(opaque, name="YouTube video player", frame={"url": "https://www.youtube.com/embed/x", "title": "Video"})
    assert bs.sensitive_label(video, "click") is None
    display = dict(opaque, frame={"url": "https://ads.example/display?slot=1", "title": "Ad"})
    assert bs.sensitive_label(display, "click") is None        # "display" is not "pay"
    # inside a payment frame the target decides; Enter in its field counts even without a form
    card = {"role": "textbox", "name": "Card number", "textEntry": True,
            "frame": {"url": "https://checkout.example/card", "title": "Checkout"}}
    assert bs.sensitive_label(card, "click") is None
    assert bs.sensitive_label(card, "enter") == 'Enter in the text field "Card number" of a payment form'


def test_sensitive_label_rules():
    button = {"role": "button", "name": "Place order"}
    assert bs.sensitive_label(button, "click") == 'button "Place order"'
    assert bs.sensitive_label({"role": "button", "name": "Next"}, "click") is None
    assert bs.sensitive_label(None, "click") is None
    # form_input: text fields never, other controls by name
    assert bs.sensitive_label({"role": "textbox", "name": "Comment", "textEntry": True}, "form_input") is None
    assert bs.sensitive_label({"role": "checkbox", "name": "Subscribe to the newsletter"}, "form_input") \
        == 'checkbox "Subscribe to the newsletter"'
    # Enter in a field submits its form: the form's submit buttons decide
    field = {"role": "textbox", "name": "Quantity", "textEntry": True, "submit": ["Place order"]}
    assert bs.sensitive_label(field, "enter") == 'button "Place order"'
    assert bs.sensitive_label(dict(field, submit=["Search"]), "enter") is None
    # a message composer
    composer = {"role": "textbox", "name": "", "placeholder": "Write a message", "textEntry": True, "multiline": True}
    assert bs.sensitive_label(composer, "enter") == 'Enter in the text field "Write a message"'
    assert bs.sensitive_label(dict(composer, placeholder="Напишите сообщение"), "enter")
    assert bs.sensitive_label(dict(composer, placeholder="Search messages"), "enter") is None
    assert bs.sensitive_label(dict(composer, inSearch=True), "enter") is None
    assert bs.sensitive_label(dict(composer, placeholder="Notes"), "enter") is None
    # a textarea's Enter adds a line, it does not submit the form
    assert bs.sensitive_label({"role": "textbox", "name": "Notes", "textEntry": True, "multiline": True,
                               "submit": ["Place order"]}, "enter") is None
    # plain text, a heading or an empty area is not an action, whatever it says
    assert bs.sensitive_label({"role": "heading", "name": "Delete your account", "actionable": False}, "click") is None
    assert bs.sensitive_label({"role": "generic", "name": "", "actionable": False}, "enter") is None
    # Enter on a focused button activates it
    assert bs.sensitive_label({"role": "button", "name": "Delete"}, "enter") == 'button "Delete"'
    assert bs.confirmation_error('button "Place order"') == (
        'Error: This looks like a consequential action (button "Place order"). '
        "Ask the user first with browser_confirm, then repeat the action.")


def test_clean_field_and_urls():
    assert bs.clean_field("a\nb\tc" + LS + "d" + PS + "e\x00f") == "a b c d e f"
    assert bs.clean_field("x" * 5000) == "x" * 4096
    # review: a cut that ends on a space made validate_content reject every tab-member result
    long_title = "a" * 4095 + " tail"
    assert bs.clean_field(long_title) == "a" * 4095
    assert bs.clean_field(bs.clean_field(long_title)) == bs.clean_field(long_title)
    entries = bs.make_tab_entries([("tab-1", long_title, "https://example.com/" + "b" * 4075 + " c")], "tab-1")
    for member in ("list_tabs", "new_tab"):
        changes = [{"type": "tab_opened", "tab_id": "tab-1"}] if member == "new_tab" else []
        content, _ = bs.assemble_content(member, entries=entries, changes=changes)
        assert bs.validate_content(member, content) == []
    assert bs.clean_field(None) == ""
    assert bs.display_url("https://user:secret@example.com/a?b=1#c") == "https://example.com/a?b=1#c"
    assert bs.display_url("about:blank") == "about:blank"
    assert bs.download_url("https://u:p@cdn.example.com/f.pdf?X-Amz-Signature=abc") == "https://cdn.example.com/f.pdf"
    assert bs.clean_text("  a \n  b ", 10) == "a b"
    assert bs.clean_text("abcdefghijkl", 5) == "abcd" + chr(0x2026)
    assert bs.quote('say "hi" \\ ok') == '"say \\"hi\\" \\\\ ok"'


def test_short_error_strips_playwright_prefixes():
    err = Exception("Page.goto: net::ERR_NAME_NOT_RESOLVED at https://x.invalid/\nCall log:\n  - navigating")
    assert bs.short_error(err) == "net::ERR_NAME_NOT_RESOLVED at https://x.invalid/"


def test_download_names(tmp_path):
    assert bs.sanitize_filename("../../etc/passwd") == "passwd"
    assert bs.sanitize_filename("a/b\\c.txt") == "c.txt"
    assert bs.sanitize_filename("") == "download"
    assert bs.sanitize_filename("..") == "download"
    assert bs.sanitize_filename("re port;rm -rf $HOME.pdf") == "re port_rm -rf _HOME.pdf"
    assert bs.sanitize_filename("отчёт 2026.xlsx") == "отчёт 2026.xlsx"
    long_name = bs.sanitize_filename("x" * 300 + ".pdf")
    assert len(long_name) <= 120 and long_name.endswith(".pdf")
    (tmp_path / "report.txt").write_text("1")
    (tmp_path / "report-2.txt").write_text("2")
    assert bs.unique_name(str(tmp_path), "report.txt") == "report-3.txt"
    assert bs.unique_name(str(tmp_path), "new.txt", reserved={"new.txt"}) == "new-2.txt"
    assert bs.unique_name(str(tmp_path), "other.txt") == "other.txt"


def test_input_validation_helpers():
    assert bs.parse_coordinate({"type": "coordinate", "x": 10, "y": 20}, 1280, 800) == (10, 20)
    assert bs.parse_coordinate({"type": "coordinate", "x": 10.6, "y": "20"}, 1280, 800) == (11, 20)
    with pytest.raises(bs.ActionError, match=r"outside the viewport \(1280x800\)"):
        bs.parse_coordinate({"type": "coordinate", "x": 1280, "y": 0}, 1280, 800)
    with pytest.raises(bs.ActionError):
        bs.parse_coordinate({"type": "ref", "ref": "ref_1"}, 1280, 800)
    with pytest.raises(bs.ActionError):
        bs.parse_coordinate({"type": "coordinate", "x": True, "y": 1}, 1280, 800)
    assert bs.parse_ref({"type": "ref", "ref": "ref_12"}) == "ref_12"
    for bad in ({"type": "ref", "ref": "12"}, {"type": "ref", "ref": "ref_0"}, {"type": "coordinate"}, None):
        with pytest.raises(bs.ActionError):
            bs.parse_ref(bad)
    assert bs.stale_ref_error("ref_3") == ("Error: ref_3 is stale or not found on the current page. "
                                           "Re-read the page to get fresh references.")
    assert bs.check_tab_id("tab-1") == "tab-1"
    for bad in ("", None, 3, "tab\n1", "x" * 5000):
        with pytest.raises(bs.ActionError):
            bs.check_tab_id(bad)
    assert bs.parse_viewport("1280x800") == (1280, 800)
    with pytest.raises(ValueError):
        bs.parse_viewport("big")


def _entries(*tabs, active=None):
    return bs.make_tab_entries([(t, f"Title {t}", f"https://example.com/{t}") for t in tabs], active)


def test_tab_entries_have_exactly_one_active():
    entries = _entries("tab-1", "tab-2", active="tab-2")
    assert [e["active"] for e in entries] == [False, True]
    assert [e["active"] for e in _entries("tab-1", "tab-2", active="gone")] == [False, True]
    assert bs.make_tab_entries([("tab-1", "a\nb", "u")], "tab-1")[0]["title"] == "a b"
    assert len(bs.make_tab_entries([("tab-1", "", ""), ("tab-1", "", "")], None)) == 1
    assert bs.make_tab_entries([], None) == []


def test_download_events_merge_to_one_per_id():
    events = [
        {"type": "download_started", "download_id": "a", "url": "u"},
        {"type": "download_started", "download_id": "b", "url": "v"},
        {"type": "download_completed", "download_id": "a", "url": "u", "path": "downloads/x", "size_bytes": 3},
    ]
    merged = bs.merge_download_events(events)
    assert [(e["download_id"], e["type"]) for e in merged] == [("a", "download_completed"), ("b", "download_started")]


def test_tab_management_results_are_exactly_one_state_block():
    entries = _entries("tab-1", "tab-2", active="tab-2")
    for member in ("list_tabs", "switch_tab", "close_tab"):
        content, _ = bs.assemble_content(member, entries=entries, text="ignored")
        assert [b["type"] for b in content] == ["browser_state"]
        assert "state_changes" not in content[0]
        assert bs.validate_content(member, content) == []
    content, _ = bs.assemble_content("new_tab", entries=entries, changes=[{"type": "tab_opened", "tab_id": "tab-2"}])
    assert content[0]["state_changes"] == [{"type": "tab_opened", "tab_id": "tab-2"}]
    assert bs.validate_content("new_tab", content) == []
    wrong, _ = bs.assemble_content("new_tab", entries=entries, changes=[{"type": "tab_opened", "tab_id": "tab-1"}])
    assert bs.validate_content("new_tab", wrong)
    closed_last, _ = bs.assemble_content("close_tab", entries=[])
    assert closed_last == [{"type": "browser_state", "tabs": []}]
    assert bs.validate_content("close_tab", closed_last) == []


def test_other_members_report_state_only_when_it_changed():
    entries = _entries("tab-1", active="tab-1")
    first, sig = bs.assemble_content("left_click", entries=entries, text="Clicked.")
    assert [b["type"] for b in first] == ["text", "browser_state"]
    again, sig2 = bs.assemble_content("left_click", entries=entries, text="Clicked.", previous_signature=sig)
    assert again == [{"type": "text", "text": "Clicked."}] and sig2 == sig
    shot, _ = bs.assemble_content("screenshot", entries=entries, image_b64="AAAA", previous_signature=sig)
    assert shot == [bs.image_block("AAAA")]
    retitled = bs.make_tab_entries([("tab-1", "New title", "https://example.com/tab-1")], "tab-1")
    changed, _ = bs.assemble_content("left_click", entries=retitled, text="Clicked.", previous_signature=sig)
    assert changed[-1]["type"] == "browser_state"
    with_download, _ = bs.assemble_content(
        "wait", entries=entries, text="Waited 1 seconds.", previous_signature=sig,
        changes=[{"type": "download_completed", "download_id": "d", "url": "u", "path": "downloads/f", "size_bytes": 1}])
    assert with_download[-1]["state_changes"][0]["type"] == "download_completed"
    empty, _ = bs.assemble_content("wait", entries=entries, changes=[], previous_signature=sig)
    assert empty == [{"type": "text", "text": "Done."}]
    for content in (first, again, shot, changed, with_download):
        assert bs.validate_content("left_click", content) == []


def test_state_block_drops_tab_opened_for_unknown_tabs_and_never_sends_empty_changes():
    entries = _entries("tab-1", active="tab-1")
    block = bs.make_state_block(entries, [{"type": "tab_opened", "tab_id": "tab-9"}])
    assert "state_changes" not in block
    assert "state_changes" not in bs.make_state_block(entries, [])


def test_validate_content_catches_what_the_api_rejects():
    good = bs.make_state_block(_entries("tab-1", active="tab-1"))
    assert bs.validate_content("navigate", [bs.text_block("x"), good]) == []
    assert bs.validate_content("navigate", [{"type": "document"}])
    assert bs.validate_content("navigate", [bs.text_block("x"), good, good])
    assert bs.validate_content("navigate", [dict(good, state_changes=[])])
    two_active = {"type": "browser_state", "tabs": [dict(t, active=True) for t in _entries("tab-1", "tab-2")]}
    assert bs.validate_content("navigate", [two_active])
    dirty = {"type": "browser_state", "tabs": [{"tab_id": "tab-1", "title": "a\nb", "url": "", "active": True}]}
    assert bs.validate_content("navigate", [dirty])
    assert bs.validate_content("list_tabs", [bs.text_block("x"), good])
    assert bs.validate_content("navigate", [])


def test_read_page_line_format():
    assert bs.format_node_line({"level": 0, "role": "link", "name": "Documentation", "ref": "ref_1"}) \
        == 'link "Documentation" [ref_1]'
    assert bs.format_node_line({"level": 2, "role": "heading", "name": "Intro", "ref": "ref_2",
                                "props": {"level": 2}}) == '    heading "Intro" [ref_2] level=2'
    assert bs.format_node_line({"level": 1, "role": "textbox", "name": "Email", "ref": "ref_3",
                                "props": {"type": "email", "value": "a@b.c", "placeholder": "you@example.com",
                                          "focused": True}}) \
        == '  textbox "Email" [ref_3] type=email value="a@b.c" placeholder="you@example.com" focused'
    assert bs.format_node_line({"level": 0, "role": "checkbox", "name": "Remember", "ref": "ref_4",
                                "props": {"checked": True, "disabled": True}}) == 'checkbox "Remember" [ref_4] checked disabled'
    assert bs.format_node_line({"level": 0, "role": "button", "name": "", "ref": "ref_5",
                                "props": {"expanded": False}}) == "button [ref_5] collapsed"
    assert bs.format_node_line({"level": 1, "role": "text", "name": 'say "hi"\nnow'}) == '  text "say \\"hi\\" now"'
    assert bs.format_node_line({"level": 3, "role": "link", "name": "x", "ref": "ref_6", "props": {"href": "/a"}},
                               indent=False) == 'link "x" [ref_6] href="/a"'


def test_render_tree_caps_and_says_so():
    nodes = [{"level": 0, "role": "link", "name": f"Link {i}", "ref": f"ref_{i}"} for i in range(1, 2001)]
    text = bs.render_tree(nodes, cap=1000)
    assert len(text) <= 1000 + len(bs.truncation_notice(1000)) + 1
    assert text.endswith(bs.truncation_notice(1000))
    assert "Output truncated at 1,000 characters" in text
    small = bs.render_tree(nodes[:2])
    assert small == 'link "Link 1" [ref_1]\nlink "Link 2" [ref_2]'
    assert bs.render_tree(nodes[:1], truncated=True).endswith(bs.truncation_notice(bs.READ_PAGE_CAP))


def test_find_ranking():
    candidates = [
        {"role": "link", "name": "Search tips"},
        {"role": "searchbox", "name": "Search", "inView": True},
        {"role": "button", "name": "Place order"},
        {"role": "textbox", "name": "Email", "placeholder": "you@example.com"},
        {"role": "heading", "name": "Your order"},
    ]
    assert bs.rank_find_candidates("search field", candidates)[0] == 1
    assert bs.rank_find_candidates("place order button", candidates)[0] == 2
    assert bs.rank_find_candidates("email input", candidates)[0] == 3
    assert bs.rank_find_candidates("nothing like this", candidates) == []
    many = [{"role": "button", "name": f"Button {i}"} for i in range(50)]
    assert len(bs.rank_find_candidates("button", many)) == bs.FIND_LIMIT


def test_page_text_choice_and_cap():
    article = "word " * 200
    assert bs.choose_page_text(article, "nav " + article + " footer") == article.strip()
    assert bs.choose_page_text("Hi", "Login form with many fields " * 20) == ("Login form with many fields " * 20).strip()
    assert bs.choose_page_text(None, "body") == "body"
    assert bs.choose_page_text("only extracted", "") == "only extracted"
    capped = bs.cap_text("x" * 70_000)
    assert capped.endswith("[Text truncated at 60,000 characters.]") and len(capped) < 60_100


def test_user_agent_is_a_desktop_chrome_without_headless():
    ua = bs.default_user_agent()
    assert "Chrome/" in ua and "Headless" not in ua and "X11; Linux x86_64" in ua


def test_member_tables_match_the_sdk_toolset():
    anthropic_types = pytest.importorskip("anthropic.types")
    names = getattr(anthropic_types, "BrowserMemberName", None)
    if names is None:
        pytest.skip("this anthropic SDK has no browser toolset types")
    sdk = set(names.__args__)
    assert set(bs.MEMBERS) | set(bs.DISABLED_MEMBERS) == sdk
    assert not set(bs.MEMBERS) & set(bs.DISABLED_MEMBERS)
    drag = anthropic_types.BrowserLeftClickDragInput.model_fields
    assert drag["from_"].alias == "from"  # the server reads "from" (and tolerates "from_")


# ---------------------------------------------------------------------------------------------
# The server object without a browser: request filter, websocket filter, dispatcher
# ---------------------------------------------------------------------------------------------

def _bare_server(tmp_path, allow_private=False):
    args = bs.parse_args(["--socket", str(tmp_path / "s.sock"), "--downloads", str(tmp_path / "dl"),
                          "--idle-seconds", "60"] + (["--allow-private"] if allow_private else []))
    server = bs.BrowserServer(args)

    async def resolver(host):
        return {"intranet.test": ["10.0.0.7"], "public.test": ["93.184.216.34"]}[host]

    server.policy._resolver = resolver
    return server


class _FakeRoute:
    def __init__(self):
        self.calls = []

    async def abort(self, code=None):
        self.calls.append(("abort", code))

    async def fallback(self):
        self.calls.append(("fallback", None))


class _FakeRequest:
    def __init__(self, url):
        self.url = url

    def is_navigation_request(self):
        return False


class _FakeWebSocket:
    def __init__(self, url):
        self.url = url
        self.calls = []

    async def close(self, code=None, reason=None):
        self.calls.append(("close", code))

    def connect_to_server(self):
        self.calls.append(("connect", None))


@pytest.mark.parametrize("url, verdict", [
    ("https://public.test/app.js", "fallback"),
    ("https://intranet.test/admin", "abort"),
    ("http://127.0.0.1:8081/bot", "abort"),
    ("http://169.254.169.254/latest/meta-data/", "abort"),
    ("http://[::1]/", "abort"),
    ("http://localhost:3000/", "abort"),
    ("file:///etc/passwd", "abort"),
    ("data:image/png;base64,AAAA", "fallback"),
])
async def test_request_filter(tmp_path, url, verdict):
    server = _bare_server(tmp_path)
    route = _FakeRoute()
    await server._route(route, _FakeRequest(url))
    assert route.calls[0][0] == verdict
    if verdict == "abort":
        assert route.calls[0][1] == "blockedbyclient"


async def test_request_filter_is_off_with_allow_private(tmp_path):
    server = _bare_server(tmp_path, allow_private=True)
    route = _FakeRoute()
    await server._route(route, _FakeRequest("http://127.0.0.1:8081/"))
    assert route.calls == [("fallback", None)]


async def test_websocket_filter(tmp_path):
    server = _bare_server(tmp_path)
    blocked, allowed = _FakeWebSocket("ws://10.1.1.1:9000/x"), _FakeWebSocket("wss://public.test/live")
    await server._route_ws(blocked)
    await server._route_ws(allowed)
    assert blocked.calls == [("close", 1008)]
    assert allowed.calls == [("connect", None)]


async def test_dispatcher_without_a_browser(tmp_path):
    server = _bare_server(tmp_path)
    assert await server.dispatch({"action": "_ping"}) == {"ok": True}
    before = server.last_request
    time.sleep(0.01)
    assert await server.dispatch({"action": "_touch"}) == {"ok": True}
    assert server.last_request > before
    status = await server.dispatch({"action": "_status"})
    assert status["ok"] and status["tabs"] == [] and status["started"] <= status["last_request"]
    peek = await server.dispatch({"action": "_peek"})
    assert peek["ok"] is False and peek["error"] == "Error: No tab is open."
    shot = await server.dispatch({"action": "screenshot", "input": {}})
    assert shot == {"ok": False, "error": "Error: No tab is open. Use new_tab or navigate first.",
                    "needs_confirmation": False, "meta": {"tab_id": None, "title": "", "url": ""}}
    assert (await server.dispatch({"action": "javascript_exec", "input": {"text": "1"}}))["error"] == \
        "Error: javascript_exec is not enabled in this environment."
    assert (await server.dispatch({"action": "nope"}))["error"] == 'Error: Unknown browser action "nope".'
    assert (await server.dispatch({"action": "wait", "input": []}))["error"] == "Error: input must be an object."
    assert (await server.dispatch({}))["ok"] is False
    assert (await server.dispatch({"action": "navigate", "input": {"url": "file:///x"}}))["error"] == \
        bs.NAV_SCHEME_ERROR
    assert server._in_flight == 0


# ---------------------------------------------------------------------------------------------
# The egress proxy (in process: a fake resolver, local upstream servers)
# ---------------------------------------------------------------------------------------------

def test_proxy_request_parsing():
    req = bs.parse_proxy_request(b"GET http://shop.example:8080/a/b?q=1 HTTP/1.1\r\nHost: shop.example:8080\r\n"
                                 b"Proxy-Connection: keep-alive\r\nProxy-Authorization: x\r\nCookie: a=1\r\n\r\n")
    assert (req.method, req.host, req.port, req.connect) == ("GET", "shop.example", 8080, False)
    head = req.head.decode()
    assert head.startswith("GET /a/b?q=1 HTTP/1.1\r\n") and head.endswith("Connection: close\r\n\r\n")
    assert "Proxy-" not in head and "Cookie: a=1" in head and "Host: shop.example:8080" in head
    tunnel = bs.parse_proxy_request(b"CONNECT [2606:4700::1]:443 HTTP/1.1\r\nHost: x\r\n\r\n")
    assert (tunnel.host, tunnel.port, tunnel.connect) == ("2606:4700::1", 443, True)
    bare = bs.parse_proxy_request(b"GET http://a.example HTTP/1.0\r\n\r\n")
    assert bare.port == 80 and bare.head.decode().startswith("GET / HTTP/1.0\r\nHost: a.example\r\n")
    upgrade = bs.parse_proxy_request(b"GET http://a.example/ws HTTP/1.1\r\nHost: a.example\r\n"
                                     b"Connection: Upgrade\r\nUpgrade: websocket\r\n\r\n")
    assert upgrade.head.decode().endswith("Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
    for bad in (b"GET /local HTTP/1.1\r\n\r\n",                      # not a proxy request
                b"GET https://a.example/ HTTP/1.1\r\n\r\n",          # https goes through CONNECT
                b"GET ftp://a.example/ HTTP/1.1\r\n\r\n",
                b"CONNECT a.example HTTP/1.1\r\n\r\n",               # no port
                b"GET http://u:p@a.example/ HTTP/1.1\r\n\r\n",
                b"GET http://a.example/ HTTP/2\r\n\r\n",
                b"GET http://a.example/ HTTP/1.1\r\n folded\r\n\r\n",
                b"GET http://a.example:99999/ HTTP/1.1\r\n\r\n"):
        with pytest.raises(bs.ProxyBadRequest):
            bs.parse_proxy_request(bad)


def test_proxy_response_heads():
    head, status = bs.rewrite_response_head(b"HTTP/1.1 200 OK\r\nConnection: keep-alive\r\nKeep-Alive: timeout=5\r\n"
                                            b"Content-Length: 3\r\n\r\n")
    assert status == 200
    assert head == b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\nProxy-Connection: close\r\n\r\n"
    early = b"HTTP/1.1 103 Early Hints\r\nLink: </a.css>; rel=preload\r\n\r\n"
    assert bs.rewrite_response_head(early) == (early, 103)
    assert bs.rewrite_response_head(b"HTTP/1.0 302\r\nLocation: /x\r\n\r\n")[1] == 302
    with pytest.raises(bs.ProxyBadRequest):
        bs.rewrite_response_head(b"garbage\r\n\r\n")
    # long header names (CSP report-only: 35 characters) and odd lines pass; only Connection goes
    long = (b"HTTP/1.1 200 OK\r\nContent-Security-Policy-Report-Only: default-src 'self'\r\n"
            b"Cross-Origin-Embedder-Policy-Report-Only: require-corp\r\nX-Folded: a\r\n b\r\n"
            b"Connection: keep-alive\r\n folded-continuation\r\n\r\n")
    assert bs.rewrite_response_head(long)[0] == (
        b"HTTP/1.1 200 OK\r\nContent-Security-Policy-Report-Only: default-src 'self'\r\n"
        b"Cross-Origin-Embedder-Policy-Report-Only: require-corp\r\nX-Folded: a\r\n b\r\n"
        b"Connection: close\r\nProxy-Connection: close\r\n\r\n")
    preflight = bs.parse_proxy_request(b"OPTIONS http://a.example/api HTTP/1.1\r\nHost: a.example\r\n"
                                       b"Access-Control-Request-Private-Network: true\r\n\r\n")
    assert b"Access-Control-Request-Private-Network: true" in preflight.head
    refusal = bs.proxy_error_response(403, "Blocked by the sandbox network policy: 10.0.0.1 is a private or local address.")
    assert refusal.startswith(b"HTTP/1.1 403 Forbidden\r\n") and refusal.endswith(b"address.\n")


async def _proxy_exchange(port: int, data: bytes, read_until_close: bool = True) -> bytes:
    import asyncio
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(data)
    await writer.drain()
    out = await asyncio.wait_for(reader.read(-1), 10) if read_until_close else b""
    writer.close()
    return out


async def test_egress_proxy_checks_every_host_and_connects_to_the_checked_address():
    import asyncio
    seen: list[bytes] = []

    async def origin(reader, writer):  # a keep-alive origin server: the proxy must close anyway
        head = await reader.readuntil(b"\r\n\r\n")
        seen.append(head)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: keep-alive\r\n\r\nhello")
        await writer.drain()

    async def echo(reader, writer):
        while data := await reader.read(100):
            writer.write(data.upper())
            await writer.drain()
        writer.close()

    web = await asyncio.start_server(origin, "127.0.0.1", 0)
    raw = await asyncio.start_server(echo, "127.0.0.1", 0)
    web_port, raw_port = web.sockets[0].getsockname()[1], raw.sockets[0].getsockname()[1]
    lookups = []

    async def resolver(host):
        lookups.append(host)
        return {"public.test": ["93.184.216.34"], "rebind.test": ["93.184.216.34", "127.0.0.1"],
                "intranet.test": ["10.0.0.7"]}[host]

    refused = []
    proxy = bs.EgressProxy(bs.HostPolicy(resolver=resolver), on_refused=refused.append)
    connected = []
    original = bs.EgressProxy._connect_one

    async def connect_one(self, address, port):  # the checked public address, mapped to the local server
        connected.append(address)
        return await original(self, "127.0.0.1" if address == "93.184.216.34" else address, port)

    proxy._connect_one = connect_one.__get__(proxy)
    port = await proxy.start()
    try:
        ok = await _proxy_exchange(port, f"GET http://public.test:{web_port}/page?x=1 HTTP/1.1\r\n"
                                         f"Host: public.test:{web_port}\r\nProxy-Connection: keep-alive\r\n\r\n".encode())
        assert ok.startswith(b"HTTP/1.1 200 OK\r\n") and b"Connection: close\r\n" in ok and ok.endswith(b"hello")
        assert seen[0].startswith(b"GET /page?x=1 HTTP/1.1\r\n") and b"Connection: close" in seen[0]
        assert connected == ["93.184.216.34"]             # the address that was checked, not a name
        for target in ("127.0.0.1", "169.254.169.254", "[::1]", "localhost", "intranet.test", "rebind.test"):
            out = await _proxy_exchange(port, f"GET http://{target}:{web_port}/secret HTTP/1.1\r\n\r\n".encode())
            assert out.startswith(b"HTTP/1.1 403 Forbidden"), (target, out)
            assert b"private or local address" in out
            out = await _proxy_exchange(port, f"CONNECT {target}:443 HTTP/1.1\r\n\r\n".encode())
            assert out.startswith(b"HTTP/1.1 403 Forbidden"), (target, out)
        assert len(seen) == 1 and connected == ["93.184.216.34"]   # nothing private was ever contacted
        assert {r.reason for r in refused} == {"private"}
        # DNS failure: refused (fail closed), not passed on
        out = await _proxy_exchange(port, b"GET http://unknown.test/ HTTP/1.1\r\n\r\n")
        assert out.startswith(b"HTTP/1.1 502 Bad Gateway") and b"could not be resolved" in out
        assert refused[-1].reason == "dns"
        # a tunnel (https, wss): bytes both ways once the host passed
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"CONNECT public.test:{raw_port} HTTP/1.1\r\nHost: public.test:{raw_port}\r\n\r\n".encode())
        await writer.drain()
        assert await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5) == b"HTTP/1.1 200 Connection established\r\n\r\n"
        writer.write(b"ping")
        await writer.drain()
        assert await asyncio.wait_for(reader.readexactly(4), 5) == b"PING"
        writer.close()
        # not a proxy request; a head that never ends within the limit
        assert (await _proxy_exchange(port, b"GET / HTTP/1.1\r\n\r\n")).startswith(b"HTTP/1.1 400")
        big = b"GET http://public.test/ HTTP/1.1\r\nX: " + b"a" * (bs.PROXY_HEAD_LIMIT + 10) + b"\r\n\r\n"
        assert (await _proxy_exchange(port, big)).startswith(b"HTTP/1.1 400")
        assert lookups.count("public.test") == 1        # cached: one lookup per host
    finally:
        proxy.close()
        web.close()
        raw.close()


async def test_egress_proxy_with_allow_private_reaches_local_servers():
    import asyncio

    async def origin(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.0 200 OK\r\n\r\nlocal")
        await writer.drain()
        writer.close()

    web = await asyncio.start_server(origin, "127.0.0.1", 0)
    proxy = bs.EgressProxy(bs.HostPolicy(allow_private=True))
    port = await proxy.start()
    try:
        out = await _proxy_exchange(port, f"GET http://127.0.0.1:{web.sockets[0].getsockname()[1]}/ HTTP/1.1\r\n\r\n".encode())
        assert out.startswith(b"HTTP/1.0 200 OK") and out.endswith(b"local")
    finally:
        proxy.close()
        web.close()


def test_the_browser_is_launched_through_the_proxy_without_loopback_bypass():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"bypass": "<-loopback>"' in source and 'proxy={"server": f"http://127.0.0.1:{port}"' in source
    assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in bs.CHROMIUM_ARGS


# =============================================================================================
# Integration tests: the real server, a local site, the unix socket
# =============================================================================================

def _find_chrome() -> str | None:
    override = os.environ.get("BROWSER_SERVER_TEST_CHROME")
    if override and os.path.exists(override):
        return override
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "", "/opt/pw-browsers",
             os.path.expanduser("~/.cache/ms-playwright")]
    found = []
    for root in roots:
        if root:
            found += glob.glob(os.path.join(root, "chromium-*", "chrome-linux", "chrome"))
    found = [f for f in found if os.access(f, os.X_OK)]
    return sorted(found)[-1] if found else None


def _playwright_available() -> bool:
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


CHROME = _find_chrome()
needs_browser = pytest.mark.skipif(not (_playwright_available() and CHROME),
                                   reason="Playwright or a Chromium binary is missing")

DOWNLOAD_BODY = b"quarterly numbers\n" * 10

FORM_PAGE = """<!doctype html><html><head><title>Form page</title></head>
<body style="margin:0; font: 16px sans-serif">
<h1>Order form</h1>
<p id="status">Nothing yet</p>
<form id="f" action="/result" method="get">
  <label for="name">Name</label> <input id="name" name="name">
  <label><input type="checkbox" id="agree" name="agree"> I agree to the terms</label>
  <select id="color" name="color"><option value="r">Red</option><option value="g">Green</option></select>
  <button type="button" id="next"
    onclick="document.getElementById('status').textContent='Next pressed'; document.title='Next pressed'">Next step</button>
</form>
<form role="search" action="/result" method="get"><input name="q" aria-label="Search query"><button>Search</button></form>
<form id="order" onsubmit="event.preventDefault(); document.title='Order placed';">
  <input name="qty" aria-label="Quantity"><button type="submit">Place order</button></form>
<textarea id="msg" aria-label="Chat box" placeholder="Write a message"></textarea>
<button id="abs" style="position:absolute; left:900px; top:600px; width:200px; height:60px"
  onclick="document.title='Coordinate click'">Abs button</button>
<div id="buy" style="cursor:pointer; position:absolute; left:900px; top:700px; width:200px; height:40px"><span>Buy now</span></div>
<script>document.getElementById('buy').addEventListener('click', () => { document.title = 'Bought'; });</script>
<a href="/other">Other page</a>
<a href="/other" target="_blank">Open in new tab</a>
<a href="/file.txt">Download report</a>
</body></html>"""

LONG_PAGE = ("<!doctype html><html><head><title>Long page</title></head><body style='margin:0'>"
             "<script>addEventListener('scroll', () => { document.title = 'y=' + Math.round(scrollY); });</script>"
             + "".join(f"<p style='height:80px;margin:0'>Paragraph {i} of the long page.</p>" for i in range(1, 61))
             + "<button onclick=\"document.title='bottom clicked'\">Bottom button</button></body></html>")

MANY_PAGE = ("<!doctype html><html><head><title>Many</title></head><body>"
             + "".join(f"<a href='/item/{i}'>Item number {i} with a fairly long link text</a><br>" for i in range(3000))
             + "</body></html>")

DIALOG_PAGE = """<!doctype html><html><head><title>Dialogs</title></head><body>
<button onclick="document.title = confirm('Delete this item permanently?') ? 'deleted' : 'kept'">Trash</button>
<button onclick="document.title = confirm('Leave this page?') ? 'left' : 'stayed'">Leave</button>
<button onclick="alert('Saved!'); document.title = 'alerted'">Save</button>
</body></html>"""

FRAMES_PAGE = """<!doctype html><html><head><title>Frames</title></head><body style="margin:0">
<div style="height:300px">spacer</div>
<iframe style="width:600px;height:200px;border:0;margin-left:200px" srcdoc="<html><body><p>Inside frame</p><button onclick='parent.document.title=&quot;frame clicked&quot;'>Frame button</button></body></html>"></iframe>
<x-card></x-card>
<script>
customElements.define('x-card', class extends HTMLElement { constructor() { super();
  const root = this.attachShadow({mode: 'open'});
  root.innerHTML = '<p>Shadow text</p><button id="b">Shadow button</button>';
  root.getElementById('b').onclick = () => { document.title = 'shadow clicked'; }; } });
</script></body></html>"""

DRAG_PAGE = """<!doctype html><html><head><title>Drag</title></head><body style="margin:0;height:700px">
<script>
let down = '';
addEventListener('mousedown', e => { down = 'down ' + e.clientX + ',' + e.clientY; document.title = down; });
addEventListener('mouseup', e => { document.title = down + ' up ' + e.clientX + ',' + e.clientY; });
addEventListener('keydown', e => { if (e.key !== 'Shift') document.title = 'key ' + e.key + (e.ctrlKey ? ' ctrl' : ''); });
</script></body></html>"""


# pages of the regression tests (path -> html, or a callable(handler) -> (body, status, headers)),
# and every path the site was asked for (to prove a private URL was never fetched)
EXTRA_PAGES: dict = {}
HITS: list[str] = []
POSTS: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="text/html; charset=utf-8", status=200, headers=()):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        HITS.append(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(min(length, 1 << 20))
        POSTS.append(self.path)
        self._send(f"<html><head><title>Order #{len(POSTS)} placed</title></head><body>placed</body></html>")

    def do_GET(self):  # noqa: N802
        HITS.append(self.path)
        parts = urlsplit(self.path)
        path, query = parts.path, parse_qs(parts.query)
        extra = EXTRA_PAGES.get(path)
        if extra is not None:
            if callable(extra):
                body, status, headers = extra(self)
                self._send(body, status=status, headers=headers)
            else:
                self._send(extra)
        elif path == "/form":
            self._send(FORM_PAGE)
        elif path == "/long":
            self._send(LONG_PAGE)
        elif path == "/many":
            self._send(MANY_PAGE)
        elif path == "/drag":
            self._send(DRAG_PAGE)
        elif path == "/dialog":
            self._send(DIALOG_PAGE)
        elif path == "/frames":
            self._send(FRAMES_PAGE)
        elif path == "/other":
            self._send("<html><head><title>Other page</title></head><body><h1>The other page</h1>"
                       "<p>Welcome to the other page of the test site.</p></body></html>")
        elif path == "/result":
            q = (query.get("q") or query.get("name") or [""])[0]
            self._send(f"<html><head><title>Result: {q}</title></head><body><p>You searched {q}</p></body></html>")
        elif path == "/file.txt":
            self._send(DOWNLOAD_BODY, "text/plain",
                       headers=[("Content-Disposition", 'attachment; filename="report.txt"')])
        elif path == "/setcookie":
            self._send("<html><head><title>Cookie set</title></head><body>cookie set</body></html>",
                       headers=[("Set-Cookie", "sid=abc123; Max-Age=3600; Path=/")])
        elif path == "/echo":
            cookie = self.headers.get("Cookie", "none")
            self._send(f"<html><head><title>Echo</title></head><body><p>Cookie header: {cookie}</p></body></html>")
        else:
            self._send("<html><head><title>Not found</title></head><body>nope</body></html>", status=404)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture(scope="module")
def site():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


class Client:
    def __init__(self, path: str, timeout: float = 90):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)
        self.file = self.sock.makefile("rwb")
        self.counter = 0

    def call(self, action: str, inp: dict | None = None, *, allow_sensitive: bool = False, check: bool = True) -> dict:
        self.counter += 1
        request = {"id": str(self.counter), "action": action, "input": inp or {}, "allow_sensitive": allow_sensitive}
        self.file.write((json.dumps(request) + "\n").encode())
        self.file.flush()
        line = self.file.readline()
        assert line, "the server closed the connection"
        response = json.loads(line)
        assert response.get("id") == str(self.counter)
        if check and action in bs.MEMBERS:
            if response["ok"]:
                assert bs.validate_content(action, response["content"]) == [], response["content"]
                assert isinstance(response["sensitive_done"], bool)
            else:
                assert "content" not in response
                assert response["error"].startswith("Error:")
                assert isinstance(response["needs_confirmation"], bool)
            assert set(response["meta"]) == {"tab_id", "title", "url"}
        return response

    def close(self):
        try:
            self.file.close()
            self.sock.close()
        except OSError:
            pass


# Starts the server with test patches: a fake DNS (BS_FAKE_DNS {host: [addresses]}) and a connect
# map for the egress proxy (BS_CONNECT_MAP {checked address: real address}), so that "public.test"
# passes the address check as 93.184.216.34 and still reaches the local test site; and
# BS_NO_FRAME_SESSIONS (cross-origin frames cannot be entered: the guard's last resort).
PATCHED_LAUNCHER = r"""
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("browser_server", os.environ["BS_SCRIPT"])
bs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bs
spec.loader.exec_module(bs)
fake = json.loads(os.environ.get("BS_FAKE_DNS") or "null")
connect_map = json.loads(os.environ.get("BS_CONNECT_MAP") or "{}")

async def resolver(host):
    if host in fake:
        return fake[host]
    raise OSError("unknown host " + host)

if fake is not None:
    original_init = bs.HostPolicy.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._resolver = resolver

    bs.HostPolicy.__init__ = init
original_connect = bs.EgressProxy._connect_one

async def connect_one(self, address, port):
    return await original_connect(self, connect_map.get(address, address), port)

bs.EgressProxy._connect_one = connect_one
if os.environ.get("BS_NO_FRAME_SESSIONS"):
    async def no_session(self, tab, frame_id):
        return None

    bs.BrowserServer._frame_session = no_session
sys.exit(bs.main(sys.argv[1:]))
"""


class ServerProcess:
    def __init__(self, workdir: str, *, profile: str | None = "profile", idle: float = 300,
                 allow_private: bool = True, fake_dns: dict | None = None, connect_map: dict | None = None,
                 no_frame_sessions: bool = False):
        self.workdir = workdir
        self.socket = os.path.join(workdir, "ctl", "browser.sock")
        self.downloads = os.path.join(workdir, "downloads")
        self.log_path = os.path.join(workdir, "server.log")
        profile_arg = "" if profile is None else os.path.join(workdir, profile)
        args = ["--socket", self.socket, "--profile", profile_arg,
                "--downloads", self.downloads, "--idle-seconds", str(idle), "--viewport", "1280x800",
                "--executable", CHROME]
        env = None
        if fake_dns is not None or no_frame_sessions:
            cmd = [sys.executable, "-c", PATCHED_LAUNCHER] + args
            env = dict(os.environ, BS_SCRIPT=str(SCRIPT), BS_FAKE_DNS=json.dumps(fake_dns),
                       BS_CONNECT_MAP=json.dumps(connect_map or {}),
                       BS_NO_FRAME_SESSIONS="1" if no_frame_sessions else "")
        else:
            cmd = [sys.executable, str(SCRIPT)] + args
        if allow_private:
            cmd.append("--allow-private")
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self._wait_ready()

    def _wait_ready(self, timeout: float = 40):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f"server exited early ({self.proc.returncode}):\n{self.logs()}")
            if os.path.exists(self.socket):
                try:
                    client = Client(self.socket, timeout=5)
                    ok = client.call("_ping")["ok"]
                    client.close()
                    if ok:
                        return
                except OSError:
                    pass
            time.sleep(0.1)
        raise AssertionError(f"server not ready:\n{self.logs()}")

    def logs(self) -> str:
        try:
            return Path(self.log_path).read_text(errors="replace")[-4000:]
        except OSError:
            return ""

    def client(self) -> Client:
        return Client(self.socket)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self):
        if self.alive():
            try:
                client = Client(self.socket, timeout=10)
                client.call("_shutdown")
                client.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log.close()


def _workdir() -> str:
    # unix socket paths are limited to ~107 bytes: keep it short
    return tempfile.mkdtemp(prefix="bsrv")


@pytest.fixture(scope="module")
def server():
    if not (_playwright_available() and CHROME):
        pytest.skip("Playwright or a Chromium binary is missing")
    workdir = _workdir()
    proc = ServerProcess(workdir)
    yield proc
    proc.stop()
    shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture
def browser(server):
    client = server.client()
    # every test starts with exactly one tab
    tabs = client.call("list_tabs")["content"][0]["tabs"]
    for tab in tabs[1:]:
        client.call("close_tab", {"tab_id": tab["tab_id"]})
    if not tabs:
        client.call("new_tab")
    yield client
    client.close()


def _text(response: dict) -> str:
    return "\n".join(b["text"] for b in response.get("content", []) if b["type"] == "text")


def _state(response: dict) -> dict | None:
    states = [b for b in response.get("content", []) if b["type"] == "browser_state"]
    return states[0] if states else None


def _active(state: dict) -> dict:
    return next(t for t in state["tabs"] if t["active"])


def _ref(read_text: str, needle: str) -> str:
    for line in read_text.splitlines():
        if needle in line and "[ref_" in line:
            return line.split("[", 1)[1].split("]", 1)[0]
    raise AssertionError(f"{needle!r} not in:\n{read_text}")


def _jpeg_size(data: bytes) -> tuple[int, int]:
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + length
    raise AssertionError("no SOF marker")


def _active_status(client: Client) -> dict:
    status = client.call("_status")
    return next(t for t in status["tabs"] if t["active"])


@needs_browser
def test_navigate_reports_state_and_read_page_gives_refs(browser, site):
    nav = browser.call("navigate", {"url": f"{site}/form"})
    assert nav["ok"], nav
    assert _text(nav) == f"Navigated to {site}/form"
    state = _state(nav)
    assert state and _active(state)["title"] == "Form page" and _active(state)["url"] == f"{site}/form"
    assert nav["meta"]["title"] == "Form page" and nav["meta"]["tab_id"] == _active(state)["tab_id"]

    page = _text(browser.call("read_page"))
    assert 'heading "Order form"' in page and "level=1" in page
    assert 'textbox "Name" [ref_' in page
    assert 'checkbox "I agree to the terms" [ref_' in page
    assert 'button "Place order" [ref_' in page
    assert 'link "Other page" [ref_' in page and 'href="/other"' in page
    assert 'text "Name"' not in page  # label text is the textbox's name, not a separate line
    assert 'clickable "Buy now" [ref_' in page  # a cursor:pointer div with a click listener
    interactive = _text(browser.call("read_page", {"filter": "interactive"}))
    assert "heading" not in interactive and 'button "Next step"' in interactive
    # refs are stable for the same element while the document lives
    assert _ref(page, 'button "Next step"') == _ref(interactive, 'button "Next step"')
    shallow = _text(browser.call("read_page", {"depth": 1}))
    assert "form [ref_" in shallow and "  textbox" not in shallow
    form_ref = _ref(page, "form [ref_")
    scoped = _text(browser.call("read_page", {"ref": form_ref}))
    assert scoped.startswith("form [ref_") and "Abs button" not in scoped
    assert browser.call("read_page", {"filter": "visible"})["error"].startswith("Error: filter must be")
    assert browser.call("read_page", {"depth": 0})["error"] == "Error: depth must be at least 1."


@needs_browser
def test_click_by_ref_and_by_coordinate(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    page = _text(browser.call("read_page"))
    click = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'button "Next step"')}})
    assert click["ok"], click
    assert _text(click).startswith("Clicked ref_") and '(button "Next step")' in _text(click)
    assert _active(_state(click))["title"] == "Next pressed"   # title changed -> state reported
    assert "Next pressed" in _text(browser.call("get_page_text"))
    coord = browser.call("left_click", {"target": {"type": "coordinate", "x": 1000, "y": 630}})
    assert coord["ok"] and _text(coord) == 'Clicked (1000, 630) (button "Abs button").'
    assert _active(_state(coord))["title"] == "Coordinate click"
    out = browser.call("left_click", {"target": {"type": "coordinate", "x": 1280, "y": 10}})
    assert out["error"] == "Error: Coordinate (1280, 10) is outside the viewport (1280x800)."
    same = browser.call("hover", {"target": {"type": "coordinate", "x": 10, "y": 10}})
    assert same["ok"] and _state(same) is None   # nothing changed: no browser_state


@needs_browser
def test_type_and_enter_submit_a_search_form(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    page = _text(browser.call("read_page"))
    assert browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'textbox "Search query"')}})["ok"]
    typed = browser.call("type", {"text": "hello"})
    assert typed["ok"] and _text(typed) == "Typed 5 characters."
    entered = browser.call("key", {"text": "Enter"})
    assert entered["ok"], entered
    assert entered["sensitive_done"] is False
    state = _state(entered)
    assert _active(state)["title"] == "Result: hello" and "/result?q=hello" in _active(state)["url"]


@needs_browser
def test_keys_chords_repeat_and_validation(browser, site):
    browser.call("navigate", {"url": f"{site}/drag"})
    assert browser.call("key", {"text": "ctrl+b"})["ok"]
    assert _active_status(browser)["title"] == "key b ctrl"
    assert browser.call("key", {"text": "Backspace Backspace", "repeat": 2})["ok"]
    assert _active_status(browser)["title"] == "key Backspace"
    assert browser.call("key", {"text": "pagedown"})["ok"]
    assert browser.call("key", {"text": "Enter", "repeat": 0})["error"] == "Error: repeat must be between 1 and 100."
    assert "Unknown key" in browser.call("key", {"text": "NoSuchKey"})["error"]
    held = browser.call("hold_key", {"text": "shift", "duration": 0.2})
    assert held["ok"] and _text(held) == "Held shift for 0.2 seconds."
    assert browser.call("hold_key", {"text": "shift", "duration": 31})["error"].startswith("Error: duration")
    waited = browser.call("wait", {"duration": 0.2})
    assert waited["ok"] and _text(waited) == "Waited 0.2 seconds."
    assert browser.call("wait", {"duration": -1})["error"].startswith("Error: duration")


@needs_browser
def test_mouse_members(browser, site):
    browser.call("navigate", {"url": f"{site}/drag"})
    drag = browser.call("left_click_drag", {"from": {"type": "coordinate", "x": 100, "y": 100},
                                            "target": {"type": "coordinate", "x": 300, "y": 200}})
    assert drag["ok"] and _text(drag) == "Dragged from (100, 100) to (300, 200)."
    assert _active_status(browser)["title"] == "down 100,100 up 300,200"
    assert browser.call("left_mouse_down", {"target": {"type": "coordinate", "x": 50, "y": 60}})["ok"]
    assert browser.call("mouse_move", {"target": {"type": "coordinate", "x": 70, "y": 80}})["ok"]
    assert browser.call("left_mouse_up", {"target": {"type": "coordinate", "x": 90, "y": 100}})["ok"]
    assert _active_status(browser)["title"] == "down 50,60 up 90,100"
    assert browser.call("right_click", {"target": {"type": "coordinate", "x": 5, "y": 5}})["ok"]
    assert browser.call("middle_click", {"target": {"type": "coordinate", "x": 5, "y": 5}})["ok"]
    assert browser.call("double_click", {"target": {"type": "coordinate", "x": 5, "y": 5}})["ok"]
    assert browser.call("triple_click", {"target": {"type": "coordinate", "x": 5, "y": 5}, "modifiers": "shift"})["ok"]
    assert browser.call("left_click", {"target": {"type": "coordinate", "x": 5, "y": 5}, "modifiers": "hyper"})["error"]


@needs_browser
def test_form_input_sets_fields_checkboxes_and_selects(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    page = _text(browser.call("read_page"))
    name, agree, color = (_ref(page, 'textbox "Name"'), _ref(page, 'checkbox "I agree'), _ref(page, "combobox"))
    assert _text(browser.call("form_input", {"target": {"type": "ref", "ref": name}, "value": "Alice"})) \
        == f'Set {name} to "Alice".'
    assert _text(browser.call("form_input", {"target": {"type": "ref", "ref": agree}, "value": True})) \
        == f"Set {agree} to checked."
    assert _text(browser.call("form_input", {"target": {"type": "ref", "ref": color}, "value": "Green"})) \
        == f'Set {color} to "Green".'
    after = _text(browser.call("read_page"))
    assert 'textbox "Name" [' + name + '] value="Alice"' in after
    assert f"[{agree}] checked" in after
    assert f'[{color}] value="Green"' in after
    button = _ref(after, 'button "Next step"')
    refused = browser.call("form_input", {"target": {"type": "ref", "ref": button}, "value": "x"})
    assert "is not a form field" in refused["error"]
    assert browser.call("form_input", {"target": {"type": "ref", "ref": color}, "value": "Purple"})["error"] \
        == f'Error: {color} has no option "Purple".'
    assert browser.call("form_input", {"target": {"type": "coordinate", "x": 1, "y": 1}, "value": "x"})["error"]


@needs_browser
def test_scroll_scroll_to_and_viewport_filter(browser, site):
    nav = browser.call("navigate", {"url": f"{site}/long"})
    assert nav["ok"]
    top = _text(browser.call("read_page"))
    assert "Paragraph 1 of the long page" in top and "Bottom button" not in top
    assert "Only elements in the viewport are listed" in top
    scrolled = browser.call("scroll", {"target": {"type": "coordinate", "x": 640, "y": 400},
                                       "scroll_direction": "down", "scroll_amount": 3})
    assert scrolled["ok"] and _text(scrolled).startswith("Scrolled down by 3 at (640, 400). The page is now at y=300")
    assert _active(_state(scrolled))["title"] == "y=300"
    assert browser.call("scroll", {"target": {"type": "coordinate", "x": 640, "y": 400},
                                   "scroll_direction": "down", "scroll_amount": 11})["error"]
    assert browser.call("scroll", {"target": {"type": "coordinate", "x": 640, "y": 400},
                                   "scroll_direction": "sideways"})["error"]
    everything = _text(browser.call("read_page", {"filter": "all"}))
    bottom = _ref(everything, 'button "Bottom button"')
    assert _text(browser.call("scroll_to", {"target": {"type": "ref", "ref": bottom}})) == f"Scrolled {bottom} into view."
    assert "Bottom button" in _text(browser.call("read_page"))
    clicked = browser.call("left_click", {"target": {"type": "ref", "ref": bottom}})
    assert _active(_state(clicked))["title"] == "bottom clicked"


@needs_browser
def test_screenshot_zoom_and_peek(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    shot = browser.call("screenshot")
    assert shot["ok"] and [b["type"] for b in shot["content"]] == ["image"]
    image = shot["content"][0]
    assert image["source"]["media_type"] == "image/jpeg"
    assert _jpeg_size(base64.b64decode(image["source"]["data"])) == (1280, 800)
    zoom = browser.call("zoom", {"region": [900, 600, 1100, 660]})
    assert zoom["ok"] and [b["type"] for b in zoom["content"]] == ["image"]
    width, height = _jpeg_size(base64.b64decode(zoom["content"][0]["source"]["data"]))
    assert max(width, height) <= bs.ZOOM_MAX_SIDE and width >= 1500
    assert abs(width / height - 200 / 60) < 0.05
    assert browser.call("zoom", {"region": [10, 10, 5, 20]})["error"].startswith("Error: region")
    assert browser.call("zoom", {"region": [0, 0, 1281, 10]})["error"].startswith("Error: region")
    peek = browser.call("_peek")
    assert peek["ok"] and peek["media_type"] == "image/jpeg"
    assert _jpeg_size(base64.b64decode(peek["image_b64"])) == (1280, 800)
    assert peek["meta"]["title"] == "Form page"


@needs_browser
def test_zoom_uses_viewport_coordinates_after_scrolling(browser, site):
    pil = pytest.importorskip("PIL.Image")
    import io

    browser.call("navigate", {"url": f"{site}/long"})
    browser.call("scroll", {"target": {"type": "coordinate", "x": 640, "y": 400}, "scroll_direction": "down",
                            "scroll_amount": 10})
    page = _text(browser.call("read_page"))
    assert "Paragraph 13 of the long page" in page   # y=1000 starts paragraph 13 (80 px each)
    shot = browser.call("zoom", {"region": [0, 0, 400, 80]})
    image = pil.open(io.BytesIO(base64.b64decode(shot["content"][0]["source"]["data"])))
    assert image.size[0] <= bs.ZOOM_MAX_SIDE
    # mostly white with dark text pixels: proves the crop rendered page content, not an empty area
    colors = image.convert("L").getextrema()
    assert colors[0] < 100 and colors[1] > 200


@needs_browser
def test_find_and_get_page_text(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    found = _text(browser.call("find", {"query": "place order button"}))
    lines = found.splitlines()
    assert lines[0].startswith("1 match") or lines[0].endswith('"place order button":')
    assert lines[1].startswith('button "Place order" [ref_')
    search = _text(browser.call("find", {"query": "search field"}))
    assert search.splitlines()[1].startswith('textbox "Search query" [ref_')
    assert _text(browser.call("find", {"query": "zebra giraffe"})).startswith('No elements match "zebra giraffe"')
    # a ref from find works for actions
    ref = _ref(found, 'button "Place order"')
    assert browser.call("scroll_to", {"target": {"type": "ref", "ref": ref}})["ok"]
    browser.call("navigate", {"url": f"{site}/other"})
    text = _text(browser.call("get_page_text"))
    assert "Welcome to the other page of the test site." in text


@needs_browser
def test_read_page_output_is_capped(browser, site):
    browser.call("navigate", {"url": f"{site}/many"})
    page = _text(browser.call("read_page", {"filter": "all"}))
    assert page.endswith(bs.truncation_notice(bs.READ_PAGE_CAP))
    assert len(page) <= bs.READ_PAGE_CAP   # the notice included (review: it came on top, 50,088)
    top = _text(browser.call("read_page"))  # the viewport note is added too: still within the cap
    assert len(top) <= bs.READ_PAGE_CAP and "Only elements in the viewport are listed" in top
    assert 'link "Item number 0 with a fairly long link text"' in page


@needs_browser
def test_stale_refs_after_navigation_and_refs_never_reused(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    page = _text(browser.call("read_page"))
    old = _ref(page, 'button "Next step"')
    browser.call("navigate", {"url": f"{site}/other"})
    stale = browser.call("left_click", {"target": {"type": "ref", "ref": old}})
    assert stale == {**stale, "ok": False, "needs_confirmation": False}
    assert stale["error"] == (f"Error: {old} is stale or not found on the current page. "
                              "Re-read the page to get fresh references.")
    assert browser.call("read_page", {"ref": old})["error"] == stale["error"]
    fresh = _text(browser.call("read_page"))
    new_numbers = [int(r.split("_")[1]) for r in
                   (line.split("[", 1)[1].split("]", 1)[0] for line in fresh.splitlines() if "[ref_" in line)]
    assert min(new_numbers) > int(old.split("_")[1])   # refs are never handed out twice in a tab


@needs_browser
def test_non_http_schemes_are_refused_live(browser, site):
    for url in ("file:///etc/passwd", "javascript:alert(1)", "chrome://version"):
        result = browser.call("navigate", {"url": url})
        assert result["ok"] is False
        assert result["error"] == "Error: Navigation refused. Only http and https URLs are allowed."
    back = browser.call("navigate", {"url": "back"})
    assert back["ok"] or back["error"].startswith("Error: There is no")


@needs_browser
def test_back_forward_reload(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    browser.call("navigate", {"url": f"{site}/other"})
    back = browser.call("navigate", {"url": "back"})
    assert back["ok"] and _text(back) == f"Navigated back to {site}/form"
    forward = browser.call("navigate", {"url": "forward"})
    assert forward["ok"] and _text(forward) == f"Navigated forward to {site}/other"
    assert _text(browser.call("navigate", {"url": "reload"})) == f"Reloaded {site}/other"
    missing = browser.call("navigate", {"url": f"{site}/missing"})
    assert missing["ok"] and _text(missing).endswith("(HTTP 404)")


@needs_browser
def test_tab_management_shapes(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    first = browser.call("list_tabs")["content"][0]["tabs"][0]["tab_id"]
    created = browser.call("new_tab")
    assert created["ok"] and len(created["content"]) == 1
    state = created["content"][0]
    new_id = _active(state)["tab_id"]
    assert new_id != first and state["state_changes"] == [{"type": "tab_opened", "tab_id": new_id}]
    assert _active(state)["url"] == "about:blank"
    listed = browser.call("list_tabs")
    assert [b["type"] for b in listed["content"]] == ["browser_state"]
    assert "state_changes" not in listed["content"][0]
    assert [t["tab_id"] for t in listed["content"][0]["tabs"]] == [first, new_id]
    nav = browser.call("navigate", {"url": f"{site}/other", "tab_id": new_id})
    assert _active(_state(nav))["title"] == "Other page"
    switched = browser.call("switch_tab", {"tab_id": first})
    assert _active(switched["content"][0])["tab_id"] == first and len(switched["content"]) == 1
    assert switched["meta"]["tab_id"] == first
    # acting on another tab by tab_id makes it the active one
    read = browser.call("get_page_text", {"tab_id": new_id})
    assert "other page" in _text(read).lower() and _active(_state(read))["tab_id"] == new_id
    closed = browser.call("close_tab", {"tab_id": new_id})
    assert [t["tab_id"] for t in closed["content"][0]["tabs"]] == [first]
    assert _active(closed["content"][0])["tab_id"] == first
    unknown = browser.call("close_tab", {"tab_id": "tab-999"})
    assert unknown["ok"] is False and unknown["error"].startswith('Error: No open tab has tab_id "tab-999"')
    assert browser.call("switch_tab", {})["error"].startswith("Error: tab_id")
    # at most 6 tabs; ids are never reused
    ids = {first}
    for _ in range(bs.MAX_TABS - 1):
        ids.add(_active(browser.call("new_tab")["content"][0])["tab_id"])
    assert len(ids) == bs.MAX_TABS and new_id not in ids
    too_many = browser.call("new_tab")
    assert too_many["ok"] is False and "maximum is 6" in too_many["error"]


@needs_browser
def test_closing_the_last_tab_and_navigating_again(browser, site):
    only = browser.call("list_tabs")["content"][0]["tabs"][0]["tab_id"]
    closed = browser.call("close_tab", {"tab_id": only})
    assert closed["content"] == [{"type": "browser_state", "tabs": []}]
    assert browser.call("screenshot")["error"] == "Error: No tab is open. Use new_tab or navigate first."
    assert browser.call("_peek")["ok"] is False
    nav = browser.call("navigate", {"url": f"{site}/other"})
    assert nav["ok"]
    state = _state(nav)
    opened = _active(state)["tab_id"]
    assert state["state_changes"] == [{"type": "tab_opened", "tab_id": opened}]


@needs_browser
def test_popup_link_opens_a_tab(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    page = _text(browser.call("read_page"))
    before = _active(browser.call("list_tabs")["content"][0])["tab_id"]
    popup = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'link "Open in new tab"')}})
    assert popup["ok"], popup
    state = _state(popup)
    opened = [c for c in state.get("state_changes", []) if c["type"] == "tab_opened"]
    assert len(opened) == 1
    new_tab = _active(state)
    assert new_tab["tab_id"] == opened[0]["tab_id"] != before
    assert new_tab["url"] == f"{site}/other" and new_tab["title"] == "Other page"


@needs_browser
def test_download_reports_state_changes_and_saves_the_file(browser, site, server):
    browser.call("navigate", {"url": f"{site}/form"})
    page = _text(browser.call("read_page"))
    result = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'link "Download report"')}})
    assert result["ok"], result
    changes = (_state(result) or {}).get("state_changes", [])
    if not any(c["type"] == "download_completed" for c in changes):  # finished after the call: next result
        changes += (_state(browser.call("wait", {"duration": 1})) or {}).get("state_changes", [])
    completed = [c for c in changes if c["type"] == "download_completed"]
    assert len(completed) == 1, changes
    event = completed[0]
    assert event["url"] == f"{site}/file.txt"
    assert event["path"].startswith("downloads/report") and event["path"].endswith(".txt")
    assert event["size_bytes"] == len(DOWNLOAD_BODY)
    assert len(event["download_id"]) == 36
    saved = Path(server.downloads) / event["path"].split("/", 1)[1]
    assert saved.read_bytes() == DOWNLOAD_BODY


@needs_browser
def test_sensitive_actions_need_confirmation(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    page = _text(browser.call("read_page"))
    place = _ref(page, 'button "Place order"')
    refused = browser.call("left_click", {"target": {"type": "ref", "ref": place}})
    assert refused["ok"] is False and refused["needs_confirmation"] is True
    assert refused["error"] == ('Error: This looks like a consequential action (button "Place order"). '
                                "Ask the user first with browser_confirm, then repeat the action.")
    assert _active_status(browser)["title"] == "Form page"     # nothing happened
    # Enter in the order form's field submits it: same guard
    browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'textbox "Quantity"')}})
    browser.call("type", {"text": "2"})
    enter = browser.call("key", {"text": "Enter"})
    assert enter["needs_confirmation"] is True and '(button "Place order")' in enter["error"]
    # Enter in a message composer
    browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'textbox "Chat box"')}})
    composer = browser.call("key", {"text": "Enter"})
    assert composer["needs_confirmation"] is True and "Enter in the text field" in composer["error"]
    typed = browser.call("type", {"text": "hi\n"})
    assert typed["needs_confirmation"] is True and "already done" not in typed["error"]
    assert 'value="hi"' not in _text(browser.call("read_page"))   # refused before typing anything
    # a div "button" without a role, clicked by coordinate on its inner span
    buy = browser.call("left_click", {"target": {"type": "coordinate", "x": 1000, "y": 720}})
    assert buy["needs_confirmation"] is True and '(clickable "Buy now")' in buy["error"]
    # clicking plain text whose words sound consequential is not an action
    assert browser.call("left_click", {"target": {"type": "coordinate", "x": 30, "y": 15}})["ok"]
    # allowed: performed and reported
    allowed = browser.call("left_click", {"target": {"type": "ref", "ref": place}}, allow_sensitive=True)
    assert allowed["ok"] is True and allowed["sensitive_done"] is True
    assert _active(_state(allowed))["title"] == "Order placed"
    # an allowance is not "used" by an ordinary action
    plain = browser.call("left_click", {"target": {"type": "coordinate", "x": 5, "y": 5}}, allow_sensitive=True)
    assert plain["ok"] and plain["sensitive_done"] is False


@needs_browser
def test_disabled_and_unknown_members(browser):
    for member in ("javascript_exec", "file_upload", "read_console", "read_network"):
        result = browser.call(member, {"text": "1"})
        assert result["ok"] is False and result["error"] == f"Error: {member} is not enabled in this environment."
    unknown = browser.call("teleport")
    assert unknown["ok"] is False and unknown["error"] == 'Error: Unknown browser action "teleport".'


@needs_browser
def test_status_and_protocol_errors(server, browser):
    status = browser.call("_status")
    assert status["ok"] and sum(1 for t in status["tabs"] if t["active"]) == 1
    assert status["started"] <= status["last_request"] <= time.time() + 1
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect(server.socket)
    stream = sock.makefile("rwb")
    stream.write(b"this is not json\n")
    stream.flush()
    assert json.loads(stream.readline())["ok"] is False
    stream.write(json.dumps({"id": "x", "action": "navigate", "input": "nope"}).encode() + b"\n")
    stream.flush()
    reply = json.loads(stream.readline())
    assert reply["id"] == "x" and reply["error"] == "Error: input must be an object."
    sock.close()


@needs_browser
def test_private_addresses_are_refused_without_allow_private(site):
    workdir = _workdir()
    proc = ServerProcess(workdir, allow_private=False, profile=None)
    try:
        client = proc.client()
        port = urlsplit(site).port
        refused = client.call("navigate", {"url": f"http://127.0.0.1:{port}/form"})
        assert refused["ok"] is False and refused["needs_confirmation"] is False
        assert refused["error"] == "Error: Navigation refused. 127.0.0.1 is a private or local address."
        assert client.call("navigate", {"url": f"http://localhost:{port}/"})["error"] == \
            "Error: Navigation refused. localhost is a private or local address."
        assert client.call("navigate", {"url": "http://169.254.169.254/latest/meta-data/"})["error"] == \
            "Error: Navigation refused. 169.254.169.254 is a private or local address."
        assert client.call("navigate", {"url": "http://[::1]:8080/"})["error"].endswith("is a private or local address.")
        client.close()
    finally:
        proc.stop()
        shutil.rmtree(workdir, ignore_errors=True)


@needs_browser
def test_cookies_persist_across_restarts_with_the_same_profile(site):
    workdir = _workdir()
    try:
        first = ServerProcess(workdir, profile="profile")
        client = first.client()
        assert client.call("navigate", {"url": f"{site}/setcookie"})["ok"]
        client.close()
        first.stop()
        assert first.proc.returncode == 0
        assert not os.path.exists(first.socket)   # removed on exit

        second = ServerProcess(workdir, profile="profile")
        client = second.client()
        client.call("navigate", {"url": f"{site}/echo"})
        assert "sid=abc123" in _text(client.call("get_page_text"))
        client.close()
        second.stop()

        ephemeral = ServerProcess(workdir, profile=None)
        client = ephemeral.client()
        client.call("navigate", {"url": f"{site}/echo"})
        assert "Cookie header: none" in _text(client.call("get_page_text"))
        client.close()
        ephemeral.stop()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@needs_browser
def test_idle_exit_ignores_peek_and_status(site):
    workdir = _workdir()
    try:
        proc = ServerProcess(workdir, profile=None, idle=2)
        started = time.monotonic()
        client = proc.client()
        while proc.alive() and time.monotonic() - started < 20:
            try:
                client.call("_peek")
                client.call("_status")
                client.call("_ping")
            except (OSError, AssertionError, ValueError):
                break
            time.sleep(0.3)
        client.close()
        proc.proc.wait(timeout=20)
        assert proc.proc.returncode == 0
        assert time.monotonic() - started < 15
        assert "idle for 2 s" in proc.logs()
        proc.stop()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@needs_browser
def test_touch_postpones_the_idle_exit(site):
    workdir = _workdir()
    try:
        proc = ServerProcess(workdir, profile=None, idle=3)
        client = proc.client()
        started = time.monotonic()
        while time.monotonic() - started < 7:
            assert client.call("_touch") == {"id": str(client.counter), "ok": True}
            time.sleep(1)
        assert proc.alive(), "the server exited although _touch kept it busy"
        # a member call counts as activity too
        assert client.call("wait", {"duration": 0})["ok"]
        stopped_touching = time.monotonic()
        proc.proc.wait(timeout=20)
        assert 2.5 <= time.monotonic() - stopped_touching < 15
        client.close()
        proc.stop()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@needs_browser
def test_sigterm_closes_cleanly(site):
    workdir = _workdir()
    try:
        proc = ServerProcess(workdir, profile="profile")
        proc.proc.terminate()
        assert proc.proc.wait(timeout=30) == 0
        assert not os.path.exists(proc.socket)
        assert not any(Path(workdir, "profile").glob("Singleton*"))
        proc.stop()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@needs_browser
def test_page_dialogs(browser, site):
    browser.call("navigate", {"url": f"{site}/dialog"})
    page = _text(browser.call("read_page"))
    trash, leave, save = (_ref(page, 'button "Trash"'), _ref(page, 'button "Leave"'), _ref(page, 'button "Save"'))
    refused = browser.call("left_click", {"target": {"type": "ref", "ref": trash}})
    assert refused["ok"] is False and refused["needs_confirmation"] is True
    assert refused["error"].startswith('Error: The page asked "Delete this item permanently?" and the browser '
                                       "answered Cancel")
    assert _active_status(browser)["title"] == "kept"
    allowed = browser.call("left_click", {"target": {"type": "ref", "ref": trash}}, allow_sensitive=True)
    assert allowed["ok"] and allowed["sensitive_done"] is True
    assert _active_status(browser)["title"] == "deleted"
    left = browser.call("left_click", {"target": {"type": "ref", "ref": leave}})
    assert left["ok"] and 'The page asked "Leave this page?" and the browser answered OK.' in _text(left)
    assert left["sensitive_done"] is False
    alerted = browser.call("left_click", {"target": {"type": "ref", "ref": save}})
    assert alerted["ok"] and 'The page showed an alert: "Saved!"' in _text(alerted)
    assert _active(_state(alerted))["title"] == "alerted"


@needs_browser
def test_long_text_is_typed_quickly(browser, site):
    browser.call("navigate", {"url": f"{site}/form"})
    page = _text(browser.call("read_page"))
    browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'textbox "Name"')}})
    text = "abc" * 3000 + "xyz"
    started = time.monotonic()
    typed = browser.call("type", {"text": text})
    assert typed["ok"] and _text(typed) == "Typed 9003 characters."
    assert time.monotonic() - started < 15
    assert 'value="abcabcabc' in _text(browser.call("read_page"))
    assert browser.call("type", {"text": "x" * (bs.MAX_TYPE_CHARS + 1)})["error"].startswith("Error: type takes at most")


@needs_browser
def test_refs_inside_same_origin_iframes_and_shadow_roots(browser, site):
    browser.call("navigate", {"url": f"{site}/frames"})
    page = _text(browser.call("read_page"))
    assert "iframe [ref_" in page and '  button "Frame button" [ref_' in page
    assert 'button "Shadow button" [ref_' in page
    framed = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'button "Frame button"')}})
    assert _active(_state(framed))["title"] == "frame clicked"
    shadow = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'button "Shadow button"')}})
    assert _active(_state(shadow))["title"] == "shadow clicked"
    found = _text(browser.call("find", {"query": "shadow button"}))
    assert found.splitlines()[1].startswith('button "Shadow button" [ref_')


@needs_browser
def test_navigating_to_a_file_downloads_it(browser, site, server):
    result = browser.call("navigate", {"url": f"{site}/file.txt"})
    assert result["ok"], result
    assert _text(result) == f"Opened {site}/file.txt: the server sent a file, which is being downloaded."
    changes = (_state(result) or {}).get("state_changes", [])
    if not any(c["type"] == "download_completed" for c in changes):
        changes += (_state(browser.call("wait", {"duration": 1})) or {}).get("state_changes", [])
    completed = [c for c in changes if c["type"] == "download_completed"]
    assert len(completed) == 1 and completed[0]["size_bytes"] == len(DOWNLOAD_BODY)
    assert (Path(server.downloads) / completed[0]["path"].split("/", 1)[1]).read_bytes() == DOWNLOAD_BODY


# ---------------------------------------------------------------------------------------------
# Regression tests from the review (real Chromium)
# ---------------------------------------------------------------------------------------------

def _redirect_to_secret(handler):
    tag = parse_qs(urlsplit(handler.path).query).get("tag", ["x"])[0]
    target = f"http://127.0.0.1:{handler.server.server_address[1]}/secret-{tag}"
    return "", 302, [("Location", target)]


SSRF_PAGE = """<!doctype html><html><head><title>ssrf main</title></head><body>
<img src="/ssrf-redir?tag=img-redirect">
<iframe src="/ssrf-redir?tag=iframe-redirect"></iframe>
<script>
fetch('/ssrf-redir?tag=fetch-redirect').catch(() => {});
try { new SharedWorker('/ssrf-shared.js'); } catch (e) {}
window.open('/ssrf-redir?tag=popup-redirect');
setTimeout(() => { document.title = 'ssrf done'; }, 1500);
</script></body></html>"""


@needs_browser
def test_private_addresses_are_unreachable_through_redirects_workers_and_popups(site):
    # review (HIGH): the route filter missed redirects (navigation, subresources, popups) and
    # SharedWorker requests. Every request now goes through the egress proxy, which checks each host.
    port = urlsplit(site).port
    secret = f"http://127.0.0.1:{port}/secret"
    EXTRA_PAGES["/ssrf-main"] = SSRF_PAGE
    EXTRA_PAGES["/ssrf-redir"] = _redirect_to_secret
    EXTRA_PAGES["/ssrf-shared.js"] = lambda h: (
        f"onconnect = () => {{}}; fetch('{secret}-sharedworker').catch(() => {{}}); "
        f"fetch('/ssrf-redir?tag=sharedworker-redirect').catch(() => {{}});", 200, [])
    workdir = _workdir()
    # "public.test" resolves to a public address for the address check; the proxy's connection to
    # that (checked) address is mapped to the local site
    proc = ServerProcess(workdir, allow_private=False, profile=None,
                         fake_dns={"public.test": ["93.184.216.34"]}, connect_map={"93.184.216.34": "127.0.0.1"})
    try:
        client = proc.client()
        public = f"http://public.test:{port}"
        nav = client.call("navigate", {"url": f"{public}/ssrf-main"})
        assert nav["ok"], nav
        time.sleep(3)
        redirected = client.call("navigate", {"url": f"{public}/ssrf-redir?tag=navigation-redirect"})
        assert redirected["ok"] is False
        assert redirected["error"] == ("Error: Navigation refused. The page redirected to 127.0.0.1, "
                                       "a private or local address.")
        assert client.call("navigate", {"url": f"http://127.0.0.1:{port}/form"})["error"] == \
            "Error: Navigation refused. 127.0.0.1 is a private or local address."
        unknown = client.call("navigate", {"url": f"http://unknown.test:{port}/"})
        assert unknown["ok"] is False and "unknown.test could not be resolved" in unknown["error"]
        client.close()
        hits = [h for h in HITS if "/secret" in h]
        assert hits == [], hits
        assert any(h.startswith("/ssrf-redir?tag=popup-redirect") for h in HITS)  # the popup did try
        assert "/ssrf-shared.js" in HITS                                        # the worker did run
        assert "proxy refused 127.0.0.1" in proc.logs()
    finally:
        proc.stop()
        shutil.rmtree(workdir, ignore_errors=True)


def _cross_site(site: str) -> str:
    return site.replace("127.0.0.1", "localhost")  # another site: an out-of-process frame


@needs_browser
def test_guard_sees_into_cross_origin_frames_and_closed_shadow_roots(browser, site):
    # review (HIGH): coordinate clicks on "Pay $500 now" inside a cross-origin iframe or a closed
    # shadow root, and Enter in a card field there, were not checked
    EXTRA_PAGES["/xo-widget"] = """<!doctype html><html><head><title>Checkout widget</title></head>
<body style="margin:0"><form action="/xo-charged" target="_top" method="get">
<input name="card" aria-label="Card number" style="position:absolute;left:0;top:0;width:200px;height:30px">
<button type="submit" style="position:absolute;left:0;top:100px;width:200px;height:50px">Pay $500 now</button>
</form></body></html>"""
    EXTRA_PAGES["/xo-shop"] = f"""<!doctype html><html><head><title>XO shop</title></head><body style="margin:0">
<iframe src="{_cross_site(site)}/xo-widget" style="position:absolute;left:100px;top:100px;width:400px;height:200px;border:0"></iframe>
<pay-button style="position:absolute;left:700px;top:100px"></pay-button>
<script>
customElements.define('pay-button', class extends HTMLElement {{ constructor() {{ super();
  const r = this.attachShadow({{mode: 'closed'}});
  r.innerHTML = '<button style="width:200px;height:50px">Buy now</button>';
  r.querySelector('button').onclick = () => {{ document.title = 'BOUGHT'; }}; }} }});
</script></body></html>"""
    EXTRA_PAGES["/xo-charged"] = "<html><head><title>CHARGED</title></head><body>charged</body></html>"
    browser.call("navigate", {"url": f"{site}/xo-shop"})
    time.sleep(0.5)
    pay = browser.call("left_click", {"target": {"type": "coordinate", "x": 200, "y": 225}})
    assert pay["needs_confirmation"] is True and '(button "Pay $500 now")' in pay["error"]
    card = browser.call("left_click", {"target": {"type": "coordinate", "x": 150, "y": 115}})
    assert card["ok"] and _text(card) == 'Clicked (150, 115) (textbox "Card number").'  # a field: fine
    typed = browser.call("type", {"text": "4242\n"})
    assert typed["needs_confirmation"] is True and '(button "Pay $500 now")' in typed["error"]
    tabbed = browser.call("key", {"text": "Tab Enter"})  # the focus moves to "Pay" inside the frame
    assert tabbed["needs_confirmation"] is True and '(button "Pay $500 now")' in tabbed["error"]
    shadow = browser.call("left_click", {"target": {"type": "coordinate", "x": 800, "y": 125}})
    assert shadow["needs_confirmation"] is True and '(button "Buy now")' in shadow["error"]
    # the focused button inside the closed root: Enter and Space are checked on it too
    browser.call("left_click", {"target": {"type": "coordinate", "x": 650, "y": 400}})
    for _ in range(6):  # card field, "Pay $500 now" (both in the frame), then the shadow button
        browser.call("key", {"text": "Tab"})
        focused = browser.call("key", {"text": "space"})
        if not focused["ok"] and '(button "Buy now")' in focused["error"]:
            break
    assert focused["needs_confirmation"] is True and '(button "Buy now")' in focused["error"]
    assert browser.call("key", {"text": "Enter"})["needs_confirmation"] is True
    assert _active_status(browser)["title"] == "XO shop"
    assert not [h for h in HITS if h.startswith("/xo-charged")]


@needs_browser
def test_middle_click_mouse_down_up_and_drags_are_guarded(browser, site):
    # review (HIGH): middle_click, left_mouse_down/up and left_click_drag never called the guard
    EXTRA_PAGES["/mouse-guard"] = """<!doctype html><html><head><title>mouse guard</title></head><body style="margin:0">
<a href="/mg-delete?id=7" style="position:absolute;left:10px;top:10px">Delete</a>
<button style="position:absolute;left:300px;top:200px;width:200px;height:60px" onclick="document.title='BOUGHT'">Buy now</button>
</body></html>"""
    EXTRA_PAGES["/mg-delete"] = "<html><head><title>item 7 deleted</title></head><body>deleted</body></html>"
    browser.call("navigate", {"url": f"{site}/mouse-guard"})
    page = _text(browser.call("read_page"))
    middle = browser.call("middle_click", {"target": {"type": "ref", "ref": _ref(page, 'link "Delete"')}})
    assert middle["needs_confirmation"] is True and '(link "Delete")' in middle["error"]
    down = browser.call("left_mouse_down", {"target": {"type": "coordinate", "x": 400, "y": 230}})
    assert down["needs_confirmation"] is True and '(button "Buy now")' in down["error"]
    up = browser.call("left_mouse_up", {"target": {"type": "coordinate", "x": 400, "y": 230}})
    assert up["needs_confirmation"] is True   # no checked press at this point before it
    drag = browser.call("left_click_drag", {"from": {"type": "coordinate", "x": 390, "y": 225},
                                            "target": {"type": "coordinate", "x": 410, "y": 235}})
    assert drag["needs_confirmation"] is True
    onto = browser.call("left_click_drag", {"from": {"type": "coordinate", "x": 700, "y": 600},
                                            "target": {"type": "coordinate", "x": 30, "y": 18}})
    assert onto["needs_confirmation"] is True and '(link "Delete")' in onto["error"]
    assert _active_status(browser)["title"] == "mouse guard"
    # allowed: the checked press covers its release at the same point
    down = browser.call("left_mouse_down", {"target": {"type": "coordinate", "x": 400, "y": 230}}, allow_sensitive=True)
    assert down["ok"] and down["sensitive_done"] is True
    up = browser.call("left_mouse_up", {"target": {"type": "coordinate", "x": 400, "y": 230}})
    assert up["ok"] and up["sensitive_done"] is False
    assert _active_status(browser)["title"] == "BOUGHT"
    assert not [h for h in HITS if h.startswith("/mg-delete")]


@needs_browser
def test_keys_are_checked_on_the_element_that_has_the_focus_right_before(browser, site):
    # review (HIGH): only Enter was checked, on the element focused BEFORE the sequence: "Tab Enter"
    # bought, Space on a focused "Place order" ordered
    EXTRA_PAGES["/keys-guard"] = """<!doctype html><html><head><title>keys guard</title></head><body style="margin:0">
<input aria-label="Coupon code" style="position:absolute;left:10px;top:10px;width:200px;height:30px">
<button style="position:absolute;left:300px;top:200px;width:200px;height:60px" onclick="document.title='BOUGHT'">Buy now</button>
<a href="#" onclick="document.title='DELETED'; return false;" style="position:absolute;left:300px;top:400px">Delete account</a>
</body></html>"""
    url = f"{site}/keys-guard"

    def focus_coupon():
        browser.call("navigate", {"url": url})
        assert browser.call("left_click", {"target": {"type": "coordinate", "x": 50, "y": 25}})["ok"]

    focus_coupon()
    tab_enter = browser.call("key", {"text": "Tab Enter"})
    assert tab_enter["needs_confirmation"] is True and '(button "Buy now")' in tab_enter["error"]
    assert "were already done" in tab_enter["error"]   # the Tab happened: the focus is on "Buy now"
    focus_coupon()
    assert browser.call("key", {"text": "Tab Tab Enter"})["error"].count('link "Delete account"') == 1
    focus_coupon()
    browser.call("key", {"text": "Tab"})
    for refused in (browser.call("key", {"text": "space"}), browser.call("type", {"text": " "}),
                    browser.call("hold_key", {"text": "space", "duration": 0.1}),
                    browser.call("key", {"text": "NumpadEnter"})):
        assert refused["needs_confirmation"] is True and '(button "Buy now")' in refused["error"], refused
    assert _active_status(browser)["title"] == "keys guard"
    # typing a sentence (spaces) into the coupon field is fine
    focus_coupon()
    typed = browser.call("type", {"text": "SAVE 10 NOW"})
    assert typed["ok"] and _text(typed) == "Typed 11 characters."
    # allowed: one activation; a second consequential key in the same action stops there
    focus_coupon()
    browser.call("key", {"text": "Tab"})
    allowed = browser.call("key", {"text": "space Tab Enter"}, allow_sensitive=True)
    assert allowed["ok"] and allowed["sensitive_done"] is True
    assert "one confirmation covers one action" in _text(allowed)
    assert _active_status(browser)["title"] == "BOUGHT"


@needs_browser
def test_a_click_on_a_container_checks_what_is_under_the_point(browser, site):
    # review (MEDIUM): clicking cell "Delete" hit its button; clicking dialog "Checkout" hit "Pay now"
    EXTRA_PAGES["/container-guard"] = """<!doctype html><html><head><title>files</title></head><body>
<table border=1><tr><td style="width:300px">report.pdf</td>
<td style="width:120px;height:40px;text-align:center"><button onclick="document.title='DELETED'">Delete</button></td></tr></table>
<div role="dialog" aria-label="Checkout" style="position:absolute;left:100px;top:200px;width:300px;height:120px;display:flex;align-items:center;justify-content:center">
<button onclick="document.title='PAID'">Pay now</button></div>
</body></html>"""
    browser.call("navigate", {"url": f"{site}/container-guard"})
    page = _text(browser.call("read_page", {"filter": "all"}))
    cell = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'cell "Delete"')}})
    assert cell["needs_confirmation"] is True and '(button "Delete")' in cell["error"]
    dialog = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'dialog "Checkout"')}})
    assert dialog["needs_confirmation"] is True and '(button "Pay now")' in dialog["error"]
    harmless = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'cell "report.pdf"')}})
    assert harmless["ok"]
    assert _active_status(browser)["title"] == "files"


@needs_browser
def test_back_forward_and_reload_never_resend_a_form_unasked(browser, site):
    # review (MEDIUM): an allowed order, then reload / back made POSTs #2 and #3 silently
    EXTRA_PAGES["/post-shop"] = """<html><head><title>shop</title></head><body>
<form method="post" action="/post-order"><button>Place order</button></form></body></html>"""
    browser.call("navigate", {"url": f"{site}/post-shop"})
    page = _text(browser.call("read_page", {"filter": "interactive"}))
    before = len(POSTS)
    order = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'button "Place order"')}},
                         allow_sensitive=True)
    assert order["ok"] and order["sensitive_done"] is True and len(POSTS) == before + 1
    reload = browser.call("navigate", {"url": "reload"})
    assert reload["needs_confirmation"] is True
    assert '(navigate "reload" would send the form of that page again (POST))' in reload["error"]
    browser.call("navigate", {"url": f"{site}/other"})
    back = browser.call("navigate", {"url": "back"})
    assert back["needs_confirmation"] is True and 'navigate "back"' in back["error"]
    assert len(POSTS) == before + 1
    assert _active_status(browser)["title"] == "Other page"
    again = browser.call("navigate", {"url": "back"}, allow_sensitive=True)
    assert again["ok"] and again["sensitive_done"] is True and "sent again" in _text(again)
    assert len(POSTS) == before + 2
    plain = browser.call("navigate", {"url": "forward"})   # back to a GET page: nothing to confirm
    assert plain["ok"] and plain["sensitive_done"] is False


@needs_browser
def test_registration_and_reply_fields_are_not_refused(browser, site):
    # review (MEDIUM-LOW): a click into "Post code" was refused, an allowed click into "Write a
    # reply" spent the allowance on focusing; (MEDIUM) a bare "Отправить" broke registration
    EXTRA_PAGES["/register-ru"] = """<!doctype html><html><head><title>Регистрация</title></head><body>
<form onsubmit="event.preventDefault(); document.title='REGISTERED'">
<label>Email <input name="email"></label> <label>Post code <input name="zip"></label>
<label>Пароль <input type="password" name="pw"></label>
<label>Ответ на секретный вопрос <input name="answer"></label>
<button type="submit">Отправить</button></form>
<form onsubmit="event.preventDefault(); document.title='MESSAGE SENT'">
<textarea aria-label="Сообщение"></textarea><button type="submit">Отправить</button></form>
<textarea aria-label="Write a reply"></textarea>
</body></html>"""
    browser.call("navigate", {"url": f"{site}/register-ru"})
    page = _text(browser.call("read_page", {"filter": "interactive"}))
    assert browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'textbox "Post code"')}})["ok"]
    reply = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'textbox "Write a reply"')}},
                         allow_sensitive=True)
    assert reply["ok"] and reply["sensitive_done"] is False   # the allowance is still unused
    answer = _ref(page, 'textbox "Ответ на секретный вопрос"')
    browser.call("left_click", {"target": {"type": "ref", "ref": answer}})
    browser.call("type", {"text": "Мурзик"})
    entered = browser.call("key", {"text": "Enter"})          # submits the registration form
    assert entered["ok"], entered
    assert _active_status(browser)["title"] == "REGISTERED"
    buttons = [line for line in page.splitlines() if 'button "Отправить"' in line]
    register_submit, message_submit = (b.split("[", 1)[1].split("]", 1)[0] for b in buttons)
    assert browser.call("left_click", {"target": {"type": "ref", "ref": register_submit}})["ok"]
    message = browser.call("left_click", {"target": {"type": "ref", "ref": message_submit}})
    assert message["needs_confirmation"] is True and '(button "Отправить")' in message["error"]


@needs_browser
def test_long_titles_never_break_tab_results(browser, site):
    # review (LOW): a 4096+ character title cut on a space made every tab member fail
    EXTRA_PAGES["/long-title"] = f"<html><head><title>{'a' * 4095} tail</title></head><body>long title</body></html>"
    nav = browser.call("navigate", {"url": f"{site}/long-title"})
    assert nav["ok"] and _active(_state(nav))["title"] == "a" * 4095
    assert browser.call("list_tabs")["ok"]
    created = browser.call("new_tab")
    assert created["ok"] and len(created["content"]) == 1


@needs_browser
def test_refs_are_unique_across_tabs(browser, site):
    # review (LOW): tab-2's ref_1 used while tab-1 was active clicked tab-1's element
    EXTRA_PAGES["/refs-a"] = "<html><head><title>A</title></head><body><button onclick=\"document.title='A clicked'\">Archive</button></body></html>"
    EXTRA_PAGES["/refs-b"] = "<html><head><title>B</title></head><body><button onclick=\"document.title='B clicked'\">Show details</button></body></html>"
    browser.call("navigate", {"url": f"{site}/refs-a"})
    first = browser.call("list_tabs")["content"][0]["tabs"][0]["tab_id"]
    page_a = _text(browser.call("read_page", {"filter": "interactive"}))
    browser.call("new_tab")
    browser.call("navigate", {"url": f"{site}/refs-b"})
    page_b = _text(browser.call("read_page", {"filter": "interactive"}))
    ref_a, ref_b = _ref(page_a, "Archive"), _ref(page_b, "Show details")
    assert ref_a != ref_b
    browser.call("switch_tab", {"tab_id": first})
    wrong = browser.call("left_click", {"target": {"type": "ref", "ref": ref_b}})
    assert wrong["ok"] is False and wrong["error"] == bs.stale_ref_error(ref_b)
    assert _active_status(browser)["title"] == "A"


@needs_browser
def test_session_cookies_survive_a_restart_with_a_persistent_profile(site):
    # review (LOW): Chromium keeps only cookies with an expiry, so a session-cookie login ended at
    # every idle exit or restart
    EXTRA_PAGES["/login-session"] = lambda h: (
        "<html><head><title>logged in</title></head><body>ok</body></html>", 200,
        [("Set-Cookie", "session=s3cr3t; Path=/; HttpOnly"), ("Set-Cookie", "remember=r1; Max-Age=86400; Path=/")])
    workdir = _workdir()
    kept = Path(workdir, "profile", bs.SESSION_COOKIES_FILE)
    try:
        first = ServerProcess(workdir, profile="profile")
        client = first.client()
        assert client.call("navigate", {"url": f"{site}/login-session"})["ok"]
        client.close()
        first.stop()
        assert kept.exists() and (kept.stat().st_mode & 0o777) == 0o600
        assert "s3cr3t" not in first.logs()
        second = ServerProcess(workdir, profile="profile")
        assert not kept.exists()          # read once, then gone
        client = second.client()
        client.call("navigate", {"url": f"{site}/echo"})
        text = _text(client.call("get_page_text"))
        assert "session=s3cr3t" in text and "remember=r1" in text
        client.close()
        second.stop()
        assert kept.exists()              # written again at this shutdown: still logged in next time
        ephemeral = ServerProcess(workdir, profile=None)
        client = ephemeral.client()
        client.call("navigate", {"url": f"{site}/login-session"})
        client.close()
        ephemeral.stop()
        assert "session cookie" not in ephemeral.logs()   # a throwaway profile keeps nothing
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@needs_browser
def test_force_state_reports_the_tabs_even_when_unchanged(browser, site):
    # review (LOW): the next turn's first read_page carried no tab list when the last report went
    # to another conversation; the bot now asks with force_state
    browser.call("navigate", {"url": f"{site}/other"})
    assert _state(browser.call("read_page")) is None           # unchanged since the last report
    forced = browser.call("read_page", {})
    assert _state(forced) is None
    browser.counter += 1
    request = {"id": str(browser.counter), "action": "read_page", "input": {}, "force_state": True}
    browser.file.write((json.dumps(request) + "\n").encode())
    browser.file.flush()
    response = json.loads(browser.file.readline())
    assert response["ok"] and bs.validate_content("read_page", response["content"]) == []
    assert _active(_state(response))["url"] == f"{site}/other"
    refused = {"id": "x", "action": "read_page", "input": {"depth": 0}, "force_state": True}
    browser.file.write((json.dumps(refused) + "\n").encode())
    browser.file.flush()
    error = json.loads(browser.file.readline())
    assert error["ok"] is False and "content" not in error    # never a browser_state on an error


@needs_browser
def test_coordinates_follow_the_viewport_of_a_popup(browser, site):
    # minor (review): a popup opened with its own size has a smaller viewport than 1280x800
    EXTRA_PAGES["/vp-opener"] = """<html><head><title>opener</title></head><body>
<button onclick="window.open('/vp-pop', 'p', 'width=420,height=300')">Open</button></body></html>"""
    EXTRA_PAGES["/vp-pop"] = "<html><head><title>pop</title></head><body>small window</body></html>"
    browser.call("navigate", {"url": f"{site}/vp-opener"})
    page = _text(browser.call("read_page"))
    opened = browser.call("left_click", {"target": {"type": "ref", "ref": _ref(page, 'button "Open"')}})
    assert _active(_state(opened))["title"] == "pop"
    outside = browser.call("left_click", {"target": {"type": "coordinate", "x": 500, "y": 10}})
    assert outside["error"] == "Error: Coordinate (500, 10) is outside the viewport (420x300)."
    assert browser.call("zoom", {"region": [0, 0, 600, 100]})["error"].startswith("Error: region")
    assert browser.call("left_click", {"target": {"type": "coordinate", "x": 100, "y": 100}})["ok"]


@needs_browser
def test_a_cross_origin_frame_the_guard_cannot_enter_counts_only_when_it_looks_like_payment(site):
    # the last resort (no DevTools session for the frame): only payment-looking frames are refused
    EXTRA_PAGES["/opaque-pay"] = "<html><body style='margin:0'><button style='width:300px;height:100px'>Go</button></body></html>"
    EXTRA_PAGES["/opaque-video"] = EXTRA_PAGES["/opaque-pay"]
    EXTRA_PAGES["/opaque-shop"] = f"""<!doctype html><html><head><title>opaque</title></head><body style="margin:0">
<iframe title="Secure payment form" src="{_cross_site(site)}/opaque-pay" style="position:absolute;left:0;top:0;width:400px;height:200px;border:0"></iframe>
<iframe title="Video" src="{_cross_site(site)}/opaque-video" style="position:absolute;left:500px;top:0;width:400px;height:200px;border:0"></iframe>
</body></html>"""
    workdir = _workdir()
    proc = ServerProcess(workdir, profile=None, no_frame_sessions=True)
    try:
        client = proc.client()
        client.call("navigate", {"url": f"{site}/opaque-shop"})
        time.sleep(0.5)
        pay = client.call("left_click", {"target": {"type": "coordinate", "x": 100, "y": 50}})
        assert pay["needs_confirmation"] is True
        assert '(frame "Secure payment form" (looks like a payment form))' in pay["error"]
        assert client.call("left_click", {"target": {"type": "coordinate", "x": 600, "y": 50}})["ok"]
        client.close()
    finally:
        proc.stop()
        shutil.rmtree(workdir, ignore_errors=True)
