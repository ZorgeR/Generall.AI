"""Tests for deploy/render_env.py, the script that turns the PROD GitHub environment
(secrets + variables) into the server's .env.

deploy/ is not a package and pytest.ini only puts app/ on sys.path, so the script is
loaded from its path."""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "render_env.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_env = _load_module()
render = render_env.render


def test_secret_wins_over_variable_with_the_same_key():
    text, keys = render({"A_KEY": "from-secret"}, {"A_KEY": "from-var", "ONLY_VAR": "v"})
    assert text == "A_KEY=from-secret\nONLY_VAR=v\n"
    assert keys == ["A_KEY", "ONLY_VAR"]


def test_deploy_only_keys_are_excluded():
    secrets = {
        "SERVER_ADDR": "1.2.3.4",
        "SERVER_USER": "root",
        "SERVER_SSH_KEY": "-----BEGIN OPENSSH PRIVATE KEY-----",
        "SERVER_CODE_ROOT_PATH": "/opt/generall.ai/Generall.AI",
        "github_token": "ghs_token",
        "TELEGRAM_BOT_TOKEN": "123:abc",
    }
    text, keys = render(secrets, {"SERVER_KNOWN_HOSTS": "host ssh-ed25519 AAAA", "GITHUB_TOKEN": "x"})
    assert keys == ["TELEGRAM_BOT_TOKEN"]
    assert text == "TELEGRAM_BOT_TOKEN=123:abc\n"
    for leaked in ("SERVER_", "ghs_token", "GITHUB_TOKEN", "1.2.3.4"):
        assert leaked not in text


def test_keys_that_are_not_upper_case_env_names_are_excluded():
    variables = {
        "lower_case": "1",
        "1STARTS_WITH_DIGIT": "2",
        "HAS-DASH": "3",
        "Mixed_Case": "4",
        "_LEADING_UNDERSCORE": "5",
        "HAS SPACE": "6",
        "": "7",
        "OK_KEY_2": "8",
    }
    text, keys = render({}, variables)
    assert keys == ["OK_KEY_2"]
    assert text == "OK_KEY_2=8\n"


def test_simple_values_are_written_bare():
    variables = {
        "URL": "http://localhost:8081",
        "IDS": "12,-34,56",
        "PHONE": "+15551234567",
        "ROOT": "/opt/generall.ai/Generall.AI",
        "B64": "abc+/=",
        "MAIL": "a@b.c",
        "EMPTY": "",
    }
    text, _ = render({}, variables)
    assert text.splitlines() == [
        "B64=abc+/=",
        "EMPTY=",
        "IDS=12,-34,56",
        "MAIL=a@b.c",
        "PHONE=+15551234567",
        "ROOT=/opt/generall.ai/Generall.AI",
        "URL=http://localhost:8081",
    ]


def test_value_with_spaces_is_double_quoted():
    text, _ = render({}, {"NAME": "General AI"})
    assert text == 'NAME="General AI"\n'


def test_quotes_and_backslashes_are_escaped_inside_double_quotes():
    text, _ = render({"MSG": 'say "hi" C:\\path'}, {})
    assert text == 'MSG="say \\"hi\\" C:\\\\path"\n'


def test_dollar_is_doubled_so_compose_does_not_interpolate_it():
    text, _ = render({"PW": "pa$$word $HOME"}, {})
    assert text == 'PW="pa$$$$word $$HOME"\n'


def test_hash_and_other_punctuation_force_quoting():
    text, _ = render({}, {"A": "abc #1", "B": "x'y", "C": "semi;colon"})
    assert text == 'A="abc #1"\nB="x\'y"\nC="semi;colon"\n'


def test_values_with_newlines_are_skipped_and_reported():
    skipped = []
    text, keys = render({"PEM": "line1\nline2", "CR": "a\rb", "OK": "1"}, {}, skipped=skipped)
    assert keys == ["OK"]
    assert text == "OK=1\n"
    assert skipped == ["CR", "PEM"]


def test_output_is_sorted_by_key():
    text, keys = render({"ZETA": "1", "ALPHA": "2"}, {"MIDDLE": "3", "BETA": "4"})
    assert keys == ["ALPHA", "BETA", "MIDDLE", "ZETA"]
    assert text.splitlines() == ["ALPHA=2", "BETA=4", "MIDDLE=3", "ZETA=1"]
    assert text.endswith("\n")


def test_empty_or_missing_inputs_render_nothing():
    assert render({}, {}) == ("", [])
    assert render(None, None) == ("", [])


def test_non_string_json_scalars_are_rendered_as_json():
    text, _ = render({}, {"FLAG": True, "N": 5, "NONE": None})
    assert text == "FLAG=true\nN=5\nNONE=\n"


def test_main_writes_a_private_file_and_prints_only_key_names(tmp_path, monkeypatch, capsys):
    out = tmp_path / "prod.env"
    monkeypatch.setenv("SECRETS_JSON", json.dumps({
        "TELEGRAM_BOT_TOKEN": "123:secret-token",
        "SERVER_ADDR": "10.0.0.1",
        "github_token": "ghs_x",
        "MULTI_LINE": "a\nb",
    }))
    monkeypatch.setenv("VARS_JSON", json.dumps({"INVITE_LIMIT": "3", "TELEGRAM_BOT_TOKEN": "ignored"}))

    assert render_env.main([str(out)]) == 0

    assert out.read_text() == "INVITE_LIMIT=3\nTELEGRAM_BOT_TOKEN=123:secret-token\n"
    assert oct(out.stat().st_mode & 0o777) == "0o600"
    captured = capsys.readouterr()
    assert "Rendered 2 keys" in captured.out
    assert "INVITE_LIMIT" in captured.out and "TELEGRAM_BOT_TOKEN" in captured.out
    assert "skipped MULTI_LINE" in captured.err
    for value in ("secret-token", "10.0.0.1", "ghs_x", "ignored"):
        assert value not in captured.out + captured.err


def test_main_fails_when_there_is_nothing_to_render(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SECRETS_JSON", json.dumps({"github_token": "x", "SERVER_ADDR": "h"}))
    monkeypatch.delenv("VARS_JSON", raising=False)
    assert render_env.main([str(tmp_path / "prod.env")]) == 1
    assert "::error::" in capsys.readouterr().err


def test_main_rejects_json_that_is_not_an_object(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRETS_JSON", "[1, 2]")
    monkeypatch.setenv("VARS_JSON", "{}")
    try:
        render_env.main([str(tmp_path / "prod.env")])
    except SystemExit as exc:
        assert "SECRETS_JSON" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_main_usage_error_without_output_path(capsys):
    assert render_env.main([]) == 2
    assert "usage" in capsys.readouterr().err
