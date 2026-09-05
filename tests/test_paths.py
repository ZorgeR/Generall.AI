import os

import pytest

from agents.paths import resolve_under


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "data" / "42"
    (base / "images").mkdir(parents=True)
    (base / "images" / "cat.jpg").write_bytes(b"x")
    (tmp_path / "data" / "41").mkdir()
    (tmp_path / "data" / "41" / "secret.txt").write_text("s")
    return base


def test_relative_sandbox_and_host_spellings_resolve_to_the_same_file(workspace):
    expected = (workspace / "images" / "cat.jpg").resolve()
    for spelling in (
        "images/cat.jpg",
        "./images/cat.jpg",
        "data/42/images/cat.jpg",
        "./data/42/images/cat.jpg",
        "/home/runner/workspace/images/cat.jpg",
        str(expected),
    ):
        assert resolve_under(workspace, spelling) == expected, spelling


def test_bare_name_is_looked_up_in_fallback_dirs(workspace):
    assert resolve_under(workspace, "cat.jpg", fallback_dirs=("images",)) == (workspace / "images" / "cat.jpg").resolve()
    assert resolve_under(workspace, "cat.jpg") is None


def test_escapes_are_rejected(workspace):
    for bad in ("../41/secret.txt", "images/../../41/secret.txt", "/etc/passwd", str((workspace.parent / "41" / "secret.txt").resolve())):
        assert resolve_under(workspace, bad) is None, bad
        assert resolve_under(workspace, bad, must_exist=False) is None, bad


def test_paths_for_new_files_stay_inside(workspace):
    assert resolve_under(workspace, "notes/new.txt", must_exist=False) == (workspace / "notes" / "new.txt").resolve()
    assert resolve_under(workspace, ".") == workspace.resolve()
    assert resolve_under(workspace, "") is None


def test_symlink_pointing_outside_is_rejected(workspace):
    link = workspace / "images" / "escape.jpg"
    os.symlink(workspace.parent / "41" / "secret.txt", link)
    assert resolve_under(workspace, "images/escape.jpg") is None
