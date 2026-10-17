import os

import pytest

from agents import paths
from agents.paths import regular_files, resolve_under


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


@pytest.fixture
def folder(workspace):
    """data/42/notes with a regular file, a non-matching name, bad UTF-8, a FIFO and links out."""
    notes = workspace / "notes"
    notes.mkdir()
    (notes / "a_note.json").write_text("mine", encoding="utf-8")
    (notes / "other.txt").write_text("not matched", encoding="utf-8")
    (notes / "b_note.json").write_bytes(b"\xff\xfe broken")
    os.mkfifo(notes / "c_note.json")
    secret = workspace.parent / "41" / "secret.txt"
    os.symlink(secret, notes / "d_note.json")
    os.symlink(workspace.parent / "41", notes / "e_note.json")
    return notes


@pytest.mark.parametrize("by_path", [False, True])
def test_regular_files_reads_only_regular_files_and_never_follows_links(folder, monkeypatch, by_path):
    if by_path:  # the fallback for platforms without O_NOFOLLOW / dir_fd
        monkeypatch.setattr(paths.os, "O_NOFOLLOW", 0, raising=False)
    assert [(p.name, text) for p, text in regular_files(folder, "*_note.json")] == [("a_note.json", "mine")]
    assert [p for p in regular_files(folder, "*_note.json")][0][0] == folder / "a_note.json"


@pytest.mark.parametrize("by_path", [False, True])
def test_regular_files_of_a_linked_or_missing_folder_is_empty(workspace, monkeypatch, by_path):
    if by_path:
        monkeypatch.setattr(paths.os, "O_NOFOLLOW", 0, raising=False)
    os.symlink(workspace.parent / "41", workspace / "linked")
    assert list(regular_files(workspace / "linked", "*")) == []
    assert list(regular_files(workspace / "missing", "*")) == []
    assert list(regular_files(workspace / "images" / "cat.jpg", "*")) == []  # a file, not a folder
