"""ContainerManager.run_script: bundled sandbox scripts run from a unique workspace copy, always in a
one-off container (run_one_shot) and without the saved-package replay."""
import importlib
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

cm_module = importlib.import_module("secure_container.container_manager")
ContainerManager = cm_module.ContainerManager

BANNER = "Secure container environment initialized\nExecuting command: python x\n"


@pytest.fixture
def manager(tmp_path):
    """A ContainerManager that never touched Docker: the constructor needs a daemon."""
    cm = ContainerManager.__new__(ContainerManager)
    cm.base_data_path = tmp_path / "data"
    cm.host_data_path = tmp_path / "data"
    cm.image_name, cm.image_tag, cm.container_prefix = "secure-container", "latest", "secure-container-"
    return cm


def test_scripts_dir_holds_the_bundled_scripts():
    assert cm_module.SCRIPTS_DIR.is_dir()
    assert (cm_module.SCRIPTS_DIR / "browse.py").is_file()
    assert (cm_module.SCRIPTS_DIR / "render_document.py").is_file()


def test_run_script_copies_quotes_and_cleans_up(manager, monkeypatch):
    calls = []

    def fake_run_one_shot(user_id, command, timeout=60, network_enabled=False, mem_limit=None, **kw):
        parts = shlex.split(command)
        copied = manager.base_data_path / user_id / Path(parts[2]).name
        calls.append(SimpleNamespace(
            user_id=user_id, parts=parts, timeout=timeout, network_enabled=network_enabled, mem_limit=mem_limit,
            existed=copied.exists(), content=copied.read_text() if copied.exists() else None, kw=kw,
        ))
        return BANNER + '{"ok": true}\n'

    monkeypatch.setattr(manager, "run_one_shot", fake_run_one_shot)
    monkeypatch.setattr(manager, "run_command", lambda *a, **k: pytest.fail("run_script never uses the session"))

    logs = manager.run_script("7", "browse.py", ["https://x.y/?a=1&b=2", "--screenshot", "downloads/s p.png"],
                              timeout=75, network_enabled=True, mem_limit="1g")

    assert logs.endswith('{"ok": true}\n')
    call = calls[0]
    assert call.user_id == "7" and call.timeout == 75 and call.network_enabled is True and call.mem_limit == "1g"
    assert call.kw == {"replay_packages": False}  # bundled scripts need no saved packages
    assert call.parts[:2] == ["python", "-I"]  # isolated: workspace files never shadow the script's imports
    assert call.parts[2].startswith("/home/runner/workspace/temp_tool_") and call.parts[2].endswith(".py")
    assert call.parts[3:] == ["https://x.y/?a=1&b=2", "--screenshot", "downloads/s p.png"]  # shell-safe quoting
    # the copy existed with the real script's content while the command ran, and is gone afterwards
    assert call.existed and call.content == (cm_module.SCRIPTS_DIR / "browse.py").read_text()
    assert list((manager.base_data_path / "7").glob("temp_tool_*")) == []


def test_run_script_defaults_and_unique_names(manager, monkeypatch):
    names = []
    monkeypatch.setattr(manager, "run_one_shot", lambda uid, cmd, timeout=60, network_enabled=False, mem_limit=None, **kw:
                        names.append((shlex.split(cmd)[2], timeout, network_enabled, mem_limit)) or "")
    for _ in range(3):
        manager.run_script("7", "render_document.py")
    assert len({n[0] for n in names}) == 3  # concurrent calls must not share a file name
    assert all(n[1:] == (90, False, None) for n in names)


def test_workspace_files_named_like_stdlib_modules_do_not_break_the_bundled_scripts(manager, monkeypatch):
    """The script's directory is the user's workspace, where the model writes files. A ``json.py`` or
    ``typing.py`` there used to replace the real module, so every pdf/docx/xlsx render and every
    browse died before printing its JSON line. The command is run here for real, with the container
    path mapped to the workspace on disk."""
    import subprocess
    import sys

    from agents.web_tools import last_json_line

    user_dir = manager.base_data_path / "7"
    user_dir.mkdir(parents=True)
    for module in ("json", "typing", "random", "copy", "zipfile"):
        (user_dir / f"{module}.py").write_text(f"raise SystemExit('planted {module}.py was imported')\n")
    (user_dir / "in.md").write_text("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")

    def run_locally(user_id, command, timeout=60, network_enabled=False, mem_limit=None, **kw):
        parts = shlex.split(command)
        assert parts[0] == "python"
        parts = [sys.executable] + [p.replace("/home/runner/workspace/", f"{user_dir}/") for p in parts[1:]]
        done = subprocess.run(parts, cwd=user_dir, capture_output=True, text=True, timeout=60)
        return BANNER + done.stdout + done.stderr

    monkeypatch.setattr(manager, "run_one_shot", run_locally)
    for fmt in ("docx", "xlsx"):
        logs = manager.run_script("7", "render_document.py", ["--format", fmt, "--input", "in.md", "--output", f"out.{fmt}"])
        data = last_json_line(logs)
        assert data and data["error"] is None and data["output"] == f"out.{fmt}", logs
        assert (user_dir / f"out.{fmt}").stat().st_size > 0


def test_run_script_cleans_up_when_run_command_raises(manager, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("docker down")

    monkeypatch.setattr(manager, "run_one_shot", boom)
    with pytest.raises(RuntimeError):
        manager.run_script("7", "browse.py", ["https://x"])
    assert list((manager.base_data_path / "7").glob("temp_tool_*")) == []


def test_run_script_refuses_unknown_scripts(manager, monkeypatch):
    monkeypatch.setattr(manager, "run_one_shot", lambda *a, **k: pytest.fail("must not run"))
    assert manager.run_script("7", "nope.py").startswith("Error: sandbox script not found")
    assert manager.run_script("7", "").startswith("Error: sandbox script not found")
    assert manager.run_script("7", None).startswith("Error: sandbox script not found")


def test_run_script_accepts_only_bare_names_of_bundled_scripts(manager, monkeypatch, tmp_path):
    """Anything with a directory part is refused before the filesystem is consulted, even when the
    file exists: ``../container_manager.py`` does exist next to SCRIPTS_DIR, and so does the
    absolute path of a bundled script."""
    monkeypatch.setattr(manager, "run_one_shot", lambda *a, **k: pytest.fail("must not run"))
    script = tmp_path / "custom.py"
    script.write_text("print('hi')\n")
    assert (cm_module.SCRIPTS_DIR / "../container_manager.py").is_file()
    for path in ("../container_manager.py", str(script), str(cm_module.SCRIPTS_DIR / "browse.py"),
                 "scripts/browse.py", "/etc/passwd", "browse.py/"):
        assert manager.run_script("7", path).startswith("Error: sandbox script not found"), path


class FakeContainer:
    def __init__(self, logs=b"out", status=0):
        self._logs, self._status, self.removed = logs, status, False

    def wait(self, timeout=None):
        return {"StatusCode": self._status}

    def logs(self):
        return self._logs

    def remove(self, force=False):
        self.removed = True


def _fake_docker(container):
    runs = []

    def run(**kwargs):
        runs.append(kwargs)
        return container

    containers = SimpleNamespace(run=run, get=lambda name: container)
    return SimpleNamespace(containers=containers), runs


def test_mem_limit_reaches_docker_run_and_defaults_to_512m(manager):
    container = FakeContainer(logs=b"hello")
    manager.docker_client, runs = _fake_docker(container)

    assert manager.run_command("7", "python x.py") == "hello"
    assert runs[0]["mem_limit"] == cm_module.DEFAULT_MEM_LIMIT == "512m"
    assert runs[0]["network_mode"] == "none"

    manager.run_command("7", "python x.py", 30, True, "1g")
    assert runs[1]["mem_limit"] == "1g" and runs[1]["network_mode"] == "bridge"
    assert runs[1]["volumes"] == {
        str((manager.host_data_path / "7").absolute()): {"bind": "/home/runner/workspace", "mode": "rw", "propagation": "shared"}
    }
    assert container.removed


def test_run_command_keyword_signature_is_backwards_compatible(manager):
    """The patcher's wrappers call run_command(user_id, command, timeout, network_enabled) positionally."""
    manager.docker_client, runs = _fake_docker(FakeContainer(status=2, logs=b"bad"))
    result = manager.run_command("7", "false", 10, False)
    assert result.startswith("Error (exit code 2):") and runs[0]["mem_limit"] == "512m"
