"""Each sandbox call gets its own workspace file (parallel tool calls must not collide)."""
from pathlib import Path


def _manager_class():
    import importlib

    return importlib.import_module("secure_container.container_manager").ContainerManager


def test_workspace_file_names_are_unique_per_call(tmp_path):
    """A fixed name like temp_setup.sh was overwritten by whichever concurrent call wrote last,
    so every container ran that one command and all callers got its output."""
    workspace_file = _manager_class()._workspace_file
    seen_host, seen_container = set(), set()
    for _ in range(50):
        host, in_container = workspace_file(tmp_path, "temp_setup", ".sh")
        assert isinstance(host, Path) and host.parent == tmp_path
        assert host.name.startswith("temp_setup_") and host.name.endswith(".sh")
        assert in_container == f"/home/runner/workspace/{host.name}"
        seen_host.add(host)
        seen_container.add(in_container)
    assert len(seen_host) == 50 and len(seen_container) == 50


def test_workspace_file_suffixes(tmp_path):
    workspace_file = _manager_class()._workspace_file
    for prefix, suffix in (("temp_code", ".py"), ("temp_script", ".sh")):
        host, in_container = workspace_file(tmp_path, prefix, suffix)
        assert host.name.startswith(f"{prefix}_") and host.name.endswith(suffix)
        assert in_container.endswith(host.name)
