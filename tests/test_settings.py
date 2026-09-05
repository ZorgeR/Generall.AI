import json

from bot.settings import DEFAULT_SETTINGS, UserSettings


def test_defaults_are_present_for_new_user(tmp_path):
    s = UserSettings("1", base_dir=str(tmp_path))
    assert s.settings == DEFAULT_SETTINGS
    assert s.settings is not DEFAULT_SETTINGS  # deep copy, never the shared default


def test_unknown_key_in_file_does_not_crash_and_is_preserved(tmp_path):
    user_dir = tmp_path / "42"
    user_dir.mkdir()
    (user_dir / "settings.json").write_text(json.dumps({
        "legacy_feature": {"enabled": True},
        "tools": {"max_iteration": 7},
        "note": "plain value",
    }))
    s = UserSettings("42", base_dir=str(tmp_path))
    assert s.get("legacy_feature", "enabled") is True
    assert s.get("tools", "max_iteration") == 7
    assert s.get("tools", "enabled") is True  # backfilled from defaults
    assert s.get("note") == "plain value"
    assert s.get("critique", "enabled") is False


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    user_dir = tmp_path / "7"
    user_dir.mkdir()
    (user_dir / "settings.json").write_text("{not json")
    s = UserSettings("7", base_dir=str(tmp_path))
    assert s.settings == DEFAULT_SETTINGS


def test_set_persists_and_reloads(tmp_path):
    s = UserSettings("9", base_dir=str(tmp_path))
    s.set("thinking", False, "enabled")
    s.set("system_prompt", "generall-ai-v1", "type")
    again = UserSettings("9", base_dir=str(tmp_path))
    assert again.get("thinking", "enabled") is False
    assert again.get("system_prompt", "type") == "generall-ai-v1"


def test_validation_clamps():
    assert UserSettings.validate_size(0) == 1
    assert UserSettings.validate_size(999) == 50
    assert UserSettings.validate_iteration(0, "tools") == 1
    assert UserSettings.validate_iteration(1000, "unknown-type") == 300
    assert UserSettings.validate_semantic_max_results(50) == 20
