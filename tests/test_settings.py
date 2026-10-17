import json

from bot.settings import (
    DEFAULT_SETTINGS,
    ITERATION_MAX,
    ITERATION_MIN,
    MODE_TYPES,
    PRESET_NAMES,
    SETTINGS_PRESETS,
    UserSettings,
    apply_preset,
    current_preset,
)


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


def test_rich_messages_default_on_and_backfilled_for_old_files(tmp_path):
    user_dir = tmp_path / "9"
    user_dir.mkdir()
    (user_dir / "settings.json").write_text(json.dumps({"thinking": {"enabled": False}}))
    s = UserSettings("9", base_dir=str(tmp_path))
    assert s.get("rich_messages", "enabled") is True
    s.set("rich_messages", False, "enabled")
    assert UserSettings("9", base_dir=str(tmp_path)).get("rich_messages", "enabled") is False


# ---- presets (F6) and the mode / ui keys (F3, F6) ----------------------------------
def test_mode_and_ui_defaults():
    assert DEFAULT_SETTINGS["mode"] == {"type": "auto"}
    assert DEFAULT_SETTINGS["ui"] == {"advanced": False}
    assert MODE_TYPES == ("auto", "speed", "intelligence")


def test_presets_are_well_formed():
    assert PRESET_NAMES == ("fast", "balanced", "deep")
    for name, values in SETTINGS_PRESETS.items():
        assert values[("mode", "type")] in MODE_TYPES, name
        assert isinstance(values[("thinking", "enabled")], bool), name
        assert ITERATION_MIN <= values[("tools", "max_iteration")] <= ITERATION_MAX, name
        for (category, key) in values:
            assert key in DEFAULT_SETTINGS[category], (name, category, key)
    # Balanced is the defaults: a new user already has it.
    assert all(DEFAULT_SETTINGS[c][k] == v for (c, k), v in SETTINGS_PRESETS["balanced"].items())


def test_presets_apply_detect_and_round_trip(tmp_path):
    s = UserSettings("5", base_dir=str(tmp_path))
    s.set("image", "fast", "engine")  # an unrelated choice a preset must leave alone
    assert current_preset(s) == "balanced"

    assert apply_preset(s, "fast") is True
    assert (s.get("mode", "type"), s.get("thinking", "enabled"), s.get("tools", "max_iteration")) == ("speed", False, 10)
    assert s.get("image", "engine") == "fast"
    again = UserSettings("5", base_dir=str(tmp_path))  # persisted in one save
    assert current_preset(again) == "fast"
    assert apply_preset(again, "fast") is False  # already active: nothing changed

    again.set("judge", True, "enabled")
    assert apply_preset(again, "DEEP") is True  # names are case-insensitive
    assert (again.get("mode", "type"), again.get("thinking", "enabled"), again.get("tools", "max_iteration")) == ("intelligence", True, 40)
    assert again.get("judge", "enabled") is False  # Deep switches the judge off
    assert current_preset(UserSettings("5", base_dir=str(tmp_path))) == "deep"

    assert apply_preset(again, "balanced") is True
    assert UserSettings("5", base_dir=str(tmp_path)).as_dict() == {**DEFAULT_SETTINGS, "image": {**DEFAULT_SETTINGS["image"], "engine": "fast"}}


def test_a_custom_state_matches_no_preset_and_unknown_presets_change_nothing(tmp_path):
    s = UserSettings("6", base_dir=str(tmp_path))
    s.set("tools", 11, "max_iteration")
    assert current_preset(s) is None
    before = s.as_dict()
    assert apply_preset(s, "turbo") is False
    assert apply_preset(s, None) is False
    assert s.as_dict() == before
    # every key must match: one differing value is a custom state
    s.set("tools", 40, "max_iteration")
    s.set("mode", "intelligence", "type")
    assert current_preset(s) == "deep"
    s.set("judge", True, "enabled")  # Deep means judge off
    assert current_preset(s) is None
    s.set("judge", False, "enabled")
    s.set("thinking", False, "enabled")
    assert current_preset(s) is None


def test_apply_preset_repairs_a_hand_edited_category(tmp_path):
    user_dir = tmp_path / "8"
    user_dir.mkdir()
    (user_dir / "settings.json").write_text(json.dumps({"mode": "speed", "tools": {"max_iteration": "lots"}}))
    s = UserSettings("8", base_dir=str(tmp_path))
    assert current_preset(s) is None
    assert apply_preset(s, "fast") is True
    assert UserSettings("8", base_dir=str(tmp_path)).get("mode", "type") == "speed"
    assert current_preset(UserSettings("8", base_dir=str(tmp_path))) == "fast"
