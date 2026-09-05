import importlib


def test_usage_rows_and_totals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import stats as stats_module

    importlib.reload(stats_module)  # STATS_DB is relative: a fresh database in tmp_path
    tracker = stats_module.StatsTracker()
    tracker._init_db()
    tracker.track_usage("1", model="claude-sonnet-5", api_calls=3, input_tokens=1000, output_tokens=200,
                        cache_read_tokens=9000, cache_write_tokens=500, tool_calls=4, duration_s=12.5, cost_usd=0.01)
    tracker.track_usage("1", model="claude-haiku-4-5", api_calls=1, input_tokens=300, output_tokens=50, cost_usd=0.001)
    tracker.track_usage("2", model="claude-sonnet-5", api_calls=1, input_tokens=100, output_tokens=10, cost_usd=0.002)

    everyone = tracker.get_usage(days=30)
    assert everyone["api_calls"] == 5 and everyone["input_tokens"] == 1400 and everyone["output_tokens"] == 260
    assert everyone["cache_read_tokens"] == 9000 and everyone["tool_calls"] == 4
    assert abs(everyone["cost_usd"] - 0.013) < 1e-9
    assert set(everyone["models"]) == {"claude-sonnet-5", "claude-haiku-4-5"}

    one = tracker.get_usage(user_id="1", days=None)
    assert one["api_calls"] == 4 and one["models"]["claude-sonnet-5"]["turns"] == 1
    assert tracker.get_usage(user_id="nobody")["api_calls"] == 0
    assert tracker.get_users_ranked_by_cost(days=30) == [("1", 0.011), ("2", 0.002)]


def test_format_usage_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from bot.handlers.stats_ui import format_usage_text

    text = format_usage_text({"api_calls": 5, "input_tokens": 1400, "output_tokens": 260, "cache_read_tokens": 9000,
                              "cache_write_tokens": 500, "tool_calls": 4, "cost_usd": 0.013,
                              "models": {"claude-sonnet-5": {"input_tokens": 1000, "output_tokens": 200, "cache_read_tokens": 9000, "cache_write_tokens": 500, "cost_usd": 0.01}}})
    assert "in *10.9k* (83% cached)" in text and "out *260*" in text and "$0.01" in text
    assert format_usage_text({}).startswith("🧮 Tokens: _no usage")
