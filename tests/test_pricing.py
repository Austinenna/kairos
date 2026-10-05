from datetime import UTC, datetime

from kairos.pricing import Usage, estimate_cost, is_peak


def test_peak_hours_are_weekday_utc_windows():
    monday = datetime(2026, 10, 5, tzinfo=UTC)  # 周一
    assert is_peak(monday.replace(hour=1))
    assert is_peak(monday.replace(hour=9, minute=59))
    assert not is_peak(monday.replace(hour=4))
    assert not is_peak(monday.replace(hour=12))
    saturday = datetime(2026, 10, 10, hour=2, tzinfo=UTC)
    assert not is_peak(saturday)


def test_cost_counts_cached_input_separately():
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=400_000, output_tokens=100_000, reasoning_tokens=0)
    off_peak = datetime(2026, 10, 5, hour=12, tzinfo=UTC)
    # 40 万命中 × 0.003 + 60 万未命中 × 0.15 + 10 万输出 × 0.60，单位是每百万 token
    expected = 0.4 * 0.003 + 0.6 * 0.15 + 0.1 * 0.60
    assert abs(estimate_cost("deepseek-flash", usage, off_peak) - expected) < 1e-9


def test_unknown_model_has_no_price():
    usage = Usage(input_tokens=10, cached_input_tokens=0, output_tokens=10, reasoning_tokens=0)
    assert estimate_cost("some-other-model", usage) is None
