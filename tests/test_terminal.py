from kairos.terminal import app


def test_two_quick_ctrl_c_mean_exit(monkeypatch):
    times = iter([100.0, 103.0, 104.5])
    monkeypatch.setattr(app.time, "monotonic", lambda: next(times))
    ctrl_c = app.DoubleCtrlC()

    assert ctrl_c.pressed() is False  # 第一次
    assert ctrl_c.pressed() is False  # 隔了 3 秒，重新算第一次
    assert ctrl_c.pressed() is True  # 1.5 秒内又按了一次


def test_status_line_shows_context_against_compaction_threshold():
    from types import SimpleNamespace

    from kairos.models import Usage
    from kairos.storage import SessionTotals

    conversation = SimpleNamespace(
        session=SimpleNamespace(model="deepseek-flash", thinking=False),
        totals=lambda: SessionTotals(Usage(0, 0, 0, 0), 0.0, 0),
        context_tokens=lambda: 480_000,
        compact_threshold=lambda: 491_288,
    )

    assert "上下文 480k/491.3k (98%) 即将压缩" in app.status_line(conversation)
