from kairos.terminal import app


def test_two_quick_ctrl_c_mean_exit(monkeypatch):
    times = iter([100.0, 103.0, 104.5])
    monkeypatch.setattr(app.time, "monotonic", lambda: next(times))
    ctrl_c = app.DoubleCtrlC()

    assert ctrl_c.pressed() is False  # 第一次
    assert ctrl_c.pressed() is False  # 隔了 3 秒，重新算第一次
    assert ctrl_c.pressed() is True  # 1.5 秒内又按了一次
