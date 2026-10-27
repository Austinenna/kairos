from datetime import datetime, timedelta

import pytest

from kairos.schedule.store import ScheduleStore, next_cron_run
from kairos.storage import open_db


@pytest.fixture
def store(tmp_path):
    db = open_db(tmp_path / "kairos.db")
    yield ScheduleStore(db)
    db.close()


def test_once_schedule_runs_once_then_done(store):
    run_at = datetime.now().astimezone() + timedelta(hours=1)
    schedule = store.create("交房租", "message", "该交房租了", "weixin", at=run_at.isoformat())

    assert store.due(run_at - timedelta(minutes=1)) == []
    (due,) = store.due(run_at)
    assert due.id == schedule.id

    store.advance(due, run_at)
    assert store.get(schedule.id).status == "done"
    assert store.due(run_at + timedelta(days=1)) == []


def test_cron_schedule_catches_up_only_once(store):
    """错过好几天后，只补跑一次，下一次从现在算起。"""
    schedule = store.create("早报", "prompt", "列出今天的待办", "weixin", cron="0 8 * * *")
    three_days_later = datetime.now().astimezone() + timedelta(days=3)

    (due,) = store.due(three_days_later)
    store.advance(due, three_days_later)

    assert store.get(schedule.id).next_run_at == next_cron_run("0 8 * * *", three_days_later)
    assert store.due(three_days_later) == []


def test_next_cron_run_uses_local_time():
    seven = datetime(2026, 10, 28, 7, 0).astimezone()
    assert next_cron_run("0 8 * * *", seven) == next_cron_run("0 8 * * *", seven.replace(hour=7, minute=59))
    assert datetime.fromisoformat(next_cron_run("0 8 * * *", seven)).astimezone().hour == 8


@pytest.mark.parametrize(
    "trigger",
    [
        {"at": "2020-01-01T08:00:00+08:00"},  # 已经过去
        {"at": "2099-01-01T08:00:00"},  # 没有时区
        {"cron": "每天八点"},
        {},
        {"at": "2099-01-01T08:00:00+08:00", "cron": "0 8 * * *"},
    ],
)
def test_create_rejects_bad_trigger(store, trigger):
    with pytest.raises(ValueError):
        store.create("坏任务", "message", "内容", "weixin", **trigger)
