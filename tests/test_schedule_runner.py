from datetime import datetime, timedelta
from threading import RLock

import pytest

from kairos.schedule.runner import ScheduleRunner
from kairos.schedule.store import ScheduleStore
from kairos.storage import open_db


@pytest.fixture
def store(tmp_path):
    db = open_db(tmp_path / "kairos.db")
    yield ScheduleStore(db)
    db.close()


def in_one_hour() -> datetime:
    return datetime.now().astimezone().replace(microsecond=0) + timedelta(hours=1)


def test_message_is_pushed_once(store):
    run_at = in_one_hour()
    schedule = store.create("交房租", "message", "该交房租了", "weixin", at=run_at.isoformat())
    pushed = []
    runner = ScheduleRunner(store, RLock(), run_prompt=None, outlets={"weixin": pushed.append})

    runner.tick(run_at)
    runner.tick(run_at + timedelta(minutes=1))

    assert pushed == ["【交房租】\n该交房租了"]
    saved = store.get(schedule.id)
    assert (saved.status, saved.last_result) == ("done", "sent")


def test_prompt_can_skip_push(store):
    run_at = in_one_hour()
    schedule = store.create("过期待办", "prompt", "有过期待办才提醒我", "weixin", at=run_at.isoformat())
    pushed = []

    def run_prompt(title, prompt, tools):
        tools.handlers["skip_push"](reason="没有过期待办")
        return "好的，这次不通知。"  # 调用 skip_push 后模型通常还会回一句，程序忽略它

    ScheduleRunner(store, RLock(), run_prompt, {"weixin": pushed.append}).tick(run_at)

    assert pushed == []
    saved = store.get(schedule.id)
    assert (saved.last_result, saved.last_note) == ("skipped", "没有过期待办")


def test_late_run_tells_model_and_marks_message(store):
    run_at = in_one_hour()
    store.create("早报", "prompt", "列出今天的待办", "weixin", cron=f"{run_at.minute} {run_at.hour} * * *")
    store.create("开会", "message", "该开会了", "weixin", at=run_at.isoformat())
    prompts, pushed = [], []

    def run_prompt(title, prompt, tools):
        prompts.append(prompt)
        return "今天没有待办。"

    ScheduleRunner(store, RLock(), run_prompt, {"weixin": pushed.append}).tick(run_at + timedelta(hours=14))

    assert "迟到" in prompts[0] and prompts[0].endswith("列出今天的待办")
    assert pushed[0] == "【早报】\n今天没有待办。"
    assert pushed[1].startswith("【开会】\n（原定") and pushed[1].endswith("该开会了")
