from datetime import datetime, timedelta

from kairos.schedule.store import ScheduleStore
from kairos.storage import open_db
from kairos.tools.schedules import schedule_tools


def test_create_update_list_schedule(tmp_path):
    db = open_db(tmp_path / "kairos.db")
    tools = schedule_tools(ScheduleStore(db)).handlers
    run_at = (datetime.now().astimezone() + timedelta(hours=2)).replace(second=0, microsecond=0)

    created = tools["create_schedule"]("交房租", "message", "该交房租了", at=run_at.isoformat())
    # 模型常把没改的字段传成 null，不能当成“要改时间”
    tools["update_schedule"](created["id"], status="paused", at=None, cron=None)

    (listed,) = tools["list_schedules"]()
    assert (listed["channel"], listed["status"]) == ("weixin", "paused")
    assert listed["next_run_at"] == run_at.isoformat(timespec="minutes")  # 显示为本地时间
