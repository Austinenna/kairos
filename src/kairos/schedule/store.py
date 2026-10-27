"""定时任务存储：schedules 表的增删改查，以及计算下次触发时间。

一条任务 = 什么时候触发 + 推什么 + 推到哪里。
  cron 有值：周期任务，下次时间用 croniter 计算；cron 为空：一次性任务，时间就是 next_run_at。
  kind = message：content 是推送原文；kind = prompt：content 是到点后交给模型执行的提示词。

时间统一存成 UTC 的 ISO 文本，数据库里可以直接按字符串比较先后。
"""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from croniter import croniter

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    kind        TEXT NOT NULL,             -- message / prompt
    content     TEXT NOT NULL,
    cron        TEXT,                      -- NULL 表示一次性任务
    channel     TEXT NOT NULL,             -- weixin / terminal
    next_run_at TEXT NOT NULL,             -- UTC
    status      TEXT NOT NULL,             -- active / paused / done
    last_run_at TEXT,
    last_result TEXT,                      -- sent / skipped / failed
    last_note   TEXT,                      -- 跳过的原因或错误信息
    created_at  TEXT NOT NULL
);
"""

KINDS = ("message", "prompt")
CHANNELS = ("weixin", "terminal")


@dataclass
class Schedule:
    id: int
    title: str
    kind: str
    content: str
    cron: str | None
    channel: str
    next_run_at: str
    status: str
    last_run_at: str | None
    last_result: str | None
    last_note: str | None
    created_at: str


class ScheduleStore:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(SCHEMA)

    def create(
        self, title: str, kind: str, content: str, channel: str, *, at: str | None = None, cron: str | None = None
    ) -> Schedule:
        """新建任务。at（带时区的 ISO 时间）和 cron 必须二选一。参数来自模型，在这里校验。"""
        _check_choice("kind", kind, KINDS)
        _check_choice("channel", channel, CHANNELS)
        if not title.strip() or not content.strip():
            raise ValueError("title 和 content 不能为空")
        next_run_at = _first_run(at, cron)
        now = _utc_now()
        cursor = self.db.execute(
            "INSERT INTO schedules (title, kind, content, cron, channel, next_run_at, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'active', ?)",
            (title.strip(), kind, content, cron, channel, next_run_at, now),
        )
        self.db.commit()
        return self.get(cursor.lastrowid)

    def get(self, schedule_id: int) -> Schedule:
        row = self.db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
        if row is None:
            raise ValueError(f"没有编号为 {schedule_id} 的定时任务")
        return Schedule(**row)

    def list_all(self) -> list[Schedule]:
        """全部任务，已结束的一次性任务也列出，方便查看上次结果。"""
        return [Schedule(**row) for row in self.db.execute("SELECT * FROM schedules ORDER BY id")]

    def update(self, schedule_id: int, **changes: str) -> Schedule:
        """修改任务。可改 title、kind、content、channel、status（active / paused），以及时间 at 或 cron。"""
        schedule = self.get(schedule_id)
        fields = {key: changes[key] for key in ("title", "kind", "content", "channel", "status") if key in changes}
        _check_choice("kind", fields.get("kind", schedule.kind), KINDS)
        _check_choice("channel", fields.get("channel", schedule.channel), CHANNELS)
        _check_choice("status", fields.get("status", schedule.status), ("active", "paused"))
        if "at" in changes or "cron" in changes:
            fields["cron"] = changes.get("cron")
            fields["next_run_at"] = _first_run(changes.get("at"), changes.get("cron"))
        elif schedule.status == "paused" and fields.get("status") == "active" and schedule.cron:
            # 暂停期间错过的不补，从现在重新算下一次
            fields["next_run_at"] = next_cron_run(schedule.cron, datetime.now(UTC))
        if fields:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            self.db.execute(f"UPDATE schedules SET {assignments} WHERE id = ?", (*fields.values(), schedule_id))
            self.db.commit()
        return self.get(schedule_id)

    def delete(self, schedule_id: int) -> Schedule:
        schedule = self.get(schedule_id)
        self.db.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        self.db.commit()
        return schedule

    def due(self, now: datetime) -> list[Schedule]:
        """到期的任务：active 且 next_run_at 不晚于 now。"""
        rows = self.db.execute(
            "SELECT * FROM schedules WHERE status = 'active' AND next_run_at <= ? ORDER BY next_run_at",
            (_utc(now),),
        )
        return [Schedule(**row) for row in rows]

    def advance(self, schedule: Schedule, now: datetime) -> None:
        """执行前调用：周期任务算出 now 之后的下一次，一次性任务标记为 done。先存再执行，崩溃也不会重复触发。"""
        if schedule.cron:
            self.db.execute(
                "UPDATE schedules SET next_run_at = ? WHERE id = ?", (next_cron_run(schedule.cron, now), schedule.id)
            )
        else:
            self.db.execute("UPDATE schedules SET status = 'done' WHERE id = ?", (schedule.id,))
        self.db.commit()

    def record(self, schedule_id: int, result: str, note: str | None, now: datetime) -> None:
        """执行后记录结果：sent / skipped / failed，note 是跳过原因或错误信息。"""
        self.db.execute(
            "UPDATE schedules SET last_run_at = ?, last_result = ?, last_note = ? WHERE id = ?",
            (_utc(now), result, note, schedule_id),
        )
        self.db.commit()


def next_cron_run(cron: str, after: datetime) -> str:
    """cron 表达式在 after 之后的下一次触发时间（UTC 文本）。按本地时区理解 cron，"0 8 * * *" 是本地早上 8 点。"""
    return _utc(croniter(cron, after.astimezone()).get_next(datetime))


def _first_run(at: str | None, cron: str | None) -> str:
    """新建或改时间时，算出第一次触发时间。"""
    if (at is None) == (cron is None):
        raise ValueError("at 和 cron 必须提供且只提供一个")
    now = datetime.now(UTC)
    if cron is not None:
        if not croniter.is_valid(cron):
            raise ValueError(f"cron 表达式无效：{cron}")
        return next_cron_run(cron, now)
    run_at = datetime.fromisoformat(at)
    if run_at.tzinfo is None:
        raise ValueError("at 必须带时区，例如 2026-10-28T15:00:00+08:00")
    if run_at <= now:
        raise ValueError(f"at 已经过去了：{at}")
    return _utc(run_at)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} 只能是 {' / '.join(choices)}，收到：{value}")


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _utc_now() -> str:
    return _utc(datetime.now(UTC))
