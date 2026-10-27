"""定时任务工具：在聊天里新建、查看、修改、删除定时推送。实际逻辑在 schedule/store.py。"""

from datetime import datetime

from ..schedule.store import Schedule, ScheduleStore
from .registry import Toolset, tool_schema

TRIGGER_PROPERTIES = {
    "at": {"type": "string", "description": "一次性任务的触发时间，带时区的 ISO 8601，例如 2026-10-28T15:00:00+08:00"},
    "cron": {"type": "string", "description": "周期任务的 5 段 cron 表达式，按用户本地时间，例如每天 8 点是 0 8 * * *"},
}

CREATE_SCHEDULE = tool_schema(
    "create_schedule",
    "新建定时任务，到点后由程序推送给用户。at 和 cron 二选一。",
    {
        "title": {"type": "string", "description": "简短标题，例如“早报”“交房租”"},
        "kind": {"type": "string", "enum": ["message", "prompt"], "description": "message 推送原文；prompt 到点执行"},
        "content": {"type": "string", "description": "message：推送给用户的原文；prompt：到点后执行的完整指令"},
        **TRIGGER_PROPERTIES,
        "channel": {"type": "string", "enum": ["weixin", "terminal"], "description": "推送到哪里，默认 weixin"},
    },
    required=["title", "kind", "content"],
)

LIST_SCHEDULES = tool_schema(
    "list_schedules", "列出全部定时任务：内容、下次触发时间、状态、上次执行结果和原因。", {}, required=[]
)

UPDATE_SCHEDULE = tool_schema(
    "update_schedule",
    "修改定时任务，只传要改的字段。改时间时 at 和 cron 二选一；暂停用 status=paused，恢复用 status=active。",
    {
        "schedule_id": {"type": "integer", "description": "任务编号"},
        "title": {"type": "string"},
        "kind": {"type": "string", "enum": ["message", "prompt"]},
        "content": {"type": "string"},
        **TRIGGER_PROPERTIES,
        "channel": {"type": "string", "enum": ["weixin", "terminal"]},
        "status": {"type": "string", "enum": ["active", "paused"]},
    },
    required=["schedule_id"],
)

DELETE_SCHEDULE = tool_schema(
    "delete_schedule", "永久删除一个定时任务。", {"schedule_id": {"type": "integer", "description": "任务编号"}}
)

RULES = """定时任务规则：
- 用户要求在某个时间、或周期性地提醒、通知、汇报时，用 create_schedule 建定时任务。
  到点后由程序推送，不需要你在对话里等待。
- 任务执行时看不到任何聊天历史。content 必须单独就能看懂：
  对话里要用到的内容（原话、名字、数字、约定）原样写进去，不要写“刚才”“上面”“今天聊的”这类指代。
- 内容现在就能确定的，kind 用 message，content 是到点直接发给用户的原文，写成对用户说的话。
- 需要到点后再查数据（待办、记忆、历史会话），或要判断这次该不该通知的，kind 用 prompt，content 写成给执行者的完整指令。
  带条件的要求（如“有过期待办才提醒”“10 点以后就不用发了”）把条件原样写进去，并写明不满足时不通知。
- 一次性任务用 at，周期任务用 cron。channel 默认 weixin，用户要求只在终端显示时才用 terminal。
- 用户只说“提醒我”时只建定时任务；用户同时要求记成待办时，再另外新建待办。
- 新建或修改后，把标题、时间、类型和内容原样告诉用户，请用户确认。删除前先用 list_schedules 确认编号，不猜测。"""


def schedule_tools(store: ScheduleStore) -> Toolset:
    def create_schedule(
        title: str, kind: str, content: str, at: str | None = None, cron: str | None = None, channel: str = "weixin"
    ) -> dict:
        return _view(store.create(title, kind, content, channel, at=at, cron=cron))

    def list_schedules() -> list[dict]:
        return [_view(schedule) for schedule in store.list_all()]

    def update_schedule(schedule_id: int, **changes: str) -> dict:
        # 模型有时会把没改的字段传成 null，只保留真正要改的
        return _view(store.update(schedule_id, **{key: value for key, value in changes.items() if value is not None}))

    def delete_schedule(schedule_id: int) -> dict:
        return {"deleted": _view(store.delete(schedule_id))}

    return Toolset(
        schemas=[CREATE_SCHEDULE, LIST_SCHEDULES, UPDATE_SCHEDULE, DELETE_SCHEDULE],
        handlers={
            "create_schedule": create_schedule,
            "list_schedules": list_schedules,
            "update_schedule": update_schedule,
            "delete_schedule": delete_schedule,
        },
        instructions=[RULES],
    )


def _view(schedule: Schedule) -> dict:
    """给模型看的任务信息：时间换成本地时间，已结束的任务不显示下次时间。"""
    return {
        "id": schedule.id,
        "title": schedule.title,
        "kind": schedule.kind,
        "content": schedule.content,
        "cron": schedule.cron,
        "channel": schedule.channel,
        "status": schedule.status,
        "next_run_at": None if schedule.status == "done" else _local(schedule.next_run_at),
        "last_run_at": schedule.last_run_at and _local(schedule.last_run_at),
        "last_result": schedule.last_result,
        "last_note": schedule.last_note,
    }


def _local(utc_text: str) -> str:
    return datetime.fromisoformat(utc_text).astimezone().isoformat(timespec="minutes")
