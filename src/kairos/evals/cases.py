"""失败用例：觉得某轮回答不对时发 /bad 备注，把这一轮连同当时的上下文存成 ~/.kairos/evals/cases/ 下的一个 JSON 文件。

先攒真实的失败，攒到二三十条再做回放评测（以后放在 runner.py）。
文件是普通 JSON，可以直接打开查看，把 expected 补上。
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .. import config
from ..compaction.compact import load_context
from ..storage import MessageRecord, Session, SessionStore


def cases_dir() -> Path:
    return config.get_app_dir() / "evals" / "cases"


def save_case(sessions: SessionStore, session: Session, note: str, memory: str) -> Path | None:
    """把会话的最近一轮存成用例，返回文件路径；会话里还没有对话时返回 None。

    文件名是“这一轮的开始时间-会话 ID 前 8 位”，对同一轮再发一次 /bad 会覆盖，相当于改备注。

    文件内容:
        note: 用户的备注，哪里不对。
        expected: 期望的表现，先留空，以后补。
        session: 会话 ID、入口、模型、思考模式。
        memory: 保存时的记忆快照（每次请求都会带上）。不一定和当时完全一样，但 /bad 通常紧跟着那一轮，差别不大。
        context: 这一轮之前发给模型的历史；压缩过的会话是摘要加保留的原文。
        input: 这一轮的用户消息。
        trace: 这一轮之后的每条消息，带用量、费用和保存时间。
    """
    records = sessions.turn(session.id)
    if not records:
        return None
    user, steps = records[0], records[1:]
    case = {
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": note,
        "expected": "",
        "session": {key: asdict(session)[key] for key in ("id", "channel", "model", "thinking")},
        "memory": memory,
        "context": load_context(sessions, session.id, before=user.id),
        "input": user.message,
        "trace": [_step(record) for record in steps],
    }
    path = cases_dir() / f"{user.created_at:%Y%m%d-%H%M%S}-{session.id[:8]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _step(record: MessageRecord) -> dict:
    return {
        "message": record.message,
        "usage": asdict(record.usage) if record.usage else None,
        "cost_usd": record.cost_usd,
        "created_at": record.created_at.isoformat(),
    }
