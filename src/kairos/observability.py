"""本地 JSONL 观测日志。

日志只在调用方显式传入路径时启用；每行都是一个完整 JSON 对象，方便用
jq、Python 或后续评估脚本逐行读取。日志内容可能包含用户输入和工具参数，
因此默认不写入文件，也不上传到远程服务。
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .images import redact_images


class JsonlLogger:
    """把 Agent 事件追加写入本地 JSONL 文件。"""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            **redact_images(event),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
