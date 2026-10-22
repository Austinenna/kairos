"""一次性脚本：把旧版的微信收消息游标转到新版的 ~/.kairos/channels/weixin/state.json。

没有游标的话，新版第一次启动可能把以前的消息再回复一遍。旧数据库不做改动。
运行：uv run python scripts/convert_weixin_state.py
"""

import json
import sqlite3
from pathlib import Path

WEIXIN = Path.home() / ".kairos" / "channels" / "weixin"
OLD_DB = Path.home() / ".kairos" / "channels" / "channels.sqlite3"


def main() -> None:
    state_path = WEIXIN / "state.json"
    if state_path.exists():
        print(f"{state_path} 已存在，不做转换。")
        return
    owner = json.loads((WEIXIN / "account.json").read_text())["owner_id"]
    row = (
        sqlite3.connect(OLD_DB)
        .execute("SELECT checkpoint FROM channel_checkpoints WHERE channel = 'weixin'")
        .fetchone()
    )
    checkpoint = json.loads(row[0])
    state = {"cursor": checkpoint.get("cursor", "")}
    if token := checkpoint.get("context_tokens", {}).get(owner):
        state["context_token"] = token
    state_path.write_text(json.dumps(state), encoding="utf-8")
    has_token = "context_token" in state
    print(f"已写入 {state_path}：游标 {'有' if state['cursor'] else '无'}，回复令牌 {'有' if has_token else '无'}")


main()
