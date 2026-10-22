"""构建请求视图：每次请求模型前，在历史末尾临时加上会变的信息。

为什么放在末尾：DeepSeek 等模型按前缀命中缓存，请求开头只要变了，整段历史都要重新计费。
所以不变的历史留在前面，会变的内容（时间，之后还有记忆）放在最后。
这些内容只存在于请求视图里，不写进会话历史。
"""

from datetime import datetime

WEEKDAYS = "一二三四五六日"


def build_request(messages: list[dict], *, now: datetime) -> list[dict]:
    """返回请求视图：原历史 + 一条说明当前时间的 system 消息。不修改原历史。"""
    return [*messages, {"role": "system", "content": time_note(now)}]


def time_note(now: datetime) -> str:
    return (
        f"当前时间：{now.strftime('%Y-%m-%d %H:%M')}（星期{WEEKDAYS[now.weekday()]}），"
        f"时区 {now.tzname()}（UTC{now.strftime('%z')}）。"
        "理解用户消息里的“今天”“明天”等相对时间时以此为准。"
    )
