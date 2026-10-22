"""构建请求视图：每次请求模型前，在历史外面临时加上规则、记忆和时间。

请求视图的顺序：
  [会话的 system 提示词] [工具使用规则] [……历史……] [记忆 + 当前时间]

为什么这样排：DeepSeek 等模型按前缀命中缓存，请求开头只要变了，整段历史都要重新计费。
所以不变的内容（提示词、规则、历史）放在前面，会变的内容（记忆、时间）放在最后。
这些附加内容只存在于请求视图里，不写进会话历史。
"""

from datetime import datetime

WEEKDAYS = "一二三四五六日"


def build_request(messages: list[dict], *, now: datetime, rules: str = "", memory: str = "") -> list[dict]:
    """返回请求视图，不修改原历史。messages[0] 是会话的 system 提示词。"""
    head = [messages[0]]
    if rules:
        head.append({"role": "system", "content": rules})
    tail = "\n\n".join(part for part in (memory, time_note(now)) if part)
    return [*head, *messages[1:], {"role": "system", "content": tail}]


def time_note(now: datetime) -> str:
    return (
        f"当前时间：{now.strftime('%Y-%m-%d %H:%M')}（星期{WEEKDAYS[now.weekday()]}），"
        f"时区 {now.tzname()}（UTC{now.strftime('%z')}）。"
        "理解用户消息里的“今天”“明天”等相对时间时以此为准。"
    )
