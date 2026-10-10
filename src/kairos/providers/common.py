"""供应商适配器共享的响应字段读取和工具调用归一化逻辑。"""

from collections.abc import Mapping
from typing import Any


def field(value: object, name: str, default: Any = None) -> Any:
    """同时读取 SDK 对象和测试中使用的字典。"""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_tool_call(tool_call: object) -> dict:
    function = field(tool_call, "function", {})
    name = field(function, "name")
    call_id = field(tool_call, "id")
    if not name or not call_id:
        raise ValueError("模型返回的工具调用缺少 id 或 name")

    arguments = field(function, "arguments", "{}")
    if not isinstance(arguments, str):
        raise TypeError("模型返回的工具参数必须是 JSON 字符串")

    return {
        "id": call_id,
        "type": field(tool_call, "type", "function"),
        "function": {"name": name, "arguments": arguments},
    }
