"""供应商适配器共享的响应字段读取和工具调用归一化逻辑。"""

import json
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ..cancellation import raise_if_cancelled


def cancellable_stream[T](stream: Iterable[T]) -> Iterator[T]:
    """在每次读取 SDK 流之前和之后检查合作取消。

    作用:
        让正文、thinking、工具参数和空帧共享取消边界，不依赖正文回调触发。
    参数:
        stream: SDK 返回的同步流；其关闭仍由 provider 的 finally 负责。
    返回:
        原样产生每个 SDK 分片的迭代器。
    行为说明:
        1. 读取前检查，避免上一帧处理时已取消却再次进入阻塞读取。
        2. 读取后检查，保证无正文分片也能观察取消；流结束前再次确认。
        3. 正在阻塞的 next() 仍需等待数据或网络超时，不强制打断其他线程。
    """
    iterator = iter(stream)
    while True:
        raise_if_cancelled()
        try:
            item = next(iterator)
        except StopIteration:
            raise_if_cancelled()
            return
        raise_if_cancelled()
        yield item


def field(value: object, name: str, default: Any = None) -> Any:
    """同时读取 SDK 对象和测试中使用的字典。"""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_tool_call(tool_call: object) -> dict:
    """校验并统一一个 OpenAI 风格的完整工具调用。

    作用:
        在协议适配器返回工具前验证调用结构和参数完整性；不执行工具、不修改历史。

    参数:
        tool_call: SDK 对象或字典，包含 id、function.name 和 function.arguments。
            arguments 必须是完整 JSON 对象的字符串。

    返回:
        包含 id、type 和 function 的统一工具调用字典（dict），保留原始参数字符串。

    异常:
        ValueError: 满足以下任一情况时抛出：
            - id 或 name 不是非空字符串；
            - 调用类型不是 function；
            - 参数无法解析为严格 JSON；
            - 参数解析结果不是对象。
        TypeError: 参数不是 JSON 字符串。

    行为说明:
        1. 检查身份和类型，再解析完整参数，拒绝 NaN、Infinity 等非 JSON 常量。
        2. 验证成功后保留参数原文，后续工具执行入口仍负责业务字段校验。
    """

    function = field(tool_call, "function", {})
    name = field(function, "name")
    call_id = field(tool_call, "id")
    if not isinstance(name, str) or not name.strip() or not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("模型返回的工具调用缺少 id 或 name")
    if field(tool_call, "type", "function") != "function":
        raise ValueError("模型返回了不支持的工具调用类型")

    arguments = field(function, "arguments")
    if not isinstance(arguments, str):
        raise TypeError("模型返回的工具参数必须是 JSON 字符串")
    try:
        parsed = json.loads(arguments, parse_constant=_reject_json_constant)
    except ValueError as error:
        raise ValueError("模型工具调用参数不是完整的 JSON 对象") from error
    if not isinstance(parsed, dict):
        raise ValueError("模型工具调用参数必须是 JSON 对象")

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def normalize_tool_calls(tool_calls: list[object]) -> list[dict]:
    """校验整批工具调用并拒绝重复调用身份。

    作用:
        为完整响应与组装完成的流式响应提供同一返回边界；整批通过后才交给 Agent Loop。

    参数:
        tool_calls: 按模型返回顺序排列的工具调用 SDK 对象或字典列表。

    返回:
        已校验的工具调用列表（list[dict]），没有调用时返回空列表。

    异常:
        ValueError: 同一批工具调用包含重复 id。

    行为说明:
        1. 逐项委托 normalize_tool_call 验证结构和 JSON 参数，不执行任何调用。
        2. 检查整批调用身份唯一后返回；任一项失败时整个响应不返回可执行工具。
    """

    calls = [normalize_tool_call(call) for call in tool_calls]
    call_ids = [call["id"] for call in calls]
    if len(set(call_ids)) != len(call_ids):
        raise ValueError("模型返回了重复的工具调用 id")
    return calls


def _reject_json_constant(value: str) -> None:
    """拒绝 JSON 解析器默认容许的非标准数字常量。

    作用:
        作为 json.loads 的 parse_constant 回调，阻止不完整协议数据进入工具调用。

    参数:
        value: 解析器遇到的 NaN、Infinity 或 -Infinity 标记。

    返回:
        不正常返回；始终抛出异常。

    异常:
        ValueError: 解析器遇到非 JSON 常量并调用本函数。

    行为说明:
        1. 抛出 ValueError，由调用方统一报告工具参数无效。
    """

    raise ValueError(f"工具参数包含非 JSON 常量：{value}")
