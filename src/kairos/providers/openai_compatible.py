"""OpenAI Chat Completions 兼容模型适配器。"""

from collections.abc import Callable
from typing import Any

from openai import OpenAI

from ..cancellation import raise_if_cancelled
from .common import cancellable_stream, field, normalize_tool_calls


def openai_user_content(content: str | list[dict]) -> str | list[dict]:
    """只转换用户内容块，内部附件元数据不进入模型协议。"""

    if not isinstance(content, list):
        return content
    converted = []
    for block in content:
        if block.get("type") == "image":
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block['mime_type']};base64,{block['data']}"},
                }
            )
        elif block.get("type") == "text":
            converted.append({"type": "text", "text": block["text"]})
        else:
            raise ValueError("OpenAI 用户消息包含不支持的内容块")
    return converted


def openai_messages(messages: list[dict]) -> list[dict]:
    """保持 assistant 与 tool 消息字段，只投影用户的图文内容。"""

    return [
        {**message, "content": openai_user_content(message.get("content", ""))}
        if message.get("role") == "user"
        else dict(message)
        for message in messages
    ]


class OpenAICompatibleModel:
    """调用 OpenAI 兼容接口，并返回 Agent Loop 使用的统一字典。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.model = model
        self.extra_body = extra_body
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict:
        raise_if_cancelled()
        request = {"model": self.model, "messages": openai_messages(messages), "tools": tools}
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**request)
        raise_if_cancelled()
        choices = field(response, "choices", [])
        if not choices:
            raise ValueError("模型响应没有 choices")

        message = field(choices[0], "message")
        if message is None:
            raise ValueError("模型响应没有 message")

        if field(message, "function_call") is not None:
            raise ValueError("模型返回了不支持的旧式 function_call")
        raw_calls = field(message, "tool_calls", []) or []
        finish_reason = field(choices[0], "finish_reason")
        _validate_finish_reason(finish_reason, has_tools=bool(raw_calls))
        tool_calls = normalize_tool_calls(raw_calls)
        result = {
            "content": field(message, "content"),
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }
        usage = field(response, "usage")
        if usage is not None:
            result["usage"] = normalize_openai_usage(usage)
        reasoning_content = field(message, "reasoning_content")
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    def stream(self, *, messages: list[dict], tools: list[dict], on_text: Callable[[str], None]) -> dict:
        """逐片检查取消并交付正文，完整响应通过校验后才返回工具调用。

        作用:
            为 Agent 循环提供实时正文和最终完整响应，不执行未收齐的工具参数。
        参数:
            messages: 已组装的模型上下文。
            tools: 可用工具定义。
            on_text: 正文片段回调；thinking 与工具参数不会作为正文发送。
        返回:
            完整、已校验的统一响应字典。
        异常:
            ValueError: 收到非字符串正文或思考、旧式 function_call、结束后内容或冲突结束标记。
        行为说明:
            1. 每个 SDK 分片前后检查当前取消标记，包括没有正文的分片。
            2. 任何结束路径都关闭 SDK 流；阻塞中的网络读取仍需等待返回或超时。
        """

        raise_if_cancelled()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages(messages),
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body

        response_stream = self.client.chat.completions.create(**request)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: dict[int, dict[str, list[str]]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        try:
            for chunk in cancellable_stream(response_stream):
                chunk_usage = field(chunk, "usage")
                if chunk_usage is not None:
                    usage = normalize_openai_usage(chunk_usage)
                for choice in field(chunk, "choices", []) or []:
                    if field(choice, "index", 0) != 0:
                        continue
                    delta = field(choice, "delta")
                    text = field(delta, "content")
                    reasoning = field(delta, "reasoning_content")
                    calls = field(delta, "tool_calls", []) or []
                    if finish_reason is not None and (text or reasoning or calls):
                        raise ValueError("模型在 finish_reason 之后仍返回内容")
                    if text is not None:
                        if not isinstance(text, str):
                            raise ValueError("模型流式正文必须是字符串")
                        text_parts.append(text)
                        if text:
                            on_text(text)
                    if reasoning is not None:
                        if not isinstance(reasoning, str):
                            raise ValueError("模型流式思考内容必须是字符串")
                        reasoning_parts.append(reasoning)
                    if field(delta, "function_call") is not None:
                        raise ValueError("模型返回了不支持的旧式 function_call")
                    for call in calls:
                        _append_tool_delta(tool_parts, call)
                    reason = field(choice, "finish_reason")
                    if reason is not None:
                        if finish_reason is not None and finish_reason != reason:
                            raise ValueError("模型返回了相互冲突的 finish_reason")
                        finish_reason = reason

            # finish_reason 后仍可能有 choices=[] 的用量帧，必须读到流结束再校验和返回。
            _validate_finish_reason(finish_reason, has_tools=bool(tool_parts))

            result: dict[str, Any] = {
                "content": "".join(text_parts) if text_parts else None,
                "tool_calls": _finish_tool_calls(tool_parts),
                "finish_reason": finish_reason,
            }
            if reasoning_parts:
                result["reasoning_content"] = "".join(reasoning_parts)
            if usage is not None:
                result["usage"] = usage
            return result
        finally:
            # 也覆盖读取异常、调用方回调异常和 KeyboardInterrupt。
            response_stream.close()


def _append_tool_delta(parts: dict[int, dict[str, list[str]]], call: object) -> None:
    index = field(call, "index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("模型工具调用缺少有效的 index")
    if field(call, "type") not in {None, "function"}:
        raise ValueError("模型返回了不支持的工具调用类型")
    item = parts.setdefault(index, {"id": [], "name": [], "arguments": []})
    function = field(call, "function")
    for key, value in (
        ("id", field(call, "id")),
        ("name", field(function, "name")),
        ("arguments", field(function, "arguments")),
    ):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"模型工具调用的 {key} 片段必须是字符串")
            item[key].append(value)


def _validate_finish_reason(finish_reason: object, *, has_tools: bool) -> None:
    """统一校验 OpenAI 完整响应与流式响应的结束原因。

    作用:
        在协议适配器返回响应前阻止截断的工具调用进入执行流程，不执行工具或写入历史。

    参数:
        finish_reason: 模型返回的结束原因；必须是当前支持的字符串。
        has_tools: 响应是否包含工具调用，结构校验由工具归一化函数负责。

    返回:
        无返回值（None）；通过校验后调用方可以继续构造响应。

    异常:
        ValueError: 满足以下任一情况时抛出：
            - 缺少 finish_reason；
            - finish_reason 不是支持的字符串；
            - 有工具调用但因 length 或 content_filter 结束；
            - 以 tool_calls 结束却没有工具调用。

    行为说明:
        1. 检查结束标记，再校验它与工具调用是否一致。
        2. 保留 stop 携带完整工具调用的兼容行为；截断纯文本可返回供界面提示。
    """

    if finish_reason is None:
        raise ValueError("模型响应未完整结束：缺少 finish_reason")
    if not isinstance(finish_reason, str) or finish_reason not in {"stop", "tool_calls", "length", "content_filter"}:
        raise ValueError(f"模型返回了不支持的 finish_reason：{finish_reason}")
    if has_tools and finish_reason in {"length", "content_filter"}:
        raise ValueError(f"模型工具调用未完整结束：finish_reason={finish_reason}")
    if finish_reason == "tool_calls" and not has_tools:
        raise ValueError("模型以 tool_calls 结束，但没有工具调用")


def _finish_tool_calls(parts: dict[int, dict[str, list[str]]]) -> list[dict]:
    """合并工具调用片段并通过整批完整性校验。

    作用:
        将流式传输积累的片段转成与非流式相同的工具结构，供统一校验入口处理。

    参数:
        parts: 以工具索引为键、以 id、name、arguments 片段列表为值的字典。

    返回:
        按工具索引排序、经完整校验的调用列表（list[dict]）。

    行为说明:
        1. 按索引拼接片段，保持工具顺序及参数原文。
        2. 委托 normalize_tool_calls 校验整批结构与 JSON，全部通过后才返回。
    """

    calls: list[dict] = []
    for index in sorted(parts):
        item = {key: "".join(fragments) for key, fragments in parts[index].items()}
        calls.append(
            {
                "id": item["id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
        )
    return normalize_tool_calls(calls)


def normalize_openai_usage(usage: object) -> dict[str, int | None]:
    """将 OpenAI 兼容用量统一转换为可区分未知值与零值的字典。

    作用:
        为完整响应、流式响应及旧用量转换入口提供相同统计口径，不计算费用或修改用量。

    参数:
        usage: SDK 用量对象或字典；供应商可能省略缓存、思考等明细字段。

    返回:
        输入、输出、总量、缓存输入及思考 token 的字典（dict[str, int | None]）。
        缺失或不是非负整数的值返回 None，显式零值保留为 0。

    行为说明:
        1. 缓存量优先读取顶层 prompt_cache_hit_tokens，再依次读取
           prompt_tokens_details.prompt_cache_hit_tokens 和 cached_tokens。
        2. 仅在字段缺失或为 None 时尝试下一来源，不以较低优先级来源覆盖显式零值。
        3. 所有计数独立校验；无效计数保留未知，不猜测缓存量或思考量。
    """

    cached = field(usage, "prompt_cache_hit_tokens")
    if cached is None:
        details = field(usage, "prompt_tokens_details")
        cached = field(details, "prompt_cache_hit_tokens")
        if cached is None:
            cached = field(details, "cached_tokens")
    return {
        "prompt_tokens": _token_count(field(usage, "prompt_tokens")),
        "completion_tokens": _token_count(field(usage, "completion_tokens")),
        "total_tokens": _token_count(field(usage, "total_tokens")),
        "cached_input_tokens": _token_count(cached),
        "reasoning_tokens": _token_count(field(field(usage, "completion_tokens_details"), "reasoning_tokens")),
    }


def _token_count(value: object) -> int | None:
    """仅保留供应商返回的非负整数计数。

    作用:
        在协议用量归一化时隔离无效字段，使统计与展示无需转换不可信的计数。

    参数:
        value: 供应商的任意原始字段值。

    返回:
        非负整数（int）或表示未知的 None；布尔值不作为整数接受。

    行为说明:
        1. 检查整数类型和取值范围，原样返回有效计数，否则返回 None。
    """

    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
