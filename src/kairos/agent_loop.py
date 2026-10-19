"""Kairos 的 Agent 循环。

这里先使用接近 OpenAI 的统一响应形状，方便用 Fake Model 测试消息顺序。
当前支持多工具、跨轮调用、工具错误回传和最大工具响应轮数。
"""

import json
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from .cancellation import current_cancellation_token, raise_if_cancelled
from .images import redact_images


class Model(Protocol):
    """模型需要提供的最小接口。"""

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict: ...


@runtime_checkable
class StreamingModel(Protocol):
    """同步消费一次模型流；正文即时回调，完整响应在流正常结束后返回。"""

    def stream(self, *, messages: list[dict], tools: list[dict], on_text: Callable[[str], None]) -> dict: ...


ToolHandler = Callable[..., object]
EventLogger = Callable[[dict[str, Any]], None]


class ExecutionRecorder(Protocol):
    """循环的持久化边界；内部身份不混入发给模型的消息。

    循环按执行顺序调用这些方法，由实现负责 ID 分配、消息配对和状态写入。
    ids 供观察事件关联当前会话、回合与步骤，消息写入方法返回内部消息 ID。
    """

    @property
    def ids(self) -> dict[str, str]: ...

    def start_step(self, index: int) -> None: ...

    def record_response(self, response: dict, duration_ms: float) -> None: ...

    def append_assistant(self, message: dict) -> str: ...

    def start_tool(self, tool_call_id: str) -> None: ...

    def append_tool_result(self, message: dict, *, status: str | None = None) -> str: ...

    def finish_step(self, status: str, *, error: str | None = None) -> None: ...


class AgentLoopLimitError(RuntimeError):
    """模型连续请求工具，超过了允许的工具响应轮数。"""


def sanitize_error(error: BaseException | str, messages: list[dict]) -> str:
    """SDK 可能把请求附在错误中；展示错误时移除已知图片字节。"""

    text = str(error)

    def remove_image_data(value: object) -> None:
        nonlocal text
        if isinstance(value, dict):
            if value.get("type") == "image" and isinstance(value.get("data"), str) and value["data"]:
                text = text.replace(value["data"].replace("/", "\\/"), "[图片数据已隐藏]")
                text = text.replace(value["data"], "[图片数据已隐藏]")
            for child in value.values():
                remove_image_data(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                remove_image_data(child)

    remove_image_data(messages)
    return redact_images(text)


def _assistant_message(response: dict, tool_calls: list[dict]) -> dict:
    message = {
        "role": "assistant",
        "content": response.get("content"),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    reasoning_content = response.get("reasoning_content")
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    # Anthropic 的内容块包含顺序和思考签名，必须原样用于后续请求。
    if "provider_content" in response:
        message["provider_content"] = response["provider_content"]
    return message


def _error_content(error: Exception) -> str:
    """把工具异常转换成模型可以继续处理的 JSON 字符串。"""
    return json.dumps(
        {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            }
        },
        ensure_ascii=False,
    )


def _execute_tool(tool_call: dict, tool_handlers: Mapping[str, ToolHandler]) -> str:
    """执行单个工具；参数或函数错误都作为工具结果返回。"""
    try:
        function = tool_call["function"]
        tool_name = function["name"]
        arguments = json.loads(function.get("arguments", "{}"))
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")

        result = tool_handlers[tool_name](**arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception as error:
        return _error_content(error)


def _display_event(event: dict[str, Any]) -> dict[str, Any]:
    """为界面投影不携带上下文、思考和工具载荷的事件。

    作用:
        将完整 observer 与入口展示分开，避免把模型请求和私人记忆交给各入口广播。
    参数:
        event: 已附带执行关联 ID、图片已脱敏的循环事件。
    返回:
        仅含事件类别、关联位置、统计和安全错误类型的独立字典。
    行为说明:
        1. model_request 只保留阶段与关联信息，model_response 不转发完整 response。
        2. 工具参数、结果及错误原文留给 logger；展示只保留工具名称与执行位置。
        3. 用量仅保留已知计数字段，不透传供应商追加字段。
    """
    allowed = (
        "event",
        "request_number",
        "tool_round",
        "session_id",
        "turn_id",
        "step_id",
        "message_id",
        "tool_call_id",
        "tool_name",
        "duration_ms",
    )
    displayed = {name: event[name] for name in allowed if name in event}
    usage = event.get("usage")
    if isinstance(usage, Mapping):
        displayed["usage"] = {
            name: value
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
            )
            if (value := usage.get(name)) is None or (type(value) is int and value >= 0)
        }
    error = event.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("type"), str):
        displayed["error"] = {"type": error["type"]}
    return displayed


def run_agent_turn(
    messages: list[dict],
    model: Model,
    tool_handlers: Mapping[str, ToolHandler],
    tool_schemas: list[dict],
    *,
    max_tool_rounds: int = 8,
    logger: EventLogger | None = None,
    on_event: EventLogger | None = None,
    recorder: ExecutionRecorder | None = None,
    prepare_messages: Callable[[list[dict]], list[dict]] | None = None,
) -> tuple[str, list[dict]]:
    """执行一次包含模型生成和工具调用的 Agent 回合。

    作用:
        接收 Runtime 准备的消息上下文，反复请求模型、执行工具并追加结果，
        直到模型给出没有工具调用的回答；通过可选记录器保存步骤和完整消息。

    参数:
        messages: 当前模型消息列表；函数会原地追加助手消息和工具结果。
        model: 提供 complete() 的模型；配置展示回调或取消作用域且支持 stream() 时使用流式接口。
        tool_handlers: 工具名称到执行函数的映射。
        tool_schemas: 发给模型的工具定义列表。
        max_tool_rounds: 大于等于 1；限制包含工具调用的模型响应轮数，一批多个工具只计一轮。
        logger: 可选日志回调，接收图片已脱敏的完整请求、响应、错误和工具结果事件。
        on_event: 可选展示回调，接收白名单阶段、关联及用量字段和实时 text_delta；
            不接收原始模型请求、完整响应、工具载荷或错误原文，正文片段不写入 logger。
        recorder: 可选执行记录器，负责 ID 关联和持久化；省略时仅维护内存历史。
        prepare_messages: 可选请求上下文构建函数，每次模型调用前接收原始历史并返回请求副本。
            构建函数不得修改原始历史；省略时沿用原始消息列表。

    返回:
        最终回答正文与更新后的消息列表（tuple[str, list[dict]]）。
        返回的列表就是传入的 messages，内部关联 ID 不混入其模型消息内容。

    异常:
        ValueError: 满足以下任一情况时显式抛出:
            - max_tool_rounds 小于 1;
            - 一批工具调用中有缺失、空字符串、非字符串或重复的调用 ID。
        AgentLoopLimitError: 已达到工具响应轮数上限，模型仍要求执行下一批工具。
        Exception / KeyboardInterrupt（显式重新抛出）: 对应本函数的三处裸 raise:
            - 模型生成或正文回调失败，发送 model_error 事件后重新抛出;
            - 工具批次处理失败，尝试补齐未配对结果，收尾成功后重新抛出;
            - 请求上下文构建或步骤执行失败，尝试记录步骤结束状态，收尾成功后重新抛出。

    行为说明:
        1. 每次迭代开始步骤，通过可选构建函数生成请求上下文；观察事件与模型使用同一副本。
           流式片段即时展示，完整模型响应随后处理；合成请求内容不追加到原始历史。
        2. 对工具响应先验证调用 ID，再保存完整助手消息和调用记录；保存成功后串行执行工具。
           每个结果通过原 tool_call_id 配对并追加到历史，再进入下一次模型生成。
        3. 无工具调用时保存最终回答并结束；工具批次中断时区分未执行和结果未知，
           尝试补齐结果并记录步骤状态，已经发生的工具副作用不会由此撤销。
           持久化补写失败时传播保存异常，遗留记录交由显式恢复处理。
        4. 使用当前取消作用域，在模型请求、正文与工具边界检查；同步工具执行期间
           不强制停止。工具正常返回时先持久保存结果，再观察取消，避免丢失已有副作用。
    """
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds 必须大于或等于 1")

    tool_rounds = 0
    request_number = 0

    def emit(event: dict[str, Any]) -> None:
        """为观察事件添加执行 ID 并清理图片载荷。

        作用:
            完整观察事件交给日志，展示回调仅收到白名单字段。

        参数:
            event: 待发送的事件字典，例如 {"event": "model_request", "request_number": 1}。

        返回:
            无返回值（None）；未配置回调时不发送事件。

        行为说明:
            1. 配置了 recorder 时加入当前会话、回合及步骤 ID，再生成图片脱敏副本。
            2. 先调用 logger，再向 on_event 发送展示投影；不改写日志和持久历史。
        """

        if recorder:
            event = {**event, **recorder.ids}
        event = redact_images(event)
        if logger:
            logger(event)
        if on_event:
            on_event(_display_event(event))

    def text_delta(text: str) -> None:
        """将正文片段及其执行位置交给展示回调。

        作用:
            在模型流读取期间显示增量正文；完整消息由外层循环在响应结束后保存。

        参数:
            text: 本次收到的正文片段；空字符串不产生事件。

        返回:
            无返回值（None）；没有 on_event 时直接结束。

        行为说明:
            1. 仅在正文非空且存在展示回调时构造 text_delta 事件。
            2. 加入请求序号和可用的执行 ID，脱敏后交给 on_event，不逐片写入持久历史。
        """

        raise_if_cancelled()
        if text and on_event:
            on_event(
                redact_images(
                    {
                        "event": "text_delta",
                        "request_number": request_number,
                        "text": text,
                        **(recorder.ids if recorder else {}),
                    }
                )
            )

    while True:
        raise_if_cancelled()
        request_number += 1
        if recorder:
            recorder.start_step(request_number)
        request_messages = messages
        try:
            if prepare_messages is not None:
                request_messages = prepare_messages(messages)
            emit(
                {
                    "event": "model_request",
                    "request_number": request_number,
                    "tool_round": tool_rounds,
                    "messages": list(request_messages),
                    "tools": list(tool_schemas),
                }
            )

            model_started = perf_counter()
            try:
                raise_if_cancelled()
                if (on_event is not None or current_cancellation_token() is not None) and isinstance(
                    model, StreamingModel
                ):
                    response = model.stream(messages=request_messages, tools=tool_schemas, on_text=text_delta)
                else:
                    response = model.complete(messages=request_messages, tools=tool_schemas)
                    text_delta(response.get("content") or "")
                raise_if_cancelled()
            except (Exception, KeyboardInterrupt) as error:
                emit(
                    {
                        "event": "model_error",
                        "request_number": request_number,
                        "tool_round": tool_rounds,
                        "error": {"type": type(error).__name__, "message": sanitize_error(error, request_messages)},
                        "duration_ms": round((perf_counter() - model_started) * 1000, 3),
                    }
                )
                raise
            duration_ms = round((perf_counter() - model_started) * 1000, 3)
            if recorder:
                recorder.record_response(response, duration_ms)
            emit(
                {
                    "event": "model_response",
                    "request_number": request_number,
                    "tool_round": tool_rounds,
                    "response": response,
                    "usage": response.get("usage"),
                    "duration_ms": duration_ms,
                }
            )
            raise_if_cancelled()
            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                final_content = response.get("content") or ""
                assistant = _assistant_message({**response, "content": final_content}, [])
                if recorder:
                    recorder.append_assistant(assistant)
                messages.append(assistant)
                if recorder:
                    recorder.finish_step("completed")
                return final_content, messages

            if tool_rounds >= max_tool_rounds:
                raise AgentLoopLimitError(f"工具调用超过最大轮数（{max_tool_rounds}）")
            tool_rounds += 1

            # 整批调用先验证、再保存；保存成功后才允许执行任何有副作用的工具。
            call_ids = [call.get("id") for call in tool_calls]
            if any(not isinstance(call_id, str) or not call_id for call_id in call_ids) or len(set(call_ids)) != len(
                call_ids
            ):
                raise ValueError("工具调用缺少或重复 id，无法回传结果")
            batch_start = len(messages)
            active_call: str | None = None
            try:
                assistant = _assistant_message(response, tool_calls)
                if recorder:
                    recorder.append_assistant(assistant)
                messages.append(assistant)
                for tool_call in tool_calls:
                    raise_if_cancelled()
                    function = tool_call.get("function", {})
                    result_started = perf_counter()
                    if recorder:
                        recorder.start_tool(tool_call["id"])
                    active_call = tool_call["id"]
                    result_content = _execute_tool(tool_call, tool_handlers)
                    result_message = {"role": "tool", "tool_call_id": tool_call["id"], "content": result_content}
                    messages.append(result_message)
                    message_id = recorder.append_tool_result(result_message) if recorder else None
                    active_call = None
                    emit(
                        {
                            "event": "tool_result",
                            "request_number": request_number,
                            "tool_round": tool_rounds,
                            "tool_call_id": tool_call["id"],
                            "tool_name": function.get("name"),
                            "arguments": function.get("arguments"),
                            "result": result_content,
                            "duration_ms": round((perf_counter() - result_started) * 1000, 3),
                            **({"message_id": message_id} if message_id else {}),
                        }
                    )
                    # 结果已经落盘再停止；不能把执行成功的同步工具误记为结果未知。
                    raise_if_cancelled()
            except (Exception, KeyboardInterrupt):
                # 执行、保存或展示中断时，补齐已保存的整批调用；恢复只能加载结果，不能重跑工具。
                completed = {m["tool_call_id"] for m in messages[batch_start + 1 :] if m["role"] == "tool"}
                for pending in tool_calls if len(messages) > batch_start else []:
                    if pending["id"] in completed:
                        continue
                    interrupted = pending["id"] == active_call
                    detail = (
                        "工具执行被中断，结果未确认；可能已产生副作用，请先查询状态，不要直接重试。"
                        if interrupted
                        else "本次工具调用因回合中断而未执行。"
                    )
                    result_message = {
                        "role": "tool",
                        "tool_call_id": pending["id"],
                        "content": _error_content(InterruptedError(detail)),
                    }
                    messages.append(result_message)
                    if recorder:
                        recorder.append_tool_result(
                            result_message, status="interrupted" if interrupted else "not_executed"
                        )
                raise
            if recorder:
                recorder.finish_step("completed")
        except (Exception, KeyboardInterrupt) as error:
            if recorder:
                recorder.finish_step(
                    "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=sanitize_error(error, request_messages),
                )
            raise
