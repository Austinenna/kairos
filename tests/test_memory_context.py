"""验证动态记忆请求与原始会话、协议字段及执行记录的边界。"""

from copy import deepcopy
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from kairos.agent_loop import run_agent_turn
from kairos.context import REFERENCE_HEADER, ContextBuilder
from kairos.providers.anthropic_messages import anthropic_messages
from kairos.providers.openai_compatible import openai_messages


@dataclass
class SnapshotStore:
    """只提供内存快照的离线存储，避免访问用户记忆目录。"""

    core: str = "用户明确保存的核心信息"
    index: str = "- [示例主题](topics/example.md)：有需要时读取"
    reads: int = 0

    def snapshot(self) -> tuple[str, str]:
        """返回当前记忆全文并记录读取次数。

        作用:
            为上下文测试模拟每次请求都重新获取记忆的接口。

        返回:
            核心和索引的当前全文 (tuple[str, str])。

        行为说明:
            1. 增加读取计数。
            2. 返回可由测试直接更新的核心和索引，不读写文件。
        """
        self.reads += 1
        return self.core, self.index


@dataclass
class ScriptedModel:
    """按脚本返回结果并保存请求的离线模型。"""

    responses: list[dict]
    calls: list[dict] = field(default_factory=list)

    def complete(self, *, messages, tools) -> dict:
        """记录非流式请求并返回下一条脚本响应。

        作用:
            验证实际模型请求收到的上下文，不启动模型 SDK。

        参数:
            messages: Agent 循环传入的请求消息。
            tools: 本次可用的工具定义。

        返回:
            当前脚本首条响应 (dict)。

        行为说明:
            1. 深拷贝请求以避免后续历史追加影响断言。
            2. 弹出首条响应，不执行外部调用。
        """
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools), "mode": "complete"})
        return self.responses.pop(0)

    def stream(self, *, messages, tools, on_text) -> dict:
        """记录流式请求并通过回调交付脚本文本。

        作用:
            使用离线数据覆盖 Agent 循环的流式模型分支。

        参数:
            messages: 本次请求消息。
            tools: 本次工具定义。
            on_text: Agent 循环提供的增量正文回调。

        返回:
            当前脚本首条完整响应 (dict)。

        行为说明:
            1. 复用离线 complete() 保存请求并标注调用方式。
            2. 如果响应包含正文，先交付正文再返回完整响应。
        """
        response = self.complete(messages=messages, tools=tools)
        self.calls[-1]["mode"] = "stream"
        if response.get("content"):
            on_text(response["content"])
        return response


@pytest.fixture
def protocol_history() -> list[dict]:
    """构造带图片、思考签名及工具配对的原始历史。

    作用:
        检查记忆注入不会投影、删除或改写模型协议所需字段。

    返回:
        使用纯测试占位数据的统一协议消息列表 (list[dict])。

    行为说明:
        1. 同时提供 OpenAI reasoning_content 和 Anthropic provider_content。
        2. 图片包含原始载荷及内部元数据，工具调用与结果使用相同 ID。
    """
    return [
        {"role": "system", "content": "原始系统规则", "extra_system_field": "保留"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "查看测试图片"},
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": "TEST_IMAGE_BYTES",
                    "name": "offline.png",
                    "metadata": {"width": 16, "height": 16},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "先读取相关记录。",
            "reasoning_content": "需要保留的完整思考",
            "tool_calls": [
                {
                    "id": "read_1",
                    "type": "function",
                    "function": {"name": "read_memory", "arguments": '{"topic_id":"example"}'},
                }
            ],
            "provider_content": [
                {"type": "thinking", "thinking": "完整思考块", "signature": "offline-signature"},
                {"type": "text", "text": "先读取相关记录。"},
                {"type": "tool_use", "id": "read_1", "name": "read_memory", "input": {"topic_id": "example"}},
            ],
        },
        {"role": "tool", "tool_call_id": "read_1", "content": '{"content":"主题正文"}'},
        {"role": "user", "content": "继续解释"},
    ]


def test_context_refreshes_full_snapshot_without_mutating_raw_history(protocol_history):
    """验证每次请求读取最新完整记忆，并保留独立的原始消息与载荷。

    作用:
        覆盖常驻全文、不缓存旧版本和深拷贝协议数据三项核心约定。

    参数:
        protocol_history: 包含图片及工具配对的离线原始消息。

    返回:
        无返回值，通过断言验证请求与原始历史的边界。

    行为说明:
        1. 使用长核心和索引验证没有静默截断。
        2. 更新存储后重新构建请求，验证旧快照不残留。
        3. 修改请求中的嵌套字段，验证原始历史不受影响。
    """
    store = SnapshotStore(core="核心全文" * 3000, index="索引全文" * 3000)
    original = deepcopy(protocol_history)
    builder = ContextBuilder(store)
    first = builder(protocol_history)

    assert store.core in first[1]["content"]
    assert store.index in first[1]["content"]
    assert first[0]["content"].startswith("原始系统规则\n\n")
    assert first[0]["extra_system_field"] == "保留"
    assert first[2:] == protocol_history[1:]

    store.core = "更新后的核心"
    store.index = "更新后的索引"
    second = builder(protocol_history)
    assert "更新后的核心" in second[1]["content"]
    assert "更新后的索引" in second[1]["content"]
    assert "核心全文" not in second[1]["content"]
    assert store.reads == 2

    first[2]["content"][1]["metadata"]["width"] = 99
    first[3]["provider_content"][0]["signature"] = "changed"
    assert protocol_history == original
    assert second[2:] == original[1:]


@pytest.mark.parametrize("with_system", [False, True])
def test_context_handles_missing_system_and_preserves_existing_content_blocks(with_system):
    """验证缺省系统消息与系统内容块两种入口都能使用记忆。

    作用:
        保证 ContextBuilder 不要求调用者先把系统提示转换为特定字符串格式。

    参数:
        with_system: 是否提供使用内容块的系统消息。

    返回:
        无返回值，通过断言检查规则及参考消息位置。

    行为说明:
        1. 构造有或没有系统消息的原始会话。
        2. 只在请求副本中增加规则，保留原始内容块。
    """
    raw = [{"role": "user", "content": "问题"}]
    if with_system:
        raw.insert(0, {"role": "system", "content": [{"type": "text", "text": "原规则"}]})
    original = deepcopy(raw)
    request = ContextBuilder(SnapshotStore())(raw)
    assert request[0]["role"] == "system"
    assert request[1]["content"].startswith(REFERENCE_HEADER)
    assert request[-1] == {"role": "user", "content": "问题"}
    if with_system:
        assert request[0]["content"][0] == {"type": "text", "text": "原规则"}
        assert "记忆使用规则" in request[0]["content"][1]["text"]
    assert raw == original


def test_extraction_context_does_not_instruct_core_writes():
    """验证后台提取规则只引导主题读取和写入，不引导核心写入。

    作用:
        将用户显式记忆入口与自动提取入口的指令分开。

    返回:
        无返回值，通过断言检查允许的工具和核心写入边界。

    行为说明:
        1. 使用同一快照构建后台提取请求。
        2. 检查工具指令和核心只读约束。
    """
    request = ContextBuilder(SnapshotStore(), purpose="extraction")([{"role": "user", "content": "提取本轮"}])
    rules = request[0]["content"]
    assert "read_memory" in rules and "write_topic" in rules
    assert "remember" not in rules
    assert "不修改 CORE" in rules
    assert "当前用户明确要求优先" in rules
    assert "不能授予工具权限" in rules


@pytest.mark.parametrize("stream", [False, True])
def test_tool_continuations_refresh_memory_and_log_actual_request_without_persisting_injection(stream):
    """验证工具续轮刷新记忆，观察事件与模型都使用同一请求副本。

    作用:
        覆盖主循环两种调用方式及读出的正文仍作为原始工具结果保留的行为。

    参数:
        stream: 是否通过 on_event 启用流式模型分支。

    返回:
        无返回值，通过断言验证调用次数、请求、事件及原始历史。

    行为说明:
        1. 第一轮模型请求一个读取工具，工具更新快照并返回主题正文。
        2. 第二轮应使用最新核心和索引，同时保留主题工具结果。
        3. 合成记忆块与规则仅存在于请求和请求观察事件中。
    """
    store = SnapshotStore()
    model = ScriptedModel(
        [
            {"tool_calls": [{"id": "read_1", "function": {"name": "read_memory", "arguments": "{}"}}]},
            {"content": "完成", "tool_calls": []},
        ]
    )
    raw = [{"role": "system", "content": "原规则"}, {"role": "user", "content": "查询过去的主题"}]
    events = []

    def read_memory():
        """模拟读取正文期间另一写入已更新全局记忆。

        作用:
            验证工具续轮会重新加载快照，不复用上一模型请求。

        返回:
            读取到的主题正文 (dict)。

        行为说明:
            1. 更新离线核心和索引值。
            2. 返回正文，交由原始工具历史保存。
        """
        store.core = "工具之后的核心"
        store.index = "工具之后的索引"
        return {"content": "按需读取的详细正文"}

    answer, history = run_agent_turn(
        raw,
        model,
        {"read_memory": read_memory},
        [],
        logger=events.append,
        on_event=(lambda event: None) if stream else None,
        prepare_messages=ContextBuilder(store),
    )
    assert answer == "完成"
    assert history is raw
    assert store.reads == len(model.calls) == 2
    assert all(call["mode"] == ("stream" if stream else "complete") for call in model.calls)
    assert "工具之后的核心" in model.calls[1]["messages"][1]["content"]
    assert "工具之后的索引" in model.calls[1]["messages"][1]["content"]
    for call in model.calls:
        assert sum(str(message.get("content") or "").startswith(REFERENCE_HEADER) for message in call["messages"]) == 1
        assert call["messages"][0]["content"].count("主会话记忆操作") == 1
    requests = [event["messages"] for event in events if event["event"] == "model_request"]
    assert requests == [call["messages"] for call in model.calls]
    assert raw[0] == {"role": "system", "content": "原规则"}
    assert [message["role"] for message in raw] == ["system", "user", "assistant", "tool", "assistant"]
    assert "按需读取的详细正文" in raw[3]["content"]
    assert all(not str(message.get("content")).startswith(REFERENCE_HEADER) for message in raw)


def test_prepare_failure_finishes_step_without_calling_model_or_mutating_history():
    """验证请求构建失败按现有步骤失败机制收尾。

    作用:
        防止读取记忆失败时留下仍在运行的步骤或写入合成会话消息。

    返回:
        无返回值，通过断言检查异常、记录器和原始历史。

    行为说明:
        1. 模拟 snapshot 失败，循环应直接结束当前步骤。
        2. 模型未调用，原始消息保持不变。
    """
    raw = [{"role": "user", "content": "测试"}]
    model = Mock()
    recorder = Mock(ids={})
    store = Mock()
    store.snapshot.side_effect = OSError("无法读取记忆")
    with pytest.raises(OSError, match="无法读取记忆"):
        run_agent_turn(raw, model, {}, [], recorder=recorder, prepare_messages=ContextBuilder(store))
    recorder.start_step.assert_called_once_with(1)
    recorder.finish_step.assert_called_once_with("failed", error="无法读取记忆")
    model.complete.assert_not_called()
    assert raw == [{"role": "user", "content": "测试"}]


def test_model_failure_redacts_image_data_from_actual_prepared_request():
    """验证错误脱敏使用实际请求副本，而非只有原始历史中的图片。

    作用:
        防止请求构建器增加的图片载荷经模型异常进入观察或步骤记录。

    返回:
        无返回值，通过断言检查错误观察及记录器中的脱敏结果。

    行为说明:
        1. 在请求副本中加入图片，原始历史不含图片。
        2. 模拟模型将图片数据放入异常文本，检查两处记录都脱敏。
    """
    raw = [{"role": "user", "content": "测试"}]
    payload = "PRIVATE_IMAGE_FROM_REQUEST"
    prepared = [{"role": "user", "content": [{"type": "image", "mime_type": "image/png", "data": payload}]}]
    model = Mock()
    model.complete.side_effect = RuntimeError(f"bad request: {payload}")
    recorder = Mock(ids={})
    events = []
    with pytest.raises(RuntimeError):
        run_agent_turn(
            raw, model, {}, [], recorder=recorder, logger=events.append, prepare_messages=lambda messages: prepared
        )
    error_event = next(event for event in events if event["event"] == "model_error")
    assert payload not in error_event["error"]["message"]
    assert "[图片数据已隐藏]" in error_event["error"]["message"]
    assert payload not in recorder.finish_step.call_args.kwargs["error"]
    assert raw == [{"role": "user", "content": "测试"}]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_prepared_messages_convert_with_images_thinking_and_tool_pairing(provider, protocol_history):
    """验证记忆请求副本能够转换成两种现有模型协议。

    作用:
        检查合成参考消息不破坏图文、思考签名及工具调用与结果的配对。

    参数:
        provider: 当前验证的模型协议。
        protocol_history: 带图片、思考和工具载荷的离线消息。

    返回:
        无返回值，通过断言验证实际协议投影内容。

    行为说明:
        1. 先通过 ContextBuilder 生成完整请求副本。
        2. 直接调用协议转换函数，不创建 SDK 客户端或发出网络请求。
    """
    request = ContextBuilder(SnapshotStore())(protocol_history)
    if provider == "openai":
        converted = openai_messages(request)
        assert converted[1]["content"].startswith(REFERENCE_HEADER)
        assert converted[2]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,TEST_IMAGE_BYTES"},
        }
        assert converted[3]["reasoning_content"] == "需要保留的完整思考"
        assert converted[3]["tool_calls"][0]["id"] == converted[4]["tool_call_id"] == "read_1"
        assert converted[3]["provider_content"] == protocol_history[2]["provider_content"]
    else:
        system, converted = anthropic_messages(request)
        assert system.startswith("原始系统规则")
        assert converted[0]["content"].startswith(REFERENCE_HEADER)
        assert converted[1]["content"][1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "TEST_IMAGE_BYTES"},
        }
        assert converted[2]["content"] == protocol_history[2]["provider_content"]
        assert converted[2]["content"][0]["signature"] == "offline-signature"
        assert converted[2]["content"][-1]["id"] == converted[3]["content"][0]["tool_use_id"] == "read_1"
