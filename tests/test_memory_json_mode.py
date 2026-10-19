"""用真实 SDK 的隔离 HTTP 请求核对后台 JSON 参数与前台边界。"""

import json

import httpx2
import pytest
from openai import OpenAI

from kairos import runtime as runtime_module
from kairos.config import Settings
from kairos.runtime import create_runtime


@pytest.mark.parametrize("thinking_enabled", [False, True])
def test_default_extractor_sends_json_mode_through_every_tool_step(tmp_path, monkeypatch, thinking_enabled):
    """验证默认调度经真实 SDK 发出后台 JSON 参数并完成工具续轮。

    作用:
        在实际 HTTP 请求层确认 extra_body 被合并到顶层，且前台仍是普通文本请求。

    参数:
        tmp_path: 隔离数据库和记忆目录。
        monkeypatch: 将默认模型构建的客户端替换为离线 HTTP transport。
        thinking_enabled: 参数化原有思考模式开关。

    返回:
        None，由断言确认请求参数、实际临时记忆写入和正常结束状态。

    行为说明:
        1. 前台完成一次回答，默认后台工厂随后创建独立模型。
        2. 后台先调用 write_core，再返回最终 JSON；两次请求均保留 JSON 模式与思考设置。
        3. SDK 使用假地址和假 Key，所有文件写入临时目录，不访问真实模型或个人数据。
    """
    requests = []
    clients = []
    original_build = runtime_module.build_model

    def respond(request):
        """捕获实际 HTTP JSON，并返回当前步骤的隔离协议响应。

        参数:
            request: SDK 经过参数合并后的 httpx2 请求。

        返回:
            非流式 Chat Completions 响应（httpx2.Response）。

        行为说明:
            1. 第一步是前台回答，后两步分别为后台工具请求和最终结构化状态。
        """
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {"role": "assistant", "content": "已收到你的称呼。"}
            finish_reason = "stop"
        elif len(requests) == 2:
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "离线协议测试",
                "tool_calls": [
                    {
                        "id": "save-name",
                        "type": "function",
                        "function": {"name": "write_core", "arguments": '{"content":"用户希望称呼为小岚。"}'},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": '{"status":"completed"}'}
            finish_reason = "stop"
        return httpx2.Response(
            200,
            json={
                "id": f"offline-{len(requests)}",
                "object": "chat.completion",
                "model": "deepseek-flash",
                "created": 0,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            },
        )

    def build(settings, client=None, *, thinking_enabled=False):
        """在原模型工厂前注入独立的离线 SDK 客户端。

        参数:
            settings: 测试传入的配置。
            client: 默认工厂不传外部客户端。
            thinking_enabled: 当前前台或后台的思考开关。

        返回:
            原工厂构造的模型适配器。

        行为说明:
            1. 只替换网络边界，保留默认后台工厂与协议参数生成过程。
        """
        assert client is None
        sdk = OpenAI(
            api_key="offline-key",
            base_url="https://offline.invalid/v1",
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
        )
        clients.append(sdk)
        return original_build(settings, client=sdk, thinking_enabled=thinking_enabled)

    monkeypatch.setattr(runtime_module, "build_model", build)
    settings = Settings(
        _env_file=None,
        kairos_model="deepseek-flash",
        kairos_api_key="offline-key",
        gateway_base_url="https://offline.invalid/v1",
    )
    with create_runtime(
        settings=settings,
        db_path=tmp_path / "sessions.sqlite3",
        memory_dir=tmp_path / "memory",
        thinking_enabled=thinking_enabled,
    ) as runtime:
        runtime.respond("以后叫我小岚。")
        assert runtime.wait_for_memory(3)
        event = runtime.drain_memory_events()[0]
        assert event["status"] == "completed"
        assert event["core_additions"] == 1
        assert "小岚" in runtime.memory_store.snapshot()[0]

    assert len(requests) == 3
    assert "response_format" not in requests[0]
    for request in requests[1:]:
        assert request["response_format"] == {"type": "json_object"}
        assert {tool["function"]["name"] for tool in request["tools"]} == {"read_memory", "write_core", "write_topic"}
    assert all(request["thinking"] == {"type": "enabled" if thinking_enabled else "disabled"} for request in requests)
    assert requests[-1]["messages"][-2]["reasoning_content"] == "离线协议测试"
    assert requests[-1]["messages"][-1]["role"] == "tool"
    assert all(client.is_closed() for client in clients)
