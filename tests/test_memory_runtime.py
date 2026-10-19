"""验证记忆工具、持久会话和后台提取在统一运行时中的边界。"""

import copy
import json
from threading import Event
from types import SimpleNamespace

import pytest

from kairos import runtime as runtime_module
from kairos.config import Settings
from kairos.context import REFERENCE_HEADER
from kairos.memory import MemoryStore
from kairos.providers import OpenAICompatibleModel
from kairos.runtime import create_runtime


def tool(name, arguments, call_id="memory-call"):
    """生成可进入正常 Agent 循环的工具请求。

    作用:
        让集成用例通过真实工具分发和会话记录执行记忆操作。

    参数:
        name: 工具名。
        arguments: 工具参数。
        call_id: 本步骤的调用 ID。

    返回:
        统一模型响应（dict）。

    行为说明:
        1. 序列化参数并返回一次完整工具调用。
    """
    return {"tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class Model:
    model = "memory-offline"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, *, messages, tools):
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools}))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def references(request):
    """找出请求中的合成记忆区，避免依赖真实用户消息的位置。

    作用:
        检查每次模型请求的常驻内容数量及更新结果。

    参数:
        request: 测试模型捕获的完整请求。

    返回:
        记忆区正文列表（list[str]）。

    行为说明:
        1. 只选择带程序参考标记的文本消息。
    """
    return [
        m["content"]
        for m in request["messages"]
        if isinstance(m.get("content"), str) and m["content"].startswith(REFERENCE_HEADER)
    ]


def test_explicit_core_updates_next_request_without_polluting_transcript(tmp_path):
    """验证显式写入、最新常驻区与真实来源形成完整流程。

    作用:
        确认 remember 的副作用在工具续轮即生效，原始历史不保存合成参考消息。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 通过模型工具调用保存核心，核对两次请求及持久来源。
        2. 再次调用完全相同的内容不会重复追加核心。
    """
    fact = "学习时先给一个最小可运行示例。"
    model = Model(
        [
            tool("remember", {"content": fact}),
            {"content": "已记住"},
            tool("remember", {"content": fact}),
            {"content": "已经记录过"},
        ]
    )
    with create_runtime(model=model, db_path=tmp_path / "sessions.db", memory_dir=tmp_path / "memory") as runtime:
        result = runtime.respond("请记住我的学习偏好")
        assert result.tool_results[0].value["status"] == "saved"
        assert fact not in references(model.requests[0])[0]
        assert fact in references(model.requests[1])[0]
        assert all(len(references(r)) == 1 for r in model.requests)
        assert REFERENCE_HEADER not in json.dumps(runtime.history(), ensure_ascii=False)
        assert REFERENCE_HEADER not in json.dumps(runtime.session_store.inspect(runtime.session_id), ensure_ascii=False)
        source_message = runtime.session_details()["turns"][0]["user_message_id"]
        assert source_message in runtime.memory_store.core_path.read_text()
        result = runtime.respond("再次记住同一条偏好")
        assert result.tool_results[0].value["status"] == "duplicate"
        assert runtime.memory_store.snapshot()[0].count(fact) == 1


def test_topic_body_is_on_demand_and_new_sessions_only_receive_index(tmp_path):
    """验证主题读取、全局索引及新会话不自动继承正文。

    作用:
        保证索引全量常驻与详细正文按需加载同时成立。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 创建主题，通过真实 read_memory 读取并记录工具结果。
        2. 新会话仍有主题索引，未读取时没有主题正文；恢复旧会话仍保留结果。
    """
    memory_dir = tmp_path / "memory"
    store = MemoryStore(memory_dir)
    body = "仅正文存在的具体依据 ZEUS-481，索引不包含这个代号。"
    item = store.write_topic(
        None,
        "设计背景",
        "讨论设计理由时参考",
        body,
        source={"session_id": "seed", "turn_id": "seed", "message_id": "seed"},
    )
    model = Model(
        [tool("read_memory", {"topic_id": item["topic_id"]}), {"content": "根据读到的记录回答"}, {"content": "新话题"}]
    )
    with create_runtime(model=model, db_path=tmp_path / "sessions.db", memory_dir=memory_dir) as runtime:
        old_id = runtime.session_id
        runtime.respond("查阅之前的设计背景")
        assert item["topic_id"] in references(model.requests[0])[0]
        assert "ZEUS-481" not in json.dumps(model.requests[0], ensure_ascii=False)
        assert "ZEUS-481" in model.requests[1]["messages"][-1]["content"]
        assert any(m["role"] == "tool" and "ZEUS-481" in m["content"] for m in runtime.history())
        runtime.new_session()
        runtime.respond("开始新话题")
        assert item["topic_id"] in references(model.requests[2])[0]
        assert "ZEUS-481" not in json.dumps(model.requests[2], ensure_ascii=False)
        runtime.resume_session(old_id)
        assert any(m["role"] == "tool" and "ZEUS-481" in m["content"] for m in runtime.history())


def test_topic_existing_fact_can_still_be_explicitly_saved_in_core(tmp_path):
    """验证主题中已存在的信息仍可按用户要求进入核心。

    作用:
        防止跨层去重把显式常驻要求降级为按需读取。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 先保存主题，再显式保存相同正文，确认核心新增且原主题保留。
    """
    root = tmp_path / "memory"
    store = MemoryStore(root)
    fact = "长期偏好先解释原因再给结论。"
    topic = store.write_topic(
        None, "协作方式", "解释需求时参考", fact, source={"session_id": "s", "turn_id": "t", "message_id": "m"}
    )
    model = Model([tool("remember", {"content": fact}), {"content": "已记住"}])
    with create_runtime(model=model, memory_dir=root, db_path=tmp_path / "sessions.db") as runtime:
        assert runtime.respond("请把这个偏好记住").tool_results[0].value["status"] == "saved"
        assert fact in store.snapshot()[0]
        assert store.read_topic(topic["topic_id"])["content"] == fact


def test_background_does_not_block_foreground_and_keeps_original_session_source(tmp_path):
    """验证后台提取与前台会话切换之间的隔离。

    作用:
        固定前台回合边界，不让之后的新会话或用户输入污染排队任务。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 用事件阻塞第一个后台请求，前台仍能切换会话并完成另一回合。
        2. 释放后核对两次任务的真实来源、原始历史不含后台回复和审计不含正文。
    """
    entered, release = Event(), Event()
    extract_models = []

    class ExtractModel(Model):
        def complete(self, **request):
            if not extract_models[0].requests:
                entered.set()
                assert release.wait(5)
            return super().complete(**request)

    def factory():
        model = ExtractModel([{"content": '{"status":"no_change"}'}])
        extract_models.append(model)
        return model

    front = Model([{"content": "第一轮已回答"}, {"content": "第二轮已回答"}])
    runtime = create_runtime(
        model=front, memory_dir=tmp_path / "memory", db_path=tmp_path / "sessions.db", extraction_model_factory=factory
    )
    try:
        first = runtime.respond("第一会话的私人内容")
        assert first.content == "第一轮已回答" and entered.wait(2)
        assert runtime.memory_pending_count == 1
        runtime.new_session()
        second = runtime.respond("第二会话的新内容")
        assert second.content == "第二轮已回答"
        assert runtime.memory_pending_count == 2
        release.set()
        assert runtime.wait_for_memory(5)
        events = runtime.drain_memory_events()
        assert [e["source"]["session_id"] for e in events] == [first.session_id, second.session_id]
        assert all(e["status"] == "no_change" for e in events)
        assert len(runtime.session_details()["turns"]) == 1
        assert len(extract_models) == 2
        first_material = extract_models[0].requests[0]["messages"][-1]["content"]
        assert "第一会话的私人内容" in first_material and "第二会话的新内容" not in first_material
        assert "第一会话的私人内容" not in (tmp_path / "memory/extraction.jsonl").read_text()
        assert runtime.drain_memory_events() == []
    finally:
        release.set()
        runtime.close()


@pytest.mark.parametrize("failure", [RuntimeError("模拟模型失败"), KeyboardInterrupt()])
def test_failed_or_interrupted_turn_never_schedules_extraction(tmp_path, failure):
    """验证没有完整结果的用户回合不触发提取。

    作用:
        区分完整回合与失败、中断，避免把未完成材料当作已完成结果。

    参数:
        tmp_path: 隔离目录。
        failure: 模型模拟错误。

    返回:
        None。

    行为说明:
        1. 请求失败后队列为空，提取工厂没有被调用。
    """
    built = []

    def factory():
        built.append(True)
        return Model([{"content": '{"status":"no_change"}'}])

    with create_runtime(
        model=Model([failure]), db_path=tmp_path / "sessions.db", extraction_model_factory=factory
    ) as runtime:
        with pytest.raises(type(failure)):
            runtime.respond("没有完整回答")
        assert runtime.wait_for_memory(1)
        assert runtime.memory_pending_count == 0 and built == []
        assert runtime.drain_memory_events() == []


def test_primary_cannot_call_background_writer_or_forge_source(tmp_path):
    """验证前台工具权限与来源参数边界。

    作用:
        防止模型越过来源绑定或调用只属于后台的核心、主题写入工具。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 模拟越权调用，正常回传工具错误，记忆文件不产生正文。
    """
    model = Model(
        [
            tool("write_topic", {"topic_id": None, "title": "越权", "description": "不应写入", "addition": "内容"}),
            tool("write_core", {"content": "前台不能自动写核心"}, "background-only"),
            tool("remember", {"content": "伪造来源", "source": {"message_id": "made-up"}}, "forged"),
            {"content": "权限不允许"},
        ]
    )
    with create_runtime(model=model, db_path=tmp_path / "sessions.db") as runtime:
        result = runtime.respond("检查工具边界")
        assert all("error" in item.value for item in result.tool_results)
        names = {s["function"]["name"] for s in model.requests[0]["tools"]}
        assert {"remember", "read_memory"} <= names
        assert not {"write_topic", "write_core"} & names
        assert "伪造来源" not in runtime.memory_store.snapshot()[0]
        assert "前台不能自动写核心" not in runtime.memory_store.snapshot()[0]
        assert list(runtime.memory_store.topics_dir.glob("*.md")) == []


def test_managed_runtime_builds_independent_extraction_client_by_default(tmp_path, monkeypatch):
    """验证真实应用入口默认启用提取并独立管理模型客户端。

    作用:
        使用真实适配器配离线客户端，检查自动工厂、工具隔离和正常关闭。

    参数:
        tmp_path: 隔离目录。
        monkeypatch: 替换模型构建中的网络客户端。

    返回:
        None。

    行为说明:
        1. 第一个适配器服务前台，第二个由后台独立创建，二者分别关闭。
    """
    clients = []

    class Client:
        def __init__(self, answer):
            self.answer, self.closed, self.requests = answer, False, []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **request):
            self.requests.append(request)
            return {"choices": [{"message": {"content": self.answer}, "finish_reason": "stop"}]}

        def close(self):
            self.closed = True

    def build(settings, client=None, *, thinking_enabled=False):
        assert client is None
        fake = Client("前台回答" if not clients else '{"status":"no_change"}')
        clients.append(fake)
        return OpenAICompatibleModel(
            model=settings.kairos_model,
            client=fake,
            extra_body={"thinking": {"type": "enabled" if thinking_enabled else "disabled"}, "keep": True},
        )

    monkeypatch.setattr(runtime_module, "build_model", build)
    settings = Settings(_env_file=None, kairos_model="deepseek-offline", gateway_base_url="https://offline.invalid")
    with create_runtime(settings=settings, db_path=tmp_path / "sessions.db") as runtime:
        assert runtime.memory_extractor is not None
        assert runtime.respond("一次普通对话").content == "前台回答"
        assert runtime.wait_for_memory(3)
        assert runtime.drain_memory_events()[0]["status"] == "no_change"
        assert len(clients) == 2 and clients[1].closed and not clients[0].closed
        assert clients[0].requests[0]["extra_body"] == {"thinking": {"type": "disabled"}, "keep": True}
        assert clients[1].requests[0]["extra_body"] == {
            "thinking": {"type": "disabled"},
            "keep": True,
            "response_format": {"type": "json_object"},
        }
        assert {t["function"]["name"] for t in clients[1].requests[0]["tools"]} == {
            "read_memory",
            "write_topic",
            "write_core",
        }
    assert all(client.closed for client in clients)


def test_completed_turn_creates_global_topic_and_new_session_reads_it(tmp_path):
    """验证自动写入索引到另一会话按需读取的完整流程。

    作用:
        连接 Runtime 成功边界、后台工具、程序索引和前台 read_memory。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 第一个回合由后台创建主题，核对实际来源及 CORE 未被自动写入。
        2. 新会话先只看到索引，读取后才得到正文，后台结果不进入用户历史。
    """
    factories = []

    def factory():
        responses = (
            [
                tool(
                    "write_topic",
                    {
                        "topic_id": None,
                        "title": "学习背景",
                        "description": "安排学习路径时参考",
                        "addition": "偏好通过小型实验学习，验收标记 ORBIT-934。",
                    },
                ),
                {"content": '{"status":"completed"}'},
            ]
            if not factories
            else [{"content": '{"status":"no_change"}'}]
        )
        model = Model(responses)
        factories.append(model)
        return model

    front = Model([{"content": "了解你的学习背景"}])
    with create_runtime(
        model=front, db_path=tmp_path / "sessions.db", memory_dir=tmp_path / "memory", extraction_model_factory=factory
    ) as runtime:
        first = runtime.respond("我喜欢通过小型实验学习")
        assert runtime.wait_for_memory(3)
        event = runtime.drain_memory_events()[0]
        assert event["status"] == "completed" and len(event["topic_ids"]) == 1
        topic_id = event["topic_ids"][0]
        record = runtime.memory_store.read_topic(topic_id)
        assert record["sources"][0]["turn_id"] == first.turn_id
        assert record["sources"][0]["session_id"] == first.session_id
        core, index = runtime.memory_store.snapshot()
        assert "ORBIT-934" not in core and "ORBIT-934" not in index and topic_id in index
        assert len(runtime.history()) == 3
        runtime.new_session()
        front.responses = iter([tool("read_memory", {"topic_id": topic_id}), {"content": "已读到先前背景"}])
        runtime.respond("查阅我的学习背景")
        assert "ORBIT-934" not in json.dumps(front.requests[1], ensure_ascii=False)
        assert "ORBIT-934" in front.requests[2]["messages"][-1]["content"]
        assert runtime.wait_for_memory(3)
        assert runtime.drain_memory_events()[0]["status"] == "no_change"


def test_completed_turn_saves_core_for_first_request_in_new_session(tmp_path):
    """验证后台核心落盘后，新会话首请求直接携带常驻信息。

    作用:
        从真实 Runtime 成功回合走通后台写入及新会话上下文加载。
        脚本模型只验证执行链，不用于声称真实模型能够正确分类姓名。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 第一个回合结束后，通过后台 write_core 写入称呼并核对来源。
        2. 等待完成后新建会话，首请求直接包含核心且无需执行 read_memory。
        3. 原始前台会话不包含后台工具调用或合成记忆正文。
    """
    fact = "用户希望被称为测试称呼 LYRA-275。"
    extract_models = [
        Model([tool("write_core", {"content": fact}), {"content": '{"status":"completed"}'}]),
        Model([{"content": '{"status":"no_change"}'}]),
    ]
    factories = iter(extract_models)
    front = Model([{"content": "好的"}, {"content": "我知道你的称呼"}])
    with create_runtime(
        model=front,
        db_path=tmp_path / "sessions.db",
        memory_dir=tmp_path / "memory",
        extraction_model_factory=lambda: next(factories),
    ) as runtime:
        first = runtime.respond("以后叫我测试称呼")
        assert runtime.wait_for_memory(3)
        event = runtime.drain_memory_events()[0]
        assert event["status"] == "completed"
        assert event["core_additions"] == 1 and event["topic_ids"] == []
        assert event["source"]["turn_id"] == first.turn_id
        assert event["source"]["session_id"] == first.session_id
        message_id = runtime.session_details()["turns"][0]["user_message_id"]
        assert event["source"]["message_id"] == message_id
        assert message_id in runtime.memory_store.core_path.read_text()
        assert len(runtime.history()) == 3
        assert "write_core" not in json.dumps(runtime.history(), ensure_ascii=False)
        assert "LYRA-275" not in json.dumps(runtime.history(), ensure_ascii=False)
        assert fact not in references(front.requests[0])[0]
        runtime.new_session()
        second = runtime.respond("我叫什么？")
        assert second.tool_results == []
        assert len(front.requests) == 2
        assert fact in references(front.requests[1])[0]
        assert all(message["role"] != "tool" for message in front.requests[1]["messages"])
        assert runtime.wait_for_memory(3)
        assert runtime.drain_memory_events()[0]["status"] == "no_change"


def test_background_core_write_deduplicates_foreground_explicit_memory(tmp_path):
    """验证前台显式保存后，后台自动保存相同正文不会重复追加。

    作用:
        确认两个保存入口复用同一存储去重规则并报告真实变化。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 前台执行 remember，回合结束后后台执行相同正文的 write_core。
        2. 核对前台保存成功、后台无新增以及文件仅保留一份正文。
    """
    fact = "默认称呼用户为测试名字。"
    front = Model([tool("remember", {"content": fact}), {"content": "已保存"}])
    background = Model([tool("write_core", {"content": fact}), {"content": '{"status":"completed"}'}])
    with create_runtime(
        model=front,
        db_path=tmp_path / "sessions.db",
        memory_dir=tmp_path / "memory",
        extraction_model_factory=lambda: background,
    ) as runtime:
        result = runtime.respond("请记住我的称呼")
        assert result.tool_results[0].value["status"] == "saved"
        assert runtime.wait_for_memory(3)
        event = runtime.drain_memory_events()[0]
        assert event["status"] == "no_change"
        assert event["core_additions"] == 0 and event["topic_ids"] == []
        assert runtime.memory_store.snapshot()[0].count(fact) == 1
        assert json.loads(background.requests[1]["messages"][-1]["content"])["status"] == "duplicate"
