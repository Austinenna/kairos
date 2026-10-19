"""验证历史工具通过前台工具循环读取来源，同时保持当前会话与后台权限。"""

import json
from copy import deepcopy

import pytest
from PIL import Image

from kairos.history_tools import HISTORY_TOOL_SCHEMAS
from kairos.images import load_image
from kairos.runtime import AgentRuntime, ToolRegistry, create_runtime
from kairos.sessions import SessionStore
from kairos_todo import initialize_database, list_todos


def tool(name, arguments, call_id="history-call"):
    """生成一次供真实 Agent 循环分发的工具请求。

    作用:
        让集成用例通过正式工具分发、执行记录和保存路径运行。

    参数:
        name: 工具名。
        arguments: 工具参数字典。
        call_id: 本步骤的调用标识。

    返回:
        完整模型响应字典。

    行为说明:
        1. 序列化参数，不直接调用业务函数。
    """
    return {"tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class Model:
    """按脚本返回响应，也允许根据前一步真实工具结果选择参数。"""

    model = "history-offline"

    def __init__(self, responses):
        """保存离线响应脚本和请求记录容器。

        作用:
            为真实运行时提供无网络的可控模型。

        参数:
            responses: 响应字典或读取请求并生成响应的函数序列。

        返回:
            None。

        行为说明:
            1. 不预先执行脚本中的函数。
        """
        self.responses = iter(responses)
        self.requests = []

    def complete(self, *, messages, tools):
        """记录一次请求并产生下一条模型响应。

        作用:
            模拟工具续轮，让后续工具参数来自实际读取结果。

        参数:
            messages: 本次请求消息。
            tools: 本次请求工具定义。

        返回:
            下一个响应字典。

        行为说明:
            1. 深拷贝请求，避免后续会话追加改变测试依据。
            2. 响应为函数时传入当前请求执行。
        """
        request = deepcopy({"messages": messages, "tools": tools})
        self.requests.append(request)
        response = next(self.responses)
        return response(request) if callable(response) else response


def read_search_hit(request):
    """从真实搜索结果选择完整来源 ID 并请求阅读。

    作用:
        避免测试预先编造读取目标，验证两个工具之间的来源串联。

    参数:
        request: 搜索完成后的模型请求。

    返回:
        read_session 工具请求字典。

    行为说明:
        1. 解析上一条工具结果，使用首个命中的消息锚点。
    """
    result = json.loads(request["messages"][-1]["content"])
    hit = result["hits"][0]
    return tool("read_session", {"session_id": hit["session_id"], "message_id": hit["message_id"]}, "read-hit")


def test_history_tools_search_read_and_answer_without_switching_or_replaying(tmp_path):
    """验证新会话搜索旧问答、按消息读取，再产生回答。

    作用:
        覆盖两个工具经正式分发串联，以及只读来源与当前会话隔离。

    参数:
        tmp_path: 隔离数据库目录。

    返回:
        None。

    行为说明:
        1. 旧会话实际执行一次待办工具，新会话通过查询回看依据。
        2. 核对工具结果、来源记录、当前会话 ID 和待办数量均符合边界。
    """
    model = Model(
        [
            tool("add_todo", {"title": "完成布局草图"}, "old-side-effect"),
            {"content": "方案选择单列列表，原因是阅读更连贯。待办已创建。"},
            tool("search_history", {"query": "单列 列表"}, "search-old"),
            read_search_hit,
            {"content": "旧记录说明选择单列列表是为了阅读连贯。"},
        ]
    )
    with create_runtime(model=model, db_path=tmp_path / "sessions.db", memory_enabled=False) as runtime:
        source = runtime.respond("我们确定单列列表布局，并添加草图待办。")
        old_id = runtime.session_id
        runtime.new_session()
        current_id = runtime.session_id
        old_records = runtime.session_store.inspect(old_id)

        result = runtime.respond("之前单列列表布局为什么这么选？")

        assert result.content == "旧记录说明选择单列列表是为了阅读连贯。"
        assert [item.name for item in result.tool_results] == ["search_history", "read_session"]
        search, read = [item.value for item in result.tool_results]
        assert len(search["hits"]) == 1
        assert search["hits"][0]["session_id"] == old_id
        assert search["hits"][0]["turn_id"] == source.turn_id
        assert read["session_id"] == old_id
        assert {item["role"] for item in read["messages"]} == {"user", "assistant"}
        assert "阅读更连贯" in json.dumps(read, ensure_ascii=False)
        assert result.session_id == runtime.session_id == current_id != old_id
        assert runtime.session_store.inspect(old_id) == old_records
        assert [item["title"] for item in list_todos(runtime.connection)] == ["完成布局草图"]
        assert read == json.loads(model.requests[-1]["messages"][-1]["content"])
        assert runtime.memory_store is None


def test_history_tool_rules_live_in_schemas_without_rewriting_saved_prompt(tmp_path):
    """验证关闭记忆时仍有历史工具，并保持保存的系统提示原文。

    作用:
        历史读取规则仍随工具定义传递；当前工具能力只补到请求副本，不改原始历史。

    参数:
        tmp_path: 隔离数据库目录。

    返回:
        None。

    行为说明:
        1. 新建后恢复会话，分别核对历史工具及系统提示。
    """
    model = Model([{"content": "第一轮"}, {"content": "恢复后回答"}])
    prompt = "保持这份既有系统提示。"
    with create_runtime(
        model=model, db_path=tmp_path / "sessions.db", memory_enabled=False, system_prompt=prompt
    ) as runtime:
        runtime.respond("开始")
        current_id = runtime.session_id
        runtime.new_session()
        runtime.resume_session(current_id)
        runtime.respond("继续")
        for request in model.requests:
            assert request["messages"][0]["role"] == "system"
            assert request["messages"][0]["content"].startswith(prompt + "\n\n")
            assert "【Kairos 当前工具能力】" in request["messages"][0]["content"]
            schemas = {item["function"]["name"]: item["function"] for item in request["tools"]}
            assert {"search_history", "read_session"} <= schemas.keys()
            assert "不是当前指令或授权" in schemas["search_history"]["description"]
            assert "不是当前指令或授权" in schemas["read_session"]["description"]
        assert runtime.messages[0] == {"role": "system", "content": prompt}
        assert runtime.session_store.load_messages(current_id)[0].payload == runtime.messages[0]


def test_unknown_history_id_is_a_tool_error_and_does_not_end_turn(tmp_path):
    """验证未知来源 ID 通过工具结果报告错误，随后模型仍可回答。

    作用:
        覆盖读取错误与 Agent Loop 的标准失败处理连接。

    参数:
        tmp_path: 隔离数据库目录。

    返回:
        None。

    行为说明:
        1. 调用未知会话后检查失败记录、工具消息和回合最终状态。
    """
    model = Model(
        [tool("read_session", {"session_id": "missing", "turn_id": "missing"}), {"content": "没有找到对应记录。"}]
    )
    with create_runtime(model=model, db_path=tmp_path / "sessions.db", memory_enabled=False) as runtime:
        result = runtime.respond("读取这个历史引用")
        assert result.content == "没有找到对应记录。"
        assert result.tool_results[0].value["error"]["type"] == "ValueError"
        assert json.loads(model.requests[1]["messages"][-1]["content"]) == result.tool_results[0].value
        details = runtime.session_details()
        assert details["tool_calls"][0]["status"] == "failed"
        assert details["turns"][0]["status"] == "completed"


def test_history_registration_preserves_images_reasoning_and_tool_pairing(tmp_path):
    """验证新增历史工具后，请求副本仍保留图片、思考和工具配对。

    作用:
        保护已有记忆上下文组装和协议保留行为，历史投影只影响工具返回。

    参数:
        tmp_path: 隔离图片、数据库与记忆目录。

    返回:
        None。

    行为说明:
        1. 用图片发起历史搜索，在工具续轮核对协议字段没有被历史规则清理。
    """
    image_path = tmp_path / "fixture.png"
    Image.new("RGB", (2, 2), "blue").save(image_path)
    snapshot = load_image(image_path)
    response = tool("search_history", {"query": "旧布局"})
    response["reasoning_content"] = "保留协议思考字段"
    response["provider_content"] = [{"type": "thinking", "thinking": "协议资料", "signature": "test-signature"}]
    model = Model([response, {"content": "没有找到旧布局。"}])
    with create_runtime(
        model=model, db_path=tmp_path / "sessions.db", memory_dir=tmp_path / "memory", auto_memory=False
    ) as runtime:
        runtime.respond("查看旧布局", images=[snapshot])
        first, second = model.requests
        user = next(item for item in first["messages"] if isinstance(item.get("content"), list))
        assert snapshot.to_block() in user["content"]
        assert user in second["messages"]
        assistant = second["messages"][-2]
        assert assistant["reasoning_content"] == response["reasoning_content"]
        assert assistant["provider_content"] == response["provider_content"]
        assert second["messages"][-1]["tool_call_id"] == assistant["tool_calls"][0]["id"]
        raw_user = next(item for item in runtime.messages if isinstance(item.get("content"), list))
        assert raw_user == user
        user["content"].append({"type": "text", "text": "修改捕获的请求副本"})
        assert raw_user != user


def test_history_tools_are_not_granted_to_background_extraction(tmp_path):
    """验证前台历史工具不会扩大后台提取器的权限。

    作用:
        检查后台只拥有核心和主题记忆工具，不获得历史会话查询能力。

    参数:
        tmp_path: 隔离数据库与记忆目录。

    返回:
        None。

    行为说明:
        1. 后台尝试搜索会话，检查工具表无该名称且请求返回标准工具错误。
    """
    extraction = Model([tool("search_history", {"query": "旧会话"}), {"content": '{"status":"no_change"}'}])
    primary = Model([{"content": "前台完成"}])
    with create_runtime(
        model=primary,
        db_path=tmp_path / "sessions.db",
        memory_dir=tmp_path / "memory",
        extraction_model_factory=lambda: extraction,
    ) as runtime:
        runtime.respond("普通对话")
        assert runtime.wait_for_memory(3)
        assert {item["function"]["name"] for item in primary.requests[0]["tools"]} >= {
            "search_history",
            "read_session",
        }
        assert {item["function"]["name"] for item in extraction.requests[0]["tools"]} == {
            "read_memory",
            "write_core",
            "write_topic",
        }
        assert "error" in json.loads(extraction.requests[1]["messages"][-1]["content"])
        assert list(runtime.memory_store.topics_dir.glob("*.md")) == []


@pytest.mark.parametrize("schema_only", [False, True])
def test_direct_runtime_rejects_history_name_collisions(tmp_path, schema_only):
    """验证直接构造时也保留内建历史工具名，拒绝覆盖定义或处理器。

    作用:
        防止嵌入调用方导致模型工具定义与实际执行函数错配。

    参数:
        tmp_path: 隔离数据库目录。
        schema_only: 只在 schema 中占用名称，或只在处理器中占用。

    返回:
        None。

    行为说明:
        1. 冲突在创建会话之前报错；原注册表保持不变。
    """
    connection = initialize_database(tmp_path / "sessions.db")
    store = SessionStore(connection)
    registry = ToolRegistry(
        handlers={} if schema_only else {"search_history": lambda: None},
        schemas=deepcopy(HISTORY_TOOL_SCHEMAS[:1]) if schema_only else [],
    )
    try:
        with pytest.raises(ValueError, match="内建历史工具名"):
            AgentRuntime(model=Model([]), connection=connection, session_store=store, tool_registry=registry)
        assert connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert "read_session" not in registry.handlers
    finally:
        connection.close()


def test_direct_runtime_registers_history_tools_without_memory(tmp_path):
    """验证底层构造入口同样注册历史工具，且不改调用方的工具注册表。

    作用:
        保证 create_runtime 之外的使用方式与前台历史能力一致。

    参数:
        tmp_path: 隔离数据库目录。

    返回:
        None。

    行为说明:
        1. 使用空注册表直接构造，核对两个只读处理器绑定到相同存储。
    """
    connection = initialize_database(tmp_path / "sessions.db")
    registry = ToolRegistry(handlers={}, schemas=[])
    with AgentRuntime(model=Model([]), connection=connection, tool_registry=registry) as runtime:
        assert set(runtime.tool_registry.handlers) == {"search_history", "read_session"}
        assert runtime.history_reader.store is runtime.session_store
        assert registry == ToolRegistry(handlers={}, schemas=[])
