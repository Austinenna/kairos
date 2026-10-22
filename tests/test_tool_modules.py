"""验证工具模块可替换、可禁用，并且当前能力不会改写会话历史。"""

import copy
import json
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest

from kairos.config import Settings, ToolSettings
from kairos.legacy_tools import load_tool_registry
from kairos.runtime import create_runtime

TOOL_CONTEXT_HEADER = "【Kairos 当前工具能力】"
FACTORY_MODULE_NAME = "kairos_test_tool_module"


class FakeModel:
    """提供固定响应，并保存每次实际发给模型的请求副本。"""

    model = "tool-module-test-model"

    def __init__(self, responses):
        """保存离线模型的响应序列。

        作用:
            为工具注册和会话恢复测试提供不访问网络的模型。

        参数:
            responses: 按调用顺序返回的统一模型响应。

        返回:
            无返回值；初始化响应迭代器及请求记录。

        行为说明:
            1. 仅保存测试数据，不读取应用配置。
        """
        self.responses = iter(responses)
        self.requests = []

    def complete(self, *, messages, tools):
        """记录请求并返回下一份响应。

        作用:
            让测试检查运行时提供的当前工具和请求上下文。

        参数:
            messages: 运行时提供的请求消息。
            tools: 本次请求可调用的工具定义。

        返回:
            下一份响应的独立副本。

        行为说明:
            1. 深拷贝请求及响应，避免后续会话操作修改测试证据。
        """
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools}))
        return copy.deepcopy(next(self.responses))


def tool_schema(name):
    """创建一个仅接收文字参数的合成工具定义。

    作用:
        为加载器校验和完整工具调用提供相同的独立测试契约。

    参数:
        name: 工具名称。

    返回:
        OpenAI 风格的函数工具定义字典。

    行为说明:
        1. 返回新字典，避免参数化测试共享可变定义。
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "回显本次调用提供的文字。",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch):
    """清除本文件测试涉及的外部环境配置。

    作用:
        确保配置测试只使用临时文件及测试显式设置的环境变量。

    参数:
        monkeypatch: 临时修改环境的测试工具。

    返回:
        无返回值；测试结束后自动恢复环境。

    行为说明:
        1. 按配置字段清除大小写两种名称，不读取或输出原值。
    """
    for name in Settings.model_fields.keys() | ToolSettings.model_fields.keys():
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def factory_module(monkeypatch):
    """安装一个仅在当前测试期间可导入的工具模块。

    作用:
        在不写入真实包、不依赖待办模块的前提下验证动态工厂加载。

    参数:
        monkeypatch: 管理临时模块注册的测试工具。

    返回:
        提供 build_tools 工厂的临时模块对象。

    行为说明:
        1. 工厂使用传入连接读取合成文字，返回普通声明字典。
        2. 注册到 sys.modules，测试完成后自动移除。
    """
    module = ModuleType(FACTORY_MODULE_NAME)

    def build_tools(connection):
        """将测试工具绑定到调用方提供的连接。

        作用:
            模拟独立业务模块的工厂，不导入 Kairos 类型。

        参数:
            connection: 应用提供的 SQLite 连接。

        返回:
            包含 handlers、schemas 和 instructions 的普通字典。

        行为说明:
            1. 绑定连接但不创建或修改任何业务表。
        """

        def echo_marker(text):
            """通过连接回显测试文字。

            作用:
                验证工厂获得的连接确实可用于工具执行。

            参数:
                text: 模型提供的合成文字。

            返回:
                包含查询结果的字典。

            行为说明:
                1. 仅执行参数绑定的 SELECT，不写入数据库。
            """
            return {"text": connection.execute("SELECT ?", (text,)).fetchone()[0]}

        return {
            "handlers": {"echo_marker": echo_marker},
            "schemas": [tool_schema("echo_marker")],
            "instructions": ["回显工具只处理本次调用提供的文字。"],
        }

    module.build_tools = build_tools
    monkeypatch.setitem(sys.modules, FACTORY_MODULE_NAME, module)
    return module


def test_disabling_tool_modules_keeps_conversation_without_creating_todo_table(tmp_path):
    """验证明确关闭工具模块后仍可完成普通对话。

    作用:
        检查核心运行时不依赖待办注册或待办表初始化。

    参数:
        tmp_path: 隔离的会话数据库目录。

    返回:
        无返回值；通过断言验证能力及持久化边界。

    行为说明:
        1. 禁用模块和记忆，保留核心会话历史工具并完成一轮对话。
        2. 检查数据库没有待办表，动态能力说明没有写入历史。
    """
    model = FakeModel([{"content": "普通对话正常"}])
    with create_runtime(
        model=model, db_path=tmp_path / "core.sqlite3", tool_factories=(), memory_enabled=False
    ) as runtime:
        assert runtime.respond("你好").content == "普通对话正常"
        assert set(runtime.tool_registry.handlers) == {"search_history", "read_session"}
        assert runtime.connection.execute("SELECT name FROM sqlite_master WHERE name = 'todos'").fetchone() is None
        request_text = json.dumps(model.requests[0]["messages"], ensure_ascii=False)
        assert TOOL_CONTEXT_HEADER in request_text
        assert "search_history" in request_text
        assert TOOL_CONTEXT_HEADER not in json.dumps(runtime.history(), ensure_ascii=False)


def test_replacement_factory_executes_and_adds_instructions_to_each_request(tmp_path, factory_module):
    """验证显式工厂替换默认待办并经过完整工具回合。

    作用:
        证明当前调用机制只依赖工具契约，且动态说明适用于每次模型请求。

    参数:
        tmp_path: 隔离的数据库目录。
        factory_module: 合成工具模块。

    返回:
        无返回值；检查执行结果、可见工具及历史不变性。

    行为说明:
        1. 让模型调用合成工具，再读取其结果生成最终回答。
        2. 检查两次请求都有能力说明，而原始会话不保存该说明。
    """
    model = FakeModel(
        [
            {
                "tool_calls": [
                    {
                        "id": "echo_1",
                        "type": "function",
                        "function": {"name": "echo_marker", "arguments": '{"text":"合成标记"}'},
                    }
                ]
            },
            {"content": "收到了合成标记"},
        ]
    )
    with create_runtime(
        model=model,
        db_path=tmp_path / "replacement.sqlite3",
        tool_factories=(f"{factory_module.__name__}:build_tools",),
        memory_enabled=False,
    ) as runtime:
        result = runtime.respond("回显合成标记")
        assert result.tool_results[0].value == {"text": "合成标记"}
        assert "add_todo" not in runtime.tool_registry.handlers
        assert runtime.connection.execute("SELECT name FROM sqlite_master WHERE name = 'todos'").fetchone() is None
        assert len(model.requests) == 2
        for request in model.requests:
            text = json.dumps(request["messages"], ensure_ascii=False)
            assert text.count(TOOL_CONTEXT_HEADER) == 1
            assert "回显工具只处理本次调用提供的文字。" in text
            assert "echo_marker" in {tool["function"]["name"] for tool in request["tools"]}
        history = json.dumps(runtime.history(), ensure_ascii=False)
        assert TOOL_CONTEXT_HEADER not in history
        assert "回显工具只处理本次调用提供的文字。" not in history


def test_restoring_legacy_session_uses_current_tools_without_rewriting_system(tmp_path):
    """验证旧会话中的待办规则不会重新注册已禁用的工具。

    作用:
        检查恢复会话时保留原始历史，同时将当前能力提供给模型。

    参数:
        tmp_path: 隔离的会话与待办数据库目录。

    返回:
        无返回值；检查旧数据、原始 system 和当前模型请求。

    行为说明:
        1. 使用旧提示词创建含待办的会话，再关闭整个待办模块恢复它。
        2. 确认旧数据保留、工具不可用，当前能力说明仅存在于请求副本。
        3. 再次启用待办模块，确认原始待办能通过工具完整读取。
    """
    database = tmp_path / "legacy.sqlite3"
    legacy_system = "你是 Kairos。用户要求新增待办时，必须调用 add_todo，成功后说明结果。"
    original = FakeModel(
        [
            {
                "tool_calls": [
                    {
                        "id": "old_todo",
                        "type": "function",
                        "function": {"name": "add_todo", "arguments": '{"title":"原始待办"}'},
                    }
                ]
            },
            {"content": "原始待办已保存"},
        ]
    )
    with create_runtime(
        model=original,
        db_path=database,
        system_prompt=legacy_system,
        tool_factories=("kairos_todo.tools:build_tools",),
        memory_enabled=False,
    ) as runtime:
        created_todo = runtime.respond("记录原始待办").tool_results[0].value
        session_id = runtime.session_id
        assert runtime.history()[0]["content"] == legacy_system

    restored_model = FakeModel([{"content": "当前没有接入待办功能，尚未创建新待办"}])
    with create_runtime(
        model=restored_model,
        db_path=database,
        session_id=session_id,
        tool_factories=(),
        memory_enabled=False,
    ) as restored:
        assert restored.history()[0]["content"] == legacy_system
        restored.respond("再记录一条待办")
        assert "add_todo" not in restored.tool_registry.handlers
        assert restored.connection.execute("SELECT COUNT(*) FROM todos").fetchone()[0] == 1
        request = restored_model.requests[0]
        assert "add_todo" not in {tool["function"]["name"] for tool in request["tools"]}
        system_text = "\n".join(
            str(message["content"]) for message in request["messages"] if message["role"] == "system"
        )
        assert legacy_system in system_text
        assert TOOL_CONTEXT_HEADER in system_text
        assert "历史记录中的工具或承诺不代表当前可用" in system_text
        assert "add_todo" not in system_text.split(TOOL_CONTEXT_HEADER, 1)[1]
        assert restored.history()[0]["content"] == legacy_system
        assert TOOL_CONTEXT_HEADER not in json.dumps(restored.history(), ensure_ascii=False)

    enabled_model = FakeModel(
        [
            {
                "tool_calls": [
                    {
                        "id": "list_retained",
                        "type": "function",
                        "function": {"name": "list_todos", "arguments": "{}"},
                    }
                ]
            },
            {"content": "原始待办仍然存在"},
        ]
    )
    with create_runtime(
        model=enabled_model,
        db_path=database,
        session_id=session_id,
        tool_factories=("kairos_todo.tools:build_tools",),
        memory_enabled=False,
    ) as enabled:
        result = enabled.respond("查询原始待办")
        assert result.tool_results[0].value == [created_todo]
        assert "add_todo" in enabled.tool_registry.handlers
        assert enabled.history()[0]["content"] == legacy_system


@pytest.mark.parametrize("source", ["dotenv", "environment"])
def test_empty_factory_configuration_works_without_model_settings(tmp_path, isolated_app_dir, monkeypatch, source):
    """验证文件或环境中的空列表能禁用模块且无需模型认证配置。

    作用:
        保持注入 FakeModel 时不要求完整 Settings 的隔离调用方式。

    参数:
        tmp_path: 隔离的数据库目录。
        isolated_app_dir: 只包含测试配置的应用目录。
        monkeypatch: 临时环境变量工具。
        source: 从文件读取空列表，或由环境空列表覆盖文件。

    返回:
        无返回值；检查配置解析与实际注册结果。

    行为说明:
        1. 不提供模型名或网关地址，仅配置工具列表。
        2. 注入离线模型启动运行时，确认空列表生效。
    """
    value = "[]" if source == "dotenv" else '["missing_configured_module:build_tools"]'
    (isolated_app_dir / ".env").write_text(f"KAIROS_TOOL_FACTORIES={value}\n", encoding="utf-8")
    if source == "environment":
        monkeypatch.setenv("KAIROS_TOOL_FACTORIES", "[]")
    assert ToolSettings().kairos_tool_factories == []
    with create_runtime(
        model=FakeModel([{"content": "配置生效"}]), db_path=tmp_path / "configured.sqlite3", memory_enabled=False
    ) as runtime:
        assert runtime.respond("测试配置").content == "配置生效"
        assert "add_todo" not in runtime.tool_registry.handlers


def test_explicit_empty_factories_override_configured_missing_module(tmp_path, isolated_app_dir):
    """验证调用方可以覆盖配置中的工具依赖。

    作用:
        使无待办包的嵌入式调用无需先修改用户配置即可启动核心。

    参数:
        tmp_path: 隔离的数据库目录。
        isolated_app_dir: 包含故意缺失模块配置的临时应用目录。

    返回:
        无返回值；检查显式空工厂列表优先。

    行为说明:
        1. 配置不可导入的模块，再显式传入空序列并执行普通对话。
    """
    (isolated_app_dir / ".env").write_text(
        'KAIROS_TOOL_FACTORIES=["missing_configured_module:build_tools"]\n', encoding="utf-8"
    )
    with create_runtime(
        model=FakeModel([{"content": "显式配置生效"}]),
        db_path=tmp_path / "override.sqlite3",
        tool_factories=(),
        memory_enabled=False,
    ) as runtime:
        assert runtime.respond("测试覆盖").content == "显式配置生效"
        assert "add_todo" not in runtime.tool_registry.handlers


def test_supplied_settings_choose_factories_when_no_explicit_override(tmp_path, monkeypatch):
    """验证已传入的 Settings 是工具配置来源。

    作用:
        防止运行时忽略调用方配置、再次从环境读取不同的工具列表。

    参数:
        tmp_path: 隔离的数据库目录。
        monkeypatch: 临时配置冲突环境变量的工具。

    返回:
        无返回值；检查显式 Settings 中的空列表生效。

    行为说明:
        1. 设置缺失模块环境值，但传入已明确禁用模块的 Settings。
    """
    monkeypatch.setenv("KAIROS_TOOL_FACTORIES", '["missing_environment_module:build_tools"]')
    settings = Settings(
        _env_file=None,
        gateway_base_url="https://offline.invalid/v1",
        kairos_model=FakeModel.model,
        kairos_tool_factories=[],
    )
    with create_runtime(
        settings=settings,
        model=FakeModel([{"content": "传入配置生效"}]),
        db_path=tmp_path / "settings.sqlite3",
        memory_enabled=False,
    ) as runtime:
        assert runtime.respond("测试传入配置").content == "传入配置生效"
        assert "add_todo" not in runtime.tool_registry.handlers


@pytest.mark.parametrize(
    "invalid_case", ["name_mismatch", "duplicate_schema", "non_callable", "text_instructions", "bad_instruction"]
)
def test_loader_rejects_invalid_factory_contract(factory_module, monkeypatch, invalid_case):
    """验证加载器拒绝无法安全调度的工具声明。

    作用:
        避免模型可见定义与实际执行函数不一致，或把非法说明传入上下文。

    参数:
        factory_module: 可替换工厂的临时模块。
        monkeypatch: 临时替换工厂的工具。
        invalid_case: 要构造的声明错误。

    返回:
        无返回值；检查加载在调用模型前明确失败。

    行为说明:
        1. 从有效声明出发，只改变当前参数对应的一项契约。
        2. 调用真实加载器，接受契约校验使用 TypeError 或 ValueError。
    """
    declaration = {"handlers": {"echo_marker": lambda text: text}, "schemas": [tool_schema("echo_marker")]}
    if invalid_case == "name_mismatch":
        declaration["schemas"] = [tool_schema("different_name")]
    elif invalid_case == "duplicate_schema":
        declaration["schemas"].append(tool_schema("echo_marker"))
    elif invalid_case == "non_callable":
        declaration["handlers"]["echo_marker"] = "not callable"
    elif invalid_case == "text_instructions":
        declaration["instructions"] = "不应逐字注册的字符串"
    else:
        declaration["instructions"] = ["合法说明", 7]
    monkeypatch.setattr(factory_module, "build_tools", lambda connection: declaration)
    with closing(sqlite3.connect(":memory:")) as connection, pytest.raises((TypeError, ValueError)):
        load_tool_registry(connection, (f"{factory_module.__name__}:build_tools",))


def test_loader_rejects_names_repeated_across_factories(factory_module):
    """验证多个模块不能静默覆盖同名工具。

    作用:
        保证当前工具名称始终对应唯一执行函数。

    参数:
        factory_module: 每次返回同名工具的合成模块。

    返回:
        无返回值；检查重复注册被拒绝。

    行为说明:
        1. 加载两次相同工厂，验证合并阶段明确失败。
    """
    factory_path = f"{factory_module.__name__}:build_tools"
    with closing(sqlite3.connect(":memory:")) as connection, pytest.raises((TypeError, ValueError)):
        load_tool_registry(connection, (factory_path, factory_path))


def test_loader_propagates_factory_dependency_failure(factory_module, monkeypatch):
    """验证工厂内部缺失依赖不会被当成可忽略模块。

    作用:
        保持加载错误可诊断，不把损坏模块伪装成正常禁用。

    参数:
        factory_module: 临时模块。
        monkeypatch: 临时替换工厂的工具。

    返回:
        无返回值；检查依赖错误的身份仍然可见。

    行为说明:
        1. 使用会抛出依赖缺失错误的工厂，检查真实加载器传播该错误。
    """

    def broken_factory(connection):
        """模拟工厂内部的依赖加载失败。

        作用:
            提供与目标模块不存在不同的失败来源。

        参数:
            connection: 未使用的隔离测试连接。

        返回:
            不返回结果。

        异常:
            ModuleNotFoundError: 始终模拟缺少业务内部依赖。

        行为说明:
            1. 在任何数据库操作之前抛出指定异常。
        """
        raise ModuleNotFoundError("合成业务依赖缺失", name="synthetic_business_dependency")

    monkeypatch.setattr(factory_module, "build_tools", broken_factory)
    with closing(sqlite3.connect(":memory:")) as connection, pytest.raises(ModuleNotFoundError) as error:
        load_tool_registry(connection, (f"{factory_module.__name__}:build_tools",))
    assert error.value.name == "synthetic_business_dependency"


def test_core_copy_runs_in_subprocess_without_todo_package(tmp_path):
    """验证只分发 Kairos 核心包时仍能禁用待办并运行。

    作用:
        防止本机 editable 安装掩盖核心对独立待办包的隐式导入依赖。

    参数:
        tmp_path: 核心副本、子进程工作目录及隔离数据库所在目录。

    返回:
        无返回值；检查隔离子进程完成对话并退出成功。

    行为说明:
        1. 只复制 src/kairos，用隔离解释器优先导入该副本。
        2. 明确阻止所有 kairos_todo 导入，禁用模块及记忆后运行一轮对话。
    """
    copied_root = tmp_path / "core-only"
    source = Path(__file__).resolve().parents[1] / "src" / "kairos"
    shutil.copytree(source, copied_root / "kairos", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from pathlib import Path

        class BlockTodo(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "kairos_todo" or fullname.startswith("kairos_todo."):
                    raise ModuleNotFoundError("Todo package is absent in this deployment", name=fullname)
                return None

        sys.meta_path.insert(0, BlockTodo())
        sys.path.insert(0, sys.argv[1])
        import kairos
        from kairos import config
        from kairos.runtime import create_runtime

        assert Path(kairos.__file__).resolve().parent == Path(sys.argv[1]) / "kairos"
        config.get_app_dir = lambda: Path(sys.argv[2]).parent / "isolated-apphome"

        class Model:
            model = "core-only-test-model"

            def complete(self, *, messages, tools):
                assert all(tool["function"]["name"] != "add_todo" for tool in tools)
                return {"content": "core works"}

        with create_runtime(model=Model(), db_path=sys.argv[2], tool_factories=(), memory_enabled=False) as runtime:
            assert runtime.respond("hello").content == "core works"
            assert runtime.connection.execute("SELECT name FROM sqlite_master WHERE name = 'todos'").fetchone() is None
        assert not any(name == "kairos_todo" or name.startswith("kairos_todo.") for name in sys.modules)
        print("core-only-ok")
        """
    )
    process = subprocess.run(
        [sys.executable, "-I", "-c", script, str(copied_root), str(tmp_path / "subprocess.sqlite3")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "core-only-ok"
