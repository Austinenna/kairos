"""加载可选的本地工具模块，组合模型定义、执行函数和当前使用规则。"""

import sqlite3
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module

from .agent_loop import ToolHandler

TOOL_CONTEXT_HEADER = "【Kairos 当前工具能力】"


@dataclass(frozen=True)
class ToolRegistry:
    """承载当前工具的模型定义、执行函数和请求期规则。"""

    handlers: dict[str, ToolHandler]
    schemas: list[dict]
    instructions: tuple[str, ...] = ()


def load_tool_registry(connection: sqlite3.Connection, factory_paths: Sequence[str]) -> ToolRegistry:
    """按显式配置加载工具工厂，合并其返回的普通数据。

    作用:
        在应用启动时连接可选业务模块与核心运行时，核心不导入具体业务包。

    参数:
        connection: 当前应用连接；工厂可初始化自己的表并绑定执行函数。
        factory_paths: module:callable 形式的本地工厂路径，空序列表示不加载模块。

    返回:
        合并后的 ToolRegistry；工厂返回的 schema 会复制，避免污染模块内常量。

    异常:
        ValueError: 工厂路径格式错误、工厂不是可调用对象、返回值不符合约定，
            或工具定义和执行函数不匹配、工具名称重复。

    行为说明:
        1. 只导入配置指定的已安装本地模块，不扫描目录或自动安装依赖。
        2. 工厂返回 handlers、schemas，以及可选的 instructions 字符串序列。
        3. 保留导入和工厂执行的原异常，不把缺失依赖或初始化错误当成空工具。
        4. 工厂只在启动时调用；正常回合直接使用注册好的执行函数。
    """
    if isinstance(factory_paths, (str, bytes)) or not isinstance(factory_paths, Sequence):
        raise ValueError("工具工厂必须是 module:callable 路径的序列")
    handlers: dict[str, ToolHandler] = {}
    schemas: list[dict] = []
    instructions: list[str] = []
    for path in factory_paths:
        if not isinstance(path, str):
            raise ValueError("工具工厂路径必须是字符串")
        module_name, separator, function_name = path.partition(":")
        if (
            not separator
            or not all(part.isidentifier() for part in module_name.split("."))
            or not function_name.isidentifier()
        ):
            raise ValueError(f"工具工厂路径必须采用 module:callable 格式：{path}")
        factory = getattr(import_module(module_name), function_name)
        if not callable(factory):
            raise ValueError(f"工具工厂不可调用：{path}")
        bundle = factory(connection)
        if not isinstance(bundle, dict):
            raise ValueError(f"工具工厂必须返回字典：{path}")
        module_handlers = bundle.get("handlers")
        module_schemas = bundle.get("schemas")
        module_instructions = bundle.get("instructions", ())
        if not isinstance(module_handlers, dict) or not all(
            isinstance(name, str) and name and callable(handler) for name, handler in module_handlers.items()
        ):
            raise ValueError(f"工具 handlers 必须是工具名称到函数的映射：{path}")
        if not isinstance(module_schemas, list):
            raise ValueError(f"工具 schemas 必须是列表：{path}")
        names: list[str] = []
        for schema in module_schemas:
            function = schema.get("function") if isinstance(schema, dict) else None
            if (
                not isinstance(schema, dict)
                or schema.get("type") != "function"
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
                or not isinstance(function.get("parameters"), dict)
            ):
                raise ValueError(f"工具 schema 必须包含 function 名称和参数定义：{path}")
            names.append(function["name"])
        if len(names) != len(set(names)) or set(names) != set(module_handlers):
            raise ValueError(f"工具定义与执行函数必须一一对应且名称不重复：{path}")
        duplicates = set(handlers).intersection(names)
        if duplicates:
            raise ValueError(f"工具名称重复：{', '.join(sorted(duplicates))}")
        if isinstance(module_instructions, (str, bytes)) or not isinstance(module_instructions, Sequence):
            raise ValueError(f"工具 instructions 必须是字符串序列：{path}")
        if not all(isinstance(item, str) and item.strip() for item in module_instructions):
            raise ValueError(f"工具 instructions 不能包含空白或非字符串内容：{path}")
        handlers.update(module_handlers)
        schemas.extend(deepcopy(module_schemas))
        instructions.extend(module_instructions)
    return ToolRegistry(handlers=handlers, schemas=schemas, instructions=tuple(instructions))


def prepare_tool_messages(messages: list[dict], registry: ToolRegistry) -> list[dict]:
    """在请求副本中加入当前工具能力和模块规则。

    作用:
        让新会话和恢复的旧会话都以当前注册的工具为准，保留原始会话历史。

    参数:
        messages: 原始消息或已加入记忆参考的请求视图。
        registry: 本次请求实际使用的工具注册表。

    返回:
        加入工具规则后的独立消息列表 (list[dict])。

    行为说明:
        1. 深拷贝输入，保留工具配对、图片及模型协议字段。
        2. 在首条 system 中追加当前工具名与规则，没有 system 时创建一条。
        3. 模块规则只随当前请求加载，不持久写入会话的 system 消息。
    """
    request = deepcopy(messages)
    names = [schema["function"]["name"] for schema in registry.schemas]
    rules = "\n".join(
        [
            TOOL_CONTEXT_HEADER,
            f"当前可用工具：{', '.join(names) if names else '无'}。",
            "本次能力以当前提供的工具定义为准；历史记录中的工具或承诺不代表当前可用。",
            "请求需要的能力未接入时，明确说明尚未执行，不能把聊天记录或记忆保存冒充业务操作完成。",
            "只有工具返回确认成功后，才能说明对应操作已完成；结果未知时如实说明。",
            *registry.instructions,
        ]
    )
    position = next((i for i, message in enumerate(request) if message.get("role") == "system"), None)
    if position is None:
        request.insert(0, {"role": "system", "content": rules})
    else:
        content = request[position].get("content")
        if isinstance(content, list):
            content.append({"type": "text", "text": rules})
        else:
            request[position]["content"] = f"{content}\n\n{rules}" if content else rules
    return request
