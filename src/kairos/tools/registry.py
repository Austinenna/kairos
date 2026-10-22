"""工具集合：模型看到的工具定义、实际执行的函数，以及告诉模型怎么用的规则。"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module


@dataclass
class Toolset:
    schemas: list[dict] = field(default_factory=list)  # 发给模型的工具定义（OpenAI 格式）
    handlers: dict[str, Callable] = field(default_factory=dict)  # 工具名 → 执行函数
    instructions: list[str] = field(default_factory=list)  # 放进请求视图的使用规则

    def add(self, other: "Toolset") -> None:
        self.schemas += other.schemas
        self.handlers |= other.handlers
        self.instructions += other.instructions


def tool_schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    """生成一个工具定义。properties 写法：{"参数名": {"type": "string", "description": "..."}}。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties) if required is None else required,
            },
        },
    }


def load_factories(paths: list[str], db: sqlite3.Connection) -> Toolset:
    """加载配置里的工具模块，例如 "kairos_todo.tools:build_tools"。

    工厂函数接收数据库连接，返回 {"schemas": [...], "handlers": {...}, "instructions": [...]}。
    """
    toolset = Toolset()
    for path in paths:
        module_name, _, function_name = path.partition(":")
        bundle = getattr(import_module(module_name), function_name)(db)
        toolset.add(Toolset(bundle["schemas"], bundle["handlers"], list(bundle.get("instructions", []))))
    return toolset
