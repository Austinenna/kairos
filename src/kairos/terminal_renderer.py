"""Kairos 终端展示层。

本模块只负责把 Runtime 返回的结构化结果渲染到终端，不参与模型请求、工具执行
或数据库操作。Rich 是终端输出实现细节，Runtime 不依赖它。
"""

from __future__ import annotations

import re
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .pricing import CONTEXT_WINDOW, Usage, estimate_cost, is_peak
from .runtime import AgentTurnResult, ToolResult

_MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_PRIORITY_LABELS = {1: "低", 2: "普通", 3: "高"}
_STATUS_LABELS = {"pending": "未完成", "completed": "已完成"}


class TerminalRenderer:
    """使用 Rich 将 Agent 结果渲染成终端内容。"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def status(self, message: str) -> None:
        self.console.print(message)

    def error(self, message: str) -> None:
        self.console.print(message)

    def history(self, messages: list[dict]) -> None:
        self.console.print_json(data=messages)

    def render_turn(self, result: AgentTurnResult, *, model: str, total_cost: float) -> float:
        """渲染一次 Agent 回合，返回本轮费用供 CLI 累加。"""

        self.render_tool_results(result.tool_results)
        self.render_content(result.content)
        if result.finish_reason == "length":
            self.status("[yellow]回复被截断了：输出达到了长度上限。[/]")

        cost = estimate_cost(model, result.usage) if result.usage else None
        self.render_usage(model, result.usage, cost, total_cost + (cost or 0.0))
        return cost or 0.0

    def render_content(self, content: str) -> None:
        """渲染普通文本、Markdown 和 Mermaid 代码块。"""

        if not content:
            return

        cursor = 0
        for match in _MERMAID_BLOCK.finditer(content):
            self._render_markdown(content[cursor : match.start()])
            self.console.print(
                Panel(
                    Syntax(match.group(1).strip(), "text", line_numbers=True),
                    title="Mermaid",
                    border_style="cyan",
                )
            )
            cursor = match.end()
        self._render_markdown(content[cursor:])

    def render_tool_results(self, tool_results: list[ToolResult]) -> None:
        """把待办查询结果渲染为表格；其他工具结果交给模型文本展示。"""

        for result in tool_results:
            if result.name == "list_todos" and isinstance(result.value, list):
                self.render_todo_table(result.value)

    def render_todo_table(self, todos: list[dict[str, Any]]) -> None:
        table = Table(title="待办列表", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("待办", min_width=20)
        table.add_column("优先级", justify="center")
        table.add_column("状态", justify="center")

        for todo in todos:
            status = str(todo.get("status", ""))
            status_style = "green" if status == "completed" else "white"
            table.add_row(
                str(todo.get("id", "")),
                str(todo.get("title", "")),
                _PRIORITY_LABELS.get(todo.get("priority"), str(todo.get("priority", ""))),
                f"[{status_style}]{_STATUS_LABELS.get(status, status)}[/{status_style}]",
            )

        self.console.print(table)

    def render_usage(self, model: str, usage: Usage | None, cost: float | None, total_cost: float) -> None:
        if usage is None:
            self.status("[dim]（这一轮没有拿到用量数据）[/]")
            return

        parts = [
            f"输入 {usage.input_tokens:,}（缓存命中 {usage.cached_input_tokens:,}）",
            f"输出 {usage.output_tokens:,}"
            + (f"（思考 {usage.reasoning_tokens:,}）" if usage.reasoning_tokens else ""),
        ]
        window = CONTEXT_WINDOW.get(model)
        if window:
            parts.append(f"上下文已用 {usage.input_tokens / window:.2%}")
        if cost is not None:
            parts.append(f"本轮 ${cost:.5f} · 累计 ${total_cost:.4f}" + (" · 高峰价" if is_peak() else ""))
        self.status("[dim]" + " · ".join(parts) + "[/]")

    def _render_markdown(self, content: str) -> None:
        if content.strip():
            self.console.print(Markdown(content))
