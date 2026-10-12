"""Kairos 终端展示层。

本模块只负责把 Runtime 的正文事件和完整结果渲染到终端，不参与模型请求、工具执行
或数据库操作。Rich 是终端输出实现细节，Runtime 不依赖它。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.syntax import Syntax
from rich.text import Text

from .pricing import CONTEXT_WINDOW, Usage, estimate_cost, is_peak
from .runtime import AgentTurnResult

_MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


class _StreamPreview:
    """活动回答超过屏高时展示末尾，确保新收到的正文仍然可见。"""

    def __init__(self, content: Group) -> None:
        self.content = content

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = console.render_lines(self.content, options, pad=False)
        height = max(1, options.size.height - 1)
        if len(lines) <= height:
            yield self.content
            return
        yield Text("…", style="dim")
        for index, line in enumerate(lines[-height:]):
            if index:
                yield Segment.line()
            yield from line


class TerminalRenderer:
    """使用 Rich 将 Agent 结果渲染成终端内容。"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._parts: list[str] = []
        self._live: Live | None = None

    @contextmanager
    def stream_output(self) -> Iterator[None]:
        """覆盖整个 Agent 回合；任何退出路径都关闭活动显示区。"""

        try:
            yield
        finally:
            self.finish_stream()

    def handle_event(self, event: dict[str, Any]) -> None:
        """只把正文作为答案展示；工具参数和结果留在运行时。"""

        kind = event["event"]
        if kind in {"model_request", "model_response", "model_error"}:
            self.finish_stream()
        elif kind == "text_delta":
            self.append_text(event["text"])

    def append_text(self, text: str) -> None:
        if not text:
            return
        self._parts.append(text)
        if not (
            getattr(self.console, "is_interactive", False)
            and getattr(self.console, "is_terminal", False)
            and not getattr(self.console, "is_dumb_terminal", False)
        ):
            # 管道和文件输出也逐段写入，避免 ANSI 重绘以及完成后重复打印。
            self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
            return

        content = _StreamPreview(self._content_renderable("".join(self._parts)))
        if self._live is None:
            self._live = Live(content, console=self.console, auto_refresh=False, vertical_overflow="ellipsis")
            self._live.start(refresh=True)
        else:
            self._live.update(content, refresh=True)

    def finish_stream(self) -> None:
        """每次模型响应完成时固定正文；Live.stop 会显示完整最终内容。"""

        live, self._live = self._live, None
        try:
            if live is not None:
                # 活动区域只显示末尾；响应结束后将全文一次性固定到终端历史。
                try:
                    live.update(self._content_renderable("".join(self._parts)), refresh=False)
                finally:
                    live.stop()
            elif self._parts:
                self.console.print()
        finally:
            self._parts.clear()

    def status(self, message: str) -> None:
        self.console.print(message)

    def error(self, message: str) -> None:
        self.console.print(message)

    def history(self, messages: list[dict]) -> None:
        self.console.print_json(data=messages)

    def render_turn(
        self, result: AgentTurnResult, *, model: str, total_cost: float, content_streamed: bool = False
    ) -> float:
        """渲染一次 Agent 回合，返回本轮费用供 CLI 累加。"""

        if not content_streamed:
            self.render_content(result.content)
        if result.finish_reason in {"length", "max_tokens"}:
            self.status("[yellow]回复被截断了：输出达到了长度上限。[/]")

        cost = estimate_cost(model, result.usage) if result.usage else None
        self.render_usage(model, result.usage, cost, total_cost + (cost or 0.0))
        return cost or 0.0

    def render_content(self, content: str) -> None:
        """渲染普通文本、Markdown 和 Mermaid 代码块。"""

        if not content:
            return

        self.console.print(self._content_renderable(content))

    def _content_renderable(self, content: str) -> Group:
        """静态和流式显示共用相同的 Markdown／Mermaid 排版。"""

        parts: list[Any] = []
        cursor = 0
        for match in _MERMAID_BLOCK.finditer(content):
            if content[cursor : match.start()].strip():
                parts.append(Markdown(content[cursor : match.start()]))
            parts.append(
                Panel(
                    Syntax(match.group(1).strip(), "text", line_numbers=True),
                    title="Mermaid",
                    border_style="cyan",
                )
            )
            cursor = match.end()
        if content[cursor:].strip():
            parts.append(Markdown(content[cursor:]))
        return Group(*parts)

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
