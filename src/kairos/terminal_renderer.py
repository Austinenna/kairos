"""Kairos 终端展示层。

本模块只负责把 Runtime 的正文事件和完整结果渲染到终端，不参与模型请求、工具执行
或数据库操作。Rich 是终端输出实现细节，Runtime 不依赖它。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pygments.style import Style as PygmentsStyle
from pygments.token import Comment, Error, Generic, Keyword, Name, Number, Operator, String, Token
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Heading, Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.syntax import PygmentsSyntaxTheme, Syntax
from rich.text import Text
from rich.theme import Theme

from .images import redact_images
from .pricing import CONTEXT_WINDOW, Usage, estimate_cost, is_peak
from .runtime import AgentTurnResult

_MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)

_INK = "#252e34"
_MUTED = "#758089"
_CYAN = "#087f8f"
_CODE_BACKGROUND = "#f3f6f7"


class _LightCodeStyle(PygmentsStyle):
    """与浅色正文协调的语法配色，不使用默认 Monokai 深色背景。"""

    background_color = _CODE_BACKGROUND
    styles = {
        Token: _INK,
        Comment: f"italic {_MUTED}",
        Keyword: "#77548f",
        Keyword.Type: _CYAN,
        Name.Builtin: _CYAN,
        Name.Function: "#275e98",
        Name.Class: _CYAN,
        Name.Exception: _CYAN,
        Name.Decorator: "#77548f",
        String: "#487743",
        Number: "#275e98",
        Operator: _MUTED,
        Generic.Heading: f"bold {_INK}",
        Generic.Subheading: f"bold {_INK}",
        Generic.Deleted: "#a43e44",
        Generic.Inserted: "#487743",
        Error: "#a43e44",
    }


_LIGHT_SYNTAX = PygmentsSyntaxTheme(_LightCodeStyle)
_MARKDOWN_THEME = Theme(
    {
        "markdown.code": Style(color=_CYAN, bgcolor="default", bold=False),
        "markdown.code_block": Style(color=_INK, bgcolor=_CODE_BACKGROUND),
        "markdown.block_quote": Style(color=_MUTED),
        "markdown.list": Style(),
        "markdown.item.number": Style(color=_MUTED),
        "markdown.table.border": Style(color="#d5dfe4"),
        "markdown.table.header": Style(color=_INK, bold=True),
        **{
            f"markdown.h{level}": Style(color=_INK, bold=True, underline=False, italic=False, dim=False)
            for level in range(1, 7)
        },
    },
    inherit=False,
)


class _LightHeading(Heading):
    LEVEL_ALIGN = {f"h{level}": "left" for level in range(1, 7)}


class _LightCodeBlock(CodeBlock):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        syntax = Syntax(str(self.text).rstrip(), self.lexer_name, theme=_LIGHT_SYNTAX, word_wrap=True)
        contents: list[Any] = []
        if self.lexer_name != "text":
            label = "Python" if self.lexer_name == "python" else self.lexer_name
            contents.append(Text(label, style=_MUTED))
        contents.append(syntax)
        yield Padding(Group(*contents), (1, 2), style=Style(bgcolor=_CODE_BACKGROUND))


class _LightMarkdown(Markdown):
    # 使用独立注册表，避免改变其他调用方的 Rich Markdown 行为。
    elements = {
        **Markdown.elements,
        "heading_open": _LightHeading,
        "fence": _LightCodeBlock,
        "code_block": _LightCodeBlock,
    }

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # 主题必须覆盖实际渲染过程，Live 重绘和最终全文也使用同一作用域。
        with console.use_theme(_MARKDOWN_THEME):
            yield from super().__rich_console__(console, options)


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
        self.console.print_json(data=redact_images(messages))

    def render_turn(self, result: AgentTurnResult, *, model: str, content_streamed: bool = False) -> float:
        """渲染一次 Agent 回合并返回本轮费用。

        作用:
            展示回答、截断提示和用量统计，不维护跨回合累计费用。

        参数:
            result: 运行时返回的完整回合结果。
            model: 用于查询价格和上下文窗口的模型名称。
            content_streamed: 正文是否已经流式显示，为真时不重复打印。

        返回:
            本轮估算费用（float，美元）；无法估算时返回 0.0，界面仍显示未知。

        行为说明:
            1. 按需展示正文和输出截断提示。
            2. 根据本轮用量估算费用，分别展示上下文窗口和本轮消耗。
        """

        if not content_streamed:
            self.render_content(result.content)
        if result.finish_reason in {"length", "max_tokens"}:
            self.status("[yellow]回复被截断了：输出达到了长度上限。[/]")

        cost = estimate_cost(model, result.usage) if result.usage else None
        self.render_usage(
            model,
            result.usage,
            cost,
            last_request_usage=result.last_request_usage,
        )
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
                parts.append(_LightMarkdown(content[cursor : match.start()]))
            parts.append(
                Panel(
                    Syntax(match.group(1).strip(), "text", theme=_LIGHT_SYNTAX, line_numbers=True, word_wrap=True),
                    title="Mermaid",
                    style=Style(color=_INK, bgcolor=_CODE_BACKGROUND),
                    border_style=_MUTED,
                )
            )
            cursor = match.end()
        if content[cursor:].strip():
            parts.append(_LightMarkdown(content[cursor:]))
        return Group(*parts)

    def render_usage(
        self,
        model: str,
        usage: Usage | None,
        cost: float | None,
        *,
        last_request_usage: Usage | None = None,
    ) -> None:
        """分行展示上下文窗口和本轮用量、费用。

        作用:
            将运行时用量投影为终端文字，避免累计输入被误认为单次上下文长度。

        参数:
            model: 用于查询上下文窗口的模型名称。
            usage: 本轮累计用量，缺失字段为 None。
            cost: 本轮估算费用，未知时为 None。
            last_request_usage: 最后一次请求的用量，其输入已包含那次发送的全部历史。

        返回:
            无返回值（None）；向终端打印统计文字。

        行为说明:
            1. 上下文窗口单独一行，按最后一次请求的完整输入计算占比，不含新输出。
            2. 本轮输入、输出与费用放在同一行；未知值不按零计费，不展示累计费用。
        """
        window = CONTEXT_WINDOW.get(model)
        latest_input = last_request_usage.input_tokens if last_request_usage is not None else None
        if latest_input is None:
            context = "上下文窗口 未知"
        elif window:
            context = f"上下文窗口 {latest_input:,} / {window:,}（{latest_input / window:.2%}）"
        else:
            context = f"上下文窗口 {latest_input:,}（上限未知）"
        self.status(f"[dim]{context}[/]")

        parts = []
        if usage is None:
            parts.append("本轮用量未知")
        else:
            counts = {
                name: f"{value:,}" if value is not None else "未知"
                for name, value in (
                    ("input", usage.input_tokens),
                    ("cached", usage.cached_input_tokens),
                    ("output", usage.output_tokens),
                    ("reasoning", usage.reasoning_tokens),
                )
            }
            parts.append(
                f"本轮输入 {counts['input']}（缓存命中 {counts['cached']}）"
                f" · 输出 {counts['output']}（思考 {counts['reasoning']}）"
            )
        if cost is not None:
            parts.append(f"本轮费用 ${cost:.5f}" + (" · 高峰价" if is_peak() else ""))
        else:
            parts.append("本轮费用未估算")
        self.status("[dim]" + " · ".join(parts) + "[/]")
