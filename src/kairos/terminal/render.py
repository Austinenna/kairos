"""终端输出：流式显示 Markdown 回答、工具调用和提示信息。

生成回答时，用 Rich 的 Live 区域实时刷新 Markdown，只显示最后一屏，长回答也能看到最新内容；
一段回答结束后收起 Live 区域，把完整的 Markdown 打印到终端历史里。
输出不是终端（比如重定向到文件）时，直接逐段打印原文。
"""

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import Segment
from rich.text import Text

CODE_THEME = "default"  # Pygments 的浅色代码配色


class Renderer:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.text = ""  # 当前这段回答已收到的正文
        self.live: Live | None = None

    def text_delta(self, chunk: str) -> None:
        """收到一段正文（传给 respond 的 on_text）。"""
        self.text += chunk
        if not self.console.is_terminal:
            print(chunk, end="", flush=True)
            return
        if self.live is None:
            self.live = Live(console=self.console, transient=True, refresh_per_second=8)
            self.live.start()
        self.live.update(_Tail(self.text))

    def end_text(self) -> None:
        """一段回答结束：收起实时区域，打印完整的 Markdown。"""
        if self.live:
            self.live.stop()
            self.live = None
        if self.text and self.console.is_terminal:
            self.console.print(Markdown(self.text, code_theme=CODE_THEME))
        elif self.text:
            print()
        self.text = ""

    def message(self, message: dict) -> None:
        """每条新消息保存后调用（传给 respond 的 on_message），这里只显示工具调用。"""
        for call in message.get("tool_calls", []):
            self.end_text()  # 先把工具调用之前的那段回答打印完，顺序才对
            function = call["function"]
            self.info(f"  → {function['name']} {function['arguments']}")

    def info(self, text: str) -> None:
        # 用 Text 而不是字符串：文字里的 [xxx] 不会被 Rich 当成样式标记。
        self.console.print(Text(text, style="dim"))

    def error(self, text: str) -> None:
        self.console.print(Text(text, style="red"))


class _Tail:
    """Rich 可渲染对象：渲染完整 Markdown，但只输出最后一屏的行。"""

    def __init__(self, text: str):
        self.text = text

    def __rich_console__(self, console, options):
        lines = console.render_lines(Markdown(self.text, code_theme=CODE_THEME), options, pad=False)
        for line in lines[-(console.height - 2) :]:
            yield from line
            yield Segment.line()
