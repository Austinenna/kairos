"""输入框：编辑文字，并支持三种方式添加图片。

- Ctrl+V：读取系统剪贴板，可以是截图，也可以是在 Finder 里复制的图片文件。
- 拖入图片文件：终端会粘贴文件路径，路径都是图片时自动转成附件。
- 输入框里只有图片路径时按 Enter：同样转成附件（有些终端拖入时不走粘贴）。

待发送的图片显示在输入框下方；输入框为空时按 Backspace 删除最后一张。
"""

import shlex
import sys
from pathlib import Path

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import VerticalAlign
from prompt_toolkit.styles import Style

from ..clipboard import ClipboardError, read_clipboard
from ..images import IMAGE_SUFFIXES, Image, load_image, prepare_image

PROMPT_STYLE = "fg:ansicyan bold"
HINT = "Enter 发送 · Alt+Enter 换行 · Ctrl+V 贴图 · Esc 中断回答 · 连按两次 Ctrl+C 退出"


class Prompt:
    def __init__(self):
        self.images: list[Image] = []  # 待发送的图片
        self.notice = ""  # 输入框下方的一次性提示，例如图片读取失败
        self.status = ""  # 输入框下方的状态行，由主循环在每次读取前传入
        self.interactive = sys.stdin.isatty()
        if self.interactive:
            self.session = PromptSession(
                key_bindings=self._key_bindings(),
                bottom_toolbar=self._toolbar,
                reserve_space_for_menu=0,  # 没有补全菜单，不用预留空行
                erase_when_done=True,  # 发送后擦掉整个输入框（横线、状态行），再由 _echo 留下一行输入内容
                prompt_continuation=" " * 5,  # 多行输入时，后面几行和“你 › ”后的文字对齐
                style=Style.from_dict({"bottom-toolbar": "noreverse #758089", "rule": "#758089"}),
            )
            # prompt_toolkit 会占满光标以下的所有行，输入区向下撑开，状态行被推到屏幕最底部。
            # 改成输入区只占实际用到的行数、内容靠上排列，下方横线才能紧贴输入框。
            self.session.layout.current_window.dont_extend_height = to_filter(True)
            self.session.layout.container.align = VerticalAlign.TOP

    def read(self, status: str = "") -> tuple[str, list[Image]]:
        """读取一次输入，返回 (文字, 图片)。Ctrl+C 抛出 KeyboardInterrupt，Ctrl+D 抛出 EOFError。"""
        if not self.interactive:  # 管道输入：只读文字
            return input().strip(), []
        self.status = status
        while True:
            text = self.session.prompt(self._message).strip()
            paths = image_paths(text)
            if not paths:
                break
            for path in paths:
                self._attach(lambda path=path: load_image(path))
        images, self.images, self.notice = self.images, [], ""
        self._echo(text, images)
        return text, images

    def _echo(self, text: str, images: list[Image]) -> None:
        """在终端历史里留下这次的输入：“你 › 内容”，加上附带的图片名。"""
        lines = [("", "\n"), (PROMPT_STYLE, "你 › "), ("", text.replace("\n", "\n     "))]
        lines += [("class:rule", f"\n     [图片 {n}] {image.name}") for n, image in enumerate(images, 1)]
        print_formatted_text(FormattedText(lines), style=self.session.style)

    def _key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("c-v")
        def paste_clipboard(event):
            try:
                content = read_clipboard()
            except ClipboardError as error:
                self.notice = str(error)
                return
            if content.image:
                self._attach(lambda: prepare_image(content.image, "剪贴板图片.png"))
            elif content.paths:
                for path in content.paths:
                    self._attach(lambda path=path: load_image(path))
            else:
                event.current_buffer.insert_text(content.text)

        @bindings.add(Keys.BracketedPaste)
        def paste_text(event):
            paths = image_paths(event.data)
            for path in paths:
                self._attach(lambda path=path: load_image(path))
            if not paths:
                event.current_buffer.insert_text(event.data)

        @bindings.add("escape", "enter")
        def newline(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("backspace")
        def backspace(event):
            if not event.current_buffer.text and self.images:
                self.images.pop()
            else:
                event.current_buffer.delete_before_cursor()

        return bindings

    def _attach(self, load) -> None:
        """加载一张图片；失败时把原因显示在输入框下方，不中断输入。"""
        try:
            self.images.append(load())
            self.notice = ""
        except Exception as error:
            self.notice = f"图片无法读取：{type(error).__name__}: {error}"

    def _message(self) -> list[tuple[str, str]]:
        """输入框上方的空行、横线和提示符。每次重绘时调用，横线跟着终端宽度变。"""
        return [("class:rule", "\n" + _rule() + "\n"), (PROMPT_STYLE, "你 › ")]

    def _toolbar(self) -> str:
        lines = [_rule()]  # 输入框下方的横线，和上方的一起把输入框框起来
        lines += [self.status] if self.status else []
        lines += [f"[图片 {n}] {image.name}  {image.width}×{image.height}" for n, image in enumerate(self.images, 1)]
        if self.images and not self.notice:
            lines.append("输入框为空时 Backspace 删除最后一张图片")
        lines.append(self.notice or HINT)
        return "\n".join(lines)


def _rule() -> str:
    return "─" * get_app().output.get_size().columns


def image_paths(text: str) -> list[Path]:
    """如果文字全部是图片文件路径，返回这些路径；否则返回空列表。

    终端粘贴路径时可能带引号，或用反斜杠转义空格，shlex.split 会按 shell 规则还原。
    """
    try:
        parts = shlex.split(text)
    except ValueError:  # 引号不配对，不是路径
        return []
    paths = [Path(part).expanduser() for part in parts]
    if paths and all(path.suffix.lower() in IMAGE_SUFFIXES and path.is_file() for path in paths):
        return paths
    return []
