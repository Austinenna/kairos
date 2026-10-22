"""输入框：编辑文字，并支持三种方式添加图片。

- Ctrl+V：读取系统剪贴板，可以是截图，也可以是在 Finder 里复制的图片文件。
- 拖入图片文件：终端会粘贴文件路径，路径都是图片时自动转成附件。
- 输入框里只有图片路径时按 Enter：同样转成附件（有些终端拖入时不走粘贴）。

待发送的图片显示在输入框下方；输入框为空时按 Backspace 删除最后一张。
"""

import shlex
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from ..clipboard import ClipboardError, read_clipboard
from ..images import IMAGE_SUFFIXES, Image, load_image, prepare_image

HINT = "Enter 发送 · Alt+Enter 换行 · Ctrl+V 粘贴图片 · 输入框为空时 Backspace 删除最后一张图片"


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
                style=Style.from_dict({"bottom-toolbar": "noreverse #758089"}),
            )

    def read(self, status: str = "") -> tuple[str, list[Image]]:
        """读取一次输入，返回 (文字, 图片)。Ctrl+C 抛出 KeyboardInterrupt，Ctrl+D 抛出 EOFError。"""
        if not self.interactive:  # 管道输入：只读文字
            return input().strip(), []
        self.status = status
        while True:
            text = self.session.prompt([("fg:ansicyan bold", "你 › ")]).strip()
            paths = image_paths(text)
            if not paths:
                break
            for path in paths:
                self._attach(lambda path=path: load_image(path))
        images, self.images, self.notice = self.images, [], ""
        return text, images

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

    def _toolbar(self) -> str:
        lines = [self.status] if self.status else []
        lines += [f"[图片 {n}] {image.name}  {image.width}×{image.height}" for n, image in enumerate(self.images, 1)]
        lines.append(self.notice or HINT)
        return "\n".join(lines)


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
