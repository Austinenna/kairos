"""带图片附件的终端输入区，不负责模型调用。

Ctrl+V 显式读取系统剪贴板；终端 bracketed paste 只处理实际粘贴文字，
不会顺带读取剪贴板中的旧图片。图片加载在单个后台线程串行执行。
"""

import asyncio
import shlex
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from urllib.parse import unquote, urlparse

from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, HSplit, Window
from prompt_toolkit.utils import is_dumb_terminal

from .clipboard import ClipboardContent, read_clipboard
from .images import ImageInput, load_image, prepare_image


@dataclass(frozen=True)
class InputDraft:
    text: str
    images: tuple[ImageInput, ...] = ()
    image_labels: tuple[str, ...] = ()


class TerminalInputUnsupportedError(RuntimeError):
    """终端缺少附件输入区需要的光标控制能力。"""


@dataclass
class _Attachment:
    number: int
    name: str
    image: ImageInput | None = None
    error: str = ""
    active: bool = True
    label: str = ""

    @property
    def marker(self) -> str:
        return f"[{self.label or f'图片 {self.number}'}]"


# 只识别明确的独立图片路径；正文里出现路径不构成自动上传。
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def _as_image_path(value: str) -> Path | None:
    if value.startswith("file://"):
        parsed = urlparse(value)
        if parsed.netloc not in ("", "localhost"):
            return None
        value = unquote(parsed.path)
    path = Path(value).expanduser()
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        return None
    if not path.is_absolute() and not value.startswith(("./", "../")):
        # 单独的文件名只在确实存在时作为路径，避免把普通文字误认为附件。
        if not path.is_file():
            return None
    return path


def _image_paths(text: str) -> tuple[Path, ...]:
    """识别整个 paste/输入恰好是图片路径（可有引号及 shell 转义）。"""
    try:
        return _parse_image_paths(text)
    except (OSError, RuntimeError, ValueError):
        # 普通正文也经过这一步；~用户名解析和过长文件名不能打断输入。
        return ()


def _parse_image_paths(text: str) -> tuple[Path, ...]:
    value = text.strip()
    if not value or "\n" in value or "\r" in value or len(value) > 8192:
        return ()
    # 允许无引号但确实存在的带空格路径。
    literal = _as_image_path(value)
    if literal is not None and literal.is_file():
        return (literal,)
    try:
        tokens = shlex.split(value)
    except ValueError:
        return ()
    if not tokens:
        return ()
    paths = tuple(_as_image_path(token) for token in tokens)
    if any(path is None for path in paths):
        return ()
    return tuple(path for path in paths if path is not None)


class TerminalInput:
    """可注入 pipe input、DummyOutput 和 clipboard_reader 的交互输入。

    read() 延续内部草稿，read(draft=...) 替换并恢复草稿。普通提交和 /file
    返回图片并清空内部草稿；其他斜杠命令只返回文字，附件继续保留。
    调用方执行 /reset 或更换 system 后应 clear()；退出时应 close()。
    """

    def __init__(
        self,
        *,
        budget_check: Callable[[Sequence[ImageInput]], None],
        input=None,
        output=None,
        clipboard_reader: Callable[[], ClipboardContent] | None = None,
    ) -> None:
        if output is None and is_dumb_terminal():
            # PromptSession 的 dumb 回退会更换 Application、跳过 pre_run，
            # 并用默认 Enter 覆盖我们的提交逻辑。不能静默退化成 str 输入。
            raise TerminalInputUnsupportedError("当前终端类型不支持图片输入区，请使用支持 ANSI 的交互终端。")
        self._budget_check = budget_check
        self._clipboard_reader = clipboard_reader or read_clipboard
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kairos-images")
        self._results: SimpleQueue[tuple[int, _Attachment, Future]] = SimpleQueue()
        self._attachments: list[_Attachment] = []
        self._next_number = 1
        self._generation = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._restoring = False
        self._text = ""
        self._cursor = 0
        self._notice = ""
        bindings = KeyBindings()

        @bindings.add("c-v")
        def clipboard_paste(event):
            attachment = self._new_attachment("剪贴板", separator=False)
            self._enqueue(attachment, self._clipboard_reader)

        @bindings.add(Keys.BracketedPaste)
        def paste(event):
            self._paste_text(event.data)

        @bindings.add("enter")
        @bindings.add("c-j")
        def submit(event):
            self._submit(event)

        @bindings.add("escape", "enter")
        def newline(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("backspace")
        def backspace(event):
            if not self._delete_attachment(backward=True):
                event.current_buffer.delete_before_cursor(count=event.arg)

        @bindings.add("delete")
        def delete(event):
            if not self._delete_attachment(backward=False):
                event.current_buffer.delete(count=event.arg)

        self._session = PromptSession(
            message=[("fg:ansicyan bold", "你 › ")],
            key_bindings=bindings,
            input=input,
            output=output,
            multiline=False,
            validate_while_typing=False,
            complete_while_typing=False,
            bottom_toolbar=self._toolbar,
            reserve_space_for_menu=0,
        )
        # 空行放在反白底栏之外，并随底栏一起隐藏，保留原有颜色和提示内容。
        for container in self._session.layout.container.children:
            if (
                isinstance(container, ConditionalContainer)
                and isinstance(container.content, Window)
                and container.content.style == "class:bottom-toolbar"
            ):
                container.content = HSplit([Window(height=1), container.content])
                break
        self._session.default_buffer.on_text_changed += self._on_change

    @property
    def _buffer(self):
        return self._session.default_buffer

    def _on_change(self, buffer) -> None:
        if self._restoring:
            return
        self._text = buffer.text
        self._cursor = buffer.cursor_position
        for attachment in self._attachments:
            if attachment.marker not in buffer.text:
                attachment.active = False
        self._attachments = [item for item in self._attachments if item.active]
        self._notice = ""

    def _new_attachment(
        self, name: str, image: ImageInput | None = None, label: str = "", *, separator: bool = True
    ) -> _Attachment:
        attachment = _Attachment(self._next_number, name, image=image, label=label)
        self._next_number += 1
        self._attachments.append(attachment)
        self._buffer.insert_text(attachment.marker + (" " if separator else ""))
        return attachment

    def _enqueue(self, attachment: _Attachment, work: Callable) -> None:
        generation = self._generation
        future = self._executor.submit(work)

        def complete(done: Future) -> None:
            self._results.put((generation, attachment, done))
            loop = self._loop
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(self._drain_results)
                except RuntimeError:
                    pass  # 命令返回的瞬间事件循环可能已关闭，下次 read 会取出结果。

        future.add_done_callback(complete)

    def _paste_text(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paths = _image_paths(text)
        if paths:
            for path in paths:
                attachment = self._new_attachment(path.name)
                self._enqueue(attachment, lambda path=path: load_image(path))
        else:
            self._buffer.insert_text(text)

    def _drain_results(self) -> None:
        while True:
            try:
                generation, attachment, future = self._results.get_nowait()
            except Empty:
                break
            if self._closed or generation != self._generation or not attachment.active:
                continue
            try:
                result = future.result()
                if isinstance(result, ClipboardContent):
                    self._accept_clipboard(attachment, result)
                    continue
                candidates = [item.image for item in self._attachments if item.image is not None]
                self._budget_check([*candidates, result])
                attachment.image = result
                attachment.name = result.name
            except Exception as exc:
                attachment.error = str(exc) or "读取图片失败，请删除附件后重试。"
        if self._session.app.is_running:
            self._session.app.invalidate()

    def _replace_marker(self, attachment: _Attachment, text: str) -> None:
        current = self._buffer.text
        index = current.find(attachment.marker)
        if index < 0:
            return
        end = index + len(attachment.marker)
        cursor = self._buffer.cursor_position
        if cursor >= end:
            cursor += len(text) - len(attachment.marker)
        elif cursor > index:
            cursor = index + len(text)
        self._buffer.document = Document(current[:index] + text + current[end:], cursor)

    def _accept_clipboard(self, attachment: _Attachment, content: ClipboardContent) -> None:
        paths = content.paths or (_image_paths(content.text) if content.image is None else ())
        if paths:
            images = tuple(path for path in paths if path.suffix.lower() in _IMAGE_SUFFIXES)
            if len(images) != len(paths):
                self._replace_marker(attachment, content.text or " ".join(map(str, content.paths)))
                return
            first, *rest = images
            attachment.name = first.name
            self._enqueue(attachment, lambda: load_image(first))
            # 其余文件也进入同一个顺序工作队列，不并发消耗图片解码内存。
            for path in rest:
                other = self._new_attachment(path.name)
                self._enqueue(other, lambda path=path: load_image(path))
        elif content.image is not None:
            attachment.name = "剪贴板.png"
            self._enqueue(attachment, lambda: prepare_image(content.image, "剪贴板.png"))
        else:
            self._replace_marker(attachment, content.text)
            if not content.text:
                self._notice = "剪贴板中没有图片或文字。"

    def _delete_attachment(self, *, backward: bool) -> bool:
        cursor = self._buffer.cursor_position
        for attachment in self._attachments:
            start = self._buffer.text.find(attachment.marker)
            end = start + len(attachment.marker)
            hit = start < cursor <= end if backward else start <= cursor < end
            if start >= 0 and hit:
                self._replace_marker(attachment, "")
                return True
        return False

    def _clean_text(self) -> str:
        text = self._buffer.text
        for attachment in self._attachments:
            text = text.replace(attachment.marker, "")
        return text.strip()

    def _submit(self, event) -> None:
        self._drain_results()
        text = self._clean_text()
        paths = _image_paths(text)
        if paths:
            # 裸终端拖放可能只是逐字输入路径而非 bracketed paste；只在整段
            # 草稿是独立图片路径时于 Enter 转附件，不扫描普通正文。
            self._restoring = True
            retained = " ".join(item.marker for item in self._attachments)
            self._buffer.document = Document(retained + (" " if retained else ""))
            self._buffer.cursor_position = len(self._buffer.text)
            self._restoring = False
            self._paste_text(text)
            return
        command = text.split(maxsplit=1)[0] if text.startswith("/") else ""
        if command and command != "/file":
            # 帮助/历史/思考等命令不消耗附件；加载中也可以退出或清空。
            self._restoring = True
            self._text = " ".join(item.marker for item in self._attachments)
            self._cursor = len(self._text)
            self._buffer.document = Document(self._text, self._cursor)
            self._restoring = False
            event.app.exit(result=InputDraft(text))
            return
        if not text and not self._attachments:
            return
        errors = [item for item in self._attachments if item.error]
        if errors:
            self._notice = "请先删除失败的图片附件，再发送。"
            return
        if any(item.image is None for item in self._attachments):
            self._notice = "图片正在加载，请稍后按 Enter 发送。"
            return
        images = tuple(item.image for item in self._attachments if item.image is not None)
        try:
            self._budget_check(images)
        except Exception as exc:
            self._notice = str(exc)
            return
        labels = tuple(item.label or f"图片 {item.number}" for item in self._attachments)
        result = InputDraft(text, images, labels)
        self.clear()
        event.app.exit(result=result)

    def _toolbar(self):
        parts = []
        for item in self._attachments:
            if item.error:
                label = f"{item.marker} {item.name} · 失败：{item.error}"
                style = "fg:ansired"
            elif item.image is None:
                label = f"{item.marker} {item.name} · 加载中"
                style = "fg:ansiyellow"
            else:
                label = (
                    f"{item.marker} {item.name} · 原始 {item.image.original_width}×{item.image.original_height}"
                    f" → 发送 {item.image.width}×{item.image.height}"
                )
                style = "fg:ansicyan"
            # FormattedText tuples，不将文件名当 HTML/Rich markup；清掉控制字符。
            label = "".join(char if char >= " " and char != "\x7f" else " " for char in label)
            parts.append((style, label + "\n"))
        notice = self._notice or "Ctrl+V 粘贴图片 · Enter 发送 · Alt+Enter 换行 · Backspace 删除图片标签"
        parts.append(("", "".join(char if char >= " " and char != "\x7f" else " " for char in notice)))
        return parts

    def read(self, draft: InputDraft | None = None) -> InputDraft:
        if self._closed:
            raise RuntimeError("输入区已关闭。")
        if draft is not None:
            self.clear()
            text = draft.text
            if draft.images and text and not text.endswith(" "):
                text += " "
            for index, image in enumerate(draft.images):
                label = draft.image_labels[index] if index < len(draft.image_labels) else ""
                attachment = _Attachment(self._next_number, image.name, image=image, label=label)
                self._next_number += 1
                if label.startswith("图片 ") and label[3:].isdigit():
                    self._next_number = max(self._next_number, int(label[3:]) + 1)
                self._attachments.append(attachment)
                text += attachment.marker + " "
            self._text = text
            self._cursor = len(text)
        self._restoring = True

        def ready() -> None:
            self._restoring = False
            self._loop = asyncio.get_running_loop()
            self._drain_results()

        try:
            return self._session.prompt(default=Document(self._text, self._cursor), pre_run=ready)
        finally:
            self._loop = None
            self._restoring = False

    def clear(self) -> None:
        self._generation += 1
        for attachment in self._attachments:
            attachment.active = False
        self._attachments.clear()
        self._text = ""
        self._cursor = 0
        self._notice = ""
        self._buffer.document = Document("")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
