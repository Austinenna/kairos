"""通过真实 PromptSession 的屏幕布局验证固定底部输入区。"""

import io
import threading
import time
from dataclasses import dataclass

import pytest
from PIL import Image
from prompt_toolkit import renderer as prompt_renderer
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from kairos.images import prepare_image
from kairos.terminal_input import InputDraft, TerminalInput
from kairos.terminal_renderer import TerminalRenderer
from kairos.usage_stats import summarize_usage

_USAGE = (
    "上下文窗口 3,206 / 1,048,576（0.31%）\n会话输入 9,585（缓存命中 6,016） · 输出 48（思考 未知） · 会话费用 $0.00116"
)


class _SizedOutput(DummyOutput):
    """提供可调整大小的终端，且不向真实终端写入内容。"""

    def __init__(self, rows: int, columns: int) -> None:
        self.size = Size(rows=rows, columns=columns)

    def get_size(self) -> Size:
        return self.size

    def get_rows_below_cursor_position(self) -> int:
        return self.size.rows


@dataclass(frozen=True)
class _Frame:
    lines: tuple[str, ...]
    done: bool
    cursor_y: int

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _capture_frames(monkeypatch, editor: TerminalInput) -> list[_Frame]:
    """捕获真实渲染器写入终端之前的屏幕，包括提交时的最终帧。

    作用:
        同时验证可见布局和最终留在滚动历史中的内容，不依赖 ANSI 字符串。

    参数:
        monkeypatch: pytest 的临时替换工具。
        editor: 当前测试专用的输入编辑器。

    返回:
        按绘制顺序追加的屏幕快照列表。

    行为说明:
        1. 只记录当前编辑器，随后仍调用原始渲染方法。
        2. 按终端格读取文本，保留行位置和中文字符宽度对应的空占位。
    """
    frames = []
    original = prompt_renderer._output_screen_diff

    def capture(app, output, screen, *args, **kwargs):
        if app is editor._session.app:
            lines = tuple(
                "".join(screen.data_buffer[y][x].char for x in range(output.get_size().columns)).rstrip()
                for y in range(screen.height)
            )
            frames.append(_Frame(lines, args[4], screen.get_cursor_position(app.layout.current_window).y))
        return original(app, output, screen, *args, **kwargs)

    monkeypatch.setattr(prompt_renderer, "_output_screen_diff", capture)
    return frames


def _until(predicate) -> None:
    """等待输入线程或渲染帧达到条件，并限制失败测试的等待时间。

    作用:
        协调测试驱动线程和 PromptSession 事件循环。

    参数:
        predicate: 无参数条件函数。

    返回:
        条件满足时返回 None。

    异常:
        AssertionError: 四秒内条件仍未满足。

    行为说明:
        1. 短暂等待允许主线程处理输入和重绘。
        2. 超时立即失败，避免测试永久等待。
    """
    deadline = time.monotonic() + 4
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("固定底栏测试等待输入或绘制超时")
        threading.Event().wait(0.01)


def _drive(pipe, editor: TerminalInput, script, *, draft: InputDraft | None = None) -> InputDraft:
    """在真实事件循环运行期间驱动输入和检查屏幕。

    作用:
        让测试覆盖 PromptSession 的实际输入、绘制和退出流程。

    参数:
        pipe: 测试专用的管道输入。
        editor: 当前输入编辑器。
        script: 后台线程运行的检查和输入步骤。
        draft: 可选的初始文字与附件。

    返回:
        输入编辑器实际提交的 InputDraft。

    异常:
        BaseException: 驱动线程发生错误时，向测试主线程重新抛出原错误。

    行为说明:
        1. 等待编辑器开始运行后再执行脚本。
        2. 脚本失败时发送 Ctrl+C 结束输入，保留原断言错误。
    """
    failures = []

    def send():
        try:
            _until(lambda: editor._session.app.is_running)
            script()
        except BaseException as error:
            failures.append(error)
            pipe.send_bytes(b"\x03")

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    try:
        return editor.read(draft=draft)
    finally:
        worker.join(timeout=5)
        if failures:
            raise failures[0]


@pytest.mark.parametrize("columns", [120, 40])
def test_footer_stays_at_bottom_updates_and_disappears_from_history(monkeypatch, columns):
    """验证宽窄终端的底部位置、统计更新和提交后的历史清理。

    作用:
        防止每轮统计重复进入历史，或窄屏换行把输入区挤离底部。

    参数:
        monkeypatch: pytest 的临时替换工具。
        columns: 当前测试的终端列数。

    返回:
        None；断言失败时由 pytest 报告差异。

    行为说明:
        1. 等待真实绘制，确认统计全文和输入框均可见。
        2. 更新统计后确认旧值消失，再提交并检查最终帧。
    """
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=_SizedOutput(24, columns), budget_check=lambda _: None)
        try:
            editor.set_usage(_USAGE)
            frames = _capture_frames(monkeypatch, editor)

            def check_and_send():
                _until(lambda: frames and "上下文窗口" in frames[-1].text)
                frame = frames[-1]
                assert len(frame.lines) == 24
                assert "Ctrl+V" in frame.lines[-1]
                assert "你 ›" in frame.lines[-3]
                assert frame.cursor_y == 21
                assert "".join(_USAGE.split()) in "".join("".join(frame.lines[:-3]).split())
                assert all(not line for line in frame.lines[:10])

                editor.set_usage(_USAGE.replace("9,585", "9,999"))
                _until(lambda: "9,999" in frames[-1].text)
                assert "9,585" not in frames[-1].text
                assert frames[-1].text.count("上下文窗口") == 1
                pipe.send_text("测试发送\r")

            assert _drive(pipe, editor, check_and_send) == InputDraft("测试发送")
            assert frames[-1].done
            assert "上下文窗口" not in frames[-1].text
            assert "会话输入" not in frames[-1].text
            assert "Ctrl+V" not in frames[-1].text
            assert len(frames[-1].lines) < 5
        finally:
            editor.close()


def test_multiline_input_and_image_attachment_keep_footer_at_bottom(monkeypatch):
    """验证多行问题和图片附件共同使用底部输入区。

    作用:
        防止固定布局破坏图片草稿和 Alt+Enter 换行。

    参数:
        monkeypatch: pytest 的临时替换工具。

    返回:
        None；断言失败时由 pytest 报告差异。

    行为说明:
        1. 使用内存生成的图片恢复附件草稿，不读取用户文件。
        2. 输入第二行后检查统计、输入和附件的顺序，再确认提交内容。
    """
    image_bytes = io.BytesIO()
    Image.new("RGB", (12, 8), "red").save(image_bytes, format="PNG")
    image = prepare_image(image_bytes.getvalue(), "示例.png")
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=_SizedOutput(24, 120), budget_check=lambda _: None)
        try:
            editor.set_usage(_USAGE)
            frames = _capture_frames(monkeypatch, editor)

            def check_and_send():
                _until(lambda: frames and "示例.png" in frames[-1].text)
                pipe.send_text("\x1b\r第二行")
                _until(lambda: "第二行" in frames[-1].text)
                frame = frames[-1]
                assert "Ctrl+V" in frame.lines[-1]
                assert "示例.png" in frame.lines[-2]
                assert "第二行" in frame.lines[-4]
                assert "你 › 第一行" in frame.lines[-5]
                assert frame.cursor_y == 20
                assert frame.text.index("上下文窗口") < frame.text.index("你 ›") < frame.text.index("示例.png")
                pipe.send_text("\r")

            result = _drive(pipe, editor, check_and_send, draft=InputDraft("第一行", (image,)))
            assert [line.rstrip() for line in result.text.splitlines()] == ["第一行", "第二行"]
            assert result.images == (image,)
            assert "上下文窗口" not in frames[-1].text
        finally:
            editor.close()


def test_resize_reflows_usage_while_preserving_input(monkeypatch):
    """验证缩小终端时统计重新换行且草稿保持不变。

    作用:
        检查固定区域在终端尺寸变化后仍有可用输入空间。

    参数:
        monkeypatch: pytest 的临时替换工具。

    返回:
        None；断言失败时由 pytest 报告差异。

    行为说明:
        1. 首次绘制后缩小输出尺寸并请求重绘。
        2. 检查用量末尾、底部快捷键与草稿后再提交。
    """
    with create_pipe_input() as pipe:
        output = _SizedOutput(24, 120)
        editor = TerminalInput(input=pipe, output=output, budget_check=lambda _: None)
        try:
            editor.set_usage(_USAGE)
            frames = _capture_frames(monkeypatch, editor)

            def check_and_send():
                _until(lambda: frames and "保留草稿" in frames[-1].text)
                output.size = Size(rows=15, columns=40)
                editor._session.app.invalidate()
                _until(lambda: len(frames[-1].lines) == 15)
                assert "$0.00116" in frames[-1].text
                assert "保留草稿" in frames[-1].lines[-3]
                assert "Ctrl+V" in frames[-1].lines[-1]
                pipe.send_text("\r")

            assert _drive(pipe, editor, check_and_send, draft=InputDraft("保留草稿")) == InputDraft("保留草稿")
        finally:
            editor.close()


def test_renderer_routes_usage_to_footer_without_printing_and_preserves_plain_output():
    """验证交互底栏与普通文字输出共用统计文本。

    作用:
        防止绑定底栏后统计仍被重复打印，同时保留非交互模式输出。

    参数:
        无。

    返回:
        None；断言失败时由 pytest 报告差异。

    行为说明:
        1. 启用回调时仅交付两行用量文本。
        2. 取消回调后仍向普通 Console 输出统计。
    """
    output = io.StringIO()
    renderer = TerminalRenderer(Console(file=output, width=160, color_system=None))
    updates = []
    renderer.usage_callback = updates.append
    renderer.render_usage("unknown-model", summarize_usage([]))
    assert output.getvalue() == ""
    assert len(updates) == 1
    assert updates[0].splitlines()[0] == "上下文窗口 未知"
    assert "会话输入 0" in updates[0].splitlines()[1]

    renderer.usage_callback = None
    renderer.render_usage("unknown-model", summarize_usage([]))
    assert output.getvalue().splitlines() == updates[0].splitlines()
