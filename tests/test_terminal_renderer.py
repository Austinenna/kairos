from io import StringIO

import pytest
from rich.console import Console

from kairos.pricing import Usage
from kairos.runtime import AgentTurnResult
from kairos.terminal_renderer import TerminalRenderer


def make_renderer():
    output = StringIO()
    console = Console(file=output, width=120, color_system=None)
    return TerminalRenderer(console), output


def test_renderer_formats_markdown_table_once():
    renderer, output = make_renderer()
    result = AgentTurnResult(
        content="""# 查询结果

共有 **1 条**待办。

| ID | 待办 | 优先级 | 状态 |
|---|---|---|---|
| 1 | 学习 Agent | 普通 | 未完成 |
""",
        messages=[],
        usage=None,
        finish_reason=None,
        events=[],
        tool_results=[],
    )

    renderer.render_turn(result, model="unknown-model")

    text = output.getvalue()
    assert "学习 Agent" in text
    assert "查询结果" in text
    assert "1 条" in text


def test_renderer_formats_mermaid_as_terminal_code_panel_and_usage():
    renderer, output = make_renderer()
    result = AgentTurnResult(
        content="""架构如下：

```mermaid
flowchart TD
    CLI --> Runtime
```
""",
        messages=[],
        usage=Usage(input_tokens=100, cached_input_tokens=10, output_tokens=20, reasoning_tokens=0),
        finish_reason=None,
        events=[],
        tool_results=[],
    )

    renderer.render_turn(result, model="unknown-model")

    text = output.getvalue()
    assert "Mermaid" in text
    assert "flowchart TD" in text
    assert "输入 100" in text
    assert "输出 20" in text


def test_pipe_output_is_immediate_ignores_tools_and_never_duplicates_final_answer():
    renderer, output = make_renderer()
    with renderer.stream_output():
        renderer.handle_event({"event": "model_request"})
        renderer.handle_event({"event": "text_delta", "text": "先查"})
        assert output.getvalue() == "先查"
        renderer.handle_event({"event": "text_delta", "text": "一下。"})
        renderer.handle_event({"event": "model_response"})
        renderer.handle_event({"event": "tool_result", "result": "不要打印工具 JSON"})
        renderer.handle_event({"event": "model_request"})
        renderer.handle_event({"event": "text_delta", "text": "已经完成。"})
        renderer.handle_event({"event": "model_response"})
    result = AgentTurnResult("已经完成。", [], None, "stop", [], [])
    renderer.render_turn(result, model="test", content_streamed=True)
    rendered = output.getvalue()
    assert rendered.startswith("先查一下。\n已经完成。\n")
    assert rendered.count("已经完成。") == 1
    assert "不要打印工具 JSON" not in rendered
    assert "\x1b" not in rendered


def test_live_terminal_keeps_latest_text_visible_beyond_screen_height(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    console = Console(file=output, force_terminal=True, force_interactive=True, color_system=None, width=60, height=8)
    renderer = TerminalRenderer(console)
    with renderer.stream_output():
        renderer.append_text("\n\n".join(f"段落{i}" for i in range(20)) + "\n\n")
        position = output.tell()
        renderer.append_text("最新到达的正文")
        assert "最新到达的正文" in output.getvalue()[position:]
        position = output.tell()
    # 收尾固定全文，包含活动期间因屏高隐藏的最初段落，同时恢复光标。
    final_frame = output.getvalue()[position:]
    assert "段落0" in final_frame
    assert "最新到达的正文" in final_frame
    assert "\x1b[?25h" in final_frame


def test_live_markdown_table_and_mermaid_are_finalized_and_interrupt_releases_display(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    console = Console(file=output, force_terminal=True, force_interactive=True, color_system=None, width=90, height=30)
    renderer = TerminalRenderer(console)
    with pytest.raises(KeyboardInterrupt), renderer.stream_output():
        renderer.append_text("| 待办 | 状态 |\n|---|---|\n| 牛奶")
        renderer.append_text(" | 未完成 |\n\n```mermaid\nA --> B\n```")
        position = output.tell()
        raise KeyboardInterrupt
    final_frame = output.getvalue()[position:]
    assert "牛奶" in final_frame and "未完成" in final_frame
    assert "Mermaid" in final_frame and "A --> B" in final_frame
    assert "\x1b[?25h" in final_frame
    # 下一个回合不能重复使用上轮 buffer。
    position = output.tell()
    with renderer.stream_output():
        renderer.append_text("下一轮回答")
    assert "下一轮回答" in output.getvalue()[position:]
    assert "牛奶" not in output.getvalue()[position:]


def test_dumb_terminal_uses_immediate_plain_output(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, force_terminal=True, force_interactive=False))
    with renderer.stream_output():
        renderer.append_text("立即可见")
        assert "立即可见" in output.getvalue()
    assert output.getvalue().count("立即可见") == 1
