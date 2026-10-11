from io import StringIO

from rich.console import Console

from kairos.pricing import Usage
from kairos.runtime import AgentTurnResult, ToolResult
from kairos.terminal_renderer import TerminalRenderer


def make_renderer():
    output = StringIO()
    console = Console(file=output, width=120, color_system=None)
    return TerminalRenderer(console), output


def test_renderer_formats_todo_results_as_table_and_renders_markdown():
    renderer, output = make_renderer()
    result = AgentTurnResult(
        content="# 查询结果\n\n共有 **1 条**待办。",
        messages=[],
        usage=None,
        finish_reason=None,
        events=[],
        tool_results=[
            ToolResult(
                call_id="call_1",
                name="list_todos",
                value=[{"id": 1, "title": "学习 Agent", "priority": 2, "status": "pending"}],
                content='[{"id": 1}]',
            )
        ],
    )

    renderer.render_turn(result, model="unknown-model", total_cost=0.0)

    text = output.getvalue()
    assert "待办列表" in text
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

    renderer.render_turn(result, model="unknown-model", total_cost=0.0)

    text = output.getvalue()
    assert "Mermaid" in text
    assert "flowchart TD" in text
    assert "输入 100" in text
    assert "输出 20" in text
