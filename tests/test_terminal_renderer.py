from io import StringIO

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

    renderer.render_turn(result, model="unknown-model", total_cost=0.0)

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

    renderer.render_turn(result, model="unknown-model", total_cost=0.0)

    text = output.getvalue()
    assert "Mermaid" in text
    assert "flowchart TD" in text
    assert "输入 100" in text
    assert "输出 20" in text
