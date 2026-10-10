"""旧版 OpenAI 流式响应辅助函数。

Agent 主流程使用统一运行时的完整回合接口；本模块保留阶段 0 的流式能力，供
实验或兼容调用方使用，不参与主程序的 Agent 编排。
"""

from openai import OpenAI
from rich.console import Console

from .pricing import Usage, usage_from_openai


def stream_reply(
    client: OpenAI, model: str, messages: list[dict], think: bool, console: Console
) -> tuple[str, Usage | None, str | None]:
    """发一次 OpenAI 兼容请求，流式打印回复。"""

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"thinking": {"type": "enabled" if think else "disabled"}},
    )

    reply_parts: list[str] = []
    usage = None
    finish_reason = None
    printed_reasoning = False

    for chunk in stream:
        if chunk.usage:
            usage = usage_from_openai(chunk.usage)
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not printed_reasoning:
                console.print("[dim]（思考中）[/]")
                printed_reasoning = True
            console.print(reasoning, end="", style="dim", markup=False, highlight=False, soft_wrap=True)

        if delta.content:
            if printed_reasoning and not reply_parts:
                console.print("\n")
            reply_parts.append(delta.content)
            console.print(delta.content, end="", markup=False, highlight=False, soft_wrap=True)

    console.print()
    return "".join(reply_parts), usage, finish_reason
