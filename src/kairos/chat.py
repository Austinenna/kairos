"""阶段 0：最小的多轮对话命令行。

这个文件要讲清楚三件事：
1. 模型 API 是无状态的。它不记得上一轮说过什么，所谓“记忆”就是我们自己维护的
   messages 列表，每一轮都把整个列表重新发过去。用 /history 可以看到它。
2. 流式输出：模型边生成边返回，我们边收边打印。用量统计在最后一个数据块里。
3. 每一轮都显示 token 和费用，建立成本直觉：历史越长，每轮的输入就越贵。
"""

from pathlib import Path

import openai
from openai import OpenAI
from pydantic import ValidationError
from rich.console import Console

from .config import Settings
from .pricing import CONTEXT_WINDOW, Usage, estimate_cost, is_peak, usage_from_openai

DEFAULT_SYSTEM_PROMPT = "你是 Kairos，用户的个人助手。回答简洁直接，使用中文。"

HELP = """\
[bold]命令[/]
  /history        查看当前的 messages 列表（下一次请求会把它整个发出去）
  /system 文本     替换 system 提示词（会清空对话，方便做对比实验）
  /think          开关思考模式
  /file 路径       把一个文本文件的内容作为你的消息发出去
  /reset          清空对话，保留 system 提示词
  /exit           退出（也可以按 Ctrl+D）
生成过程中按 Ctrl+C 可以中断这一轮。"""


def stream_reply(
    client: OpenAI, model: str, messages: list[dict], think: bool, console: Console
) -> tuple[str, Usage | None, str | None]:
    """发一次请求，流式打印回复。返回（回复全文，用量，结束原因）。"""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        # 让服务端在最后一个数据块里附上 token 用量
        stream_options={"include_usage": True},
        # DeepSeek 自己的参数，OpenAI SDK 不认识，所以放在 extra_body 里原样发过去
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

        # 思考内容是 DeepSeek 额外加的字段，SDK 的类型里没有，用 getattr 读
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


def format_stats(model: str, usage: Usage | None, cost: float | None, total_cost: float) -> str:
    if usage is None:
        return "[dim]（这一轮没有拿到用量数据）[/]"
    parts = [
        f"输入 {usage.input_tokens:,}（缓存命中 {usage.cached_input_tokens:,}）",
        f"输出 {usage.output_tokens:,}" + (f"（思考 {usage.reasoning_tokens:,}）" if usage.reasoning_tokens else ""),
    ]
    window = CONTEXT_WINDOW.get(model)
    if window:
        parts.append(f"上下文已用 {usage.input_tokens / window:.2%}")
    if cost is not None:
        parts.append(f"本轮 ${cost:.5f} · 累计 ${total_cost:.4f}" + (" · 高峰价" if is_peak() else ""))
    return "[dim]" + " · ".join(parts) + "[/]"


def run() -> None:
    console = Console()
    try:
        settings = Settings()
    except ValidationError:
        console.print("[red]配置无效。[/]把 .env.example 复制成 .env，并检查模型配置后再运行。")
        return

    if not settings.openai_api_key:
        console.print(
            "[red]没有找到 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。[/]把 .env.example 复制成 .env，填入你的 key 再运行。"
        )
        return

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    model = settings.kairos_model
    system_prompt = DEFAULT_SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    think = False
    total_cost = 0.0

    console.print(f"[bold]Kairos[/] · 模型 {model} · 输入 /help 查看命令")

    while True:
        try:
            user_input = console.input("\n[bold cyan]你 › [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user_input:
            continue

        # ---- 命令 ----
        if user_input.startswith("/"):
            cmd, _, arg = user_input.partition(" ")
            if cmd == "/exit":
                break
            elif cmd == "/help":
                console.print(HELP)
            elif cmd == "/history":
                console.print_json(data=messages)
            elif cmd == "/reset":
                messages = [{"role": "system", "content": system_prompt}]
                console.print("[dim]对话已清空。[/]")
            elif cmd == "/system":
                if not arg:
                    console.print(f"[dim]当前 system 提示词：{system_prompt}[/]")
                    continue
                system_prompt = arg
                messages = [{"role": "system", "content": system_prompt}]
                console.print("[dim]system 提示词已替换，对话已清空。[/]")
            elif cmd == "/think":
                think = not think
                console.print(f"[dim]思考模式：{'开' if think else '关'}[/]")
            elif cmd == "/file":
                path = Path(arg).expanduser()
                try:
                    user_input = path.read_text(encoding="utf-8")
                except OSError as e:
                    console.print(f"[red]读不了这个文件：{e}[/]")
                    continue
                console.print(f"[dim]已读取 {path.name}，{len(user_input):,} 个字符。[/]")
            else:
                console.print(f"[red]不认识的命令 {cmd}[/]，输入 /help 查看。")
            if cmd != "/file":
                continue

        # ---- 一轮对话 ----
        messages.append({"role": "user", "content": user_input})
        console.print("[bold green]Kairos › [/]", end="")
        try:
            reply, usage, finish_reason = stream_reply(client, model, messages, think, console)
        except KeyboardInterrupt:
            # 中断的这一轮不算数：撤回刚加的用户消息，让历史保持干净
            messages.pop()
            console.print("\n[yellow]已中断，这一轮不计入对话历史。[/]")
            continue
        except openai.AuthenticationError:
            messages.pop()
            console.print("\n[red]API key 无效。[/]检查 .env 里的 DEEPSEEK_API_KEY。")
            continue
        except openai.APIStatusError as e:
            messages.pop()
            console.print(f"\n[red]请求失败（HTTP {e.status_code}）：[/]{e.message}")
            continue
        except openai.APIConnectionError:
            messages.pop()
            console.print("\n[red]连不上服务器。[/]检查网络后重试。")
            continue

        # 只把最终回答放回历史。没有带 tools 参数时，DeepSeek 不需要也不会使用之前的思考内容。
        # 注意：阶段 1 加上 tools 以后规则会变，到时候必须把 reasoning_content 也带回去，否则报 400。
        messages.append({"role": "assistant", "content": reply})

        if finish_reason == "length":
            console.print("[yellow]回复被截断了：输出达到了长度上限。[/]")

        cost = estimate_cost(model, usage) if usage else None
        total_cost += cost or 0.0
        console.print(format_stats(model, usage, cost, total_cost))
