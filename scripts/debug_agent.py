"""阶段 1 的一次性真实模型调试脚本。

示例：

    uv run python scripts/debug_agent.py "把学习 Agent 加入待办"

正式的 Agent 编排位于 ``kairos.agent_app``；本文件只负责命令行输入输出，和正式源码分开。
"""

import argparse

from kairos.agent_app import run_cli


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="用真实 DeepSeek 模型调试 Kairos 待办工具循环")
    parser.add_argument("prompt", nargs="*", help="本轮要发送给 Kairos 的问题；省略时从 stdin 读取")
    args = parser.parse_args(argv)
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = input("你 › ").strip()
    if not prompt:
        parser.error("请输入要发送的问题")

    print(run_cli(prompt))


if __name__ == "__main__":
    main()
