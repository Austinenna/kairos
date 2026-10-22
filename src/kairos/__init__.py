"""Kairos：个人工作与生活助手。"""


def main() -> None:
    """`uv run kairos` 的入口，启动终端界面。旧版入口是 `uv run kairos-legacy`。"""
    from .terminal.app import main as terminal_main  # 用到时才导入，导入 kairos 包本身保持轻量

    terminal_main()
