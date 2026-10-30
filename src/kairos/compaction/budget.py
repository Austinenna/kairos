"""压缩阈值和上下文长度估算。

阈值照 Claude Code 的公式（autoCompact.ts），只是窗口先乘上配置的百分比：
    窗口     = 模型窗口 × KAIROS_AUTO_COMPACT_WINDOW_PERCENT%
    压缩阈值 = 窗口 − 摘要输出预留 20K − 余量 13K
    警告线   = 压缩阈值 − 20K
"""

SUMMARY_OUTPUT_RESERVE = 20_000  # 写摘要那次请求的输出预留
AUTO_COMPACT_BUFFER = 13_000  # 检查发生在请求之前，给这一轮的增长留余量
WARNING_BUFFER = 20_000  # 离阈值还有这么多时，状态栏提示即将压缩

# 每个 token 大约对应的字符数。2026-10-30 用本地会话校准：同一轮相邻两次请求的 input 差值和新增字符数相比，
# 34 组样本在 0.35～0.57 token/字符之间，中间值约 0.5。参考项目常用的 ÷4 是按英文定的，对中文会低估一半。
CHARS_PER_TOKEN = 2
IMAGE_TOKENS = 1200  # 每张图片按固定值算，照搬 pi，没有对 DeepSeek 校准


def compact_threshold(window: int | None, percent: int) -> int | None:
    """超过这个 token 数就压缩。模型窗口未知时返回 None，表示不自动压缩。"""
    if window is None:
        return None
    return window * percent // 100 - SUMMARY_OUTPUT_RESERVE - AUTO_COMPACT_BUFFER


def estimate_tokens(messages: list[dict]) -> int:
    """按字符数估算消息的 token 数：正文、思考内容、工具调用的名字和参数、工具结果都算，图片按固定值。"""
    chars = images = 0
    for message in messages:
        content = message.get("content") or ""
        if isinstance(content, str):
            chars += len(content)
        else:
            for block in content:
                if block["type"] == "text":
                    chars += len(block["text"])
                else:
                    images += 1
        chars += len(message.get("reasoning_content", ""))
        for call in message.get("tool_calls", []):
            chars += len(call["function"]["name"]) + len(call["function"]["arguments"])
    return chars // CHARS_PER_TOKEN + images * IMAGE_TOKENS
