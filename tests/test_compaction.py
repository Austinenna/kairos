from kairos.compaction.budget import compact_threshold, estimate_tokens


def test_threshold_follows_claude_code_formula_on_a_share_of_the_window():
    assert compact_threshold(1_048_576, 50) == 524_288 - 20_000 - 13_000
    assert compact_threshold(None, 50) is None  # 窗口未知：不自动压缩


def test_estimate_counts_text_tool_calls_and_images():
    call = {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"p": 1}'}}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "看图" * 50}, {"type": "image_ref", "image_id": 1}]},
        {"role": "assistant", "content": "", "reasoning_content": "想" * 20, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "结果" * 100},
    ]
    chars = 100 + 20 + len("read") + len('{"p": 1}') + 200
    assert estimate_tokens(messages) == chars // 2 + 1200
