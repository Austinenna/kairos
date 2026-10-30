import pytest

from kairos.compaction.budget import compact_threshold, estimate_tokens
from kairos.compaction.summary import SummaryFailed, summarize
from kairos.models import ModelResponse, Usage
from kairos.storage import SessionStore, open_db


@pytest.fixture
def store(tmp_path):
    db = open_db(tmp_path / "kairos.db")
    yield SessionStore(db)
    db.close()


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


def test_rows_keep_system_prompt_and_usage_and_latest_compaction_wins(store):
    session = store.create(provider="openai", model="deepseek-flash", system_prompt="你是助手")
    for text in ("一", "二"):
        store.add_message(session.id, {"role": "user", "content": text})
        store.add_message(session.id, {"role": "assistant", "content": "好"}, Usage(100, 0, 20))
    rows = store.load_rows(session.id)
    assert [row.context_tokens for row in rows] == [None, None, 120, None, 120]

    store.add_compaction(session.id, "旧摘要", rows[1].id, rows[2].id, 1000)
    store.add_compaction(session.id, "新摘要", rows[3].id, rows[4].id, 2000)

    compaction = store.latest_compaction(session.id)
    assert (compaction.summary, compaction.first_kept_message_id) == ("新摘要", rows[3].id)
    kept = store.load_rows(session.id, from_id=compaction.first_kept_message_id)
    assert [row.message["content"] for row in kept] == ["你是助手", "二", "好"]
    assert len(store.load_messages(session.id)) == 5  # 原文一条都没少


class ReplyModel:
    model = "deepseek-flash"

    def __init__(self, response):
        self.response = response
        self.requests = []

    def stream(self, messages, tools, on_text=None):
        self.requests.append((messages, tools))
        return self.response


def test_summary_request_is_plain_text_without_tools_or_reasoning():
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}
    messages = [
        {"role": "user", "content": "读一下 a.txt"},
        {"role": "assistant", "content": "", "reasoning_content": "我在想", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "很长" * 2000},
    ]
    model = ReplyModel(ModelResponse(content="1. 主要请求和意图：读文件", finish_reason="stop"))

    assert summarize(model, messages, previous="之前聊过天气", focus="文件名") == "1. 主要请求和意图：读文件"

    request, tools = model.requests[0]
    assert tools == [] and "补充要求：文件名" in request[0]["content"]
    text = request[1]["content"]
    assert text.startswith("【之前的摘要】\n之前聊过天气") and "[用户] 读一下 a.txt" in text
    assert "[调用工具] read_file" in text and "我在想" not in text
    assert "后面 2000 字省略" in text  # 工具结果截到 2000 字符


def test_truncated_summary_is_a_failure():
    model = ReplyModel(ModelResponse(content="1. 主要请求", finish_reason="length"))
    with pytest.raises(SummaryFailed):
        summarize(model, [{"role": "user", "content": "你好"}])
