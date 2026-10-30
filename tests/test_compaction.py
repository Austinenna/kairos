import pytest

from kairos.compaction.budget import compact_threshold, estimate_tokens
from kairos.models import Usage
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
