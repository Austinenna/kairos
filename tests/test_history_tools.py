from kairos.storage import SessionStore, open_db
from kairos.tools.history import history_tools


def test_search_then_read_session(tmp_path):
    db = open_db(tmp_path / "kairos.db")
    store = SessionStore(db)
    session = store.create(provider="openai", model="m", system_prompt="规则")
    store.add_message(session.id, {"role": "user", "content": "为什么主题要平铺？"})
    store.add_message(session.id, {"role": "assistant", "content": "因为平铺更容易按标题查找。"})
    tools = history_tools(db).handlers

    hits = tools["search_history"]("平铺 标题")
    page = tools["read_session"](hits[0]["session_id"])

    assert [hit["role"] for hit in hits] == ["assistant"]
    assert [m["content"] for m in page["messages"]] == ["为什么主题要平铺？", "因为平铺更容易按标题查找。"]
    assert page["next_offset"] is None
