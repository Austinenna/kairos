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


def test_list_sessions_by_date_shows_channel(tmp_path):
    db = open_db(tmp_path / "kairos.db")
    store = SessionStore(db)
    old = store.create(provider="openai", model="m", system_prompt="规则")
    store.add_message(old.id, {"role": "user", "content": "昨天的话"})
    db.execute("UPDATE messages SET created_at = '2026-10-22T09:00:00+08:00' WHERE session_id = ?", (old.id,))
    today = store.create(provider="openai", model="m", system_prompt="规则", channel="weixin")
    store.add_message(today.id, {"role": "user", "content": "今天的话"})
    store.create(provider="openai", model="m", system_prompt="规则")  # 没说过话的空会话不列出
    list_sessions = history_tools(db).handlers["list_sessions"]

    listed = list_sessions(date=today.created_at[:10])

    assert [(s["id"], s["channel"], s["messages"]) for s in listed] == [(today.id, "weixin", 1)]
    assert {s["id"] for s in list_sessions()} == {old.id, today.id}
