"""使用临时会话库验证历史搜索、原文定位、续读和只读边界。"""

import json
import sqlite3

import pytest

from kairos.history import HistoryReader
from kairos.sessions import SessionStore


@pytest.fixture
def connection(tmp_path):
    """创建测试专用数据库，并在用例结束时关闭连接。"""
    connection = sqlite3.connect(tmp_path / "history.sqlite3")
    yield connection
    connection.close()


@pytest.fixture
def store(connection):
    """装配只使用临时数据库的真实会话存储。"""
    return SessionStore(connection)


@pytest.fixture
def reader(store):
    """创建供搜索和阅读行为测试使用的查询服务。"""
    return HistoryReader(store)


def new_session(store, title="历史测试"):
    """建立带显式标题的隔离会话。

    作用:
        让测试通过真实保存接口生成来源 ID，不访问用户的正式数据库。

    参数:
        store: 当前用例的临时会话存储。
        title: 会话标题，默认使用不含业务关键词的固定标题。

    返回:
        已创建的 SessionInfo。

    行为说明:
        创建一个含系统消息的会话，后续回合由 record_turn 写入。
    """
    return store.create_session("系统提示仅作内部规则", provider="openai", model="fake", title=title)


def record_turn(store, session_id, user_content, answer="已完成", *, tools=(), status="completed", answer_extra=None):
    """通过正常保存路径建立含可选工具执行的一个回合。

    作用:
        为查询测试提供真实关联的用户消息、助手消息、工具调用和结果。

    参数:
        store: 当前用例的临时会话存储。
        session_id: 已存在的会话 ID。
        user_content: 用户正文，可以是字符串或标准内容块列表。
        answer: 最后助手消息的正文，None 表示不保存助手回答。
        tools: 按顺序排列的 (工具名, 参数对象, 结果正文) 三元组。
        status: 回合结束状态，running 表示保持未结束。
        answer_extra: 附加的协议字段，用来验证其不参与历史查询。

    返回:
        创建回合时返回的 TurnRecord，包含真实回合 ID 和用户消息 ID。

    行为说明:
        工具调用及结果保存到第一步；最终回答另存一步。
        未结束的回合仍保留全部已保存消息，以测试查询对状态的筛选。
    """
    turn = store.begin_turn(session_id, {"role": "user", "content": user_content})
    step_index = 1
    if tools:
        step = store.start_step(turn.turn_id, step_index, "fake")
        calls = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
            for index, (name, arguments, _) in enumerate(tools)
        ]
        store.append_assistant(step.step_id, {"role": "assistant", "content": "正在查证", "tool_calls": calls})
        for index, (name, _, result) in enumerate(tools):
            store.start_tool(step.step_id, f"call_{index}")
            store.append_tool_result(
                step.step_id,
                {"role": "tool", "tool_call_id": f"call_{index}", "name": name, "content": result},
            )
        store.finish_step(step.step_id, "completed")
        step_index += 1
    if answer is not None:
        step = store.start_step(turn.turn_id, step_index, "fake")
        store.append_assistant(step.step_id, {"role": "assistant", "content": answer, **(answer_extra or {})})
        store.finish_step(step.step_id, "completed")
    if status != "running":
        store.finish_turn(turn.turn_id, status, "stop" if status == "completed" else status)
    return turn


def set_turn_time(connection, turn_id, timestamp):
    """把隔离回合的开始时间设置为确定值，验证带时区的筛选边界。"""
    with connection:
        connection.execute("UPDATE session_turns SET created_at = ? WHERE turn_id = ?", (timestamp, turn_id))


def hit_ids(result):
    """从工具公开结果中取出命中回合 ID，简化行为断言。"""
    return [hit["turn_id"] for hit in result["hits"]]


def test_search_requires_all_literal_chinese_terms_in_one_turn(store, reader):
    """搜索可跨同一回合的正文匹配，但不能拼接不同回合凑齐关键词。"""
    session = new_session(store)
    wanted = record_turn(store, session.session_id, "长期记忆放在哪里？", "主题采用平铺组织")
    record_turn(store, session.session_id, "长期记忆是什么？", "说明概念")
    record_turn(store, session.session_id, "文件组织方式", "主题采用平铺组织")

    result = reader.search_history("记忆 平铺")

    assert result["source_type"] == "historical_reference"
    assert hit_ids(result) == [wanted.turn_id]
    hit = result["hits"][0]
    assert hit["session_id"] == session.session_id
    assert hit["session_title"] == session.title
    assert hit["status"] == "completed"
    messages = {message.message_id: message for message in store.load_messages(session.session_id)}
    assert hit["message_id"] in messages
    assert {match["role"] for match in hit["matches"]} == {"user", "assistant"}
    assert all(match["message_id"] in messages and match["snippet"] for match in hit["matches"])


def test_search_treats_sql_wildcards_as_literal_text(store, reader):
    """百分号与下划线是正文字符，不会变成数据库通配符。"""
    session = new_session(store)
    literal = record_turn(store, session.session_id, "格式是 100%_done")
    record_turn(store, session.session_id, "格式是 100xyzdone")
    assert hit_ids(reader.search_history("%_")) == [literal.turn_id]


def test_search_snippet_preserves_match_after_casefold_expansion(store, reader):
    """大小写折叠改变字符长度时，摘录仍围绕原文中的真实命中。

    作用:
        防止搜索匹配成功却返回不含命中词的片段，保留定位结果的可用性。

    参数:
        store: 当前测试的临时会话库。
        reader: 绑定该会话库的历史查询服务。

    返回:
        无返回值(None)。

    行为说明:
        用折叠后长度翻倍的字符前缀制造位置偏移，只检查公开摘录结果。
    """
    session = new_session(store)
    record_turn(store, session.session_id, "ß" * 300 + "唯一命中词")
    matches = reader.search_history("唯一命中词")["hits"][0]["matches"]
    assert any("唯一命中词" in match["snippet"] for match in matches)
    assert all(len(match["snippet"]) <= 240 for match in matches)


def test_search_combines_session_and_absolute_time_filters(store, reader, connection):
    """时间按实际时刻比较，并与会话范围共同约束命中。"""
    first = new_session(store)
    second = new_session(store)
    before = record_turn(store, first.session_id, "检索时间")
    inside = record_turn(store, first.session_id, "检索时间")
    upper = record_turn(store, first.session_id, "检索时间")
    another = record_turn(store, second.session_id, "检索时间")
    for turn, timestamp in (
        (before, "2026-10-18T15:59:59+00:00"),
        (inside, "2026-10-18T16:00:00+00:00"),
        (upper, "2026-10-19T16:00:00+00:00"),
        (another, "2026-10-19T08:00:00+00:00"),
    ):
        set_turn_time(connection, turn.turn_id, timestamp)

    filters = {"since": "2026-10-19T00:00:00+08:00", "until": "2026-10-20T00:00:00+08:00"}
    assert set(hit_ids(reader.search_history("检索", **filters))) == {inside.turn_id, another.turn_id}
    assert hit_ids(reader.search_history("检索", session_id=first.session_id, **filters)) == [inside.turn_id]


def test_search_time_bounds_preserve_microsecond_precision(store, reader, connection):
    """时间筛选保持微秒精度，下界包含、上界排除。

    作用:
        验证数据库时间转换不会把相邻时刻舍入成同一个边界。

    参数:
        store、reader: 当前用例的临时存储和查询服务。
        connection: 同一临时库连接，用于设置确定的回合时间。

    返回:
        无返回值(None)。

    行为说明:
        建立相差一微秒的三个回合，区间仅应包含中间一个。
    """
    session = new_session(store)
    turns = [record_turn(store, session.session_id, "微秒边界") for _ in range(3)]
    for index, turn in enumerate(turns, start=1):
        set_turn_time(connection, turn.turn_id, f"2026-10-19T00:00:00.00000{index}+00:00")
    result = reader.search_history("微秒边界", since="2026-10-19T00:00:00.000002Z", until="2026-10-19T00:00:00.000003Z")
    assert hit_ids(result) == [turns[1].turn_id]


def test_equal_relevance_is_sorted_by_actual_time_across_offsets(store, reader, connection):
    """相同相关性的结果按真实时刻排序，时区字符串不影响先后。

    作用:
        防止不同 UTC 偏移下的本地钟面时间被误当作时间顺序。

    参数:
        store、reader: 当前用例的临时存储和查询服务。
        connection: 同一临时库连接，用于设置带不同偏移的时间。

    返回:
        无返回值(None)。

    行为说明:
        比较东八区上午十点与 UTC 凌晨三点，后者实际晚一小时。
    """
    session = new_session(store)
    older = record_turn(store, session.session_id, "相同相关性")
    newer = record_turn(store, session.session_id, "相同相关性")
    set_turn_time(connection, older.turn_id, "2026-10-19T10:00:00+08:00")
    set_turn_time(connection, newer.turn_id, "2026-10-19T03:00:00+00:00")
    assert hit_ids(reader.search_history("相同相关性")) == [newer.turn_id, older.turn_id]


def test_empty_query_requires_a_scope_and_supports_browsing(store, reader):
    """无关键词时必须提供范围，按会话或时间浏览仍可找到记录。"""
    session = new_session(store)
    turn = record_turn(store, session.session_id, "最近一次决定")
    for query in (None, "", "  \n"):
        with pytest.raises(ValueError):
            reader.search_history(query)
    assert hit_ids(reader.search_history(session_id=session.session_id)) == [turn.turn_id]
    assert hit_ids(reader.search_history(since="2000-01-01T00:00:00Z")) == [turn.turn_id]


def test_search_top_results_are_bounded_stable_and_report_more(store, reader, connection):
    """相同条件的排序稳定，并准确区分有剩余命中与结果已完整。"""
    session = new_session(store)
    for index in range(7):
        turn = record_turn(store, session.session_id, f"稳定排序资料 {index}")
        set_turn_time(connection, turn.turn_id, "2026-10-19T00:00:00+00:00")
    first = reader.search_history("稳定排序", limit=3)
    second = reader.search_history("稳定排序", limit=3)
    full = reader.search_history("稳定排序", limit=10)
    assert hit_ids(first) == hit_ids(second) == hit_ids(full)[:3]
    assert first["has_more"] is True
    assert full["has_more"] is False
    assert len(set(hit_ids(full))) == 7


def test_search_output_budget_includes_json_escaping(store, reader):
    """多个命中的短摘录经 JSON 转义后，整体结果仍受字符预算约束。"""
    session = new_session(store)
    text = "\x01" * 100 + "输出预算" + "\x01" * 300
    for _ in range(10):
        record_turn(store, session.session_id, text, text, tools=[("check", {}, text)])
    result = reader.search_history("输出预算", include_tools=True, limit=10)
    assert 0 < len(result["hits"]) < 10
    assert result["has_more"] is True
    assert len(json.dumps(result, ensure_ascii=False)) <= 16000


def test_tool_arguments_and_results_are_opt_in_but_history_echoes_are_not_searchable(store, reader):
    """工具参数和结果需显式开启；历史工具回显不能成为新的搜索命中。"""
    session = new_session(store)
    ordinary = record_turn(
        store,
        session.session_id,
        "检查项目状态",
        tools=[("inspect_project", {"path": "参数独有词"}, "结果独有词")],
    )
    history = record_turn(
        store,
        session.session_id,
        "查旧资料",
        tools=[
            ("search_history", {"query": "历史参数独有词"}, '{"snippet":"检索回显独有词"}'),
            ("read_session", {"session_id": "其他会话"}, '{"content":"读取回显独有词"}'),
        ],
    )
    for query in ("参数独有词", "结果独有词"):
        assert reader.search_history(query)["hits"] == []
        assert hit_ids(reader.search_history(query, include_tools=True)) == [ordinary.turn_id]
    for query in ("历史参数独有词", "检索回显独有词", "读取回显独有词"):
        assert reader.search_history(query, include_tools=True)["hits"] == []
    ordinary_read = reader.read_session(session.session_id, turn_id=ordinary.turn_id)
    expanded = reader.read_session(session.session_id, turn_id=ordinary.turn_id, include_tools=True)
    assert all(message["role"] != "tool" for message in ordinary_read["messages"])
    assert "参数独有词" not in json.dumps(ordinary_read, ensure_ascii=False)
    expanded_text = json.dumps(expanded, ensure_ascii=False)
    assert "参数独有词" in expanded_text and "结果独有词" in expanded_text
    historical_read = reader.read_session(session.session_id, turn_id=history.turn_id, include_tools=True)
    historical_text = json.dumps(historical_read, ensure_ascii=False)
    assert "检索回显独有词" not in historical_text and "读取回显独有词" not in historical_text


def test_running_and_inactive_records_are_excluded_but_failure_status_is_preserved(store, reader):
    """未结束回合和撤回输入不进入回忆，失败与取消记录保留真实状态。"""
    session = new_session(store)
    other = new_session(store)
    record_turn(store, other.session_id, "共同关键词 别处运行中", status="running")
    withdrawn = store.begin_turn(session.session_id, {"role": "user", "content": "共同关键词 已撤回"})
    store.finish_turn(withdrawn.turn_id, "failed", "model_error", retain_input=False)
    failed = record_turn(store, session.session_id, "共同关键词 失败", status="failed")
    cancelled = record_turn(store, session.session_id, "共同关键词 取消", status="cancelled")
    running = record_turn(store, session.session_id, "共同关键词 未完成", status="running")

    result = reader.search_history("共同关键词")
    assert {hit["turn_id"]: hit["status"] for hit in result["hits"]} == {
        failed.turn_id: "failed",
        cancelled.turn_id: "cancelled",
    }
    for turn in (running, withdrawn):
        with pytest.raises(ValueError):
            reader.read_session(session.session_id, turn_id=turn.turn_id)
    for turn, expected in ((failed, "failed"), (cancelled, "cancelled")):
        page = reader.read_session(session.session_id, turn_id=turn.turn_id)
        assert page["selected_turns"][0]["status"] == expected


def test_images_reasoning_and_protocol_data_never_enter_history_results(store, reader):
    """历史输出仅保留可读资料，不回传图片编码、内部思考和协议扩展。"""
    encoded = "iVBORw0KGgo" + "A" * 180
    data_url = "data:image/png;base64," + encoded
    session = new_session(store)
    turn = record_turn(
        store,
        session.session_id,
        [
            {"type": "text", "text": "请分析这张布局图"},
            {"type": "image", "media_type": "image/png", "data": encoded},
        ],
        "布局采用单列。图片原文是 " + data_url,
        tools=[("analyze", {"image_url": data_url}, json.dumps({"image": data_url}))],
        answer_extra={
            "reasoning_content": "隐藏思考独有词",
            "provider_content": [{"type": "thinking", "thinking": "供应商字段独有词"}],
            "provider_id": "内部协议独有词",
        },
    )
    for query in (encoded, "隐藏思考独有词", "供应商字段独有词", "内部协议独有词"):
        assert reader.search_history(query, include_tools=True)["hits"] == []
    search = reader.search_history("布局", include_tools=True)
    page = reader.read_session(session.session_id, turn_id=turn.turn_id, include_tools=True)
    serialized = json.dumps([search, page], ensure_ascii=False)
    assert "布局采用单列" in serialized
    assert encoded not in serialized and "data:image/png;base64," not in serialized
    assert "隐藏思考独有词" not in serialized and "供应商字段独有词" not in serialized
    original = json.dumps([message.payload for message in store.load_messages(session.session_id)])
    assert encoded in original


def test_inactive_message_is_excluded_from_an_otherwise_readable_turn(store, reader, connection):
    """回合仍可读取时，其中单独撤出的消息也不能参加搜索或阅读。"""
    session = new_session(store)
    turn = record_turn(store, session.session_id, "有效问题", "已撤出正文独有词")
    answer = next(
        message
        for message in store.load_messages(session.session_id)
        if message.turn_id == turn.turn_id and message.role == "assistant"
    )
    with connection:
        connection.execute("UPDATE session_messages SET in_context = 0 WHERE message_id = ?", (answer.message_id,))
    assert reader.search_history("已撤出正文独有词")["hits"] == []
    page = reader.read_session(session.session_id, turn_id=turn.turn_id)
    assert [message["content"] for message in page["messages"]] == ["有效问题"]
    with pytest.raises(ValueError):
        reader.read_session(session.session_id, message_id=answer.message_id)


def test_message_anchor_expands_to_whole_turn_and_requested_neighbors(store, reader):
    """消息锚点定位后按回合扩展，返回准确的前后边界。"""
    session = new_session(store)
    turns = [record_turn(store, session.session_id, f"问题 {index}", f"回答 {index}") for index in range(6)]
    messages = store.load_messages(session.session_id)
    anchor = next(
        message for message in messages if message.turn_id == turns[2].turn_id and message.role == "assistant"
    )
    page = reader.read_session(session.session_id, message_id=anchor.message_id, before_turns=1, after_turns=2)
    assert page["source_type"] == "historical_reference"
    assert page["anchor"] == {"message_id": anchor.message_id, "turn_id": turns[2].turn_id}
    assert [turn["turn_id"] for turn in page["selected_turns"]] == [turn.turn_id for turn in turns[1:5]]
    assert page["previous_turn_id"] == turns[0].turn_id
    assert page["next_turn_id"] == turns[5].turn_id
    assert [message["content"] for message in page["messages"]] == [
        text for index in range(1, 5) for text in (f"问题 {index}", f"回答 {index}")
    ]
    assert page["has_more"] is False and page["next_cursor"] is None
    assert all(
        message["start_char"] == 0 and message["end_char"] == message["total_chars"] for message in page["messages"]
    )


def test_foreign_or_inconsistent_anchors_are_rejected(store, reader):
    """会话、回合、消息必须属于同一来源，系统消息也不能作为正文入口。"""
    first = new_session(store)
    second = new_session(store)
    left = record_turn(store, first.session_id, "第一回合")
    right = record_turn(store, first.session_id, "第二回合")
    foreign = record_turn(store, second.session_id, "其他会话")
    system_id = store.load_messages(first.session_id)[0].message_id
    for parameters in (
        {"message_id": foreign.user_message_id},
        {"turn_id": foreign.turn_id},
        {"message_id": left.user_message_id, "turn_id": right.turn_id},
        {"message_id": system_id},
        {"message_id": "missing-message"},
        {"turn_id": "missing-turn"},
        {},
    ):
        with pytest.raises(ValueError):
            reader.read_session(first.session_id, **parameters)


@pytest.mark.parametrize("limit", [True, False, 0, -1, 11, 1.5, "5"])
def test_search_rejects_invalid_result_limits(reader, limit):
    """结果数量必须是规定范围内的整数，布尔值不能冒充数量。"""
    with pytest.raises(ValueError):
        reader.search_history("关键词", limit=limit)


@pytest.mark.parametrize(
    "parameters",
    [
        {"before_turns": True},
        {"before_turns": -1},
        {"before_turns": 3},
        {"after_turns": 1.5},
        {"after_turns": "1"},
        {"include_tools": 1},
    ],
)
def test_read_rejects_invalid_windows_and_switches(store, reader, parameters):
    """阅读窗口和工具开关保持明确类型，拒绝越界与隐式转换。"""
    session = new_session(store)
    turn = record_turn(store, session.session_id, "测试输入")
    with pytest.raises(ValueError):
        reader.read_session(session.session_id, turn_id=turn.turn_id, **parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"include_tools": 1},
        {"since": "2026-10-19T00:00:00"},
        {"since": "不合法日期"},
        {"since": "2026-10-20T00:00:00Z", "until": "2026-10-19T00:00:00Z"},
    ],
)
def test_search_rejects_ambiguous_filters(reader, parameters):
    """搜索拒绝无时区时间、倒置区间和非布尔工具开关。"""
    with pytest.raises(ValueError):
        reader.search_history("关键词", **parameters)


def test_pages_reconstruct_long_messages_without_gaps_or_duplicates(store):
    """连续续读完整还原长消息和后续消息，字符坐标连续且每页有界。"""
    session = new_session(store)
    first = record_turn(store, session.session_id, "长问题" * 90, "长回答" * 120)
    second = record_turn(store, session.session_id, "后续追问" * 30, "后续结论" * 40)
    expected = {
        message.message_id: message.payload["content"]
        for message in store.load_messages(session.session_id)
        if message.turn_id in (first.turn_id, second.turn_id)
    }
    reader = HistoryReader(store, page_chars=80)
    page = reader.read_session(session.session_id, turn_id=first.turn_id, after_turns=1)
    reconstructed = {message_id: "" for message_id in expected}
    seen_cursors = set()
    count = 0
    while True:
        count += 1
        assert count < 100
        assert 0 < sum(len(message["content"]) for message in page["messages"]) <= 80
        assert [turn["turn_id"] for turn in page["selected_turns"]] == [first.turn_id, second.turn_id]
        for message in page["messages"]:
            message_id = message["message_id"]
            assert message["start_char"] == len(reconstructed[message_id])
            reconstructed[message_id] += message["content"]
            assert message["end_char"] == len(reconstructed[message_id])
            assert message["total_chars"] == len(expected[message_id])
        if not page["has_more"]:
            assert page["next_cursor"] is None
            break
        cursor = page["next_cursor"]
        assert cursor and cursor not in seen_cursors
        seen_cursors.add(cursor)
        page = reader.read_session(session.session_id, cursor=cursor)
    assert count > 2
    assert reconstructed == expected


def test_turn_without_visible_text_returns_a_complete_empty_page(store, reader):
    """没有可见正文的已结束回合能正常读取空结果，不生成无效续读。"""
    session = new_session(store)
    turn = record_turn(store, session.session_id, [], answer=None)
    assert reader.search_history(session_id=session.session_id)["hits"] == []
    page = reader.read_session(session.session_id, turn_id=turn.turn_id)
    assert page["messages"] == []
    assert page["has_more"] is False and page["next_cursor"] is None
    assert page["selected_turns"][0]["turn_id"] == turn.turn_id


def test_many_tool_messages_are_paginated_with_a_message_count_limit(store, reader):
    """短工具消息也受单页数量限制，续读仍完整覆盖选定回合。"""
    session = new_session(store)
    turn = record_turn(
        store,
        session.session_id,
        "批量核对",
        tools=[("check", {"index": index}, f"结果 {index}") for index in range(25)],
    )
    expected_ids = [
        message.message_id for message in store.load_messages(session.session_id) if message.turn_id == turn.turn_id
    ]
    page = reader.read_session(session.session_id, turn_id=turn.turn_id, include_tools=True)
    assert page["has_more"] is True
    received_ids = []
    for _ in range(10):
        assert 0 < len(page["messages"]) <= 20
        received_ids.extend(message["message_id"] for message in page["messages"])
        if not page["has_more"]:
            break
        page = reader.read_session(session.session_id, cursor=page["next_cursor"])
    assert received_ids == expected_ids
    assert page["has_more"] is False


def test_serialized_page_budget_accounts_for_escaped_characters(store, reader):
    """JSON 转义会放大正文，单页仍有大小上限且续读不丢失字符。"""
    session = new_session(store)
    original = "\x01" * 11000
    turn = record_turn(store, session.session_id, original, answer=None)
    page = reader.read_session(session.session_id, turn_id=turn.turn_id)
    assert page["has_more"] is True
    reconstructed = ""
    for _ in range(20):
        assert len(json.dumps(page, ensure_ascii=False)) <= 28000
        reconstructed += "".join(message["content"] for message in page["messages"])
        if not page["has_more"]:
            break
        page = reader.read_session(session.session_id, cursor=page["next_cursor"])
    assert reconstructed == original
    assert page["has_more"] is False


def test_cursor_is_bound_to_reader_session_range_and_unchanged_source(store, connection):
    """游标不能篡改、换会话、换服务或在原文改变后继续使用。"""
    session = new_session(store)
    other = new_session(store)
    turn = record_turn(store, session.session_id, "需要分页的原始内容" * 50)
    reader = HistoryReader(store, page_chars=80)
    page = reader.read_session(session.session_id, turn_id=turn.turn_id)
    cursor = page["next_cursor"]
    assert cursor
    tampered = ("A" if cursor[0] != "A" else "B") + cursor[1:]
    for service, session_id, parameters in (
        (reader, session.session_id, {"cursor": tampered}),
        (reader, other.session_id, {"cursor": cursor}),
        (HistoryReader(store, page_chars=80), session.session_id, {"cursor": cursor}),
        (reader, session.session_id, {"cursor": cursor, "before_turns": 1}),
        (reader, session.session_id, {"cursor": cursor, "include_tools": True}),
    ):
        with pytest.raises(ValueError):
            service.read_session(session_id, **parameters)
    with connection:
        connection.execute(
            "UPDATE session_messages SET payload = ? WHERE message_id = ?",
            (json.dumps({"role": "user", "content": "内容已变更"}), turn.user_message_id),
        )
    with pytest.raises(ValueError):
        reader.read_session(session.session_id, cursor=cursor)


def test_history_operations_do_not_mutate_records_or_session_activity(store, reader, connection):
    """查询不改写聊天内容、活动时间或当前运行回合，也不留下事务。"""
    old = new_session(store)
    saved = record_turn(store, old.session_id, "只读核验关键词")
    current = new_session(store)
    record_turn(store, current.session_id, "当前正在处理", status="running")
    before = "\n".join(connection.iterdump())
    latest = store.latest_session()

    reader.search_history("只读核验")
    reader.read_session(old.session_id, turn_id=saved.turn_id)
    with pytest.raises(ValueError):
        reader.read_session(current.session_id, turn_id=saved.turn_id)

    assert "\n".join(connection.iterdump()) == before
    assert store.latest_session() == latest
    assert connection.in_transaction is False
