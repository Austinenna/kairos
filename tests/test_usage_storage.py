"""用临时 SQLite 验证费用快照、旧记录迁移和步骤统计边界。"""

import json
import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest

from kairos import execution, pricing
from kairos.agent_loop import run_agent_turn
from kairos.execution import TurnRecorder
from kairos.pricing import Usage, estimate_cost_snapshot, usage_from_normalized
from kairos.sessions import SESSION_SCHEMA, SessionStore

MODEL = "usage-storage-test"
RAW_USAGE = {"prompt_tokens": 10000, "cached_input_tokens": 4000, "completion_tokens": 100, "reasoning_tokens": 20}
PEAK_START = datetime(2026, 10, 19, 1, 59, tzinfo=UTC)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """建立隔离会话存储和固定测试价格。

    作用:
        为每项测试提供临时数据库，防止访问个人数据或依赖真实模型价格。

    参数:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest 自动恢复全局价格表的工具。

    返回:
        会话存储生成器；测试结束后关闭数据库连接。

    行为说明:
        1. 注入测试模型价格并打开临时 SQLite。
        2. 测试后仅关闭本项测试创建的连接。
    """
    monkeypatch.setitem(pricing.PRICES, MODEL, {"peak": (2, 4, 6), "off_peak": (1, 2, 3)})
    connection = sqlite3.connect(tmp_path / "usage.sqlite3")
    try:
        yield SessionStore(connection)
    finally:
        connection.close()


def begin_recording(store):
    """创建供步骤记录测试使用的会话、回合和记录器。

    作用:
        复用合法的对象关联，让测试聚焦统计的保存与恢复。

    参数:
        store: 隔离的 SessionStore。

    返回:
        会话、回合、记录器的元组(tuple)。

    行为说明:
        1. 创建测试会话和用户消息，但不提前创建模型步骤。
    """
    session = store.create_session("测试系统提示", provider="openai", model=MODEL)
    turn = store.begin_turn(session.session_id, {"role": "user", "content": "测试问题"})
    return session, turn, TurnRecorder(store, session.session_id, turn.turn_id, MODEL)


@pytest.mark.parametrize("raw", [None, {}, [], "not-usage"])
def test_absent_normalized_usage_stays_unknown(raw):
    """验证没有有效用量对象时不构造零用量。

    参数:
        raw: 缺失或不符合字典约定的输入。

    返回:
        None，由断言确认结果。

    行为说明:
        1. 对缺失对象保持未知，不参与费用计算。
    """
    assert usage_from_normalized(raw) is None


def test_normalized_usage_keeps_valid_zero_and_rejects_invalid_counts():
    """验证各用量字段独立校验且显式零得到保留。

    返回:
        None，由断言确认解析结果。

    行为说明:
        1. 布尔值、负数和字符串保持未知，零仍是有效输出量。
    """
    assert usage_from_normalized(
        {"prompt_tokens": True, "cached_input_tokens": -1, "completion_tokens": 0, "reasoning_tokens": "20"}
    ) == Usage(None, None, 0, None)
    assert usage_from_normalized(RAW_USAGE) == Usage(10000, 4000, 100, 20)


def test_snapshot_uses_utc_request_time_and_copies_rates(store):
    """验证费用快照按请求时间取价并保留 UTC 时点。

    参数:
        store: 提供固定价格的隔离存储夹具。

    返回:
        None，由断言确认费用和时间。

    行为说明:
        1. 用北京时间表示同一高峰时点，确认估算值与 UTC 一致。
        2. 未知模型仍保留来源、时段和时间，费用与费率未知。
    """
    local_time = PEAK_START.astimezone(timezone(timedelta(hours=8)))
    snapshot = estimate_cost_snapshot(MODEL, RAW_USAGE, local_time)
    assert snapshot == {
        "cost_usd": pytest.approx(0.0326),
        "rates": [2, 4, 6],
        "period": "peak",
        "priced_at": PEAK_START.isoformat(),
        "source": "response",
    }
    missing = estimate_cost_snapshot("unknown-model", RAW_USAGE, local_time)
    assert missing["cost_usd"] is None and missing["rates"] is None
    assert missing["priced_at"] == PEAK_START.isoformat()


def test_failed_step_and_reopen_preserve_original_snapshot(store, monkeypatch):
    """验证收尾和重新打开数据库不会重新计算已经保存的费用。

    作用:
        覆盖模型返回后工具执行失败，以及之后价格变化和会话恢复的路径。

    参数:
        store: 隔离存储。
        monkeypatch: 固定请求时钟并修改后续价格。

    返回:
        None，由断言确认费用与用量均已稳定保存。

    行为说明:
        1. 固定请求起点，保存响应后再切换费率并让步骤失败。
        2. 再次初始化存储和恢复会话，确认快照不变。
    """

    class Clock:
        @staticmethod
        def now(tz):
            """返回测试固定的 UTC 请求起点。"""
            return PEAK_START

    monkeypatch.setattr(execution, "datetime", Clock)
    session, turn, recorder = begin_recording(store)
    recorder.start_step(1)
    recorder.record_response({"usage": RAW_USAGE, "finish_reason": "tool_calls"}, 2.0)
    before = store.usage_steps(session.session_id)[0]
    assert before["status"] == "running"
    assert before["cost_snapshot"]["priced_at"] == PEAK_START.isoformat()
    monkeypatch.setitem(pricing.PRICES, MODEL, {"peak": (20, 40, 60), "off_peak": (10, 20, 30)})
    recorder.finish_step("failed", error="工具执行失败")
    after = store.usage_steps(session.session_id)[0]
    assert after["status"] == "failed"
    assert after["usage"] == RAW_USAGE
    assert after["cost_snapshot"] == before["cost_snapshot"]
    store.finish_step(recorder.step_id, "failed", usage=RAW_USAGE)
    assert store.usage_steps(session.session_id)[0]["cost_snapshot"] == before["cost_snapshot"]
    reopened = SessionStore(store.connection)
    reopened.recover_session(session.session_id)
    assert reopened.inspect(session.session_id)["steps"][0]["cost_snapshot"] == before["cost_snapshot"]
    assert reopened.usage_steps(session.session_id)[0]["turn_id"] == turn.turn_id


def test_tool_interruption_keeps_already_received_usage_and_cost(store):
    """通过真实 Agent 循环验证工具中断不会丢失模型消耗。

    参数:
        store: 隔离存储。

    返回:
        None，由断言确认已取消步骤仍保留响应统计。

    行为说明:
        1. 模型先返回完整用量和工具调用，工具执行再触发用户中断。
        2. 循环补齐工具结果并取消步骤，费用快照仍来自已收到的响应。
    """

    class Model:
        def complete(self, *, messages, tools):
            """返回包含工具调用和完整用量的固定模型响应。"""
            return {
                "tool_calls": [{"id": "interrupt", "function": {"name": "interrupt", "arguments": "{}"}}],
                "usage": RAW_USAGE,
                "finish_reason": "tool_calls",
            }

    def interrupt():
        """模拟工具执行期间的用户中断。

        异常:
            KeyboardInterrupt: 每次执行都触发，交由 Agent 循环收尾。
        """
        raise KeyboardInterrupt

    session, _, recorder = begin_recording(store)
    messages = [{"role": "user", "content": "执行工具"}]
    with pytest.raises(KeyboardInterrupt):
        run_agent_turn(messages, Model(), {"interrupt": interrupt}, [], recorder=recorder)
    step = store.usage_steps(session.session_id)[0]
    assert step["status"] == "cancelled"
    assert step["usage"] == RAW_USAGE
    assert step["cost_snapshot"]["cost_usd"] is not None
    assert step["cost_snapshot"]["source"] == "response"


def test_response_write_failure_rolls_back_usage_and_snapshot_together(store):
    """验证响应持久化失败时不会只保存用量或费用的一部分。

    参数:
        store: 隔离存储。

    返回:
        None，由断言确认事务回滚和记录器故障标记。

    行为说明:
        1. 在步骤创建后让同一事务中的会话更新时间写入失败。
        2. 确认步骤的用量、费用均未提交，并结束 SQLite 事务。
    """
    session, _, recorder = begin_recording(store)
    recorder.start_step(1)
    store.connection.execute(
        """CREATE TRIGGER fail_session_touch BEFORE UPDATE ON sessions
            BEGIN SELECT RAISE(ABORT, 'test response transaction failure'); END"""
    )
    with pytest.raises(sqlite3.IntegrityError, match="test response transaction failure"):
        recorder.record_response({"usage": RAW_USAGE}, 1)
    step = store.usage_steps(session.session_id)[0]
    assert step["usage"] is None and step["cost_snapshot"] is None
    assert recorder.storage_failed
    assert not store.connection.in_transaction


def test_usage_steps_filters_sessions_and_orders_turns_without_payloads(store):
    """验证统计只读取当前会话且依照真实执行顺序返回。

    参数:
        store: 隔离存储。

    返回:
        None，由断言确认会话隔离、排序和返回字段。

    行为说明:
        1. 创建两个会话，其中一个包含多回合、多步骤。
        2. 校验明细无消息载荷，并确认空会话没有步骤。
    """
    session, turn, recorder = begin_recording(store)
    for index in (1, 2):
        recorder.start_step(index)
        recorder.record_response({"usage": RAW_USAGE}, 1)
        recorder.finish_step("completed")
    store.finish_turn(turn.turn_id, "completed", "final_answer")
    next_turn = store.begin_turn(session.session_id, {"role": "user", "content": "第二轮"})
    later = store.start_step(next_turn.turn_id, 1, MODEL)
    other, _, other_recorder = begin_recording(store)
    other_recorder.start_step(1)
    other_recorder.record_response({"usage": {**RAW_USAGE, "prompt_tokens": 90000}}, 1)
    rows = store.usage_steps(session.session_id)
    assert [row["step_id"] for row in rows] == recorder.step_ids + [later.step_id]
    assert [row["turn_id"] for row in rows] == [turn.turn_id, turn.turn_id, next_turn.turn_id]
    assert len(store.usage_steps(other.session_id)) == 1
    assert all("payload" not in row for row in rows)
    assert rows[-1]["usage"] is None and rows[-1]["cost_snapshot"] is None
    empty = store.create_session("空会话", provider="openai", model=MODEL)
    assert store.usage_steps(empty.session_id) == []


def test_legacy_schema_migration_preserves_records_and_never_reprices(tmp_path, monkeypatch):
    """验证旧数据库加列、补估和再次打开均保留原始记录。

    参数:
        tmp_path: 隔离数据库目录。
        monkeypatch: 控制迁移前后模型价格。

    返回:
        None，由断言确认迁移幂等、未知值和原字段保留。

    行为说明:
        1. 构造缺少快照列的旧库，包含有效时间、损坏时间和缺失用量。
        2. 首次迁移后改变价格重开，已知和未知快照都不重新估算。
    """
    monkeypatch.setitem(pricing.PRICES, MODEL, {"peak": (2, 4, 6), "off_peak": (1, 2, 3)})
    connection = sqlite3.connect(tmp_path / "legacy.sqlite3")
    try:
        connection.executescript(SESSION_SCHEMA.replace("    cost_snapshot TEXT,\n", ""))
        connection.execute(
            """INSERT INTO sessions(session_id, title, provider, model, created_at, updated_at)
                VALUES ('legacy', '旧会话', 'openai', ?, ?, ?)""",
            (MODEL, PEAK_START.isoformat(), PEAK_START.isoformat()),
        )
        connection.execute(
            """INSERT INTO session_turns(turn_id, session_id, user_message_id, status, created_at)
                VALUES ('turn', 'legacy', 'message', 'completed', ?)""",
            (PEAK_START.isoformat(),),
        )
        connection.execute(
            """INSERT INTO session_messages(message_id, session_id, seq, turn_id, role, payload, created_at)
                VALUES ('message', 'legacy', 0, 'turn', 'user', ?, ?)""",
            (json.dumps({"role": "user", "content": "保留的历史正文"}), PEAK_START.isoformat()),
        )
        for index, (created_at, usage) in enumerate(
            [
                (PEAK_START.isoformat(), RAW_USAGE),
                ("not-a-date", RAW_USAGE),
                ("2026-10-19T01:59:00", RAW_USAGE),
                (PEAK_START.isoformat(), {"prompt_tokens": 15}),
                (PEAK_START.isoformat(), None),
            ],
            start=1,
        ):
            connection.execute(
                """INSERT INTO session_steps(step_id, turn_id, step_index, model, status, usage, created_at)
                    VALUES (?, 'turn', ?, ?, 'completed', ?, ?)""",
                (f"step-{index}", index, MODEL, json.dumps(usage) if usage is not None else None, created_at),
            )
        connection.commit()
        original_query = (
            "SELECT step_id, turn_id, step_index, model, status, usage, created_at "
            "FROM session_steps ORDER BY step_index"
        )
        originals = connection.execute(original_query).fetchall()
        migrated = SessionStore(connection)
        rows = migrated.usage_steps("legacy")
        assert rows[0]["cost_snapshot"]["cost_usd"] == pytest.approx(0.0326)
        assert rows[0]["cost_snapshot"]["source"] == "legacy_estimate"
        for row in rows[1:3]:
            assert row["cost_snapshot"]["cost_usd"] is None
            assert row["cost_snapshot"]["priced_at"] is None
            assert row["cost_snapshot"]["period"] is None
        assert rows[3]["cost_snapshot"]["cost_usd"] is None
        assert rows[4]["cost_snapshot"] is None
        monkeypatch.setitem(pricing.PRICES, MODEL, {"peak": (20, 40, 60), "off_peak": (10, 20, 30)})
        reopened = SessionStore(connection)
        assert reopened.usage_steps("legacy") == rows
        assert connection.execute(original_query).fetchall() == originals
        assert reopened.inspect("legacy")["messages"][0]["payload"]["content"] == "保留的历史正文"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert not connection.in_transaction
    finally:
        connection.close()
