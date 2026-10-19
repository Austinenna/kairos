"""用隔离 SQLite 验证通道收件、路由和发送边界，不调用网络或 Runtime。"""

import sqlite3
import stat
from dataclasses import replace

import pytest

from kairos.channels.base import InboundMessage, ReceiveBatch
from kairos.channels.store import ChannelStore


@pytest.fixture
def store(tmp_path):
    """提供测试专用通道数据库。

    作用:
        隔离每个测试的通道状态并保证连接释放。
    参数:
        tmp_path: pytest 临时目录。
    返回:
        一个 ChannelStore 实例。
    行为说明:
        1. 不读取真实配置或业务数据。
    """
    result = ChannelStore(tmp_path / "channels.sqlite3")
    try:
        yield result
    finally:
        result.close()


def message(message_id="1", text="测试输入", **changes):
    """构造有固定身份的合成通道输入。

    作用:
        让各测试只覆盖当前需要改变的身份或正文。
    参数:
        message_id: 原平台消息 ID。
        text: 原输入文字，None 表示不支持的类型。
        changes: 需要替换的身份字段。
    返回:
        统一 InboundMessage。
    行为说明:
        1. 所有默认标识均为测试数据。
    """
    return replace(InboundMessage("weixin", "bot", "chat", "owner", message_id, text), **changes)


def enqueue(store, incoming):
    """保存一条输入并返回对应记录。

    作用:
        为状态变化用例准备真正经过收件事务的记录。
    参数:
        store: 临时通道存储。
        incoming: 合成输入。
    返回:
        当前最早 queued 记录。
    行为说明:
        1. 使用正常公开接口保存输入。
    """
    store.accept_batch(incoming.channel, incoming.account_id, ReceiveBatch((incoming,), {}))
    return store.next_queued()


def prepared(
    store, incoming=None, *, session="session-a", text="完整模型回答", parts=("第一段", "第二段"), kind="model"
):
    """准备一条已生成但尚未发送的回复。

    作用:
        为分段、恢复和缓存测试建立处理成功边界。
    参数:
        store: 临时存储。
        incoming: 可选合成输入。
        session: 内部会话标识。
        text: 完整回复。
        parts: 发送分段。
        kind: 回复类型。
    返回:
        ready 状态的记录。
    行为说明:
        1. 只推进存储状态，不调用真实模型。
    """
    incoming = incoming or message()
    record = enqueue(store, incoming)
    store.bind(incoming, session)
    store.claim(record.id, session)
    store.prepare_reply(record.id, text, parts, kind=kind, turn_id="turn-test")
    return store.get(record.id)


def test_inbox_deduplicates_without_rewriting_original_and_saves_checkpoint(store):
    """验证入站去重、原文保留和游标推进。

    作用:
        重复投递只更新接收位置，不重建任务或改写已接收文字。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 使用超大平台 ID、重复输入及不支持类型检查完整保存。
    """
    large_id = str(2**63 - 1)
    first = message(large_id, "原始文字 '); DROP TABLE channel_inbox; --")
    other = message("unsupported", None)
    store.seed_checkpoint("weixin", "bot", {"cursor": "seed"})
    store.seed_checkpoint("weixin", "bot", {"cursor": "ignored"})
    assert store.checkpoint("weixin", "bot") == {"cursor": "seed"}
    batch = ReceiveBatch((first, first, other), {"cursor": "next", "contexts": {"chat": "synthetic-token"}})
    assert store.accept_batch("weixin", "bot", batch) == 2
    record = store.next_queued()
    assert record.message == first and record.message.message_id == large_id
    store.claim(record.id, "session-a")
    assert store.next_queued().message == other
    assert (
        store.accept_batch("weixin", "bot", ReceiveBatch((replace(first, text="重复载荷"),), {"cursor": "last"})) == 0
    )
    assert store.get(record.id).message.text == first.text
    assert store.checkpoint("weixin", "bot") == {"cursor": "last"}
    assert store.checkpoint("weixin", "another-bot") is None


def test_checkpoint_write_failure_rolls_back_entire_inbox_batch(store):
    """验证游标保存失败会回滚此前插入的全部消息。

    作用:
        用真实 SQLite 写入故障检查输入与游标的原子关系。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 在检查点更新时用触发器模拟失败，确认没有残留消息。
    """
    store.seed_checkpoint("weixin", "bot", {"cursor": "before"})
    store.connection.execute(
        """CREATE TRIGGER fail_checkpoint BEFORE UPDATE ON channel_checkpoints
           BEGIN SELECT RAISE(ABORT, 'simulated checkpoint failure'); END"""
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.accept_batch("weixin", "bot", ReceiveBatch((message("1"), message("2")), {"cursor": "after"}))
    assert store.next_queued() is None
    assert store.checkpoint("weixin", "bot") == {"cursor": "before"}
    assert not store.connection.in_transaction


@pytest.mark.parametrize("changed", [{"channel": "other"}, {"account_id": "other"}, {"text": 123}])
def test_invalid_batch_never_partially_accepts(store, changed):
    """验证混入其他账号或无效输入时整批拒绝。

    作用:
        防止跨账号入站及半批推进接收位置。
    参数:
        store: 临时存储。
        changed: 第二条输入的非法变化。
    返回:
        None。
    行为说明:
        1. 第一条合法、第二条非法，确认两者均未写入。
    """
    invalid = replace(message("2"), **changed)
    with pytest.raises(ValueError):
        store.accept_batch("weixin", "bot", ReceiveBatch((message(), invalid), {"cursor": "after"}))
    assert store.next_queued() is None
    assert store.checkpoint("weixin", "bot") is None


def test_binding_and_deduplication_are_isolated_by_channel_account_conversation(store):
    """验证绑定和消息去重使用完整外部身份。

    作用:
        防止不同账号或对话恰好使用相同消息 ID 时串话。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 四个身份分别绑定；重绑一个身份不会改变其余记录。
    """
    identities = [message(), message(channel="other"), message(account_id="other"), message(conversation_id="other")]
    for index, incoming in enumerate(identities):
        assert store.binding(incoming) is None
        store.bind(incoming, f"session-{index}")
        assert store.accept_batch(incoming.channel, incoming.account_id, ReceiveBatch((incoming,), {})) == 1
    store.bind(identities[0], "new-session")
    assert [store.binding(item) for item in identities] == ["new-session", "session-1", "session-2", "session-3"]
    assert store.connection.execute("SELECT count(*) FROM channel_inbox").fetchone()[0] == 4


def test_new_database_permissions_are_private_without_changing_existing_paths(tmp_path):
    """验证新目录和数据库权限，并保留已有路径权限。

    作用:
        私有保存消息和回复上下文，同时不擅自修改已有目录配置。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 检查多层新目录为 700、新文件为 600，重开时不改已有权限。
    """
    path = tmp_path / "private" / "channels" / "state.sqlite3"
    store = ChannelStore(path)
    store.close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700
    path.chmod(0o640)
    path.parent.chmod(0o750)
    store = ChannelStore(path)
    store.close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o750


def test_segment_progress_is_ordered_persistent_and_keeps_client_ids(store):
    """验证分段顺序、确认依据及稳定发送标识。

    作用:
        确保回复只沿已记录状态前进，不跳段或重新生成标识。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 拒绝跳段和重复开始，分别保存 HTTP 接受及业务确认。
    """
    record = prepared(store)
    identifiers = [part["client_id"] for part in record.parts]
    assert len(set(identifiers)) == 2
    assert all(part["acknowledged"] is None for part in record.parts)
    with pytest.raises(RuntimeError):
        store.mark_sending(record.id, 1)
    store.mark_sending(record.id, 0)
    assert store.get(record.id).status == "sending"
    assert store.ready_replies() == []
    with pytest.raises(RuntimeError):
        store.mark_sending(record.id, 0)
    store.mark_submitted(record.id, 0, False)
    assert store.get(record.id).status == "ready"
    assert store.get(record.id).parts[0]["status"] == "submitted"
    store.mark_sending(record.id, 1)
    store.mark_submitted(record.id, 1, True)
    saved = store.get(record.id)
    assert saved.status == "sent"
    assert [part["client_id"] for part in saved.parts] == identifiers
    assert [part["acknowledged"] for part in saved.parts] == [False, True]
    assert store.ready_replies() == [] and store.next_queued() is None
    with pytest.raises(RuntimeError):
        store.claim(record.id, "session-a")
    with pytest.raises(RuntimeError):
        store.prepare_reply(record.id, "覆盖内容", ["覆盖内容"])


def test_send_failure_halts_remaining_parts_but_keeps_model_response(store):
    """验证发送异常停止整条回复并保留可显式重发的原答案。

    作用:
        在平台可能已经收到文字时避免自动重复或继续后续分段。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 标记第一段结果未知，保留原文、稳定 ID、错误码和未发送第二段。
    """
    record = prepared(store)
    store.mark_sending(record.id, 0)
    store.mark_uncertain(record.id, 0, "network_timeout")
    saved = store.get(record.id)
    assert saved.status == "delivery_uncertain" and saved.error_code == "network_timeout"
    assert [part["status"] for part in saved.parts] == ["uncertain", "ready"]
    assert saved.parts[0]["client_id"] == record.parts[0]["client_id"]
    assert saved.turn_id == "turn-test" and saved.response == "完整模型回答"
    assert store.latest_model_reply(record.message) == "完整模型回答"
    assert store.ready_replies() == [] and store.next_queued() is None
    with pytest.raises(RuntimeError):
        store.mark_sending(record.id, 1)
    assert store.recover() == 0


def test_processing_recovery_produces_traceable_error_without_requeuing(tmp_path):
    """验证处理期间中断只生成通知，不重新排队执行原请求。

    作用:
        避免未知工具副作用在重启后被自动重复执行。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 真正关闭并重开数据库，核对原输入、会话和中断错误可追溯。
    """
    path = tmp_path / "state.sqlite3"
    store = ChannelStore(path)
    first = enqueue(store, message())
    store.bind(first.message, "session-a")
    store.claim(first.id, "session-a")
    enqueue(store, message("2", "尚未开始"))
    store.close()
    store = ChannelStore(path)
    try:
        assert store.recover() == 1
        recovered = store.get(first.id)
        assert recovered.status == "ready" and recovered.kind == "error"
        assert recovered.error_code == "interrupted" and recovered.session_id == "session-a"
        assert recovered.message == first.message
        assert "没有自动重新执行模型或工具" in recovered.response
        assert store.next_queued().message.message_id == "2"
        assert store.latest_model_reply(first.message) is None
        assert store.ready_replies() == [recovered]
        assert store.recover() == 0
        assert store.get(first.id).parts == recovered.parts
    finally:
        store.close()


def test_sending_recovery_never_resends_unknown_part(tmp_path):
    """验证发送途中退出会隔离未知分段并保留其他进度。

    作用:
        防止重启把已经发送但尚未确认的文字再次投递。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 第一段已提交、第二段正在发送、第三段未开始，重开后整体停止发送。
    """
    path = tmp_path / "state.sqlite3"
    store = ChannelStore(path)
    record = prepared(store, parts=["一", "二", "三"])
    store.mark_sending(record.id, 0)
    store.mark_submitted(record.id, 0, True)
    store.mark_sending(record.id, 1)
    store.close()
    store = ChannelStore(path)
    try:
        assert store.recover() == 1
        saved = store.get(record.id)
        assert saved.status == "delivery_uncertain" and saved.error_code == "delivery_interrupted"
        assert [part["status"] for part in saved.parts] == ["submitted", "uncertain", "ready"]
        assert [part["client_id"] for part in saved.parts] == [part["client_id"] for part in record.parts]
        assert store.ready_replies() == [] and store.next_queued() is None
        assert store.latest_model_reply(record.message) == record.response
        assert store.recover() == 0
    finally:
        store.close()


def test_completed_part_and_ready_remainder_survive_restart(tmp_path):
    """验证已确认分段不重发，剩余 ready 分段可以继续。

    作用:
        区分安全继续发送与结果未知时必须停止的恢复路径。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 第一段提交后退出，重启仅对第二段继续状态变化。
    """
    path = tmp_path / "state.sqlite3"
    store = ChannelStore(path)
    record = prepared(store)
    store.mark_sending(record.id, 0)
    store.mark_submitted(record.id, 0, False)
    store.close()
    store = ChannelStore(path)
    try:
        assert store.recover() == 0
        saved = store.ready_replies()[0]
        assert [part["status"] for part in saved.parts] == ["submitted", "ready"]
        assert saved.parts[1]["client_id"] == record.parts[1]["client_id"]
        with pytest.raises(RuntimeError):
            store.mark_sending(record.id, 0)
        store.mark_sending(record.id, 1)
        store.mark_submitted(record.id, 1, True)
        assert store.get(record.id).status == "sent"
    finally:
        store.close()


def test_cached_reply_uses_current_binding_and_excludes_commands_errors(store):
    """验证显式重发只取当前会话最近模型答案。

    作用:
        防止新建会话后误发旧答案，或把帮助、错误通知当作模型回复。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 混入命令、错误与另一对话，再切换绑定检查查询边界。
    """
    first = prepared(store, message("1"), text="旧模型答案")
    prepared(store, message("2"), text="最新模型答案")
    prepared(store, message("3"), text="帮助信息", kind="command")
    prepared(store, message("4"), text="错误通知", kind="error")
    prepared(store, message("5", conversation_id="other"), text="其他聊天答案")
    assert store.latest_model_reply(first.message) == "最新模型答案"
    store.bind(first.message, "new-session")
    assert store.latest_model_reply(first.message) is None
    prepared(store, message("6"), session="new-session", text="新会话答案")
    assert store.latest_model_reply(first.message) == "新会话答案"
    store.bind(first.message, "session-a")
    assert store.latest_model_reply(first.message) == "最新模型答案"


def test_returned_state_is_detached_and_invalid_transition_keeps_storage(store):
    """验证读到的可变对象不污染存储，非法变化无部分写入。

    作用:
        保持持久状态只通过受控方法前进。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 修改读出的分段和检查点，确认下次读取不受影响。
        2. 无效布尔值及索引不改变状态。
    """
    record = prepared(store)
    checkpoint = store.checkpoint("weixin", "bot")
    checkpoint["unexpected"] = True
    assert store.checkpoint("weixin", "bot") == {}
    record.parts[0]["text"] = "外部篡改"
    assert store.get(record.id).parts[0]["text"] == "第一段"
    with pytest.raises(ValueError):
        store.mark_sending(record.id, True)
    with pytest.raises(ValueError):
        store.mark_sending(record.id, -1)
    store.mark_sending(record.id, 0)
    with pytest.raises(ValueError):
        store.mark_submitted(record.id, 0, 1)
    assert store.get(record.id).status == "sending"
    with pytest.raises(KeyError):
        store.get(record.id + 100)
    with pytest.raises(ValueError):
        store.get(True)


def test_scoped_queue_skips_older_messages_from_other_accounts(store):
    """验证当前账号只取得自己的等待输入。

    作用:
        换 Bot 后旧账号积压不能阻塞或混入当前处理。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 旧账号及其他通道先入队，当前账号仍能取得自身消息。
    """
    old = message("old")
    other_channel = message("other-channel", channel="other", account_id="new-bot")
    current = message("current", account_id="new-bot")
    for incoming in (old, other_channel, current):
        store.accept_batch(incoming.channel, incoming.account_id, ReceiveBatch((incoming,), {}))
    assert store.next_queued().message == old
    assert store.next_queued("weixin", "new-bot").message == current
    assert store.next_queued("other", "new-bot").message == other_channel
    assert store.next_queued("weixin", "missing-bot") is None


def test_scoped_replies_and_recovery_leave_other_accounts_untouched(store):
    """验证待发列表和中断恢复都限定当前通道账号。

    作用:
        防止新账号发送旧账号答案，或改变另一个账号的执行状态。
    参数:
        store: 临时存储。
    返回:
        None。
    行为说明:
        1. 准备各账号的回复、处理中输入和发送中分段。
        2. 范围恢复只处理当前账号；无范围恢复仍可显式处理剩余账号。
    """
    old_ready = prepared(store, message("old-ready"))
    current_ready = prepared(store, message("current-ready", account_id="new-bot"))
    old_processing = enqueue(store, message("old-processing"))
    store.claim(old_processing.id, "session-old")
    current_processing = enqueue(store, message("current-processing", account_id="new-bot"))
    store.claim(current_processing.id, "session-current")
    other_processing = enqueue(store, message("other-processing", channel="other", account_id="new-bot"))
    store.claim(other_processing.id, "session-other")
    old_sending = prepared(store, message("old-sending"))
    store.mark_sending(old_sending.id, 0)
    current_sending = prepared(store, message("current-sending", account_id="new-bot"))
    store.mark_sending(current_sending.id, 0)

    assert [record.id for record in store.ready_replies()] == [old_ready.id, current_ready.id]
    assert store.ready_replies("weixin", "new-bot") == [current_ready]
    assert store.ready_replies("other", "new-bot") == []
    assert store.recover("weixin", "new-bot") == 2
    assert store.get(current_processing.id).kind == "error"
    assert store.get(current_sending.id).status == "delivery_uncertain"
    assert store.get(old_processing.id).status == "processing"
    assert store.get(other_processing.id).status == "processing"
    assert store.get(old_sending.id).status == "sending"
    assert [record.id for record in store.ready_replies("weixin", "new-bot")] == [
        current_ready.id,
        current_processing.id,
    ]
    assert store.recover("weixin", "new-bot") == 0
    assert store.recover() == 3


@pytest.mark.parametrize("method", ["next_queued", "ready_replies", "recover"])
@pytest.mark.parametrize("scope", [{"channel": "weixin"}, {"account_id": "bot"}])
def test_partial_scope_is_rejected_without_changes(store, method, scope):
    """验证不完整的账号范围不会退化为全局操作。

    作用:
        防止调用方漏传身份字段时意外读取或恢复其他账号。
    参数:
        store: 临时存储。
        method: 查询或恢复方法名称。
        scope: 只含通道或账号的参数。
    返回:
        None。
    行为说明:
        1. 调用必须明确失败，原处理中输入保持原样。
    """
    record = enqueue(store, message())
    store.claim(record.id, "session-a")
    with pytest.raises(ValueError):
        getattr(store, method)(**scope)
    assert store.get(record.id).status == "processing"
