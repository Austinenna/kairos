"""离线验证后台提取的调度边界、工具权限、来源和失败报告。"""

import json
from copy import deepcopy
from threading import Event, Thread

import pytest

import kairos.memory as memory_module
from kairos.memory import MemoryStore
from kairos.memory_extraction import ExtractionJob, MemoryExtractor


def final(status="no_change"):
    return {"content": json.dumps({"status": status})}


def call(name, arguments, call_id="call-1"):
    return {
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


def new_topic(addition="用户有十年 Python 经验。"):
    return {"topic_id": None, "title": "学习背景", "description": "调整讲解深度时参考。", "addition": addition}


class FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeModel:
    def __init__(self, responses, *, entered=None, release=None):
        self.responses = iter(responses)
        self.requests = []
        self.client = FakeClient()
        self.entered = entered
        self.release = release

    def complete(self, *, messages, tools):
        self.requests.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        if self.entered:
            self.entered.set()
        if self.release:
            assert self.release.wait(3), "测试没有释放假模型"
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def job(factory, turn="turn-1", *, messages=None, start=1):
    return ExtractionJob(
        source={"session_id": "session-1", "turn_id": turn, "message_id": f"message-{turn}"},
        messages=messages
        or [
            {"role": "system", "content": "前台系统提示"},
            {"role": "user", "content": "我有十年 Python 经验。"},
            {"role": "assistant", "content": "我会结合你的背景解释。"},
        ],
        current_turn_start=start,
        model_factory=factory,
    )


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory")


def run_job(store, model, **kwargs):
    extractor = MemoryExtractor(store, **kwargs)
    try:
        assert extractor.submit(job(lambda: model))
        assert extractor.wait(3)
        return extractor.drain_events()[0]
    finally:
        extractor.close()


def test_submit_is_nonblocking_serial_deduplicated_and_models_are_independent(store):
    entered, release = Event(), Event()
    models = []

    def factory():
        model = FakeModel([final()], entered=entered if not models else None, release=release if not models else None)
        models.append(model)
        return model

    extractor = MemoryExtractor(store)
    try:
        assert extractor.submit(job(factory))
        assert entered.wait(2)
        assert not extractor.submit(job(factory))
        assert extractor.submit(job(factory, "turn-2"))
        assert extractor.pending_count == 2
        assert len(models) == 1
        assert not extractor.wait(0.01)
        release.set()
        assert extractor.wait(3)
        assert len(models) == 2
        assert models[0] is not models[1]
        assert all(model.client.closed for model in models)
        events = extractor.drain_events()
        assert [event["source"]["turn_id"] for event in events] == ["turn-1", "turn-2"]
        assert all(event["status"] == "no_change" for event in events)
        assert extractor.drain_events() == []
        assert extractor.pending_count == 0
    finally:
        release.set()
        extractor.close()
    assert not extractor.submit(job(factory, "turn-3"))


def test_queued_snapshot_keeps_boundary_and_omits_system_reasoning_and_image_bytes(store):
    entered, release = Event(), Event()
    blocker = FakeModel([final()], entered=entered, release=release)
    model = FakeModel([final()])
    material = [
        {"role": "system", "content": "不要发送的前台系统指令"},
        {"role": "user", "content": "旧话题只作指代背景"},
        {
            "role": "assistant",
            "content": "选择一或二",
            "reasoning_content": "内部推理不能提取",
            "provider_content": [{"type": "thinking", "thinking": "供应商隐藏思考"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "我选择第二个"},
                {"type": "image", "data": "PRIVATE_BASE64", "mime_type": "image/png"},
            ],
        },
        {"role": "assistant", "content": "好的，按第二个方案。"},
    ]
    task = job(lambda: model, "turn-2", messages=material, start=3)
    extractor = MemoryExtractor(store)
    try:
        extractor.submit(job(lambda: blocker))
        assert entered.wait(2)
        extractor.submit(task)
        material[3]["content"][0]["text"] = "错误：前台后来修改"
        task.source["message_id"] = "mutated-source"
        material.append({"role": "user", "content": "不应进入快照的下轮消息"})
        store.remember("排队之后保存的最新核心内容", source=job(lambda: None).source)
        release.set()
        assert extractor.wait(3)
        event = extractor.drain_events()[-1]
    finally:
        release.set()
        extractor.close()
    payload = json.loads(model.requests[0]["messages"][-1]["content"])
    assert payload["current_turn"][0]["content"][0]["text"] == "我选择第二个"
    assert len(payload["current_turn"]) == 2
    assert len(payload["earlier_context"]) == 2
    encoded = json.dumps(payload, ensure_ascii=False)
    for omitted in ("前台系统指令", "内部推理", "隐藏思考", "PRIVATE_BASE64", "不应进入快照"):
        assert omitted not in encoded
    assert "图片数据已省略" in encoded
    assert event["source"]["message_id"] == "message-turn-2"
    assert "排队之后保存的最新核心内容" in str(model.requests[0])


def test_create_uses_runtime_source_only_and_records_safe_metadata(store):
    model = FakeModel(
        [
            {
                **call("write_topic", new_topic("PERSONAL_TOPIC_BODY")),
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "debug": "PRIVATE_USAGE"},
            },
            {"content": '{"status":"completed","summary":"PRIVATE_SUMMARY"}'},
        ]
    )
    result = run_job(store, model)
    assert result["status"] == "completed"
    assert len(result["topic_ids"]) == 1
    topic = store.read_topic(result["topic_ids"][0])
    assert topic["content"] == "PERSONAL_TOPIC_BODY"
    assert "message-turn-1" in json.dumps(topic)
    assert result["usage"][0]["prompt_tokens"] == 100
    assert result["usage"][1]["prompt_tokens"] is None
    assert result["request_count"] == 2
    for request in model.requests:
        assert {tool["function"]["name"] for tool in request["tools"]} == {"read_memory", "write_topic"}
        assert "CORE.md 完整正文" in str(request["messages"])
        assert "MEMORY.md 完整根索引" in str(request["messages"])
    record = (store.root / "extraction.jsonl").read_text()
    for secret in ("PERSONAL_TOPIC_BODY", "PRIVATE_USAGE", "PRIVATE_SUMMARY", "十年 Python", "学习背景"):
        assert secret not in record
    assert json.loads(record)["status"] == "completed"


def test_existing_topic_requires_read_before_append(store):
    created = store.write_topic(**new_topic(), source=job(lambda: None).source)
    topic_id = created["topic_id"]
    addition = {"topic_id": topic_id, "title": None, "description": None, "addition": "开始学习 Rust。"}
    denied = FakeModel([call("write_topic", addition), final("completed")])
    failed = run_job(store, denied)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "tool_failed"
    assert "开始学习 Rust" not in store.read_topic(topic_id)["content"]
    allowed = FakeModel(
        [call("read_memory", {"topic_id": topic_id}), call("write_topic", addition, "append"), final("completed")]
    )
    saved = run_job(store, allowed)
    assert saved["status"] == "completed"
    assert saved["topic_ids"] == [topic_id]
    assert "开始学习 Rust" in store.read_topic(topic_id)["content"]


@pytest.mark.parametrize(
    "name,arguments", [("remember", {"content": "不能修改CORE"}), ("add_todo", {"title": "不能执行"})]
)
def test_background_cannot_run_core_or_business_tools(store, name, arguments):
    before = store.snapshot()
    model = FakeModel([call(name, arguments), final("completed")])
    result = run_job(store, model)
    assert result["status"] == "failed"
    assert result["error_code"] == "tool_failed"
    assert store.snapshot() == before


def test_model_cannot_supply_source(store):
    model = FakeModel([call("write_topic", {**new_topic(), "source": {"message_id": "spoofed"}}), final("completed")])
    result = run_job(store, model)
    assert result["status"] == "failed"
    assert result["topic_ids"] == []
    assert not list((store.root / "topics").glob("*.md"))


def test_existing_or_duplicate_results_are_not_counted_as_written(store):
    existing = store.write_topic(**new_topic(), source=job(lambda: None).source)
    model = FakeModel([call("write_topic", new_topic()), final("completed")])
    result = run_job(store, model)
    assert result["status"] == "no_change"
    assert result["topic_ids"] == []
    duplicate = FakeModel(
        [
            call("read_memory", {"topic_id": existing["topic_id"]}),
            call("write_topic", {**new_topic(), "topic_id": existing["topic_id"]}),
            final("completed"),
        ]
    )
    assert run_job(store, duplicate)["status"] == "no_change"


@pytest.mark.parametrize(
    "responses,status,code",
    [
        ([final("completed")], "no_change", None),
        ([final("conflict")], "conflict", "memory_conflict"),
        ([{"content": "声称已经保存，但没有结构化状态"}], "failed", "invalid_completion"),
        ([RuntimeError("PRIVATE_RAW_ERROR")], "failed", "extraction_failed"),
        ([call("write_topic", new_topic()), RuntimeError("PRIVATE_RAW_ERROR")], "partial", "extraction_failed"),
        ([call("write_topic", new_topic()), final("conflict")], "partial", "memory_conflict"),
        ([call("write_topic", new_topic()), call("remember", {"content": "no"}), final()], "partial", "tool_failed"),
    ],
)
def test_noop_failure_and_partial_write_statuses(store, responses, status, code):
    model = FakeModel(responses)
    result = run_job(store, model)
    assert result["status"] == status
    assert result["error_code"] == code
    assert "PRIVATE_RAW_ERROR" not in (store.root / "extraction.jsonl").read_text()
    assert model.client.closed


def test_loop_limit_is_not_success_after_partial_write(store):
    model = FakeModel([call("write_topic", new_topic()), call("write_topic", new_topic("第二条"))])
    result = run_job(store, model, max_tool_rounds=1)
    assert result["status"] == "partial"
    assert result["error_code"] == "tool_round_limit"
    assert len(result["topic_ids"]) == 1


@pytest.mark.parametrize("append", [False, True])
def test_topic_saved_before_index_error_is_partial_and_snapshot_repairs_index(store, monkeypatch, append):
    existing = store.write_topic(**new_topic(), source=job(lambda: None).source) if append else None
    original_write = memory_module._atomic_write
    index_failures = 0

    def fail_index_once(path, content):
        nonlocal index_failures
        if path == store.index_path and index_failures == 0:
            index_failures += 1
            raise OSError("PRIVATE_INDEX_FAILURE")
        return original_write(path, content)

    # 新建会改变索引；追加正文不改变标题/描述，故先删索引以便目标写入走到索引更新。
    # 在模型首次执行 write_topic 前删除，避免 ContextBuilder 先修复影响故障位置。
    responses = [
        call(
            "write_topic",
            {
                "topic_id": existing["topic_id"] if existing else None,
                "title": None if existing else "学习背景",
                "description": None if existing else "调整讲解深度时参考。",
                "addition": "PRIVATE_NEW_TOPIC_DETAIL",
            },
        ),
        final("completed"),
    ]
    if existing:
        responses.insert(0, call("read_memory", {"topic_id": existing["topic_id"]}))
    model = FakeModel(responses)
    original_complete = model.complete

    def complete_and_inject_failure(*, messages, tools):
        response = original_complete(messages=messages, tools=tools)
        calls = response.get("tool_calls") or []
        if calls and calls[0]["function"]["name"] == "write_topic":
            if existing:
                store.index_path.unlink()
            monkeypatch.setattr(memory_module, "_atomic_write", fail_index_once)
        return response

    monkeypatch.setattr(model, "complete", complete_and_inject_failure)
    result = run_job(store, model)
    assert index_failures == 1
    assert result["status"] == "partial"
    assert result["error_code"] == "tool_failed"
    assert len(result["topic_ids"]) == 1
    topic_id = result["topic_ids"][0]
    if existing:
        assert topic_id == existing["topic_id"]
    assert "PRIVATE_NEW_TOPIC_DETAIL" in store.read_topic(topic_id)["content"]
    assert topic_id in store.snapshot()[1]
    assert topic_id in store.index_path.read_text()
    record = (store.root / "extraction.jsonl").read_text()
    assert "PRIVATE_INDEX_FAILURE" not in record
    assert "PRIVATE_NEW_TOPIC_DETAIL" not in record


def test_factory_failure_does_not_stop_later_jobs(store):
    def broken_factory():
        raise RuntimeError("不应记录的认证细节")

    extractor = MemoryExtractor(store)
    try:
        extractor.submit(job(broken_factory))
        extractor.submit(job(lambda: FakeModel([final()]), "turn-2"))
        assert extractor.wait(3)
        events = extractor.drain_events()
        assert events[0]["error_code"] == "model_factory_failed"
        assert events[1]["status"] == "no_change"
    finally:
        extractor.close()
    assert "认证细节" not in (store.root / "extraction.jsonl").read_text()


def test_close_drains_active_and_queued_jobs(store):
    entered, release, closed = Event(), Event(), Event()
    first = FakeModel([final()], entered=entered, release=release)
    second = FakeModel([final()])
    extractor = MemoryExtractor(store)
    extractor.submit(job(lambda: first))
    assert entered.wait(2)
    extractor.submit(job(lambda: second, "turn-2"))

    def shutdown():
        extractor.close()
        closed.set()

    closer = Thread(target=shutdown)
    closer.start()
    try:
        assert not closed.wait(0.02)
        release.set()
        assert closed.wait(3)
        assert second.requests
        assert first.client.closed and second.client.closed
        assert extractor.pending_count == 0
        extractor.close()
    finally:
        release.set()
        closer.join(3)


def test_log_failure_is_visible_to_main_thread(store, monkeypatch):
    extractor = MemoryExtractor(store)

    def fail_record(_):
        raise OSError("private path")

    monkeypatch.setattr(extractor, "_record", fail_record)
    try:
        extractor.submit(job(lambda: FakeModel([final()])))
        assert extractor.wait(3)
        event = extractor.drain_events()[0]
        assert event["status"] == "failed"
        assert event["error_code"] == "result_record_failed"
    finally:
        extractor.close()


def test_source_keeps_context_boundaries_but_drops_uncontrolled_fields(store):
    extractor = MemoryExtractor(store)
    task = job(lambda: FakeModel([call("write_topic", new_topic()), final("completed")]))
    task.source.update(
        context_start_message_id="previous-message",
        context_end_message_id="latest-message",
        arbitrary_note="PRIVATE_SOURCE_NOTE",
    )
    try:
        extractor.submit(task)
        assert extractor.wait(3)
        result = extractor.drain_events()[0]
    finally:
        extractor.close()
    assert result["source"]["context_start_message_id"] == "previous-message"
    assert result["source"]["context_end_message_id"] == "latest-message"
    record = (store.root / "extraction.jsonl").read_text()
    assert "PRIVATE_SOURCE_NOTE" not in record
    assert "previous-message" in json.dumps(store.read_topic(result["topic_ids"][0]))


def test_audit_does_not_follow_symlink_outside_memory_directory(store, tmp_path):
    destination = tmp_path / "outside.txt"
    destination.write_text("keep intact")
    (store.root / "extraction.jsonl").symlink_to(destination)
    result = run_job(store, FakeModel([final()]))
    assert result["status"] == "failed"
    assert result["error_code"] == "result_record_failed"
    assert destination.read_text() == "keep intact"


@pytest.mark.parametrize("start", [-1, 0, 2, 9, True])
def test_rejects_invalid_turn_boundary(store, start):
    extractor = MemoryExtractor(store)
    try:
        with pytest.raises(ValueError, match="current_turn_start"):
            extractor.submit(job(lambda: FakeModel([final()]), start=start))
    finally:
        extractor.close()
