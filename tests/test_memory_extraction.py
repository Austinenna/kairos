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
        assert {tool["function"]["name"] for tool in request["tools"]} == {"read_memory", "write_topic", "write_core"}
        assert "CORE.md 完整正文" in str(request["messages"])
        assert "MEMORY.md 完整根索引" in str(request["messages"])
    record = (store.root / "extraction.jsonl").read_text()
    for secret in ("PERSONAL_TOPIC_BODY", "PRIVATE_USAGE", "PRIVATE_SUMMARY", "十年 Python", "学习背景"):
        assert secret not in record
    assert json.loads(record)["status"] == "completed"


def test_core_only_write_binds_source_refreshes_request_and_records_safe_metadata(store):
    """验证后台核心写入真实落盘、来源绑定及续轮读取最新正文。

    作用:
        覆盖自动提取选择核心后的实际执行链，不用脚本模型证明语义分类。

    参数:
        store: 使用隔离目录的真实记忆存储。

    返回:
        None。

    行为说明:
        1. 通过真实后台工具保存核心并核对实际写入计数与程序来源。
        2. 检查第二次请求已经携带新核心，审计文件只记录受控元数据。
    """
    fact = "PRIVATE_CORE_BODY"
    model = FakeModel([call("write_core", {"content": fact}), final("completed")])
    result = run_job(store, model)

    assert result["status"] == "completed"
    assert result["core_additions"] == 1
    assert result["topic_ids"] == []
    core, _ = store.snapshot()
    assert fact in core
    assert list(store.topics_dir.glob("*.md")) == []
    persisted = store.core_path.read_text()
    for key, value in job(lambda: None).source.items():
        assert f'"{key}": "{value}"' in persisted
    assert fact not in model.requests[0]["messages"][1]["content"]
    assert fact in model.requests[1]["messages"][1]["content"]
    saved = json.loads(model.requests[1]["messages"][-1]["content"])
    assert saved["status"] == "saved" and saved["source"] == result["source"]
    record = (store.root / "extraction.jsonl").read_text()
    assert fact not in record
    assert json.loads(record)["core_additions"] == 1


def test_repeated_core_writes_count_only_actual_additions(store):
    """验证核心重复写入不制造新增或错误的完成状态。

    作用:
        保持工具去重结果、后台统计和文件内容一致。

    参数:
        store: 使用隔离目录的真实记忆存储。

    返回:
        None。

    行为说明:
        1. 同一任务写入相同正文两次，只统计第一次。
        2. 后续任务再次写入，必须报告无变化且不改写来源。
    """
    fact = "用户希望被称为测试称呼。"
    model = FakeModel(
        [call("write_core", {"content": fact}), call("write_core", {"content": fact}, "duplicate"), final("completed")]
    )
    first = run_job(store, model)
    original = store.core_path.read_bytes()
    assert first["status"] == "completed" and first["core_additions"] == 1
    assert store.snapshot()[0].count(fact) == 1
    assert json.loads(model.requests[2]["messages"][-1]["content"])["status"] == "duplicate"
    repeated = run_job(store, FakeModel([call("write_core", {"content": fact}), final("completed")]))
    assert repeated["status"] == "no_change" and repeated["core_additions"] == 0
    assert repeated["topic_ids"] == []
    assert store.core_path.read_bytes() == original


def test_core_and_topic_writes_share_one_completed_result(store):
    """验证同一后台任务可以分别保存常驻核心和主题详情。

    作用:
        防止增加核心入口后丢失原有主题流程或混淆两种写入统计。

    参数:
        store: 使用隔离目录的真实记忆存储。

    返回:
        None。

    行为说明:
        1. 依次执行核心和主题写入，并核对两个文件的独立正文。
    """
    model = FakeModel(
        [
            call("write_core", {"content": "核心里的称呼"}),
            call("write_topic", new_topic("主题里的项目经历"), "topic"),
            final("completed"),
        ]
    )
    result = run_job(store, model)
    assert result["status"] == "completed"
    assert result["core_additions"] == 1 and len(result["topic_ids"]) == 1
    assert "核心里的称呼" in store.snapshot()[0]
    assert "主题里的项目经历" not in store.snapshot()[0]
    assert store.read_topic(result["topic_ids"][0])["content"] == "主题里的项目经历"


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
def test_background_cannot_run_primary_remember_or_business_tools(store, name, arguments):
    before = store.snapshot()
    model = FakeModel([call(name, arguments), final("completed")])
    result = run_job(store, model)
    assert result["status"] == "failed"
    assert result["error_code"] == "tool_failed"
    assert store.snapshot() == before


@pytest.mark.parametrize("name,arguments", [("write_topic", new_topic()), ("write_core", {"content": "伪造来源"})])
def test_model_cannot_supply_source(store, name, arguments):
    """验证后台两种写入入口都不能接受模型伪造来源。

    作用:
        确认来源仅由已完成回合绑定，模型参数不能覆盖追溯身份。

    参数:
        store: 使用隔离目录的真实记忆存储。
        name: 后台写入工具名。
        arguments: 工具的合法业务参数。

    返回:
        None。

    行为说明:
        1. 额外提供伪造 source，核对工具失败且两个存储层均不变化。
    """
    before = store.snapshot()
    model = FakeModel([call(name, {**arguments, "source": {"message_id": "spoofed"}}), final("completed")])
    result = run_job(store, model)
    assert result["status"] == "failed"
    assert result["topic_ids"] == []
    assert result["core_additions"] == 0
    assert store.snapshot() == before
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
        (
            [call("write_core", {"content": "已落盘核心"}), RuntimeError("PRIVATE_RAW_ERROR")],
            "partial",
            "extraction_failed",
        ),
        ([call("write_core", {"content": "已落盘核心"}), final("conflict")], "partial", "memory_conflict"),
        (
            [call("write_core", {"content": "已落盘核心"}), call("remember", {"content": "no"}), final()],
            "partial",
            "tool_failed",
        ),
    ],
)
def test_noop_failure_and_partial_write_statuses(store, responses, status, code):
    model = FakeModel(responses)
    result = run_job(store, model)
    assert result["status"] == status
    assert result["error_code"] == code
    assert "PRIVATE_RAW_ERROR" not in (store.root / "extraction.jsonl").read_text()
    assert model.client.closed


@pytest.mark.parametrize("status", ["no_change", "completed", "conflict"])
@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "```\n{}\n```"])
def test_completion_accepts_only_supported_statuses_in_whole_json_or_fences(store, status, wrapper):
    """验证合法结束状态及整段单一代码围栏的兼容解析。

    作用:
        保持模型格式差异的有限容错，同时以实际工具写入判断保存结果。

    参数:
        store: 临时记忆目录中的真实存储。
        status: 三个允许的结束状态之一。
        wrapper: 原始 JSON、json 围栏或无语言围栏的完整包装。

    返回:
        None，由断言检查结束原因、正文长度和实际保存状态。

    行为说明:
        1. 完整运行一次离线提取，不调用写入工具。
        2. completed 文字声明不会被当成实际保存，冲突仍单独报告。
    """
    content = wrapper.format(json.dumps({"status": status}))
    result = run_job(store, FakeModel([{"content": content, "finish_reason": "stop"}]))
    assert result["status"] == ("conflict" if status == "conflict" else "no_change")
    assert result["error_code"] == ("memory_conflict" if status == "conflict" else None)
    assert result["completion_reason"] is None
    assert result["completion_chars"] == len(content)
    assert result["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "content,reason",
    [
        ("", "empty"),
        (" \n\t", "empty"),
        ("```json\n\n```", "empty"),
        ("PRIVATE_COMPLETION_TEXT", "invalid_json"),
        ('说明：{"status":"no_change"}', "invalid_json"),
        ('```json\n{"status":"no_change"}\n```\n说明', "invalid_json"),
        ('{"status":"no_change"}\n{"status":"completed"}', "invalid_json"),
        ('```json\n{"status":"no_change"}\n```\n```json\n{"status":"completed"}\n```', "invalid_json"),
        ('[{"status":"no_change"}]', "not_object"),
        ('"no_change"', "not_object"),
        ("null", "not_object"),
        ("{}", "invalid_status"),
        ('{"status":null}', "invalid_status"),
        ('{"status":[]}', "invalid_status"),
        ('{"status":{}}', "invalid_status"),
        ('{"status":true}', "invalid_status"),
        ('{"status":12}', "invalid_status"),
        ('{"status":"PRIVATE_INVALID_STATUS"}', "invalid_status"),
    ],
)
def test_invalid_completion_records_reason_without_private_content(store, content, reason):
    """验证不同无效结束格式均保留受控诊断且不泄露正文。

    作用:
        区分未返回内容、JSON 错误和状态错误，防止非字符串状态变成意外执行异常。

    参数:
        store: 临时记忆目录中的真实存储。
        content: 离线模型最终正文，覆盖无效包装与无效状态类型。
        reason: 预期的受控失败原因。

    返回:
        None，由断言检查失败分类、原文隔离和请求次数。

    行为说明:
        1. 一次模型响应后结束，不把格式错误降为无新增，也不自动重试。
        2. 审计只保留长度与受控原因，不记录模型提供的私人文字或状态值。
    """
    result = run_job(store, FakeModel([{"content": content, "finish_reason": "stop"}]))
    assert result["status"] == "failed"
    assert result["error_code"] == "invalid_completion"
    assert result["completion_reason"] == reason
    assert result["completion_chars"] == len(content)
    assert result["request_count"] == 1
    assert "PRIVATE_" not in (store.root / "extraction.jsonl").read_text()


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
@pytest.mark.parametrize("save_core", [False, True])
def test_truncated_completion_is_rejected_even_when_json_is_complete(store, finish_reason, save_core):
    """验证截断优先于合法 JSON，已写入内容仍报告部分完成。

    作用:
        防止输出长度上限碰巧落在完整 JSON 后就被误认成正常结束。

    参数:
        store: 临时记忆目录中的真实存储。
        finish_reason: 两种已知的长度上限结束原因。
        save_core: 截断前是否通过工具保存一条核心记忆。

    返回:
        None，由断言检查失败原因和实际写入保留。

    行为说明:
        1. 模拟最终正文可解析但供应商明确报告截断的响应。
        2. 已保存核心不回滚，没有写入时保持失败状态。
    """
    responses = [call("write_core", {"content": "测试核心记录"})] if save_core else []
    responses.append({"content": '{"status":"completed"}', "finish_reason": finish_reason})
    result = run_job(store, FakeModel(responses))
    assert result["status"] == ("partial" if save_core else "failed")
    assert result["error_code"] == "invalid_completion"
    assert result["completion_reason"] == "truncated"
    assert result["finish_reason"] == finish_reason
    assert result["core_additions"] == int(save_core)


@pytest.mark.parametrize(
    "finish_reason",
    ["stop", "end_turn", "tool_calls", "stop_sequence", "pause_turn", "content_filter", "refusal"],
)
def test_completion_audit_accepts_only_known_finish_reasons(store, finish_reason):
    """验证结束原因白名单只保存受控协议值。

    作用:
        保留诊断需要的供应商结束信号，不记录完整模型响应。

    参数:
        store: 临时记忆目录中的真实存储。
        finish_reason: 白名单内的非截断结束原因。

    返回:
        None，由断言检查审计中保存的受控结束原因。

    行为说明:
        1. 用离线模型直接验证提取器的元数据过滤；供应商适配器的协议校验由其独立测试覆盖。
    """
    result = run_job(store, FakeModel([{**final(), "finish_reason": finish_reason}]))
    assert result["finish_reason"] == finish_reason
    assert json.loads((store.root / "extraction.jsonl").read_text())["finish_reason"] == finish_reason


@pytest.mark.parametrize("finish_reason", [None, "PRIVATE_FINISH_REASON", [], {}])
def test_last_finish_reason_replaces_previous_value_and_extra_json_fields_stay_private(store, finish_reason):
    """验证未知末次原因覆盖旧值且额外 JSON 字段不会进入审计。

    作用:
        避免旧工具调用的结束原因冒充最终原因，或模型自由字段泄露私人正文。

    参数:
        store: 临时记忆目录中的真实存储。
        finish_reason: 缺失、未知字符串或非法类型的最终结束原因。

    返回:
        None，由断言检查真实写入、最后结束原因和审计内容。

    行为说明:
        1. 第一请求保存核心并带 tool_calls 原因，第二请求返回未知原因。
        2. 最终 JSON 的额外字段被忽略，不进入日志或完成事件。
    """
    first = {**call("write_core", {"content": "测试核心记录"}), "finish_reason": "tool_calls"}
    last = {"content": '{"status":"completed","detail":"PRIVATE_EXTRA_FIELD"}', "finish_reason": finish_reason}
    result = run_job(store, FakeModel([first, last]))
    assert result["status"] == "completed"
    assert result["finish_reason"] is None
    assert result["completion_reason"] is None
    assert "PRIVATE_" not in json.dumps(result)
    assert "PRIVATE_" not in (store.root / "extraction.jsonl").read_text()


def test_tool_failure_keeps_priority_over_invalid_completion_after_write(store):
    """验证工具失败优先于结束格式错误且已保存内容不丢失。

    作用:
        在新增格式诊断后保留已有工具故障和部分完成的状态语义。

    参数:
        store: 临时记忆目录中的真实存储。

    返回:
        None，由断言检查优先错误码、解析原因和写入数量。

    行为说明:
        1. 先合法写入，再调用不允许的工具，最终返回无效正文。
        2. 主要错误仍为 tool_failed，格式诊断作为独立元数据保留。
    """
    responses = [
        call("write_core", {"content": "测试核心记录"}),
        call("remember", {"content": "不应写入"}),
        {"content": "PRIVATE_INVALID_COMPLETION"},
    ]
    result = run_job(store, FakeModel(responses))
    assert result["status"] == "partial"
    assert result["core_additions"] == 1
    assert result["error_code"] == "tool_failed"
    assert result["completion_reason"] == "invalid_json"
    assert "PRIVATE_" not in (store.root / "extraction.jsonl").read_text()


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


@pytest.mark.parametrize("save_core", [False, True])
def test_log_failure_is_visible_to_main_thread(store, monkeypatch, save_core):
    """验证审计失败按实际核心落盘情况报告失败或部分完成。

    作用:
        避免日志不可写时把已保存的核心记忆误报为完全没有执行。

    参数:
        store: 使用隔离目录的真实记忆存储。
        monkeypatch: 临时替换审计写入。
        save_core: 是否在审计失败前实际保存核心。

    返回:
        None。

    行为说明:
        1. 模拟审计失败，检查前台事件和已落盘核心保持一致。
    """
    extractor = MemoryExtractor(store)

    def fail_record(_):
        raise OSError("private path")

    monkeypatch.setattr(extractor, "_record", fail_record)
    try:
        responses = (
            [call("write_core", {"content": "日志失败前已保存"}), final("completed")] if save_core else [final()]
        )
        extractor.submit(job(lambda: FakeModel(responses)))
        assert extractor.wait(3)
        event = extractor.drain_events()[0]
        assert event["status"] == ("partial" if save_core else "failed")
        assert event["core_additions"] == int(save_core)
        assert ("日志失败前已保存" in store.snapshot()[0]) == save_core
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
