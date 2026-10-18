"""验证文件记忆的持久化、增量维护与路径边界。"""

import json
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from kairos import memory
from kairos.memory import MemoryIndexUpdateError, MemoryStore
from kairos.memory_tools import (
    EXTRACTION_MEMORY_TOOL_SCHEMAS,
    PRIMARY_MEMORY_TOOL_SCHEMAS,
    WRITE_TOPIC_TOOL_SCHEMA,
)

SOURCE = {"session_id": "session-1", "turn_id": "turn-1", "message_id": "message-1"}


def test_initial_store_is_private_and_does_not_create_topics(tmp_path):
    store = MemoryStore(tmp_path / "memory")

    core, index = store.snapshot()
    assert core == "# 核心记忆\n"
    assert "暂无主题记忆" in index
    assert list(store.topics_dir.iterdir()) == []
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.topics_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.core_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.index_path.stat().st_mode) == 0o600


def test_remember_preserves_original_content_and_source_and_deduplicates(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    content = "  请保留原文。\n\n- 第一条\n- 第二条  "
    source = dict(SOURCE)

    first = store.remember(content, source=source)
    original_file = store.core_path.read_text()
    source["message_id"] = "mutated-after-saving"
    second = store.remember(content, source={**SOURCE, "message_id": "message-2"})

    assert first["status"] == "saved"
    assert first["source"] == SOURCE
    assert second["status"] == "duplicate"
    assert store.core_path.read_text() == original_file
    assert content in store.snapshot()[0]
    assert "kairos-memory" not in store.snapshot()[0]
    assert '"message_id": "message-1"' in original_file


def test_core_exact_check_does_not_claim_semantic_deduplication(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.remember("请使用中文回答。", source=SOURCE)

    result = store.remember("回答时请使用中文。", source=SOURCE)

    assert result["status"] == "saved"
    assert "请使用中文回答。" in store.snapshot()[0]
    assert "回答时请使用中文。" in store.snapshot()[0]


def test_topic_metadata_source_and_index_do_not_load_body(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    source = {
        **SOURCE,
        "context_start_message_id": "previous-message",
        "context_end_message_id": "last-message",
    }
    result = store.write_topic(None, "Kairos 学习背景", "安排学习内容时读取", "只有读正文才能发现的细节", source=source)

    topic = store.read_topic(result["topic_id"])
    core, index = store.snapshot()

    assert result["status"] == "created"
    assert topic["title"] == "Kairos 学习背景"
    assert topic["description"] == "安排学习内容时读取"
    assert topic["content"] == "只有读正文才能发现的细节"
    assert topic["sources"] == [source]
    assert topic["created_at"] == topic["updated_at"]
    assert "Kairos 学习背景" in index
    assert result["topic_id"] in index
    assert "安排学习内容时读取" in index
    assert "只有读正文才能发现的细节" not in core + index
    assert stat.S_IMODE((store.root / topic["path"]).stat().st_mode) == 0o600


def test_same_topic_appends_and_keeps_sources_without_replacing_body(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    created = store.write_topic(None, "学习", "继续学习时读取", "已经学习 Python。", source=SOURCE)
    topic_id = created["topic_id"]
    second_source = {**SOURCE, "turn_id": "turn-2", "message_id": "message-2"}

    result = store.write_topic(topic_id, None, None, "下一阶段学习 Agent。", source=second_source)
    topic = store.read_topic(topic_id)
    before_duplicate = (store.root / topic["path"]).read_bytes()
    duplicate = store.write_topic(topic_id, None, None, "下一阶段学习 Agent。", source=SOURCE)

    assert result["status"] == "appended"
    assert topic["content"] == "已经学习 Python。\n\n下一阶段学习 Agent。"
    assert topic["sources"] == [SOURCE, second_source]
    assert topic["updated_at"] >= topic["created_at"]
    assert duplicate["status"] == "duplicate"
    assert (store.root / topic["path"]).read_bytes() == before_duplicate
    assert len(list(store.topics_dir.glob("*.md"))) == 1


def test_duplicate_title_returns_existing_id_without_adding_content(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    first = store.write_topic(None, "Kairos", "讨论时读取", "第一份内容", source=SOURCE)

    second = store.write_topic(None, "kairos", "不同描述", "不能未读就追加的内容", source=SOURCE)

    assert second["status"] == "existing"
    assert second["topic_id"] == first["topic_id"]
    assert store.read_topic(first["topic_id"])["content"] == "第一份内容"
    assert len(list(store.topics_dir.glob("*.md"))) == 1


@pytest.mark.parametrize("title,description", [("新标题", None), (None, "新描述")])
def test_append_does_not_rename_or_rewrite_topic_metadata(tmp_path, title, description):
    store = MemoryStore(tmp_path / "memory")
    result = store.write_topic(None, "旧标题", "旧描述", "旧正文", source=SOURCE)

    with pytest.raises(ValueError, match="不修改已有主题"):
        store.write_topic(result["topic_id"], title, description, "新正文", source=SOURCE)

    assert store.read_topic(result["topic_id"])["content"] == "旧正文"


def test_instances_reread_core_and_topics_and_repair_index(tmp_path):
    first = MemoryStore(tmp_path / "memory")
    second = MemoryStore(tmp_path / "memory")

    first.remember("使用中文。", source=SOURCE)
    second.remember("先说明整体。", source=SOURCE)
    created = second.write_topic(None, "某主题", "相关时读取", "正文", source=SOURCE)
    first.index_path.write_text("过期索引", encoding="utf-8")

    core, index = first.snapshot()
    assert "使用中文。" in core and "先说明整体。" in core
    assert created["topic_id"] in index
    assert first.index_path.read_text() == index
    assert first.read_topic(created["topic_id"])["content"] == "正文"


def test_index_has_stable_order_and_escapes_metadata(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.write_topic(None, "Z 主题", "第一行\n- 伪造索引", "正文 z", source=SOURCE)
    store.write_topic(None, "A [主题]", "正常用途", "正文 a", source=SOURCE)

    _, index = store.snapshot()

    assert index.index("A \\[主题\\]") < index.index("Z 主题")
    assert "\n- 伪造索引" not in index
    assert store.snapshot()[1] == index


def test_model_title_cannot_escape_flat_topics_directory(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    result = store.write_topic(None, "../../中文/../CORE.md", "用途", "主题正文", source=SOURCE)
    topic = store.read_topic(result["topic_id"])

    path = store.root / topic["path"]
    assert path.parent == store.topics_dir
    assert "中文" in path.name
    assert "主题正文" not in store.snapshot()[0]
    assert len(list(store.topics_dir.iterdir())) == 1


def test_short_id_filename_collision_does_not_overwrite_other_topic(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory")
    identifiers = iter([UUID("12345678-1111-4111-8111-111111111111"), UUID("12345678-2222-4222-8222-222222222222")])
    monkeypatch.setattr(memory, "uuid4", lambda: next(identifiers))

    first = store.write_topic(None, "A B", "用途一", "正文一", source=SOURCE)
    second = store.write_topic(None, "A/B", "用途二", "正文二", source=SOURCE)

    assert first["path"] != second["path"]
    assert store.read_topic(first["topic_id"])["content"] == "正文一"
    assert store.read_topic(second["topic_id"])["content"] == "正文二"


@pytest.mark.parametrize("topic_id", ["../CORE.md", "/tmp/secret", "not-a-uuid", "", None, 123])
def test_read_rejects_paths_and_non_ids(tmp_path, topic_id):
    store = MemoryStore(tmp_path / "memory")
    with pytest.raises(ValueError, match="topic_id"):
        store.read_topic(topic_id)


def test_unknown_uuid_does_not_create_or_read_anything(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    unknown = str(uuid4())

    with pytest.raises(KeyError, match="主题不存在"):
        store.read_topic(unknown)
    with pytest.raises(KeyError, match="主题不存在"):
        store.write_topic(unknown, None, None, "内容", source=SOURCE)

    assert list(store.topics_dir.iterdir()) == []


@pytest.mark.parametrize("location", ["root", "topics", "core", "index", "topic"])
def test_symlink_targets_are_never_read_or_written(tmp_path, location):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "secret.md"
    sentinel.write_text("不得读取或修改的内容", encoding="utf-8")
    root = tmp_path / "memory"
    if location == "root":
        root.symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="符号链接"):
            MemoryStore(root)
    else:
        store = MemoryStore(root)
        if location == "topics":
            store.topics_dir.rmdir()
            store.topics_dir.symlink_to(outside, target_is_directory=True)
        elif location == "core":
            store.core_path.unlink()
            store.core_path.symlink_to(sentinel)
        elif location == "index":
            store.index_path.unlink()
            store.index_path.symlink_to(sentinel)
        else:
            (store.topics_dir / "malicious.md").symlink_to(sentinel)
        with pytest.raises(ValueError, match="符号链接"):
            store.snapshot()

    assert sentinel.read_text() == "不得读取或修改的内容"
    assert list(outside.iterdir()) == [sentinel]


def test_broken_symlink_is_rejected_during_initialization(tmp_path):
    root = tmp_path / "memory"
    root.mkdir()
    (root / "CORE.md").symlink_to(tmp_path / "nonexistent")

    with pytest.raises(ValueError, match="符号链接"):
        MemoryStore(root)

    assert not (tmp_path / "nonexistent").exists()


def test_failed_atomic_replace_keeps_core_unchanged_and_cleans_temp(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory")
    store.remember("原有内容", source=SOURCE)
    original = store.core_path.read_bytes()

    def fail_replace(source, target):
        raise OSError("模拟替换失败")

    monkeypatch.setattr(memory.os, "replace", fail_replace)

    with pytest.raises(OSError, match="模拟替换失败"):
        store.remember("不能丢掉旧内容", source=SOURCE)

    assert store.core_path.read_bytes() == original
    assert list(store.root.glob(".memory-*.tmp")) == []


def test_failed_index_write_can_be_rebuilt_from_saved_topic(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory")
    replace = memory.os.replace

    def fail_index_replace(source, target):
        if Path(target) == store.index_path:
            raise OSError("索引保存失败")
        replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(memory.os, "replace", fail_index_replace)
        with pytest.raises(MemoryIndexUpdateError, match="主题正文已保存") as captured:
            store.write_topic(None, "已保存主题", "读取用途", "正文仍保存", source=SOURCE)

    assert captured.value.status == "created"
    assert captured.value.topic_id
    assert store.read_topic(captured.value.topic_id)["content"] == "正文仍保存"
    assert str(captured.value.__cause__) == "索引保存失败"
    assert "正文仍保存" not in str(captured.value)
    assert "索引保存失败" not in str(captured.value)
    _, index = store.snapshot()
    assert "已保存主题" in index
    assert len(list(store.topics_dir.glob("*.md"))) == 1
    assert store.write_topic(None, "已保存主题", "读取用途", "正文仍保存", source=SOURCE)["status"] == "existing"


def test_failed_index_rebuild_reports_saved_append_and_preserves_old_body(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory")
    created = store.write_topic(None, "学习背景", "讨论学习时使用", "已有 Python 经验。", source=SOURCE)
    rebuild = store._rebuild_index

    def fail_index_rebuild(topics):
        raise RuntimeError("PRIVATE_INDEX_FAILURE")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_rebuild_index", fail_index_rebuild)
        with pytest.raises(MemoryIndexUpdateError) as captured:
            store.write_topic(created["topic_id"], None, None, "开始学习 Rust。", source=SOURCE)

    assert captured.value.topic_id == created["topic_id"]
    assert captured.value.status == "appended"
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "PRIVATE_INDEX_FAILURE" not in str(captured.value)
    assert store.read_topic(created["topic_id"])["content"] == "已有 Python 经验。\n\n开始学习 Rust。"
    assert created["topic_id"] in store.snapshot()[1]
    assert store._rebuild_index == rebuild


def test_duplicate_index_failure_does_not_claim_a_new_write(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory")
    created = store.write_topic(None, "主题", "用途", "相同正文", source=SOURCE)
    before = (store.root / created["path"]).read_bytes()

    def fail_index_rebuild(topics):
        raise OSError("索引暂不可用")

    monkeypatch.setattr(store, "_rebuild_index", fail_index_rebuild)
    with pytest.raises(OSError, match="索引暂不可用") as captured:
        store.write_topic(created["topic_id"], None, None, "相同正文", source=SOURCE)

    assert not isinstance(captured.value, MemoryIndexUpdateError)
    assert (store.root / created["path"]).read_bytes() == before


def test_invalid_source_fails_before_writing(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    initial = store.core_path.read_bytes()

    with pytest.raises(ValueError, match="来源"):
        store.remember("没有来源的内容", source={"session_id": "only-session"})
    with pytest.raises(ValueError, match="来源"):
        store.write_topic(None, "主题", "用途", "内容", source={**SOURCE, "message_id": ""})

    assert store.core_path.read_bytes() == initial
    assert list(store.topics_dir.iterdir()) == []


@pytest.mark.parametrize("kind", ["core", "topic"])
def test_corrupt_metadata_is_not_silently_overwritten(tmp_path, kind):
    store = MemoryStore(tmp_path / "memory")
    if kind == "core":
        path = store.core_path
    else:
        result = store.write_topic(None, "主题", "用途", "内容", source=SOURCE)
        path = store.root / result["path"]
    path.write_text("手工写入的未知格式", encoding="utf-8")

    with pytest.raises(ValueError, match="元数据"):
        store.snapshot()

    assert path.read_text() == "手工写入的未知格式"


def test_metadata_comment_closer_does_not_break_parser(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    result = store.write_topic(None, "包含 --> 的标题", "用途 --> 后半段", "正文 <!-- 示例 -->", source=SOURCE)

    assert store.read_topic(result["topic_id"])["content"] == "正文 <!-- 示例 -->"
    assert store.read_topic(result["topic_id"])["title"] == "包含 --> 的标题"


def test_in_process_parallel_appends_do_not_lose_entries(tmp_path):
    store = MemoryStore(tmp_path / "memory")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(store.remember, f"第 {index} 条", source=SOURCE) for index in range(12)]
        assert all(future.result()["status"] == "saved" for future in futures)

    core, _ = store.snapshot()
    assert all(core.count(f"第 {index} 条") == 1 for index in range(12))


def test_tool_sets_separate_primary_and_extraction_permissions():
    primary = {tool["function"]["name"] for tool in PRIMARY_MEMORY_TOOL_SCHEMAS}
    extraction = {tool["function"]["name"] for tool in EXTRACTION_MEMORY_TOOL_SCHEMAS}

    assert primary == {"remember", "read_memory"}
    assert extraction == {"read_memory", "write_topic"}
    parameters = WRITE_TOPIC_TOOL_SCHEMA["function"]["parameters"]
    assert set(parameters["required"]) == {"topic_id", "title", "description", "addition"}
    assert parameters["additionalProperties"] is False
    assert "source" not in json.dumps(PRIMARY_MEMORY_TOOL_SCHEMAS + EXTRACTION_MEMORY_TOOL_SCHEMAS)


def test_memory_file_io_without_platform_nofollow_flag(tmp_path, monkeypatch):
    """验证缺少 POSIX 专用标志的平台仍能使用普通记忆文件。

    作用:
        保持用户目录布局在其他系统上的基本读写能力。

    参数:
        tmp_path: 隔离目录。
        monkeypatch: 临时模拟没有 O_NOFOLLOW 的平台。

    返回:
        None。

    行为说明:
        1. 保留普通路径校验，验证核心保存和主题读取不依赖专用标志。
    """
    monkeypatch.delattr(memory.os, "O_NOFOLLOW", raising=False)
    store = MemoryStore(tmp_path / "memory")
    store.remember("跨平台核心", source=SOURCE)
    saved = store.write_topic(None, "跨平台主题", "普通文件读取", "正文", source=SOURCE)
    assert "跨平台核心" in store.snapshot()[0]
    assert store.read_topic(saved["topic_id"])["content"] == "正文"
