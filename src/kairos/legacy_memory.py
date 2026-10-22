"""以 Markdown 保存全局核心和主题记忆，并生成按需读取的完整根索引。

文件头中的 JSON 注释保存程序元数据，正文保留 Markdown 原文。
主 Agent 只能追加 CORE，后台提取器只能追加主题；本期不提供覆盖、删除或精炼。
"""

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid4

_HEADER = "<!-- kairos-memory\n"
_HEADER_END = "\n-->\n\n"
_SOURCE_KEYS = {"session_id", "turn_id", "message_id"}


class MemoryIndexUpdateError(OSError):
    """表示主题正文已保存，但派生索引尚未同步。"""

    def __init__(self, topic_id: str, status: str):
        """保存已完成写入的身份，供调用方如实记录部分成功。

        作用:
            让后台提取器区分正文完全未保存与仅索引更新失败，不暴露私人内容。

        参数:
            topic_id: 已持久化的主题 UUID。
            status: 已完成的正文操作，created 或 appended。

        返回:
            无。

        行为说明:
            1. 对外消息固定；底层错误由抛出位置通过异常因果链保留。
            2. 下次快照会依据主题主文件再次尝试重建索引。
        """
        self.topic_id = topic_id
        self.status = status
        super().__init__("主题正文已保存，但记忆索引更新失败；下次加载将重建索引。")


def _nonempty(value: str, name: str) -> str:
    """校验记忆文本并保留原文。

    作用:
        在写入前拒绝空内容，不改变用户明确要求保留的空白或换行。

    参数:
        value: 待校验的字符串。
        name: 用于错误提示的字段名。

    返回:
        未修改的字符串（str）。

    异常:
        ValueError: 参数不是非空字符串。

    行为说明:
        1. strip 仅用于检查，不用于保存。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _validate_source(source: dict[str, str]) -> dict[str, str]:
    """校验并复制运行时绑定的来源。

    作用:
        在磁盘操作前确保记忆关联实际会话、回合和用户消息。

    参数:
        source: 至少包含 session_id、turn_id、message_id 的字符串字典。

    返回:
        与调用者独立的来源副本（dict[str, str]）。

    异常:
        ValueError: 来源不是字典、缺少必要字段，或存在非空字符串以外的键值。

    行为说明:
        1. 来源由运行时传入，不暴露给模型工具参数。
    """
    if (
        not isinstance(source, dict)
        or not _SOURCE_KEYS.issubset(source)
        or any(not isinstance(key, str) or not key.strip() for key in source)
        or any(not isinstance(value, str) or not value.strip() for value in source.values())
    ):
        raise ValueError("记忆来源必须包含非空的 session_id、turn_id 和 message_id")
    return dict(source)


def _validate_topic_id(topic_id: str) -> str:
    """校验稳定主题 ID，拒绝模型提供文件路径。

    作用:
        在主题查找前限定 UUID 格式，避免把外部输入拼接为任意路径。

    参数:
        topic_id: 索引提供的规范 UUID 字符串。

    返回:
        原始合法 ID（str）。

    异常:
        ValueError: ID 不是字符串、不是 UUID，或不是规范格式。

    行为说明:
        1. 文件位置只由扫描到的有效主题元数据决定。
    """
    if not isinstance(topic_id, str):
        raise ValueError("topic_id 必须是索引中的 UUID")
    try:
        parsed = UUID(topic_id)
    except ValueError:
        raise ValueError("topic_id 必须是索引中的 UUID，不能是路径") from None
    if str(parsed) != topic_id:
        raise ValueError("topic_id 必须使用索引中的规范 UUID 格式")
    return topic_id


def _check_file(path: Path) -> None:
    """检查记忆目标是否为普通文件或尚未创建。

    作用:
        在读写前阻止符号链接、目录和特殊设备进入记忆存储。

    参数:
        path: 由存储程序确定的文件路径。

    返回:
        无。

    异常:
        ValueError: 已存在的路径不是普通文件，包括符号链接。

    行为说明:
        1. lstat 不跟随链接，断开的链接也会被拒绝。
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise ValueError(f"记忆路径必须是普通文件，不能使用符号链接：{path.name}")


def _read_text(path: Path) -> str:
    """读取普通记忆文件，不跟随文件符号链接。

    作用:
        为跨实例最新读取提供统一入口。

    参数:
        path: 已经由存储程序确定的文件路径。

    返回:
        完整 UTF-8 文本（str）。

    行为说明:
        1. 先检查文件类型，再使用 O_NOFOLLOW 打开，防止末级链接被读取。
        2. 文件检查和标准库的读写错误交由调用者处理。
    """
    _check_file(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, encoding="utf-8") as stream:
        return stream.read()


def _atomic_write(path: Path, content: str) -> None:
    """以私有临时文件原子替换一份记忆文件。

    作用:
        避免写入中断时截断旧内容；替换前失败时保留旧文件。

    参数:
        path: CORE、程序索引或主题文件路径。
        content: 准备保存的完整文本。

    返回:
        无。

    行为说明:
        1. 在目标目录创建权限为 0600 的临时文件，写入并同步。
        2. 再检查目标文件，使用 replace 完成单文件原子切换。
        3. 无论成功或失败都清理临时文件；不提供跨进程事务。
    """
    _check_file(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".memory-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _check_file(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _encode(metadata: dict[str, Any], content: str) -> str:
    """组合程序元数据与原始 Markdown 正文。

    作用:
        将来源及增量校验值保存在同一主文件中，不另建第二份正文。

    参数:
        metadata: 可序列化的文件头字段。
        content: 不作精炼的正文。

    返回:
        可保存的完整 Markdown 文件（str）。

    行为说明:
        1. 转义元数据中的右尖括号，避免值提前关闭 HTML 注释。
    """
    header = json.dumps(metadata, ensure_ascii=False, sort_keys=True).replace(">", "\\u003e")
    return f"{_HEADER}{header}{_HEADER_END}{content}"


def _decode(path: Path, kind: str) -> tuple[dict[str, Any], str]:
    """解析并校验程序维护的记忆文件。

    作用:
        在读取和增量修改前识别损坏的元数据，避免覆盖未知内容。

    参数:
        path: 存储目录内的普通文件。
        kind: 预期类型，core 或 topic。

    返回:
        元数据及完整正文（tuple[dict[str, Any], str]）。

    异常:
        ValueError: 文件头缺失、JSON 无效、类型或版本错误，或增量记录无效。

    行为说明:
        1. 只解析首个程序文件头，正文中的同名文本保持原样。
    """
    text = _read_text(path)
    if not text.startswith(_HEADER) or _HEADER_END not in text:
        raise ValueError(f"记忆文件缺少有效元数据：{path.name}")
    header, content = text[len(_HEADER) :].split(_HEADER_END, 1)
    try:
        metadata = json.loads(header)
    except json.JSONDecodeError:
        raise ValueError(f"记忆文件元数据不是有效 JSON：{path.name}") from None
    if not isinstance(metadata, dict) or metadata.get("version") != 1 or metadata.get("kind") != kind:
        raise ValueError(f"记忆文件版本或类型无效：{path.name}")
    additions = metadata.get("additions")
    if not isinstance(additions, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("digest"), str)
        or not isinstance(item.get("created_at"), str)
        or not isinstance(item.get("source"), dict)
        for item in additions
    ):
        raise ValueError(f"记忆文件增量记录无效：{path.name}")
    for item in additions:
        _validate_source(item["source"])
    return metadata, content


def _addition_record(content: str, source: dict[str, str]) -> dict[str, Any]:
    """为一次追加生成精确校验值和可信来源记录。

    作用:
        防止相同文本被重试追加，同时保留每次新增的依据。

    参数:
        content: 原始新增内容。
        source: 已校验的来源副本。

    返回:
        包含 digest、source 和 created_at 的字典（dict[str, Any]）。

    行为说明:
        1. 校验值只防完全相同的追加，语义重复仍由模型判断。
    """
    return {
        "digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "source": source,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _index_text(value: str) -> str:
    """将标题和用途转成不会破坏索引结构的单行文本。

    作用:
        避免元数据中的换行或 Markdown 标记产生额外索引条目。

    参数:
        value: 标题或用途描述。

    返回:
        转义后的单行文本（str）。

    行为说明:
        1. 正文不经过此处理，只有索引展示字段会折叠空白。
    """
    return re.sub(r"([\\`*_{}\[\]<>])", r"\\\1", " ".join(value.split()))


class MemoryStore:
    """保存一套全局记忆，使用实例内锁串行协调核心、主题和索引操作。"""

    def __init__(self, root: Path):
        """建立私有记忆目录并初始化空核心和索引。

        作用:
            由运行时以用户记忆路径创建，测试则传入临时目录。

        参数:
            root: 记忆根目录，不接受已有符号链接。

        返回:
            无。

        行为说明:
            1. 不创建预设主题或分类，只初始化 CORE、MEMORY 和 topics。
            2. 已有正文保持原样，索引从有效主题元数据重建。
        """
        self.root = Path(root).expanduser().absolute()
        self.topics_dir = self.root / "topics"
        self.core_path = self.root / "CORE.md"
        self.index_path = self.root / "MEMORY.md"
        self._lock = RLock()
        self._prepare_directories()
        with self._lock:
            _check_file(self.core_path)
            if not self.core_path.exists():
                _atomic_write(self.core_path, _encode({"version": 1, "kind": "core", "additions": []}, "# 核心记忆\n"))
            _decode(self.core_path, "core")
            self._rebuild_index(self._topics())

    def _prepare_directories(self) -> None:
        """创建并检查记忆专用目录。

        作用:
            为初始化提供私有目录，并在每次访问时阻止目录被替换为链接。

        参数:
            无。

        返回:
            无。

        异常:
            ValueError: 根目录或 topics 为符号链接，或不是目录。

        行为说明:
            1. 仅调整记忆根目录和 topics 的权限，不修改祖先目录。
        """
        for directory in (self.root, self.topics_dir):
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                raise ValueError(f"记忆目录不能是符号链接或普通文件：{directory.name}")
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory.chmod(0o700)

    def _topics(self) -> list[dict[str, Any]]:
        """重新读取平铺主题，避免长期缓存隐藏其他实例写入。

        作用:
            为读取、去重和索引提供同一份最新主题集合。

        参数:
            无。

        返回:
            按标题、ID 确定排序的完整主题列表（list[dict[str, Any]]）。

        异常:
            ValueError: 主题缺少必要字段，或多个文件使用同一 ID。

        行为说明:
            1. 只扫描 topics 直属 Markdown 文件，不递归子目录。
            2. 读取前校验元数据，不沿符号链接访问其他位置。
        """
        self._prepare_directories()
        topics = []
        known_ids = set()
        for path in sorted(self.topics_dir.glob("*.md")):
            metadata, content = _decode(path, "topic")
            topic_id = _validate_topic_id(metadata.get("topic_id"))
            for field in ("title", "description", "created_at", "updated_at"):
                if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                    raise ValueError(f"主题缺少有效的 {field}：{path.name}")
            if topic_id in known_ids:
                raise ValueError(f"多个记忆文件使用同一个主题 ID：{topic_id}")
            known_ids.add(topic_id)
            topics.append({**metadata, "content": content, "path": f"topics/{path.name}"})
        return sorted(topics, key=lambda item: (item["title"].casefold(), item["topic_id"]))

    def _rebuild_index(self, topics: list[dict[str, Any]]) -> str:
        """从主题元数据生成完整且可重建的根索引。

        作用:
            统一维护所有主题入口，索引只描述用途、不复制正文。

        参数:
            topics: 按确定顺序排列的主题记录。

        返回:
            与磁盘同步的完整索引（str）。

        行为说明:
            1. 内容相同时不重写；磁盘索引缺失或过期时自动恢复。
            2. 主题文件是主数据，索引生成失败不会删除主题正文。
        """
        lines = ["# 记忆索引", "", "由程序生成；根据用途选择主题 ID，再调用 read_memory 读取正文。", ""]
        if not topics:
            lines.append("暂无主题记忆。")
        for topic in topics:
            lines.extend(
                [
                    f"- [{_index_text(topic['title'])}]({quote(topic['path'], safe='/')})",
                    f"  - ID：`{topic['topic_id']}`",
                    f"  - 用途：{_index_text(topic['description'])}",
                ]
            )
        content = "\n".join(lines) + "\n"
        _check_file(self.index_path)
        if not self.index_path.exists() or _read_text(self.index_path) != content:
            _atomic_write(self.index_path, content)
        return content

    def snapshot(self) -> tuple[str, str]:
        """读取最新核心正文和完整根索引。

        作用:
            在每次模型请求前提供常驻记忆，不把主题正文提前放进上下文。

        参数:
            无。

        返回:
            核心正文及完整索引（tuple[str, str]），不包含内部 JSON 文件头。

        行为说明:
            1. 每次从主文件重读，保证实例外的已完成写入也能被看到。
            2. 根索引从主题重建，不静默截断。
        """
        with self._lock:
            self._prepare_directories()
            _, core = _decode(self.core_path, "core")
            return core, self._rebuild_index(self._topics())

    def remember(self, content: str, *, source: dict[str, str]) -> dict[str, Any]:
        """将已选定的新增内容追加到核心记忆。

        作用:
            供前台显式 remember 与后台 write_core 共用，不覆盖旧事实。

        参数:
            content: 用户明确要求保存的内容，或后台筛选出的常驻信息。
            source: 运行时绑定的真实用户消息来源。

        返回:
            status 为 saved 或 duplicate，附带内容及来源（dict[str, Any]）。

        行为说明:
            1. 调用方负责显式或自动筛选及语义查重；程序只防完全相同文本再次追加。
            2. 每次重读 CORE，追加成功后下一次 snapshot 即可看到。
        """
        content = _nonempty(content, "content")
        source = _validate_source(source)
        record = _addition_record(content, source)
        with self._lock:
            self._prepare_directories()
            metadata, body = _decode(self.core_path, "core")
            if any(item["digest"] == record["digest"] for item in metadata["additions"]):
                return {"status": "duplicate", "content": content, "source": source}
            metadata["additions"].append(record)
            _atomic_write(self.core_path, _encode(metadata, body + "\n" + content + "\n"))
            return {"status": "saved", "content": content, "source": source}

    def read_topic(self, topic_id: str) -> dict[str, Any]:
        """按稳定 ID 读取一个主题的完整内容。

        作用:
            供主 Agent 按需回忆，也供后台提取器先读后做增量去重。

        参数:
            topic_id: 根索引中的 UUID，不能是任意路径。

        返回:
            主题 ID、标题、用途、正文、来源、时间及相对路径（dict[str, Any]）。

        异常:
            KeyError: 存储中没有该主题 ID。

        行为说明:
            1. 每次重新读取，不复用过期的正文缓存。
        """
        topic_id = _validate_topic_id(topic_id)
        with self._lock:
            for topic in self._topics():
                if topic["topic_id"] == topic_id:
                    return self._public_topic(topic)
        raise KeyError(f"主题不存在：{topic_id}")

    @staticmethod
    def _public_topic(topic: dict[str, Any]) -> dict[str, Any]:
        """将内部文件字段转为工具可读的主题结果。

        作用:
            返回业务内容与来源，隐藏内部校验值和序列化字段。

        参数:
            topic: 完整内部主题记录。

        返回:
            可序列化的主题字典（dict[str, Any]）。

        行为说明:
            1. sources 按追加顺序保留，不把多个来源折叠为一个。
        """
        return {
            key: topic[key]
            for key in ("topic_id", "title", "description", "content", "created_at", "updated_at", "path")
        } | {"sources": [dict(record["source"]) for record in topic["additions"]]}

    def write_topic(
        self,
        topic_id: str | None,
        title: str | None,
        description: str | None,
        addition: str,
        *,
        source: dict[str, str],
    ) -> dict[str, Any]:
        """创建一个主题或向已有主题追加新信息。

        作用:
            由后台提取工具调用，正文和元数据成功保存后重建完整索引。

        参数:
            topic_id: None 创建主题；已有 UUID 表示追加。
            title: 创建时必填，追加时为 None 或与原标题相同。
            description: 创建时必填，追加时为 None 或与原描述相同。
            addition: 本次新增的正文，不是整篇替换内容。
            source: 运行时固定的回合和消息来源。

        返回:
            状态及主题 ID、标题、路径（dict[str, Any]）。
            同标题创建返回 existing，要求先读取已有主题再决定增量内容。

        异常:
            KeyError: 指定的主题 ID 不存在。
            ValueError: 追加请求试图修改原标题或用途描述，或新主题 ID、文件名发生冲突。
            MemoryIndexUpdateError: 本次主题正文已保存，但随后重建索引失败。
            捕获的原异常（显式重新抛出）: 未新增正文的重复请求遇到索引重建失败。

        行为说明:
            1. 新主题使用程序生成的 UUID 和安全文件名，保持平铺。
            2. 已有主题只追加，精确重复跳过；不提供删除或覆盖接口。
            3. 索引是派生数据；索引写入失败后仍可从已保存主题恢复。
        """
        addition = _nonempty(addition, "addition")
        source = _validate_source(source)
        record = _addition_record(addition, source)
        if topic_id is None:
            title = _nonempty(title, "title").strip()
            description = _nonempty(description, "description").strip()
        else:
            topic_id = _validate_topic_id(topic_id)
        with self._lock:
            topics = self._topics()
            if topic_id is None:
                existing = next((item for item in topics if item["title"].casefold() == title.casefold()), None)
                if existing is not None:
                    self._rebuild_index(topics)
                    return {
                        "status": "existing",
                        "topic_id": existing["topic_id"],
                        "title": existing["title"],
                        "path": existing["path"],
                        "message": "已有同名主题；先调用 read_memory，确认不重复后向该 ID 追加。",
                    }
                topic_id = str(uuid4())
                if any(item["topic_id"] == topic_id for item in topics):
                    raise ValueError("生成的新主题 ID 与已有主题冲突，请重试")
                safe_title = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", title, flags=re.UNICODE).strip("-_")[:48]
                filename = f"{safe_title or '主题'}-{topic_id[:8]}.md"
                if (self.topics_dir / filename).exists() or (self.topics_dir / filename).is_symlink():
                    filename = f"{safe_title or '主题'}-{topic_id}.md"
                if (self.topics_dir / filename).exists() or (self.topics_dir / filename).is_symlink():
                    raise ValueError("生成的新主题文件名与已有文件冲突，请重试")
                topic = {
                    "version": 1,
                    "kind": "topic",
                    "topic_id": topic_id,
                    "title": title,
                    "description": description,
                    "created_at": record["created_at"],
                    "updated_at": record["created_at"],
                    "additions": [record],
                    "content": addition,
                    "path": f"topics/{filename}",
                }
                topics.append(topic)
                status = "created"
            else:
                topic = next((item for item in topics if item["topic_id"] == topic_id), None)
                if topic is None:
                    raise KeyError(f"主题不存在：{topic_id}")
                if (title is not None and title != topic["title"]) or (
                    description is not None and description != topic["description"]
                ):
                    raise ValueError("第一期只追加正文，不修改已有主题标题或用途描述")
                status = "duplicate"
                if not any(item["digest"] == record["digest"] for item in topic["additions"]):
                    topic["content"] += "\n\n" + addition
                    topic["additions"].append(record)
                    topic["updated_at"] = record["created_at"]
                    status = "appended"
            if status != "duplicate":
                metadata = {key: value for key, value in topic.items() if key not in {"content", "path"}}
                _atomic_write(self.root / topic["path"], _encode(metadata, topic["content"]))
            topics.sort(key=lambda item: (item["title"].casefold(), item["topic_id"]))
            try:
                self._rebuild_index(topics)
            except Exception as error:
                if status in {"created", "appended"}:
                    raise MemoryIndexUpdateError(topic_id, status) from error
                raise
            return {"status": status, "topic_id": topic_id, "title": topic["title"], "path": topic["path"]}
