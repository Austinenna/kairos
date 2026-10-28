"""共享服务：所有会话共用的东西都在这里创建一次。

Services（共享）                   Conversation（每个会话一份）
  配置、数据库、会话存储              会话信息、历史消息、模型
  工具集合、记忆存储、后台提取器
  图片归档、后台分类、定时任务      文件工具（记着这个会话读过哪些文件）
  文件路径规则
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from .config import Settings, get_workspace_dir
from .files.policy import FilePolicy
from .images.archive import ImageArchive
from .images.classifier import ImageClassifier
from .memory.extractor import MemoryExtractor
from .memory.store import MemoryStore
from .models import build_model
from .schedule.store import ScheduleStore
from .storage import SessionStore, open_db
from .tools import Toolset, load_factories
from .tools.history import history_tools
from .tools.images import image_tools
from .tools.memory import primary_memory_tools
from .tools.schedules import schedule_tools


@dataclass
class Services:
    settings: Settings
    db: sqlite3.Connection
    sessions: SessionStore
    tools: Toolset
    memory: MemoryStore
    extractor: MemoryExtractor | None  # None 表示不做后台记忆提取（测试里常用）
    images: ImageArchive
    classifier: ImageClassifier  # 后台给图片分类，用自己的数据库连接
    schedules: ScheduleStore
    file_policy: FilePolicy  # 文件工具能读写哪些路径
    # 入口名 → 推送函数。入口启动时登记，定时任务到点后用它推送，并追加进该入口的当前会话。
    outlets: dict[str, Callable[[str], None]] = field(default_factory=dict)
    # 终端和微信在不同线程里运行。处理一条输入时持有这把锁，两边轮流使用数据库。
    lock: RLock = field(default_factory=RLock)

    def close(self) -> None:
        """退出前调用：等后台记忆提取和图片分类做完，再关闭数据库。"""
        if self.extractor:
            self.extractor.close()
        self.classifier.close()
        self.db.close()


def open_services(settings: Settings, data_dir: Path, *, auto_memory: bool = True) -> Services:
    """创建全部共享服务。数据都放在 data_dir（正常运行时是 ~/.kairos）。"""
    db_path = data_dir / "kairos.db"
    db = open_db(db_path)
    memory = MemoryStore(data_dir / "memories")
    tools = load_factories(settings.kairos_tool_factories, db)  # 业务工具，默认是待办
    tools.add(primary_memory_tools(memory))
    tools.add(history_tools(db))
    schedules = ScheduleStore(db)
    tools.add(schedule_tools(schedules))
    images = ImageArchive(db, data_dir / "files", lambda: build_model(settings))
    # 后台任务带上核心记忆，用户的长期指令在哪里都生效。没有 read_memory，所以不给主题索引。
    classifier = ImageClassifier(db_path, data_dir / "files", lambda: build_model(settings), memory.core)
    tools.add(image_tools(images))
    extractor = MemoryExtractor(memory, lambda: build_model(settings)) if auto_memory else None
    file_policy = FilePolicy(home=Path.home(), workspace=get_workspace_dir())
    return Services(
        settings, db, SessionStore(db), tools, memory, extractor, images, classifier, schedules, file_policy
    )
