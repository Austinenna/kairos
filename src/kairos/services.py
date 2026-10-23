"""共享服务：所有会话共用的东西都在这里创建一次。

Services（共享）                   Conversation（每个会话一份）
  配置、数据库、会话存储              会话信息、历史消息、模型
  工具集合、记忆存储、后台提取器
  图片归档
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from .archive import ImageArchive
from .config import Settings
from .memory.extractor import MemoryExtractor
from .memory.store import MemoryStore
from .models import build_model
from .storage import SessionStore, open_db
from .tools import Toolset, load_factories
from .tools.history import history_tools
from .tools.images import image_tools
from .tools.memory import primary_memory_tools


@dataclass
class Services:
    settings: Settings
    db: sqlite3.Connection
    sessions: SessionStore
    tools: Toolset
    memory: MemoryStore
    extractor: MemoryExtractor | None  # None 表示不做后台记忆提取（测试里常用）
    images: ImageArchive
    # 终端和微信在不同线程里运行。处理一条输入时持有这把锁，两边轮流使用数据库。
    lock: RLock = field(default_factory=RLock)

    def close(self) -> None:
        """退出前调用：等后台记忆提取做完，再关闭数据库。"""
        if self.extractor:
            self.extractor.close()
        self.db.close()


def open_services(settings: Settings, data_dir: Path, *, auto_memory: bool = True) -> Services:
    """创建全部共享服务。数据都放在 data_dir（正常运行时是 ~/.kairos）。"""
    db = open_db(data_dir / "kairos.db")
    memory = MemoryStore(data_dir / "memories")
    tools = load_factories(settings.kairos_tool_factories, db)  # 业务工具，默认是待办
    tools.add(primary_memory_tools(memory))
    tools.add(history_tools(db))
    images = ImageArchive(db, data_dir / "files", lambda: build_model(settings))
    tools.add(image_tools(images))
    extractor = MemoryExtractor(memory, lambda: build_model(settings)) if auto_memory else None
    return Services(settings, db, SessionStore(db), tools, memory, extractor, images)
