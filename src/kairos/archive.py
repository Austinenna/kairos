"""图片归档：收到的图片先原样存进 inbox，再让模型看图给出分类、标题和描述，移到分类目录。

目录结构：files/<分类>/<日期>-<标题>.<扩展名>，files 在 ~/.kairos 下。
images 表记每张图的路径和归档信息，以数据库为准；描述里有图中的关键文字，以后靠它搜索。
"""

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image as PILImage

from .images import prepare_image
from .models import Model
from .tools.registry import tool_schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL,   -- 相对 files 目录的路径
    category    TEXT,            -- NULL 表示还没归档，图片在 inbox 里
    title       TEXT,
    description TEXT,
    source      TEXT NOT NULL,   -- 从哪个入口收到：weixin / terminal
    created_at  TEXT NOT NULL
);
"""

ARCHIVE_PROMPT = """你是 Kairos 的图片归档员。用户发来一张图片，让你帮忙存档备忘。
看图后调用一次 file_image：
- category：分类名，2～6 个字。优先使用已有分类，确实都不合适才新建。
- title：标题，简短说明这是什么，例如“停车票”“午饭发票 38 元”。
- description：一两句描述。把图里的关键文字和数字写进去（金额、日期、名称、报错信息等），以后靠它搜索。"""

FILE_IMAGE = tool_schema(
    "file_image",
    "提交这张图片的归档结果。",
    {
        "category": {"type": "string", "description": "分类名"},
        "title": {"type": "string", "description": "标题"},
        "description": {"type": "string", "description": "描述，包含图中关键文字"},
    },
)

INBOX = "inbox"
# 文件名里不能出现的字符（macOS 和 Windows 的并集），以及空白
UNSAFE_NAME = re.compile(r'[\\/:*?"<>|\s]+')


@dataclass
class ArchivedImage:
    id: int
    path: str  # 相对 files 目录
    category: str | None
    title: str | None
    description: str | None


class ImageArchive:
    def __init__(self, db: sqlite3.Connection, root: Path, model_factory: Callable[[], Model]):
        """root 是 files 目录；model_factory 每次归档时创建模型，需要能看图。"""
        self.db = db
        self.root = root
        self.model_factory = model_factory
        self.db.executescript(SCHEMA)

    def store(self, data: bytes, source: str) -> ArchivedImage:
        """把原图存进 inbox 并记一行。之后分类失败，图片也不会丢。"""
        name = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}.{_extension(data)}"
        path = self.root / INBOX / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        relative = f"{INBOX}/{name}"
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO images (path, source, created_at) VALUES (?, ?, ?)",
                (relative, source, datetime.now().astimezone().isoformat(timespec="seconds")),
            )
        return ArchivedImage(cursor.lastrowid, relative, None, None, None)

    def classify(self, image: ArchivedImage) -> ArchivedImage:
        """让模型看图给出分类、标题和描述，把图片从 inbox 移到分类目录。"""
        filing = self._ask_model(self.root / image.path)
        category, title = _safe_name(filing["category"]), _safe_name(filing["title"])
        source = self.root / image.path
        target = _free_path(self.root / category / f"{datetime.now():%Y-%m-%d}-{title}{source.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        relative = target.relative_to(self.root).as_posix()
        with self.db:
            self.db.execute(
                "UPDATE images SET path = ?, category = ?, title = ?, description = ? WHERE id = ?",
                (relative, category, filing["title"], filing["description"], image.id),
            )
        return ArchivedImage(image.id, relative, category, filing["title"], filing["description"])

    def categories(self) -> list[str]:
        rows = self.db.execute("SELECT DISTINCT category FROM images WHERE category IS NOT NULL ORDER BY category")
        return [row["category"] for row in rows]

    def _ask_model(self, path: Path) -> dict:
        """请求一次模型，取出 file_image 的参数。模型没有调用工具时抛出 ValueError。"""
        existing = "、".join(self.categories()) or "（还没有）"
        image = prepare_image(path.read_bytes(), path.name)  # 发给模型的是缩小后的版本，原图不动
        messages = [
            {"role": "system", "content": f"{ARCHIVE_PROMPT}\n\n已有分类：{existing}"},
            {"role": "user", "content": [{"type": "text", "text": "请归档这张图片。"}, image.to_block()]},
        ]
        response = self.model_factory().stream(messages, [FILE_IMAGE])
        for call in response.tool_calls:
            if call["function"]["name"] == "file_image":
                filing = json.loads(call["function"]["arguments"])
                if all(isinstance(filing.get(key), str) for key in ("category", "title", "description")):
                    return filing
        raise ValueError("模型没有给出归档结果")


def _extension(data: bytes) -> str:
    with PILImage.open(BytesIO(data)) as image:
        return {"JPEG": "jpg"}.get(image.format, image.format.lower())


def _safe_name(name: str) -> str:
    """把模型给的分类或标题变成可用的文件名：去掉路径符号等特殊字符，限制长度。"""
    return UNSAFE_NAME.sub("-", name).strip("-.")[:40] or "未命名"


def _free_path(path: Path) -> Path:
    """同名文件已存在时在后面加 -2、-3……，不覆盖旧图。"""
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_stem(f"{path.stem}-{n}")
        n += 1
    return candidate
