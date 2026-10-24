"""图片归档：收到的图片先原样存进 inbox，再让模型看图给出分类、标题和描述，移到分类目录。

目录结构：files/<分类>/<日期>-<标题>.<扩展名>，files 在 ~/.kairos 下。
images 表记每张图的路径和归档信息，以数据库为准；描述里有图中的关键文字，以后靠它搜索。

存进 inbox 很快，在收消息的线程里做；看图分类要请求模型，交给后台的 ImageClassifier，
它用自己的数据库连接，不拿 services.lock，不耽误对话。
"""

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Thread
from uuid import uuid4

from PIL import Image as PILImage

from .images import Image, prepare_image
from .models import Model
from .storage import open_db
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
        self._pixels: dict[str, Image] = {}  # 路径 → 处理好的图片，每次请求都要用，不重复读文件和缩放

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
        """让模型看图给出分类、标题和描述，把图片从 inbox 移到分类目录。

        对话可能正在另一个线程里读这张图。先在新位置建硬链接、更新数据库，再删掉 inbox 里的，
        这样任何时候数据库记的路径上都有文件。
        """
        filing = self._ask_model(self.root / image.path)
        category, title = _safe_name(filing["category"]), _safe_name(filing["title"])
        source = self.root / image.path
        target = _free_path(self.root / category / f"{datetime.now():%Y-%m-%d}-{title}{source.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(source)  # 同一个文件多一个名字，不复制数据
        relative = target.relative_to(self.root).as_posix()
        with self.db:
            self.db.execute(
                "UPDATE images SET path = ?, category = ?, title = ?, description = ? WHERE id = ?",
                (relative, category, filing["title"], filing["description"], image.id),
            )
        source.unlink()
        return ArchivedImage(image.id, relative, category, filing["title"], filing["description"])

    def get(self, image_id: int) -> ArchivedImage | None:
        row = self.db.execute(
            "SELECT id, path, category, title, description FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        return ArchivedImage(**row) if row else None

    def pixels(self, image: ArchivedImage) -> Image | None:
        """读取原图并处理成发给模型的版本。文件被移走或删除时返回 None。"""
        path = self.root / image.path
        if image.path not in self._pixels:
            if not path.exists():
                return None
            self._pixels[image.path] = prepare_image(path.read_bytes(), path.name)
        return self._pixels[image.path]

    def search(self, query: str = "", limit: int = 10) -> list[dict]:
        """按关键词搜索归档的图片，最新的在前。关键词用空格分开，需同时出现在分类、标题或描述里。"""
        words = query.split()
        searchable = "(IFNULL(category, '') || ' ' || IFNULL(title, '') || ' ' || IFNULL(description, ''))"
        conditions = " AND ".join(f"{searchable} LIKE ?" for _ in words) or "1"
        rows = self.db.execute(
            f"SELECT * FROM images WHERE {conditions} ORDER BY id DESC LIMIT ?",
            [*(f"%{word}%" for word in words), limit],
        )
        return [dict(row) | {"path": str(self.root / row["path"])} for row in rows]

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


class ImageClassifier:
    """后台分类线程：排队给 inbox 里的图片分类，做完一张调用一次 on_done。"""

    def __init__(self, db_path: Path, root: Path, model_factory: Callable[[], Model]):
        self.db_path = db_path
        self.root = root
        self.model_factory = model_factory
        self.jobs: Queue[tuple[int, Callable] | None] = Queue()
        self.thread = Thread(target=self._run, name="kairos-classifier", daemon=True)
        self.thread.start()

    def submit(self, image_id: int, on_done: Callable[[ArchivedImage | Exception], None]) -> None:
        """排队分类一张图。on_done 在后台线程里调用，参数是归档结果，失败时是异常。"""
        self.jobs.put((image_id, on_done))

    def close(self) -> None:
        """等排队的分类全部做完再结束。没做完就退出的图片留在 inbox 里，不会丢。"""
        self.jobs.put(None)
        self.thread.join()

    def _run(self) -> None:
        # 自己的连接：和对话用的连接互不等待，SQLite 用文件锁协调两边的写入。
        archive = ImageArchive(open_db(self.db_path), self.root, self.model_factory)
        while (job := self.jobs.get()) is not None:
            image_id, on_done = job
            try:
                result = archive.classify(archive.get(image_id))
            except Exception as error:
                result = error
            on_done(result)
        archive.db.close()


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
