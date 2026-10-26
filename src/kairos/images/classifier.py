"""后台分类线程：给 inbox 里的图片排队分类，不耽误对话。

看图分类要请求模型，耗时几秒。这个线程用自己的数据库连接，不拿 services.lock，
SQLite 用文件锁协调它和对话线程的写入。
"""

from collections.abc import Callable
from pathlib import Path
from queue import Queue
from threading import Thread

from ..models import Model
from ..storage import open_db
from .archive import ArchivedImage, ImageArchive


class ImageClassifier:
    """后台分类线程：排队给 inbox 里的图片分类，做完一张调用一次 on_done。"""

    def __init__(self, db_path: Path, root: Path, model_factory: Callable[[], Model], memory: Callable[[], str]):
        """memory 返回用户记忆快照，每张图分类前取一次，用户刚说的要求也能用上。"""
        self.db_path = db_path
        self.root = root
        self.model_factory = model_factory
        self.memory = memory
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
                result = archive.classify(archive.get(image_id), self.memory())
            except Exception as error:
                result = error
            on_done(result)
        archive.db.close()
