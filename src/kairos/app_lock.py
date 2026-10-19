"""保证一份 Kairos 用户数据由一个程序进程管理，进程内可同时启用多个入口。"""

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class AppAlreadyRunning(RuntimeError):
    """表示另一个程序进程已经持有用户数据目录的运行锁。"""


def _create_private_directory(directory: Path) -> None:
    """创建缺失的数据目录并保留已有目录权限。

    作用:
        在获取运行锁前准备存放锁文件的位置，不访问配置或数据库。

    参数:
        directory: 调用方选择的用户数据目录。

    返回:
        无返回值（None）。

    行为说明:
        1. 从最近的已有祖先开始，以 0700 权限逐层创建缺失目录。
        2. 不修改已存在目录的权限，避免改变调用方的目录设置。
    """
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    for current in reversed(missing):
        current.mkdir(mode=0o700, exist_ok=True)


@contextmanager
def application_lock(data_dir: Path, label: str = "Kairos") -> Iterator[None]:
    """独占用户数据目录直到当前程序完成收尾。

    作用:
        在统一程序启动 Runtime 前取得运行锁，避免不同进程同时维护会话、
        记忆与数据库状态；同进程的终端和微信入口共用这一次加锁。

    参数:
        data_dir: 当前入口使用的数据目录；测试和隔离验收可传入临时目录。
        label: 由入口写死的名称，如 Kairos CLI；不得传入路径或个人信息。

    返回:
        持有排他文件锁的上下文管理器，进入时产出 None。

    异常:
        AppAlreadyRunning: 已有另一个程序持有同一数据目录的运行锁。

    行为说明:
        1. 创建私有数据目录，以 0600 权限新建 .runtime.lock 文件。
        2. 使用非阻塞 flock 获取排他锁；只在成功后写入 PID 与入口固定名称。
        3. 持锁覆盖整个上下文，正常返回或异常退出时关闭文件并释放锁。
           不删除锁文件，避免其他进程锁定不同文件而失去互斥。
    """
    _create_private_directory(data_dir)
    fd = os.open(data_dir / ".runtime.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AppAlreadyRunning("另一个 Kairos 程序正在使用此数据目录；请在同一个程序中启用所需入口") from None
        try:
            stream.seek(0)
            stream.truncate()
            json.dump({"pid": os.getpid(), "label": label}, stream, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
