"""生成回答时监听 Esc：按下后中断这一轮回答。

输入框（prompt_toolkit）只在等待输入时读键盘，生成回答期间没有人读。
这里在生成期间开一个后台线程读按键，读到 Esc 就给主线程发 SIGINT，
主线程里出现的 KeyboardInterrupt 和按 Ctrl+C 一样，沿用已有的中断处理。
"""

import os
import select
import signal
import sys
import termios
import threading
import tty

ESC = b"\x1b"


class EscInterrupt:
    """用法：with EscInterrupt() as esc: …；结束后 esc.pressed 表示是不是按了 Esc。"""

    def __init__(self):
        self.pressed = False
        self._saved = None  # 进入前的终端设置，退出时恢复
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def __enter__(self) -> "EscInterrupt":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # 按键不用等回车就能读到，也不回显；Ctrl+C 仍然有效
            self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        if self._saved is None:
            return
        self._stop.set()
        self._thread.join()
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def _listen(self) -> None:
        main = threading.main_thread().ident
        while not self._stop.is_set():
            readable, _, _ = select.select([self._fd], [], [], 0.05)  # 每 0.05 秒检查一次是否该停止
            if not readable:
                continue
            # 方向键等会一次发来以 Esc 开头的一串字符，只有单独一个 Esc 才是按了 Esc 键。
            if os.read(self._fd, 64) == ESC:
                self.pressed = True
                signal.pthread_kill(main, signal.SIGINT)
                return
