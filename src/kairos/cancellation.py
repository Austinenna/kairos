"""在线程之间请求合作取消，不依赖终端、微信或模型协议。"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class RequestCancelled(KeyboardInterrupt):
    """表示执行已观察到取消请求，沿用 Runtime 的回合中断保存语义。"""


class CancellationToken:
    """保存一次请求的线程安全取消标记，不强制终止线程或撤销工具副作用。"""

    def __init__(self) -> None:
        """创建尚未请求取消的标记。

        作用:
            由应用层为一次任务创建，并交给控制入口和执行线程共享。
        返回:
            None。
        行为说明:
            1. 使用 Event 保存单向状态；不启动线程或定时器。
        """
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        """读取是否已经有人请求取消。

        作用:
            供应用层展示取消请求已提交；True 不代表执行已经停止。
        返回:
            是否请求过取消（bool）。
        行为说明:
            1. 只读共享事件，不等待执行线程确认。
        """
        return self._event.is_set()

    def cancel(self) -> None:
        """提交幂等的合作取消请求。

        作用:
            允许控制入口在执行线程忙于模型或工具时独立设置标记。
        返回:
            None。
        行为说明:
            1. 设置事件；执行方到达检查点后自行停止，不发送进程信号。
        """
        self._event.set()

    def raise_if_cancelled(self) -> None:
        """在检查点确认取消并中断当前执行。

        作用:
            将外部取消标记转换成沿用既有回合清理流程的异常。
        返回:
            未请求取消时返回 None。
        异常:
            RequestCancelled: 取消事件已经设置。
        行为说明:
            1. 只在调用线程抛出异常；不保证能打断阻塞中的同步工具或网络等待。
        """
        if self._event.is_set():
            raise RequestCancelled("本轮已取消")


_current_token: ContextVar[CancellationToken | None] = ContextVar("kairos_cancellation_token", default=None)


@contextmanager
def cancellation_scope(token: CancellationToken | None) -> Iterator[None]:
    """在当前执行上下文临时安装本轮取消标记。

    作用:
        让 Agent 循环和内置 provider 共用取消检查，保持已有 Model 方法签名兼容。
    参数:
        token: 本轮标记；None 表示此作用域不启用合作取消。
    返回:
        不产生绑定值的上下文管理器。
    行为说明:
        1. 标记仅供当前上下文读取，不自动传播到另一个工作线程。
        2. 正常返回或异常退出后恢复原上下文，避免取消污染下一个请求。
    """
    previous = _current_token.set(token)
    try:
        yield
    finally:
        _current_token.reset(previous)


def current_cancellation_token() -> CancellationToken | None:
    """读取当前执行上下文的合作取消标记。

    作用:
        让循环在启用取消时优先使用可逐片检查的流式模型接口。
    返回:
        当前 CancellationToken，未安装时为 None。
    行为说明:
        1. 不创建、修改或跨线程传递标记。
    """
    return _current_token.get()


def raise_if_cancelled() -> None:
    """检查当前作用域并在已请求取消时停止执行。

    作用:
        为循环、工具边界和 provider 流读取提供同一个无参数检查点。
    返回:
        未启用取消或尚未请求取消时返回 None。
    行为说明:
        1. 委托当前标记检查；异常由 CancellationToken 抛出。
    """
    token = _current_token.get()
    if token is not None:
        token.raise_if_cancelled()
