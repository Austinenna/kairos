"""同进程微信通道后台：接收与处理各用自己的 SQLite 和网络客户端。"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from .channel_app import ChannelApp
from .channels.base import ChannelAdapter, ReceiveBatch, TransportError
from .channels.store import ChannelStore

if TYPE_CHECKING:
    from .application import RuntimeClient


class WeixinChannelRunner:
    """在同一应用中独立接收微信输入，并通过统一服务串行执行任务。"""

    def __init__(
        self,
        store_path: Path,
        adapter_factory: Callable[[], ChannelAdapter],
        client_factory: Callable[..., RuntimeClient],
        *,
        initial_checkpoint: dict | None = None,
        report: Callable[[str], None] = print,
        max_messages: int = 0,
        idle_timeout: float | None = None,
        store_factory: Callable[..., ChannelStore] = ChannelStore,
    ) -> None:
        """保存后台通道依赖，不在构造时创建线程或读取用户数据。

        作用:
            让主入口控制总应用锁和统一服务生命周期，微信只管理自己的收发。
        参数:
            store_path: 通道 SQLite 路径。
            adapter_factory: 每次创建独立网络适配器的工厂。
            client_factory: 统一 ApplicationService.client，接受 source 与 session_id。
            initial_checkpoint: 仅在没有既有 checkpoint 时保存的初始状态。
            report: 只接收受控状态说明，不接收消息正文或凭据。
            max_messages: 验收用处理上限，零表示持续运行。
            idle_timeout: 可选空闲退出秒数。
            store_factory: 通道存储工厂，测试可观察连接线程。
        返回:
            无返回值。
        异常:
            ValueError: 消息上限或空闲时限无效。
        行为说明:
            1. 接收线程和处理线程分别创建并关闭自己的存储与适配器。
            2. 两条线程均不是 daemon，主入口必须 join 后才能关闭统一服务和释放锁。
        """
        if (
            type(max_messages) is not int
            or max_messages < 0
            or (
                idle_timeout is not None
                and (type(idle_timeout) not in (int, float) or not math.isfinite(idle_timeout) or idle_timeout <= 0)
            )
        ):
            raise ValueError("微信处理上限必须为非负整数，空闲时限必须为有限正数")
        self.store_path = Path(store_path)
        self.adapter_factory = adapter_factory
        self.client_factory = client_factory
        self.initial_checkpoint = initial_checkpoint or {}
        self.report = report
        self.max_messages = max_messages
        self.idle_timeout = idle_timeout
        self.store_factory = store_factory
        self.stop_event = threading.Event()
        self._receiver_ready = threading.Event()
        self._identity: tuple[str, str, str] | None = None
        self._threads: list[threading.Thread] = []
        self._error_lock = threading.Lock()
        self.error: TransportError | None = None
        self.processed = 0

    @property
    def running(self) -> bool:
        """判断是否仍有微信工作线程未退出。

        作用:
            为主入口显示停止等待状态，不把 stop 信号当作已经停完。
        参数:
            无。
        返回:
            任一工作线程仍存活时为 True。
        行为说明:
            1. 不读取数据库或调用模型。
        """
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        """准备存储并启动微信接收及处理线程。

        作用:
            在模型执行期间仍可把新微信输入和游标原子落盘。
        参数:
            无。
        返回:
            None。
        异常:
            RuntimeError: 当前 runner 已经启动过或已被停止。
            捕获的原异常: 任一线程启动失败，等待已启动线程退出后重新抛出。
        行为说明:
            1. 调用线程仅短暂创建并关闭存储，提前完成纯新增表初始化。
            2. 接收线程完成 checkpoint 初始化后，处理线程才读取已有输入。
        """
        if self._threads or self.stop_event.is_set():
            raise RuntimeError("微信通道实例不能重复启动，请创建新实例")
        with closing(self.store_factory(self.store_path)):
            pass
        self._threads = [
            threading.Thread(target=self._worker, args=(self._receive,), name="kairos-weixin-receive"),
            threading.Thread(target=self._worker, args=(self._process,), name="kairos-weixin-process"),
        ]
        try:
            for thread in self._threads:
                thread.start()
        except BaseException:
            self.stop_event.set()
            self._receiver_ready.set()
            self.join()
            raise

    def stop(self) -> None:
        """请求微信线程在安全边界退出。

        作用:
            停止继续接收和提交新任务，不在调用线程关闭其他线程的 SQLite。
        参数:
            无。
        返回:
            None。
        行为说明:
            1. 已发起长轮询等待自身超时；已执行任务由主入口的应用取消及收尾处理。
        """
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> bool:
        """等待微信线程实际完成资源收尾。

        作用:
            为主入口提供关闭统一服务、释放总锁之前的等待屏障。
        参数:
            timeout: 所有线程合计等待秒数，None 表示等到全部退出。
        返回:
            全部已退出时为 True；仍有线程运行时为 False。
        行为说明:
            1. 不强杀线程，不把超时当作已关闭。
        """
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        for thread in self._threads:
            if thread.ident is None:
                continue
            thread.join(None if deadline is None else max(0, deadline - time.monotonic()))
        return not self.running

    def _worker(self, run: Callable[[], None]) -> None:
        """运行一个通道任务并收敛线程异常。

        作用:
            避免 provider 异常或取消的 traceback 直接暴露到共享终端。
        参数:
            run: 接收或处理入口。
        返回:
            None。
        行为说明:
            1. KeyboardInterrupt 及其取消子类只产生受控中断状态。
            2. 任一线程结束都会通知同伴停止；错误仅保留安全说明和类别。
        """
        try:
            run()
        except KeyboardInterrupt:
            self.report("微信当前处理已中断；原请求和已执行工具状态会保留，未自动重跑。")
        except BaseException as error:
            safe_error = (
                error
                if isinstance(error, TransportError)
                else TransportError(f"微信通道停止：{type(error).__name__}；未输出原始错误内容")
            )
            with self._error_lock:
                if self.error is None:
                    self.error = safe_error
            self.report(str(safe_error))
        finally:
            self.stop_event.set()
            self._receiver_ready.set()

    def _receive(self) -> None:
        """持续拉取获准微信输入，在接收线程原子保存游标和消息。

        作用:
            接收不等待模型排队或工具执行，避免长任务期间无法保存新输入。
        参数:
            无。
        返回:
            None。
        异常:
            捕获的 TransportError: 认证失效、不可恢复错误或连续五次失败时停止。
        行为说明:
            1. 存储与适配器均在当前线程创建；连接关闭也在当前线程。
            2. 已收到的批次即使此刻收到 stop，也先完整保存再退出。
        """
        adapter = self.adapter_factory()
        self._identity = (adapter.channel, adapter.account_id, adapter.owner_id)
        with closing(self.store_factory(self.store_path)) as store:
            store.seed_checkpoint(adapter.channel, adapter.account_id, self.initial_checkpoint)
            self._receiver_ready.set()
            failures = 0
            while not self.stop_event.is_set():
                try:
                    batch = adapter.poll(store.checkpoint(adapter.channel, adapter.account_id) or {})
                    messages = tuple(
                        message
                        for message in batch.messages
                        if message.channel == adapter.channel
                        and message.account_id == adapter.account_id
                        and message.sender_id == adapter.owner_id
                    )
                    store.accept_batch(adapter.channel, adapter.account_id, ReceiveBatch(messages, batch.checkpoint))
                    failures = 0
                    if not messages:
                        self.stop_event.wait(0.2)
                except TransportError as error:
                    failures += 1
                    if error.authentication or not error.retryable or failures >= 5:
                        raise
                    self.report("微信接收暂时不可用，正在等待重连；已保存输入不会重跑。")
                    self.stop_event.wait(min(2**failures, 30))

    def _process(self) -> None:
        """按接收顺序处理输入并发送缓存答案。

        作用:
            在处理线程持有自己的存储和发送客户端，业务 Runtime 始终属于应用服务。
        参数:
            无。
        返回:
            None。
        异常:
            TransportError: 两次工厂调用返回不同的通道账号身份。
        行为说明:
            1. 等待接收初始化，不把 SQLite 连接从主线程或接收线程传入。
            2. stop 或取消不会重新执行已提交请求，未确认结果留待账本恢复。
        """
        while not self._receiver_ready.wait(0.1):
            if self.stop_event.is_set():
                return
        if self.stop_event.is_set():
            return
        adapter = self.adapter_factory()
        if (adapter.channel, adapter.account_id, adapter.owner_id) != self._identity:
            raise TransportError("微信通道初始化身份不一致，已停止收发")
        with closing(self.store_factory(self.store_path)) as store:
            app = ChannelApp(store, adapter, self.client_factory, report=self.report, stop_event=self.stop_event)
            self.processed = app.run(max_messages=self.max_messages, idle_timeout=self.idle_timeout, receive=False)
