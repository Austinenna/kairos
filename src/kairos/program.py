"""在一个进程中装配终端、微信和共用的串行应用服务。"""

import signal
import threading
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from queue import Empty, SimpleQueue

from pydantic import ValidationError

from . import cli, config
from .app_lock import AppAlreadyRunning, application_lock
from .application import ApplicationService
from .channels.weixin import WeixinAdapter, import_probe, read_credentials
from .config import Settings
from .runtime import create_runtime
from .weixin_channel import WeixinChannelRunner


class ProgramStopped(BaseException):
    """表示系统要求停止整个程序，不是单个聊天请求的取消。"""


def parse_channels(value: str | Sequence[str]) -> tuple[str, ...]:
    """校验并去重需要启用的接入模块。

    作用:
        将命令行逗号列表或配置数组转换为程序装配选择。
    参数:
        value: terminal、weixin 组成的逗号列表或字符串序列。
    返回:
        保持首次出现顺序的模块名元组。
    异常:
        ValueError: 列表为空或包含不支持的模块名。
    行为说明:
        1. 不读取配置、不创建目录，也不启动任何入口。
    """
    entries = value.split(",") if isinstance(value, str) else value
    channels = tuple(dict.fromkeys(item.strip() for item in entries))
    if not channels or any(item not in {"terminal", "weixin"} for item in channels):
        raise ValueError("入口只能选择 terminal、weixin，且至少启用一个")
    return channels


def _weixin_runner(
    service: ApplicationService,
    data_dir: Path,
    report: Callable[[str], None],
    *,
    max_messages: int,
    idle_timeout: float | None,
) -> WeixinChannelRunner:
    """准备微信收发模块，不创建另一个 Runtime。

    作用:
        读取登录凭据并将微信处理端接入当前程序的应用服务。
    参数:
        service: 当前进程唯一的应用服务。
        data_dir: 已取得运行锁的用户目录。
        report: 不含正文和凭据的状态消息接收器。
        max_messages: 验收输入上限，零表示不限制。
        idle_timeout: 可选空闲停止秒数。
    返回:
        尚未启动的微信运行器。
    行为说明:
        1. 缺少正式账号时导入已验证的探针凭据，不改动原探针数据。
        2. 收发线程各自创建适配器与数据库连接。
    """
    account_path = data_dir / "channels" / "weixin" / "account.json"
    if not account_path.exists():
        import_probe(data_dir / "weixin-probe" / "state.json", account_path)
        report("已复用探针微信登录状态，原探针数据保留。")
    credentials = read_credentials(account_path)
    return WeixinChannelRunner(
        data_dir / "channels" / "channels.sqlite3",
        lambda: WeixinAdapter(dict(credentials)),
        service.client,
        initial_checkpoint=credentials.get("initial_checkpoint", {}),
        report=report,
        max_messages=max_messages,
        idle_timeout=idle_timeout,
    )


def run(
    *,
    session_id: str | None = None,
    resume_latest: bool = False,
    channels: str | Sequence[str] | None = None,
    settings: Settings | None = None,
    max_messages: int = 0,
    idle_timeout: float | None = None,
) -> int:
    """启动单进程多入口程序，在释放运行锁前收尾所有资源。

    作用:
        集中管理配置、唯一应用服务、终端和微信模块的生命周期。
    参数:
        session_id, resume_latest: 仅用于终端的启动会话选择，二者互斥。
        channels: 可选入口覆盖，省略时读取 KAIROS_CHANNELS。
        settings: 可注入的设置，省略时在持锁后读取用户配置。
        max_messages, idle_timeout: 微信验收输入数及空闲时限。
    返回:
        正常停止为 0，启动或入口运行失败为 1。
    行为说明:
        1. Runtime 与业务数据库只在应用服务的固定执行线程创建和关闭。
        2. 微信失败不关闭仍在使用的终端；/detach 留下微信继续运行。
        3. /exit、输入结束和 SIGTERM 停止整个程序；生成时 Ctrl+C 只取消终端请求。
        4. 先停止接收和取消请求，等待收发线程退出，再关闭 Runtime 和释放总锁。
    """
    notices: SimpleQueue[str] = SimpleQueue()

    def drain_notices() -> list[str]:
        """取出等待在主线程展示的入口状态。

        作用:
            避免微信线程直接打印并破坏终端输入界面。
        返回:
            当前已入队的状态文字列表。
        行为说明:
            1. 不等待网络或模型任务。
        """
        messages = []
        while True:
            try:
                messages.append(notices.get_nowait())
            except Empty:
                return messages

    def print_notices() -> None:
        """在没有输入界面占用终端时展示等待中的状态。

        作用:
            为仅微信模式和程序收尾提供本机状态输出。
        返回:
            None。
        行为说明:
            1. 队列中只保存受控状态，不包含正文或凭据。
        """
        for message in drain_notices():
            print(message, flush=True)

    def stop_program(signum, frame) -> None:
        """将系统终止信号转为整个程序的收尾请求。

        作用:
            防止 SIGTERM 被终端当作单轮取消后继续等待输入。
        参数:
            signum, frame: Python 信号机制传入的参数。
        返回:
            不正常返回。
        异常:
            ProgramStopped: 系统请求停止整个程序。
        行为说明:
            1. 资源由外层 finally 关闭。
        """
        raise ProgramStopped

    previous_signal = None
    if threading.current_thread() is threading.main_thread():
        previous_signal = signal.signal(signal.SIGTERM, stop_program)
    try:
        with application_lock(config.get_app_dir(), label="Kairos"):
            settings = settings or Settings()
            selected = parse_channels(channels if channels is not None else settings.kairos_channels)
            if session_id is not None and resume_latest:
                print("--session 和 --continue 不能同时使用。", flush=True)
                return 1
            if "terminal" not in selected and (session_id is not None or resume_latest):
                print("会话启动参数只用于终端；微信沿用自己的持久会话绑定。", flush=True)
                return 1
            if max_messages < 0 or (idle_timeout is not None and idle_timeout <= 0):
                print("微信消息上限不能为负，空闲时限必须为正。", flush=True)
                return 1
            if not settings.kairos_api_key:
                print("缺少模型 API key，请检查 ~/.kairos/.env。", flush=True)
                return 1
            data_dir = config.get_app_dir()
            service = ApplicationService(
                partial(
                    create_runtime,
                    settings=settings,
                    db_path=data_dir / "kairos.sqlite3",
                    memory_dir=data_dir / "memory",
                ),
                request_path=data_dir / "application" / "requests.sqlite3",
            )
            runner = None
            terminal = None
            channel_failed = False
            try:
                # 微信先创建会话不能影响终端 --continue 的选择。
                if "terminal" in selected:
                    terminal = service.client("terminal", session_id=session_id, resume_latest=resume_latest)
                if "weixin" in selected:
                    try:
                        runner = _weixin_runner(
                            service, data_dir, notices.put, max_messages=max_messages, idle_timeout=idle_timeout
                        )
                        runner.start()
                        notices.put("微信入口已启动，与终端共用同一程序的 Runtime。")
                    except Exception as error:
                        channel_failed = True
                        notices.put(f"微信入口未能启动：{type(error).__name__}；请检查登录状态，未输出私人错误内容。")
                reason = "detach"
                if terminal is not None:
                    reason = cli.run(
                        runtime=terminal,
                        notice_reader=drain_notices,
                        allow_detach=runner is not None and runner.running,
                    )
                print_notices()
                if reason == "detach" and runner is not None:
                    if runner.running:
                        print("Kairos 正在运行微信入口；Ctrl+C 停止整个程序。", flush=True)
                    while not runner.join(timeout=0.2):
                        print_notices()
                channel_failed = channel_failed or (runner is not None and runner.error is not None)
                return 1 if channel_failed else 0
            finally:
                saved_signals = {}
                if threading.current_thread() is threading.main_thread():
                    # 请求已进入收尾；再次按 Ctrl+C 不能跳过线程 join 后提前释放总锁。
                    for number in (signal.SIGINT, signal.SIGTERM):
                        saved_signals[number] = signal.signal(number, signal.SIG_IGN)
                try:
                    _close_resources(service, runner, terminal, print_notices)
                finally:
                    for number, handler in saved_signals.items():
                        signal.signal(number, handler)
    except AppAlreadyRunning as error:
        print(str(error), flush=True)
        return 1
    except ValidationError:
        print("配置无效，请检查 ~/.kairos/.env；未输出可能含凭据的原始错误。", flush=True)
        return 1
    except (KeyboardInterrupt, ProgramStopped):
        print("Kairos 已停止，已执行的工具不会自动重跑。", flush=True)
        return 0
    except Exception as error:
        print(f"Kairos 停止：{type(error).__name__}；未输出可能含私人信息的原始错误。", flush=True)
        return 1
    finally:
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)


def _close_resources(service, runner, terminal, print_notices: Callable[[], None]) -> None:
    """按接收、请求、入口、核心的顺序完成程序收尾。

    作用:
        在调用方仍持有运行锁时等待工作线程结束，不把关闭核心交给某一个入口。
    参数:
        service: 唯一应用服务。
        runner: 可选微信运行器。
        terminal: 可选终端客户端。
        print_notices: 在主线程显示剩余状态的回调。
    返回:
        None。
    行为说明:
        1. 取消只是合作请求，网络或同步工具必须等待实际返回。
        2. 即使入口关闭发生异常，也尝试关闭核心资源。
    """
    if runner is not None:
        runner.stop()
    service.stop_accepting()
    service.cancel_all()
    try:
        if runner is not None:
            print("正在停止微信收发并等待当前操作结束。", flush=True)
            while not runner.join(timeout=0.2):
                print_notices()
    finally:
        try:
            if terminal is not None:
                terminal.close()
            print_notices()
        finally:
            service.close()
