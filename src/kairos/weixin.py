"""微信私聊文字入口：python -m kairos.weixin run。"""

import argparse
import signal
from contextlib import ExitStack
from functools import partial

from pydantic import ValidationError

from . import config
from .app_lock import AppAlreadyRunning, application_lock
from .channel_app import ChannelApp
from .channels.base import TransportError
from .channels.store import ChannelStore
from .channels.weixin import WeixinAdapter, import_probe, login, read_credentials
from .config import Settings
from .runtime import create_runtime


def _stop(signum, frame) -> None:
    """将进程终止信号转为可正常收尾的中断。

    作用:
        让微信前台进程停止后释放锁、模型客户端和数据库。
    参数:
        signum: 操作系统信号编号。
        frame: Python 当前执行栈，由信号机制传入。
    返回:
        不正常返回。
    异常:
        KeyboardInterrupt: 接收到 SIGTERM 时中断前台操作。
    行为说明:
        1. 使用 Runtime 原有中断保存逻辑，不尝试撤销已经执行的工具。
    """
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    """解析微信入口命令，在共享应用锁内登录或运行文字通道。

    作用:
        复用当前用户的模型配置、会话库、记忆与可选工具，微信传输状态单独保存。
    参数:
        argv: 可选参数列表，测试可以显式提供；None 读取命令行。
    返回:
        正常退出或用户中断为 0，启动或运行错误为 1 (int)。
    行为说明:
        1. 沿用用户 ~/.kairos/.env，不新增模型凭据配置，也不回显原始错误。
        2. 首次运行可复制此前已验证的探针登录信息，不改动原探针状态。
        3. CLI 与微信入口共用同一数据目录锁，运行期禁止两个入口同时写入。
        4. 默认持续运行；验收可限定输入数量或空闲时限，结束时等待后台收尾。
    """
    parser = argparse.ArgumentParser(description="Kairos 微信私聊文字通道；与 CLI 共用会话库、记忆和工具")
    parser.add_argument("command", nargs="?", choices=("run", "login"), default="run")
    parser.add_argument("--max-messages", type=int, default=0, help="验收时限制处理输入数，默认持续运行")
    parser.add_argument("--idle-timeout", type=float, default=None, help="验收时设置空闲秒数，默认持续等待")
    parser.add_argument("--replace-login", action="store_true", help="重新扫码并替换微信入口凭据，仅用于 login")
    args = parser.parse_args(argv)
    if args.max_messages < 0 or (args.idle_timeout is not None and args.idle_timeout <= 0):
        parser.error("消息上限不能为负，空闲时限必须为正")
    if args.replace_login and args.command != "login":
        parser.error("--replace-login 仅用于 login")

    data_dir = config.get_app_dir()
    account_path = data_dir / "channels" / "weixin" / "account.json"
    previous_signal = signal.signal(signal.SIGTERM, _stop)
    try:
        with application_lock(data_dir, label="Kairos Weixin"):
            if args.command == "login":
                login(account_path, replace=args.replace_login)
                return 0
            settings = Settings()
            if not settings.kairos_api_key:
                print("缺少模型 API key，请检查 ~/.kairos/.env。", flush=True)
                return 1
            if not account_path.exists():
                import_probe(data_dir / "weixin-probe" / "state.json", account_path)
                print("已复用探针登录状态；原探针数据保留。", flush=True)
            credentials = read_credentials(account_path)
            adapter = WeixinAdapter(credentials)
            with ExitStack() as resources:
                store = ChannelStore(data_dir / "channels" / "channels.sqlite3")
                resources.callback(store.close)
                store.seed_checkpoint(adapter.channel, adapter.account_id, credentials.get("initial_checkpoint", {}))
                app = ChannelApp(
                    store,
                    adapter,
                    partial(
                        create_runtime,
                        settings=settings,
                        db_path=data_dir / "kairos.sqlite3",
                        memory_dir=data_dir / "memory",
                    ),
                    report=lambda message: print(message, flush=True),
                )
                resources.callback(app.close)
                print("Kairos 微信文字通道已启动。只接受本人私聊，输入 /help 查看命令；Ctrl+C 停止。", flush=True)
                processed = app.run(max_messages=args.max_messages, idle_timeout=args.idle_timeout)
                print(f"微信入口已停止，本次处理 {processed} 条输入。", flush=True)
            return 0
    except AppAlreadyRunning as error:
        print(str(error), flush=True)
        return 1
    except ValidationError:
        print("模型配置无效，请检查 ~/.kairos/.env；未输出可能含凭据的原始错误。", flush=True)
        return 1
    except TransportError as error:
        if error.authentication:
            print("微信登录已失效，请停止旧入口并重新扫码登录。", flush=True)
        else:
            print(f"微信入口停止：{error}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("微信入口已停止，已执行的工具不会自动重跑。", flush=True)
        return 0
    except Exception as error:
        print(f"微信入口停止：{type(error).__name__}；未输出可能含私人信息的原始错误。", flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


if __name__ == "__main__":
    raise SystemExit(main())
