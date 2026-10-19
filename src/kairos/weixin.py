"""微信登录命令，以及只启用微信模块的统一程序兼容入口。"""

import argparse
import signal

from . import config, program
from .app_lock import AppAlreadyRunning, application_lock
from .channels.base import TransportError
from .channels.weixin import login


def _stop(signum, frame) -> None:
    """将登录期间的系统终止转为正常中断。

    作用:
        让扫码登录退出时释放运行锁。
    参数:
        signum, frame: Python 信号机制传入的参数。
    返回:
        不正常返回。
    异常:
        KeyboardInterrupt: 系统要求停止登录。
    行为说明:
        1. 聊天程序的信号与资源生命周期由 program.run 管理。
    """
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    """解析微信登录或兼容运行命令。

    作用:
        登录只处理微信凭据；run 启动同一套程序并仅启用微信模块。
    参数:
        argv: 可注入的参数列表，None 读取命令行。
    返回:
        正常停止为 0，登录或运行失败为 1。
    行为说明:
        1. 此模块不创建 Runtime，也不另建一套微信业务后台。
        2. run 不能与另一个 Kairos 进程同时占用数据目录；双入口应一次启动。
        3. 帮助和非法参数不读取用户配置或凭据。
    """
    parser = argparse.ArgumentParser(description="Kairos 微信登录与仅微信运行入口；双入口请使用 kairos")
    parser.add_argument("command", nargs="?", choices=("run", "login"), default="run")
    parser.add_argument("--max-messages", type=int, default=0, help="验收时限制微信输入数，默认持续运行")
    parser.add_argument("--idle-timeout", type=float, default=None, help="验收时设置空闲秒数")
    parser.add_argument("--replace-login", action="store_true", help="重新扫码并替换凭据，仅用于 login")
    args = parser.parse_args(argv)
    if args.max_messages < 0 or (args.idle_timeout is not None and args.idle_timeout <= 0):
        parser.error("消息上限不能为负，空闲时限必须为正")
    if args.replace_login and args.command != "login":
        parser.error("--replace-login 仅用于 login")
    if args.command == "run":
        return program.run(channels=("weixin",), max_messages=args.max_messages, idle_timeout=args.idle_timeout)

    previous_signal = signal.signal(signal.SIGTERM, _stop)
    try:
        data_dir = config.get_app_dir()
        with application_lock(data_dir, label="Kairos login"):
            login(data_dir / "channels" / "weixin" / "account.json", replace=args.replace_login)
        return 0
    except AppAlreadyRunning as error:
        print(str(error), flush=True)
        return 1
    except TransportError as error:
        print(f"微信登录失败：{error}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("微信登录已停止。", flush=True)
        return 0
    except Exception as error:
        print(f"微信登录失败：{type(error).__name__}；未输出可能含私人信息的原始错误。", flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


if __name__ == "__main__":
    raise SystemExit(main())
