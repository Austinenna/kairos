def main() -> None:
    """解析会话启动参数并进入 CLI。

    作用:
        作为 kairos 命令入口读取进程参数，将新建或恢复的选择交给 cli.run()，
        并在运行期间独占用户数据目录，避免 CLI 与微信通道同时使用它。

    参数:
        无显式参数；argparse 从 sys.argv 读取 --session ID 或 --continue，两者互斥。

    返回:
        无返回值（None），CLI 正常结束后返回。

    异常:
        SystemExit: 其他入口正在使用此数据目录时，以状态码 1 退出。

    行为说明:
        1. 创建参数解析器，参数合法性、互斥检查和帮助退出由 argparse 处理。
        2. 参数解析完成后取得共享运行锁；帮助命令不创建运行锁。
        3. 将 session_id 和 resume_latest 传入 CLI，由其负责会话创建、恢复及资源管理。
    """

    import argparse
    import sys

    from .app_lock import AppAlreadyRunning, application_lock
    from .cli import run
    from .config import get_app_dir

    parser = argparse.ArgumentParser(description="Kairos：可以保存和恢复会话的个人助手")
    sessions = parser.add_mutually_exclusive_group()
    sessions.add_argument("--continue", dest="resume_latest", action="store_true", help="继续最近的会话")
    sessions.add_argument("--session", dest="session_id", metavar="ID", help="恢复指定会话（支持唯一 ID 前缀）")
    args = parser.parse_args()
    try:
        with application_lock(get_app_dir(), label="Kairos CLI"):
            run(session_id=args.session_id, resume_latest=args.resume_latest)
    except AppAlreadyRunning as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
