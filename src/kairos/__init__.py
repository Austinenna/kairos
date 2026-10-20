def main() -> None:
    """解析参数并启动单进程多入口 Kairos。

    作用:
        将入口选择与终端会话选项交给统一程序，由其创建应用层和 Runtime。
    返回:
        None，正常停止后返回。
    异常:
        SystemExit: 程序启动或运行失败时以非零状态退出。
    行为说明:
        1. 帮助和参数互斥校验不读取用户配置或启动入口。
        2. 未指定 --channels 时同时启用终端和微信，指定时按参数选择。
    """
    import argparse

    from .program import parse_channels, run

    parser = argparse.ArgumentParser(description="Kairos：同一程序中的终端和微信，共用会话与执行后台")
    sessions = parser.add_mutually_exclusive_group()
    sessions.add_argument("--continue", dest="resume_latest", action="store_true", help="终端继续最近的会话")
    sessions.add_argument("--session", dest="session_id", metavar="ID", help="终端恢复指定会话（支持唯一 ID 前缀）")
    parser.add_argument(
        "--channels", type=parse_channels, metavar="terminal,weixin", help="选择启动入口；省略时启动终端和微信"
    )
    args = parser.parse_args()
    status = run(session_id=args.session_id, resume_latest=args.resume_latest, channels=args.channels)
    if status:
        raise SystemExit(status)
