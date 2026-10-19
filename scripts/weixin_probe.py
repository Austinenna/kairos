"""验证个人微信 iLink 文字通路；不接模型、工具或 Kairos 业务数据库。

运行：uv run --no-project --with 'qrcode[pil]>=8.2' python scripts/weixin_probe.py login
随后：python3 scripts/weixin_probe.py run
协议依据：https://github.com/Tencent/openclaw-weixin/blob/main/docs/protocol.md
"""

import argparse
import base64
import fcntl
import getpass
import hashlib
import json
import os
import secrets
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

DEFAULT_BASE = "https://ilinkai.weixin.qq.com"
CHANNEL_VERSION = "2.4.10"
APP_HEADERS = {"iLink-App-Id": "bot", "iLink-App-ClientVersion": "132106"}
BASE_INFO = {"channel_version": CHANNEL_VERSION, "bot_agent": "KairosProbe/0.1"}
REPLY_TEXT = "Kairos 微信通路测试成功：已收到测试口令。这是固定回复，未调用模型。"


class ProbeError(Exception):
    """表示可以直接展示、没有凭据及原始响应正文的探针错误。"""


class ProbeTimeout(ProbeError):
    """表示本次网络等待超时；收消息可继续，发送结果可能未知。"""


def validate_base_url(url: str) -> str:
    """校验服务端地址，限制凭据只发往微信 HTTPS 域名。

    作用:
        用于登录重定向和已登录 API 地址，避免将凭据发送至任意目标。
    参数:
        url: 微信返回的 API 根地址。
    返回:
        去除末尾斜杠的根地址 (str)。
    异常:
        ProbeError: 地址不是微信 HTTPS 根地址，或包含认证、端口、查询及片段。
    行为说明:
        1. 不联网，不允许任意 HTTP 重定向；业务重定向重新经过本函数。
    """
    if not isinstance(url, str):
        raise ProbeError("微信 API 地址格式无效")
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or not host.endswith(".weixin.qq.com")
        or parsed.netloc != host
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ProbeError("拒绝使用微信 HTTPS 域名以外的 API 地址")
    return f"https://{host}"


def save_state(path: Path, state: dict) -> None:
    """以私有权限原子保存探针状态。

    作用:
        保存凭据、游标和测试发送状态，不接触 Kairos 业务数据。
    参数:
        path: 专用状态文件路径。
        state: 可序列化的完整状态。
    返回:
        None。
    行为说明:
        1. 专用目录权限为 700，临时文件和最终文件为 600。
        2. 写入并同步后原子替换，失败时清理本次临时文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(state, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path: Path) -> dict:
    """读取专用探针状态，缺失时返回空状态。

    作用:
        为登录、重启和状态查询提供同一存储入口。
    参数:
        path: 专用状态文件路径。
    返回:
        状态字典 (dict)，不会打印其中的凭据。
    异常:
        ProbeError: 状态 JSON 损坏或不是对象。
    行为说明:
        1. 不读取 .env、会话数据库或其他应用凭据。
    """
    if not path.exists():
        return {"cursor": "", "replies": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise ProbeError("探针状态文件损坏，请保留现场后检查") from None
    if not isinstance(state, dict) or not isinstance(state.get("replies", {}), dict):
        raise ProbeError("探针状态文件结构无效")
    return state


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁止 HTTP 层自动转发请求和凭据。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """拒绝自动重定向，由调用方将其作为 HTTP 错误处理。

        作用:
            防止带 Bearer 的 POST 或登录查询跨主机跳转。
        参数:
            req, fp, code, msg, headers, newurl: urllib 提供的重定向信息。
        返回:
            None，不创建新的请求。
        行为说明:
            1. 微信 JSON 中的合法重定向由登录状态机单独校验。
        """
        return None


class ProbeClient:
    """发送对齐官方客户端结构的有界 HTTPS JSON 请求。"""

    def __init__(self, base_url: str = DEFAULT_BASE, token: str = ""):
        """绑定已校验地址和可选 Bot 凭据。

        作用:
            隔离网络协议，不读取磁盘或启动后台任务。
        参数:
            base_url: 微信 HTTPS API 根地址。
            token: 已登录 Bot 凭据，登录前为空。
        返回:
            无返回值。
        行为说明:
            1. 使用系统网络设置；禁止 HTTP 自动重定向。
        """
        self.base_url = validate_base_url(base_url)
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, endpoint: str, payload: dict | None = None, timeout: float = 35.0) -> dict:
        """发送一个请求，并区分 HTTP、业务和超时错误。

        作用:
            供扫码、接收和发送复用，不自动重试具有发送副作用的请求。
        参数:
            endpoint: 本探针生成的相对 API 路径及已编码查询。
            payload: JSON 对象；None 使用 GET，否则使用 POST。
            timeout: 网络等待上限秒数。
        返回:
            已解析且未报告业务失败的 JSON 对象 (dict)。
        异常:
            ProbeError: 路径不合法、HTTP/网络/格式错误，或服务端返回业务错误。
            ProbeTimeout: 网络等待超时。
        行为说明:
            1. GET 仅带应用头；POST 增加协议头，登录后才带 Bearer。
            2. 响应最大 2 MiB；错误不包含原始正文、URL、令牌或账号标识。
        """
        if not endpoint.startswith("/ilink/bot/") or "\n" in endpoint or "\r" in endpoint:
            raise ProbeError("非法探针 API 路径")
        headers = dict(APP_HEADERS)
        data = None
        if payload is not None:
            headers.update(
                {
                    "Content-Type": "application/json",
                    "AuthorizationType": "ilink_bot_token",
                    "X-WECHAT-UIN": base64.b64encode(str(secrets.randbits(32)).encode()).decode(),
                }
            )
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + endpoint, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            raise ProbeError(f"微信 HTTP 请求失败，状态码 {error.code}") from None
        except TimeoutError:
            raise ProbeTimeout("微信请求等待超时") from None
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise ProbeTimeout("微信请求等待超时") from None
            raise ProbeError("微信网络请求失败；未输出可能含敏感信息的原始错误") from None
        if len(raw) > 2 * 1024 * 1024:
            raise ProbeError("微信响应超过探针读取上限")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ProbeError("微信响应不是有效 JSON") from None
        if not isinstance(result, dict):
            raise ProbeError("微信响应不是 JSON 对象")
        for field in ("ret", "errcode"):
            value = result.get(field)
            if value not in (None, 0):
                code = str(value) if type(value) is int else "未知"
                raise ProbeError(f"微信返回业务错误：{field}={code}")
        return result


def login(path: Path, timeout: float) -> None:
    """生成二维码并等待本人扫码，将登录结果保存到专用目录。

    作用:
        仅建立测试 Bot 连接，不读取微信历史或自动发送消息。
    参数:
        path: 专用状态文件路径。
        timeout: 本次登录最长等待秒数。
    返回:
        None；成功后打印不含账号信息的确认。
    异常:
        ProbeError: 缺少二维码依赖、响应字段不完整、二维码失效、验证被阻止、
            没有可复用凭据的已有绑定、未知登录状态，或等待超时。
    行为说明:
        1. 二维码负载只在本地编码为 PNG，不打印其 URL。
        2. 仅保存成功响应中的 Bot 凭据与扫码用户 ID；已存在凭据时不覆盖。
    """
    state = load_state(path)
    if state.get("credentials"):
        print("已有探针登录凭据，请直接运行 run 验证收发；本次没有覆盖登录状态。", flush=True)
        return
    try:
        import qrcode
    except ImportError:
        raise ProbeError("生成二维码需要使用 uv run --no-project --with 'qrcode[pil]>=8.2' 运行 login") from None
    client = ProbeClient()
    qr = client.request("/ilink/bot/get_bot_qrcode?bot_type=3", {"local_token_list": []})
    if not all(isinstance(qr.get(key), str) and qr[key] for key in ("qrcode", "qrcode_img_content")):
        raise ProbeError("微信未返回完整二维码字段")
    save_state(path, state)
    image_path = path.parent / "login.png"
    qrcode.make(qr["qrcode_img_content"]).save(image_path)
    image_path.chmod(0o600)
    print(f"二维码已生成：{image_path}\n请用手机微信扫码并确认授权。", flush=True)
    deadline = time.monotonic() + timeout
    verify_code = None
    last_status = None
    while time.monotonic() < deadline:
        query = {"qrcode": qr["qrcode"]}
        if verify_code:
            query["verify_code"] = verify_code
        try:
            result = client.request(
                "/ilink/bot/get_qrcode_status?" + urllib.parse.urlencode(query),
                timeout=min(35.0, max(1.0, deadline - time.monotonic())),
            )
        except ProbeTimeout:
            continue
        status = result.get("status")
        if status == "confirmed":
            if not all(
                isinstance(result.get(key), str) and result[key]
                for key in ("bot_token", "ilink_bot_id", "ilink_user_id")
            ):
                raise ProbeError("已确认扫码，但缺少 Bot 凭据或扫码用户标识；停止测试")
            state["credentials"] = {
                "bot_token": result["bot_token"],
                "bot_id": result["ilink_bot_id"],
                "owner_id": result["ilink_user_id"],
                "base_url": validate_base_url(result.get("baseurl") or DEFAULT_BASE),
            }
            save_state(path, state)
            image_path.unlink(missing_ok=True)
            print("扫码登录成功；凭据已私有保存，二维码图片已清除。", flush=True)
            return
        if status == "scaned_but_redirect":
            host = result.get("redirect_host")
            if not isinstance(host, str) or not host:
                raise ProbeError("扫码重定向缺少目标地址")
            client = ProbeClient(validate_base_url("https://" + host))
        elif status == "need_verifycode":
            verify_code = getpass.getpass("请在本机终端输入手机显示的配对码（输入不回显）：").strip()
        elif status == "scaned":
            verify_code = None
            if status != last_status:
                print("已扫码，正在等待微信授权确认。", flush=True)
        elif status == "expired":
            raise ProbeError("二维码已过期，请重新运行 login")
        elif status == "verify_code_blocked":
            raise ProbeError("微信已阻止本次配对码验证，请停止尝试并检查手机提示")
        elif status == "binded_redirect":
            raise ProbeError("微信提示已有绑定，但本探针没有可复用凭据；尚未完成通路验证")
        elif status != "wait":
            raise ProbeError("微信返回未识别的登录状态；未输出原始响应")
        last_status = status
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise ProbeError("扫码等待超时，请重新运行 login")


def process_batch(client: ProbeClient, state: dict, path: Path, batch: dict, expected: str) -> bool:
    """只回复本人私聊中的指定测试口令，并保存去重与游标状态。

    作用:
        验证真实入站到出站的通路，不把任意微信内容交给模型或写入日志。
    参数:
        client: 已登录的协议客户端；测试可注入 FakeClient。
        state: 含 credentials、cursor 和 replies 的探针状态。
        path: 状态保存路径。
        batch: getupdates 返回的消息批次。
        expected: 本次必须精确匹配的测试口令。
    返回:
        本批是否提交过一条未被接口明确拒绝的测试回复 (bool)。
    异常:
        ProbeError: 批次格式无效、匹配消息缺少 ID/回复令牌，或发送响应明确失败。
    行为说明:
        1. 只接受扫码用户、非群聊、非 Bot 的纯文字；其他消息直接忽略。
        2. 发送前保存 sending；失败保存 uncertain，不自动重发结果未知的消息。
        3. 数值零业务码保存 sent；未提供业务码保存 submitted，仍须手机确认。
        4. 已见消息跨重启去重；本批处理完成后才保存新游标。
    """
    owner = state["credentials"]["owner_id"]
    messages = batch.get("msgs", [])
    if not isinstance(messages, list):
        raise ProbeError("微信消息批次格式无效")
    replied = False
    records = state.setdefault("replies", {})
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("from_user_id") != owner
            or message.get("message_type") not in (None, 1)
            or message.get("to_user_id") not in (None, "", state["credentials"]["bot_id"])
            or message.get("group_id")
        ):
            continue
        items = message.get("item_list", [])
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, dict) or item.get("type") != 1 for item in items)
        ):
            continue
        texts = [item.get("text_item", {}).get("text") for item in items if isinstance(item.get("text_item"), dict)]
        if len(texts) != len(items) or any(not isinstance(text, str) for text in texts):
            continue
        if "\n".join(texts) != expected:
            continue
        message_id = message.get("message_id")
        context = message.get("context_token")
        if type(message_id) not in (int, str) or str(message_id) == "" or not isinstance(context, str) or not context:
            raise ProbeError("已收到测试口令，但缺少消息 ID 或回复令牌")
        key = hashlib.sha256(json.dumps([state["credentials"]["bot_id"], owner, str(message_id)]).encode()).hexdigest()
        if key in records or replied:
            continue
        client_id = f"kairos-probe:{int(time.time() * 1000)}-{secrets.token_hex(4)}"
        records[key] = {"status": "sending", "client_id": client_id}
        save_state(path, state)
        try:
            result = client.request(
                "/ilink/bot/sendmessage",
                {
                    "msg": {
                        "from_user_id": "",
                        "to_user_id": owner,
                        "client_id": client_id,
                        "message_type": 2,
                        "message_state": 2,
                        "context_token": context,
                        "item_list": [{"type": 1, "text_item": {"text": REPLY_TEXT}}],
                    },
                    "base_info": dict(BASE_INFO),
                },
                timeout=15.0,
            )
            if result.get("ret") not in (None, 0) or result.get("errcode") not in (None, 0):
                raise ProbeError("发送接口返回业务错误，请先检查手机；不会自动重发")
        except BaseException:
            records[key]["status"] = "uncertain"
            save_state(path, state)
            raise
        business_ack = any(type(result.get(field)) is int and result[field] == 0 for field in ("ret", "errcode"))
        records[key]["status"] = "sent" if business_ack else "submitted"
        records[key]["acknowledgement"] = "business" if business_ack else "http_only"
        save_state(path, state)
        if business_ack:
            print("发送接口返回了业务成功码。", flush=True)
        else:
            print("发送接口已返回 HTTP 成功，但未提供业务状态码；投递结果以手机确认为准。", flush=True)
        replied = True
    cursor = batch.get("get_updates_buf")
    if isinstance(cursor, str) and cursor:
        state["cursor"] = cursor
    save_state(path, state)
    return replied


def run_probe(path: Path, timeout: float) -> None:
    """等待本次唯一测试口令，发送一条固定回复后停止。

    作用:
        独立验证收发和已保存登录状态的复用，不常驻、不回应其他消息。
    参数:
        path: 专用状态文件。
        timeout: 等待测试消息的最长秒数。
    返回:
        None；发送请求返回后仍需用户确认手机实际收到。
    异常:
        ProbeError: 没有登录、凭据结构无效，或等待测试消息超时。
    行为说明:
        1. 每次启动生成新测试口令；恢复此前游标及发送记录。
        2. 只有接收超时会继续轮询，发送错误直接停止。
    """
    state = load_state(path)
    credentials = state.get("credentials")
    if not isinstance(credentials, dict) or not all(
        isinstance(credentials.get(key), str) and credentials[key]
        for key in ("bot_token", "bot_id", "owner_id", "base_url")
    ):
        raise ProbeError("缺少完整登录凭据，请先运行 login")
    client = ProbeClient(credentials["base_url"], credentials["bot_token"])
    expected = "Kairos测试-" + secrets.token_hex(3).upper()
    count = len(state.get("replies", {}))
    print(f"已加载登录状态；此前记录 {count} 次发送。", flush=True)
    print(f"请在手机微信的 Bot 私聊中发送：{expected}", flush=True)
    print("只回复本次口令，成功提交一条固定回复后自动停止。", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            batch = client.request(
                "/ilink/bot/getupdates",
                {"get_updates_buf": state.get("cursor", ""), "base_info": dict(BASE_INFO)},
                timeout=min(40.0, max(1.0, deadline - time.monotonic())),
            )
        except ProbeTimeout:
            continue
        if process_batch(client, state, path, batch, expected):
            print("已收到本人测试口令，发送请求已提交，状态已保存。请确认手机实际收到固定回复。", flush=True)
            return
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    raise ProbeError("等待测试口令超时；本次没有发送回复")


@contextmanager
def state_lock(directory: Path):
    """限制同一状态目录同时只运行一个探针进程。

    作用:
        避免重复轮询、凭据覆盖及重复发送。
    参数:
        directory: 探针专用目录。
    返回:
        持锁的上下文管理器。
    异常:
        ProbeError: 已有探针进程占用同一目录。
    行为说明:
        1. 非阻塞取得文件锁，进程退出时自动释放。
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ProbeError("另一个微信探针正在使用此状态目录") from None
        yield


def main() -> int:
    """解析命令并展示不含凭据的运行结果。

    作用:
        提供 login、run、status 三个独立操作，默认私有保存到 ~/.kairos/weixin-probe。
    返回:
        成功为 0，预期失败为 1，用户中断为 130 (int)。
    行为说明:
        1. 不读取模型配置；不打印异常堆栈和原始网络错误。
        2. status 仅显示登录状态、游标是否存在与发送计数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("login", "run", "status"))
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".kairos" / "weixin-probe")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if not 5 <= args.timeout <= 900:
        parser.error("--timeout 须在 5 到 900 秒之间")
    path = args.state_dir / "state.json"
    try:
        with state_lock(args.state_dir):
            if args.command == "login":
                login(path, args.timeout)
            elif args.command == "run":
                run_probe(path, args.timeout)
            else:
                state = load_state(path)
                records = state.get("replies", {}).values()
                print(
                    json.dumps(
                        {
                            "logged_in": bool(state.get("credentials")),
                            "cursor_saved": bool(state.get("cursor")),
                            "sent": sum(record.get("status") == "sent" for record in records),
                            "submitted": sum(record.get("status") == "submitted" for record in records),
                            "uncertain": sum(record.get("status") in ("sending", "uncertain") for record in records),
                        },
                        ensure_ascii=False,
                    )
                )
    except ProbeError as error:
        print(f"探针停止：{error}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("探针已停止。", flush=True)
        return 130
    except Exception as error:
        print(f"探针停止：本地错误 {type(error).__name__}；未输出原始异常以避免泄露凭据。", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
