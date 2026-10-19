"""通过微信 iLink 收发本人私聊；协议状态不进入模型或通用消息对象。"""

import base64
import copy
import fcntl
import getpass
import http.client
import json
import math
import os
import secrets
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from kairos.channels.base import InboundMessage, ReceiveBatch, ReplyTarget, SendReceipt, TransportError

DEFAULT_BASE = "https://ilinkai.weixin.qq.com"
APP_HEADERS = {"iLink-App-Id": "bot", "iLink-App-ClientVersion": "132106"}
BASE_INFO = {"channel_version": "2.4.10", "bot_agent": "Kairos/0.1"}
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class WeixinTimeout(TransportError):
    """区分接收的正常等待超时与发送结果未知。"""


def validate_base_url(url: str) -> str:
    """校验微信 HTTPS 根地址。

    作用:
        在凭据加载及扫码业务重定向时限制请求目标。
    参数:
        url: 配置或服务端返回的根地址。
    返回:
        去除末尾斜杠的地址 (str)。
    异常:
        TransportError: 地址包含非微信域名、非 HTTPS、认证、端口或额外路径。
    行为说明:
        1. 不发网络请求，也不回显被拒绝的地址。
    """
    try:
        parsed = urllib.parse.urlsplit(url) if isinstance(url, str) else None
        host = parsed.hostname if parsed else None
        valid = (
            parsed is not None
            and parsed.scheme == "https"
            and host
            and host.endswith(".weixin.qq.com")
            and parsed.netloc == host
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
            and not any(character.isspace() for character in url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise TransportError("微信 API 地址必须是微信域名下的 HTTPS 根地址")
    return f"https://{host}"


def _credentials(value: dict) -> dict:
    """检查并复制正式通道所需的凭据。

    作用:
        为文件导入、登录和适配器提供一致的字段校验。
    参数:
        value: 含 bot_token、bot_id、owner_id、base_url 的对象。
    返回:
        含必要字段与可选初始 checkpoint 的独立字典 (dict)。
    异常:
        TransportError: 凭据不是对象，必要字段无效，或初始 checkpoint 结构无效。
    行为说明:
        1. 不保留探针回复记录，不输出凭据内容。
    """
    keys = ("bot_token", "bot_id", "owner_id", "base_url")
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), str)
        or not value[key].strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value[key])
        for key in keys
    ):
        raise TransportError("微信登录凭据不完整或格式无效", authentication=True)
    if any(not 33 <= ord(character) <= 126 for character in value["bot_token"]):
        raise TransportError("微信登录凭据格式无效", authentication=True)
    result = {**{key: value[key] for key in keys}, "base_url": validate_base_url(value["base_url"])}
    if "initial_checkpoint" in value:
        initial = value["initial_checkpoint"]
        if not isinstance(initial, dict) or not isinstance(initial.get("cursor", ""), str):
            raise TransportError("微信初始接收位置格式无效")
        tokens = initial.get("context_tokens", {})
        if not isinstance(tokens, dict) or any(
            not isinstance(key, str) or not isinstance(token, str) or not token for key, token in tokens.items()
        ):
            raise TransportError("微信初始回复状态格式无效")
        result["initial_checkpoint"] = {
            **({"cursor": initial["cursor"]} if "cursor" in initial else {}),
            **({"context_tokens": dict(tokens)} if "context_tokens" in initial else {}),
        }
    return result


def _check_business(result: dict) -> None:
    """检查微信业务返回码。

    作用:
        防止 HTTP 成功掩盖平台拒绝，供真客户端与假客户端共用。
    参数:
        result: 已解析的响应对象。
    返回:
        None。
    异常:
        TransportError: 响应不是对象，或任一非空业务码不为零。
    行为说明:
        1. 缺少业务码不推断失败；错误只展示数值码，不展示服务端正文。
        2. 任一错误码为 -14 时立即标记需要重新登录。
    """
    if not isinstance(result, dict):
        raise TransportError("微信响应不是 JSON 对象")
    if any(type(result.get(key)) in (int, float) and result[key] == -14 for key in ("ret", "errcode")):
        raise TransportError("微信登录凭据已失效，请重新登录", authentication=True)
    for key in ("ret", "errcode"):
        code = result.get(key)
        if code not in (None, 0):
            label = str(code) if type(code) is int else "格式未知"
            raise TransportError(f"微信返回业务错误：{key}={label}", retryable=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁止 urllib 自动重定向登录查询和带凭据的请求。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """拒绝创建重定向请求。

        作用:
            保证请求仅发往事先校验的微信地址。
        参数:
            req, fp, code, msg, headers, newurl: urllib 提供的重定向参数。
        返回:
            None，让 urllib 报告 HTTP 错误。
        行为说明:
            1. 业务响应中的扫码重定向由登录函数单独校验。
        """
        return None


class WeixinClient:
    """实现有界、无自动重试的微信 HTTPS JSON 请求。"""

    def __init__(self, base_url: str = DEFAULT_BASE, token: str = ""):
        """绑定微信地址及可选凭据。

        作用:
            为登录与正式通道提供独立的网络边界。
        参数:
            base_url: 微信 HTTPS 根地址。
            token: 登录前为空，登录后为 Bot token。
        返回:
            无返回值。
        异常:
            TransportError: token 不是可安全放入请求头的 ASCII 字符串。
        行为说明:
            1. 不读取用户配置，不建立网络连接。
        """
        self.base_url = validate_base_url(base_url)
        if not isinstance(token, str) or any(not 33 <= ord(character) <= 126 for character in token):
            raise TransportError("微信登录凭据格式无效", authentication=True)
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, endpoint: str, payload: dict | None = None, timeout: float = 35.0) -> dict:
        """提交一次微信请求并返回脱离传输层的对象。

        作用:
            收发均复用同一请求结构，但发送失败保留结果未知标记。
        参数:
            endpoint: 代码生成的 /ilink/bot/ 路径及查询。
            payload: None 表示 GET，字典表示 JSON POST。
            timeout: 等待响应的上限秒数。
        返回:
            响应字典；空响应为 {}。
        异常:
            TransportError: 路径非法、请求无法编码、HTTP 或网络失败、响应超限或格式错误。
            WeixinTimeout: 网络等待超时。
        行为说明:
            1. GET 仅带应用头；POST 增加协议头和可选 Bearer。
            2. 不输出原始异常、响应正文、URL 或令牌，不自动重发。
        """
        if (
            not isinstance(endpoint, str)
            or not endpoint.startswith("/ilink/bot/")
            or any(ord(character) < 32 for character in endpoint)
        ):
            raise TransportError("微信 API 路径无效")
        sending = endpoint == "/ilink/bot/sendmessage"
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
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except (ValueError, TypeError, UnicodeError):
                raise TransportError("微信请求内容无法编码") from None
        request = urllib.request.Request(self.base_url + endpoint, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            authentication = error.code in (401, 403)
            raise TransportError(
                f"微信 HTTP 请求失败，状态码 {error.code}",
                retryable=not sending and (error.code == 429 or error.code >= 500),
                uncertain=sending and error.code >= 500,
                authentication=authentication,
            ) from None
        except (TimeoutError, urllib.error.URLError, OSError, http.client.HTTPException) as error:
            timed_out = isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError)
            error_type = WeixinTimeout if timed_out else TransportError
            raise error_type(
                "微信请求等待超时" if timed_out else "微信网络请求失败",
                retryable=not sending,
                uncertain=sending,
            ) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise TransportError("微信响应超过读取上限", uncertain=sending)
        try:
            result = json.loads(raw) if raw.strip() else {}
        except (ValueError, UnicodeError):
            raise TransportError("微信响应不是有效 JSON", uncertain=sending) from None
        if not isinstance(result, dict):
            raise TransportError("微信响应不是 JSON 对象", uncertain=sending)
        _check_business(result)
        return result


class WeixinAdapter:
    """把本人微信私聊转换为通用输入，并保持协议状态与模型隔离。"""

    channel = "weixin"

    def __init__(self, credentials: dict, client=None):
        """绑定一个微信 Bot 与其扫码用户。

        作用:
            明确当前实例的账号和唯一允许交互的本人身份。
        参数:
            credentials: 正式通道凭据。
            client: 可注入假客户端；默认创建标准库 HTTPS 客户端。
        返回:
            无返回值。
        行为说明:
            1. 不读取文件，不调用模型，不开始轮询。
        """
        credentials = _credentials(credentials)
        self.account_id = credentials["bot_id"]
        self.owner_id = credentials["owner_id"]
        self.client = client if client is not None else WeixinClient(credentials["base_url"], credentials["bot_token"])

    def _checkpoint(self, checkpoint: dict) -> dict:
        """复制并检查当前账号的收消息位置与回复令牌。

        作用:
            避免适配器修改协调层持有的已持久化状态。
        参数:
            checkpoint: 含可选 cursor 与 context_tokens 的字典。
        返回:
            仅保留本人对话令牌的新字典 (dict)。
        异常:
            TransportError: 游标或令牌存储结构无效。
        行为说明:
            1. 不向通用消息复制令牌，也不保留其他人的令牌。
        """
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("cursor", ""), str):
            raise TransportError("微信接收游标格式无效")
        tokens = checkpoint.get("context_tokens", {})
        if not isinstance(tokens, dict):
            raise TransportError("微信回复状态格式无效")
        token = tokens.get(self.owner_id)
        if token is not None and (not isinstance(token, str) or not token):
            raise TransportError("微信回复令牌格式无效")
        return {"cursor": checkpoint.get("cursor", ""), "context_tokens": {self.owner_id: token} if token else {}}

    def poll(self, checkpoint: dict) -> ReceiveBatch:
        """读取一批本人私聊并返回可原子保存的协议状态。

        作用:
            将微信输入交给协调层；游标与消息由协调层一起持久化。
        参数:
            checkpoint: 上次持久化的游标与按对话保存的回复令牌。
        返回:
            消息元组及新 checkpoint (ReceiveBatch)。超时返回空批次。
        异常:
            TransportError: 批次、游标或本人消息的标识与回复令牌无效。
        行为说明:
            1. 忽略他人、Bot、群组与尚在生成的片段。
            2. 完整纯文本保留正文，媒体或混合消息的 text 为 None。
            3. 不写磁盘，不修改传入状态，失败时不推进游标。
        """
        state = self._checkpoint(checkpoint)
        try:
            result = self.client.request(
                "/ilink/bot/getupdates",
                {"get_updates_buf": state["cursor"], "base_info": dict(BASE_INFO)},
                timeout=35.0,
            )
        except WeixinTimeout:
            return ReceiveBatch((), copy.deepcopy(checkpoint))
        _check_business(result)
        messages = result.get("msgs", [])
        cursor = result.get("get_updates_buf")
        if not isinstance(messages, list) or (cursor is not None and not isinstance(cursor, str)):
            raise TransportError("微信消息批次或游标格式无效")
        received = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("from_user_id") != self.owner_id
                or message.get("message_type") not in (None, 1)
                or (message.get("message_type") is not None and type(message["message_type"]) is not int)
                or message.get("to_user_id") not in (None, "", self.account_id)
                or message.get("group_id")
                or message.get("message_state") == 1
            ):
                continue
            message_state = message.get("message_state")
            if message_state is not None and (type(message_state) is not int or message_state not in (0, 2)):
                raise TransportError("微信本人消息状态无效")
            items = message.get("item_list", [])
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise TransportError("微信本人消息内容结构无效")
            if any(item.get("is_completed") is False for item in items):
                continue
            message_id = message.get("message_id")
            context_token = message.get("context_token")
            if (
                type(message_id) not in (int, str)
                or not str(message_id).strip()
                or (type(message_id) is int and message_id < 0)
                or not isinstance(context_token, str)
                or not context_token
            ):
                raise TransportError("微信本人消息缺少有效消息 ID 或回复令牌")
            text = None
            if items and all(
                type(item.get("type")) is int
                and item["type"] == 1
                and isinstance(item.get("text_item"), dict)
                and isinstance(item["text_item"].get("text"), str)
                for item in items
            ):
                text = "\n".join(item["text_item"]["text"] for item in items)
            received.append(
                InboundMessage(self.channel, self.account_id, self.owner_id, self.owner_id, str(message_id), text)
            )
            state["context_tokens"][self.owner_id] = context_token
        if cursor:
            state["cursor"] = cursor
        return ReceiveBatch(tuple(received), state)

    def send(self, target: ReplyTarget, text: str, client_id: str, checkpoint: dict) -> SendReceipt:
        """使用已有回复令牌向本人提交一次文字回复。

        作用:
            只负责传输已生成答案，复用协调层持久化的发送标识。
        参数:
            target: 当前微信账号的本人对话。
            text: 待发送的非空文字。
            client_id: 协调层生成并保存的唯一发送标识。
            checkpoint: 包含本人 context_token 的协议状态。
        返回:
            是否收到明确数值零业务码 (SendReceipt)，不是手机回执。
        异常:
            TransportError: 目标、文字、发送标识或回复令牌无效。
        行为说明:
            1. 不调用模型，不重试发送；网络失败的未知结果由协调层处理。
            2. 账号鉴权失效立即上抛，不尝试自动续期。
        """
        if not isinstance(target, ReplyTarget) or (
            target.channel != self.channel
            or target.account_id != self.account_id
            or target.conversation_id != self.owner_id
        ):
            raise TransportError("拒绝向当前微信本人对话以外的目标发送消息")
        if not isinstance(text, str) or not text or not isinstance(client_id, str) or not client_id.strip():
            raise TransportError("微信回复文字或发送标识无效")
        token = self._checkpoint(checkpoint)["context_tokens"].get(self.owner_id)
        if not token:
            raise TransportError("尚无微信回复令牌，请先从手机发送消息")
        result = self.client.request(
            "/ilink/bot/sendmessage",
            {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": self.owner_id,
                    "client_id": client_id,
                    "message_type": 2,
                    "message_state": 2,
                    "context_token": token,
                    "item_list": [{"type": 1, "text_item": {"text": text}}],
                },
                "base_info": dict(BASE_INFO),
            },
            timeout=15.0,
        )
        _check_business(result)
        return SendReceipt(
            any(type(result.get(key)) in (int, float) and result[key] == 0 for key in ("ret", "errcode"))
        )


def _read_json(path: Path, limit: int) -> dict:
    """有界读取私有 JSON 文件。

    作用:
        读取明确指定的账号或探针文件，不扩展到其他用户数据。
    参数:
        path: 指定文件路径。
        limit: 允许读取的最大字节数。
    返回:
        JSON 对象 (dict)。
    异常:
        TransportError: 文件为符号链接、不可读、超限或 JSON 结构无效。
    行为说明:
        1. 不在异常中包含文件内容或原始系统错误。
    """
    try:
        if path.is_symlink():
            raise TransportError("拒绝读取符号链接形式的微信凭据文件")
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        value = json.loads(raw) if len(raw) <= limit else None
    except (OSError, ValueError, UnicodeError):
        raise TransportError("微信状态文件不可读或内容无效") from None
    if not isinstance(value, dict):
        raise TransportError("微信状态文件超出上限或不是对象")
    return value


def read_credentials(path: Path) -> dict:
    """读取并校验正式微信账号凭据。

    作用:
        为 CLI 构造适配器提供唯一账号文件入口。
    参数:
        path: 正式账号 JSON 文件路径。
    返回:
        经校验的凭据副本 (dict)。
    行为说明:
        1. 最多读取 64 KiB，不读取 .env、模型配置或探针数据。
    """
    return _credentials(_read_json(Path(path), 64 * 1024))


def _private_directory(path: Path) -> None:
    """创建专用私有目录。

    作用:
        为账号原子写入和二维码临时文件准备存储位置。
    参数:
        path: 专用目录路径。
    返回:
        None。
    异常:
        TransportError: 目录是符号链接或无法创建及调整权限。
    行为说明:
        1. 仅修改指定目录权限为 700。
    """
    try:
        if path.is_symlink():
            raise TransportError("拒绝使用符号链接形式的微信状态目录")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
    except OSError:
        raise TransportError("无法准备微信私有状态目录") from None


def _save_credentials(path: Path, credentials: dict, *, replace: bool) -> None:
    """通过私有临时文件原子保存账号凭据。

    作用:
        保证登录和导入不会写出半份凭据，也不会意外覆盖现有账号。
    参数:
        path: 正式账号路径。
        credentials: 已校验的四字段凭据。
        replace: 是否明确允许替换现有文件。
    返回:
        None。
    异常:
        TransportError: 目标已存在而未允许覆盖，或写入失败。
    行为说明:
        1. 文件权限 600，写入并 fsync 后使用原子替换或独占链接。
        2. 失败也清理本次临时文件；原有账号保持完整。
    """
    _private_directory(path.parent)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".account-", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(credentials, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    except FileExistsError:
        raise TransportError("微信账号文件已存在；没有覆盖现有凭据") from None
    except OSError:
        raise TransportError("微信账号凭据保存失败；未输出原始系统错误") from None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_credentials(path: Path, creds: dict) -> None:
    """校验并私有保存正式账号凭据。

    作用:
        为明确需要更新凭据的调用方提供原子写入口。
    参数:
        path: 正式账号文件路径。
        creds: 完整凭据，可附带 initial_checkpoint。
    返回:
        None。
    行为说明:
        1. 校验成功后替换指定文件；导入与登录另行限制覆盖。
    """
    _save_credentials(Path(path), _credentials(creds), replace=True)


def import_probe(probe_path: Path, account_path: Path) -> dict:
    """从已验证探针复制凭据并返回正式通道初始游标。

    作用:
        复用扫码结果，保持探针文件原样，避免重新触发旧测试消息。
    参数:
        probe_path: 用户指定的探针状态文件。
        account_path: 尚不存在的正式账号文件。
    返回:
        含 cursor 和空 context_tokens 的初始 checkpoint (dict)。
    异常:
        TransportError: 账号已存在、探针锁不可用或被占用，或探针缺少有效游标。
    行为说明:
        1. 持有探针非阻塞互斥锁，不复制回复记录；校验完成后才写账号。
        2. 账号文件同时保留初始游标，保证后续数据库初始化可恢复。
    """
    account_path = Path(account_path)
    if account_path.exists() or account_path.is_symlink():
        raise TransportError("微信账号已存在；没有覆盖账号或导入探针")
    probe_path = Path(probe_path)
    lock_path = probe_path.parent / ".lock"
    if lock_path.is_symlink():
        raise TransportError("拒绝使用符号链接形式的微信探针锁")
    try:
        lock_file = lock_path.open("a+")
        os.fchmod(lock_file.fileno(), 0o600)
    except OSError:
        raise TransportError("无法打开微信探针互斥锁") from None
    with lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise TransportError("微信探针仍在运行，请停止探针后再导入") from None
        except OSError:
            raise TransportError("无法取得微信探针互斥锁") from None
        try:
            state = _read_json(probe_path, MAX_RESPONSE_BYTES)
            credentials = _credentials(state.get("credentials"))
            if not isinstance(state.get("cursor"), str):
                raise TransportError("探针状态缺少有效接收游标")
            checkpoint = {"cursor": state["cursor"], "context_tokens": {}}
            credentials["initial_checkpoint"] = checkpoint
            _save_credentials(account_path, credentials, replace=False)
            return checkpoint
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def login(account_path: Path, *, timeout: float = 300, replace: bool = False) -> None:
    """生成本地二维码并等待本人扫码登录。

    作用:
        仅获取通道凭据，不读取业务配置或自动发送微信消息。
    参数:
        account_path: 正式账号保存路径。
        timeout: 本次等待扫码的最长秒数，必须为有限正数。
        replace: 是否明确允许替换已存在账号。
    返回:
        None，成功后保存凭据并打印不含账号信息的确认。
    异常:
        TransportError: 账号已存在、等待时间无效、缺少二维码依赖、二维码字段
            不完整、生成失败、重定向无效、微信拒绝登录或等待超时。
    行为说明:
        1. 二维码负载只在本地编码；仅打印私有图片路径。
        2. 配对码由 getpass 输入不回显；所有退出路径清除二维码。
        3. 新凭据写入成功前不改原账号，未允许覆盖时原子检查目标不存在。
    """
    account_path = Path(account_path)
    if not replace and (account_path.exists() or account_path.is_symlink()):
        raise TransportError("微信账号已存在；重新登录须明确允许替换")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise TransportError("微信扫码等待时间必须为有限正数")
    try:
        import qrcode
    except ImportError:
        raise TransportError("生成微信二维码需要可选依赖 qrcode[pil]") from None
    client = WeixinClient()
    qr = client.request("/ilink/bot/get_bot_qrcode?bot_type=3", {"local_token_list": []})
    if not all(isinstance(qr.get(key), str) and qr[key] for key in ("qrcode", "qrcode_img_content")):
        raise TransportError("微信未返回完整二维码字段")
    _private_directory(account_path.parent)
    image_path = None
    try:
        try:
            fd, filename = tempfile.mkstemp(prefix="login-", suffix=".png", dir=account_path.parent)
            image_path = Path(filename)
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                qrcode.make(qr["qrcode_img_content"]).save(stream, format="PNG")
        except Exception:
            raise TransportError("微信二维码生成失败") from None
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
                    timeout=min(35.0, max(0.1, deadline - time.monotonic())),
                )
            except WeixinTimeout:
                continue
            _check_business(result)
            status = result.get("status")
            if not isinstance(status, str):
                raise TransportError("微信登录状态格式无效")
            if status == "confirmed":
                credentials = _credentials(
                    {
                        "bot_token": result.get("bot_token"),
                        "bot_id": result.get("ilink_bot_id"),
                        "owner_id": result.get("ilink_user_id"),
                        "base_url": result.get("baseurl") or DEFAULT_BASE,
                        "initial_checkpoint": {},
                    }
                )
                _save_credentials(account_path, credentials, replace=replace)
                print("微信扫码登录成功；凭据已私有保存。", flush=True)
                return
            if status == "scaned_but_redirect":
                host = result.get("redirect_host")
                if not isinstance(host, str) or not host:
                    raise TransportError("微信扫码重定向缺少目标地址")
                client = WeixinClient(validate_base_url("https://" + host))
            elif status == "need_verifycode":
                verify_code = getpass.getpass("请输入手机显示的配对码（输入不回显）：").strip()
            elif status == "scaned":
                verify_code = None
                if last_status != status:
                    print("已扫码，正在等待微信授权确认。", flush=True)
            elif status != "wait":
                errors = {
                    "expired": "微信二维码已过期，请重新登录",
                    "verify_code_blocked": "微信已阻止本次配对码验证，请检查手机提示",
                    "binded_redirect": "微信提示已有绑定，但未返回新凭据；原账号没有变化",
                }
                raise TransportError(errors.get(status, "微信返回未识别的登录状态"))
            last_status = status
            if status != "need_verifycode":
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise TransportError("微信扫码等待超时，请重新登录")
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
