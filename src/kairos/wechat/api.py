"""微信 iLink Bot 协议：扫码登录、长轮询收消息、发文字消息、下载图片。

一个 Bot 只和扫码登录的那个人（owner）私聊，其他人的消息一律忽略。
收消息靠“游标”（get_updates_buf）：每次请求带上一次返回的游标，服务端只返回之后的新消息。
回复必须带上对方最近一条消息里的 context_token。
图片不在消息里，消息只给 CDN 下载地址和密钥；下载到的是 AES-128-ECB 加密的数据。
"""

import base64
import json
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
APP_HEADERS = {"iLink-App-Id": "bot", "iLink-App-ClientVersion": "132106"}
BASE_INFO = {"channel_version": "2.4.10", "bot_agent": "Kairos/0.1"}
POLL_SECONDS = 35  # 长轮询：服务端最多挂起这么久，没有新消息就返回空列表
TEXT, IMAGE = 1, 2  # item_list 里每一项的 type


class WeixinError(RuntimeError):
    def __init__(self, message: str, *, expired: bool = False):
        super().__init__(message)
        self.expired = expired  # True 表示登录凭据失效，需要重新扫码


@dataclass
class Credentials:
    bot_token: str
    bot_id: str
    owner_id: str  # 扫码登录的人，也是唯一会回复的人
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def load(cls, path: Path) -> "Credentials":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["bot_token"], data["bot_id"], data["owner_id"], data.get("base_url", DEFAULT_BASE_URL))


@dataclass
class Incoming:
    text: str | None  # 文字正文，只有图片时是空字符串；含有语音、文件等不支持的内容时为 None
    images: list[dict]  # 图片的 image_item，用 WeixinClient.download_image 下载
    context_token: str  # 回复这条消息时要带上


class WeixinClient:
    def __init__(self, credentials: Credentials):
        self.credentials = credentials

    def poll(self, cursor: str) -> tuple[list[Incoming], str]:
        """长轮询一次，返回 (owner 发来的新消息, 新游标)。等待超时就返回空列表和原游标。"""
        try:
            result = self._request(
                "/ilink/bot/getupdates", {"get_updates_buf": cursor, "base_info": BASE_INFO}, POLL_SECONDS + 5
            )
        except TimeoutError:
            return [], cursor
        messages = []
        for message in result.get("msgs", []):
            items = message.get("item_list", [])
            if (
                message.get("from_user_id") != self.credentials.owner_id
                or message.get("group_id")
                or message.get("message_state") == 1  # 1 表示还在输入中，等完整消息
                or any(item.get("is_completed") is False for item in items)
            ):
                continue
            texts = [item["text_item"]["text"] for item in items if item.get("type") == TEXT]
            images = [item["image_item"] for item in items if item.get("type") == IMAGE]
            if len(texts) + len(images) == len(items):
                messages.append(Incoming("\n".join(texts), images, message["context_token"]))
            else:
                messages.append(Incoming(None, [], message["context_token"]))
        return messages, result.get("get_updates_buf") or cursor

    def send(self, text: str, context_token: str) -> None:
        """给 owner 发一条文字消息。"""
        message = {
            "from_user_id": "",
            "to_user_id": self.credentials.owner_id,
            "client_id": secrets.token_hex(16),  # 每条消息一个唯一 ID
            "message_type": 2,  # 2 表示 Bot 发出的消息
            "message_state": 2,  # 2 表示消息已完整
            "context_token": context_token,
            "item_list": [{"type": TEXT, "text_item": {"text": text}}],
        }
        self._request("/ilink/bot/sendmessage", {"msg": message, "base_info": BASE_INFO}, 15)

    def download_image(self, image_item: dict) -> bytes:
        """下载并解密一张图片，返回原图字节。密钥是 image_item["aeskey"]（32 位 hex）。"""
        with urllib.request.urlopen(image_item["media"]["full_url"], timeout=30) as reply:
            encrypted = reply.read()
        decryptor = Cipher(algorithms.AES(bytes.fromhex(image_item["aeskey"])), modes.ECB()).decryptor()
        padded = decryptor.update(encrypted) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()  # 去掉加密时补齐到 16 字节整数倍的填充
        return unpadder.update(padded) + unpadder.finalize()

    def _request(self, endpoint: str, payload: dict, timeout: float) -> dict:
        return request(self.credentials.base_url, endpoint, payload, timeout, token=self.credentials.bot_token)


def request(base_url: str, endpoint: str, payload: dict | None, timeout: float, token: str = "") -> dict:
    """发一次 iLink 请求。payload 为 None 时是 GET，否则是 JSON POST。业务错误码非 0 时抛出 WeixinError。"""
    headers = dict(APP_HEADERS)
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers |= {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": base64.b64encode(str(secrets.randbits(32)).encode()).decode(),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(base_url + endpoint, data, headers), timeout=timeout) as reply:
        body = reply.read()
    result = json.loads(body) if body.strip() else {}
    for key in ("ret", "errcode"):
        code = result.get(key)
        if code == -14:
            raise WeixinError("微信登录已失效，请重新扫码登录", expired=True)
        if code not in (None, 0):
            raise WeixinError(f"微信返回错误：{key}={code}")
    return result


def login(account_path: Path) -> None:
    """生成二维码，等待手机扫码确认，把凭据保存到 account_path。需要可选依赖 qrcode[pil]。"""
    import qrcode  # 只有登录时才需要，运行时不依赖

    base_url = DEFAULT_BASE_URL
    qr = request(base_url, "/ilink/bot/get_bot_qrcode?bot_type=3", {"local_token_list": []}, 15)
    account_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = account_path.parent / "login-qrcode.png"
    qrcode.make(qr["qrcode_img_content"]).save(image_path)
    print(f"请用手机微信扫描二维码：{image_path}")
    try:
        query = {"qrcode": qr["qrcode"]}
        while True:
            try:
                status = request(base_url, "/ilink/bot/get_qrcode_status?" + urllib.parse.urlencode(query), None, 40)
            except TimeoutError:
                continue
            state = status.get("status")
            if state == "confirmed":
                credentials = {
                    "bot_token": status["bot_token"],
                    "bot_id": status["ilink_bot_id"],
                    "owner_id": status["ilink_user_id"],
                    "base_url": status.get("baseurl") or DEFAULT_BASE_URL,
                }
                account_path.write_text(json.dumps(credentials), encoding="utf-8")
                account_path.chmod(0o600)  # 凭据只允许当前用户读取
                print("登录成功。")
                return
            if state == "scaned_but_redirect":  # 服务端要求换到另一个地址继续查询
                base_url = "https://" + status["redirect_host"]
            elif state == "need_verifycode":
                query["verify_code"] = input("请输入手机上显示的配对码：").strip()
            elif state == "scaned":
                print("已扫码，请在手机上确认。")
            elif state != "wait":
                raise WeixinError(f"登录失败：{state}")
            time.sleep(1)
    finally:
        image_path.unlink(missing_ok=True)
