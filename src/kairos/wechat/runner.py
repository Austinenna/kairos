"""微信入口：后台线程里循环“收消息 → 图片归档、文字交给会话 → 回复”。

微信和终端共用同一个 Services，各自持有自己的 Conversation。
处理每条消息时持有 services.lock，和终端轮流使用数据库，不会同时写。

状态文件 state.json 记录三样东西，重启后接着用：
  cursor         收消息的游标，保证重启后不会把旧消息再回复一遍
  context_token  回复时要带的令牌，来自对方最近一条消息
  session_id     微信当前对应的会话
"""

import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from ..conversation import Conversation
from ..services import Services
from .api import Credentials, Incoming, WeixinClient, WeixinError

HELP = "直接发文字就可以对话；发图片会自动存档并分类。\n/new 开始新会话\n/help 显示本帮助"
REPLY_BYTES = 1800  # 单条微信消息的长度上限（UTF-8 字节），更长的回答拆成几条发
RETRY_SECONDS = 10  # 网络出错后隔多久再试


class WeixinRunner:
    def __init__(self, services: Services, data_dir: Path):
        """data_dir 是 ~/.kairos/channels/weixin，里面有 account.json（凭据）和 state.json（状态）。"""
        self.services = services
        self.client = WeixinClient(Credentials.load(data_dir / "account.json"))
        self.state_path = data_dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.conversation: Conversation | None = None
        self.notices: Queue[str] = Queue()  # 给终端显示的状态，例如断线、登录失效
        self.stopping = Event()
        self.thread = Thread(target=self._run, name="kairos-weixin", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        """请求停止。正在进行的长轮询最多再等几十秒，线程是 daemon，退出时不必等它。"""
        self.stopping.set()

    def drain_notices(self) -> list[str]:
        notices = []
        while True:
            try:
                notices.append(self.notices.get_nowait())
            except Empty:
                return notices

    def _run(self) -> None:
        failing = False  # 正在连续出错时只提示一次，恢复后再提示一次
        while not self.stopping.is_set():
            try:
                messages, self.state["cursor"] = self.client.poll(self.state.get("cursor", ""))
                self._save_state()
                for message in messages:
                    self._handle(message)
                if failing:
                    self.notices.put("微信连接已恢复。")
                    failing = False
            except WeixinError as error:
                if error.expired:
                    self.notices.put(f"{error}。微信入口已停止，重新登录：uv run python -m kairos.wechat")
                    return
                failing = self._report(error, failing)
            except Exception as error:  # 网络中断等：隔一会儿重试
                failing = self._report(error, failing)

    def _report(self, error: Exception, failing: bool) -> bool:
        if not failing:
            self.notices.put(f"微信连接出错，{RETRY_SECONDS} 秒后重试：{type(error).__name__}: {error}")
        self.stopping.wait(RETRY_SECONDS)
        return True

    def _handle(self, message: Incoming) -> None:
        """处理 owner 发来的一条消息：图片逐张归档，文字交给会话，全部处理完再回复。"""
        self.state["context_token"] = message.context_token
        self._save_state()
        if message.text is None:
            self._reply("目前只支持文字和图片。")
            return
        with self.services.lock:
            if self.stopping.is_set():
                replies = ["Kairos 正在关闭，这条消息没有处理，请稍后再发。"]
            else:
                replies = [self._archive(image_item) for image_item in message.images]
                if message.text.strip():
                    replies.append(self._answer(message.text.strip()))
        for reply in replies:
            self._reply(reply)

    def _archive(self, image_item: dict) -> str:
        """下载一张图片并归档，返回回执。"""
        try:
            data = self.client.download_image(image_item)
        except Exception as error:
            return f"图片下载失败：{type(error).__name__}。可以再发一次。"
        image = self.services.images.store(data, "weixin")
        try:
            image = self.services.images.classify(image)
        except Exception as error:
            return f"图片已存到 {image.path}，自动分类失败：{type(error).__name__}: {error}"
        return f"已存到 {image.path}\n{image.description}"

    def _answer(self, text: str) -> str:
        """回答一条文字消息；/new 和 /help 是命令。"""
        if text == "/new":
            self.conversation = Conversation.create(self.services, channel="weixin")
            self._remember_session()
            return "已开始新会话。"
        if text == "/help":
            return HELP
        try:
            return self._conversation().respond(text).content or "（没有生成回答）"
        except Exception as error:
            return f"出错了：{type(error).__name__}。没有自动重试，可以稍后再发一次。"

    def _conversation(self) -> Conversation:
        """取微信当前的会话：先用内存里的，再按 state.json 恢复，都没有就新建。"""
        if self.conversation is None:
            session_id = self.state.get("session_id")
            if session_id:
                self.conversation = Conversation(self.services, self.services.sessions.get(session_id))
            else:
                self.conversation = Conversation.create(self.services, channel="weixin")
                self._remember_session()
        return self.conversation

    def _remember_session(self) -> None:
        self.state["session_id"] = self.conversation.session.id
        self._save_state()

    def _reply(self, text: str) -> None:
        for part in split_reply(text):
            self.client.send(part, self.state["context_token"])

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state), encoding="utf-8")
        self.state_path.chmod(0o600)  # 里面有回复令牌，只允许当前用户读取


def split_reply(text: str, limit: int = REPLY_BYTES) -> list[str]:
    """把长回答按 UTF-8 字节数拆成几段，尽量在换行处断开。"""
    parts = []
    while len(text.encode("utf-8")) > limit:
        cut = len(text.encode("utf-8")[:limit].decode("utf-8", errors="ignore"))  # 不超过 limit 字节的最长前缀
        newline = text.rfind("\n", 0, cut)
        cut = newline + 1 if newline > 0 else cut
        parts.append(text[:cut])
        text = text[cut:]
    return [*parts, text]
