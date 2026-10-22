"""定义通道收发与协调层之间的最小通用数据结构。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InboundMessage:
    """表示一条平台输入；text 为 None 时表示暂不支持的消息类型。"""

    channel: str
    account_id: str
    conversation_id: str
    sender_id: str
    message_id: str
    text: str | None


@dataclass(frozen=True)
class ReceiveBatch:
    """保存新消息及接收进度；connection_confirmed 区分实际响应与超时空批次。"""

    messages: tuple[InboundMessage, ...]
    checkpoint: dict
    connection_confirmed: bool = True


@dataclass(frozen=True)
class ReplyTarget:
    """标识回复所属的平台、机器人账号与对话，不携带平台凭据。"""

    channel: str
    account_id: str
    conversation_id: str


@dataclass(frozen=True)
class SendReceipt:
    """表示请求已被接口接受；acknowledged 仅指明确的业务成功码，不是手机回执。"""

    acknowledged: bool


class TransportError(Exception):
    """表示可安全展示的收发错误，不包含原始消息、地址或凭据。"""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        uncertain: bool = False,
        authentication: bool = False,
    ) -> None:
        """保存网络错误的处理边界。

        作用:
            让协调层区分可重连、投递结果未知和需要重新登录的情况。
        参数:
            message: 已脱敏、可向用户展示的说明。
            retryable: 是否适合稍后重试接收请求；不授权重跑模型或工具。
            uncertain: 发送是否可能已经被平台接收。
            authentication: 是否需要处理账号凭据。
        返回:
            无返回值。
        行为说明:
            1. 仅保存标记，不执行重试或记录原始异常。
        """
        super().__init__(message)
        self.retryable = retryable
        self.uncertain = uncertain
        self.authentication = authentication


class ChannelAdapter(Protocol):
    """约束同步收发接口；checkpoint 的内部格式只由适配器解释。"""

    channel: str
    account_id: str
    owner_id: str

    def poll(self, checkpoint: dict) -> ReceiveBatch:
        """从已持久化的接收位置读取一批输入。

        作用:
            把平台协议消息转换为通用输入，不执行模型或写入通道存储。
        参数:
            checkpoint: 上次已保存、仅由适配器解释的协议状态。
        返回:
            新消息与对应新状态组成的 ReceiveBatch。
        行为说明:
            1. 返回值由协调层原子保存，不能提前推进持久接收进度。
        """
        ...

    def send(self, target: ReplyTarget, text: str, client_id: str, checkpoint: dict) -> SendReceipt:
        """向指定对话提交一段已经生成的文字。

        作用:
            提供独立于 Runtime 的平台发送边界。
        参数:
            target: 账号与对话目标。
            text: 已保存的答案片段。
            client_id: 发送前已经持久化的本次片段标识。
            checkpoint: 适配器所需的当前回复上下文，不传入模型。
        返回:
            接口接受请求的依据 (SendReceipt)，不代表手机收到或已读。
        行为说明:
            1. 不重新生成答案，不自行重试结果未知的发送。
        """
        ...
