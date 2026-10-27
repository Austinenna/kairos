"""调度线程：每隔一会儿找出到期的定时任务，生成要推送的文字，交给对应入口推送。

一次执行：
  ① advance：先存好下次时间（一次性任务改为 done），再执行。中途崩溃也不会重复触发
  ② 生成文字：message 直接用原文；prompt 在新会话里跑一轮，模型可以调用 skip_push 表示这次不推送
  ③ 推送：交给入口登记的推送函数（outlets），入口负责发出并追加进自己的当前会话
  ④ record：记下 sent / skipped / failed 和原因

推送由程序完成，模型没有发送工具。执行时持有 services.lock，和终端、微信轮流使用数据库。
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from queue import Empty, Queue
from threading import Event, RLock, Thread

from ..tools.registry import Toolset, tool_schema
from .store import Schedule, ScheduleStore

TICK_SECONDS = 30
LATE_AFTER = timedelta(minutes=5)  # 比原定时间晚这么久才执行，就当作迟到（程序当时没有运行）

SKIP_PUSH = tool_schema(
    "skip_push",
    "按任务要求，这次不需要通知用户时调用。调用后不必再写内容。",
    {"reason": {"type": "string", "description": "为什么这次不通知，一句话"}},
)

TASK_RULES = """你正在执行一个定时任务，执行完成后，程序会把你的最终回答推送给用户。
如果按任务要求这次不需要通知用户，就调用 skip_push 并写明原因；否则直接写要推送的完整内容，不要加开场白。"""

# run_prompt(会话标题, 输入, 额外工具) → 模型的最终回答。由入口层组装，这里不导入 Conversation。
RunPrompt = Callable[[str, str, Toolset], str]


class ScheduleRunner:
    def __init__(
        self, store: ScheduleStore, lock: RLock, run_prompt: RunPrompt, outlets: dict[str, Callable[[str], None]]
    ):
        """outlets：入口名 → 推送函数，入口启动时登记进来，例如 outlets["weixin"]。"""
        self.store = store
        self.lock = lock
        self.run_prompt = run_prompt
        self.outlets = outlets
        self.notices: Queue[str] = Queue()  # 给终端显示的失败提示
        self.stopping = Event()
        self.thread = Thread(target=self._run, name="kairos-schedule", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        """不再开始新的任务。正在执行的任务由调用方拿 services.lock 等它结束。"""
        self.stopping.set()

    def drain_notices(self) -> list[str]:
        notices = []
        while True:
            try:
                notices.append(self.notices.get_nowait())
            except Empty:
                return notices

    def tick(self, now: datetime) -> None:
        """执行所有到期任务。启动后马上检查一次，程序关着时错过的任务会在这里补跑。"""
        with self.lock:
            for schedule in self.store.due(now):
                if self.stopping.is_set():
                    return
                self._execute(schedule, now)

    def _run(self) -> None:
        while not self.stopping.is_set():
            self.tick(datetime.now().astimezone())
            self.stopping.wait(TICK_SECONDS)

    def _execute(self, schedule: Schedule, now: datetime) -> None:
        planned = datetime.fromisoformat(schedule.next_run_at)
        self.store.advance(schedule, now)
        late = now - planned if now - planned > LATE_AFTER else None
        try:
            text, skip_reason = self._compose(schedule, now, planned, late)
            if skip_reason is not None:
                self.store.record(schedule.id, "skipped", skip_reason, now)
                return
            if not text.strip():
                raise RuntimeError("模型没有给出内容")
            if schedule.channel not in self.outlets:
                raise RuntimeError(f"{schedule.channel} 入口没有启动")
            self.outlets[schedule.channel](f"【{schedule.title}】\n{text}")
            self.store.record(schedule.id, "sent", None, now)
        except Exception as error:  # 后台线程里没人接异常：记下来，并提示终端
            note = f"{type(error).__name__}: {error}"
            self.store.record(schedule.id, "failed", note, now)
            self.notices.put(f"定时任务 #{schedule.id}「{schedule.title}」没有推送：{note}")

    def _compose(
        self, schedule: Schedule, now: datetime, planned: datetime, late: timedelta | None
    ) -> tuple[str, str | None]:
        """生成要推送的文字。返回 (文字, 跳过原因)，跳过原因不是 None 表示这次不推送。"""
        planned_text = planned.astimezone().strftime("%m-%d %H:%M")
        if schedule.kind == "message":
            note = f"（原定 {planned_text} 的提醒，当时程序没有运行）\n" if late else ""
            return note + schedule.content, None

        skipped: list[str] = []  # skip_push 被调用时把原因放进来

        def skip_push(reason: str) -> dict:
            skipped.append(reason)
            return {"ok": True, "note": "这次不推送，不需要再写内容，直接结束。"}

        prompt = schedule.content
        if late:
            prompt = (
                f"这次执行迟到了：原定 {planned_text}，现在 {now.strftime('%m-%d %H:%M')}（晚了 {_duration(late)}），"
                "程序当时没有运行。如果迟到后这条推送已经没有意义，就调用 skip_push 并写明原因。\n\n"
                f"任务：\n{prompt}"
            )
        tools = Toolset([SKIP_PUSH], {"skip_push": skip_push}, [TASK_RULES])
        text = self.run_prompt(f"定时任务 #{schedule.id} {schedule.title}", prompt, tools)
        return text, (skipped[0] if skipped else None)


def _duration(delta: timedelta) -> str:
    hours, minutes = divmod(int(delta.total_seconds()) // 60, 60)
    return f"{hours} 小时 {minutes} 分钟" if hours else f"{minutes} 分钟"
