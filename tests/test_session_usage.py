"""通过离线模型和临时数据库验证会话用量、计价快照及终端切换。"""

import sqlite3
from copy import deepcopy
from io import StringIO

import pytest
from rich.console import Console

from kairos import cli, pricing
from kairos.pricing import Usage
from kairos.runtime import create_runtime
from kairos.terminal_input import InputDraft

FIRST = {"prompt_tokens": 1000, "cached_input_tokens": 400, "completion_tokens": 30, "reasoning_tokens": 10}
SECOND = {"prompt_tokens": 1500, "cached_input_tokens": 500, "completion_tokens": 40, "reasoning_tokens": 15}
THIRD = {"prompt_tokens": 2000, "cached_input_tokens": 600, "completion_tokens": 50, "reasoning_tokens": 20}
FIRST_COST = 0.00169
SECOND_COST = 0.00262
THIRD_COST = 0.00355
TOOL_CALL = {"id": "list_once", "function": {"name": "list_todos", "arguments": "{}"}}


class MeteredModel:
    """按测试脚本提供完整响应或错误，不访问模型服务。"""

    model = "session-usage-test"

    def __init__(self, responses):
        """保存当前测试使用的离线响应序列。

        作用:
            为实际 Runtime 调用提供可重复的输入，不持有数据库或网络资源。

        参数:
            responses: 按调用顺序排列的响应字典或异常对象。

        返回:
            无返回值（None）。

        行为说明:
            1. 使用迭代器，意外的额外请求会耗尽响应并使测试失败。
        """
        self.responses = iter(responses)

    def complete(self, *, messages, tools):
        """返回下一条离线响应或触发预设的请求错误。

        作用:
            替代模型边界，让生产 Agent 循环和保存逻辑照常执行。

        参数:
            messages: Runtime 准备的完整请求消息。
            tools: Runtime 提供的工具定义。

        返回:
            下一条响应的独立副本（dict）。

        异常:
            BaseException: 下一条预设值是异常对象时显式抛出该对象。

        行为说明:
            1. 每次调用消费一条脚本记录，不模拟存储或统计结果。
        """
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)


class UsageInput:
    """记录固定统计栏的刷新内容和每次输入开始时的可见状态。"""

    def __init__(self, inputs, output):
        """保存离线输入并准备统计栏的观察记录。

        作用:
            替代交互输入边界，使真实 CLI、渲染器和 Runtime 连接到可检查的底栏。

        参数:
            inputs: 按读取顺序排列的输入文字。
            output: 收集真实 Rich 输出的内存文本流。

        返回:
            无返回值（None）。

        行为说明:
            1. 分别记录底栏更新和输入时看到的统计，检查刷新时机。
        """
        self.inputs = iter(inputs)
        self.output = output
        self.usage = None
        self.updates = []
        self.read_usages = []
        self.outputs_at_update = []
        self.clear_count = 0
        self.closed = False

    def set_usage(self, text):
        """记录 CLI 推送的最新两行统计。

        作用:
            观察生产渲染器传入底栏的内容及该时刻已输出的消息。

        参数:
            text: 渲染器生成的纯文字统计。

        返回:
            无返回值（None）。

        行为说明:
            1. 覆盖当前内容，同时保留更新历史用于验证切换和失败收尾。
        """
        self.usage = text
        self.updates.append(text)
        self.outputs_at_update.append(self.output.getvalue())

    def read(self, draft=None):
        """记录输入开始前的统计并返回下一条离线草稿。

        作用:
            驱动真实 CLI 的对话与命令循环，确认用户输入前底栏已经更新。

        参数:
            draft: CLI 传入的可选恢复草稿，本测试使用预设输入替代。

        返回:
            下一条离线输入草稿（InputDraft）。

        行为说明:
            1. 每次读取只消费一条输入，不自行修改会话统计。
        """
        self.read_usages.append(self.usage)
        return InputDraft(next(self.inputs))

    def clear(self):
        """记录 CLI 在切换会话后清理输入的动作。

        作用:
            保持输入编辑器接口完整，统计内容仍由生产渲染器更新。

        返回:
            无返回值（None）。

        行为说明:
            1. 只累计清理次数，不模拟会话数据变化。
        """
        self.clear_count += 1

    def close(self):
        """记录 CLI 退出时关闭输入编辑器的动作。

        作用:
            让测试确认注入的交互编辑器正常结束生命周期。

        返回:
            无返回值（None）。

        行为说明:
            1. 将关闭状态置为真，不再触发统计更新。
        """
        self.closed = True


@pytest.fixture(autouse=True)
def fixed_prices(monkeypatch):
    """为测试模型配置与时间无关的独立费率。

    作用:
        在请求开始前提供已知价格，让保存过程实际生成费用快照。

    参数:
        monkeypatch: pytest 的临时替换工具。

    返回:
        无返回值（None）。

    行为说明:
        1. 高峰和非高峰使用相同费率，不修改真实模型的价格表。
        2. 设置独立窗口，便于验证上下文没有使用累计输入。
    """
    monkeypatch.setitem(pricing.PRICES, MeteredModel.model, {"peak": (1, 2, 3), "off_peak": (1, 2, 3)})
    monkeypatch.setitem(pricing.CONTEXT_WINDOW, MeteredModel.model, 10000)


def test_multiple_turns_include_every_tool_step_once(tmp_path):
    """验证多回合累计包含工具续轮且保持单轮结果独立。

    作用:
        检查同一会话内三次真实模型调用只计入一次，并保留可查询的单轮数据。

    参数:
        tmp_path: 当前测试独立的临时目录。

    返回:
        无返回值（None）；通过断言验证公开结果和持久化步骤关联。

    行为说明:
        1. 第一轮查询待办并续轮回答，第二轮直接回答。
        2. 分别核对回合累计、会话累计、最新请求和数据库中的回合关联。
    """
    model = MeteredModel(
        [
            {"tool_calls": [TOOL_CALL], "usage": FIRST},
            {"content": "没有待办", "usage": SECOND},
            {"content": "继续回答", "usage": THIRD},
        ]
    )
    with create_runtime(model=model, db_path=tmp_path / "usage.sqlite3") as runtime:
        first = runtime.respond("查询待办")
        second = runtime.respond("继续")
        summary = runtime.session_usage()
        steps = runtime.session_details()["steps"]

    assert first.usage == Usage(2500, 900, 70, 25)
    assert first.session_usage.usage == first.usage
    assert first.session_usage.step_count == 2
    assert second.usage == Usage(2000, 600, 50, 20)
    assert second.session_usage == summary
    assert summary.usage == Usage(4500, 1500, 120, 45)
    assert summary.cost == pytest.approx(FIRST_COST + SECOND_COST + THIRD_COST)
    assert summary.step_count == 3
    assert summary.unpriced_steps == summary.legacy_estimated_steps == 0
    assert summary.last_request_usage == Usage(2000, 600, 50, 20)
    assert [step["turn_id"] for step in steps] == [first.turn_id, first.turn_id, second.turn_id]


def test_new_reset_and_resume_keep_session_totals_separate(tmp_path):
    """验证新建与重置归零且恢复会话后接续原有统计。

    作用:
        保护会话切换的边界，避免进程累计值串入其他会话。

    参数:
        tmp_path: 当前测试独立的临时目录。

    返回:
        无返回值（None）；断言每个会话只含自己的模型调用。

    行为说明:
        1. 分别在两个会话执行请求，再重置出第三个空会话。
        2. 切回首个会话继续请求，并再次核对其他会话未变化。
    """
    model = MeteredModel(
        [{"content": "A", "usage": FIRST}, {"content": "B", "usage": SECOND}, {"content": "A 继续", "usage": THIRD}]
    )
    with create_runtime(model=model, db_path=tmp_path / "switch.sqlite3") as runtime:
        session_a = runtime.session_id
        runtime.respond("A 的问题")
        runtime.new_session("B")
        session_b = runtime.session_id
        empty = runtime.session_usage()
        assert empty.usage == Usage(0, 0, 0, 0)
        assert empty.cost == empty.step_count == empty.unpriced_steps == 0
        assert empty.last_request_usage is None
        runtime.respond("B 的问题")
        runtime.reset()
        session_c = runtime.session_id
        assert runtime.session_usage() == empty

        runtime.resume_session(session_a)
        assert runtime.session_usage().usage == Usage(1000, 400, 30, 10)
        result = runtime.respond("A 的后续问题")
        assert result.session_usage.usage == Usage(3000, 1000, 80, 30)
        assert result.session_usage.cost == pytest.approx(FIRST_COST + THIRD_COST)
        runtime.resume_session(session_b)
        assert runtime.session_usage().usage == Usage(1500, 500, 40, 15)
        assert runtime.session_usage().cost == pytest.approx(SECOND_COST)
        runtime.resume_session(session_c)
        assert runtime.session_usage() == empty


def test_restart_preserves_saved_costs_and_prices_only_new_requests(tmp_path, monkeypatch):
    """验证重启不会重算旧费用且新请求使用当前费率。

    作用:
        检查费用来自持久化快照，避免换价或重复恢复改变已保存的累计结果。

    参数:
        tmp_path: 当前测试独立的临时目录。
        monkeypatch: pytest 的临时替换工具。

    返回:
        无返回值（None）；断言两次重启前后的用量和费用保持一致。

    行为说明:
        1. 保存第一轮后关闭连接，将测试费率整体提高十倍。
        2. 恢复并执行新轮，再次关闭与恢复，核对只增加一次新轮费用。
    """
    database = tmp_path / "restart.sqlite3"
    with create_runtime(model=MeteredModel([{"content": "初次回答", "usage": FIRST}]), db_path=database) as runtime:
        session_id = runtime.session_id
        runtime.respond("第一次提问")

    monkeypatch.setitem(pricing.PRICES, MeteredModel.model, {"peak": (10, 20, 30), "off_peak": (10, 20, 30)})
    with create_runtime(
        model=MeteredModel([{"content": "继续回答", "usage": THIRD}]), db_path=database, session_id=session_id
    ) as restored:
        assert restored.session_usage().cost == pytest.approx(FIRST_COST)
        result = restored.respond("恢复后继续")
        summary = result.session_usage
        assert summary.usage == Usage(3000, 1000, 80, 30)
        assert summary.cost == pytest.approx(FIRST_COST + 10 * THIRD_COST)
        assert summary.step_count == 2

    with create_runtime(model=MeteredModel([]), db_path=database, session_id=session_id) as reopened:
        assert reopened.session_usage() == summary
        assert reopened.session_usage() == summary
        snapshots = [step["cost_snapshot"] for step in reopened.session_details()["steps"]]
        assert [snapshot["cost_usd"] for snapshot in snapshots] == pytest.approx([FIRST_COST, 10 * THIRD_COST])


@pytest.mark.parametrize("error", [OSError("离线模拟失败"), KeyboardInterrupt()])
def test_failed_or_cancelled_request_retains_saved_usage_and_unknown_gap(tmp_path, error):
    """验证失败和中断不会丢弃此前完成请求的费用或伪造未知用量。

    作用:
        覆盖工具续轮第二次请求未返回用量的实际失败路径，并检查恢复后的统计。

    参数:
        tmp_path: 当前测试独立的临时目录。
        error: 离线模型在第二次请求显式抛出的错误或中断。

    返回:
        无返回值（None）；断言已知费用保留，缺失步骤仍标为未知。

    行为说明:
        1. 第一请求正常返回工具调用和用量，第二请求失败或中断。
        2. 重启恢复后再完成新回合，核对未重复计费且未知缺项仍保留。
    """
    database = tmp_path / "failure.sqlite3"
    model = MeteredModel([{"tool_calls": [TOOL_CALL], "usage": FIRST}, error])
    with create_runtime(model=model, db_path=database) as runtime:
        session_id = runtime.session_id
        with pytest.raises(type(error)):
            runtime.respond("查询待办")
        summary = runtime.session_usage()
        assert summary.usage == Usage(None, None, None, None)
        assert summary.cost == pytest.approx(FIRST_COST)
        assert summary.step_count == 2
        assert summary.unpriced_steps == 1
        assert summary.last_request_usage is None

    with create_runtime(
        model=MeteredModel([{"content": "已恢复", "usage": SECOND}]), db_path=database, session_id=session_id
    ) as restored:
        assert restored.session_usage() == summary
        result = restored.respond("继续")
        assert result.session_usage.usage == Usage(None, None, None, None)
        assert result.session_usage.cost == pytest.approx(FIRST_COST + SECOND_COST)
        assert result.session_usage.step_count == 3
        assert result.session_usage.unpriced_steps == 1
        assert result.session_usage.last_request_usage == Usage(1500, 500, 40, 15)


@pytest.mark.parametrize(
    ("field", "expected", "expected_cost", "unpriced"),
    [
        ("cached_input_tokens", Usage(2500, None, 70, 25), FIRST_COST, 1),
        ("reasoning_tokens", Usage(2500, 900, 70, None), FIRST_COST + SECOND_COST, 0),
    ],
)
def test_missing_field_preserves_other_totals_and_known_costs(tmp_path, field, expected, expected_cost, unpriced):
    """验证单个未知字段不会抹掉其他统计且不会被补成零。

    作用:
        检查供应商只返回部分用量时的会话累计和费用完整性标记。

    参数:
        tmp_path: 当前测试独立的临时目录。
        field: 第二次响应中缺失的字段名称。
        expected: 逐字段累计后的预期用量。
        expected_cost: 可以从已保存信息估算出的费用。
        unpriced: 无法估算费用的预期请求数量。

    返回:
        无返回值（None）；断言未知值和已知值分别正确保留。

    行为说明:
        1. 完成两轮真实请求，只从第二条响应移除一个用量字段。
        2. 验证缓存缺失阻止该步计价，思考明细缺失不影响输出计费。
    """
    partial = {name: value for name, value in SECOND.items() if name != field}
    model = MeteredModel([{"content": "第一轮", "usage": FIRST}, {"content": "第二轮", "usage": partial}])
    with create_runtime(model=model, db_path=tmp_path / "partial.sqlite3") as runtime:
        runtime.respond("第一次")
        result = runtime.respond("第二次")
        assert result.session_usage.usage == expected
        assert result.session_usage.cost == pytest.approx(expected_cost)
        assert result.session_usage.unpriced_steps == unpriced


@pytest.mark.parametrize("footer", [False, True])
def test_cli_shows_current_session_on_startup_turn_and_each_switch(tmp_path, monkeypatch, footer):
    """验证终端启动、回答结束及会话切换均立即展示当前统计。

    作用:
        使用实际 CLI 与 Rich 输出验证用户可见结果，不替换渲染或统计接口。

    参数:
        tmp_path: 当前测试独立的临时目录。
        monkeypatch: pytest 的临时替换工具，用于提供离线终端输入。
        footer: 为真时通过交互输入编辑器显示固定统计栏。

    返回:
        无返回值（None）；断言展示顺序、费用与上下文分别属于当前会话。

    行为说明:
        1. 保存旧会话后关闭连接，通过 CLI 启动选项恢复旧会话并连续提问两轮。
        2. 执行新建、恢复及重置，检查无需发送新请求就显示对应统计。
        3. 交互输入只更新底栏，普通终端继续输出统计文字。
    """
    database = tmp_path / "cli.sqlite3"
    with create_runtime(model=MeteredModel([{"content": "旧回答", "usage": FIRST}]), db_path=database) as runtime:
        session_id = runtime.session_id
        runtime.respond("旧问题")

    output = StringIO()
    console = Console(file=output, width=240, color_system=None)
    inputs = iter(["继续回答", "再次回答", "/new 新会话", f"/resume {session_id}", "/reset", "/exit"])
    editor = UsageInput(inputs, output) if footer else None
    monkeypatch.setattr(console, "input", lambda prompt: next(inputs))
    with create_runtime(
        model=MeteredModel([{"content": "新回答", "usage": SECOND}, {"content": "再次回答", "usage": THIRD}]),
        db_path=database,
    ) as runtime:
        cli.run(console=console, runtime=runtime, session_id=session_id, input_editor=editor)

    text = output.getvalue()
    if footer:
        assert editor.read_usages == editor.updates
        assert len(editor.updates) == 6
        assert all(len(update.splitlines()) == 2 for update in editor.updates)
        assert all("[" not in update for update in editor.updates)
        assert editor.clear_count == 3
        assert editor.closed
        assert "上下文窗口" not in text
        assert "会话输入" not in text
        assert "会话费用" not in text
        stats = "\n".join(editor.updates)
    else:
        stats = text
    usage_lines = [line for line in stats.splitlines() if line.startswith("会话输入")]
    assert len(usage_lines) == 6
    assert [line.split("（", 1)[0] for line in usage_lines] == [
        "会话输入 1,000",
        "会话输入 2,500",
        "会话输入 4,500",
        "会话输入 0",
        "会话输入 4,500",
        "会话输入 0",
    ]
    assert [line.rsplit(" · ", 1)[1] for line in usage_lines] == [
        "会话费用 $0.00169",
        "会话费用 $0.00431",
        "会话费用 $0.00786",
        "会话费用 $0.00000",
        "会话费用 $0.00786",
        "会话费用 $0.00000",
    ]
    assert "上下文窗口 1,000 / 10,000（10.00%）" in stats
    assert stats.count("上下文窗口 1,500 / 10,000（15.00%）") == 1
    assert stats.count("上下文窗口 2,000 / 10,000（20.00%）") == 2
    assert "25.00%" not in stats
    assert "45.00%" not in stats
    assert "本轮输入" not in stats
    assert "本轮费用" not in stats


@pytest.mark.parametrize(
    ("error", "notice"),
    [(OSError("离线续轮失败"), "本轮执行失败：OSError: 离线续轮失败"), (KeyboardInterrupt(), "已中断。")],
)
@pytest.mark.parametrize("footer", [False, True])
def test_cli_immediately_shows_partial_cost_after_failure_or_interrupt(tmp_path, monkeypatch, error, notice, footer):
    """验证工具续轮失败或中断后立即展示已保存的费用。

    作用:
        保护失败收尾的用户可见统计，避免等待下一次成功回答才刷新已发生的消耗。

    参数:
        tmp_path: 当前测试独立的临时目录。
        monkeypatch: pytest 的临时替换工具，用于提供离线终端输入。
        error: 第二次模型请求触发的错误或中断。
        notice: CLI 对应错误分支应展示的文字。
        footer: 为真时通过交互输入编辑器显示固定统计栏。

    返回:
        无返回值（None）；断言失败提示后立即出现部分费用且能够退出。

    行为说明:
        1. 第一请求正常返回用量并执行工具，第二请求失败或中断。
        2. 随后直接退出，不发送成功回合，检查已知费用与未知缺项均已显示。
    """
    model = MeteredModel([{"tool_calls": [TOOL_CALL], "usage": FIRST}, error])
    output = StringIO()
    console = Console(file=output, width=240, color_system=None)
    inputs = iter(["查询待办", "/exit"])
    editor = UsageInput(inputs, output) if footer else None
    monkeypatch.setattr(console, "input", lambda prompt: next(inputs))
    with create_runtime(model=model, db_path=tmp_path / "cli-failure.sqlite3") as runtime:
        cli.run(console=console, runtime=runtime, input_editor=editor)
        assert runtime.session_usage().step_count == 2

    text = output.getvalue()
    if footer:
        assert editor.read_usages == editor.updates
        assert len(editor.updates) == 2
        assert notice in editor.outputs_at_update[1]
        assert "上下文窗口 未知" in editor.updates[1]
        assert "上下文窗口" not in text
        assert "会话输入" not in text
        assert "会话费用" not in text
        assert editor.closed
        stats = "\n".join(editor.updates)
    else:
        stats = text
    usage_lines = [line for line in stats.splitlines() if line.startswith("会话输入")]
    assert len(usage_lines) == 2
    assert usage_lines[0].startswith("会话输入 0")
    assert usage_lines[1].startswith("会话输入 未知")
    assert "会话费用 $0.00169（部分未估算）" in usage_lines[1]
    if not footer:
        assert text.index(notice) < text.index(usage_lines[1])
        assert "上下文窗口 未知" in text[text.index(notice) :]


@pytest.mark.parametrize("footer", [False, True])
def test_cli_usage_read_error_preserves_request_error_and_interaction(tmp_path, monkeypatch, footer):
    """验证统计读取失败不会覆盖原请求错误或终止命令交互。

    作用:
        检查失败收尾的统计属于辅助展示，数据库读取出错时仍能说明原始故障。

    参数:
        tmp_path: 当前测试独立的临时目录。
        monkeypatch: pytest 的临时替换工具，用于离线输入和统计读故障注入。
        footer: 为真时通过交互输入编辑器显示固定统计栏。

    返回:
        无返回值（None）；断言原错误、未知统计和后续帮助按顺序展示。

    行为说明:
        1. 使用实际 Runtime 触发模型错误，仅让随后的统计读取抛出数据库异常。
        2. 接着执行帮助和退出，验证 CLI 仍可交互且没有改写已保存的用量。
    """
    model = MeteredModel([{"tool_calls": [TOOL_CALL], "usage": FIRST}, OSError("原请求失败")])
    output = StringIO()
    console = Console(file=output, width=240, color_system=None)
    inputs = iter(["查询待办", "/help", "/exit"])
    editor = UsageInput(inputs, output) if footer else None
    monkeypatch.setattr(console, "input", lambda prompt: next(inputs))
    with create_runtime(model=model, db_path=tmp_path / "cli-read-failure.sqlite3") as runtime:
        read_usage = runtime.session_usage
        reads = []

        def fail_second_read():
            """仅在请求错误后的统计读取注入数据库故障。

            作用:
                保留启动时的真实读取，让故障发生在本测试关注的失败收尾边界。

            返回:
                原接口提供的会话统计，故障调用没有返回值。

            异常:
                sqlite3.OperationalError: 第二次调用时模拟统计数据库不可读。

            行为说明:
                1. 记录调用次数，除第二次外均委托真实 Runtime 查询。
            """
            reads.append(None)
            if len(reads) == 2:
                raise sqlite3.OperationalError("模拟统计读取失败")
            return read_usage()

        monkeypatch.setattr(runtime, "session_usage", fail_second_read)
        cli.run(console=console, runtime=runtime, input_editor=editor)
        assert len(reads) == 2
        assert read_usage().cost == pytest.approx(FIRST_COST)

    text = output.getvalue()
    assert "本轮执行失败：OSError: 原请求失败" in text
    if footer:
        assert len(editor.updates) == 2
        assert editor.read_usages == [editor.updates[0], editor.updates[1], editor.updates[1]]
        assert editor.updates[1] == "上下文窗口 未知\n会话用量未知 · 会话费用未估算"
        assert "原请求失败" in editor.outputs_at_update[1]
        assert "/new 标题" not in editor.outputs_at_update[1]
        assert "/new 标题" in text
        assert "上下文窗口" not in text
        assert "会话输入" not in text
        assert "会话用量未知" not in text
        assert "会话费用" not in text
        assert editor.closed
    else:
        assert "会话用量未知 · 会话费用未估算" in text
        assert text.index("原请求失败") < text.index("会话用量未知") < text.index("/new 标题")
    assert "模拟统计读取失败" not in text
