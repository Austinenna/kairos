from kairos.cli import run
from kairos.runtime import AgentTurnResult, create_runtime
from kairos.terminal_input import TerminalInputUnsupportedError


class FakeConsole:
    def __init__(self, inputs):
        self.inputs = iter(inputs)
        self.printed = []

    def input(self, prompt):
        self.printed.append(("input", prompt))
        return next(self.inputs)

    def print(self, *values, **kwargs):
        self.printed.append(("print", values, kwargs))

    def print_json(self, **kwargs):
        self.printed.append(("json", kwargs))


class FakeRuntime:
    model_name = "fake-model"
    system_prompt = "系统提示"

    def __init__(self):
        self.prompts = []
        self.resets = []
        self.thinking = []

    def respond(self, prompt, *, on_event=None):
        self.prompts.append(prompt)
        if on_event:
            on_event({"event": "model_request", "request_number": 1})
            on_event({"event": "text_delta", "request_number": 1, "text": f"收到：{prompt}"})
            on_event({"event": "model_response", "request_number": 1})
        return AgentTurnResult(
            content=f"收到：{prompt}",
            messages=[{"role": "assistant", "content": f"收到：{prompt}"}],
            usage=None,
            finish_reason=None,
            events=[],
            tool_results=[],
        )

    def history(self):
        return [{"role": "assistant", "content": "历史"}]

    def reset(self, system_prompt=None):
        self.resets.append(system_prompt)

    def set_thinking(self, enabled):
        self.thinking.append(enabled)


def test_main_chat_uses_injected_agent_runtime_and_keeps_commands():
    console = FakeConsole(["创建一个待办", "/reset", "/think", "/history", "/exit"])
    runtime = FakeRuntime()

    run(console=console, runtime=runtime)

    assert runtime.prompts == ["创建一个待办"]
    assert runtime.resets == [None]
    assert runtime.thinking == [True]
    assert any(item[0] == "json" for item in console.printed)
    assert sum("收到：创建一个待办" in str(item) for item in console.printed) == 1


def test_cli_streams_during_respond_and_can_continue_after_interrupt(tmp_path):
    console = FakeConsole(["第一次", "第二次", "/exit"])

    class Model:
        calls = 0

        def stream(self, *, messages, tools, on_text):
            self.calls += 1
            if self.calls == 1:
                on_text("第一轮的部分正文")
                assert any("第一轮的部分正文" in str(item) for item in console.printed)
                raise KeyboardInterrupt
            assert [m["role"] for m in messages] == ["system", "user"]
            assert messages[-1]["content"] == "第二次"
            on_text("第二轮的完整回答")
            assert any("第二轮的完整回答" in str(item) for item in console.printed)
            return {"content": "第二轮的完整回答", "tool_calls": [], "finish_reason": "stop"}

    with create_runtime(model=Model(), db_path=tmp_path / "todos.sqlite3") as runtime:
        run(console=console, runtime=runtime)
        assert runtime.history()[-1]["content"] == "第二轮的完整回答"
    assert sum("第二轮的完整回答" in str(item) for item in console.printed) == 1
    assert any("已中断" in str(item) for item in console.printed)


def test_dumb_terminal_falls_back_to_plain_text_input(monkeypatch):
    from types import SimpleNamespace

    from kairos import cli

    def unsupported(**kwargs):
        raise TerminalInputUnsupportedError("模拟受限终端")

    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli, "TerminalInput", unsupported)
    console = FakeConsole(["你好", "/exit"])
    console.is_terminal = True
    runtime = FakeRuntime()
    runtime.check_image_budget = lambda _: None
    run(console=console, runtime=runtime)
    assert runtime.prompts == ["你好"]
    assert any("文字输入模式" in str(item) for item in console.printed)
