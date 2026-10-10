from kairos.chat import run
from kairos.runtime import AgentTurnResult


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

    def respond(self, prompt):
        self.prompts.append(prompt)
        return AgentTurnResult(
            content=f"收到：{prompt}",
            messages=[{"role": "assistant", "content": f"收到：{prompt}"}],
            usage=None,
            finish_reason=None,
            events=[],
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
