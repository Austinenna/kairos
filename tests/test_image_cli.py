"""图文草稿进入真实 Runtime 后的 CLI 提交与恢复。"""

import io

from PIL import Image

from kairos.cli import run
from kairos.images import prepare_image
from kairos.runtime import create_runtime
from kairos.terminal_input import InputDraft
from kairos.todos import list_todos


class Console:
    def __init__(self):
        self.output = []

    def print(self, *values, **kwargs):
        self.output.append(str(values))

    def print_json(self, **kwargs):
        self.output.append(str(kwargs))


class Editor:
    def __init__(self, drafts):
        self.drafts = iter(drafts)
        self.restored = []
        self.cleared = 0
        self.closed = False

    def read(self, draft=None):
        self.restored.append(draft)
        next_draft = next(self.drafts)
        return next_draft(draft) if callable(next_draft) else next_draft

    def clear(self):
        self.cleared += 1

    def close(self):
        self.closed = True


def picture():
    data = io.BytesIO()
    Image.new("RGB", (40, 30), "blue").save(data, "PNG")
    return prepare_image(data.getvalue(), "截图.png")


def test_cli_restores_processed_image_draft_after_rolled_back_turn(tmp_path):
    image = picture()
    first = InputDraft("这是什么颜色？", (image,))

    def retry(restored):
        assert restored == first
        assert restored.images[0] is image
        return InputDraft("请重新看图", restored.images)

    editor = Editor([first, retry, InputDraft("/history"), InputDraft("/reset"), InputDraft("/exit")])
    console = Console()

    class Model:
        calls = 0

        def stream(self, *, messages, tools, on_text):
            self.calls += 1
            assert len(messages) == 2
            block = next(block for block in messages[-1]["content"] if block["type"] == "image")
            assert block["data"] == image.data
            if self.calls == 1:
                raise OSError("模拟断流")
            on_text("蓝色")
            return {"content": "蓝色", "tool_calls": [], "finish_reason": "stop"}

    with create_runtime(model=Model(), db_path=tmp_path / "todos.sqlite3") as runtime:
        run(console=console, runtime=runtime, input_editor=editor)
        assert len(runtime.messages) == 1
    assert editor.cleared == 1
    assert editor.closed
    assert image.data not in "".join(console.output)
    assert any("草稿已恢复" in line for line in console.output)


def test_cli_does_not_restore_or_repeat_image_turn_after_tool_side_effect(tmp_path):
    image = picture()
    editor = Editor([InputDraft("根据图片创建待办", (image,)), InputDraft("检查结果"), InputDraft("/exit")])

    class Model:
        calls = 0

        def stream(self, *, messages, tools, on_text):
            self.calls += 1
            if self.calls == 1:
                return {
                    "content": None,
                    "tool_calls": [{"id": "add_1", "function": {"name": "add_todo", "arguments": '{"title":"测试"}'}}],
                }
            if self.calls == 2:
                raise KeyboardInterrupt
            assert [m["role"] for m in messages] == ["system", "user", "user", "assistant", "tool", "user"]
            assert messages[1]["content"].startswith("【Kairos 全局记忆参考】")
            assert messages[-1]["content"] == "检查结果"
            return {"content": "已保留", "tool_calls": [], "finish_reason": "stop"}

    console = Console()
    with create_runtime(model=Model(), db_path=tmp_path / "todos.sqlite3") as runtime:
        run(console=console, runtime=runtime, input_editor=editor)
        assert len(list_todos(runtime.connection)) == 1
    assert editor.restored == [None, None, None]
    assert any("本轮历史已保留" in line for line in console.output)


def test_file_command_submits_its_text_with_images_and_system_clears_draft(tmp_path):
    text_file = tmp_path / "问题.txt"
    text_file.write_text("描述图片中的颜色", encoding="utf-8")
    image = picture()
    editor = Editor([InputDraft(f"/file {text_file}", (image,)), InputDraft("/system 新提示词"), InputDraft("/exit")])

    class Model:
        def complete(self, *, messages, tools):
            content = messages[-1]["content"]
            assert any(b.get("text") == "描述图片中的颜色" for b in content)
            assert any(b.get("data") == image.data for b in content)
            return {"content": "蓝色", "tool_calls": []}

    with create_runtime(model=Model(), db_path=tmp_path / "todos.sqlite3") as runtime:
        run(console=Console(), runtime=runtime, input_editor=editor)
        assert runtime.system_prompt == "新提示词"
    assert editor.cleared == 1


def test_file_retry_never_reinterprets_file_body_as_control_command(tmp_path):
    text_file = tmp_path / "prompt.txt"
    text_file.write_text("/reset", encoding="utf-8")
    draft = InputDraft(f"/file {text_file}", (picture(),))

    def retry(restored):
        assert restored == draft
        return restored

    editor = Editor([draft, retry, InputDraft("/exit")])

    class Model:
        calls = 0

        def complete(self, *, messages, tools):
            self.calls += 1
            assert messages[-1]["content"][0]["text"] == "/reset"
            if self.calls == 1:
                raise OSError("模拟网络失败")
            return {"content": "已收到文件正文", "tool_calls": []}

    model = Model()
    with create_runtime(model=model, db_path=tmp_path / "test.sqlite3") as runtime:
        run(console=Console(), runtime=runtime, input_editor=editor)
        assert model.calls == 2
        assert len(runtime.messages) == 3
        assert editor.cleared == 0
