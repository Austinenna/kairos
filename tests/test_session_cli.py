"""会话命令、启动入口与输入草稿的离线集成验证。"""

import io
import sqlite3
from types import SimpleNamespace

import pytest
from PIL import Image
from rich.console import Console

from kairos import cli, main
from kairos.images import prepare_image
from kairos.runtime import create_runtime
from kairos.terminal_input import InputDraft


class CaptureConsole:
    def __init__(self, inputs=()):
        self.inputs = iter(inputs)
        self.output = []
        self.json = []

    def input(self, prompt):
        return next(self.inputs)

    def print(self, *values, **kwargs):
        self.output.append(" ".join(str(value) for value in values))

    def print_json(self, *, data):
        self.json.append(data)


class Model:
    model = "offline-model"

    def complete(self, *, messages, tools):
        return {"content": "已收到", "tool_calls": [], "finish_reason": "stop"}


class Editor:
    def __init__(self, drafts):
        self.drafts = iter(drafts)
        self.cleared = 0
        self.closed = False

    def read(self, draft=None):
        value = next(self.drafts)
        return value(self, draft) if callable(value) else value

    def clear(self):
        self.cleared += 1

    def close(self):
        self.closed = True


def test_commands_switch_persisted_history_and_show_ids(tmp_path):
    with create_runtime(model=Model(), db_path=tmp_path / "sessions.sqlite3") as runtime:
        runtime.rename_session("会话 A")
        runtime.respond("A 的内容")
        original_id = runtime.session_id
        console = CaptureConsole(
            [
                "/new 会话 B",
                "B 的内容",
                "/sessions",
                f"/resume {original_id[:8]}",
                "/history --ids",
                "/rename [red]会话 A",
                "/think",
                "/exit",
            ]
        )

        cli.run(console=console, runtime=runtime)

        assert runtime.session_id == original_id
        assert runtime.session_info.title == "[red]会话 A"
        assert not runtime.thinking_enabled
        assert not runtime.session_info.thinking_enabled
        assert "A 的内容" in str(runtime.history())
        assert "B 的内容" not in str(runtime.history())
        assert len(runtime.list_sessions()) == 2
        assert set(console.json[0]) >= {"session", "turns", "steps", "messages", "tool_calls"}
        assert original_id in str(console.json[0])
        output = "\n".join(console.output)
        assert original_id[:8] in output
        assert "* " in output
        assert "\\[red]会话 A" in output
        assert "模型 offline-model" in output
        assert "思考切换未接入" in output
        assert "Kairos 尚未为当前模型接入思考开关" in output


def test_empty_session_list_and_invalid_commands_remain_interactive(tmp_path):
    console = CaptureConsole(["/sessions", "/resume", "/rename", "/history --wrong", "/resume missing", "/exit"])
    with create_runtime(model=Model(), db_path=tmp_path / "sessions.sqlite3") as runtime:
        original_id = runtime.session_id
        cli.run(console=console, runtime=runtime)
        assert runtime.session_id == original_id
    output = "\n".join(console.output)
    assert "还没有已保存的会话" in output
    assert output.count("命令执行失败") == 4


def test_failed_resume_keeps_attachments_successful_switch_clears_them(tmp_path):
    image_file = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(image_file, "PNG")
    image = prepare_image(image_file.getvalue(), "附件.png")

    def submit_after_failed_resume(editor, restored):
        # 斜杠命令的附件由输入编辑器持有。失败不能 clear，也不能用空附件草稿覆盖它。
        assert editor.cleared == 0
        assert restored is None
        return InputDraft("继续看图", (image,))

    with create_runtime(model=Model(), db_path=tmp_path / "sessions.sqlite3") as runtime:
        runtime.rename_session("原会话")
        original_id = runtime.session_id
        editor = Editor(
            [
                InputDraft("/resume missing"),
                submit_after_failed_resume,
                InputDraft("/new 新会话"),
                InputDraft(f"/resume {original_id}"),
                InputDraft("/exit"),
            ]
        )
        cli.run(console=CaptureConsole(), runtime=runtime, input_editor=editor)
        assert runtime.messages[1]["content"][-1]["data"] == image.data
        assert editor.cleared == 2
        assert editor.closed


def test_startup_restore_routes_to_runtime_and_does_not_close_injected_runtime(tmp_path):
    with create_runtime(model=Model(), db_path=tmp_path / "sessions.sqlite3") as runtime:
        runtime.rename_session("最初会话")
        original_id = runtime.session_id
        runtime.new_session(title="最近会话")
        latest_id = runtime.session_id
        cli.run(console=CaptureConsole(["/exit"]), runtime=runtime, session_id=original_id[:8])
        assert runtime.session_id == original_id
        # list_sessions 的排序由 Runtime 维护，--continue 使用它给出的首项。
        newest_id = runtime.list_sessions()[0].session_id
        cli.run(console=CaptureConsole(["/exit"]), runtime=runtime, resume_latest=True)
        assert runtime.session_id == newest_id
        assert {session.session_id for session in runtime.list_sessions()} == {original_id, latest_id}


def test_no_previous_session_reports_error_and_closes_editor(tmp_path):
    editor = Editor([])
    console = CaptureConsole()
    with create_runtime(model=Model(), db_path=tmp_path / "sessions.sqlite3") as runtime:
        cli.run(console=console, runtime=runtime, input_editor=editor, resume_latest=True)
        assert runtime.list_sessions() == []
    assert editor.closed
    assert "没有可以恢复的会话" in "\n".join(console.output)


def test_model_and_thinking_display_follow_restored_session(monkeypatch):
    class Runtime:
        session_id = "session-aaaaaaaa"
        model_name = "model-a"
        thinking_enabled = False
        system_prompt = "提示"
        recovered_turns = 0
        supports_thinking = True
        thinking_state_corrected = False

        @property
        def session_info(self):
            return SimpleNamespace(session_id=self.session_id, title=self.model_name)

        def resume_session(self, reference):
            self.session_id = "session-bbbbbbbb"
            self.model_name = "model-b"
            self.thinking_enabled = True

        def set_thinking(self, value):
            self.thinking_enabled = value

        def respond(self, *args, **kwargs):
            return object()

    models = []
    monkeypatch.setattr(
        cli.TerminalRenderer,
        "render_turn",
        lambda self, result, *, model, **kwargs: models.append(model) or 0.0,
    )
    console = CaptureConsole(["/resume b", "/think", "继续", "/exit"])
    runtime = Runtime()
    cli.run(console=console, runtime=runtime)
    assert models == ["model-b"]
    assert not runtime.thinking_enabled
    output = "\n".join(console.output)
    assert "模型 model-a · 思考 关" in output
    assert "模型 model-b · 思考 开" in output
    assert "思考模式：关" in output


@pytest.mark.parametrize(("initial", "restored"), [(1, 0), (0, 1)])
def test_startup_and_resume_report_interrupted_execution(initial, restored):
    class Runtime:
        session_id = "session-aaaaaaaa"
        session_info = SimpleNamespace(session_id=session_id, title="恢复会话")
        model_name = "offline-model"
        thinking_enabled = False
        recovered_turns = initial
        supports_thinking = False
        thinking_state_corrected = False

        def resume_session(self, reference):
            self.recovered_turns = restored

    console = CaptureConsole(["/resume saved", "/exit"])
    cli.run(console=console, runtime=Runtime())
    output = "\n".join(console.output)
    assert output.count("上次未完成的执行已标记中断") == 1
    assert "工具不会自动重跑" in output


def test_session_list_disambiguates_prefixes_and_escapes_titles():
    output = io.StringIO()
    runtime = SimpleNamespace(
        session_id="abcdefgh-1",
        list_sessions=lambda: [
            SimpleNamespace(session_id="abcdefgh-1", title="[red]原样标题[/]", updated_at="2026-10-16T00:00:00Z"),
            SimpleNamespace(session_id="abcdefgh-2", title="第二个会话", updated_at="2026-10-15T00:00:00Z"),
        ],
    )
    cli._show_sessions(runtime, cli.TerminalRenderer(Console(file=output, color_system=None)))
    text = output.getvalue()
    assert "* abcdefgh-1" in text
    assert "  abcdefgh-2" in text
    assert "[red]原样标题[/]" in text


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], {"session_id": None, "resume_latest": False}),
        (["--continue"], {"session_id": None, "resume_latest": True}),
        (["--session", "abc"], {"session_id": "abc", "resume_latest": False}),
    ],
)
def test_main_routes_session_arguments(monkeypatch, arguments, expected):
    calls = []
    monkeypatch.setattr(cli.sys, "argv", ["kairos", *arguments])
    monkeypatch.setattr(cli, "run", lambda **kwargs: calls.append(kwargs))
    main()
    assert calls == [expected]


def test_main_rejects_conflicting_session_options(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["kairos", "--continue", "--session", "abc"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--continue" in capsys.readouterr().err


def test_created_runtime_receives_options_and_is_closed(monkeypatch, tmp_path):
    runtime = create_runtime(model=Model(), db_path=tmp_path / "sessions.sqlite3")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return runtime

    monkeypatch.setattr(cli, "create_runtime", factory)
    settings = SimpleNamespace(kairos_api_key="offline-placeholder")
    cli.run(console=CaptureConsole(["/exit"]), settings=settings, session_id="saved")
    assert calls == [{"settings": settings, "session_id": "saved", "resume_latest": False}]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        runtime.connection.execute("SELECT 1")


def test_create_runtime_failure_is_friendly_and_closes_editor(monkeypatch):
    def factory(**kwargs):
        raise ValueError("找不到指定会话")

    monkeypatch.setattr(cli, "create_runtime", factory)
    console = CaptureConsole()
    editor = Editor([])
    cli.run(
        console=console,
        settings=SimpleNamespace(kairos_api_key="offline-placeholder"),
        input_editor=editor,
        session_id="missing",
    )
    assert editor.closed
    assert "无法启动会话：找不到指定会话" in "\n".join(console.output)
