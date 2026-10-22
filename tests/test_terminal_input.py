import io
import threading
import time
from pathlib import Path

import pytest
from PIL import Image
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from kairos import terminal_input
from kairos.clipboard import ClipboardContent
from kairos.legacy_images import prepare_image
from kairos.terminal_input import InputDraft, TerminalInput, TerminalInputUnsupportedError


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (12, 8), "red").save(output, format="PNG")
    return output.getvalue()


def until(predicate):
    deadline = time.monotonic() + 4
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("终端输入测试等待状态超时")
        threading.Event().wait(0.01)


def paste(pipe, text):
    pipe.send_text("\x1b[200~" + text + "\x1b[201~")


def drive(pipe, editor, script, *, draft=None):
    failures = []

    def send():
        try:
            until(lambda: editor._session.app.is_running)
            script()
            until(lambda: not editor._session.app.is_running)
        except BaseException as exc:
            failures.append(exc)
            pipe.send_bytes(b"\x03")

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    try:
        result = editor.read(draft=draft)
    finally:
        worker.join(timeout=5)
        if failures:
            raise failures[0]
    return result


def test_plain_paste_never_reads_old_clipboard_image_and_empty_enter_stays_open():
    def unexpected():
        raise AssertionError("文字粘贴不应读取剪贴板")

    with create_pipe_input() as pipe:
        editor = TerminalInput(
            input=pipe, output=DummyOutput(), budget_check=lambda _: None, clipboard_reader=unexpected
        )
        try:

            def send():
                pipe.send_text("\r")
                paste(pipe, "解释 /tmp/photo.png 的文件名")
                pipe.send_text("\r")

            assert drive(pipe, editor, send) == InputDraft("解释 /tmp/photo.png 的文件名")
        finally:
            editor.close()


def test_ctrl_v_picture_and_question_submit_together():
    with create_pipe_input() as pipe:
        editor = TerminalInput(
            input=pipe,
            output=DummyOutput(),
            budget_check=lambda _: None,
            clipboard_reader=lambda: ClipboardContent(image=png_bytes()),
        )
        try:

            def send():
                pipe.send_text("\x16这是什么？")
                until(lambda: bool(editor._attachments) and editor._attachments[0].image is not None)
                pipe.send_text("\r")

            result = drive(pipe, editor, send)
            assert result.text == "这是什么？"
            assert len(result.images) == 1
            assert result.images[0].width == 12
            assert not editor._attachments
        finally:
            editor.close()


def test_backspace_removes_entire_image_and_unbinds_payload():
    image = prepare_image(png_bytes(), "[red]截图.png")
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=lambda _: None)
        try:
            result = drive(pipe, editor, lambda: pipe.send_text("\x7f\x7f正常文字\r"), draft=InputDraft("", (image,)))
            assert result == InputDraft("正常文字")
        finally:
            editor.close()


def test_removing_a_loading_attachment_prevents_late_reappearance(monkeypatch):
    started, release = threading.Event(), threading.Event()
    image = prepare_image(png_bytes(), "a.png")

    def load(path):
        started.set()
        assert release.wait(4)
        return image

    monkeypatch.setattr(terminal_input, "load_image", load)
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=lambda _: None)
        try:

            def send():
                paste(pipe, "/tmp/a.png")
                assert started.wait(4)
                pipe.send_text("\x7f\x7f保留这段文字")
                until(lambda: not editor._attachments)
                release.set()
                pipe.send_text("\r")

            assert drive(pipe, editor, send) == InputDraft("保留这段文字")
        finally:
            release.set()
            editor.close()


def test_images_load_serially_while_text_remains_editable(monkeypatch):
    first_started, release = threading.Event(), threading.Event()
    calls = []

    def load(path):
        calls.append(path.name)
        if len(calls) == 1:
            first_started.set()
            assert release.wait(4)
        return prepare_image(png_bytes(), path.name)

    monkeypatch.setattr(terminal_input, "load_image", load)
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=lambda _: None)
        try:

            def send():
                paste(pipe, "/tmp/a.png /tmp/b.png")
                assert first_started.wait(4)
                pipe.send_text("请比较两张图\r")
                until(lambda: "请比较" in editor._buffer.text)
                assert calls == ["a.png"]
                assert editor._session.app.is_running
                release.set()
                until(lambda: len(editor._attachments) == 2 and all(item.image for item in editor._attachments))
                pipe.send_text("\r")

            result = drive(pipe, editor, send)
            assert result.text == "请比较两张图"
            assert [image.name for image in result.images] == ["a.png", "b.png"]
        finally:
            release.set()
            editor.close()


def test_failed_budget_preserves_other_images_and_allows_reset(monkeypatch):
    monkeypatch.setattr(terminal_input, "load_image", lambda path: prepare_image(png_bytes(), path.name))

    def budget(images):
        if len(images) > 1:
            raise ValueError("最多一张图片")

    image = prepare_image(png_bytes(), "a.png")
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=budget)
        try:

            def send():
                paste(pipe, "/tmp/b.png")
                until(lambda: any(item.error for item in editor._attachments))
                pipe.send_text("\r/reset\r")

            assert drive(pipe, editor, send, draft=InputDraft("", (image,))) == InputDraft("/reset")
            assert len(editor._attachments) == 2
            editor.clear()
            assert drive(pipe, editor, lambda: pipe.send_text("重新开始\r")) == InputDraft("重新开始")
        finally:
            editor.close()


def test_help_keeps_attachments_and_file_command_submits_them():
    image = prepare_image(png_bytes(), "a.png")
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=lambda _: None)
        try:
            assert drive(pipe, editor, lambda: pipe.send_text("/help\r"), draft=InputDraft("", (image,))) == InputDraft(
                "/help"
            )
            result = drive(pipe, editor, lambda: pipe.send_text("/file notes.txt\r"))
            assert result.text == "/file notes.txt"
            assert result.images == (image,)
        finally:
            editor.close()


def test_raw_independent_image_path_becomes_attachment_on_enter(monkeypatch):
    calls = []

    def load(path):
        calls.append(path)
        return prepare_image(png_bytes(), path.name)

    monkeypatch.setattr(terminal_input, "load_image", load)
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=lambda _: None)
        try:

            def send():
                pipe.send_text("/tmp/a.png\r")
                until(lambda: bool(editor._attachments) and editor._attachments[0].image is not None)
                pipe.send_text("看一下\r")

            result = drive(pipe, editor, send)
            assert calls == [Path("/tmp/a.png")]
            assert result.text == "看一下"
            assert len(result.images) == 1
        finally:
            editor.close()


def test_plain_text_from_explicit_clipboard_shortcut():
    with create_pipe_input() as pipe:
        editor = TerminalInput(
            input=pipe,
            output=DummyOutput(),
            budget_check=lambda _: None,
            clipboard_reader=lambda: ClipboardContent(text="剪贴板文字"),
        )
        try:

            def send():
                pipe.send_text("前\x16后")
                until(lambda: "剪贴板文字" in editor._buffer.text)
                pipe.send_text("\r")

            assert drive(pipe, editor, send) == InputDraft("前剪贴板文字后")
        finally:
            editor.close()


def test_finder_clipboard_paths_are_attached_without_a_command(monkeypatch):
    calls = []

    def load(path):
        calls.append(path)
        return prepare_image(png_bytes(), path.name)

    monkeypatch.setattr(terminal_input, "load_image", load)
    with create_pipe_input() as pipe:
        editor = TerminalInput(
            input=pipe,
            output=DummyOutput(),
            budget_check=lambda _: None,
            clipboard_reader=lambda: ClipboardContent(paths=(Path("/tmp/a b.png"),)),
        )
        try:

            def send():
                pipe.send_text("\x16看一下")
                until(lambda: bool(editor._attachments) and editor._attachments[0].image is not None)
                pipe.send_text("\r")

            result = drive(pipe, editor, send)
            assert calls == [Path("/tmp/a b.png")]
            assert result.text == "看一下"
            assert result.image_labels == ("图片 1",)
        finally:
            editor.close()


def test_stable_image_labels_survive_deletion_and_draft_replay():
    images = (prepare_image(png_bytes(), "a.png"), prepare_image(png_bytes(), "b.png"))
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=lambda _: None)
        try:
            # Home + Delete 整体删除第一张；End 后输入仍使用第二张原有编号。
            result = drive(
                pipe,
                editor,
                lambda: pipe.send_text("\x01\x1b[3~\x05解释图片 7\r"),
                draft=InputDraft("", images, ("图片 3", "图片 7")),
            )
            assert result.text == "解释图片 7"
            assert result.images == (images[1],)
            assert result.image_labels == ("图片 7",)
            replay = drive(pipe, editor, lambda: pipe.send_text("\r"), draft=result)
            assert replay == result
        finally:
            editor.close()


def test_reset_during_load_cannot_restore_cleared_attachment(monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def load(path):
        started.set()
        assert release.wait(4)
        finished.set()
        return prepare_image(png_bytes(), path.name)

    monkeypatch.setattr(terminal_input, "load_image", load)
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=lambda _: None)
        try:

            def reset():
                paste(pipe, "/tmp/a.png")
                assert started.wait(4)
                pipe.send_text("/reset\r")

            assert drive(pipe, editor, reset) == InputDraft("/reset")
            editor.clear()
            release.set()
            assert finished.wait(4)
            assert drive(pipe, editor, lambda: pipe.send_text("新问题\r")) == InputDraft("新问题")
        finally:
            release.set()
            editor.close()


def test_dumb_terminal_fails_explicitly_instead_of_returning_unprocessed_text(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    with pytest.raises(TerminalInputUnsupportedError, match="终端类型不支持图片输入区"):
        TerminalInput(budget_check=lambda _: None)


def test_linefeed_uses_the_same_draft_submission_as_enter():
    with create_pipe_input() as pipe:
        editor = TerminalInput(input=pipe, output=DummyOutput(), budget_check=lambda _: None)
        try:
            assert drive(pipe, editor, lambda: pipe.send_text("普通文字\n")) == InputDraft("普通文字")
        finally:
            editor.close()
