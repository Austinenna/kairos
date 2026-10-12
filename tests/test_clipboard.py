import base64
import json
import subprocess
from pathlib import Path

import pytest

from kairos import clipboard


def returned(monkeypatch, payload):
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

    monkeypatch.setattr(clipboard.subprocess, "run", run)
    return calls


def test_macos_clipboard_decodes_images_with_fixed_script(monkeypatch):
    calls = returned(monkeypatch, {"image": base64.b64encode(b"image bytes").decode()})
    assert clipboard.read_clipboard() == clipboard.ClipboardContent(image=b"image bytes")
    args, options = calls[0]
    assert args[:4] == ["osascript", "-l", "JavaScript", "-e"]
    assert options["timeout"] == 5
    assert "shell" not in options


def test_finder_file_urls_remain_data_and_decode_spaces(monkeypatch):
    returned(monkeypatch, {"urls": ["file:///tmp/a%20b.png", "file:///tmp/%24%28oops%29.png"]})
    assert clipboard.read_clipboard().paths == (Path("/tmp/a b.png"), Path("/tmp/$(oops).png"))


def test_plain_clipboard_text(monkeypatch):
    returned(monkeypatch, {"text": "普通文字", "image": None, "urls": []})
    assert clipboard.read_clipboard() == clipboard.ClipboardContent(text="普通文字")


@pytest.mark.parametrize("payload", [[], {"image": "!invalid"}, {"urls": ["https://example.com/a.png"]}])
def test_malformed_clipboard_is_actionable_error(monkeypatch, payload):
    returned(monkeypatch, payload)
    with pytest.raises(clipboard.ClipboardError, match="剪贴板内容无法解析"):
        clipboard.read_clipboard()


def test_clipboard_timeout_does_not_expose_subprocess_output(monkeypatch):
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("osascript", 5, output="private clipboard text")

    monkeypatch.setattr(clipboard.subprocess, "run", timeout)
    with pytest.raises(clipboard.ClipboardError, match="超时") as caught:
        clipboard.read_clipboard()
    assert "private" not in str(caught.value)
