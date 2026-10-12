"""真实终端与接口错误复核发现的边界。"""

import io
import json

import pytest
from PIL import Image

from kairos.images import prepare_image
from kairos.runtime import create_runtime
from kairos.terminal_input import _image_paths


@pytest.mark.parametrize("text", ["~10MB 的图片怎么压缩？", "x" * 500 + ".png", "~不存在的用户/a.png"])
def test_ordinary_text_path_probe_never_raises(text):
    assert _image_paths(text) == ()


def test_slash_escaped_api_error_never_exposes_jpeg_in_exception_or_logs(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 15), "blue").save(buffer, "JPEG")
    image = prepare_image(buffer.getvalue())
    error_text = json.dumps({"image": f"data:image/jpeg;base64,{image.data}", "data": image.data}).replace("/", "\\/")

    class Model:
        def complete(self, *, messages, tools):
            raise RuntimeError(error_text)

    events = []
    log = tmp_path / "error.jsonl"
    with create_runtime(model=Model(), db_path=tmp_path / "test.sqlite3", log_path=log) as runtime:
        with pytest.raises(RuntimeError) as raised:
            runtime.respond("读图", images=[image], on_event=events.append)
        for observed in (str(raised.value), json.dumps(events), log.read_text()):
            assert image.data not in observed.replace("\\/", "/")
            assert "data:image" not in observed
