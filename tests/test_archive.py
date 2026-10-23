import json
from io import BytesIO

import pytest
from PIL import Image as PILImage

from kairos.archive import ImageArchive
from kairos.models import ModelResponse
from kairos.storage import open_db


def jpeg_bytes():
    buffer = BytesIO()
    PILImage.new("RGB", (20, 10), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


class FilingModel:
    """假模型：每次都把图片归到同一个分类和标题，并记下收到的请求。"""

    model = "deepseek-flash"

    def __init__(self, filing):
        self.filing = filing
        self.requests = []

    def stream(self, messages, tools, on_text=None):
        self.requests.append(messages)
        function = {"name": "file_image", "arguments": json.dumps(self.filing)}
        call = {"id": "c1", "type": "function", "function": function}
        return ModelResponse(content="", tool_calls=[call])


@pytest.fixture
def archive(tmp_path):
    model = FilingModel({"category": "票据", "title": "停车/票", "description": "停车 12 元"})
    archive = ImageArchive(open_db(tmp_path / "kairos.db"), tmp_path / "files", lambda: model)
    archive.model = model
    return archive


def test_image_is_filed_into_category_without_overwriting(archive, tmp_path):
    first = archive.classify(archive.store(jpeg_bytes(), "weixin"))
    second = archive.classify(archive.store(jpeg_bytes(), "weixin"))

    assert first.path.startswith("票据/") and first.path.endswith("-停车-票.jpg")  # 标题里的 / 不会变成子目录
    assert second.path.endswith("-停车-票-2.jpg")  # 同名时不覆盖第一张
    assert (tmp_path / "files" / first.path).read_bytes() == jpeg_bytes()  # 存的是原图
    assert not list((tmp_path / "files" / "inbox").iterdir())
    assert "已有分类：票据" in archive.model.requests[1][0]["content"]  # 第二次归档时带上已有分类


def test_image_stays_in_inbox_when_model_gives_no_filing(archive, tmp_path):
    archive.model.stream = lambda messages, tools, on_text=None: ModelResponse(content="这是一张图")
    image = archive.store(jpeg_bytes(), "weixin")

    with pytest.raises(ValueError):
        archive.classify(image)
    assert (tmp_path / "files" / image.path).exists() and image.path.startswith("inbox/")
