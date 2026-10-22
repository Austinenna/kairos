import base64
from io import BytesIO

from PIL import Image as PILImage

from kairos.images import prepare_image, text_of
from kairos.models.anthropic import to_anthropic_messages
from kairos.models.openai_compat import _to_openai
from kairos.terminal.prompt import image_paths


def png_bytes(width, height):
    buffer = BytesIO()
    PILImage.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_large_image_is_resized_and_becomes_a_block():
    image = prepare_image(png_bytes(4000, 1000), "截图.png")

    assert (image.width, image.height) == (2000, 500)
    block = image.to_block()
    assert block["type"] == "image" and block["mime_type"] == "image/png"
    assert PILImage.open(BytesIO(base64.b64decode(block["data"]))).size == (2000, 500)


def test_both_adapters_convert_image_blocks():
    image = prepare_image(png_bytes(10, 10), "a.png")
    user = {"role": "user", "content": [{"type": "text", "text": "看图"}, image.to_block()]}

    openai_content = _to_openai(user)["content"]
    _, anthropic_messages = to_anthropic_messages([user])

    assert openai_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert anthropic_messages[0]["content"][1]["source"]["media_type"] == "image/png"
    assert text_of(user["content"]) == "看图 [图片：a.png]"


def test_dragged_paths_are_recognized_as_images(tmp_path):
    picture = tmp_path / "my photo.png"
    picture.write_bytes(png_bytes(10, 10))

    assert image_paths(f"'{picture}'") == [picture]  # 带引号
    assert image_paths(str(picture).replace(" ", "\\ ")) == [picture]  # 反斜杠转义空格
    assert image_paths(f"看看这个 {picture}") == []  # 夹杂普通文字时不当作路径
