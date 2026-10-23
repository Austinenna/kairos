import base64
from io import BytesIO

from PIL import Image as PILImage

from kairos.images import prepare_image, text_of
from kairos.models.anthropic import to_anthropic_messages
from kairos.models.openai_compat import _to_openai, _to_openai_messages
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


def test_image_in_tool_result_is_converted_for_both_protocols():
    image = prepare_image(png_bytes(10, 10), "a.png").to_block()
    call = {"id": "c1", "type": "function", "function": {"name": "view_image", "arguments": "{}"}}
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "结果"}, image]},
        {"role": "system", "content": "时间"},
    ]

    openai = _to_openai_messages(messages)
    _, anthropic = to_anthropic_messages(messages)

    # OpenAI：工具结果只留文字，图片放进紧跟其后的用户消息
    assert [m["role"] for m in openai] == ["assistant", "tool", "user", "system"]
    assert openai[1]["content"] == "结果" and openai[2]["content"][1]["type"] == "image_url"
    # Anthropic：图片直接放在 tool_result 里
    assert [block["type"] for block in anthropic[1]["content"][0]["content"]] == ["text", "image"]
