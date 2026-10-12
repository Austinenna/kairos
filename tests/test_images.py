import base64
import copy
import json
import random
from dataclasses import FrozenInstanceError, replace
from io import BytesIO

import pytest
from PIL import Image

from kairos import images
from kairos.images import (
    ImageInputError,
    image_budget_usage,
    load_image,
    prepare_image,
    redact_images,
    validate_image_budget,
)


def image_bytes(size=(12, 6), *, image_format="PNG", mode="RGB", color="red", exif=None):
    image = Image.new(mode, size, color)
    output = BytesIO()
    options = {"exif": exif} if exif is not None else {}
    image.save(output, format=image_format, **options)
    image.close()
    return output.getvalue()


def test_load_uses_real_format_and_keeps_a_snapshot_without_changing_source(tmp_path):
    original = image_bytes(image_format="JPEG")
    path = tmp_path / "格式实际是JPEG.png"
    path.write_bytes(original)

    snapshot = load_image(path)

    assert snapshot.mime_type == "image/jpeg"
    assert snapshot.name == path.name
    assert base64.b64decode(snapshot.data) == original
    assert path.read_bytes() == original
    assert snapshot.data not in repr(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.name = "其他图片"
    path.write_bytes("原文件后来已经变化".encode())
    assert base64.b64decode(snapshot.data) == original


def test_exif_orientation_is_applied_once_and_original_dimensions_are_retained():
    exif = Image.Exif()
    exif[274] = 6
    snapshot = prepare_image(image_bytes(size=(20, 10), image_format="JPEG", exif=exif))

    assert (snapshot.original_width, snapshot.original_height) == (20, 10)
    assert (snapshot.width, snapshot.height) == (10, 20)
    with Image.open(BytesIO(base64.b64decode(snapshot.data))) as decoded:
        assert decoded.getexif().get(274, 1) == 1
    repeated = prepare_image(base64.b64decode(snapshot.data))
    assert (repeated.width, repeated.height) == (10, 20)


@pytest.mark.parametrize(
    "image_format,mime_type", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")]
)
def test_supported_static_formats_remain_unchanged_when_within_limits(image_format, mime_type):
    payload = image_bytes(image_format=image_format)
    snapshot = prepare_image(payload)
    assert snapshot.mime_type == mime_type
    assert base64.b64decode(snapshot.data) == payload


def test_resize_preserves_aspect_ratio_and_transparency_without_upscaling(monkeypatch):
    monkeypatch.setattr(images, "MAX_DIMENSION", 10)
    snapshot = prepare_image(image_bytes(size=(40, 20), mode="RGBA", color=(20, 80, 160, 90)))

    assert (snapshot.width, snapshot.height) == (10, 5)
    assert snapshot.mime_type == "image/png"
    with Image.open(BytesIO(base64.b64decode(snapshot.data))) as decoded:
        assert decoded.mode == "RGBA"
        assert decoded.getpixel((2, 2))[3] == 90
    small = prepare_image(image_bytes(size=(6, 3)))
    assert (small.width, small.height) == (6, 3)


def test_png_transparency_color_key_is_preserved_when_resizing(monkeypatch):
    monkeypatch.setattr(images, "MAX_DIMENSION", 10)
    source = Image.new("RGB", (20, 10), (255, 0, 0))
    output = BytesIO()
    source.save(output, format="PNG", transparency=(255, 0, 0))
    source.close()

    snapshot = prepare_image(output.getvalue())

    with Image.open(BytesIO(base64.b64decode(snapshot.data))) as decoded:
        assert decoded.mode == "RGBA"
        assert decoded.getpixel((2, 2))[3] == 0


def test_oversized_png_can_be_compressed_with_a_reported_new_mime_type(monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 9000)
    noisy = Image.frombytes("RGB", (128, 128), random.Random(13).randbytes(128 * 128 * 3))
    output = BytesIO()
    noisy.save(output, format="PNG")
    noisy.close()
    assert len(output.getvalue()) > 9000

    snapshot = prepare_image(output.getvalue(), name="噪声.png")

    assert snapshot.mime_type == "image/jpeg"
    assert snapshot.size_bytes <= 9000
    assert (snapshot.width, snapshot.height) == (128, 128)


def test_finite_compression_reports_failure_instead_of_dropping_image(monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ImageInputError, match="有限压缩"):
        prepare_image(image_bytes())


@pytest.mark.parametrize("data", [b"", b"this is not an image", image_bytes()[:-15]])
def test_empty_invalid_and_truncated_images_are_rejected(data):
    with pytest.raises(ImageInputError):
        prepare_image(data)


def test_unsupported_actual_format_is_rejected_despite_png_filename():
    with pytest.raises(ImageInputError, match="仅支持"):
        prepare_image(image_bytes(image_format="BMP"), name="fake.png")


def test_animated_png_is_rejected():
    first = Image.new("RGB", (4, 4), "red")
    second = Image.new("RGB", (4, 4), "blue")
    output = BytesIO()
    first.save(output, format="PNG", save_all=True, append_images=[second], duration=100, loop=0)
    first.close()
    second.close()

    with pytest.raises(ImageInputError, match="动画"):
        prepare_image(output.getvalue())


def test_file_limit_and_pixel_limit_are_checked_before_full_decode(tmp_path, monkeypatch):
    payload = image_bytes(size=(10, 10))
    path = tmp_path / "large.png"
    path.write_bytes(payload)
    monkeypatch.setattr(images, "MAX_FILE_BYTES", len(payload) - 1)
    with pytest.raises(ImageInputError, match="原始图片"):
        load_image(path)
    assert path.read_bytes() == payload

    monkeypatch.setattr(images, "MAX_FILE_BYTES", len(payload))
    monkeypatch.setattr(images, "MAX_PIXELS", 99)
    with pytest.raises(ImageInputError, match="像素"):
        prepare_image(payload)


def test_pillow_decompression_warning_becomes_an_input_error(monkeypatch):
    payload = image_bytes(size=(10, 10))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 75)
    with pytest.raises(ImageInputError, match="像素"):
        prepare_image(payload)


def test_invalid_snapshot_cannot_bypass_format_dimensions_or_size_validation():
    snapshot = prepare_image(image_bytes())
    for changes in (
        {"data": "not-base64"},
        {"mime_type": "image/jpeg"},
        {"width": snapshot.width + 1},
        {"size_bytes": snapshot.size_bytes - 1},
        {"original_width": 0},
    ):
        with pytest.raises(ImageInputError):
            replace(snapshot, **changes)


def test_image_budget_counts_actual_payload_and_never_mutates_history(monkeypatch):
    snapshot = prepare_image(image_bytes())
    history = [
        {"role": "system", "content": "普通文本"},
        {"role": "user", "content": [{"type": "text", "text": "第一张"}, snapshot.to_block()]},
    ]
    before = copy.deepcopy(history)
    assert image_budget_usage(history) == (1, len(snapshot.data))
    monkeypatch.setattr(images, "MAX_SESSION_IMAGES", 2)
    monkeypatch.setattr(images, "MAX_SESSION_BASE64_BYTES", 2 * len(snapshot.data))
    validate_image_budget(history, [snapshot])
    with pytest.raises(ImageInputError, match="最多"):
        validate_image_budget(history, [snapshot, snapshot])
    monkeypatch.setattr(images, "MAX_SESSION_BASE64_BYTES", 2 * len(snapshot.data) - 1)
    with pytest.raises(ImageInputError, match="总量"):
        validate_image_budget(history, [snapshot])
    with pytest.raises(ImageInputError, match="加载和验证"):
        validate_image_budget(history, [snapshot.to_block()])
    assert history == before


def test_redaction_handles_all_wire_shapes_error_echoes_and_nested_values_without_mutation():
    snapshot = prepare_image(image_bytes())
    url = f"data:{snapshot.mime_type};base64,{snapshot.data}"
    value = {
        "messages": [{"role": "user", "content": [snapshot.to_block()]}],
        "openai": {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
        "anthropic": {
            "type": "image",
            "source": {"type": "base64", "media_type": snapshot.mime_type, "data": snapshot.data},
        },
        "error": f"请求失败：{url}；原始数据：{snapshot.data}",
        "ordinary": "图片已经加载。字段 data 和路径 ./a.png 都是普通文字。",
    }
    before = copy.deepcopy(value)

    result = redact_images(value)

    assert snapshot.data not in json.dumps(result, ensure_ascii=False)
    assert "data:image" not in json.dumps(result, ensure_ascii=False)
    assert result["messages"][0]["content"][0]["name"] == snapshot.name
    assert result["messages"][0]["content"][0]["width"] == snapshot.width
    assert result["anthropic"]["mime_type"] == snapshot.mime_type
    assert result["openai"]["size_bytes"] == snapshot.size_bytes
    assert result["ordinary"] == value["ordinary"]
    assert value == before
    assert redact_images(result) == result


def test_standalone_error_string_and_dataclass_are_redacted():
    snapshot = prepare_image(image_bytes())
    assert snapshot.data not in redact_images(f"API 回显：{snapshot.data}")
    assert snapshot.data not in redact_images(f"API 回显：data:image/png;base64,{snapshot.data}")
    assert snapshot.data not in json.dumps(redact_images(snapshot), ensure_ascii=False)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
def test_json_slash_escaped_image_payload_is_redacted(image_format):
    data = BytesIO()
    Image.new("RGB", (30, 20), "green").save(data, format=image_format)
    snapshot = prepare_image(data.getvalue())
    url = f"data:{snapshot.mime_type};base64,{snapshot.data}"
    escaped = json.dumps({"image_url": url, "raw": snapshot.data}).replace("/", "\\/")
    safe = redact_images(escaped)
    assert snapshot.data not in safe.replace("\\/", "/")
    assert "data:image" not in safe
