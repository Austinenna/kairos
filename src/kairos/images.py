"""本地图片快照、预处理、会话预算与展示摘要。"""

from __future__ import annotations

import base64
import binascii
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 64_000_000
MAX_DIMENSION = 2000
MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_SESSION_IMAGES = 8
MAX_SESSION_BASE64_BYTES = 16 * 1024 * 1024

_FORMAT_MIMES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_IMAGE_DATA_REMOVED = "[图片数据已省略]"
_DATA_URL = re.compile(r"data:image\\?/[\w.+-]+(?:;[^\s,\"'<>\\]*)?,(?:[A-Za-z0-9+_=-]|\\?/)*", re.IGNORECASE)
_IMAGE_BASE64 = re.compile(r"(?:iVBORw0KGgo|\\?/9j\\?/|UklGR|R0lGOD)(?:[A-Za-z0-9+]|\\?/){16,}={0,2}")


class ImageInputError(ValueError):
    """图片无法作为本轮输入；错误说明不包含图片载荷。"""


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _decode_image(data: bytes, *, processed: bool = False) -> tuple[Image.Image, str, tuple[int, int]]:
    """先检查头部尺寸，再完整校验和解码；返回与输入流无关的像素副本。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                image_format = source.format
                if image_format not in _FORMAT_MIMES:
                    raise ImageInputError("仅支持 PNG、JPEG 和 WebP 图片。")
                width, height = source.size
                if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                    raise ImageInputError(f"图片像素数超过上限（{MAX_PIXELS:,}）。")
                if processed and max(width, height) > MAX_DIMENSION:
                    raise ImageInputError(f"处理后的图片最长边不能超过 {MAX_DIMENSION} 像素。")
                if getattr(source, "is_animated", False) or getattr(source, "n_frames", 1) != 1:
                    raise ImageInputError("暂不支持动画图片，请先导出一张静态图片。")
                source.verify()
            # verify() 会使解码器失效；从同一份内存字节重新打开，不重读源文件。
            with Image.open(BytesIO(data)) as source:
                source.load()
                return source.copy(), image_format, (width, height)
    except ImageInputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageInputError("图片像素过多，已停止解码。") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise ImageInputError("图片损坏或无法解码，请使用完整的 PNG、JPEG 或 WebP 图片。") from None


@dataclass(frozen=True)
class ImageInput:
    """已经验证并固定内容的图片；后续请求不依赖原文件。"""

    name: str
    mime_type: str
    data: str = field(repr=False)
    width: int
    height: int
    original_width: int
    original_height: int
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ImageInputError("图片名称不能为空。")
        if self.mime_type not in _FORMAT_MIMES.values():
            raise ImageInputError("图片媒体类型不受支持。")
        dimensions = (self.width, self.height, self.original_width, self.original_height)
        if not all(_positive_int(value) for value in dimensions):
            raise ImageInputError("图片尺寸必须是正整数。")
        if max(self.width, self.height) > MAX_DIMENSION or self.original_width * self.original_height > MAX_PIXELS:
            raise ImageInputError("图片尺寸超过上限。")
        if not _positive_int(self.size_bytes) or self.size_bytes > MAX_IMAGE_BYTES:
            raise ImageInputError("处理后的图片大小超过上限。")
        if not isinstance(self.data, str) or not self.data or len(self.data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ImageInputError("图片 Base64 数据为空或超过上限。")
        try:
            raw = base64.b64decode(self.data, validate=True)
        except (ValueError, binascii.Error):
            raise ImageInputError("图片 Base64 数据无效。") from None
        if len(raw) != self.size_bytes or base64.b64encode(raw).decode("ascii") != self.data:
            raise ImageInputError("图片大小或 Base64 数据与快照不一致。")
        image, image_format, size = _decode_image(raw, processed=True)
        image.close()
        if size != (self.width, self.height) or _FORMAT_MIMES[image_format] != self.mime_type:
            raise ImageInputError("图片尺寸或媒体类型与实际内容不一致。")
        if self.width * self.height > self.original_width * self.original_height:
            raise ImageInputError("处理后的图片不能大于原始图片。")

    def to_block(self) -> dict:
        """返回内部内容块；provider 负责挑选协议允许的字段。"""
        return {
            "type": "image",
            "name": self.name,
            "mime_type": self.mime_type,
            "data": self.data,
            "width": self.width,
            "height": self.height,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "size_bytes": self.size_bytes,
        }


def _encode_image(image: Image.Image, image_format: str, quality: int | None = None) -> bytes:
    stream = BytesIO()
    if image_format == "PNG":
        image.save(stream, format="PNG", optimize=True, compress_level=9)
    elif image_format == "JPEG":
        image.save(stream, format="JPEG", quality=quality or 95, optimize=True)
    else:
        options = {"lossless": True} if quality is None else {"quality": quality}
        image.save(stream, format="WEBP", method=4, **options)
    return stream.getvalue()


def _fit_encoded_image(image: Image.Image, image_format: str) -> tuple[bytes, str]:
    """先保留格式；必要时有限降质，带透明通道时保持透明。"""
    first = _encode_image(image, image_format)
    if len(first) <= MAX_IMAGE_BYTES:
        return first, image_format
    has_alpha = "A" in image.getbands()
    fallback_format = "WEBP" if has_alpha or image_format == "WEBP" else "JPEG"
    candidate_image = image if has_alpha or image.mode in {"RGB", "L"} else image.convert("RGB")
    try:
        for quality in (90, 80, 65, 50):
            candidate = _encode_image(candidate_image, fallback_format, quality)
            if len(candidate) <= MAX_IMAGE_BYTES:
                return candidate, fallback_format
    finally:
        if candidate_image is not image:
            candidate_image.close()
    raise ImageInputError("图片在有限压缩后仍超过单图大小上限，请裁剪或手动压缩后重试。")


def prepare_image(data: bytes, name: str = "剪贴板.png") -> ImageInput:
    """验证真实格式、校正方向并按需缩放，生成可以直接提交的图片快照。"""
    if not isinstance(data, bytes) or not data:
        raise ImageInputError("图片文件为空或数据无效。")
    if len(data) > MAX_FILE_BYTES:
        raise ImageInputError(f"原始图片不能超过 {MAX_FILE_BYTES // (1024 * 1024)} MiB。")
    if not isinstance(name, str) or not name.strip():
        raise ImageInputError("图片名称不能为空。")
    display_name = name.replace("\\", "/").rsplit("/", 1)[-1]
    if not display_name.strip():
        raise ImageInputError("图片名称不能为空。")
    source, image_format, original_size = _decode_image(data)
    normalized: Image.Image | None = None
    try:
        orientation = source.getexif().get(274, 1)
        if orientation == 1 and max(source.size) <= MAX_DIMENSION and len(data) <= MAX_IMAGE_BYTES:
            encoded, output_format = data, image_format
            size = source.size
        else:
            normalized = ImageOps.exif_transpose(source)
            # 调整调色板图片前转为真正的颜色通道，避免缩放退化成最近邻或丢失透明。
            if "transparency" in normalized.info and "A" not in normalized.getbands():
                converted = normalized.convert("RGBA")
                normalized.close()
                normalized = converted
            elif normalized.mode == "P":
                converted = normalized.convert("RGBA" if "transparency" in normalized.info else "RGB")
                normalized.close()
                normalized = converted
            elif image_format == "JPEG" and normalized.mode not in {"RGB", "L"}:
                converted = normalized.convert("RGB")
                normalized.close()
                normalized = converted
            # 重编码不携带原方向标记，避免下一轮再次旋转。
            normalized.info.clear()
            normalized.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
            encoded, output_format = _fit_encoded_image(normalized, image_format)
            size = normalized.size
        return ImageInput(
            name=display_name,
            mime_type=_FORMAT_MIMES[output_format],
            data=base64.b64encode(encoded).decode("ascii"),
            width=size[0],
            height=size[1],
            original_width=original_size[0],
            original_height=original_size[1],
            size_bytes=len(encoded),
        )
    except ImageInputError:
        raise
    except (OSError, ValueError, KeyError):
        raise ImageInputError("图片方向校正或压缩失败，请先转换成普通静态图片。") from None
    finally:
        source.close()
        if normalized is not None:
            normalized.close()


def load_image(path: str | Path) -> ImageInput:
    """只读一次本地普通文件；大小检查既在读取前，也在有界读取后执行。"""
    try:
        image_path = Path(path).expanduser()
        if not image_path.is_file():
            raise ImageInputError("图片路径不存在或不是普通文件。")
        if image_path.stat().st_size > MAX_FILE_BYTES:
            raise ImageInputError(f"原始图片不能超过 {MAX_FILE_BYTES // (1024 * 1024)} MiB。")
        with image_path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
    except ImageInputError:
        raise
    except (OSError, ValueError, TypeError):
        raise ImageInputError("无法读取图片文件，请检查路径和读取权限。") from None
    return prepare_image(data, name=image_path.name)


def image_budget_usage(messages: list[dict]) -> tuple[int, int]:
    """统计实际内部图片块数量及 Base64 字节，不信任元数据中的大小。"""
    count = total = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            data = block.get("data")
            if not isinstance(data, str) or not data.isascii():
                raise ImageInputError("历史图片数据无效。")
            count += 1
            total += len(data)
    return count, total


def validate_image_budget(messages: list[dict], images: Sequence[ImageInput]) -> None:
    """新增附件只做准入检查，不修改原历史，也不自动删除旧图片。"""
    count, total = image_budget_usage(messages)
    for image in images:
        if not isinstance(image, ImageInput):
            raise ImageInputError("图片必须先经过加载和验证。")
        count += 1
        total += len(image.data)
    if count > MAX_SESSION_IMAGES:
        raise ImageInputError(f"本段对话最多保留 {MAX_SESSION_IMAGES} 张图片，请 /reset 后开始新对话。")
    if total > MAX_SESSION_BASE64_BYTES:
        raise ImageInputError("本段对话的图片总量超过上限，请移除新附件或 /reset 后开始新对话。")


def _image_payload(value: dict) -> tuple[str | None, str | None]:
    # 观察事件也包含 JSON Schema；其 type 可以是允许 null 的类型列表。
    if not isinstance(value.get("type"), str):
        return None, None
    source = value.get("source")
    if value.get("type") == "image":
        if isinstance(source, dict):
            return source.get("data"), source.get("media_type")
        return value.get("data"), value.get("mime_type") or value.get("mimeType")
    if value.get("type") in {"image_url", "input_image"}:
        url = value.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
        if isinstance(url, str) and url.startswith("data:image/") and "," in url:
            header, data = url.split(",", 1)
            return data, header[5:].split(";", 1)[0]
    return None, None


def redact_images(value: Any) -> Any:
    """递归生成展示副本，也清理异常字符串中回显的图片 Base64。"""
    payloads: set[str] = set()

    def collect(item: Any) -> None:
        if isinstance(item, ImageInput):
            payloads.add(item.data)
        elif isinstance(item, dict):
            data, _ = _image_payload(item)
            if isinstance(data, str) and data and data != _IMAGE_DATA_REMOVED:
                payloads.add(data)
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                collect(nested)

    collect(value)
    known_payloads = sorted(payloads, key=len, reverse=True)

    def clean_text(text: str) -> str:
        text = _DATA_URL.sub(_IMAGE_DATA_REMOVED, text)
        for data in known_payloads:
            text = text.replace(data.replace("/", "\\/"), _IMAGE_DATA_REMOVED)
            text = text.replace(data, _IMAGE_DATA_REMOVED)
        return _IMAGE_BASE64.sub(_IMAGE_DATA_REMOVED, text)

    def project(item: Any) -> Any:
        if isinstance(item, ImageInput):
            return project(item.to_block())
        if isinstance(item, dict):
            data, mime_type = _image_payload(item)
            if isinstance(data, str):
                summary = {"type": "image", "data": _IMAGE_DATA_REMOVED}
                if isinstance(mime_type, str):
                    summary["mime_type"] = clean_text(mime_type)
                for key in ("name", "width", "height", "original_width", "original_height", "size_bytes"):
                    if key in item:
                        summary[key] = project(item[key])
                if "size_bytes" not in summary and data != _IMAGE_DATA_REMOVED:
                    padding = len(data) - len(data.rstrip("="))
                    summary["size_bytes"] = max(0, len(data) * 3 // 4 - padding)
                return summary
            return {key: project(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [project(nested) for nested in item]
        if isinstance(item, tuple):
            return tuple(project(nested) for nested in item)
        return clean_text(item) if isinstance(item, str) else item

    return project(value)
