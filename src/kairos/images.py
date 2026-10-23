"""图片输入：把文件或剪贴板里的图片处理成可以发给模型的内容块。

处理只做三件事：按 EXIF 校正方向、缩到最长边 2000 像素、编码成 base64。
用户消息里的图片块格式：{"type": "image", "name", "mime_type", "data": base64}，
各模型适配器再把它转换成自己协议的格式。
微信发来的图片已经归档，历史里只存引用 {"type": "image_ref", "image_id"}，请求前再展开（见 archive.py）。
"""

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageOps

MAX_SIDE = 2000  # 最长边像素；更大的图片会等比缩小
MAX_PNG_BYTES = 3 * 1024 * 1024  # PNG 超过这个大小且没有透明通道时改存 JPEG，减小请求体积
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"}


@dataclass
class Image:
    name: str
    mime_type: str
    data: str  # base64
    width: int
    height: int

    def to_block(self) -> dict:
        return {"type": "image", "name": self.name, "mime_type": self.mime_type, "data": self.data}


def load_image(path: str | Path) -> Image:
    path = Path(path).expanduser()
    return prepare_image(path.read_bytes(), path.name)


def prepare_image(data: bytes, name: str) -> Image:
    """校正方向、缩小、重新编码。PNG、JPEG、WebP 保留原格式，其他格式转成 PNG。"""
    with PILImage.open(BytesIO(data)) as source:
        image_format = source.format if source.format in ("PNG", "JPEG", "WEBP") else "PNG"
        image = ImageOps.exif_transpose(source)
    image.thumbnail((MAX_SIDE, MAX_SIDE))

    encoded = _encode(image, image_format)
    if image_format == "PNG" and len(encoded) > MAX_PNG_BYTES and "A" not in image.getbands():
        image_format = "JPEG"
        encoded = _encode(image, image_format)
    return Image(
        name=name,
        mime_type=f"image/{image_format.lower()}",
        data=base64.b64encode(encoded).decode("ascii"),
        width=image.width,
        height=image.height,
    )


def _encode(image: PILImage.Image, image_format: str) -> bytes:
    if image_format == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")  # JPEG 不支持透明通道和调色板
    buffer = BytesIO()
    image.save(buffer, format=image_format, quality=85)
    return buffer.getvalue()


def text_of(content: str | list[dict]) -> str:
    """取出消息正文里的文字；图片换成“[图片：文件名]”或“[图片 #编号]”，不带 base64。

    用在会话标题、历史搜索、记忆提取等只需要文字的地方。
    """
    if isinstance(content, str):
        return content
    return " ".join(_block_text(block) for block in content)


def _block_text(block: dict) -> str:
    if block["type"] == "text":
        return block["text"]
    if block["type"] == "image_ref":
        return f"[图片 #{block['image_id']}]"
    return f"[图片：{block['name']}]"
