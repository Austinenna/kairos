"""macOS 系统剪贴板读取；普通终端文字粘贴不调用此模块。"""

import base64
import binascii
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class ClipboardContent:
    text: str = ""
    paths: tuple[Path, ...] = ()
    image: bytes | None = None


class ClipboardError(ValueError):
    """读取失败时可直接在输入区显示的错误。"""


# 脚本固定不拼接文件名或剪贴板文字，返回数据通过 JSON 传递。
# Finder 复制文件优先返回 file URL；截图则由 AppKit 解码再转 PNG。
_MACOS_SCRIPT = r"""
ObjC.import('AppKit');
ObjC.import('Foundation');
const pb = $.NSPasteboard.generalPasteboard;
const result = {text: '', urls: [], image: null};
const items = pb.pasteboardItems;
if (ObjC.unwrap(items)) {
  for (let i = 0; i < items.count; i++) {
    const value = ObjC.unwrap(items.objectAtIndex(i).stringForType('public.file-url'));
    if (typeof value === 'string' && value) result.urls.push(value);
  }
}
const text = ObjC.unwrap(pb.stringForType('public.utf8-plain-text'));
if (typeof text === 'string') result.text = text;
if (result.urls.length === 0) {
  const image = $.NSImage.alloc.initWithPasteboard(pb);
  if (ObjC.unwrap(image)) {
    const tiff = image.TIFFRepresentation;
    const rep = ObjC.unwrap(tiff) ? $.NSBitmapImageRep.imageRepWithData(tiff) : null;
    const png = ObjC.unwrap(rep) ? rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $()) : null;
    if (ObjC.unwrap(png)) {
      if (png.length > 20 * 1024 * 1024) {
        result.error = '剪贴板图片超过 20 MiB，请先保存为文件并缩小图片。';
      } else {
        result.image = ObjC.unwrap(png.base64EncodedStringWithOptions(0));
      }
    }
  }
}
JSON.stringify(result);
"""


def read_clipboard() -> ClipboardContent:
    """读取当前图片、Finder 文件 URL 或文字，最多等待 5 秒。"""
    if sys.platform != "darwin":
        raise ClipboardError("当前只支持 macOS 剪贴板图片；也可以粘贴本地图片路径。")
    try:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", _MACOS_SCRIPT],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClipboardError("读取剪贴板超时，请重试或粘贴图片文件路径。") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ClipboardError("无法读取系统剪贴板，请重试或粘贴图片文件路径。") from exc
    try:
        content = json.loads(result.stdout)
        if not isinstance(content, dict):
            raise ValueError("invalid clipboard object")
        if content.get("error"):
            raise ClipboardError(str(content["error"]))
        if not isinstance(content.get("text", ""), str):
            raise ValueError("invalid clipboard text")
        if not isinstance(content.get("urls", []), list):
            raise ValueError("invalid clipboard URLs")
        paths = []
        for value in content.get("urls", []):
            if not isinstance(value, str):
                raise ValueError("invalid file URL")
            url = urlparse(value)
            if url.scheme != "file" or url.netloc not in ("", "localhost"):
                raise ValueError("unsupported file URL")
            paths.append(Path(unquote(url.path)))
        image = content.get("image")
        return ClipboardContent(
            text=content.get("text") or "",
            paths=tuple(paths),
            image=base64.b64decode(image, validate=True) if image else None,
        )
    except ClipboardError:
        raise
    except (ValueError, TypeError, AttributeError, binascii.Error) as exc:
        raise ClipboardError("剪贴板内容无法解析，请重新复制图片。") from exc
