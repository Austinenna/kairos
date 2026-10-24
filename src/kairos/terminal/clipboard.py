"""读取 macOS 系统剪贴板：截图、Finder 里复制的文件，或者文字。

Python 标准库读不了剪贴板里的图片，这里用 macOS 自带的 osascript 运行一段 JavaScript，
调用系统的 NSPasteboard 接口，把结果以 JSON 返回。
"""

import base64
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass
class ClipboardContent:
    text: str = ""
    paths: tuple[Path, ...] = ()  # Finder 里复制的文件
    image: bytes | None = None  # 截图等位图，统一转成 PNG


class ClipboardError(RuntimeError):
    """读取失败，消息可以直接显示给用户。"""


# 优先取 Finder 复制的文件路径；没有文件时再取图片，转成 PNG 后用 base64 返回。
_SCRIPT = r"""
ObjC.import('AppKit');
const pb = $.NSPasteboard.generalPasteboard;
const result = {text: '', urls: [], image: null};
const items = pb.pasteboardItems;
for (let i = 0; i < items.count; i++) {
  const url = ObjC.unwrap(items.objectAtIndex(i).stringForType('public.file-url'));
  if (url) result.urls.push(url);
}
result.text = ObjC.unwrap(pb.stringForType('public.utf8-plain-text')) || '';
if (result.urls.length === 0) {
  const image = $.NSImage.alloc.initWithPasteboard(pb);
  if (ObjC.unwrap(image)) {
    const rep = $.NSBitmapImageRep.imageRepWithData(image.TIFFRepresentation);
    const png = rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $());
    result.image = ObjC.unwrap(png.base64EncodedStringWithOptions(0));
  }
}
JSON.stringify(result);
"""


def read_clipboard() -> ClipboardContent:
    if sys.platform != "darwin":
        raise ClipboardError("目前只支持 macOS 剪贴板；也可以把图片文件拖进输入框。")
    try:
        output = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", _SCRIPT], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ClipboardError(f"读取剪贴板失败：{type(error).__name__}") from error
    data = json.loads(output)
    return ClipboardContent(
        text=data["text"],
        paths=tuple(Path(unquote(urlparse(url).path)) for url in data["urls"]),
        image=base64.b64decode(data["image"]) if data["image"] else None,
    )
