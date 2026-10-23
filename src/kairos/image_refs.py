"""图片引用的请求视图：每次请求模型前，把历史里的图片引用换成模型能看的内容，不修改原历史。

微信发来的图片已经归档，历史里只存 {"type": "image_ref", "image_id"}。展开规则：
- 最近一条带图片的用户消息：给原图。连续发的几张图配给同一句话，所以会一起给原图。
- 当前这一轮里 view_image 的结果：附上原图。
- 其余的：只给一行文字（标题、路径、描述）。

这样旧图不会每轮都重复发送。只有发了新图时，上一批图才从原图变成文字，
请求从那一轮开始变化一次，之后前缀缓存照常命中。
"""

import json

from .archive import ArchivedImage, ImageArchive

VIEW_IMAGE = "view_image"


def expand_image_refs(messages: list[dict], archive: ImageArchive) -> list[dict]:
    users = [i for i, message in enumerate(messages) if message["role"] == "user"]
    with_refs = [i for i in users if _has_refs(messages[i])]
    latest = with_refs[-1] if with_refs else None
    turn_start = users[-1] if users else len(messages)  # 当前这一轮从最后一条用户消息开始
    view_calls = _view_call_ids(messages)
    view = []
    for i, message in enumerate(messages):
        if i in with_refs:
            content = [part for block in message["content"] for part in _expand(block, archive, i == latest)]
            message = {**message, "content": content}
        elif message["role"] == "tool" and i > turn_start and message["tool_call_id"] in view_calls:
            message = {**message, "content": _attach_viewed(message["content"], archive)}
        view.append(message)
    return view


def _has_refs(message: dict) -> bool:
    content = message["content"]
    return isinstance(content, list) and any(block["type"] == "image_ref" for block in content)


def _view_call_ids(messages: list[dict]) -> set[str]:
    """历史里所有 view_image 调用的 ID，用来认出哪些工具结果是 view_image 的。"""
    return {
        call["id"]
        for message in messages
        if message["role"] == "assistant"
        for call in message.get("tool_calls", [])
        if call["function"]["name"] == VIEW_IMAGE
    }


def _expand(block: dict, archive: ImageArchive, pixels: bool) -> list[dict]:
    """展开一个内容块：图片引用换成“说明 + 原图”或一行文字，其他块原样保留。"""
    if block["type"] != "image_ref":
        return [block]
    image = archive.get(block["image_id"])
    if image is None:
        return [{"type": "text", "text": f"[图片 #{block['image_id']}：记录不存在]"}]
    label = _label(image, archive)
    if not pixels:
        return [{"type": "text", "text": f"{label} 描述：{image.description or '无'}"}]
    picture = archive.pixels(image)
    if picture is None:
        return [{"type": "text", "text": f"{label}（原图文件找不到了）描述：{image.description or '无'}"}]
    return [{"type": "text", "text": label}, picture.to_block()]


def _attach_viewed(content: str, archive: ImageArchive) -> str | list[dict]:
    """view_image 成功时，在工具结果文字后面附上原图。"""
    result = json.loads(content)
    image = archive.get(result["image_id"]) if "image_id" in result else None
    picture = archive.pixels(image) if image else None
    if picture is None:
        return content
    return [{"type": "text", "text": content}, picture.to_block()]


def _label(image: ArchivedImage, archive: ImageArchive) -> str:
    return f"[图片 #{image.id} · {image.title or '未分类'} · 已存档：{archive.root / image.path}]"
