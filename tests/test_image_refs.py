import json

from test_archive import archive, jpeg_bytes  # noqa: F401  archive 是 fixture

from kairos.images.refs import expand_image_refs


def view_call(call_id, image_id):
    function = {"name": "view_image", "arguments": json.dumps({"image_id": image_id})}
    call = {"id": call_id, "type": "function", "function": function}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def view_result(call_id, image_id):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"image_id": image_id})}


def kinds(message):
    return [block["type"] for block in message["content"]] if isinstance(message["content"], list) else "text"


def test_only_latest_images_and_current_views_get_pixels(archive):  # noqa: F811
    first = archive.classify(archive.store(jpeg_bytes(), "weixin")).id
    second = archive.store(jpeg_bytes(), "weixin").id  # 没分类也能看
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": [{"type": "text", "text": "第一张"}, {"type": "image_ref", "image_id": first}]},
        view_call("old", first),
        view_result("old", first),  # 上一轮的 view_image：不再附原图
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": [{"type": "text", "text": "第二张"}, {"type": "image_ref", "image_id": second}]},
        view_call("new", first),
        view_result("new", first),  # 本轮的 view_image：附原图
    ]

    view = expand_image_refs(messages, archive)

    assert kinds(view[1]) == ["text", "text"] and "停车 12 元" in view[1]["content"][1]["text"]
    assert kinds(view[3]) == "text"
    assert kinds(view[5]) == ["text", "text", "image"]
    assert kinds(view[7]) == ["text", "image"]
    assert messages[1]["content"][1] == {"type": "image_ref", "image_id": first}  # 原历史不变
