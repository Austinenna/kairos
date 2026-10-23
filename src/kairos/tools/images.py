"""图片归档工具：搜索已经存档的图片，或取回一张旧图的原图来看。只读。"""

from ..archive import ImageArchive
from ..image_refs import VIEW_IMAGE
from .registry import Toolset, tool_schema

SEARCH_IMAGES = tool_schema(
    "search_images",
    "搜索已经存档的图片（微信发来的图片会自动存档），返回编号、分类、标题、描述、文件路径和存档时间，最新的在前。",
    {
        "query": {"type": "string", "description": "关键词，多个用空格分开，需同时包含；不填则列出最近存档的图片"},
        "limit": {"type": "integer", "description": "最多返回几张，默认 10"},
    },
    required=[],
)

VIEW_IMAGE_SCHEMA = tool_schema(
    VIEW_IMAGE,
    "按编号取回一张已存档图片的原图，原图会附在工具结果里，只在本轮可见。",
    {"image_id": {"type": "integer", "description": "图片编号，即 [图片 #编号] 或 search_images 结果里的 id"}},
)

RULES = """微信发来的图片会自动存档，并带进紧随其后的那句话；图片和用户的话无关时忽略它们。
对话里较早的图片只显示标题和描述，需要看清原图时用 view_image 取回。
用户问起存过的图片、照片、截图、票据等时，用 search_images 查找，回答时给出文件路径。
关键词尽量短，一次搜不到可以换同义词再搜。目前只能给出路径，不能把图片发回微信。"""


def image_tools(archive: ImageArchive) -> Toolset:
    def view_image(image_id: int) -> dict:
        image = archive.get(image_id)
        if image is None or archive.pixels(image) is None:
            raise ValueError(f"找不到图片 #{image_id} 的原图")
        return {"image_id": image.id, "title": image.title, "path": str(archive.root / image.path)}

    return Toolset(
        schemas=[SEARCH_IMAGES, VIEW_IMAGE_SCHEMA],
        handlers={"search_images": archive.search, VIEW_IMAGE: view_image},
        instructions=[RULES],
    )
