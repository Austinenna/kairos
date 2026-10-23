"""图片归档工具：让模型搜索已经存档的图片。只读。"""

from ..archive import ImageArchive
from .registry import Toolset, tool_schema

SEARCH_IMAGES = tool_schema(
    "search_images",
    "搜索已经存档的图片（微信发来的图片会自动存档），返回分类、标题、描述、文件路径和存档时间，最新的在前。",
    {
        "query": {"type": "string", "description": "关键词，多个用空格分开，需同时包含；不填则列出最近存档的图片"},
        "limit": {"type": "integer", "description": "最多返回几张，默认 10"},
    },
    required=[],
)

RULES = """用户问起存过的图片、照片、截图、票据等时，用 search_images 查找，回答时给出文件路径。
关键词尽量短，一次搜不到可以换同义词再搜。目前只能给出路径，不能把图片发回微信。"""


def image_tools(archive: ImageArchive) -> Toolset:
    return Toolset(schemas=[SEARCH_IMAGES], handlers={"search_images": archive.search}, instructions=[RULES])
