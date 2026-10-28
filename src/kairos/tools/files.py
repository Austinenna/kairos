"""文件工具：读、写、改文本文件，列目录，搜索文件。实际逻辑和路径规则在 files/ 包里。

每个会话一份（读过哪些文件是按会话记的），由 Conversation 创建。
"""

from collections.abc import Callable

from ..files.search import list_dir, search_files
from ..files.text import MAX_LINES, TextFiles
from .registry import Toolset, tool_schema

PATH = {"type": "string", "description": "文件路径：绝对路径、~ 开头，或相对工作目录的路径"}

READ_FILE = tool_schema(
    "read_file",
    f"读取 UTF-8 文本文件，每行前面带“行号<Tab>”。一次最多 {MAX_LINES} 行、50KB，没读完时结果里有 next_offset。",
    {
        "path": PATH,
        "offset": {"type": "integer", "description": "从第几行开始读，从 1 开始，默认 1"},
        "limit": {"type": "integer", "description": f"最多读几行，默认 {MAX_LINES}"},
    },
    required=["path"],
)

WRITE_FILE = tool_schema(
    "write_file",
    "新建文本文件，或用 content 整份覆盖已有文件。上级目录不存在时会自动创建。覆盖已有文件前必须先 read_file。",
    {"path": PATH, "content": {"type": "string", "description": "文件的完整内容"}},
)

EDIT_FILE = tool_schema(
    "edit_file",
    "把文件里的一段原文 old_string 换成 new_string，返回 diff。必须先 read_file；old_string 要精确匹配且在文件里唯一。",
    {
        "path": PATH,
        "old_string": {"type": "string", "description": "要替换的原文，从 read_file 结果里原样复制，不带行号"},
        "new_string": {"type": "string", "description": "替换后的文字；传空字符串表示删除这段原文"},
        "replace_all": {"type": "boolean", "description": "替换所有出现的地方，默认 false"},
    },
    required=["path", "old_string", "new_string"],
)

LIST_DIR = tool_schema(
    "list_dir",
    "列出目录里的文件和子目录，子目录名以 / 结尾。",
    {"path": {**PATH, "description": "目录路径，默认是工作目录"}},
    required=[],
)

SEARCH_FILES = tool_schema(
    "search_files",
    "在目录下递归搜索文件，跳过隐藏文件和目录。mode=name 按文件名搜，mode=content 按内容搜。",
    {
        "query": {"type": "string", "description": "name：文件名关键词或通配符（如 *.md）；content：正则表达式"},
        "path": {**PATH, "description": "从哪个目录开始搜，默认是工作目录"},
        "mode": {"type": "string", "enum": ["name", "content"], "description": "默认 name"},
        "glob": {"type": "string", "description": "mode=content 时限定文件名，例如 *.md"},
    },
    required=["query"],
)


def file_tools(files: TextFiles) -> Toolset:
    policy = files.policy
    rules = f"""文件工具规则：
- 工作目录是 {policy.workspace}。只能在工作目录里新建和修改文件，相对路径也按它解析；其他位置的文件只能读。
- 不知道文件在哪时，先用 list_dir 或 search_files 查找，不要猜路径。
- read_file 结果每行开头的“行号<Tab>”不是文件内容。文件没读完时按 next_offset 接着读，或先搜索再读需要的部分。
- 修改已有文件用 edit_file：先 read_file，再从结果里原样复制要换掉的文字作为 old_string，带上刚好能让它唯一的几行。
- 新建文件或整篇重写用 write_file。写完把文件路径告诉用户。
- 密钥凭证、系统文件和 Kairos 自己的数据不能读写。被拒绝时如实告诉用户，不要换路径或换工具绕过。"""
    return Toolset(
        schemas=[READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, SEARCH_FILES],
        handlers={
            "read_file": _skip_none(files.read),
            "write_file": files.write,
            "edit_file": _skip_none(files.edit),
            "list_dir": _skip_none(lambda path=".": list_dir(policy, path)),
            "search_files": _skip_none(lambda query, **options: search_files(policy, query, **options)),
        },
        instructions=[rules],
    )


def _skip_none(handler: Callable) -> Callable:
    """模型有时把没填的可选参数传成 null，去掉它们，让函数用自己的默认值。"""
    return lambda **arguments: handler(**{key: value for key, value in arguments.items() if value is not None})
