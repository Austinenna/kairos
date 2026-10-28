"""列目录和搜索文件，只读。

搜索会跳过隐藏文件和隐藏目录（.git、.ssh、.env 之类），密钥凭证也就不会出现在结果里。
结果和扫描量都有上限，从很大的目录（比如整个主目录）开始搜也不会卡太久。
"""

import os
import re
from fnmatch import fnmatch
from pathlib import Path

from .policy import FilePolicy

MAX_ENTRIES = 200  # 列目录最多返回几项
MAX_MATCHES = 100  # 搜索最多返回几条
MAX_SCANNED = 20000  # 搜索最多检查几个文件
MAX_SEARCH_FILE_BYTES = 1024 * 1024  # 搜内容时跳过更大的文件
MAX_LINE_CHARS = 200  # 搜内容时每条结果最多显示的字符数


def list_dir(policy: FilePolicy, path: str = ".") -> dict:
    """列出目录里的文件和子目录，目录排在前面；子目录名以 / 结尾。"""
    target = policy.readable(path)
    if not target.is_dir():
        raise NotADirectoryError(f"{target} 不是目录")
    children = sorted(target.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower()))
    entries = [
        {"name": f"{child.name}/"} if child.is_dir() else {"name": child.name, "bytes": child.stat().st_size}
        for child in children[:MAX_ENTRIES]
    ]
    result = {"path": str(target), "entries": entries}
    if len(children) > MAX_ENTRIES:
        result["note"] = f"共 {len(children)} 项，只显示前 {MAX_ENTRIES} 项"
    return result


def search_files(policy: FilePolicy, query: str, path: str = ".", mode: str = "name", glob: str | None = None) -> dict:
    """在目录下递归搜索。

    参数:
        query: mode="name" 时是文件名关键词或通配符（*.md、周报*）；mode="content" 时是正则表达式。
            都不区分大小写。
        path: 从哪个目录开始搜，相对路径按工作目录解析。
        mode: "name" 按文件名搜，"content" 按文件内容搜。
        glob: 只在 mode="content" 时有用，限定要搜的文件名，例如 "*.md"。

    返回:
        {"matches": [...], "note"?}。按文件名搜时每条是文件路径；按内容搜时每条是 {path, line, text}。
    """
    root = policy.readable(path)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} 不是目录")
    if mode == "name":
        pattern = query.lower() if any(c in query for c in "*?[") else f"*{query.lower()}*"

        def match(file: Path) -> list:
            return [str(file)] if fnmatch(file.name.lower(), pattern) else []
    else:
        regex = re.compile(query, re.IGNORECASE)

        def match(file: Path) -> list:
            return _grep(policy, file, regex, glob)

    matches, scanned = [], 0
    for file in _walk(root):
        scanned += 1
        matches += match(file)
        if len(matches) >= MAX_MATCHES or scanned >= MAX_SCANNED:
            return {"matches": matches[:MAX_MATCHES], "note": "结果太多，只显示一部分。请缩小目录或换更具体的关键词"}
    return {"matches": matches}


def _walk(root: Path):
    """逐个给出目录下的文件，跳过隐藏的文件和目录。"""
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
        for name in sorted(names):
            if not name.startswith("."):
                yield Path(directory, name)


def _grep(policy: FilePolicy, file: Path, regex: re.Pattern, glob: str | None) -> list[dict]:
    """在一个文件里找匹配的行。跳过大文件、非 UTF-8 文件，以及软链接指向的密钥文件。"""
    if glob and not fnmatch(file.name.lower(), glob.lower()):
        return []
    try:
        policy.readable(str(file))
        if file.stat().st_size > MAX_SEARCH_FILE_BYTES:
            return []
        text = file.read_bytes().decode()
    except (OSError, UnicodeDecodeError):  # PermissionError 也是 OSError
        return []
    return [
        {"path": str(file), "line": number, "text": line.strip()[:MAX_LINE_CHARS]}
        for number, line in enumerate(text.splitlines(), start=1)
        if regex.search(line)
    ]
