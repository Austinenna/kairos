"""列目录和搜索文件，只读。

搜索会跳过隐藏文件和隐藏目录（.git、.ssh、.env 之类），密钥凭证也就不会出现在结果里。
按内容搜时逐行读文件，内存里只放一行，所以不按文件大小跳过；开头含空字节的当作二进制文件跳过。
检查到的文件都读完、结果都数清，但进结果的有上限：条数或总字数超出时截断，并说明总数和怎么翻页。
"""

import json
import os
import re
from collections.abc import Iterable, Iterator
from fnmatch import fnmatch
from pathlib import Path

from .policy import FilePolicy

MAX_ENTRIES = 200  # 列目录最多返回几项
MAX_MATCHES = 100  # 搜索一次最多返回几条（文件或匹配行）
MAX_RESULT_CHARS = 20_000  # 搜索结果的总字数上限：条数没超、但每条都很长时也要截断
MAX_SCANNED = 20000  # 搜索最多检查几个文件
MAX_LINE_CHARS = 500  # 按内容搜时每条匹配行最多显示的字符数
BINARY_SNIFF_BYTES = 8192  # 读文件开头这么多字节，含空字节就当作二进制


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


def search_files(
    policy: FilePolicy,
    query: str,
    path: str = ".",
    mode: str = "name",
    glob: str | None = None,
    output: str = "files",
    offset: int = 0,
) -> dict:
    """在目录下递归搜索。

    参数:
        query: mode="name" 时是文件名关键词或通配符（*.md、周报*）；mode="content" 时是正则表达式。
            都不区分大小写。
        path: 从哪个目录开始搜，也可以是单个文件（只搜这个文件）；相对路径按工作目录解析。
        mode: "name" 按文件名搜，"content" 按文件内容搜。
        glob: 只在 mode="content" 时有用，限定要搜的文件名，例如 "*.md"。
        output: 只在 mode="content" 时有用。"files"（默认）列出匹配的文件和每个文件匹配了几行，
            "content" 列出匹配行。先用 files 找到文件，再缩小 path 用 content 看具体内容。
        offset: 跳过前几条结果，用来翻页。

    返回:
        {"matches": [...], "total": 结果总条数, "note"?}。按文件名搜时每条是文件路径；按内容搜时
        output="files" 每条是 {path, count}，output="content" 每条是 {path, line, text}。
        没有全部显示，或检查的文件数到了上限时，note 里说明。
    """
    root = policy.readable(path)
    if not root.exists():
        raise FileNotFoundError(f"{root} 不存在")
    files = list(_walk(root)) if root.is_dir() else [root]
    if mode == "name":
        pattern = query.lower() if any(c in query for c in "*?[") else f"*{query.lower()}*"
        found = (str(file) for file in files[:MAX_SCANNED] if fnmatch(file.name.lower(), pattern))
    else:
        regex = re.compile(query, re.IGNORECASE)
        found = _grep(policy, files[:MAX_SCANNED], regex, glob, output)
    result = _page(found, offset)
    if len(files) > MAX_SCANNED:
        note = f"检查了 {MAX_SCANNED} 个文件后停止，后面的没有搜，请缩小目录"
        result["note"] = f"{result['note']}；{note}" if "note" in result else note
    return result


def _page(found: Iterable, offset: int) -> dict:
    """数出全部结果，只留下从 offset 开始、不超过条数和字数上限的一页。"""
    matches, chars, total, full = [], 0, 0, False
    for total, match in enumerate(found, start=1):
        if total <= offset or full:
            continue
        chars += len(json.dumps(match, ensure_ascii=False))
        full = len(matches) >= MAX_MATCHES or (chars > MAX_RESULT_CHARS and bool(matches))
        if not full:
            matches.append(match)
    result = {"matches": matches, "total": total}
    end = offset + len(matches)
    if end < total:
        result["note"] = (
            f"共 {total} 条，显示了第 {offset + 1}–{end} 条。用 offset={end} 看后面的，或缩小目录、换更具体的关键词"
        )
    return result


def _walk(root: Path) -> Iterator[Path]:
    """逐个给出目录下的文件，跳过隐藏的文件和目录。多给一个，调用方才知道有没有超过 MAX_SCANNED。"""
    count = 0
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
        for name in sorted(names):
            if not name.startswith("."):
                yield Path(directory, name)
                count += 1
                if count > MAX_SCANNED:
                    return


def _grep(policy: FilePolicy, files: list[Path], regex: re.Pattern, glob: str | None, output: str) -> Iterator[dict]:
    """按内容搜：output="files" 时每个有匹配的文件给出一条 {path, count}，否则每条匹配行给出一条。

    匹配行边读边交出去，不在内存里攒着，用 "." 这种正则搜大文件也不会占很多内存。
    """
    for file in files:
        if glob and not fnmatch(file.name.lower(), glob.lower()):
            continue
        lines = _matching_lines(policy, file, regex)
        if output == "files":
            if count := sum(1 for _ in lines):
                yield {"path": str(file), "count": count}
        else:
            yield from ({"path": str(file), "line": number, "text": text} for number, text in lines)


def _matching_lines(policy: FilePolicy, file: Path, regex: re.Pattern) -> Iterator[tuple[int, str]]:
    """逐行读一个文件，给出匹配的 (行号, 截断后的行)。跳过二进制、非 UTF-8 文件，以及软链接指向的密钥文件。

    文件读到一半才发现不是 UTF-8 时停下，已经给出的匹配保留。
    """
    try:
        policy.readable(str(file))
        with file.open("rb") as handle:
            if b"\0" in handle.read(BINARY_SNIFF_BYTES):
                return
            handle.seek(0)
            for number, raw in enumerate(handle, start=1):
                line = raw.decode()
                if regex.search(line):
                    yield number, line.strip()[:MAX_LINE_CHARS]
    except (OSError, UnicodeDecodeError):  # PermissionError 也是 OSError
        return
