"""文本文件的读、写、改。一个会话一份 TextFiles，它记得这个会话读过哪些文件。

- 读：返回带行号的内容。一次最多 2000 行、50KB，超出时给出 next_offset，让模型接着读。
- 写：整份覆盖或新建。
- 改：把文件里唯一的一段原文（old_string）换成新文字（new_string），只做精确匹配。

先读后改：写已有文件、改文件之前，必须在这个会话里读过它，并且读过之后文件没有再被改过，
否则拒绝，让模型重新读。这样不会覆盖掉用户刚在别处改的内容。读过一部分（offset/limit）也算读过。
写入时先写临时文件再替换原文件，写到一半出错时原文件不受影响。
"""

import difflib
import os
from pathlib import Path

from .policy import FilePolicy

MAX_LINES = 2000  # 一次最多返回的行数
MAX_LINE_CHARS = 2000  # 单行超过这个长度时截断
MAX_BYTES = 50 * 1024  # 一次最多返回的内容大小
MAX_FILE_BYTES = 10 * 1024 * 1024  # 超过这个大小的文件不读进内存，让模型改用搜索


class TextFiles:
    def __init__(self, policy: FilePolicy):
        self.policy = policy
        self.read_versions: dict[Path, int] = {}  # 读过的文件 → 读取时的修改时间（纳秒）

    def read(self, path: str, offset: int = 1, limit: int = MAX_LINES) -> dict:
        """读取文本文件的一段，返回带行号的内容（“行号<Tab>内容”），并记下读取时的修改时间。"""
        if offset < 1 or limit < 1:
            raise ValueError("offset 和 limit 都必须是正整数")
        target = self.policy.readable(path)
        lines = _load(target).split("\n")
        if lines[-1] == "":
            lines.pop()  # 文件以换行结尾时，最后一个空元素不算一行
        if offset > max(len(lines), 1):
            raise ValueError(f"offset={offset} 超出了文件行数（共 {len(lines)} 行）")
        shown, size = [], 0
        for number, line in enumerate(lines[offset - 1 : offset - 1 + min(limit, MAX_LINES)], start=offset):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + "…（本行过长，已截断）"
            text = f"{number}\t{line.removesuffix(chr(13))}"
            size += len(text.encode()) + 1
            if size > MAX_BYTES and shown:
                break
            shown.append(text)
        self.read_versions[target] = target.stat().st_mtime_ns
        result = {"path": str(target), "total_lines": len(lines), "content": "\n".join(shown)}
        end = offset + len(shown) - 1
        if end < len(lines):
            result["next_offset"] = end + 1
            result["note"] = f"已显示第 {offset}–{end} 行，用 offset={end + 1} 继续读。"
        elif not lines:
            result["note"] = "文件是空的。"
        return result

    def write(self, path: str, content: str) -> dict:
        """新建文件，或整份覆盖已有文件（需要先读过）。不存在的上级目录会自动创建。"""
        target = self.policy.writable(path)
        existed = target.exists()
        if existed:
            self._check_fresh(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._save(target, content)
        return {"path": str(target), "created": not existed, "lines": len(content.splitlines())}

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict:
        """把文件里的 old_string 换成 new_string，返回改动的 diff。

        异常:
            ValueError: 满足以下任一情况时抛出:
                - old_string 为空，或和 new_string 相同;
                - 文件里找不到 old_string;
                - old_string 出现多次而 replace_all 为 False。

        行为说明:
            1. 匹配前把换行统一成 \\n（CRLF 归一化），写回时再换回文件原来的换行方式。
            2. 只做精确匹配，找不到时把错误交回模型，让它重新读文件、复制原文再试。
        """
        if not old_string:
            raise ValueError("old_string 不能为空；新建文件请用 write_file")
        if old_string == new_string:
            raise ValueError("old_string 和 new_string 相同，没有要改的内容")
        target = self.policy.writable(path)
        self._check_fresh(target)
        original = _load(target)
        crlf = "\r\n" in original
        text, old, new = (s.replace("\r\n", "\n") for s in (original, old_string, new_string))
        count = text.count(old)
        if count == 0:
            raise ValueError("文件里找不到 old_string。请重新读取文件，原样复制要替换的文字（不要带行号）")
        if count > 1 and not replace_all:
            raise ValueError(f"old_string 在文件里出现了 {count} 次。请多带几行上下文让它唯一，或设 replace_all=true")
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        self._save(target, updated.replace("\n", "\r\n") if crlf else updated)
        return {"path": str(target), "replaced": count if replace_all else 1, "diff": _diff(text, updated)}

    def _check_fresh(self, target: Path) -> None:
        """写已有文件前确认：这个会话读过它，而且读过之后没被改过。"""
        version = self.read_versions.get(target)
        if version is None:
            raise PermissionError(f"还没有读过 {target}，请先用 read_file 读取再修改")
        if target.stat().st_mtime_ns != version:
            raise PermissionError(f"{target} 在读取之后被改过，请重新读取再修改")

    def _save(self, target: Path, content: str) -> None:
        """先写同目录下的临时文件，再原子替换原文件；然后记下新的修改时间，下次可以接着改。"""
        temp = target.with_name(f".{target.name}.kairos-tmp")
        temp.write_bytes(content.encode())
        os.replace(temp, target)
        self.read_versions[target] = target.stat().st_mtime_ns


def _load(target: Path) -> str:
    """读出整个文件。太大或不是 UTF-8 文本时报错。"""
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{target} 超过 {MAX_FILE_BYTES // 1024 // 1024}MB，请用 search_files 查找需要的部分")
    try:
        return target.read_bytes().decode()
    except UnicodeDecodeError:
        raise ValueError(f"{target} 不是 UTF-8 文本文件") from None


def _diff(before: str, after: str) -> str:
    lines = difflib.unified_diff(before.splitlines(), after.splitlines(), "修改前", "修改后", n=2, lineterm="")
    return "\n".join(lines)
