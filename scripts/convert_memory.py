"""一次性脚本：把旧版记忆（~/.kairos/memory）转换成新版格式，写到 ~/.kairos/memories。

旧目录保持不动，旧版 `uv run kairos` 仍然可以使用。新目录已存在时不做任何事。
运行：uv run python scripts/convert_memory.py
"""

import json
from pathlib import Path

OLD = Path.home() / ".kairos" / "memory"
NEW = Path.home() / ".kairos" / "memories"
HEADER_END = "\n-->\n\n"


def split_old_file(path: Path) -> tuple[dict, str]:
    """旧文件 = “<!-- kairos-memory {JSON 元数据} -->” + Markdown 正文。"""
    header, body = path.read_text(encoding="utf-8").split(HEADER_END, 1)
    return json.loads(header.removeprefix("<!-- kairos-memory\n")), body.strip()


def main() -> None:
    if NEW.exists():
        print(f"{NEW} 已存在，不做转换。")
        return
    (NEW / "topics").mkdir(parents=True)

    _, core = split_old_file(OLD / "CORE.md")
    # 旧 CORE：标题下每段一条；新 CORE：每条一行“- 内容”。
    entries = [paragraph for paragraph in core.split("\n\n") if not paragraph.startswith("#")]
    (NEW / "CORE.md").write_text("# 核心记忆\n\n" + "".join(f"- {entry}\n" for entry in entries), encoding="utf-8")
    print(f"CORE.md：{len(entries)} 条")

    for path in sorted((OLD / "topics").glob("*.md")):
        metadata, body = split_old_file(path)
        title = metadata["title"].replace("/", "-")
        text = f"# {title}\n\n用途：{metadata['description']}\n\n{body}\n"
        (NEW / "topics" / f"{title}.md").write_text(text, encoding="utf-8")
        print(f"主题：{title}")


main()
