"""记忆存储：全部是普通 Markdown 文件，可以直接打开查看和手工修改。

memories/
  CORE.md          核心记忆，每条一行“- 内容”，每次请求都完整加载
  topics/标题.md    一个主题一个文件，按需用 read_memory 读取

主题文件格式：
  # 标题

  用途：什么时候需要读这个主题

  正文，每次追加一段
"""

from dataclasses import dataclass
from pathlib import Path

CORE_HEADER = "# 核心记忆\n\n"
PURPOSE_PREFIX = "用途："


@dataclass
class Topic:
    title: str
    purpose: str


class MemoryStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.core_path = self.root / "CORE.md"
        self.topics_dir = self.root / "topics"
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        if not self.core_path.exists():
            self.core_path.write_text(CORE_HEADER, encoding="utf-8")

    def core(self) -> str:
        return self.core_path.read_text(encoding="utf-8")

    def add_core(self, content: str) -> None:
        """在 CORE 末尾追加一条。多行内容合并成一行，保持“一条一行”。"""
        with self.core_path.open("a", encoding="utf-8") as file:
            file.write(f"- {' '.join(content.split())}\n")

    def topics(self) -> list[Topic]:
        """读取所有主题的标题和用途（只读文件开头，不读正文）。"""
        topics = []
        for path in sorted(self.topics_dir.glob("*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            purpose = next((line.removeprefix(PURPOSE_PREFIX) for line in lines if line.startswith(PURPOSE_PREFIX)), "")
            topics.append(Topic(title=path.stem, purpose=purpose))
        return topics

    def read_topic(self, title: str) -> str:
        path = self._topic_path(title)
        if not path.exists():
            raise KeyError(f"没有名为“{title}”的记忆主题")
        return path.read_text(encoding="utf-8")

    def write_topic(self, title: str, addition: str, purpose: str | None = None) -> str:
        """主题不存在就新建（需要 purpose），存在就在末尾追加一段。返回 created 或 appended。"""
        path = self._topic_path(title)
        if path.exists():
            with path.open("a", encoding="utf-8") as file:
                file.write(f"\n{addition.strip()}\n")
            return "appended"
        if not purpose:
            raise ValueError("新建主题需要提供 purpose（用途）")
        path.write_text(f"# {title}\n\n{PURPOSE_PREFIX}{purpose}\n\n{addition.strip()}\n", encoding="utf-8")
        return "created"

    def snapshot(self) -> str:
        """每次请求附带的记忆：CORE 全文 + 主题索引。"""
        index = "\n".join(f"- {topic.title}：{topic.purpose}" for topic in self.topics()) or "（暂无主题）"
        return f"{self.core().strip()}\n\n# 记忆主题索引（需要时用 read_memory 按标题读取正文）\n\n{index}"

    def _topic_path(self, title: str) -> Path:
        # 标题直接作文件名；去掉路径分隔符，避免写到 topics 目录之外。
        name = title.strip().replace("/", "-").replace("\\", "-")
        return self.topics_dir / f"{name}.md"
