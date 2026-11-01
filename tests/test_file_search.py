import pytest

from kairos.files.policy import FilePolicy
from kairos.files.search import list_dir, search_files


@pytest.fixture
def policy(tmp_path):
    home = tmp_path / "home"
    workspace = home / "Documents" / "kairos"
    (workspace / "周报").mkdir(parents=True)
    (workspace / "周报" / "第一周.md").write_text("完成了登录页\n下周做支付\n")
    (workspace / "购物清单.txt").write_text("牛奶\n鸡蛋\n")
    (workspace / ".env").write_text("API_KEY=secret 支付\n")
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("支付")
    (workspace / "钥匙").symlink_to(home / ".ssh" / "id_rsa")  # 软链接到密钥，搜内容时也要跳过
    return FilePolicy(home=home, workspace=workspace)


def test_list_dir_puts_directories_first(policy):
    names = [entry["name"] for entry in list_dir(policy)["entries"]]
    assert names[0] == "周报/"
    assert "购物清单.txt" in names


def test_search_by_name_and_content_skips_secrets(policy):
    assert [p.rsplit("/", 1)[-1] for p in search_files(policy, "*.md")["matches"]] == ["第一周.md"]

    files = search_files(policy, "支付", mode="content")["matches"]
    assert [(m["path"].rsplit("/", 1)[-1], m["count"]) for m in files] == [("第一周.md", 1)]
    lines = search_files(policy, "支付", mode="content", output="content")["matches"]
    assert [(m["path"].rsplit("/", 1)[-1], m["line"]) for m in lines] == [("第一周.md", 2)]


def test_many_matching_lines_do_not_hide_other_files(policy):
    """复现 2026-11-01 的失败：每个会话文件每行都匹配，按行截断时只看得到前两三个文件。"""
    for name in ("a.jsonl", "b.jsonl", "c.jsonl"):
        lines = "".join(f'{{"timestamp":"2026-11-01T0{i % 5}:00","text":"第 {i} 条"}}\n' for i in range(150))
        (policy.workspace / name).write_text(lines)

    files = search_files(policy, '"timestamp":"2026-11-01T0[0-4]', mode="content", glob="*.jsonl")
    assert [(m["path"].rsplit("/", 1)[-1], m["count"]) for m in files["matches"]] == [
        ("a.jsonl", 150),
        ("b.jsonl", 150),
        ("c.jsonl", 150),
    ]
    first = search_files(policy, "timestamp", mode="content", output="content", glob="*.jsonl")
    shown = len(first["matches"])  # 100 条或 2 万字，先到哪个算哪个
    assert first["total"] == 450 and shown <= 100 and f"offset={shown}" in first["note"]
    second = search_files(policy, "timestamp", mode="content", output="content", glob="*.jsonl", offset=shown)
    assert second["matches"][0]["line"] == shown + 1


def test_large_files_are_searched_and_binary_files_skipped(policy):
    big = policy.workspace / "big.log"
    big.write_text("INFO 正常\n" * 200_000 + "ERROR 出错了\n")  # 约 2.6MB，以前超过 1MB 会被静默跳过
    (policy.workspace / "image.bin").write_bytes(b"\0\x01ERROR")

    result = search_files(policy, "error", mode="content", output="content")
    assert [(m["path"].rsplit("/", 1)[-1], m["line"]) for m in result["matches"]] == [("big.log", 200_001)]
    single = search_files(policy, "正常", path=str(big), mode="content")
    assert single["matches"] == [{"path": str(big), "count": 200_000}]
