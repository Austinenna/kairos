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

    matches = search_files(policy, "支付", mode="content")["matches"]
    assert [(m["path"].rsplit("/", 1)[-1], m["line"]) for m in matches] == [("第一周.md", 2)]
