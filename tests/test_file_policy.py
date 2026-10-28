import pytest

from kairos.files.policy import FilePolicy


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    (home / "Documents" / "kairos").mkdir(parents=True)
    return home


def test_write_only_inside_workspace(home):
    policy = FilePolicy(home=home, workspace=home / "Documents" / "kairos")

    assert policy.writable("笔记/周报.md") == (home / "Documents" / "kairos" / "笔记" / "周报.md").resolve()
    for outside in [str(home / "Documents" / "other.md"), "../escape.md", "/etc/hosts"]:
        with pytest.raises(PermissionError):
            policy.writable(outside)


def test_deny_list_wins_even_if_workspace_is_too_wide(home):
    policy = FilePolicy(home=home, workspace=home)  # 模拟误把整个主目录加进白名单
    (home / "notes").mkdir()
    (home / "notes" / "keys").symlink_to(home / ".ssh", target_is_directory=True)

    for denied in [".ssh/config", ".SSH/config", "notes/keys/id_rsa", ".zshrc", "proj/.env.local",
                   "proj/.git/config", ".kairos/.env"]:  # fmt: skip
        with pytest.raises(PermissionError):
            policy.writable(denied)
    assert policy.writable("notes/todo.md")


def test_read_blocks_only_secrets(home):
    policy = FilePolicy(home=home, workspace=home / "Documents" / "kairos")

    assert policy.readable(str(home / ".zshrc"))
    assert policy.readable("/etc/hosts")
    for secret in [home / ".ssh" / "id_ed25519", home / ".kairos" / "kairos.db", home / "proj" / ".env"]:
        with pytest.raises(PermissionError):
            policy.readable(str(secret))
