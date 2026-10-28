import os

import pytest

from kairos.files.policy import FilePolicy
from kairos.files.text import TextFiles


@pytest.fixture
def files(tmp_path):
    workspace = tmp_path / "home" / "Documents" / "kairos"
    workspace.mkdir(parents=True)
    return TextFiles(FilePolicy(home=tmp_path / "home", workspace=workspace))


def test_read_in_pages(files):
    files.write("长文.md", "".join(f"第 {i} 行\n" for i in range(1, 2501)))

    first = files.read("长文.md")
    assert first["total_lines"] == 2500
    assert first["content"].splitlines()[0] == "1\t第 1 行"
    assert first["next_offset"] == 2001
    rest = files.read("长文.md", offset=first["next_offset"])
    assert rest["content"].splitlines()[-1] == "2500\t第 2500 行"
    assert "next_offset" not in rest


def test_edit_requires_fresh_read(files):
    files.write("待办.md", "- 周三交报告\n- 周五体检\n")
    path = files.policy.workspace / "待办.md"
    files.read_versions.clear()  # 模拟新会话：没读过

    with pytest.raises(PermissionError, match="还没有读过"):
        files.edit("待办.md", "周五", "周六")
    files.read("待办.md")
    os.utime(path, ns=(0, 0))  # 模拟用户在别处改过文件
    with pytest.raises(PermissionError, match="被改过"):
        files.edit("待办.md", "周五", "周六")

    files.read("待办.md")
    result = files.edit("待办.md", "周五", "周六")
    assert "+- 周六体检" in result["diff"]
    files.edit("待办.md", "周三", "周四")  # 自己刚改过的文件可以接着改，不用重新读
    assert path.read_text() == "- 周四交报告\n- 周六体检\n"


def test_edit_matches_crlf_file_and_keeps_line_endings(files):
    path = files.policy.workspace / "windows.txt"
    path.write_bytes(b"a\r\nb\r\nb\r\n")
    files.read("windows.txt")

    with pytest.raises(ValueError, match="出现了 2 次"):
        files.edit("windows.txt", "b", "c")
    files.edit("windows.txt", "a\nb", "a\nB")  # 模型给的换行总是 \n
    assert path.read_bytes() == b"a\r\nB\r\nb\r\n"
