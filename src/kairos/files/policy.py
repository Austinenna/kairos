"""文件路径规则：哪些路径能读、哪些能写。读写文件前都先经过这里。

写入要过两层：
1. 拒绝清单：系统目录、密钥凭证、会自动执行的配置、Git 内部文件、Kairos 自己的数据。
   命中就拒绝，即使在白名单里，误加白名单时靠它兜底。
2. 白名单：只有工作目录能写。
读取只拦密钥凭证（内容会发给模型服务商），其余路径都能读。

比较时同时检查原路径和解析软链接后的真实路径，用软链接绕不过去；
并且不区分大小写，因为 macOS 的文件系统默认不区分，~/.SSH 就是 ~/.ssh。
规则参考 hermes-agent 的 file_safety.py 和 Claude Code 的 permissions/filesystem.ts。
"""

import os
from dataclasses import dataclass
from pathlib import Path

SYSTEM_DIRS = (
    "/etc", "/private/etc", "/private/var/db", "/private/var/root",
    "/System", "/Library", "/usr", "/bin", "/sbin",
)  # fmt: skip
# 下面三组都相对用户主目录。~/.kairos 里是配置密钥、数据库和记忆，只能通过各自的工具改。
SECRET_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh", ".config/gcloud", ".kairos")
SECRET_FILES = (".netrc", ".pgpass", ".npmrc", ".pypirc", ".git-credentials")
AUTORUN_FILES = (".zshrc", ".zprofile", ".bashrc", ".bash_profile", ".profile", ".gitconfig")


@dataclass(frozen=True)
class FilePolicy:
    home: Path  # 用户主目录，测试时换成临时目录
    workspace: Path  # 唯一可写的目录；模型给的相对路径也按它解析

    def readable(self, path: str) -> Path:
        """检查能否读取，返回解析后的真实路径。命中密钥凭证时抛 PermissionError。"""
        real, candidates = self._resolve(path)
        if self._is_secret(candidates):
            raise PermissionError(f"禁止读取 {path}：密钥或凭证文件")
        return real

    def writable(self, path: str) -> Path:
        """检查能否写入，返回解析后的真实路径。

        异常:
            PermissionError: 满足以下任一情况时抛出:
                - 路径命中拒绝清单;
                - 真实路径不在工作目录里。
        """
        real, candidates = self._resolve(path)
        reason = self._write_denied(candidates)
        if reason:
            raise PermissionError(f"禁止写入 {path}：{reason}")
        if not _under(candidates[-1], [_key(self.workspace.resolve())]):
            raise PermissionError(f"禁止写入 {path}：只能写入 {self.workspace} 里的文件")
        return real

    def _resolve(self, path: str) -> tuple[Path, list[str]]:
        """展开 ~、补全相对路径，返回真实路径，以及用来比较的两个小写路径：原路径、真实路径。"""
        given = Path(path).expanduser()
        if not given.is_absolute():
            given = self.workspace / given
        lexical = Path(os.path.normpath(given))  # 去掉 ..，但不跟随软链接
        real = lexical.resolve()
        return real, [_key(lexical), _key(real)]

    def _write_denied(self, candidates: list[str]) -> str | None:
        if any(_under(path, [_key(Path(d)) for d in SYSTEM_DIRS]) for path in candidates):
            return "系统目录"
        if self._is_secret(candidates):
            return "密钥或凭证文件"
        if any(path in self._home_paths(AUTORUN_FILES) for path in candidates):
            return "会自动执行的配置文件"
        if any(".git" in path.split("/") for path in candidates):
            return "Git 内部文件"
        return None

    def _is_secret(self, candidates: list[str]) -> bool:
        return any(
            _under(path, self._home_paths(SECRET_DIRS)) or path in self._home_paths(SECRET_FILES) or _is_env_file(path)
            for path in candidates
        )

    def _home_paths(self, names: tuple[str, ...]) -> list[str]:
        """主目录下的规则路径。主目录本身也可能是软链接，所以原路径和真实路径各算一份。"""
        homes = {_key(self.home), _key(self.home.resolve())}
        return [f"{home}/{name.lower()}" for home in homes for name in names]


def _key(path: Path) -> str:
    return str(path).lower()


def _under(path: str, roots: list[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def _is_env_file(path: str) -> bool:
    """项目里的 .env、.env.local、.envrc 等，通常放着密钥，在哪个目录都拦。"""
    name = path.rsplit("/", 1)[-1]
    return name in (".env", ".envrc") or name.startswith(".env.")
