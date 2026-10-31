"""调用网关转发的 DeepSeek 搜索和抓取接口。请求体与 DeepSeek 原生接口相同，不带 model。"""

import json
import urllib.error
import urllib.request

TIMEOUT = 60  # 秒；抓取慢的网页需要更久


class WebError(Exception):
    """网关或 DeepSeek 返回的错误。消息会原样交给模型，让它换关键词或换来源。"""


class WebClient:
    def __init__(self, base_url: str, api_key: str | None):
        self.base_url = base_url.rstrip("/")  # 网关地址，含 /v1
        self.api_key = api_key

    def search(self, query: str, count: int) -> list[dict]:
        return self._post("/tools/search", {"query": query, "count": count}).get("results", [])

    def fetch(self, url: str) -> dict:
        """返回 {"title": ..., "content": Markdown 正文}。"""
        return self._post("/tools/fetch", {"url": url})

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise WebError(f"HTTP {error.code}：{_error_message(error.read())}") from None


def _error_message(raw: bytes) -> str:
    """网关的错误是 {"error": {"message": ...}}；其他格式截取原文。"""
    try:
        return json.loads(raw)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return raw.decode(errors="replace")[:300]
