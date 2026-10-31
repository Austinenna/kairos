import io
import json
import urllib.error

import pytest

from kairos.tools.web import web_tools
from kairos.web import client as web_client
from kairos.web.client import WebClient, WebError

GATEWAY = "https://gateway.test/v1"


def fake_gateway(monkeypatch, response: dict | None = None, status: int = 200):
    """替换 urlopen：记录发出的请求，返回给定的 JSON 或错误状态。"""
    sent = []

    def urlopen(request, timeout):
        sent.append(request)
        body = json.dumps(response or {}).encode()
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "error", {}, io.BytesIO(body))
        return io.BytesIO(body)

    monkeypatch.setattr(web_client.urllib.request, "urlopen", urlopen)
    return sent


def test_search_sends_native_body_and_trims_summaries(monkeypatch):
    long_summary = "正文" * 1000
    result = {
        "title": "标题",
        "url": "https://a.test/",
        "site_name": "站点",
        "date": "2025-12-03T16:00:00+00:00",
        "summary": long_summary,
        "snippet": "片段",
        "site_icon": "x",
        "rerank_score": 1.9,
        "authority": 4,
    }
    sent = fake_gateway(monkeypatch, {"results": [result]})
    search = web_tools(WebClient(GATEWAY, "gw_token")).handlers["web_search"]

    short = search("DeepSeek 价格")
    full = search("DeepSeek 价格", count=50, full=True)

    assert sent[0].full_url == GATEWAY + "/tools/search"
    assert sent[0].get_header("Authorization") == "Bearer gw_token"
    assert json.loads(sent[0].data) == {"query": "DeepSeek 价格", "count": 5}
    assert json.loads(sent[1].data)["count"] == 20  # 超过上限时按 20 条
    assert short["note"] and short["results"][0] == {
        "title": "标题",
        "url": "https://a.test/",
        "site": "站点",
        "date": "2025-12-03",
        "summary": long_summary[:800] + "……",
    }
    assert full["results"][0]["summary"] == long_summary


def test_fetch_truncates_long_pages_and_rejects_bad_input(monkeypatch):
    sent = fake_gateway(monkeypatch, {"title": "长文", "content": "字" * 60_000})
    tools = web_tools(WebClient(GATEWAY, "gw_token")).handlers

    page = tools["web_fetch"]("https://a.test/long")

    assert sent[0].full_url == GATEWAY + "/tools/fetch"
    assert json.loads(sent[0].data) == {"url": "https://a.test/long"}
    assert len(page["content"]) == 50_000 and "60000" in page["truncated"]
    with pytest.raises(ValueError):
        tools["web_fetch"]("file:///etc/passwd")
    with pytest.raises(ValueError):
        tools["web_search"]("长" * 129)
    assert len(sent) == 1  # 参数不合法时不发请求


def test_gateway_error_message_reaches_the_model(monkeypatch):
    fake_gateway(monkeypatch, {"error": {"message": "此项目未获搜索或抓取授权"}}, status=403)

    with pytest.raises(WebError, match="HTTP 403：此项目未获搜索或抓取授权"):
        WebClient(GATEWAY, "gw_token").search("x", 5)
