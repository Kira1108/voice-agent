"""DeepSeek Search / Fetch tools, against a fake HTTP server."""
import json

import httpx

from app.plugins.llm.web_tools import DeepSeekWebTools


def tools(handler, **kwargs) -> DeepSeekWebTools:
    web = DeepSeekWebTools(api_key="k", **kwargs)
    web._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return web


async def test_search_sends_the_query_and_formats_the_results_for_the_model():
    seen = {}

    def handler(request):
        seen["path"], seen["auth"], seen["body"] = request.url.path, request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"results": [
            {"title": "北京天气", "url": "https://w.cn/bj", "summary": "明天多云", "snippet": "短", "date": "2026-11-25T00:00:00+00:00"},
        ]})

    out = await tools(handler).search("北京天气", count=50)
    assert seen == {"path": "/v1/tools/search", "auth": "Bearer k", "body": {"query": "北京天气", "count": 10}}
    assert out == "[1] 北京天气（2026-11-25）\nhttps://w.cn/bj\n明天多云"


async def test_fetch_returns_markdown_cut_to_size_and_errors_as_text():
    page = tools(lambda r: httpx.Response(200, json={"title": "T", "content": "x" * 50}), max_page_chars=10)
    assert await page.fetch("https://a.cn") == "# T\n\n" + "x" * 10 + "\n…（后面省略）"
    missing = tools(lambda r: httpx.Response(404, json={"error": {"message": "failed to fetch URL"}}))
    assert await missing.fetch("https://a.cn") == "读取失败：HTTP 404: failed to fetch URL"
    empty = tools(lambda r: httpx.Response(200, json={"results": []}))
    assert await empty.search("x") == "没有搜到结果"
