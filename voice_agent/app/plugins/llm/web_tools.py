"""Web tools for the thinker: DeepSeek Search and Fetch APIs (https://api-docs.deepseek.com/guides/search/).

    search(query)  POST /v1/tools/search  ranked results, each with a long summary written for a model
    fetch(url)     POST /v1/tools/fetch   one page as Markdown

Both use DEEPSEEK_API_KEY. Errors come back as text, so the thinker can read them and change course instead
of the whole task failing.
"""
import httpx

from app.config import get_config

BASE_URL = "https://api.deepseek.com/v1/tools"


class DeepSeekWebTools:
    def __init__(self, api_key: str | None = None, timeout: float = 20.0, max_page_chars: int = 6000):
        self.api_key = api_key or get_config().deepseek_api_key
        self.max_page_chars = max_page_chars
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    async def _post(self, path: str, body: dict) -> dict:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        response = await self._client.post(
            f"{BASE_URL}/{path}", json=body, headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if response.status_code != 200:
            try:
                message = response.json()["error"]["message"]
            except Exception:
                message = response.text[:200]
            raise RuntimeError(f"HTTP {response.status_code}: {message}")
        return response.json()

    async def search(self, query: str, count: int = 5) -> str:
        """Search the web. Supports operators such as site:, lang:zh, after:2025-01-01.

        Args:
            query: what to search for, 1 to 128 characters.
            count: how many results, 1 to 10.
        """
        try:
            data = await self._post("search", {"query": query[:128], "count": max(1, min(count, 10))})
        except Exception as e:
            return f"搜索失败：{e}"
        results = data.get("results") or []
        if not results:
            return "没有搜到结果"
        blocks = []
        for i, r in enumerate(results, 1):
            date = f"（{r['date'][:10]}）" if r.get("date") else ""
            blocks.append(f"[{i}] {r.get('title', '')}{date}\n{r.get('url', '')}\n{r.get('summary') or r.get('snippet', '')}")
        return "\n\n".join(blocks)

    async def fetch(self, url: str) -> str:
        """Read a web page as Markdown, e.g. a result of search worth reading in full.

        Args:
            url: an http(s) URL.
        """
        try:
            data = await self._post("fetch", {"url": url})
        except Exception as e:
            return f"读取失败：{e}"
        content = data.get("content") or ""
        if len(content) > self.max_page_chars:
            content = content[:self.max_page_chars] + "\n…（后面省略）"
        return f"# {data.get('title', '')}\n\n{content}".strip()

    async def aclose(self):
        await self._client.aclose()


_default: DeepSeekWebTools | None = None


def _tools() -> DeepSeekWebTools:
    global _default
    if _default is None:
        _default = DeepSeekWebTools()
    return _default


async def web_search(query: str, count: int = 5) -> str:
    """Search the web for current information. Supports operators such as site:github.com, lang:zh,
    after:2025-01-01. Returns the top results, each with title, URL, date and a summary.

    Args:
        query: what to search for, 1 to 128 characters.
        count: how many results, 1 to 10.
    """
    return await _tools().search(query, count)


async def web_fetch(url: str) -> str:
    """Read a web page as Markdown, e.g. a search result worth reading in full.

    Args:
        url: an http(s) URL.
    """
    return await _tools().fetch(url)
