import httpx
import pytest
from unittest.mock import patch

import tools.webplus_backend as webplus_backend


class _FakeSyncResponse:
    def __init__(self, text: str, *, url: str = "https://html.duckduckgo.com/html/?q=test"):
        self.text = text
        self.url = httpx.URL(url)

    def raise_for_status(self):
        return None


class _FakeAsyncResponse:
    def __init__(self, text: str, *, url: str = "https://example.com/final", headers: dict | None = None):
        self.text = text
        self.url = httpx.URL(url)
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}

    def raise_for_status(self):
        return None

    def json(self):
        raise AssertionError("json() should not be called for HTML responses")


class _FakeAsyncClient:
    def __init__(self, response: _FakeAsyncResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str):
        return self._response


def test_duckduckgo_html_search_parses_results_with_html_parser():
    html = """
    <html>
      <body>
        <div class="results">
          <div class="result results_links_deep web-result">
            <h2><a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Ffire-pdf">Introducing Fire-PDF</a></h2>
            <a class="result__snippet">Fire-PDF is a fast Rust-based PDF parsing engine.</a>
          </div>
          <div class="result results_links_deep web-result">
            <h2><a class="result__a" href="https://example.org/docs">Second Result</a></h2>
            <div class="result__snippet">Documentation page for parser internals.</div>
          </div>
        </div>
      </body>
    </html>
    """

    with patch("tools.webplus_backend.httpx.get", return_value=_FakeSyncResponse(html)):
        results = webplus_backend._duckduckgo_html_search("fire pdf", limit=2)

    assert results == [
        {
            "title": "Introducing Fire-PDF",
            "url": "https://example.com/fire-pdf",
            "description": "Fire-PDF is a fast Rust-based PDF parsing engine.",
        },
        {
            "title": "Second Result",
            "url": "https://example.org/docs",
            "description": "Documentation page for parser internals.",
        },
    ]


@pytest.mark.asyncio
async def test_local_web_extract_prefers_structured_main_content_for_html_pages():
    html = """
    <html>
      <head>
        <title>Fire PDF Docs</title>
      </head>
      <body>
        <header>Site Header</header>
        <nav>Top Navigation</nav>
        <main>
          <article>
            <h1>Fire PDF Docs</h1>
            <p>Fast PDF parsing for local extraction.</p>
            <ul>
              <li>Structured content</li>
              <li>Lower noise</li>
            </ul>
          </article>
        </main>
        <footer>Footer Links</footer>
        <script>window.__bot = true;</script>
      </body>
    </html>
    """
    fake_response = _FakeAsyncResponse(html)

    with patch.object(webplus_backend, "check_website_access", return_value=None), patch(
        "tools.webplus_backend.httpx.AsyncClient",
        side_effect=lambda **kwargs: _FakeAsyncClient(fake_response),
    ):
        result = await webplus_backend.local_web_extract("https://example.com/docs")

    assert result["url"] == "https://example.com/final"
    assert result["title"] == "Fire PDF Docs"
    assert "Fast PDF parsing for local extraction." in result["content"]
    assert "Structured content" in result["content"]
    assert "Top Navigation" not in result["content"]
    assert "Footer Links" not in result["content"]
    assert "window.__bot" not in result["content"]
    assert result["raw_content"] == result["content"]