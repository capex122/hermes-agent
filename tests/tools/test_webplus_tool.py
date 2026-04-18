import json
from unittest.mock import AsyncMock, patch

import pytest

import tools.webplus_tool as webplus_tool


@pytest.mark.asyncio
async def test_web_fetch_prefers_browser_when_render_requested():
    expected = {"success": True, "mode": "browser", "url": "https://example.com"}

    with patch.object(webplus_tool, "_browser_available", return_value=True), patch.object(
        webplus_tool,
        "_browser_fetch",
        return_value=json.dumps(expected),
    ) as mock_browser_fetch:
        result = json.loads(
            await webplus_tool.web_fetch_tool(
                "https://example.com",
                render=True,
                task_id="task-1",
            )
        )

    assert result == expected
    mock_browser_fetch.assert_called_once_with(
        "https://example.com",
        selector=None,
        include_console=False,
        task_id="task-1",
    )


@pytest.mark.asyncio
async def test_web_fetch_prefers_managed_service_when_available():
    service_payload = {"success": True, "url": "https://example.com", "content": "from service"}

    with patch.object(webplus_tool, "ensure_bundled_web_service", return_value=True), patch.object(
        webplus_tool,
        "_call_bundled_backend",
        new=AsyncMock(return_value=service_payload),
    ) as mock_remote:
        result = json.loads(await webplus_tool.web_fetch_tool("https://example.com"))

    assert result["success"] is True
    assert result["mode"] == "bundled-service"
    assert result["content"] == "from service"
    mock_remote.assert_awaited_once()


@pytest.mark.asyncio
async def test_web_deep_search_aggregates_search_and_extract_results():
    search_payload = {
        "success": True,
        "data": {
            "web": [
                {"title": "One", "url": "https://example.com/1", "description": "first", "position": 1},
                {"title": "Two", "url": "https://example.com/2", "description": "second", "position": 2},
            ]
        },
    }
    extract_payload = {
        "results": [
            {"url": "https://example.com/1", "title": "One", "content": "page one"},
            {"url": "https://example.com/2", "title": "Two", "content": "page two"},
        ]
    }

    with patch("tools.web_tools.web_search_tool", return_value=json.dumps(search_payload)), patch(
        "tools.web_tools.web_extract_tool",
        new=AsyncMock(return_value=json.dumps(extract_payload)),
    ):
        result = json.loads(
            await webplus_tool.web_deep_search_tool(
                "test query",
                top_k=5,
                extract_top=2,
            )
        )

    assert result["success"] is True
    assert result["mode"] == "native-deep-search"
    assert len(result["search_results"]) == 2
    assert len(result["extracted_pages"]) == 2


def test_youtube_search_uses_browser_fallback_when_web_backend_missing():
    payload = {"success": True, "data": {"web": [{"title": "Video", "url": "https://youtube.com/watch?v=abc"}]}}

    with patch.object(webplus_tool, "_web_available", return_value=False), patch.object(
        webplus_tool,
        "_browser_available",
        return_value=True,
    ), patch.object(
        webplus_tool,
        "_youtube_search_via_browser",
        return_value=json.dumps(payload),
    ) as mock_browser:
        result = json.loads(webplus_tool.youtube_search_tool("cats", limit=3, task_id="yt-task"))

    assert result == payload
    mock_browser.assert_called_once_with("cats", 3, "yt-task")


def test_youtube_search_prefers_managed_service_when_available():
    payload = {"success": True, "data": {"web": [{"title": "Video"}]}}

    with patch.object(webplus_tool, "ensure_bundled_web_service", return_value=True), patch(
        "tools.webplus_backend._call_bundled_backend_sync",
        return_value=payload,
    ) as mock_remote:
        result = json.loads(webplus_tool.youtube_search_tool("cats", limit=3, task_id="yt-task"))

    assert result == payload
    mock_remote.assert_called_once()


def test_youtube_transcript_tool_serializes_local_helper_result():
    payload = {"success": True, "video_id": "abc123def45", "full_text": "hello world"}

    with patch.object(webplus_tool, "fetch_youtube_transcript_local", return_value=payload) as mock_fetch:
        result = json.loads(
            webplus_tool.youtube_transcript_tool(
                "https://youtu.be/abc123def45",
                language="en,tr",
                include_timestamps=True,
            )
        )

    assert result == payload
    mock_fetch.assert_called_once_with(
        "https://youtu.be/abc123def45",
        languages=["en", "tr"],
        include_timestamps=True,
    )


def test_youtube_transcript_prefers_managed_service_when_available():
    payload = {"success": True, "video_id": "abc123def45", "full_text": "service transcript"}

    with patch.object(webplus_tool, "ensure_bundled_web_service", return_value=True), patch(
        "tools.webplus_backend._call_bundled_backend_sync",
        return_value=payload,
    ) as mock_remote:
        result = json.loads(
            webplus_tool.youtube_transcript_tool(
                "https://youtu.be/abc123def45",
                language="en,tr",
                include_timestamps=True,
            )
        )

    assert result == payload
    mock_remote.assert_called_once()


def test_web_inspect_requires_browser_backend():
    with patch.object(webplus_tool, "_browser_available", return_value=False):
        result = json.loads(webplus_tool.web_inspect_tool(url="https://example.com"))

    assert result["success"] is False
    assert "Browser backend is required" in result["error"]


def test_web_inspect_prefers_managed_service_when_available():
    payload = {
        "success": True,
        "mode": "inspect",
        "network": {"count": 2},
        "development_capabilities": {"console": True, "eval": True, "network": True},
    }

    with patch.object(webplus_tool, "ensure_bundled_web_service", return_value=True), patch(
        "tools.webplus_backend._call_bundled_backend_sync",
        return_value=payload,
    ) as mock_remote:
        result = json.loads(
            webplus_tool.web_inspect_tool(
                url="https://example.com",
                selector="#app",
                include_console=True,
                include_network=True,
                expression="document.title",
                task_id="inspect-1",
            )
        )

    assert result == payload
    mock_remote.assert_called_once()