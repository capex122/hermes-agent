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


def test_web_inspect_local_collects_richer_devtools_data():
    def fake_browser_console(*, clear=False, expression=None, task_id=None):
        assert task_id == "inspect-extended"
        if expression is None:
            assert clear is True
            return json.dumps(
                {
                    "success": True,
                    "console_messages": [{"type": "log", "text": "ready", "source": "console"}],
                    "js_errors": [{"message": "warn", "source": "exception"}],
                    "total_messages": 1,
                    "total_errors": 1,
                }
            )
        if "document.readyState" in expression:
            return json.dumps(
                {
                    "success": True,
                    "result": {
                        "title": "Example Page",
                        "url": "https://example.com",
                        "readyState": "complete",
                    },
                    "result_type": "dict",
                }
            )
        if "document.querySelector" in expression:
            return json.dumps(
                {
                    "success": True,
                    "result": {"found": True, "selector": "#app", "tagName": "DIV"},
                    "result_type": "dict",
                }
            )
        if expression == "document.title":
            return json.dumps({"success": True, "result": "Example Page", "result_type": "str"})
        if "performance.getEntriesByType('resource')" in expression:
            return json.dumps(
                {
                    "success": True,
                    "result": {"count": 2, "byInitiator": {"fetch": 1, "script": 1}},
                    "result_type": "dict",
                }
            )
        if "localStorage" in expression:
            return json.dumps(
                {
                    "success": True,
                    "result": {"cookiesEnabled": True, "localStorage": {"count": 1, "items": [{"key": "token", "value": "abc"}]}},
                    "result_type": "dict",
                }
            )
        raise AssertionError(f"Unexpected expression: {expression}")

    with patch.object(webplus_tool, "ensure_bundled_web_service", return_value=False), patch.object(
        webplus_tool,
        "_browser_available",
        return_value=True,
    ), patch(
        "tools.browser_tool.browser_snapshot",
        return_value=json.dumps({"success": True, "snapshot": "snapshot text", "element_count": 4}),
    ), patch(
        "tools.browser_tool.browser_console",
        side_effect=fake_browser_console,
    ), patch(
        "tools.browser_tool.browser_get_images",
        return_value=json.dumps({"success": True, "count": 1, "images": [{"src": "https://example.com/a.png", "alt": "diagram"}]}),
    ):
        result = json.loads(
            webplus_tool.web_inspect_tool(
                selector="#app",
                include_console=True,
                include_network=True,
                include_images=True,
                include_storage=True,
                clear_console=True,
                full_snapshot=True,
                expression="document.title",
                task_id="inspect-extended",
            )
        )

    assert result["success"] is True
    assert result["page"]["title"] == "Example Page"
    assert result["selector_result"]["selector"] == "#app"
    assert result["eval_result"] == "Example Page"
    assert result["console_summary"] == {"message_count": 1, "error_count": 1}
    assert result["network"]["count"] == 2
    assert result["storage"]["cookiesEnabled"] is True
    assert result["images"]["count"] == 1
    assert result["requested_checks"]["full_snapshot"] is True


def test_web_inspect_forwards_extended_payload_to_managed_service():
    payload = {"success": True, "mode": "inspect"}

    with patch.object(webplus_tool, "ensure_bundled_web_service", return_value=True), patch(
        "tools.webplus_backend._call_bundled_backend_sync",
        return_value=payload,
    ) as mock_remote:
        result = json.loads(
            webplus_tool.web_inspect_tool(
                url="https://example.com",
                selector="#app",
                selectors=["main", "footer"],
                include_console=False,
                include_network=True,
                include_images=True,
                include_storage=True,
                clear_console=True,
                full_snapshot=True,
                expression="document.title",
                expressions=["location.href", "document.readyState"],
                task_id="inspect-remote",
            )
        )

    assert result == payload
    mock_remote.assert_called_once_with(
        "/v1/inspect",
        {
            "url": "https://example.com",
            "selector": "#app",
            "selectors": ["main", "footer"],
            "include_console": False,
            "include_network": True,
            "include_images": True,
            "include_storage": True,
            "expression": "document.title",
            "expressions": ["location.href", "document.readyState"],
            "clear_console": True,
            "full_snapshot": True,
            "task_id": "inspect-remote",
        },
    )