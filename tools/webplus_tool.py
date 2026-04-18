#!/usr/bin/env python3
"""Built-in webplus tools for bundled/local web research workflows."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from tools.registry import registry, tool_error, tool_result
from tools.webplus_backend import (
    _call_bundled_backend,
    bundled_backend_is_available,
    fetch_youtube_transcript_local,
    youtube_transcript_support_available,
)
from tools.webplus_service_manager import ensure_bundled_web_service

logger = logging.getLogger(__name__)


def _json_load(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"success": True, "result": parsed}
    except Exception:
        return {"success": False, "error": "Invalid JSON response", "raw": value}


def _browser_available() -> bool:
    from tools.browser_tool import check_browser_requirements

    return check_browser_requirements()


def _web_available() -> bool:
    from tools.web_tools import check_web_api_key

    return check_web_api_key()


def check_web_fetch_requirements() -> bool:
    return _web_available() or _browser_available()


def check_web_deep_search_requirements() -> bool:
    return _web_available()


def check_youtube_search_requirements() -> bool:
    return _web_available() or _browser_available()


def check_youtube_transcript_requirements() -> bool:
    return youtube_transcript_support_available() or bundled_backend_is_available()


def check_web_inspect_requirements() -> bool:
    return _browser_available()


def _selector_eval_expression(selector: str) -> str:
    selector_json = json.dumps(selector)
    return (
        "(() => {"
        f"const selector = {selector_json};"
        "const el = document.querySelector(selector);"
        "if (!el) return {found: false, selector};"
        "const rect = el.getBoundingClientRect();"
        "const attrs = {};"
        "for (const attr of Array.from(el.attributes || [])) attrs[attr.name] = attr.value;"
        "return {"
        "found: true,"
        "selector,"
        "tagName: el.tagName,"
        "text: ((el.innerText || el.textContent || '').trim()).slice(0, 4000),"
        "html: (el.outerHTML || '').slice(0, 4000),"
        "attributes: attrs,"
        "rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height}"
        "};"
        "})()"
    )


def _browser_fetch(url: str, *, selector: Optional[str], include_console: bool, task_id: Optional[str]) -> str:
    from tools.browser_tool import browser_console, browser_navigate, browser_snapshot

    nav = _json_load(browser_navigate(url, task_id=task_id))
    if not nav.get("success"):
        return json.dumps(nav, ensure_ascii=False)

    snapshot = _json_load(browser_snapshot(full=False, task_id=task_id))
    response: Dict[str, Any] = {
        "success": True,
        "mode": "browser",
        "url": nav.get("url", url),
        "title": nav.get("title", ""),
        "snapshot": snapshot.get("snapshot", nav.get("snapshot", "")),
        "element_count": snapshot.get("element_count", nav.get("element_count", 0)),
    }

    if selector:
        selector_result = _json_load(
            browser_console(expression=_selector_eval_expression(selector), task_id=task_id)
        )
        response["selector_result"] = selector_result.get("result", selector_result)

    if include_console:
        console_output = _json_load(browser_console(task_id=task_id))
        response["console_messages"] = console_output.get("console_messages", [])
        response["js_errors"] = console_output.get("js_errors", [])

    return tool_result(response)


async def web_fetch_tool(
    url: str,
    *,
    render: bool = False,
    selector: Optional[str] = None,
    include_console: bool = False,
    task_id: Optional[str] = None,
) -> str:
    """Fetch a page with extract-first fallback and optional browser inspection."""
    if ensure_bundled_web_service():
        payload = {
            "url": url,
            "selector": selector or "",
            "include_console": include_console,
            "full_snapshot": False,
            "task_id": task_id or "",
        }
        remote_path = "/v1/inspect" if (render or selector or include_console) else "/v1/extract"
        remote = await _call_bundled_backend(remote_path, payload)
        if isinstance(remote, dict) and (remote.get("success") or remote.get("content") or remote.get("snapshot") or remote.get("error")):
            if remote_path == "/v1/extract":
                normalized = {
                    "success": not bool(remote.get("error")),
                    "mode": "bundled-service",
                    "url": remote.get("url", url),
                    "title": remote.get("title", ""),
                    "content": remote.get("content", ""),
                }
                if remote.get("error"):
                    normalized["error"] = remote["error"]
                if "blocked_by_policy" in remote:
                    normalized["blocked_by_policy"] = remote["blocked_by_policy"]
                return tool_result(normalized)
            return json.dumps(remote, ensure_ascii=False)

    if render or selector or include_console:
        if _browser_available():
            return _browser_fetch(url, selector=selector, include_console=include_console, task_id=task_id)
        if selector or include_console:
            return tool_error(
                "Browser inspection requires the browser tool backend to be available",
                success=False,
            )

    from tools.web_tools import web_extract_tool

    extracted = _json_load(
        await web_extract_tool([url], format="markdown", use_llm_processing=False)
    )
    if extracted.get("error"):
        if _browser_available():
            return _browser_fetch(url, selector=None, include_console=False, task_id=task_id)
        return json.dumps(extracted, ensure_ascii=False)

    results = extracted.get("results", [])
    if not results:
        if _browser_available():
            return _browser_fetch(url, selector=None, include_console=False, task_id=task_id)
        return tool_error("No content extracted", success=False)

    first = results[0]
    response = {
        "success": not bool(first.get("error")),
        "mode": "extract",
        "url": first.get("url", url),
        "title": first.get("title", ""),
        "content": first.get("content", ""),
    }
    if first.get("error"):
        response["error"] = first["error"]
    if "blocked_by_policy" in first:
        response["blocked_by_policy"] = first["blocked_by_policy"]
    return tool_result(response)


async def web_deep_search_tool(
    query: str,
    *,
    top_k: int = 5,
    extract_top: int = 3,
    site: Optional[str] = None,
) -> str:
    """Run a search-plus-extract workflow and return combined results."""
    if ensure_bundled_web_service():
        remote = await _call_bundled_backend(
            "/v1/research/deep",
            {
                "query": query,
                "top_k": top_k,
                "extract_top": extract_top,
                "site": site or "",
            },
        )
        if isinstance(remote, dict) and remote.get("success"):
            return json.dumps(remote, ensure_ascii=False)

    from tools.web_tools import web_extract_tool, web_search_tool

    effective_query = f"site:{site} {query}" if site else query
    search_result = _json_load(web_search_tool(effective_query, limit=max(1, min(top_k, 10))))
    if not search_result.get("success"):
        return json.dumps(search_result, ensure_ascii=False)

    web_results = search_result.get("data", {}).get("web", [])
    urls: List[str] = []
    for item in web_results:
        url = str(item.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)
        if len(urls) >= max(1, min(extract_top, 5)):
            break

    extracted = {"results": []}
    if urls:
        extracted = _json_load(
            await web_extract_tool(urls, format="markdown", use_llm_processing=True)
        )

    return tool_result(
        success=True,
        mode="native-deep-search",
        query=query,
        site=site or "",
        search_results=web_results,
        extracted_pages=extracted.get("results", []),
    )


def _youtube_search_via_browser(query: str, limit: int, task_id: Optional[str]) -> str:
    from tools.browser_tool import browser_console, browser_navigate

    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    nav = _json_load(browser_navigate(search_url, task_id=task_id))
    if not nav.get("success"):
        return json.dumps(nav, ensure_ascii=False)

    expression = (
        "(() => Array.from(document.querySelectorAll('a#video-title'))"
        f".slice(0, {max(1, min(limit, 10))})"
        ".map((el, index) => ({"
        "title: (el.textContent || '').trim(),"
        "url: el.href,"
        "description: (el.getAttribute('aria-label') || '').trim(),"
        "position: index + 1"
        "})))()"
    )
    evaluated = _json_load(browser_console(expression=expression, task_id=task_id))
    if not evaluated.get("success"):
        return json.dumps(evaluated, ensure_ascii=False)

    return tool_result(
        success=True,
        source="youtube-browser",
        data={"web": evaluated.get("result", [])},
    )


def youtube_search_tool(query: str, *, limit: int = 5, task_id: Optional[str] = None) -> str:
    """Search YouTube using bundled/local web search with browser fallback."""
    if ensure_bundled_web_service():
        from tools.webplus_backend import _call_bundled_backend_sync

        remote = _call_bundled_backend_sync(
            "/v1/youtube/search",
            {"query": query, "limit": max(1, min(limit, 10))},
        )
        if isinstance(remote, dict) and remote.get("success"):
            return json.dumps(remote, ensure_ascii=False)

    if _web_available():
        from tools.web_tools import web_search_tool

        results = _json_load(
            web_search_tool(
                f"site:youtube.com/watch {query}".strip(),
                limit=max(1, min(limit, 10)),
            )
        )
        if results.get("success"):
            results["source"] = results.get("source", "youtube-search")
        return json.dumps(results, ensure_ascii=False)

    if _browser_available():
        return _youtube_search_via_browser(query, limit, task_id)

    return tool_error("No web or browser backend available for YouTube search", success=False)


def youtube_transcript_tool(
    url: str,
    *,
    language: str = "",
    include_timestamps: bool = False,
) -> str:
    """Fetch a YouTube transcript locally using youtube-transcript-api when available."""
    if ensure_bundled_web_service():
        from tools.webplus_backend import _call_bundled_backend_sync

        remote = _call_bundled_backend_sync(
            "/v1/youtube/transcript",
            {
                "url": url,
                "languages": language,
                "include_timestamps": include_timestamps,
            },
        )
        if isinstance(remote, dict) and (remote.get("success") or remote.get("error")):
            return json.dumps(remote, ensure_ascii=False)

    languages = [part.strip() for part in language.split(",") if part.strip()] if language else None
    result = fetch_youtube_transcript_local(
        url,
        languages=languages,
        include_timestamps=include_timestamps,
    )
    return json.dumps(result, ensure_ascii=False)


def web_inspect_tool(
    url: str = "",
    *,
    selector: str = "",
    include_console: bool = True,
    include_network: bool = True,
    expression: str = "",
    task_id: Optional[str] = None,
) -> str:
    """Inspect a rendered page using the browser accessibility tree and DOM eval."""
    if ensure_bundled_web_service():
        from tools.webplus_backend import _call_bundled_backend_sync

        remote = _call_bundled_backend_sync(
            "/v1/inspect",
            {
                "url": url,
                "selector": selector,
                "include_console": include_console,
                "include_network": include_network,
                "expression": expression,
                "clear_console": False,
                "task_id": task_id or "",
            },
        )
        if isinstance(remote, dict) and (remote.get("success") or remote.get("error")):
            return json.dumps(remote, ensure_ascii=False)

    if not _browser_available():
        return tool_error("Browser backend is required for web inspection", success=False)

    effective_url = url.strip()
    effective_selector = selector.strip() or None
    if effective_url:
        result = _browser_fetch(
            effective_url,
            selector=effective_selector,
            include_console=include_console,
            task_id=task_id,
        )
        parsed = _json_load(result)
        parsed["mode"] = "inspect"
        return json.dumps(parsed, ensure_ascii=False)

    from tools.browser_tool import browser_console, browser_snapshot

    snapshot = _json_load(browser_snapshot(full=False, task_id=task_id))
    if not snapshot.get("success"):
        return json.dumps(snapshot, ensure_ascii=False)

    response: Dict[str, Any] = {
        "success": True,
        "mode": "inspect",
        "snapshot": snapshot.get("snapshot", ""),
        "element_count": snapshot.get("element_count", 0),
        "development_capabilities": {
            "console": True,
            "eval": True,
            "network": include_network,
            "raw_js_eval_tool": "browser_console",
        },
    }
    if effective_selector:
        selector_result = _json_load(
            browser_console(expression=_selector_eval_expression(effective_selector), task_id=task_id)
        )
        response["selector_result"] = selector_result.get("result", selector_result)
    if expression:
        eval_result = _json_load(browser_console(expression=expression, task_id=task_id))
        response["eval_result"] = eval_result.get("result", eval_result)
    if include_console:
        console_output = _json_load(browser_console(task_id=task_id))
        response["console_messages"] = console_output.get("console_messages", [])
        response["js_errors"] = console_output.get("js_errors", [])
    if include_network:
        network_output = _json_load(
            browser_console(
                expression=(
                    "(() => {"
                    "const entries = performance.getEntriesByType('resource').slice(-100).map((entry, index) => ({"
                    "index, name: entry.name, initiatorType: entry.initiatorType || '', startTime: entry.startTime, duration: entry.duration,"
                    "transferSize: entry.transferSize ?? null, encodedBodySize: entry.encodedBodySize ?? null, decodedBodySize: entry.decodedBodySize ?? null,"
                    "nextHopProtocol: entry.nextHopProtocol || ''"
                    "}));"
                    "const nav = performance.getEntriesByType('navigation')[0];"
                    "return {"
                    "count: entries.length,"
                    "document: nav ? {type: nav.type, domComplete: nav.domComplete, loadEventEnd: nav.loadEventEnd, transferSize: nav.transferSize ?? null} : null,"
                    "entries"
                    "};"
                    "})()"
                ),
                task_id=task_id,
            )
        )
        response["network"] = network_output.get("result", network_output)

    return tool_result(response)


WEB_FETCH_SCHEMA = {
    "name": "web_fetch",
    "description": "Fetch a web page through Hermes' bundled/local web workflow. Use render=true for a browser-backed fetch when you need a rendered page, and selector/include_console for DOM inspection.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch"},
            "render": {"type": "boolean", "description": "Use the browser-backed render path when available", "default": False},
            "selector": {"type": "string", "description": "Optional CSS selector to inspect on the page"},
            "include_console": {"type": "boolean", "description": "Include browser console output and JS errors when using browser fetch", "default": False},
        },
        "required": ["url"],
    },
}

WEB_DEEP_SEARCH_SCHEMA = {
    "name": "web_deep_search",
    "description": "Run a search-plus-fetch workflow over the top search results and return both search hits and extracted page content.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The research query"},
            "top_k": {"type": "integer", "description": "How many search results to collect", "default": 5, "minimum": 1, "maximum": 10},
            "extract_top": {"type": "integer", "description": "How many of the top results to fetch and extract", "default": 3, "minimum": 1, "maximum": 5},
            "site": {"type": "string", "description": "Optional site/domain filter, e.g. docs.python.org"},
        },
        "required": ["query"],
    },
}

YOUTUBE_SEARCH_SCHEMA = {
    "name": "youtube_search",
    "description": "Search YouTube for videos using the bundled/local web workflow with a browser fallback when needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The YouTube search query"},
            "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 5, "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
}

YOUTUBE_TRANSCRIPT_SCHEMA = {
    "name": "youtube_transcript",
    "description": "Fetch a YouTube transcript locally when transcripts are available. Optionally provide preferred languages as a comma-separated string.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "YouTube URL or video ID"},
            "language": {"type": "string", "description": "Optional comma-separated language codes, e.g. en,tr"},
            "include_timestamps": {"type": "boolean", "description": "Include a timestamped transcript view", "default": False},
        },
        "required": ["url"],
    },
}

WEB_INSPECT_SCHEMA = {
    "name": "web_inspect",
    "description": "Inspect a rendered web page using a higher-level development diagnostics workflow. Returns snapshot data, optional selector lookup, console messages, JS errors, optional resource-timing network summary, and optional evaluated expression output. Use browser_console for raw ad hoc JavaScript evaluation.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Optional URL to open before inspection. Leave empty to inspect the current browser page."},
            "selector": {"type": "string", "description": "Optional CSS selector to inspect in the current page"},
            "include_console": {"type": "boolean", "description": "Include console messages and JS errors", "default": True},
            "include_network": {"type": "boolean", "description": "Include a best-effort network summary from the browser Performance API", "default": True},
            "expression": {"type": "string", "description": "Optional JavaScript expression to evaluate and include in the inspect response. For general raw JS usage, prefer browser_console."},
        },
        "required": [],
    },
}


registry.register(
    name="web_fetch",
    toolset="webplus",
    schema=WEB_FETCH_SCHEMA,
    handler=lambda args, **kw: web_fetch_tool(
        url=args.get("url", ""),
        render=bool(args.get("render", False)),
        selector=args.get("selector") or None,
        include_console=bool(args.get("include_console", False)),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_web_fetch_requirements,
    is_async=True,
    emoji="🌐",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_deep_search",
    toolset="webplus",
    schema=WEB_DEEP_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_deep_search_tool(
        query=args.get("query", ""),
        top_k=int(args.get("top_k", 5) or 5),
        extract_top=int(args.get("extract_top", 3) or 3),
        site=args.get("site") or None,
    ),
    check_fn=check_web_deep_search_requirements,
    is_async=True,
    emoji="🧭",
    max_result_size_chars=100_000,
)
registry.register(
    name="youtube_search",
    toolset="webplus",
    schema=YOUTUBE_SEARCH_SCHEMA,
    handler=lambda args, **kw: youtube_search_tool(
        query=args.get("query", ""),
        limit=int(args.get("limit", 5) or 5),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_youtube_search_requirements,
    emoji="▶️",
    max_result_size_chars=100_000,
)
registry.register(
    name="youtube_transcript",
    toolset="webplus",
    schema=YOUTUBE_TRANSCRIPT_SCHEMA,
    handler=lambda args, **kw: youtube_transcript_tool(
        url=args.get("url", ""),
        language=args.get("language", ""),
        include_timestamps=bool(args.get("include_timestamps", False)),
    ),
    check_fn=check_youtube_transcript_requirements,
    emoji="📝",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_inspect",
    toolset="webplus",
    schema=WEB_INSPECT_SCHEMA,
    handler=lambda args, **kw: web_inspect_tool(
        url=args.get("url", ""),
        selector=args.get("selector", ""),
        include_console=bool(args.get("include_console", True)),
        include_network=bool(args.get("include_network", True)),
        expression=args.get("expression", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_web_inspect_requirements,
    emoji="🧪",
    max_result_size_chars=100_000,
)