#!/usr/bin/env python3
"""Shared browser inspection helpers for webplus local and bundled flows."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

MAX_INSPECT_SELECTORS = 10
MAX_INSPECT_EXPRESSIONS = 10
MAX_IMAGE_ITEMS = 25


def _json_load(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"success": True, "result": parsed}
    except Exception:
        return {"success": False, "error": "Invalid JSON response", "raw": raw}


def _normalize_string_list(
    primary: str = "",
    items: Optional[Iterable[Any]] = None,
    *,
    limit: int,
) -> List[str]:
    normalized: List[str] = []

    def _append(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)

    _append(primary)
    if items is not None:
        for item in items:
            _append(item)
            if len(normalized) >= limit:
                break

    return normalized[:limit]


def normalize_selector_inputs(selector: str = "", selectors: Optional[Iterable[Any]] = None) -> List[str]:
    return _normalize_string_list(selector, selectors, limit=MAX_INSPECT_SELECTORS)


def normalize_expression_inputs(expression: str = "", expressions: Optional[Iterable[Any]] = None) -> List[str]:
    return _normalize_string_list(expression, expressions, limit=MAX_INSPECT_EXPRESSIONS)


def selector_expression(selector: str) -> str:
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
        "id: el.id || '',"
        "className: el.className || '',"
        "role: el.getAttribute('role') || '',"
        "text: ((el.innerText || el.textContent || '').trim()).slice(0, 4000),"
        "html: (el.outerHTML || '').slice(0, 4000),"
        "attributes: attrs,"
        "rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height}"
        "};"
        "})()"
    )


PAGE_METADATA_EXPRESSION = (
    "(() => ({"
    "title: document.title || '',"
    "url: location.href,"
    "readyState: document.readyState || '',"
    "lang: document.documentElement?.lang || '',"
    "charset: document.characterSet || '',"
    "metaDescription: document.querySelector('meta[name=\"description\"]')?.content || '',"
    "canonicalUrl: document.querySelector('link[rel=\"canonical\"]')?.href || '',"
    "viewport: {width: window.innerWidth, height: window.innerHeight, devicePixelRatio: window.devicePixelRatio || 1},"
    "counts: {links: document.links.length, forms: document.forms.length, images: document.images.length, iframes: document.querySelectorAll('iframe').length, scripts: document.scripts.length}"
    "}))()"
)


NETWORK_SUMMARY_EXPRESSION = (
    "(() => {"
    "const resources = performance.getEntriesByType('resource').slice(-200).map((entry, index) => ({"
    "index,"
    "name: entry.name,"
    "initiatorType: entry.initiatorType || 'other',"
    "startTime: entry.startTime,"
    "duration: entry.duration,"
    "transferSize: entry.transferSize ?? null,"
    "encodedBodySize: entry.encodedBodySize ?? null,"
    "decodedBodySize: entry.decodedBodySize ?? null,"
    "nextHopProtocol: entry.nextHopProtocol || ''"
    "}));"
    "const byInitiator = {};"
    "let totalTransferSize = 0;"
    "for (const item of resources) {"
    "byInitiator[item.initiatorType] = (byInitiator[item.initiatorType] || 0) + 1;"
    "if (typeof item.transferSize === 'number') totalTransferSize += item.transferSize;"
    "}"
    "const slowest = [...resources].sort((a, b) => b.duration - a.duration).slice(0, 10);"
    "const recent = resources.slice(-15);"
    "const nav = performance.getEntriesByType('navigation')[0];"
    "return {"
    "count: resources.length,"
    "byInitiator,"
    "totalTransferSize: totalTransferSize || null,"
    "document: nav ? {type: nav.type, domComplete: nav.domComplete, loadEventEnd: nav.loadEventEnd, responseEnd: nav.responseEnd, transferSize: nav.transferSize ?? null} : null,"
    "slowest,"
    "recent"
    "};"
    "})()"
)


STORAGE_SUMMARY_EXPRESSION = (
    "(() => {"
    "const serializeStore = (store) => {"
    "try {"
    "const items = [];"
    "for (let i = 0; i < Math.min(store.length, 25); i += 1) {"
    "const key = store.key(i);"
    "items.push({key, value: String(store.getItem(key) || '').slice(0, 500)});"
    "}"
    "return {count: store.length, items};"
    "} catch (error) {"
    "return {error: String(error)};"
    "}"
    "};"
    "return {"
    "cookiesEnabled: navigator.cookieEnabled,"
    "cookiePreview: String(document.cookie || '').slice(0, 1000),"
    "localStorage: serializeStore(window.localStorage),"
    "sessionStorage: serializeStore(window.sessionStorage)"
    "};"
    "})()"
)


def _selector_results(selectors: List[str], task_id: str) -> List[Dict[str, Any]]:
    from tools.browser_tool import browser_console

    results: List[Dict[str, Any]] = []
    for selector in selectors:
        raw = _json_load(browser_console(expression=selector_expression(selector), task_id=task_id))
        item: Dict[str, Any] = {"selector": selector, "success": bool(raw.get("success"))}
        if raw.get("success"):
            item["result"] = raw.get("result", raw)
        else:
            item["error"] = raw.get("error", "Selector inspection failed")
        results.append(item)
    return results


def _expression_results(expressions: List[str], task_id: str) -> List[Dict[str, Any]]:
    from tools.browser_tool import browser_console

    results: List[Dict[str, Any]] = []
    for expression in expressions:
        raw = _json_load(browser_console(expression=expression, task_id=task_id))
        item: Dict[str, Any] = {"expression": expression, "success": bool(raw.get("success"))}
        if raw.get("success"):
            item["result"] = raw.get("result", raw)
            item["result_type"] = raw.get("result_type", type(item["result"]).__name__)
        else:
            item["error"] = raw.get("error", "Expression evaluation failed")
        results.append(item)
    return results


def collect_browser_inspect_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    from tools.browser_tool import browser_console, browser_get_images, browser_navigate, browser_snapshot

    task_id = str(payload.get("task_id") or "bundled-web-inspect")
    url = str(payload.get("url") or "").strip()
    include_console = bool(payload.get("include_console", True))
    include_network = bool(payload.get("include_network", False))
    include_images = bool(payload.get("include_images", False))
    include_storage = bool(payload.get("include_storage", False))
    clear_console = bool(payload.get("clear_console", False))
    full_snapshot = bool(payload.get("full_snapshot", False))

    selectors = normalize_selector_inputs(
        str(payload.get("selector") or ""),
        payload.get("selectors") if isinstance(payload.get("selectors"), list) else None,
    )
    expressions = normalize_expression_inputs(
        str(payload.get("expression") or ""),
        payload.get("expressions") if isinstance(payload.get("expressions"), list) else None,
    )

    response: Dict[str, Any] = {
        "success": True,
        "mode": "inspect",
        "requested_checks": {
            "selectors": len(selectors),
            "expressions": len(expressions),
            "console": include_console,
            "network": include_network,
            "images": include_images,
            "storage": include_storage,
            "full_snapshot": full_snapshot,
        },
        "development_capabilities": {
            "console": True,
            "eval": True,
            "network": True,
            "images": True,
            "storage": True,
            "multi_selector": True,
            "multi_expression": True,
            "raw_js_eval_tool": "browser_console",
        },
    }

    if url:
        nav = _json_load(browser_navigate(url, task_id=task_id))
        if not nav.get("success"):
            return nav
        response["url"] = nav.get("url", url)
        response["title"] = nav.get("title", "")

    snapshot = _json_load(browser_snapshot(full=full_snapshot, task_id=task_id))
    if not snapshot.get("success"):
        return snapshot

    response["snapshot"] = snapshot.get("snapshot", "")
    response["element_count"] = snapshot.get("element_count", 0)

    page = _json_load(browser_console(expression=PAGE_METADATA_EXPRESSION, task_id=task_id))
    response["page"] = page.get("result", page)
    if isinstance(response["page"], dict):
        response["page"].setdefault("url", response.get("url", url))
        if response.get("title"):
            response["page"].setdefault("title", response["title"])

    if selectors:
        selector_results = _selector_results(selectors, task_id)
        response["inspected_selectors"] = selector_results
        if len(selector_results) == 1 and selector_results[0].get("success"):
            response["selector_result"] = selector_results[0].get("result")

    if expressions:
        expression_results = _expression_results(expressions, task_id)
        response["evaluated_expressions"] = expression_results
        if len(expression_results) == 1 and expression_results[0].get("success"):
            response["eval_result"] = expression_results[0].get("result")

    if include_console:
        console = _json_load(browser_console(clear=clear_console, task_id=task_id))
        response["console_messages"] = console.get("console_messages", [])
        response["js_errors"] = console.get("js_errors", [])
        response["console_summary"] = {
            "message_count": console.get("total_messages", len(response["console_messages"])),
            "error_count": console.get("total_errors", len(response["js_errors"])),
        }

    if include_network:
        network = _json_load(browser_console(expression=NETWORK_SUMMARY_EXPRESSION, task_id=task_id))
        response["network"] = network.get("result", network)

    if include_storage:
        storage = _json_load(browser_console(expression=STORAGE_SUMMARY_EXPRESSION, task_id=task_id))
        response["storage"] = storage.get("result", storage)

    if include_images:
        images = _json_load(browser_get_images(task_id=task_id))
        image_items = images.get("images", []) if isinstance(images.get("images", []), list) else []
        response["images"] = {
            "count": images.get("count", len(image_items)),
            "items": image_items[:MAX_IMAGE_ITEMS],
        }
        if images.get("warning"):
            response["images"]["warning"] = images["warning"]

    return response