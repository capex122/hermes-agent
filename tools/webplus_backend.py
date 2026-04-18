#!/usr/bin/env python3
"""Helpers for Hermes' bundled local web backend.

This module provides a thin adapter layer that can talk to a Hermes-managed
local web service when configured, while still offering local fallback
behaviour for search, fetch/extract, and YouTube transcripts.

It intentionally does not register tools directly. Built-in tool modules can
reuse these helpers without creating import-order or registry coupling.
"""

from __future__ import annotations

import asyncio
import html
import importlib.util
import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from tools.url_safety import is_safe_url
from tools.webplus_service_manager import ensure_bundled_web_service
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_SOURCE_ADAPTERS = {
    "general": {"label": "general web", "query_prefix": ""},
    "docs": {
        "label": "documentation",
        "query_prefix": "(site:docs.python.org OR site:developer.mozilla.org OR site:readthedocs.io OR site:learn.microsoft.com)",
    },
    "github": {"label": "GitHub", "query_prefix": "site:github.com"},
    "stackoverflow": {"label": "Stack Overflow", "query_prefix": "site:stackoverflow.com"},
    "reddit": {"label": "Reddit", "query_prefix": "site:reddit.com"},
    "wikipedia": {"label": "Wikipedia", "query_prefix": "site:wikipedia.org"},
    "arxiv": {"label": "arXiv", "query_prefix": "site:arxiv.org"},
    "hackernews": {"label": "Hacker News", "query_prefix": "site:news.ycombinator.com"},
}

_SOURCE_PROFILES = {
    "mixed": ["general", "docs", "github", "stackoverflow", "reddit", "wikipedia"],
    "code": ["github", "docs", "stackoverflow"],
    "research": ["general", "wikipedia", "arxiv", "hackernews"],
    "community": ["reddit", "stackoverflow", "hackernews"],
    "docs": ["docs", "wikipedia"],
}

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    )
}

_DEFAULT_CONFIG = {
    "enabled": False,
    "base_url": "http://127.0.0.1:8765",
    "timeout": 45,
    "fallback_to_local": True,
    "max_local_chars": 20000,
}

_RESULT_LINK_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_RESULT_SNIPPET_RE = re.compile(
    r'class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</',
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def load_bundled_web_config() -> Dict[str, Any]:
    """Return merged bundled-web configuration from config.yaml and env vars."""
    cfg = deepcopy(_DEFAULT_CONFIG)

    try:
        from hermes_cli.config import load_config

        loaded = load_config()
        web_cfg = loaded.get("web", {}) if isinstance(loaded, dict) else {}
        if isinstance(web_cfg, dict):
            bundled_cfg = web_cfg.get("bundled", {})
            if isinstance(bundled_cfg, dict):
                cfg.update(bundled_cfg)
            backend = str(web_cfg.get("backend") or "").strip().lower()
            cfg["backend"] = backend
            if backend == "bundled" and "enabled" not in bundled_cfg:
                cfg["enabled"] = True
    except Exception:
        cfg["backend"] = ""

    env_url = os.getenv("HERMES_BUNDLED_WEB_URL", "").strip()
    if env_url:
        cfg["base_url"] = env_url
        cfg["enabled"] = True

    env_timeout = os.getenv("HERMES_BUNDLED_WEB_TIMEOUT", "").strip()
    if env_timeout:
        try:
            cfg["timeout"] = max(1, int(env_timeout))
        except ValueError:
            logger.debug("Ignoring invalid HERMES_BUNDLED_WEB_TIMEOUT=%r", env_timeout)

    cfg["base_url"] = str(cfg.get("base_url") or "").rstrip("/")
    cfg["timeout"] = max(1, int(cfg.get("timeout") or _DEFAULT_CONFIG["timeout"]))
    cfg["max_local_chars"] = max(
        1000,
        int(cfg.get("max_local_chars") or _DEFAULT_CONFIG["max_local_chars"]),
    )
    cfg["enabled"] = bool(cfg.get("enabled"))
    cfg["fallback_to_local"] = bool(cfg.get("fallback_to_local", True))
    return cfg


def bundled_backend_is_available() -> bool:
    """Fast availability check for the bundled-web mode.

    This is intentionally configuration-based and does not perform network I/O.
    Tool `check_fn` calls must stay cheap.
    """
    cfg = load_bundled_web_config()
    return bool(cfg.get("enabled"))


def youtube_transcript_support_available() -> bool:
    """Return True when local YouTube transcript extraction is importable."""
    return importlib.util.find_spec("youtube_transcript_api") is not None


def _clean_html_fragment(value: str) -> str:
    """Collapse HTML into readable plain text."""
    cleaned = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", value)
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</(p|div|section|article|li|h[1-6])>", "\n", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = cleaned.replace("\r", "")
    cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_title(html_text: str) -> str:
    match = _TITLE_RE.search(html_text)
    if not match:
        return ""
    return _clean_html_fragment(match.group(1))


def _decode_duckduckgo_href(href: str) -> str:
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = urljoin("https://duckduckgo.com", href)

    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return unquote(qs["uddg"][0])
    return href


def _normalize_search_results(items: List[Dict[str, Any]], limit: int, source: str) -> Dict[str, Any]:
    web_results: List[Dict[str, Any]] = []
    for index, item in enumerate(items[:limit]):
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        web_results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "description": str(
                    item.get("description")
                    or item.get("snippet")
                    or item.get("content")
                    or ""
                ).strip(),
                "position": index + 1,
            }
        )

    return {
        "success": True,
        "data": {"web": web_results},
        "source": source,
    }


def _normalize_remote_search_payload(payload: Any, limit: int) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("web"), list):
        normalized = _normalize_search_results(payload["data"]["web"], limit, "bundled-service")
        if isinstance(payload.get("grouped_results"), dict):
            normalized["grouped_results"] = payload["grouped_results"]
        if isinstance(payload.get("adapters_used"), list):
            normalized["adapters_used"] = payload["adapters_used"]
        if payload.get("source_profile"):
            normalized["source_profile"] = str(payload["source_profile"])
        return normalized

    for key in ("results", "items", "web"):
        values = payload.get(key)
        if isinstance(values, list):
            normalized = _normalize_search_results(values, limit, "bundled-service")
            if isinstance(payload.get("grouped_results"), dict):
                normalized["grouped_results"] = payload["grouped_results"]
            if isinstance(payload.get("adapters_used"), list):
                normalized["adapters_used"] = payload["adapters_used"]
            if payload.get("source_profile"):
                normalized["source_profile"] = str(payload["source_profile"])
            return normalized

    return None


def normalize_source_adapters(
    sources: Optional[List[str]] = None,
    *,
    source_profile: str = "",
) -> List[str]:
    normalized: List[str] = []
    profile_name = str(source_profile or "").strip().lower()

    if sources:
        for source in sources:
            key = str(source or "").strip().lower()
            if key in _SOURCE_ADAPTERS and key not in normalized:
                normalized.append(key)
    elif profile_name in _SOURCE_PROFILES:
        normalized.extend(_SOURCE_PROFILES[profile_name])

    if not normalized:
        normalized.append("general")

    return normalized


def _build_search_query(
    query: str,
    *,
    site: Optional[str] = None,
    source_adapter: Optional[str] = None,
) -> str:
    parts: List[str] = []
    adapter = str(source_adapter or "").strip().lower()
    adapter_cfg = _SOURCE_ADAPTERS.get(adapter)
    if adapter_cfg and adapter_cfg.get("query_prefix"):
        parts.append(str(adapter_cfg["query_prefix"]))
    if site:
        parts.append(f"site:{site}")
    parts.append(query.strip())
    return " ".join(part for part in parts if part).strip()


def _duckduckgo_html_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    response = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers=_DEFAULT_HEADERS,
        follow_redirects=True,
        timeout=20,
    )
    response.raise_for_status()

    html_text = response.text
    items: List[Dict[str, Any]] = []
    matches = list(_RESULT_LINK_RE.finditer(html_text))
    for idx, match in enumerate(matches[: limit * 2]):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(html_text), match.end() + 1800)
        window = html_text[match.end() : next_start]
        snippet_match = _RESULT_SNIPPET_RE.search(window)
        items.append(
            {
                "title": _clean_html_fragment(match.group("title")),
                "url": _decode_duckduckgo_href(match.group("href")),
                "description": _clean_html_fragment(snippet_match.group("snippet")) if snippet_match else "",
            }
        )
    return items


async def _call_bundled_backend(path: str, payload: Dict[str, Any]) -> Optional[Any]:
    cfg = load_bundled_web_config()
    if not cfg.get("enabled") or not cfg.get("base_url"):
        return None
    ensure_bundled_web_service()

    timeout = httpx.Timeout(
        cfg["timeout"],
        connect=min(2.0, float(cfg["timeout"])),
        read=float(cfg["timeout"]),
        write=min(10.0, float(cfg["timeout"])),
        pool=min(2.0, float(cfg["timeout"])),
    )

    try:
        async with httpx.AsyncClient(
            base_url=cfg["base_url"],
            headers=_DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            response = await client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        logger.debug("Bundled web backend call failed for %s: %s", path, exc)
        return None


def _call_bundled_backend_sync(path: str, payload: Dict[str, Any]) -> Optional[Any]:
    cfg = load_bundled_web_config()
    if not cfg.get("enabled") or not cfg.get("base_url"):
        return None
    ensure_bundled_web_service()

    timeout = httpx.Timeout(
        cfg["timeout"],
        connect=min(2.0, float(cfg["timeout"])),
        read=float(cfg["timeout"]),
        write=min(10.0, float(cfg["timeout"])),
        pool=min(2.0, float(cfg["timeout"])),
    )

    try:
        with httpx.Client(
            base_url=cfg["base_url"],
            headers=_DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            response = client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        logger.debug("Bundled web backend sync call failed for %s: %s", path, exc)
        return None


def local_web_search(query: str, limit: int = 5, site: Optional[str] = None) -> Dict[str, Any]:
    """Run a free local DuckDuckGo HTML search and normalize the result."""
    return _normalize_search_results(
        _duckduckgo_html_search(_build_search_query(query, site=site), limit=limit),
        limit,
        "bundled-local",
    )


def local_source_search(
    query: str,
    *,
    limit: int = 5,
    site: Optional[str] = None,
    sources: Optional[List[str]] = None,
    source_profile: str = "",
) -> Dict[str, Any]:
    """Run a grouped multi-source local search using adapter-specific DuckDuckGo queries."""
    adapters = normalize_source_adapters(sources, source_profile=source_profile)
    grouped_results: Dict[str, List[Dict[str, Any]]] = {}
    merged_results: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    per_adapter_limit = max(2, min(limit, 10))

    for adapter in adapters:
        query_text = _build_search_query(query, site=site, source_adapter=adapter)
        adapter_items = _duckduckgo_html_search(query_text, limit=per_adapter_limit)
        adapter_results: List[Dict[str, Any]] = []
        for item in adapter_items:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            normalized_item = {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "description": str(item.get("description") or "").strip(),
                "source_adapter": adapter,
                "source_label": _SOURCE_ADAPTERS.get(adapter, {}).get("label", adapter),
            }
            adapter_results.append(normalized_item)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if len(merged_results) < limit:
                normalized_item["position"] = len(merged_results) + 1
                merged_results.append(normalized_item)
        grouped_results[adapter] = adapter_results[:per_adapter_limit]

    return {
        "success": True,
        "data": {"web": merged_results[:limit]},
        "source": "bundled-local",
        "adapters_used": adapters,
        "source_profile": str(source_profile or "").strip().lower() or "custom",
        "grouped_results": grouped_results,
    }


def bundled_web_search(
    query: str,
    limit: int = 5,
    site: Optional[str] = None,
    *,
    sources: Optional[List[str]] = None,
    source_profile: str = "",
) -> Dict[str, Any]:
    """Use the configured bundled backend when possible, else local search."""
    adapters = normalize_source_adapters(sources, source_profile=source_profile)
    payload = {
        "query": query,
        "limit": limit,
        "site": site,
        "sources": adapters,
        "source_profile": str(source_profile or "").strip().lower(),
        "output_mode": "clean",
    }
    remote = _call_bundled_backend_sync("/v1/search", payload)
    normalized = _normalize_remote_search_payload(remote, limit) if remote is not None else None
    if normalized is not None:
        return normalized
    return local_source_search(
        query,
        limit=limit,
        site=site,
        sources=adapters,
        source_profile=source_profile,
    )


def _error_result(url: str, message: str, *, blocked_by_policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = {
        "url": url,
        "title": "",
        "content": "",
        "raw_content": "",
        "error": message,
    }
    if blocked_by_policy:
        result["blocked_by_policy"] = blocked_by_policy
    return result


def _normalize_extract_entry(payload: Any, fallback_url: str) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list) and payload["results"]:
            return _normalize_extract_entry(payload["results"][0], fallback_url)
        if isinstance(payload.get("data"), dict):
            return _normalize_extract_entry(payload["data"], fallback_url)

        url = str(payload.get("url") or payload.get("source_url") or fallback_url)
        title = str(payload.get("title") or payload.get("metadata", {}).get("title") or "")
        content = str(
            payload.get("content")
            or payload.get("markdown")
            or payload.get("text")
            or payload.get("full_text")
            or ""
        )
        error = payload.get("error")
        result = {
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
        }
        if error:
            result["error"] = str(error)
        return result

    return _error_result(fallback_url, "Unsupported extract payload")


async def local_web_extract(url: str, *, max_chars: Optional[int] = None) -> Dict[str, Any]:
    """Fetch a URL locally and return a normalized extract result."""
    cfg = load_bundled_web_config()
    limit = max_chars or cfg["max_local_chars"]

    if not is_safe_url(url):
        return _error_result(url, "Blocked: URL targets a private or internal network address")

    blocked = check_website_access(url)
    if blocked:
        return _error_result(
            url,
            blocked["message"],
            blocked_by_policy={"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]},
        )

    timeout = httpx.Timeout(cfg["timeout"], connect=min(5.0, float(cfg["timeout"])))
    try:
        async with httpx.AsyncClient(
            headers=_DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:
        return _error_result(url, f"Local fetch failed: {exc}")

    final_url = str(response.url)
    final_blocked = check_website_access(final_url)
    if final_blocked:
        return _error_result(
            final_url,
            final_blocked["message"],
            blocked_by_policy={
                "host": final_blocked["host"],
                "rule": final_blocked["rule"],
                "source": final_blocked["source"],
            },
        )

    content_type = (response.headers.get("content-type") or "").lower()
    title = ""
    content = ""
    if "html" in content_type:
        title = _extract_title(response.text)
        content = _clean_html_fragment(response.text)
    elif "json" in content_type:
        try:
            parsed = response.json()
            content = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            content = response.text
    elif content_type.startswith("text/") or not content_type:
        content = response.text
    else:
        return _error_result(final_url, f"Unsupported content type: {content_type or 'unknown'}")

    content = content.strip()
    if not content:
        return _error_result(final_url, "Fetched content was empty")
    if len(content) > limit:
        content = f"{content[:limit].rstrip()}\n\n[truncated after {limit} characters]"

    return {
        "url": final_url,
        "title": title,
        "content": content,
        "raw_content": content,
    }


async def bundled_web_extract(urls: List[str], *, max_chars: Optional[int] = None) -> List[Dict[str, Any]]:
    """Extract one or more URLs using the bundled backend with local fallback."""

    async def _extract_one(url: str) -> Dict[str, Any]:
        remote = await _call_bundled_backend(
            "/v1/extract",
            {"url": url, "output_mode": "clean"},
        )
        if remote is not None:
            normalized = _normalize_extract_entry(remote, url)
            if normalized.get("content") or normalized.get("error"):
                return normalized
        return await local_web_extract(url, max_chars=max_chars)

    tasks = [_extract_one(url) for url in urls[:5]]
    return list(await asyncio.gather(*tasks))


def extract_youtube_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    for pattern in (
        r"(?:v=|youtu\.be/|shorts/|embed/|live/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ):
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return value


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def fetch_youtube_transcript_local(
    url_or_id: str,
    *,
    languages: Optional[List[str]] = None,
    include_timestamps: bool = False,
) -> Dict[str, Any]:
    """Fetch a YouTube transcript locally with youtube-transcript-api when installed."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return {
            "success": False,
            "error": "youtube-transcript-api is not installed",
            "missing_dependency": "youtube-transcript-api",
        }

    video_id = extract_youtube_video_id(url_or_id)
    api = YouTubeTranscriptApi()
    try:
        raw = api.fetch(video_id, languages=languages or None)
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "video_id": video_id,
        }

    segments: List[Dict[str, Any]] = []
    for item in raw:
        if hasattr(item, "text"):
            segments.append(
                {
                    "text": item.text,
                    "start": float(getattr(item, "start", 0.0)),
                    "duration": float(getattr(item, "duration", 0.0)),
                }
            )
        elif isinstance(item, dict):
            segments.append(
                {
                    "text": str(item.get("text") or ""),
                    "start": float(item.get("start") or 0.0),
                    "duration": float(item.get("duration") or 0.0),
                }
            )

    full_text = " ".join(segment["text"] for segment in segments if segment.get("text")).strip()
    result = {
        "success": True,
        "video_id": video_id,
        "segment_count": len(segments),
        "full_text": full_text,
        "segments": segments,
        "source": "bundled-local",
    }
    if include_timestamps:
        result["timestamped_text"] = "\n".join(
            f"{_format_timestamp(segment['start'])} {segment['text']}".strip()
            for segment in segments
            if segment.get("text")
        )
    return result