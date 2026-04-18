#!/usr/bin/env python3
"""Local bundled web service for Hermes-managed search/fetch/research flows."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from tools.webplus_inspect import collect_browser_inspect_data
from tools.webplus_backend import (
    fetch_youtube_transcript_local,
    local_source_search,
    local_web_extract,
    local_web_search,
    youtube_transcript_support_available,
)

logger = logging.getLogger(__name__)

_SERVER_START_TIME = time.time()
_HTTPD: Optional[ThreadingHTTPServer] = None


def _browser_available() -> bool:
    from tools.browser_tool import check_browser_requirements

    return check_browser_requirements()


def _inspect_with_browser(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _browser_available():
        return {"success": False, "error": "Browser backend is not available"}

    return collect_browser_inspect_data(payload)


async def _deep_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    top_k = max(1, min(int(payload.get("top_k", 5) or 5), 10))
    extract_top = max(1, min(int(payload.get("extract_top", 3) or 3), 5))
    site = str(payload.get("site") or "").strip() or None
    sources = payload.get("sources") if isinstance(payload.get("sources"), list) else None
    source_profile = str(payload.get("source_profile") or "").strip()

    search = local_source_search(
        query,
        limit=top_k,
        site=site,
        sources=sources,
        source_profile=source_profile,
    )
    urls = []
    for item in search.get("data", {}).get("web", []):
        url = str(item.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)
        if len(urls) >= extract_top:
            break

    extracted = []
    if urls:
        extracted = list(await asyncio.gather(*(local_web_extract(url) for url in urls)))

    return {
        "success": True,
        "mode": "bundled-service-deep-search",
        "query": query,
        "site": site or "",
        "adapters_used": search.get("adapters_used", []),
        "source_profile": search.get("source_profile", "custom"),
        "grouped_results": search.get("grouped_results", {}),
        "search_results": search.get("data", {}).get("web", []),
        "extracted_pages": extracted,
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "HermesBundledWeb/0.1"

    def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("webplus-service: " + format, *args)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/healthz":
            self._json_response(
                200,
                {
                    "ok": True,
                    "service": "bundled-web",
                    "uptime_seconds": max(0, int(time.time() - _SERVER_START_TIME)),
                    "capabilities": {
                        "search": True,
                        "extract": True,
                        "deep_search": True,
                        "youtube_search": True,
                        "youtube_transcript": youtube_transcript_support_available(),
                        "inspect": _browser_available(),
                    },
                },
            )
            return

        self._json_response(404, {"success": False, "error": "Not found"})

    def do_POST(self) -> None:
        payload = self._read_json_body()
        path = self.path.rstrip("/")

        if path == "/shutdown":
            self._json_response(200, {"success": True, "message": "Shutting down"})

            def _shutdown() -> None:
                if _HTTPD is not None:
                    _HTTPD.shutdown()

            threading.Thread(target=_shutdown, daemon=True).start()
            return

        if path == "/v1/search":
            sources = payload.get("sources") if isinstance(payload.get("sources"), list) else None
            source_profile = str(payload.get("source_profile") or "").strip()
            result = local_source_search(
                str(payload.get("query") or ""),
                limit=max(1, min(int(payload.get("limit", 5) or 5), 10)),
                site=str(payload.get("site") or "").strip() or None,
                sources=sources,
                source_profile=source_profile,
            )
            self._json_response(200, result)
            return

        if path == "/v1/extract":
            url = str(payload.get("url") or "").strip()
            if not url:
                self._json_response(400, {"success": False, "error": "Missing url"})
                return
            result = asyncio.run(local_web_extract(url))
            self._json_response(200, result)
            return

        if path == "/v1/research/deep":
            result = asyncio.run(_deep_search(payload))
            self._json_response(200, result)
            return

        if path == "/v1/youtube/search":
            result = local_web_search(
                str(payload.get("query") or ""),
                limit=max(1, min(int(payload.get("limit", 5) or 5), 10)),
                site="youtube.com/watch",
            )
            result["source"] = "bundled-service"
            self._json_response(200, result)
            return

        if path == "/v1/youtube/transcript":
            languages = payload.get("languages")
            if isinstance(languages, str):
                languages = [part.strip() for part in languages.split(",") if part.strip()]
            result = fetch_youtube_transcript_local(
                str(payload.get("url") or ""),
                languages=languages if isinstance(languages, list) else None,
                include_timestamps=bool(payload.get("include_timestamps", False)),
            )
            if result.get("success"):
                result["source"] = "bundled-service"
            self._json_response(200, result)
            return

        if path == "/v1/inspect":
            result = _inspect_with_browser(payload)
            self._json_response(200, result)
            return

        self._json_response(404, {"success": False, "error": "Not found"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes bundled local web service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    global _HTTPD
    _HTTPD = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info("Bundled web service listening on http://%s:%s", args.host, args.port)
    try:
        _HTTPD.serve_forever()
    finally:
        _HTTPD.server_close()


if __name__ == "__main__":
    main()