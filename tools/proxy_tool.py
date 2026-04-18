"""proxy_control tool — let the agent flip HTTP/HTTPS proxy env vars at
runtime so subsequent in-process httpx / urllib calls route through the new
proxy. Designed to pair with an upstream rotator like mubeng running on
localhost (https://github.com/mubeng/mubeng).

The tool only mutates ``os.environ`` of the running process. To actually
spin mubeng up the agent should use the ``terminal`` tool (mubeng install +
launch as a background process with ``background=true``).

Why this works
--------------
All of Hermes' free HTTP fallbacks (DDG-Lite, Wikipedia, the http-only
navigate path, webplus DDG HTML scrape) and the official ``web_search`` /
``web_extract`` tools use ``httpx`` with default ``trust_env=True`` and
construct a fresh client per call. Setting ``HTTPS_PROXY`` /
``HTTP_PROXY`` therefore takes effect on the very next call without any
restart.

Limitations
-----------
* Affects only the parent Python process. Subprocesses launched via
  ``terminal`` after the change will inherit the new env vars; running
  subprocesses do NOT. This is by design — the agent should not silently
  poison long-running user shells.
* Does NOT change the proxy used by the Chromium ``agent-browser`` CLI;
  for that, set ``HERMES_BROWSER_PROXY`` before launching Hermes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error

# Env vars we manage in lock-step. Both upper and lowercase to cover libraries
# that read either convention.
_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _current_state() -> Dict[str, Optional[str]]:
    return {var: os.environ.get(var) for var in _PROXY_ENV_VARS}


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        # Bare host:port → assume http
        url = f"http://{url}"
    return url


def proxy_control(
    action: str = "status",
    url: Optional[str] = None,
    no_proxy: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Inspect or mutate the in-process HTTP/HTTPS proxy env vars.

    Args:
        action: one of ``status``, ``set``, ``clear``.
        url: required when ``action == "set"``. Examples:
             ``http://localhost:8089``, ``socks5://127.0.0.1:1080``.
        no_proxy: optional value for ``NO_PROXY`` (comma-separated hosts).
    """
    act = (action or "status").strip().lower()

    if act == "status":
        state = _current_state()
        active = {k: v for k, v in state.items() if v}
        return json.dumps(
            {
                "success": True,
                "action": "status",
                "proxy_set": bool(active),
                "vars": state,
                "no_proxy": os.environ.get("NO_PROXY") or os.environ.get("no_proxy"),
            },
            indent=2,
        )

    if act == "clear":
        cleared = []
        for var in _PROXY_ENV_VARS:
            if os.environ.pop(var, None) is not None:
                cleared.append(var)
        for var in ("NO_PROXY", "no_proxy"):
            os.environ.pop(var, None)
        return json.dumps(
            {
                "success": True,
                "action": "clear",
                "cleared_vars": cleared,
                "note": (
                    "In-process proxy env vars unset. Subsequent httpx / "
                    "urllib calls will use a direct connection again."
                ),
            },
            indent=2,
        )

    if act == "set":
        normalized = _normalize_url(url or "")
        if not normalized:
            return tool_error("'url' is required for action='set' (e.g. http://localhost:8089).")
        for var in _PROXY_ENV_VARS:
            os.environ[var] = normalized
        if no_proxy is not None:
            os.environ["NO_PROXY"] = no_proxy
            os.environ["no_proxy"] = no_proxy
        return json.dumps(
            {
                "success": True,
                "action": "set",
                "proxy": normalized,
                "no_proxy": os.environ.get("NO_PROXY"),
                "applies_to": [
                    "web_search / web_extract (httpx)",
                    "browser_search free fallbacks (DDG-Lite, Wikipedia)",
                    "_http_only_browser_navigate (plain HTTPS recovery)",
                    "webplus DDG HTML scrape",
                    "any other in-process httpx / urllib HTTP call",
                ],
                "does_not_affect": [
                    "Chromium agent-browser CLI (set HERMES_BROWSER_PROXY before launch)",
                    "subprocesses already running",
                ],
            },
            indent=2,
        )

    return tool_error(f"Unknown action {act!r}. Expected one of: status, set, clear.")


_SCHEMA = {
    "name": "proxy_control",
    "description": (
        "Set, clear, or inspect the HTTP/HTTPS proxy used by Hermes' in-process "
        "HTTP calls (web_search, web_extract, all browser_search free fallbacks, "
        "and the http-only navigate recovery path). The new proxy takes effect "
        "on the very next HTTP call -- no restart needed. Pair this with an "
        "upstream rotator like mubeng "
        "(https://github.com/mubeng/mubeng) when search engines start blocking "
        "your IP: install mubeng via `terminal`, write proxies.txt, launch it "
        "as a background process with `terminal(background=true)`, then call "
        "`proxy_control(action='set', url='http://localhost:8089')`. Use "
        "`action='clear'` when done. NOTE: does not affect the Chromium "
        "agent-browser CLI -- for that, set HERMES_BROWSER_PROXY before "
        "launching Hermes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "set", "clear"],
                "description": (
                    "status -> report current proxy env vars; set -> apply a "
                    "new proxy URL to all in-process HTTP calls; clear -> "
                    "remove the proxy and go back to direct connection."
                ),
            },
            "url": {
                "type": "string",
                "description": (
                    "Proxy URL. Required for action='set'. Examples: "
                    "'http://localhost:8089', 'http://user:pass@host:port', "
                    "'socks5://127.0.0.1:1080'. Bare 'host:port' is auto-prefixed with http://."
                ),
            },
            "no_proxy": {
                "type": "string",
                "description": (
                    "Optional comma-separated list of hosts to bypass the proxy "
                    "for. Sets NO_PROXY env var (e.g. 'localhost,127.0.0.1,.internal')."
                ),
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="proxy_control",
    toolset="proxy",
    schema=_SCHEMA,
    handler=lambda args, **kw: proxy_control(
        action=args.get("action", "status"),
        url=args.get("url"),
        no_proxy=args.get("no_proxy"),
        task_id=kw.get("task_id"),
    ),
    check_fn=lambda: True,
    requires_env=[],
)
