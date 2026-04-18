#!/usr/bin/env python3
"""Lifecycle management for the Hermes bundled local web service."""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_service_process: Optional[subprocess.Popen] = None
_service_log_handle = None


def _load_cfg() -> Dict[str, Any]:
    from tools.webplus_backend import load_bundled_web_config

    return load_bundled_web_config()


def _runtime_dir() -> Path:
    path = get_hermes_home() / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log_path() -> Path:
    path = get_hermes_home() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / "bundled-web-service.log"


def _request_json(url: str, *, method: str = "GET", payload: Optional[Dict[str, Any]] = None, timeout: float = 1.5) -> Optional[Dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {}
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return None


def bundled_web_service_health(timeout: float = 1.0) -> Optional[Dict[str, Any]]:
    cfg = _load_cfg()
    if not cfg.get("enabled") or not cfg.get("base_url"):
        return None
    return _request_json(f"{cfg['base_url']}/healthz", timeout=timeout)


def bundled_web_service_is_healthy(timeout: float = 1.0) -> bool:
    payload = bundled_web_service_health(timeout=timeout)
    return bool(payload and payload.get("ok"))


def _spawn_service_process(base_url: str) -> Optional[subprocess.Popen]:
    global _service_log_handle

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    log_handle = open(_log_path(), "a", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "tools.webplus_service",
        "--host",
        host,
        "--port",
        str(port),
    ]
    kwargs: Dict[str, Any] = {
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "close_fds": os.name != "nt",
        "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        process = subprocess.Popen(cmd, **kwargs)
    except Exception:
        log_handle.close()
        raise

    _service_log_handle = log_handle
    logger.info("Started bundled web service on %s (pid=%s)", base_url, process.pid)
    return process


def ensure_bundled_web_service() -> bool:
    """Ensure the bundled local web service is healthy.

    Returns True if the service is healthy or was started successfully.
    """
    global _service_process

    cfg = _load_cfg()
    if not cfg.get("enabled"):
        return False

    if bundled_web_service_is_healthy():
        return True

    if not cfg.get("auto_start", True):
        return False

    with _lock:
        if bundled_web_service_is_healthy():
            return True

        if _service_process is not None and _service_process.poll() is None:
            deadline = time.time() + 8
            while time.time() < deadline:
                if bundled_web_service_is_healthy():
                    return True
                time.sleep(0.2)
            return False

        try:
            _service_process = _spawn_service_process(cfg["base_url"])
        except Exception as exc:
            logger.warning("Could not start bundled web service: %s", exc)
            _service_process = None
            return False

        deadline = time.time() + 12
        while time.time() < deadline:
            if _service_process is not None and _service_process.poll() is not None:
                logger.warning("Bundled web service exited early with code %s", _service_process.returncode)
                break
            if bundled_web_service_is_healthy(timeout=1.0):
                return True
            time.sleep(0.25)

        return False


def shutdown_bundled_web_service(force: bool = False) -> None:
    """Shut down the Hermes-managed bundled web service, if we started it."""
    global _service_process, _service_log_handle

    with _lock:
        process = _service_process
        if process is None:
            return

        cfg = _load_cfg()
        base_url = cfg.get("base_url", "").rstrip("/")

        if not force and process.poll() is None and base_url:
            _request_json(f"{base_url}/shutdown", method="POST", payload={}, timeout=2.0)
            deadline = time.time() + 5
            while time.time() < deadline and process.poll() is None:
                time.sleep(0.1)

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

        if _service_log_handle is not None:
            try:
                _service_log_handle.close()
            except Exception:
                pass

        _service_log_handle = None
        _service_process = None


atexit.register(shutdown_bundled_web_service)