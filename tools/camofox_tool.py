"""camofox_control tool — let Hermes install / start / stop / inspect the
local Camofox anti-detection browser server.

Camofox (https://github.com/jo-inc/camofox-browser) is the canonical
local backend for browser_navigate / browser_search / browser_click etc.
When ``CAMOFOX_URL`` is set, ``tools/browser_camofox.py`` routes every
browser operation through it, giving you Firefox-with-fingerprint-spoofing
under your own control instead of a remote service.

This tool gives the agent full lifecycle control:

    camofox_control(action="status")     -> is it running? where? version?
    camofox_control(action="install")    -> npm install @askjo/camofox-browser
    camofox_control(action="start")      -> launch as background daemon, set CAMOFOX_URL
    camofox_control(action="stop")       -> kill the daemon
    camofox_control(action="restart")    -> stop + start
    camofox_control(action="logs", lines=200) -> tail recent server log
    camofox_control(action="set_url", url="http://...") -> point Hermes at an
                                                          existing Camofox server

Files written:
    $HERMES_HOME/camofox.pid        — PID of the running daemon
    $HERMES_HOME/logs/camofox.log   — combined stdout/stderr from the server
    $HERMES_HOME/.env               — CAMOFOX_URL persisted on start/set_url

Both ``CAMOFOX_URL`` env var and the persisted ``.env`` file are updated
together so the next Hermes process inherits the setting.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from hermes_constants import display_hermes_home, get_hermes_home
from tools.registry import registry, tool_error

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DEFAULT_PORT = 9377
NPM_PACKAGE = "@askjo/camofox-browser"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _pid_file() -> Path:
    return get_hermes_home() / "camofox.pid"


def _log_file() -> Path:
    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "camofox.log"


def _env_file() -> Path:
    return get_hermes_home() / ".env"


def _node_modules_dir() -> Path:
    return PROJECT_ROOT / "node_modules"


def _is_installed() -> bool:
    pkg_dir = _node_modules_dir() / "@askjo" / "camofox-browser"
    return pkg_dir.exists() and (pkg_dir / "package.json").exists()


# ---------------------------------------------------------------------------
# .env persistence — write CAMOFOX_URL so next Hermes process inherits it.
# ---------------------------------------------------------------------------


def _persist_env_var(key: str, value: Optional[str]) -> None:
    """Set/clear ``key`` in ``$HERMES_HOME/.env`` (KEY=value format)."""
    env_path = _env_file()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if env_path.exists():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
    out: list[str] = []
    found = False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            found = True
            if value is not None:
                out.append(f"{key}={value}")
            # else: drop the line (clear)
        else:
            out.append(ln)
    if not found and value is not None:
        out.append(f"{key}={value}")
    try:
        env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Process lifecycle
# ---------------------------------------------------------------------------


def _read_pid() -> Optional[int]:
    pid_path = _pid_file()
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return pid


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            # Windows: tasklist /FI "PID eq <pid>" returns "INFO: No tasks..." if dead
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in out.stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def _kill_pid(pid: int) -> bool:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=10)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            # Give it a moment, then SIGKILL if still alive
            for _ in range(20):
                if not _pid_alive(pid):
                    return True
                time.sleep(0.1)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def _detect_url(port: Optional[int] = None) -> str:
    cur = (os.getenv("CAMOFOX_URL") or "").rstrip("/")
    if cur:
        return cur
    return f"http://localhost:{port or DEFAULT_PORT}"


def _health(url: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text[:500]}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# npm / docker discovery
# ---------------------------------------------------------------------------


def _npm() -> Optional[str]:
    return shutil.which("npm")


def _npx() -> Optional[str]:
    return shutil.which("npx")


def _docker() -> Optional[str]:
    return shutil.which("docker")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _action_status(port: Optional[int]) -> Dict[str, Any]:
    pid = _read_pid()
    pid_alive = bool(pid and _pid_alive(pid))
    url = _detect_url(port)
    health = _health(url, timeout=2.0)
    installed = _is_installed()
    return {
        "success": True,
        "action": "status",
        "installed": installed,
        "running": bool(health),
        "pid": pid if pid_alive else None,
        "pid_file": str(_pid_file()),
        "url": url,
        "configured_url": os.getenv("CAMOFOX_URL") or None,
        "health": health,
        "log_file": str(_log_file()),
        "node_modules_dir": str(_node_modules_dir()),
        "package": NPM_PACKAGE,
        "npm": _npm(),
        "docker": _docker(),
        "hint": (
            None if health else
            ("Camofox is not responding. Try action='install' (if not installed) "
             "or action='start' to launch it.")
        ),
    }


def _action_install() -> Dict[str, Any]:
    npm = _npm()
    if not npm:
        return {
            "success": False,
            "action": "install",
            "error": (
                "npm not found in PATH. Install Node.js first (the Hermes "
                "installer normally does this) or use Docker: "
                "`docker run -d -p 9377:9377 -e CAMOFOX_PORT=9377 "
                "jo-inc/camofox-browser` and then call "
                "`camofox_control(action='set_url', url='http://localhost:9377')`."
            ),
        }
    if _is_installed():
        return {
            "success": True,
            "action": "install",
            "already_installed": True,
            "node_modules_dir": str(_node_modules_dir()),
        }
    try:
        proc = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund", NPM_PACKAGE],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "action": "install",
            "error": "npm install timed out after 10 minutes. First-run downloads ~300MB.",
        }
    ok = proc.returncode == 0 and _is_installed()
    return {
        "success": ok,
        "action": "install",
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "node_modules_dir": str(_node_modules_dir()),
        "next_step": "Call action='start' to launch the server." if ok else None,
    }


def _action_start(port: Optional[int]) -> Dict[str, Any]:
    # Already running?
    existing_pid = _read_pid()
    url = _detect_url(port)
    if existing_pid and _pid_alive(existing_pid) and _health(url, timeout=2.0):
        return {
            "success": True,
            "action": "start",
            "already_running": True,
            "pid": existing_pid,
            "url": url,
        }

    if not _is_installed():
        # Auto-install
        ins = _action_install()
        if not ins.get("success"):
            return {
                "success": False,
                "action": "start",
                "error": "Camofox is not installed and auto-install failed.",
                "install_result": ins,
            }

    npx = _npx()
    node = shutil.which("node")
    if not node:
        return {
            "success": False,
            "action": "start",
            "error": "node not found in PATH. Install Node.js first.",
        }

    # Resolve the package's actual entry point. The npm package
    # @askjo/camofox-browser has NO `bin` field, so `npx <pkg>` fails. The
    # right invocation is `node <pkgdir>/<main>` with cwd set to the package
    # dir so relative requires/files resolve.
    pkg_dir = _node_modules_dir() / "@askjo" / "camofox-browser"
    pkg_json = pkg_dir / "package.json"
    entry_script: Optional[str] = None
    bin_entry: Optional[str] = None
    try:
        meta = json.loads(pkg_json.read_text(encoding="utf-8"))
        bin_field = meta.get("bin")
        if isinstance(bin_field, str):
            bin_entry = bin_field
        elif isinstance(bin_field, dict) and bin_field:
            bin_entry = next(iter(bin_field.values()))
        entry_script = meta.get("main") or "server.js"
    except Exception:
        entry_script = "server.js"

    use_port = port or DEFAULT_PORT
    log_path = _log_file()
    env = os.environ.copy()
    env["CAMOFOX_PORT"] = str(use_port)
    env["PORT"] = str(use_port)  # some forks read PORT instead

    try:
        log_fh = open(log_path, "ab")
    except Exception as exc:
        return {"success": False, "action": "start", "error": f"Cannot open log file: {exc}"}

    # Build argv: prefer bin (executable), fall back to main script.
    if bin_entry:
        cmd = [node, str((pkg_dir / bin_entry).resolve())]
    else:
        cmd = [node, str((pkg_dir / (entry_script or "server.js")).resolve())]

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(pkg_dir),  # run FROM the package dir so its relative paths work
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": env,
    }
    if os.name == "nt":
        # Windows: detach into new process group, hide window
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        popen_kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True  # detach from this process group

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as exc:
        log_fh.close()
        return {"success": False, "action": "start", "error": f"Failed to spawn: {exc}", "cmd": cmd}

    pid = proc.pid
    try:
        _pid_file().write_text(str(pid), encoding="utf-8")
    except Exception:
        pass

    # Poll /health for up to 60s (first run downloads Camoufox ~300MB)
    new_url = f"http://localhost:{use_port}"
    health: Optional[Dict[str, Any]] = None
    deadline = time.time() + 60
    while time.time() < deadline:
        if not _pid_alive(pid):
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except Exception:
                pass
            return {
                "success": False,
                "action": "start",
                "error": "Camofox process exited before becoming healthy.",
                "pid": pid,
                "log_tail": tail,
                "log_file": str(log_path),
            }
        health = _health(new_url, timeout=2.0)
        if health:
            break
        time.sleep(1.0)

    if not health:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            pass
        return {
            "success": False,
            "action": "start",
            "error": "Camofox started but /health did not respond within 60s. It may still be downloading the Camoufox engine on first run; check logs and retry status in a minute.",
            "pid": pid,
            "url": new_url,
            "cmd": cmd,
            "log_file": str(log_path),
            "log_tail": tail,
        }

    # Persist + update in-process env
    os.environ["CAMOFOX_URL"] = new_url
    _persist_env_var("CAMOFOX_URL", new_url)

    return {
        "success": True,
        "action": "start",
        "pid": pid,
        "url": new_url,
        "health": health,
        "log_file": str(log_path),
        "note": (
            "CAMOFOX_URL is now set in this process and persisted to "
            f"{display_hermes_home()}/.env. All browser_* tools will route through "
            "Camofox from now on."
        ),
    }


def _action_stop() -> Dict[str, Any]:
    pid = _read_pid()
    if not pid:
        return {"success": True, "action": "stop", "was_running": False, "message": "No PID file."}
    if not _pid_alive(pid):
        try:
            _pid_file().unlink()
        except Exception:
            pass
        return {"success": True, "action": "stop", "was_running": False, "stale_pid": pid}
    killed = _kill_pid(pid)
    try:
        _pid_file().unlink()
    except Exception:
        pass
    return {"success": killed, "action": "stop", "killed_pid": pid}


def _action_restart(port: Optional[int]) -> Dict[str, Any]:
    stop_res = _action_stop()
    time.sleep(1.0)
    start_res = _action_start(port)
    return {"success": start_res.get("success", False), "action": "restart",
            "stop": stop_res, "start": start_res}


def _action_logs(lines: int) -> Dict[str, Any]:
    log_path = _log_file()
    if not log_path.exists():
        return {"success": True, "action": "logs", "log_file": str(log_path),
                "lines": [], "message": "Log file does not exist yet."}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return tool_error(f"Cannot read log: {exc}")  # type: ignore[return-value]
    tail = text.splitlines()[-max(1, int(lines)):]
    return {
        "success": True,
        "action": "logs",
        "log_file": str(log_path),
        "line_count": len(tail),
        "lines": tail,
    }


def _action_set_url(url: Optional[str]) -> Dict[str, Any]:
    if not url:
        return {"success": False, "action": "set_url", "error": "'url' is required."}
    url = url.rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    os.environ["CAMOFOX_URL"] = url
    _persist_env_var("CAMOFOX_URL", url)
    health = _health(url, timeout=3.0)
    return {
        "success": True,
        "action": "set_url",
        "url": url,
        "reachable": bool(health),
        "health": health,
        "persisted_to": str(_env_file()),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def camofox_control(
    action: str = "status",
    port: Optional[int] = None,
    url: Optional[str] = None,
    lines: int = 100,
    task_id: Optional[str] = None,
) -> str:
    act = (action or "status").strip().lower()
    try:
        if act == "status":
            res = _action_status(port)
        elif act == "install":
            res = _action_install()
        elif act == "start":
            res = _action_start(port)
        elif act == "stop":
            res = _action_stop()
        elif act == "restart":
            res = _action_restart(port)
        elif act == "logs":
            res = _action_logs(lines or 100)
        elif act == "set_url":
            res = _action_set_url(url)
        else:
            return tool_error(
                f"Unknown action {act!r}. Expected: status, install, start, stop, restart, logs, set_url."
            )
    except Exception as exc:
        return tool_error(f"camofox_control({act}) failed: {exc}")
    return json.dumps(res, indent=2, default=str)


_SCHEMA = {
    "name": "camofox_control",
    "description": (
        "Install, start, stop, inspect, or reconfigure the local Camofox "
        "anti-detection browser server (https://github.com/jo-inc/camofox-browser). "
        "Camofox is the canonical local backend for browser_navigate / "
        "browser_search / browser_click etc. -- when CAMOFOX_URL is set, every "
        "browser_* tool routes through it for fingerprint-spoofed Firefox "
        "automation under your own control. "
        "Actions: 'status' (running? installed? URL? health?), "
        "'install' (npm install @askjo/camofox-browser), "
        "'start' (launch as background daemon, sets+persists CAMOFOX_URL), "
        "'stop' (kill the daemon), 'restart', "
        "'logs' (tail server log), "
        "'set_url' (point Hermes at an existing Camofox server elsewhere -- "
        "e.g. on another host or in a Docker container). "
        "On 'start' the tool waits up to 60s for /health to respond (first run "
        "downloads the Camoufox engine ~300MB)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "install", "start", "stop", "restart", "logs", "set_url"],
                "description": "Lifecycle action to perform.",
            },
            "port": {
                "type": "integer",
                "description": (
                    "Port for the local Camofox server (default 9377). Used by "
                    "'start' and 'restart'."
                ),
            },
            "url": {
                "type": "string",
                "description": (
                    "Required for action='set_url'. Full URL of an existing "
                    "Camofox server, e.g. 'http://localhost:9377' or "
                    "'http://camofox.internal:9377'."
                ),
            },
            "lines": {
                "type": "integer",
                "description": "How many trailing log lines to return for action='logs' (default 100).",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="camofox_control",
    toolset="camofox",
    schema=_SCHEMA,
    handler=lambda args, **kw: camofox_control(
        action=args.get("action", "status"),
        port=args.get("port"),
        url=args.get("url"),
        lines=args.get("lines", 100),
        task_id=kw.get("task_id"),
    ),
    check_fn=lambda: True,
    requires_env=[],
)
