"""Smoke tests for tools/camofox_tool.py.

These mock subprocess + requests so no real npm/npx/network is required.
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))


def test_status_when_nothing_installed(monkeypatch):
    from tools import camofox_tool
    monkeypatch.setattr(camofox_tool, "_is_installed", lambda: False)
    monkeypatch.setattr(camofox_tool, "_read_pid", lambda: None)
    monkeypatch.setattr(camofox_tool, "_health", lambda url, timeout=2.0: None)
    res = json.loads(camofox_tool.camofox_control(action="status"))
    assert res["success"] is True
    assert res["installed"] is False
    assert res["running"] is False
    assert res["pid"] is None
    assert "url" in res
    assert res["hint"]


def test_status_running(monkeypatch):
    from tools import camofox_tool
    monkeypatch.setattr(camofox_tool, "_is_installed", lambda: True)
    monkeypatch.setattr(camofox_tool, "_read_pid", lambda: 12345)
    monkeypatch.setattr(camofox_tool, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(camofox_tool, "_health",
                        lambda url, timeout=2.0: {"status": "ok", "vncPort": 5900})
    res = json.loads(camofox_tool.camofox_control(action="status"))
    assert res["installed"] is True
    assert res["running"] is True
    assert res["pid"] == 12345
    assert res["health"]["status"] == "ok"


def test_install_when_npm_missing(monkeypatch):
    from tools import camofox_tool
    monkeypatch.setattr(camofox_tool, "_npm", lambda: None)
    monkeypatch.setattr(camofox_tool, "_is_installed", lambda: False)
    res = json.loads(camofox_tool.camofox_control(action="install"))
    assert res["success"] is False
    assert "npm" in res["error"].lower()


def test_install_already_installed(monkeypatch):
    from tools import camofox_tool
    monkeypatch.setattr(camofox_tool, "_npm", lambda: "/usr/bin/npm")
    monkeypatch.setattr(camofox_tool, "_is_installed", lambda: True)
    res = json.loads(camofox_tool.camofox_control(action="install"))
    assert res["success"] is True
    assert res["already_installed"] is True


def test_set_url_persists(monkeypatch, tmp_path):
    from tools import camofox_tool
    monkeypatch.setattr(camofox_tool, "_health", lambda url, timeout=3.0: {"ok": True})
    res = json.loads(camofox_tool.camofox_control(action="set_url",
                                                   url="localhost:9999"))
    assert res["success"] is True
    assert res["url"] == "http://localhost:9999"
    assert os.environ["CAMOFOX_URL"] == "http://localhost:9999"
    env_path = Path(os.environ["HERMES_HOME"]) / ".env"
    assert env_path.exists()
    content = env_path.read_text(encoding="utf-8")
    assert "CAMOFOX_URL=http://localhost:9999" in content


def test_set_url_required():
    from tools import camofox_tool
    res = json.loads(camofox_tool.camofox_control(action="set_url"))
    assert res["success"] is False


def test_unknown_action():
    from tools import camofox_tool
    res = json.loads(camofox_tool.camofox_control(action="bogus"))
    assert "error" in res


def test_stop_no_pid(monkeypatch):
    from tools import camofox_tool
    monkeypatch.setattr(camofox_tool, "_read_pid", lambda: None)
    res = json.loads(camofox_tool.camofox_control(action="stop"))
    assert res["success"] is True
    assert res["was_running"] is False


def test_stop_stale_pid(monkeypatch, tmp_path):
    from tools import camofox_tool
    pid_file = camofox_tool._pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("99999")
    monkeypatch.setattr(camofox_tool, "_pid_alive", lambda pid: False)
    res = json.loads(camofox_tool.camofox_control(action="stop"))
    assert res["success"] is True
    assert res["was_running"] is False
    assert not pid_file.exists()


def test_logs_no_file():
    from tools import camofox_tool
    res = json.loads(camofox_tool.camofox_control(action="logs", lines=10))
    assert res["success"] is True
    assert res["lines"] == []


def test_logs_tail():
    from tools import camofox_tool
    log = camofox_tool._log_file()
    log.write_text("\n".join(f"line {i}" for i in range(50)), encoding="utf-8")
    res = json.loads(camofox_tool.camofox_control(action="logs", lines=5))
    assert res["success"] is True
    assert len(res["lines"]) == 5
    assert res["lines"][-1] == "line 49"


def test_registered_in_registry():
    from tools.registry import registry
    import tools.camofox_tool  # noqa: F401
    assert "camofox_control" in registry.get_all_tool_names()
    schema = registry.get_schema("camofox_control")
    assert schema is not None and schema["name"] == "camofox_control"


def test_persist_env_replaces_existing(monkeypatch):
    from tools import camofox_tool
    env_path = camofox_tool._env_file()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("OTHER=foo\nCAMOFOX_URL=http://old\nMORE=bar\n", encoding="utf-8")
    camofox_tool._persist_env_var("CAMOFOX_URL", "http://new")
    text = env_path.read_text(encoding="utf-8")
    assert "CAMOFOX_URL=http://new" in text
    assert "CAMOFOX_URL=http://old" not in text
    assert "OTHER=foo" in text
    assert "MORE=bar" in text


def test_persist_env_clear(monkeypatch):
    from tools import camofox_tool
    env_path = camofox_tool._env_file()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("CAMOFOX_URL=http://old\nKEEP=yes\n", encoding="utf-8")
    camofox_tool._persist_env_var("CAMOFOX_URL", None)
    text = env_path.read_text(encoding="utf-8")
    assert "CAMOFOX_URL" not in text
    assert "KEEP=yes" in text
