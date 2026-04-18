"""Smoke tests for tools/proxy_tool.py."""
import json
import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure each test starts with no proxy env vars set."""
    for var in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


def test_status_reports_no_proxy():
    from tools.proxy_tool import proxy_control
    res = json.loads(proxy_control(action="status"))
    assert res["success"] is True
    assert res["proxy_set"] is False
    assert all(v is None for v in res["vars"].values())


def test_set_applies_to_all_proxy_vars():
    from tools.proxy_tool import proxy_control
    res = json.loads(proxy_control(action="set", url="http://localhost:8089"))
    assert res["success"] is True
    assert res["proxy"] == "http://localhost:8089"
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert os.environ.get(var) == "http://localhost:8089"


def test_set_normalizes_bare_host_port():
    from tools.proxy_tool import proxy_control
    res = json.loads(proxy_control(action="set", url="localhost:8089"))
    assert res["success"] is True
    assert res["proxy"] == "http://localhost:8089"


def test_set_with_no_proxy():
    from tools.proxy_tool import proxy_control
    res = json.loads(proxy_control(
        action="set", url="http://localhost:8089",
        no_proxy="localhost,127.0.0.1",
    ))
    assert res["success"] is True
    assert os.environ.get("NO_PROXY") == "localhost,127.0.0.1"
    assert os.environ.get("no_proxy") == "localhost,127.0.0.1"


def test_set_requires_url():
    from tools.proxy_tool import proxy_control
    res = json.loads(proxy_control(action="set"))
    assert "error" in res


def test_clear_removes_all_proxy_vars():
    from tools.proxy_tool import proxy_control
    proxy_control(action="set", url="http://localhost:8089", no_proxy="x")
    res = json.loads(proxy_control(action="clear"))
    assert res["success"] is True
    for var in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "NO_PROXY", "no_proxy",
    ):
        assert os.environ.get(var) is None


def test_unknown_action_errors():
    from tools.proxy_tool import proxy_control
    res = json.loads(proxy_control(action="bogus"))
    assert "error" in res


def test_registered_in_registry():
    from tools.registry import registry
    import tools.proxy_tool  # noqa: F401  ensure import-time registration
    assert "proxy_control" in registry.get_all_tool_names()
    schema = registry.get_schema("proxy_control")
    assert schema is not None and schema["name"] == "proxy_control"
