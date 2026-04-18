"""Tests for tools/mcp_create_tool.py."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Each test gets a fresh HERMES_HOME and the gate enabled."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_ALLOW_MCP_CREATE", "1")
    return tmp_path


@pytest.fixture
def closed_gate(tmp_path, monkeypatch):
    """HERMES_HOME set, but gate explicitly closed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ALLOW_MCP_CREATE", raising=False)
    monkeypatch.setenv("HERMES_DISABLE_MCP_CREATE", "1")
    return tmp_path


def _basic_spec():
    return [
        {
            "name": "echo",
            "description": "Echoes a message",
            "input_schema": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
            "code": (
                'msg = arguments.get("msg", "")\n'
                'return [TextContent(type="text", text=msg)]'
            ),
        }
    ]


def test_gate_blocks_when_disabled(closed_gate):
    from tools.mcp_create_tool import mcp_create_server

    res = json.loads(mcp_create_server(name="x", tools=_basic_spec(), register_now=False))
    assert "error" in res
    assert "disabled" in res["error"].lower()


def test_invalid_name_rejected(isolated_home):
    from tools.mcp_create_tool import mcp_create_server

    res = json.loads(mcp_create_server(name="1bad", tools=_basic_spec(), register_now=False))
    assert "error" in res
    assert "name" in res["error"].lower()


def test_empty_tools_rejected(isolated_home):
    from tools.mcp_create_tool import mcp_create_server

    res = json.loads(mcp_create_server(name="ok", tools=[], register_now=False))
    assert "error" in res


def test_creates_server_file_and_config(isolated_home):
    from tools.mcp_create_tool import mcp_create_server

    res = json.loads(
        mcp_create_server(name="echoer", tools=_basic_spec(), register_now=False)
    )
    assert res["success"] is True

    server_path = isolated_home / "mcp_servers" / "echoer" / "server.py"
    assert server_path.exists()

    src = server_path.read_text(encoding="utf-8")
    assert "from mcp.server import Server" in src
    assert "name='echo'" in src or 'name="echo"' in src

    # Generated server must syntax-compile
    import py_compile

    py_compile.compile(str(server_path), doraise=True)

    config_path = isolated_home / "config.yaml"
    assert config_path.exists()
    text = config_path.read_text(encoding="utf-8")
    assert "mcp_servers:" in text
    assert "echoer:" in text


def test_overwrite_protection(isolated_home):
    from tools.mcp_create_tool import mcp_create_server

    res1 = json.loads(
        mcp_create_server(name="dup", tools=_basic_spec(), register_now=False)
    )
    assert res1["success"]

    # Second call without overwrite should fail
    res2 = json.loads(
        mcp_create_server(name="dup", tools=_basic_spec(), register_now=False)
    )
    assert "error" in res2
    assert "exists" in res2["error"].lower()

    # With overwrite=True it succeeds
    res3 = json.loads(
        mcp_create_server(
            name="dup", tools=_basic_spec(), register_now=False, overwrite=True
        )
    )
    assert res3["success"]


def test_requirements_file_written(isolated_home):
    from tools.mcp_create_tool import mcp_create_server

    res = json.loads(
        mcp_create_server(
            name="hasreqs",
            tools=_basic_spec(),
            requirements=["requests>=2.0", "pyyaml"],
            register_now=False,
        )
    )
    assert res["success"]
    reqs_path = isolated_home / "mcp_servers" / "hasreqs" / "requirements.txt"
    assert reqs_path.exists()


# ─── Approval-callback override path ─────────────────────────────────────────


def test_approval_callback_deny_blocks(closed_gate, monkeypatch):
    """When the gate is closed and the approval callback returns 'deny',
    the tool must refuse and surface the user's decision."""
    from tools import mcp_create_tool as mod

    # Reset session-grant in case a prior test set it.
    monkeypatch.setattr(mod, "_session_approval_granted", False, raising=False)

    calls = []

    def cb(reason, server_name):
        calls.append((reason, server_name))
        return "deny"

    mod.set_mcp_create_approval_callback(cb)
    try:
        res = json.loads(mod.mcp_create_server(name="x", tools=_basic_spec(), register_now=False))
        assert "error" in res
        assert "declined" in res["error"].lower()
        assert calls and calls[0][1] == "x"
    finally:
        mod.set_mcp_create_approval_callback(None)


def test_approval_callback_once_allows_single_call(closed_gate, monkeypatch):
    """'once' allows the current call but does NOT grant for the next one."""
    from tools import mcp_create_tool as mod

    monkeypatch.setattr(mod, "_session_approval_granted", False, raising=False)

    responses = ["once", "deny"]

    def cb(reason, server_name):
        return responses.pop(0)

    mod.set_mcp_create_approval_callback(cb)
    try:
        res1 = json.loads(mod.mcp_create_server(name="oneoff", tools=_basic_spec(), register_now=False))
        assert res1.get("success") is True
        # Second call should re-prompt and this time deny.
        res2 = json.loads(mod.mcp_create_server(name="second", tools=_basic_spec(), register_now=False))
        assert "error" in res2
    finally:
        mod.set_mcp_create_approval_callback(None)


def test_approval_callback_session_caches_grant(closed_gate, monkeypatch):
    """'session' grants permission for all subsequent calls without re-prompting."""
    from tools import mcp_create_tool as mod

    monkeypatch.setattr(mod, "_session_approval_granted", False, raising=False)

    call_count = {"n": 0}

    def cb(reason, server_name):
        call_count["n"] += 1
        return "session"

    mod.set_mcp_create_approval_callback(cb)
    try:
        res1 = json.loads(mod.mcp_create_server(name="sess1", tools=_basic_spec(), register_now=False))
        res2 = json.loads(mod.mcp_create_server(name="sess2", tools=_basic_spec(), register_now=False))
        assert res1.get("success") is True
        assert res2.get("success") is True
        # Second call must NOT have re-prompted.
        assert call_count["n"] == 1
    finally:
        mod.set_mcp_create_approval_callback(None)
        # Reset to avoid leaking session grant to other tests in the same worker.
        monkeypatch.setattr(mod, "_session_approval_granted", False, raising=False)


def test_check_fn_true_when_callback_registered(closed_gate, monkeypatch):
    """Tool must remain available (check_fn returns True) when the gate is
    closed but an approval callback is registered, so the agent can attempt
    the call and trigger the prompt."""
    from tools import mcp_create_tool as mod

    monkeypatch.setattr(mod, "_session_approval_granted", False, raising=False)
    mod.set_mcp_create_approval_callback(lambda r, n: "deny")
    try:
        assert mod._check_fn() is True
    finally:
        mod.set_mcp_create_approval_callback(None)
    # With no callback and gate closed, check_fn must be False.
    assert mod._check_fn() is False


def test_invalid_tool_name_rejected(isolated_home):
    from tools.mcp_create_tool import mcp_create_server

    bad = [{"name": "not-a-valid-py-id", "description": "x", "input_schema": {}, "code": "return []"}]
    res = json.loads(mcp_create_server(name="ok", tools=bad, register_now=False))
    assert "error" in res


def test_multi_tool_server_compiles(isolated_home):
    from tools.mcp_create_tool import mcp_create_server

    spec = [
        {
            "name": "add",
            "description": "Add",
            "input_schema": {"type": "object"},
            "code": 'return [TextContent(type="text", text=str(arguments.get("a",0)+arguments.get("b",0)))]',
        },
        {
            "name": "sub",
            "description": "Sub",
            "input_schema": {"type": "object"},
            "code": 'return [TextContent(type="text", text=str(arguments.get("a",0)-arguments.get("b",0)))]',
        },
    ]
    res = json.loads(mcp_create_server(name="multi", tools=spec, register_now=False))
    assert res["success"]
    server_path = isolated_home / "mcp_servers" / "multi" / "server.py"
    import py_compile

    py_compile.compile(str(server_path), doraise=True)
    src = server_path.read_text(encoding="utf-8")
    assert "if name == 'add'" in src
    assert "if name == 'sub'" in src
