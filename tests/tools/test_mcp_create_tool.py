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
    content = reqs_path.read_text(encoding="utf-8")
    assert "requests>=2.0" in content
    assert "pyyaml" in content


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
