"""
mcp_create_server tool — let the agent generate, install, and register a new
MCP (Model Context Protocol) server from a spec.

This tool is ENABLED BY DEFAULT so the agent can extend itself with new tools
on demand. To explicitly disable it, set one of:
  - env var ``HERMES_DISABLE_MCP_CREATE=1`` (or true/yes/on)
  - config flag ``tools.mcp_create.enabled: false`` in ~/.hermes/config.yaml

Legacy explicit-allow knobs are still honoured for back-compat:
  - env var ``HERMES_ALLOW_MCP_CREATE=0`` will disable
  - ``tools.mcp_create.enabled: false`` will disable

When disabled, the tool is registered but ``check_fn`` returns False, so the
model cannot call it.

What it does
------------
Given a name + a spec describing one or more tools, this:
  1. Writes a Python stdio MCP server to
     ``$HERMES_HOME/mcp_servers/<name>/server.py``
  2. Optionally writes a ``requirements.txt``
  3. Adds an entry to ``mcp_servers.<name>`` in ``$HERMES_HOME/config.yaml``
     so it loads on next start (and via /reload-mcp)
  4. Optionally calls ``register_mcp_servers`` to load it now

Server template
---------------
Generated servers use the official ``mcp`` Python SDK with stdio transport.
Each tool the spec defines becomes an ``@server.list_tools()`` entry plus a
handler under ``@server.call_tool()``. Handler bodies come straight from the
spec ``code`` field — the agent is responsible for writing valid Python that
returns a string.

Safety
------
- Refuses to overwrite an existing server unless ``overwrite=True``.
- Server names are sanitized to ``[a-zA-Z0-9_-]+`` and bounded to 40 chars.
- The generated file is written into the Hermes home dir only — no path
  traversal possible.
- Generated handler code is *executed by the MCP server process*, which is
  spawned as a subprocess by Hermes. It runs with the same privileges as
  Hermes itself. Set HERMES_DISABLE_MCP_CREATE=1 if you want the gate closed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hermes_constants import get_hermes_home, display_hermes_home
from tools.registry import registry, tool_error

# ─── Gate ────────────────────────────────────────────────────────────────────


def _gate_open() -> bool:
    """Return True if MCP server creation is allowed.

    Enabled by default. Honours both an explicit-disable knob
    (``HERMES_DISABLE_MCP_CREATE``) and the legacy explicit-allow knob
    (``HERMES_ALLOW_MCP_CREATE``) for back-compat.
    """
    # Explicit disable wins.
    disable_env = os.environ.get("HERMES_DISABLE_MCP_CREATE", "").strip().lower()
    if disable_env in ("1", "true", "yes", "on"):
        return False

    # Legacy explicit allow knob (still honoured both directions for back-compat).
    allow_env = os.environ.get("HERMES_ALLOW_MCP_CREATE", "").strip().lower()
    if allow_env in ("0", "false", "no", "off"):
        return False
    if allow_env in ("1", "true", "yes", "on"):
        return True

    # Config file: explicit ``enabled: false`` disables; otherwise default-on.
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        section = cfg.get("tools", {}) or {}
        sub = section.get("mcp_create", {}) or {}
        if "enabled" in sub:
            return bool(sub.get("enabled"))
    except Exception:
        pass

    return True


# ─── Helpers ─────────────────────────────────────────────────────────────────

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_\-]{0,39}$")


def _validate_name(name: str) -> Optional[str]:
    """Return None if OK, else error message."""
    if not isinstance(name, str) or not name.strip():
        return "name must be a non-empty string"
    if not _NAME_RE.match(name):
        return (
            "name must match [a-zA-Z][a-zA-Z0-9_-]{0,39} "
            "(letters/digits/underscore/dash, start with a letter, max 40 chars)"
        )
    return None


def _validate_tool_spec(idx: int, t: Any) -> Optional[str]:
    """Validate one tool entry in the spec."""
    if not isinstance(t, dict):
        return f"tools[{idx}] must be an object"
    name = t.get("name")
    if not isinstance(name, str) or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$", name):
        return f"tools[{idx}].name must be a valid Python identifier"
    if not isinstance(t.get("description", ""), str):
        return f"tools[{idx}].description must be a string"
    schema = t.get("input_schema", {})
    if not isinstance(schema, dict):
        return f"tools[{idx}].input_schema must be a JSON-schema object"
    code = t.get("code", "")
    if not isinstance(code, str):
        return f"tools[{idx}].code must be a string of Python"
    return None


def _render_server_py(name: str, description: str, tools: List[Dict[str, Any]]) -> str:
    """Build the server.py source for the new MCP server."""
    # Build tool list block
    tool_descriptors_lines = []
    for t in tools:
        tname = t["name"]
        tdesc = (t.get("description") or "").replace('"""', "'''")
        tschema = t.get("input_schema") or {"type": "object", "properties": {}}
        tool_descriptors_lines.append(
            f'        Tool(\n'
            f'            name={tname!r},\n'
            f'            description={tdesc!r},\n'
            f'            inputSchema={json.dumps(tschema, indent=12).rstrip()},\n'
            f'        ),'
        )
    tool_descriptors = "\n".join(tool_descriptors_lines) or "        # (no tools defined)"

    # Build dispatcher branches
    handler_branches = []
    for t in tools:
        tname = t["name"]
        body = textwrap.dedent(t.get("code", "") or "").strip() or 'return [TextContent(type="text", text="(no result)")]'
        # Body sits inside `try:` which is at 8 spaces, so body needs 12.
        indented = textwrap.indent(body, " " * 12)
        handler_branches.append(
            f'    if name == {tname!r}:\n'
            f'        # User-supplied handler body. `arguments` is a dict.\n'
            f'        try:\n'
            f'{indented}\n'
            f'        except Exception as exc:\n'
            f'            return [TextContent(type="text", text=f"Error in {tname}: {{exc}}")]'
        )
    handlers = "\n".join(handler_branches) or "    pass"

    src = f'''"""Auto-generated MCP server: {name}

{description or "(no description)"}

Generated by Hermes mcp_create_server. Edit freely — Hermes will not
overwrite this file unless mcp_create_server is called again with
overwrite=True for this server name.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


server = Server({name!r})


@server.list_tools()
async def _list_tools() -> list[Tool]:
    return [
{tool_descriptors}
    ]


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    arguments = arguments or {{}}
{handlers}

    return [TextContent(type="text", text=f"Unknown tool: {{name}}")]


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_run())
'''
    return src


def _write_config_entry(name: str, server_dir: Path, python_exec: Optional[str]) -> Path:
    """Add the new server under mcp_servers.<name> in config.yaml."""
    config_path = get_hermes_home() / "config.yaml"
    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                config = loaded
        except Exception:
            # Keep going with empty config; we'll overwrite cleanly
            config = {}

    servers = config.get("mcp_servers")
    if not isinstance(servers, dict):
        servers = {}
        config["mcp_servers"] = servers

    servers[name] = {
        "command": python_exec or sys.executable,
        "args": [str(server_dir / "server.py")],
        "enabled": True,
    }

    # Atomic-ish write
    tmp = config_path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    tmp.replace(config_path)
    return config_path


# ─── Tool implementation ─────────────────────────────────────────────────────


def mcp_create_server(
    name: str,
    description: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    requirements: Optional[List[str]] = None,
    overwrite: bool = False,
    register_now: bool = True,
    python_exec: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Create + register a new MCP server.

    Returns a JSON string describing the result.
    """
    # ── Gate ────────────────────────────────────────────────────────────
    if not _gate_open():
        return tool_error(
            "mcp_create_server is disabled. Unset HERMES_DISABLE_MCP_CREATE in "
            "~/.hermes/.env or set tools.mcp_create.enabled: true in "
            "~/.hermes/config.yaml to re-enable."
        )

    # ── Validate inputs ─────────────────────────────────────────────────
    err = _validate_name(name)
    if err:
        return tool_error(err)

    if tools is None or not isinstance(tools, list) or not tools:
        return tool_error("tools must be a non-empty list of tool specs")

    for i, t in enumerate(tools):
        err = _validate_tool_spec(i, t)
        if err:
            return tool_error(err)

    # ── Set up directory ────────────────────────────────────────────────
    base = get_hermes_home() / "mcp_servers" / name
    server_file = base / "server.py"

    if server_file.exists() and not overwrite:
        return tool_error(
            f"MCP server {name!r} already exists at {server_file}. "
            "Pass overwrite=True to replace it."
        )

    base.mkdir(parents=True, exist_ok=True)

    # ── Write server.py ─────────────────────────────────────────────────
    src = _render_server_py(name, description, tools)
    server_file.write_text(src, encoding="utf-8")

    # ── Optional requirements.txt ───────────────────────────────────────
    reqs_written: Optional[Path] = None
    if requirements:
        if not isinstance(requirements, list) or not all(isinstance(r, str) for r in requirements):
            return tool_error("requirements must be a list of strings")
        reqs_written = base / "requirements.txt"
        reqs_written.write_text("\n".join(requirements) + "\n", encoding="utf-8")

    # ── Update config.yaml ──────────────────────────────────────────────
    try:
        config_path = _write_config_entry(name, base, python_exec)
    except Exception as exc:
        return tool_error(f"Wrote server file but failed to update config.yaml: {exc}")

    # ── Optional: try to register immediately ───────────────────────────
    register_result: Dict[str, Any] = {"attempted": False}
    if register_now:
        register_result["attempted"] = True
        try:
            from tools.mcp_tool import register_mcp_servers, _MCP_AVAILABLE  # type: ignore

            if not _MCP_AVAILABLE:
                register_result["error"] = (
                    "MCP SDK not installed in Hermes environment — server file is "
                    "written but cannot be loaded. Install: pip install mcp"
                )
            else:
                cmd = python_exec or sys.executable
                tool_names = register_mcp_servers({
                    name: {
                        "command": cmd,
                        "args": [str(server_file)],
                        "enabled": True,
                    }
                })
                # Filter to tool names from this server (prefix `mcp_<name>_`)
                from tools.mcp_tool import sanitize_mcp_name_component  # type: ignore

                prefix = f"mcp_{sanitize_mcp_name_component(name)}_"
                this_server_tools = [t for t in tool_names if t.startswith(prefix)]
                register_result["registered_tools"] = this_server_tools
                register_result["all_mcp_tools"] = tool_names
        except Exception as exc:
            register_result["error"] = f"Registration failed: {exc}"

    home_display = display_hermes_home()
    rel = f"{home_display}/mcp_servers/{name}/server.py"
    cfg_display = f"{home_display}/config.yaml"

    response = {
        "success": True,
        "name": name,
        "server_file": rel,
        "config_path": cfg_display,
        "tools_defined": [t["name"] for t in tools],
        "requirements_file": (
            f"{home_display}/mcp_servers/{name}/requirements.txt" if reqs_written else None
        ),
        "registration": register_result,
        "next_steps": [
            (
                "Use /reload-mcp to (re)load all configured MCP servers."
                if not register_now
                else "Server should be live. Run /reload-mcp if you don't see its tools."
            ),
            f"Edit the server at {rel} to refine behavior.",
            (
                f"Install dependencies into the Hermes Python env: "
                f"pip install -r {home_display}/mcp_servers/{name}/requirements.txt"
            )
            if reqs_written
            else None,
        ],
    }
    response["next_steps"] = [s for s in response["next_steps"] if s]
    return json.dumps(response, indent=2)


# ─── Schema + registration ───────────────────────────────────────────────────


_SCHEMA = {
    "name": "mcp_create_server",
    "description": (
        "Generate, install, and register a new Model Context Protocol (MCP) server "
        "so you can give yourself a brand-new tool on demand. Enabled by default; "
        "the owner can disable with HERMES_DISABLE_MCP_CREATE=1 or "
        "tools.mcp_create.enabled: false in ~/.hermes/config.yaml. "
        "Writes a stdio Python MCP server to "
        f"{display_hermes_home()}/mcp_servers/<name>/server.py and adds an entry "
        "under mcp_servers.<name> in config.yaml. Each entry in `tools` becomes a "
        "callable tool (named `mcp_<server>_<tool>`) exposed to this and future "
        "Hermes sessions. `code` is the Python body of the handler (must `return` a "
        "string or `[TextContent(type='text', text=...)]`). Use this when the user "
        "asks for a capability you do not have, or when you find yourself running the "
        "same custom workflow repeatedly — wrap it in an MCP server so it becomes a "
        "first-class tool next time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Server name. [a-zA-Z][a-zA-Z0-9_-]{0,39}.",
            },
            "description": {
                "type": "string",
                "description": "Free-text description of what this server does (goes in the file header).",
            },
            "tools": {
                "type": "array",
                "description": "List of tools to expose. Each item: {name, description, input_schema, code}.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "input_schema": {
                            "type": "object",
                            "description": "JSON Schema object for the tool's `arguments` dict.",
                        },
                        "code": {
                            "type": "string",
                            "description": (
                                "Python statements that form the body of the handler. "
                                "`arguments` is a dict; must `return [TextContent(type='text', text=...)]` "
                                "(TextContent is already imported)."
                            ),
                        },
                    },
                    "required": ["name", "description", "code"],
                },
            },
            "requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of pip requirements to write to requirements.txt.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "If true, overwrite an existing server file. Defaults to false.",
            },
            "register_now": {
                "type": "boolean",
                "description": "If true (default), connect to the server and register its tools immediately.",
            },
            "python_exec": {
                "type": "string",
                "description": "Optional path to the python interpreter to launch the server with. Defaults to the current interpreter.",
            },
        },
        "required": ["name", "tools"],
    },
}


registry.register(
    name="mcp_create_server",
    toolset="mcp_create",
    schema=_SCHEMA,
    handler=lambda args, **kw: mcp_create_server(
        name=args.get("name", ""),
        description=args.get("description", ""),
        tools=args.get("tools"),
        requirements=args.get("requirements"),
        overwrite=bool(args.get("overwrite", False)),
        register_now=bool(args.get("register_now", True)),
        python_exec=args.get("python_exec"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_gate_open,
    requires_env=[],
)
