"""MCP servers configured under ``tools.mcpServers`` must reach GroupChatEngine.

Regression: b677a0101 (2026-04-27) removed the old AgentLoop construction and,
with it, the only ``mcp_servers=config.tools.mcp_servers`` wiring. Every place
that built GroupChatEngine afterwards (gateway, DI container, headless driver)
spelled out its own kwargs without ``mcp_servers``, so the engine's lazy MCP
connect saw an empty dict and no MCP server (memory, exa, local_rag) was ever
connected in production. ``GroupChatEngine.from_config`` is now the single
constructor for a full app config; these tests pin it and its callers.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

from nanobot.config.schema import Config, MCPServerConfig
from nanobot.groupchat.runtime.engine import GroupChatEngine

REPO = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> Config:
    cfg = Config()
    agents = tmp_path / "ws" / "agents" / "alpha" / "workspace"
    agents.mkdir(parents=True)
    (agents / "SOUL.md").write_text("你叫Alpha", encoding="utf-8")
    cfg.agents.defaults.workspace = str(tmp_path / "ws")
    cfg.groupchat.agents_dir = str(tmp_path / "ws" / "agents")
    cfg.tools.mcp_servers = {
        "mempalace": MCPServerConfig(type="streamableHttp", url="http://127.0.0.1:8765/mcp"),
    }
    return cfg


def test_from_config_passes_mcp_servers(tmp_path):
    cfg = _config(tmp_path)
    engine = GroupChatEngine.from_config(cfg, provider=Mock(), state_dir=tmp_path / "state")
    assert set(engine._mcp_servers) == {"mempalace"}
    assert engine._mcp_servers["mempalace"].url == "http://127.0.0.1:8765/mcp"
    assert engine.workspace == cfg.workspace_path
    assert engine.config is cfg.groupchat


def _engine_constructions(path: Path) -> list[str]:
    """How each GroupChatEngine is built in ``path``: 'from_config' or 'direct'."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == "GroupChatEngine":
            found.append("direct")
        elif isinstance(f, ast.Attribute) and f.attr == "from_config" and getattr(f.value, "id", "") == "GroupChatEngine":
            found.append("from_config")
    return found


def test_app_entry_points_build_engine_from_config():
    for rel in ("nanobot/cli/commands.py", "nanobot/state/gateway_wiring.py", "scripts/headless_drive.py"):
        assert _engine_constructions(REPO / rel) == ["from_config"], rel
