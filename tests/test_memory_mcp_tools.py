"""Group chat memory goes through the MemPalace MCP server, not an in-process tool.

The old built-in ``memory_palace`` tool wrapped the mempalace library inside the
gateway (and silently failed wherever the library was not installed, e.g. the
Singapore gateway). Memory is now the ``mempalace`` MCP server's tools, named
``mcp_mempalace_mempalace_<tool>`` by ``connect_mcp_servers``. These tests pin:

- which tool names count as memory tools (and which of them write),
- that every broadcast agent always gets the memory tools, whatever its
  per-agent tool toggles, and no longer gets a ``memory_palace`` tool,
- the display brief and the delivery gate for the MCP memory tools.
"""

from __future__ import annotations

import asyncio
from typing import Any

from nanobot.groupchat.display.display import tool_result_brief
from nanobot.groupchat.history.component_manager import synthesis_quality_check
from nanobot.groupchat.runtime.broadcast import BroadcastOrchestrator, always_on_tool_names
from nanobot.tools.base import Tool
from nanobot.tools.memory_mcp import is_memory_tool, is_memory_write_tool, memory_tool
from nanobot.tools.registry import ToolRegistry

SEARCH = "mcp_mempalace_mempalace_search"
ADD = "mcp_mempalace_mempalace_add_drawer"


class _NamedTool(Tool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_memory_tool_names():
    assert memory_tool("search") == SEARCH
    assert is_memory_tool(SEARCH) and is_memory_tool(ADD)
    assert not is_memory_tool("memory_palace")
    assert not is_memory_tool("mcp_exa_web_search_exa")
    assert is_memory_write_tool(ADD)
    assert is_memory_write_tool(memory_tool("diary_write"))
    assert not is_memory_write_tool(SEARCH)


def test_always_on_tools_include_memory_tools_for_every_agent():
    reg = ToolRegistry()
    for name in ("chatroom_send", "wait", "web_search", SEARCH, ADD, "end_discussion"):
        reg.register(_NamedTool(name))

    member = always_on_tool_names(reg, is_leader=False)
    assert member[:2] == ["chatroom_send", "wait"]
    assert SEARCH in member and ADD in member
    assert "web_search" not in member  # still governed by per-agent toggles
    assert "end_discussion" not in member
    assert "memory_palace" not in member

    leader = always_on_tool_names(reg, is_leader=True)
    assert SEARCH in leader and "end_discussion" in leader


def test_always_on_tools_without_memory_server():
    reg = ToolRegistry()
    for name in ("chatroom_send", "wait"):
        reg.register(_NamedTool(name))
    assert always_on_tool_names(reg, is_leader=False) == ["chatroom_send", "wait"]


class _FakeMailbox:
    def set_pool(self, pool: Any) -> None:
        self.pool = pool


class _FakeEngine:
    def __init__(self, agents: list[str], base: ToolRegistry):
        self.registry = {a: {"rank": "pawn"} for a in agents}
        self._base = base
        self.sent: list[str] = []

    async def _send(self, text: str) -> None:
        self.sent.append(text)

    def _get_agent_registry(self, name: str) -> ToolRegistry:
        return self._base


def test_round_registries_carry_mcp_memory_tools_not_builtin():
    base = ToolRegistry()
    for name in ("web_search", SEARCH, ADD):
        base.register(_NamedTool(name))
    engine = _FakeEngine(["alice", "bob"], base)
    orch = BroadcastOrchestrator(["alice", "bob"], engine, _FakeMailbox())

    asyncio.run(orch.setup_tools_and_pools(spawn_fn=lambda *_: None))

    for agent in ("alice", "bob"):
        names = set(orch.agent_tool_registries[agent].tool_names)
        assert {SEARCH, ADD} <= names
        assert "memory_palace" not in names
        assert {"chatroom_send", "wait"} <= names


def test_display_brief_for_memory_tools():
    assert tool_result_brief("alice", ADD, '{"success": true, "drawer_id": "d1"}') == "    └ ✅ stored"
    assert tool_result_brief("alice", SEARCH, '{"query": "x", "results": [{"text": "y"}]}') == "    └ 🔍 found"


def test_delivery_gate_on_memory_write_without_inline_data():
    thin = "## 结论\n已存储到记忆。"
    ok, reason = synthesis_quality_check(thin, tools_used=[ADD])
    assert not ok and "记忆" in reason
    # Reading memory is not "storing data the user cannot see".
    assert synthesis_quality_check(thin, tools_used=[SEARCH])[0]
    assert synthesis_quality_check(thin, tools_used=["chatroom_send"])[0]
