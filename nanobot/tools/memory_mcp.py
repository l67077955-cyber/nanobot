"""Names of the shared-memory tools.

Group chat memory is the MemPalace MCP server configured under
``tools.mcpServers.mempalace``. ``connect_mcp_servers`` registers its tools as
``mcp_<server>_<tool>``, so every MemPalace tool here is
``mcp_mempalace_mempalace_<action>``.
"""

from __future__ import annotations

MEMORY_MCP_SERVER = "mempalace"
_PREFIX = f"mcp_{MEMORY_MCP_SERVER}_"
_WRITE_ACTIONS = ("add_drawer", "update_drawer", "diary_write", "kg_add", "kg_supersede")


def memory_tool(action: str) -> str:
    """Registered tool name of a MemPalace action, e.g. ``search``."""
    return f"{_PREFIX}mempalace_{action}"


def is_memory_tool(name: str) -> bool:
    return name.startswith(_PREFIX)


def is_memory_write_tool(name: str) -> bool:
    return name in {memory_tool(a) for a in _WRITE_ACTIONS}
