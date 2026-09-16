"""Cached getters for the user-level groupchat runtime settings file.

``~/.nanobot/groupchat_settings.json`` was previously read raw from three
call sites (broadcast.py, run_loop.py, tool_loop.py) — each with its own
path literal, its own JSON parse, and no caching (a disk read per round,
or worse, per tool-loop iteration). Centralise: one path, one cached
parse (mtime-invalidated), one getter per key. Settings changes take
effect without a gateway restart.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_cache: tuple[float, dict[str, Any]] | None = None  # (mtime, data)

# Keys with production defaults; mirrors the merge dict broadcast.py used.
_DEFAULTS: dict[str, Any] = {
    "search_initial": 2,
    "search_earn_interval": 4,
    "allocate_timeout": 15,
    "call_timeout": 90,
    "conv_keep_turns": 3,
    "global_timeout": 600,
    "stream_llm_calls": False,
}


def settings_path() -> Path:
    return Path.home() / ".nanobot" / "groupchat_settings.json"


def _load() -> dict[str, Any]:
    """Parsed settings dict; empty dict when the file is absent/broken."""
    global _cache
    with _LOCK:
        try:
            mtime = settings_path().stat().st_mtime
        except OSError:
            _cache = None
            return {}
        if _cache is not None and _cache[0] == mtime:
            return _cache[1]
        try:
            data = json.loads(settings_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        _cache = (mtime, data)
        return data


def load_dict() -> dict[str, Any]:
    """Full settings dict merged over defaults (ctx-style consumers)."""
    merged = dict(_DEFAULTS)
    merged.update(_load())
    return merged


def get(key: str, default: Any = None) -> Any:
    return _load().get(key, default)


def stream_llm_calls() -> bool:
    """Pure-metrics streaming switch (real TTFT in request_logs; 045bd3b2c)."""
    return bool(get("stream_llm_calls", False))


def global_timeout(default: float = 600.0) -> float:
    """Global round timeout in seconds, floored at 30s."""
    try:
        return max(30.0, float(get("global_timeout", default)))
    except (TypeError, ValueError):
        return default
