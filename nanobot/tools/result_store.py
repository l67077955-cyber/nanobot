"""Result store — offload old tool results, keep a preview stub inline.

Context-offloading primitive (plan.md 2026-09-13 backlog #1, realigned
2026-09-16 to Anthropic's "tool result clearing" + just-in-time retrieval
semantics): a 20-50KB web_fetch/tool output pasted into shared history
inflates every agent's view (observed 216K chars → thinking models degrade
to text-protocol output and prefix-cache hit-rates collapse to 3-33%).

v2 semantics (age-first, hybrid strategy):

- tool results **outside the keep_recent window are offloaded regardless
  of size** as long as they exceed ``inline_max_chars`` — matching the
  official tool-result-clearing rule "clear old, keep recent", rather
  than a size threshold;
- tiny results (≤ ``inline_max_chars``) stay verbatim, and stubs carry a
  one-line preview (~120 chars) so agents usually don't need a second
  retrieval round-trip;
- ``get_tool_result`` supports offset/limit paging for huge payloads.

Storage: in-memory LRU (entry-count + total-bytes caps) with an
append-only jsonl on disk for restart survival. Disk drift past its
budget triggers a compact to the live set (atomic tmp+rename). Evicted
entries degrade gracefully — ``get`` returns a friendly error and the
stub left in history stays harmless.

Dedup: identical content → identical index (sha1 prefix).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.tools.base import Tool

_INDEX_PREFIX = "tr-"
_STUB_HEAD = "[已归档"  # canonical stub marker; detection uses startswith
_DEFAULT_INLINE_MAX = 1_000  # ≤ this stays verbatim (hybrid strategy)
_DEFAULT_KEEP_RECENT = 3
_DEFAULT_PAGE = 4_000
_PREVIEW_CHARS = 120
_DEFAULT_MAX_ENTRIES = 2_000
_DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024

_STUB_TEMPLATE = (
    "[已归档 {index} | 原文 {size:,} 字符 | 预览: {preview} | "
    "调用 get_tool_result(index=\"{index}\", offset=0, limit=4000) 取回原文]"
)


def _one_line(s: str, n: int) -> str:
    return " ".join(s.split())[:n]


class ResultStore:
    """Content-addressed LRU archive for offloaded tool results."""

    def __init__(
        self,
        store_dir: Path | None = None,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self._dir = store_dir or (Path.home() / ".nanobot" / "tool_results")
        self._file = self._dir / "store.jsonl"
        self._max_entries = max(1, int(max_entries))
        self._max_total_bytes = max(1, int(max_total_bytes))
        # index -> content; OrderedDict order = insertion order = LRU age
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._loaded = False

    # ── boot / persistence ────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                if self._file.exists():
                    # Stream line-by-line (never read_text a possibly huge
                    # file); file order is chronological, so on overflow the
                    # oldest entries at the front are evicted naturally.
                    with open(self._file, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            idx = rec.get("index")
                            content = rec.get("content")
                            if isinstance(idx, str) and isinstance(content, str):
                                if idx in self._cache:
                                    continue  # dedup on replay
                                self._cache[idx] = content
                                self._bytes += len(content.encode("utf-8", errors="replace"))
                                self._evict_locked(hold=idx)
            except Exception as e:  # noqa: BLE001
                logger.warning("ResultStore: load failed (starting empty): {}", e)
            self._loaded = True

    def _append_disk(self, index: str, content: str) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"index": index, "content": content, "ts": time.time()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("ResultStore: disk append failed (memory-only): {}", e)

    def _compact_locked(self) -> None:
        """Rewrite the jsonl down to the live set (atomic tmp+rename)."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._file.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for idx, content in self._cache.items():
                    f.write(
                        json.dumps(
                            {"index": idx, "content": content, "ts": time.time()},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            os.replace(tmp, self._file)
        except Exception as e:  # noqa: BLE001
            logger.warning("ResultStore: compact failed (will retry later): {}", e)

    # ── eviction ──────────────────────────────────────────────────────────

    def _evict_locked(self, *, hold: str | None = None) -> None:
        evicted = False
        while len(self._cache) > self._max_entries or self._bytes > self._max_total_bytes:
            if not self._cache:
                break
            oldest = None
            for candidate in self._cache:  # first key not being held
                if candidate != hold:
                    oldest = candidate
                    break
            if oldest is None:
                break  # only the held entry remains — nothing to evict
            content = self._cache.pop(oldest)
            self._bytes -= len(content.encode("utf-8", errors="replace"))
            evicted = True
        if evicted:
            try:
                if self._file.exists() and self._file.stat().st_size > self._max_total_bytes * 2:
                    self._compact_locked()
            except Exception as e:  # noqa: BLE001
                logger.debug("ResultStore: compact check failed: {}", e)

    # ── API ───────────────────────────────────────────────────────────────

    def put(self, content: str) -> str:
        """Archive content, return its stable index (content-addressed)."""
        self._ensure_loaded()
        digest = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:10]
        index = f"{_INDEX_PREFIX}{digest}"
        with self._lock:
            if index in self._cache:
                self._cache.move_to_end(index)  # refresh recency
                return index
            self._cache[index] = content
            self._bytes += len(content.encode("utf-8", errors="replace"))
            self._append_disk(index, content)
            self._evict_locked(hold=index)
        return index

    def has(self, index: str) -> bool:
        self._ensure_loaded()
        with self._lock:
            return index in self._cache

    def get(self, index: str, offset: int = 0, limit: int | None = None) -> str:
        """Fetch (a page of) archived content by index."""
        self._ensure_loaded()
        with self._lock:
            content = self._cache.get(index)
            if content is not None:
                self._cache.move_to_end(index)  # refresh recency
        if content is None:
            return (
                f"Error: index '{index}' not found. It may reference a result "
                f"archived before the store existed or already evicted."
            )
        total = len(content)
        limit = _DEFAULT_PAGE if limit is None else max(1, min(int(limit), 50_000))
        offset = max(0, int(offset))
        chunk = content[offset : offset + limit]
        more = (
            f" | 还有 {total - offset - len(chunk):,} 字符，继续用 offset={offset + len(chunk)} 翻页"
            if offset + len(chunk) < total
            else ""
        )
        header = f"[{index} 共 {total:,} 字符 本页 {len(chunk):,}]" if total > limit else ""
        return (header + "\n" if header else "") + chunk + (more if more else "")

    def stub_for(self, index: str, original_size: int | None = None) -> str:
        """Render the inline stub; preview is derived from stored content."""
        with self._lock:
            content = self._cache.get(index)
        preview = _one_line(content, _PREVIEW_CHARS) if content else ""
        size = original_size if original_size is not None else (len(content) if content else 0)
        return _STUB_TEMPLATE.format(index=index, size=size, preview=preview)

    def size(self) -> int:
        self._ensure_loaded()
        with self._lock:
            return len(self._cache)

    def total_bytes(self) -> int:
        self._ensure_loaded()
        with self._lock:
            return self._bytes


_store: ResultStore | None = None


def get_result_store() -> ResultStore:
    """Process-global store singleton."""
    global _store
    if _store is None:
        _store = ResultStore()
    return _store


# ── Message-side offloading ───────────────────────────────────────────────


def _settings() -> tuple[bool, int, int]:
    try:
        from nanobot.groupchat.history.history_settings import (  # noqa: PLC0415
            result_store_enabled,
            result_store_inline_max,
            result_store_keep_recent,
        )
        return (
            result_store_enabled(),
            result_store_inline_max(),
            result_store_keep_recent(),
        )
    except Exception:
        return True, _DEFAULT_INLINE_MAX, _DEFAULT_KEEP_RECENT


def compress_tool_results(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int | None = None,
    inline_max: int | None = None,
) -> list[dict[str, Any]]:
    """Offload old tool results, replacing them with preview stubs.

    Age-first (official tool-result-clearing semantics): every tool result
    outside the newest ``keep_recent`` is offloaded as soon as it exceeds
    ``inline_max`` chars — size only decides "tiny stays verbatim", not
    "old gets cleared". Returns the original list when nothing changes
    (copy-on-write otherwise).
    """
    enabled, cfg_inline, cfg_keep = _settings()
    if not enabled:
        return messages
    inline_max = cfg_inline if inline_max is None else inline_max
    keep_recent = cfg_keep if keep_recent is None else keep_recent

    tool_positions = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool" and isinstance(m.get("content"), str)
    ]
    archivable = tool_positions[:-keep_recent] if keep_recent > 0 else tool_positions
    # Detection uses the canonical stub head (startwith), NOT a substring —
    # "str-…" style prefixes must not be mistaken for archived stubs.
    to_archive = [
        i
        for i in archivable
        if len(messages[i]["content"]) > inline_max
        and not messages[i]["content"].startswith(_STUB_HEAD)
    ]
    if not to_archive:
        return messages

    store = get_result_store()
    out = list(messages)
    for i in to_archive:
        original = out[i]["content"]
        index = store.put(original)
        out[i] = {**out[i], "content": store.stub_for(index, len(original))}
    logger.debug(
        "ResultStore: offloaded {} tool result(s), {} chars → preview stubs",
        len(to_archive),
        sum(len(messages[i]["content"]) for i in to_archive),
    )
    return out


class GetToolResultTool(Tool):
    """Fetch an offloaded tool result by its index (with paging)."""

    @property
    def name(self) -> str:
        return "get_tool_result"

    @property
    def description(self) -> str:
        return (
            "按索引取回被卸载（offload）的大体积工具结果原文。历史中旧的工具输出会被替换为"
            " '[已归档 tr-xxxx | 预览: …]' 存根行；需要查看原文细节时用本工具。"
            "支持 offset/limit 分页浏览超长内容。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index": {"type": "string", "description": "归档索引，如 tr-1a2b3c4d5e"},
                "offset": {"type": "integer", "description": "起始字符偏移（默认 0）"},
                "limit": {"type": "integer", "description": "本页字符数（默认 4000，最大 50000）"},
            },
            "required": ["index"],
        }

    async def execute(self, index: str = "", offset: int = 0, limit: int | None = None, **_: Any) -> str:
        if not index or not isinstance(index, str):
            return "Error: index is required (e.g. 'tr-1a2b3c4d5e')"
        return get_result_store().get(index.strip(), offset=offset, limit=limit)
