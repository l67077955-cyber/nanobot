"""Result store: offload old tool results, preview stubs inline.

Pins the 2026-09-13 context-inflation fix (216K-char agent views) and the
2026-09-16 v2 realignment to Anthropic tool-result-clearing semantics:
age-first offloading (old results cleared regardless of size, tiny results
stay verbatim), stubs carry a preview, store is a bounded LRU with disk
compaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.tools.result_store import (
    GetToolResultTool,
    ResultStore,
    compress_tool_results,
)

BIG = "x" * 8_000
MID = "m" * 1_500   # old default (6000) would NOT archive this; v2 does
SMALL = "ok"


@pytest.fixture()
def store(tmp_path: Path) -> ResultStore:
    return ResultStore(store_dir=tmp_path / "rs")


def test_put_get_roundtrip(store: ResultStore) -> None:
    idx = store.put(BIG)
    assert idx.startswith("tr-") and len(idx) == 13
    assert store.get(idx, limit=100_000) == BIG


def test_content_addressed_dedup(store: ResultStore) -> None:
    a = store.put("same content")
    b = store.put("same content")
    assert a == b
    assert store.size() == 1


def test_paging(store: ResultStore) -> None:
    idx = store.put("0123456789" * 100)  # 1000 chars
    page1 = store.get(idx, offset=0, limit=400)
    assert "0123456789" in page1 and "offset=400" in page1
    page2 = store.get(idx, offset=990, limit=400)
    assert "0123456789" in page2  # last 10 chars
    assert "offset=" not in page2  # last page: no more-pages hint


def test_missing_index_returns_error(store: ResultStore) -> None:
    out = store.get("tr-doesnotexist")
    assert out.startswith("Error: index")


def test_persistence_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "rs2"
    s1 = ResultStore(store_dir=d)
    idx = s1.put("persist me")
    s2 = ResultStore(store_dir=d)
    assert s2.get(idx, limit=100) == "persist me"


# ── v2: age-first + preview stubs ─────────────────────────────────────────


def test_compress_replaces_old_large_results(store: ResultStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanobot.tools import result_store as rs_mod
    monkeypatch.setattr(rs_mod, "get_result_store", lambda: store)
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "web_fetch", "arguments": "{}"}},
            {"id": "t2", "type": "function", "function": {"name": "exec", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "t1", "content": BIG},
        {"role": "tool", "tool_call_id": "t2", "content": BIG},
        {"role": "tool", "tool_call_id": "t3", "content": BIG},
        {"role": "tool", "tool_call_id": "t4", "content": BIG},  # recent, kept verbatim
        {"role": "tool", "tool_call_id": "t5", "content": SMALL},  # recent, kept verbatim
    ]
    out = compress_tool_results(msgs, keep_recent=3, inline_max=1_000)
    assert out[2]["content"].startswith("[已归档 tr-")
    assert out[3]["content"].startswith("[已归档 tr-")
    assert out[4]["content"] == BIG
    assert out[5]["content"] == BIG
    assert out[6]["content"] == SMALL
    # caller list untouched (copy-on-write)
    assert msgs[2]["content"] == BIG
    # archived content retrievable
    idx = out[2]["content"].split("tr-")[1].split()[0]
    assert store.get("tr-" + idx, limit=100_000) == BIG


def test_age_first_midsize_archived(store: ResultStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old results are cleared regardless of size (official semantics):
    1500 chars would pass the old 6000 threshold — v2 offloads it."""
    from nanobot.tools import result_store as rs_mod
    monkeypatch.setattr(rs_mod, "get_result_store", lambda: store)
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": MID},
        {"role": "tool", "tool_call_id": "t2", "content": MID},
    ]
    out = compress_tool_results(msgs, keep_recent=0, inline_max=1_000)
    assert out[0]["content"].startswith("[已归档 tr-")
    assert out[1]["content"].startswith("[已归档 tr-")


def test_stub_carries_preview(store: ResultStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hybrid strategy: the stub previews the head of the content so most
    retrievals never happen."""
    from nanobot.tools import result_store as rs_mod
    monkeypatch.setattr(rs_mod, "get_result_store", lambda: store)
    content = "QUERY_RESULT_DISK_USAGE_42\n" + "z" * 9_000
    msgs = [{"role": "tool", "tool_call_id": "t1", "content": content}]
    out = compress_tool_results(msgs, keep_recent=0, inline_max=1_000)
    assert "QUERY_RESULT_DISK_USAGE_42" in out[0]["content"]
    assert "\n" not in out[0]["content"]  # preview is one line
    assert len(out[0]["content"]) < 400


def test_str_prefix_not_mistaken_for_stub(store: ResultStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: 'tr-' substring detection let 'str-…' outputs slip
    through unarchived (2026-09-16 review P2). Detection is now startswith."""
    from nanobot.tools import result_store as rs_mod
    monkeypatch.setattr(rs_mod, "get_result_store", lambda: store)
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": "str-object: " + "y" * 9_000},
    ]
    out = compress_tool_results(msgs, keep_recent=0, inline_max=1_000)
    assert out[0]["content"].startswith("[已归档 tr-")


def test_compress_idempotent_on_stubs() -> None:
    stub = "[已归档 tr-abc1234567 | 原文 8,000 字符 | 预览: x | 调用 get_tool_result(index=\"tr-abc1234567\")]"
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": stub},
        {"role": "tool", "tool_call_id": "t2", "content": "x"},
    ]
    out = compress_tool_results(msgs, keep_recent=0, inline_max=10)
    assert out is msgs or out == msgs  # stub not re-processed


def test_small_results_untouched() -> None:
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": SMALL},
        {"role": "tool", "tool_call_id": "t2", "content": "also small"},
    ]
    out = compress_tool_results(msgs, keep_recent=0, inline_max=1_000)
    assert out is msgs


# ── v2: bounded LRU + disk compaction ─────────────────────────────────────


def test_lru_eviction(tmp_path: Path) -> None:
    s = ResultStore(store_dir=tmp_path / "lru", max_entries=3)
    first = s.put("first" * 10)
    s.put("second" * 10)
    s.put("third" * 10)
    s.put("fourth" * 10)  # evicts `first`
    assert s.get(first).startswith("Error: index")  # evicted → friendly error
    assert "fourth" in s.get(s.put("fourth" * 10))  # newest still alive


def test_lru_get_refreshes_recency(tmp_path: Path) -> None:
    s = ResultStore(store_dir=tmp_path / "lru2", max_entries=2)
    a = s.put("aaa")
    b = s.put("bbb")
    _ = s.get(a)          # refresh a → b becomes coldest
    s.put("ccc")      # evicts b, keeps a
    assert "aaa" in s.get(a)
    assert s.get(b).startswith("Error: index")


def test_disk_compact_under_budget(tmp_path: Path) -> None:
    s = ResultStore(store_dir=tmp_path / "cmp", max_entries=2, max_total_bytes=500)
    s.put("x" * 300)
    s.put("y" * 300)
    s.put("z" * 300)  # triggers eviction + (eventually) compaction
    f = tmp_path / "cmp" / "store.jsonl"
    if f.exists():
        # compaction rewrites to the live set — at most the budget worth of
        # content entries (2 × 300 chars) plus json overhead tolerance
        assert f.stat().st_size < 2_000


def test_eviction_survives_restart(tmp_path: Path) -> None:
    """Replay keeps the newest max_entries entries; older disk rows drop
    with the same friendly-error degradation (in-memory and replay agree)."""
    d = tmp_path / "restart"
    s1 = ResultStore(store_dir=d, max_entries=2)
    oldest = s1.put("oldest")
    s1.put("second")
    s1.put("third")  # in-memory evicts 'oldest'
    assert s1.get(oldest).startswith("Error: index")
    s2 = ResultStore(store_dir=d, max_entries=2)  # replay keeps newest 2
    assert s2.get(oldest).startswith("Error: index")
    assert "third" in s2.get(s2.put("third"))  # dedup → same index, alive


async def test_get_tool_result_tool(store: ResultStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanobot.tools import result_store as rs_mod
    monkeypatch.setattr(rs_mod, "get_result_store", lambda: store)
    tool = GetToolResultTool()
    idx = store.put("A" * 10)
    out = await tool.execute(index=idx, limit=5)
    assert "AAAAA" in out and "offset=5" in out
    err = await tool.execute(index="")
    assert err.startswith("Error")
    missing = await tool.execute(index="tr-nope")
    assert missing.startswith("Error: index")


def test_history_context_archives_oversized_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Groupchat views must not accumulate >inline_max text blobs."""
    from nanobot.groupchat.history.context import HistoryContext
    from nanobot.tools import result_store as rs_mod

    store = ResultStore(store_dir=tmp_path / "rs3")
    monkeypatch.setattr(rs_mod, "get_result_store", lambda: store)

    state = type("S", (), {"save_message": lambda *a, **k: None})()
    ctx = HistoryContext(state=state, provider=None)
    ctx.set_active_agents(["A"])
    big = "历史大块文本" + "Z" * 9_000
    ctx.add_message("Harper", big)
    stored = ctx.messages[-1]["content"]
    assert stored.startswith("[已归档 tr-") and len(stored) < 600
    view = ctx.view_for("A")
    assert len(view[-1]["content"]) < 600
    # preview inside the stub
    assert "历史大块文本" in stored
    # original retrievable
    idx = stored.split("tr-")[1].split()[0]
    assert store.get("tr-" + idx, limit=100_000) == big
    # small content passes through verbatim
    ctx.add_message("Harper", "小消息")
    assert ctx.messages[-1]["content"] == "小消息"
