"""Poison-message guard for the user_listener requeue path.

The 2026-09-13 anti-drop patch (94166f5f4) requeues any message whose
handling raises — correct for transient failures, but a poison message
(the same exception on every retry) live-locks the listener and starves
the queue. These tests pin the cap-and-dump decision logic added in the
2026-09-16 review (P2).
"""

from __future__ import annotations

from pathlib import Path

from nanobot.groupchat.runtime.broadcast import (
    _REQUEUE_MAX_ATTEMPTS,
    _dump_dropped_message,
    _msg_key,
    _poison_decision,
)


def test_two_requeues_then_drop() -> None:
    counts: dict[str, int] = {}
    a1 = _poison_decision("poison", counts)
    a2 = _poison_decision("poison", counts)
    a3 = _poison_decision("poison", counts)
    assert a1[0] == "requeue" and a2[0] == "requeue"
    assert a3[0] == "drop"
    assert counts == {}  # drop clears the tally


def test_distinct_messages_tracked_independently() -> None:
    counts: dict[str, int] = {}
    _poison_decision("a", counts)
    _poison_decision("b", counts)
    assert counts[_msg_key("a")] == 1
    assert counts[_msg_key("b")] == 1


def test_drop_then_fresh_start() -> None:
    counts: dict[str, int] = {}
    for _ in range(_REQUEUE_MAX_ATTEMPTS):
        _poison_decision("x", counts)
    # tally was cleared on drop → a new burst starts from scratch
    assert _poison_decision("x", counts)[0] == "requeue"


def test_success_path_clears_via_key() -> None:
    """The listener's else-branch pops by key; verify key stability."""
    counts: dict[str, int] = {}
    _, key = _poison_decision("m", counts)
    counts.pop(_msg_key("m"), None)  # mirrors listener else-branch
    assert _poison_decision("m", counts)[0] == "requeue"  # back to attempt 1


def test_dump_writes_log(tmp_path: Path) -> None:
    p = _dump_dropped_message("bad message", log_dir=tmp_path)
    assert p is not None and p.exists()
    content = p.read_text(encoding="utf-8")
    assert "bad message" in content and "dropped" in content


def test_dump_never_raises(tmp_path: Path) -> None:
    # log_dir under a regular file → mkdir fails → None, no raise
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    p = _dump_dropped_message("boom", log_dir=blocker / "sub")
    assert p is None
