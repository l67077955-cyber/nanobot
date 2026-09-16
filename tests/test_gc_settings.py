"""gc_settings: cached, mtime-invalidated getters for the user-level
groupchat runtime settings file (was read raw from three call sites)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanobot.groupchat.runtime import gc_settings


@pytest.fixture()
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "groupchat_settings.json"
    monkeypatch.setattr(gc_settings, "settings_path", lambda: p)
    monkeypatch.setattr(gc_settings, "_cache", None)
    return p


def _bump_mtime(p: Path, delta: float = 10.0) -> None:
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + delta))


def test_missing_file_returns_defaults(cfg: Path) -> None:
    assert gc_settings.get("anything") is None
    assert gc_settings.stream_llm_calls() is False
    assert gc_settings.global_timeout() == 600.0


def test_roundtrip(cfg: Path) -> None:
    cfg.write_text(
        json.dumps({"stream_llm_calls": True, "global_timeout": 900}),
        encoding="utf-8",
    )
    assert gc_settings.stream_llm_calls() is True
    assert gc_settings.global_timeout() == 900.0


def test_mtime_invalidation(cfg: Path) -> None:
    cfg.write_text('{"stream_llm_calls": false}', encoding="utf-8")
    assert gc_settings.stream_llm_calls() is False
    cfg.write_text('{"stream_llm_calls": true}', encoding="utf-8")
    _bump_mtime(cfg)  # defeat coarse fs timestamps
    assert gc_settings.stream_llm_calls() is True


def test_broken_json_falls_back(cfg: Path) -> None:
    cfg.write_text("{not json", encoding="utf-8")
    assert gc_settings.stream_llm_calls() is False
    assert gc_settings.global_timeout() == 600.0


def test_global_timeout_floor_and_garbage(cfg: Path) -> None:
    cfg.write_text('{"global_timeout": 5}', encoding="utf-8")
    assert gc_settings.global_timeout() == 30.0
    cfg.write_text('{"global_timeout": "oops"}', encoding="utf-8")
    _bump_mtime(cfg)
    assert gc_settings.global_timeout() == 600.0


def test_load_dict_merges_defaults(cfg: Path) -> None:
    cfg.write_text('{"conv_keep_turns": 5}', encoding="utf-8")
    d = gc_settings.load_dict()
    assert d["conv_keep_turns"] == 5  # user override wins
    assert d["search_initial"] == 2  # untouched default survives
