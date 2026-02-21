"""
Tests for the pure helper utilities in server.py:
    _read(), _load_json(), _save_json(), _now()

These have zero dependencies on config.json and don't need isolated_server.
"""

import json
from pathlib import Path

import pytest

import server as srv


# ──────────────────────────────────────────────────────────────────────────────
# _read
# ──────────────────────────────────────────────────────────────────────────────

class TestRead:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        assert srv._read(f) == "hello world"

    def test_missing_file_returns_error_string(self, tmp_path):
        result = srv._read(tmp_path / "nonexistent.txt")
        assert result.startswith("[Error reading")

    def test_unicode_content(self, tmp_path):
        f = tmp_path / "unicode.txt"
        f.write_text("café résumé naïve", encoding="utf-8")
        assert "café" in srv._read(f)


# ──────────────────────────────────────────────────────────────────────────────
# _load_json / _save_json
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadSaveJson:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "data.json"
        original = {"key": "value", "num": 42, "list": [1, 2, 3]}
        srv._save_json(path, original)
        loaded = srv._load_json(path, {})
        assert loaded == original

    def test_load_missing_returns_default(self, tmp_path):
        result = srv._load_json(tmp_path / "missing.json", {"default": True})
        assert result == {"default": True}

    def test_load_corrupt_returns_default(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("this is not json {{{{", encoding="utf-8")
        result = srv._load_json(path, {"fallback": 1})
        assert result == {"fallback": 1}

    def test_save_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "data.json"
        srv._save_json(deep, {"nested": True})
        assert deep.exists()
        assert json.loads(deep.read_text())["nested"] is True

    def test_unicode_persisted_correctly(self, tmp_path):
        path = tmp_path / "unicode.json"
        srv._save_json(path, {"name": "Frank MacBride", "note": "café ✓"})
        raw = path.read_text(encoding="utf-8")
        assert "café" in raw   # ensure_ascii=False in _save_json
        loaded = srv._load_json(path, {})
        assert loaded["note"] == "café ✓"


# ──────────────────────────────────────────────────────────────────────────────
# _now
# ──────────────────────────────────────────────────────────────────────────────

class TestNow:
    def test_returns_string(self):
        result = srv._now()
        assert isinstance(result, str)

    def test_format_matches_expected(self):
        import re
        result = srv._now()
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result), (
            f"_now() returned unexpected format: {result}"
        )
