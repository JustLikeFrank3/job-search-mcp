"""
Tests for log_personal_story() and get_personal_context().

These tools are the v3 personal story library — they accumulate context
that enriches cover letters, STAR answers, and behavioral interview responses.
All tests use the isolated_server fixture to work against a clean tmp directory.
"""

import json
from pathlib import Path

import pytest

import server as srv


# ──────────────────────────────────────────────────────────────────────────────
# log_personal_story
# ──────────────────────────────────────────────────────────────────────────────

class TestLogPersonalStory:
    def test_story_is_persisted(self, isolated_server):
        srv.log_personal_story(
            story="Grandfather stopped workers from torching 1934 Ford fire truck brass threads.",
            tags=["ford", "quality", "grandfather"],
            title="Ford fire truck threads",
        )
        data = json.loads(srv.PERSONAL_CONTEXT_FILE.read_text())
        assert len(data["stories"]) == 1
        assert data["stories"][0]["title"] == "Ford fire truck threads"

    def test_return_value_includes_id_and_title(self, isolated_server):
        result = srv.log_personal_story(
            story="Test story.",
            tags=["test"],
        )
        assert "#1" in result
        assert "✓" in result

    def test_ids_are_sequential(self, isolated_server):
        srv.log_personal_story(story="Story A", tags=["a"])
        srv.log_personal_story(story="Story B", tags=["b"])
        srv.log_personal_story(story="Story C", tags=["c"])
        data = json.loads(srv.PERSONAL_CONTEXT_FILE.read_text())
        ids = [s["id"] for s in data["stories"]]
        assert ids == [1, 2, 3]

    def test_tags_normalised_to_lowercase(self, isolated_server):
        srv.log_personal_story(story="Story.", tags=["Ford", "QUALITY", "Testing"])
        data = json.loads(srv.PERSONAL_CONTEXT_FILE.read_text())
        tags = data["stories"][0]["tags"]
        assert tags == ["ford", "quality", "testing"]

    def test_auto_title_truncated_to_60_chars(self, isolated_server):
        long_story = "x" * 120
        srv.log_personal_story(story=long_story, tags=["generic"])
        data = json.loads(srv.PERSONAL_CONTEXT_FILE.read_text())
        title = data["stories"][0]["title"]
        assert title.endswith("...")
        assert len(title) <= 63  # 60 chars + "..."

    def test_people_field_stored(self, isolated_server):
        srv.log_personal_story(
            story="Sean Evans introduced me to Less Than Jake's Just Like Frank.",
            tags=["music", "friendship"],
            people=["Sean Evans"],
        )
        data = json.loads(srv.PERSONAL_CONTEXT_FILE.read_text())
        assert "Sean Evans" in data["stories"][0]["people"]

    def test_appends_without_overwriting(self, isolated_server):
        for i in range(5):
            srv.log_personal_story(story=f"Story {i}", tags=["generic"])
        data = json.loads(srv.PERSONAL_CONTEXT_FILE.read_text())
        assert len(data["stories"]) == 5


# ──────────────────────────────────────────────────────────────────────────────
# get_personal_context
# ──────────────────────────────────────────────────────────────────────────────

class TestGetPersonalContext:
    def _seed(self):
        """Seed a known set of stories for retrieval tests."""
        srv.log_personal_story(story="Grandfather at Ford.", tags=["ford", "family"], people=["Grandpa"])
        srv.log_personal_story(story="Sean Evans and LTJ.", tags=["music", "friendship"], people=["Sean Evans"])
        srv.log_personal_story(story="Fire truck threads.", tags=["ford", "quality", "testing"])

    def test_returns_all_stories_when_no_filter(self, isolated_server):
        self._seed()
        result = srv.get_personal_context()
        assert "Grandfather at Ford" in result
        assert "Sean Evans" in result
        assert "Fire truck threads" in result

    def test_tag_filter_returns_only_matching(self, isolated_server):
        self._seed()
        result = srv.get_personal_context(tag="music")
        assert "Sean Evans" in result
        assert "Grandfather" not in result

    def test_tag_filter_case_insensitive(self, isolated_server):
        self._seed()
        result = srv.get_personal_context(tag="FORD")
        assert "Grandfather at Ford" in result

    def test_person_filter(self, isolated_server):
        self._seed()
        result = srv.get_personal_context(person="Sean Evans")
        assert "Sean Evans" in result
        assert "Grandfather at Ford" not in result

    def test_combined_tag_and_person_filter(self, isolated_server):
        self._seed()
        result = srv.get_personal_context(tag="ford", person="Grandpa")
        assert "Grandfather at Ford" in result
        # fire truck story has ford tag but no person — should not appear
        assert "Fire truck threads" not in result

    def test_no_stories_returns_empty_message(self, isolated_server):
        result = srv.get_personal_context()
        assert "No personal stories" in result

    def test_no_match_on_tag_returns_empty_message(self, isolated_server):
        self._seed()
        result = srv.get_personal_context(tag="nonexistent_tag_xyz")
        assert "No personal stories" in result

    def test_output_includes_story_numbers(self, isolated_server):
        self._seed()
        result = srv.get_personal_context()
        assert "#1" in result
        assert "#2" in result
        assert "#3" in result
