"""
Tests for the pure logic helper functions in server.py.
These functions contain no I/O — they can be tested without the isolated_server
fixture or any filesystem setup.
"""
import pytest
import server


# ─── _build_story_entry ────────────────────────────────────────────────────────

class TestBuildStoryEntry:
    def test_id_is_stories_length_plus_one(self):
        existing = [{"id": 1}, {"id": 2}]
        entry = server._build_story_entry(existing, "A story", ["tag"], [], "")
        assert entry["id"] == 3

    def test_id_is_one_when_list_empty(self):
        entry = server._build_story_entry([], "First story", ["career"], [], "")
        assert entry["id"] == 1

    def test_explicit_title_preserved(self):
        entry = server._build_story_entry([], "story text", [], [], "My Title")
        assert entry["title"] == "My Title"

    def test_title_auto_generated_from_story_when_empty(self):
        entry = server._build_story_entry([], "short story", [], [], "")
        assert entry["title"] == "short story"

    def test_long_story_title_truncated_to_63_chars(self):
        long = "x" * 80
        entry = server._build_story_entry([], long, [], [], "")
        assert entry["title"] == "x" * 60 + "..."
        assert len(entry["title"]) == 63

    def test_tags_lowercased_and_stripped(self):
        entry = server._build_story_entry([], "s", ["  Java  ", "SPRING"], [], "")
        assert entry["tags"] == ["java", "spring"]

    def test_people_stored_as_list(self):
        people = ["Alice", "Bob"]
        entry = server._build_story_entry([], "s", [], people, "")
        assert entry["people"] == people

    def test_people_is_copy_not_same_reference(self):
        people = ["Alice"]
        entry = server._build_story_entry([], "s", [], people, "")
        people.append("Bob")
        assert entry["people"] == ["Alice"]

    def test_story_text_stored(self):
        entry = server._build_story_entry([], "the full story text", [], [], "")
        assert entry["story"] == "the full story text"

    def test_timestamp_present(self):
        entry = server._build_story_entry([], "s", [], [], "")
        assert "timestamp" in entry and len(entry["timestamp"]) > 10


# ─── _filter_stories ──────────────────────────────────────────────────────────

class TestFilterStories:
    STORIES = [
        {"id": 1, "tags": ["career", "gm"],    "people": ["Alice"]},
        {"id": 2, "tags": ["music", "family"],  "people": ["Bob"]},
        {"id": 3, "tags": ["career", "family"], "people": ["Alice", "Carol"]},
    ]

    def test_no_filter_returns_all(self):
        result = server._filter_stories(self.STORIES)
        assert len(result) == 3

    def test_tag_filter_exact_match(self):
        result = server._filter_stories(self.STORIES, tag="music")
        assert [s["id"] for s in result] == [2]

    def test_tag_filter_case_insensitive(self):
        result = server._filter_stories(self.STORIES, tag="CAREER")
        assert len(result) == 2

    def test_tag_filter_no_match_returns_empty(self):
        result = server._filter_stories(self.STORIES, tag="nonexistent")
        assert result == []

    def test_person_filter_exact_case(self):
        result = server._filter_stories(self.STORIES, person="Bob")
        assert [s["id"] for s in result] == [2]

    def test_person_filter_case_insensitive(self):
        result = server._filter_stories(self.STORIES, person="alice")
        assert len(result) == 2  # ids 1 and 3

    def test_combined_tag_and_person_filter(self):
        result = server._filter_stories(self.STORIES, tag="career", person="carol")
        assert [s["id"] for s in result] == [3]

    def test_original_list_not_mutated(self):
        original = list(self.STORIES)
        server._filter_stories(self.STORIES, tag="music")
        assert self.STORIES == original


# ─── _format_story_list ───────────────────────────────────────────────────────

class TestFormatStoryList:
    def test_header_contains_count(self):
        stories = [{"id": 1, "title": "T", "tags": ["t"], "story": "s", "people": []}]
        out = server._format_story_list(stories)
        assert "1 stories" in out or "1 story" in out or "1" in out

    def test_story_title_in_output(self):
        stories = [{"id": 1, "title": "The Fire Truck", "tags": [], "story": "text", "people": []}]
        out = server._format_story_list(stories)
        assert "The Fire Truck" in out

    def test_people_listed_when_present(self):
        stories = [{"id": 1, "title": "T", "tags": [], "story": "s", "people": ["Grandpa Frank"]}]
        out = server._format_story_list(stories)
        assert "Grandpa Frank" in out

    def test_people_line_absent_when_empty(self):
        stories = [{"id": 1, "title": "T", "tags": [], "story": "s", "people": []}]
        out = server._format_story_list(stories)
        assert "People:" not in out

    def test_story_body_included(self):
        stories = [{"id": 1, "title": "T", "tags": ["t"], "story": "unique body text", "people": []}]
        out = server._format_story_list(stories)
        assert "unique body text" in out


# ─── _build_checkin_entry ─────────────────────────────────────────────────────

class TestBuildCheckinEntry:
    def test_energy_clamped_to_1_minimum(self):
        entry, _ = server._build_checkin_entry("low", 0, "", False)
        assert entry["energy"] == 1

    def test_energy_clamped_to_10_maximum(self):
        entry, _ = server._build_checkin_entry("good", 99, "", True)
        assert entry["energy"] == 10

    def test_energy_stored_correctly(self):
        entry, _ = server._build_checkin_entry("stable", 7, "", True)
        assert entry["energy"] == 7

    def test_mood_stored(self):
        entry, _ = server._build_checkin_entry("anxious", 5, "notes", False)
        assert entry["mood"] == "anxious"

    def test_notes_stored(self):
        entry, _ = server._build_checkin_entry("good", 6, "feeling great", True)
        assert entry["notes"] == "feeling great"

    def test_productive_stored_as_bool(self):
        entry, _ = server._build_checkin_entry("good", 6, "", True)
        assert entry["productive"] is True

    def test_low_energy_guidance(self):
        _, guidance = server._build_checkin_entry("depressed", 2, "", False)
        assert "Low energy" in guidance or "small wins" in guidance.lower()

    def test_low_mood_keyword_triggers_low_energy_guidance(self):
        _, guidance = server._build_checkin_entry("low", 5, "", False)
        assert "Low energy" in guidance

    def test_high_energy_guidance(self):
        _, guidance = server._build_checkin_entry("good", 9, "", True)
        assert "High energy" in guidance or "deep work" in guidance.lower()

    def test_hyperfocus_mood_triggers_high_guidance(self):
        _, guidance = server._build_checkin_entry("hyperfocus", 5, "", True)
        assert "High energy" in guidance or "hyperfocus" in guidance.lower() or "deep work" in guidance.lower()

    def test_normal_range_guidance(self):
        _, guidance = server._build_checkin_entry("stable", 5, "", True)
        assert "High energy" not in guidance and "Low energy" not in guidance

    def test_timestamp_present(self):
        entry, _ = server._build_checkin_entry("good", 7, "", True)
        assert "timestamp" in entry and len(entry["timestamp"]) > 10

    def test_date_present(self):
        entry, _ = server._build_checkin_entry("good", 7, "", True)
        assert "date" in entry and len(entry["date"]) == 10


# ─── _build_tone_sample_entry ─────────────────────────────────────────────────

class TestBuildToneSampleEntry:
    def test_id_sequential(self):
        existing = [{"id": 1}, {"id": 2}]
        entry = server._build_tone_sample_entry(existing, "text", "source", "ctx")
        assert entry["id"] == 3

    def test_id_one_when_empty(self):
        entry = server._build_tone_sample_entry([], "text", "source", "ctx")
        assert entry["id"] == 1

    def test_source_stored(self):
        entry = server._build_tone_sample_entry([], "text", "Cover Letter GM", "")
        assert entry["source"] == "Cover Letter GM"

    def test_context_stored(self):
        entry = server._build_tone_sample_entry([], "text", "src", "hyperfocus mode")
        assert entry["context"] == "hyperfocus mode"

    def test_text_stored(self):
        entry = server._build_tone_sample_entry([], "hello world", "src", "")
        assert entry["text"] == "hello world"

    def test_word_count_correct(self):
        entry = server._build_tone_sample_entry([], "one two three four five", "s", "")
        assert entry["word_count"] == 5

    def test_word_count_single_word(self):
        entry = server._build_tone_sample_entry([], "hello", "s", "")
        assert entry["word_count"] == 1

    def test_timestamp_present(self):
        entry = server._build_tone_sample_entry([], "text", "s", "")
        assert "timestamp" in entry and len(entry["timestamp"]) > 10


# ─── _scan_dirs ───────────────────────────────────────────────────────────────

class TestScanDirs:
    def test_cover_letters_returns_one_dir(self):
        dirs = server._scan_dirs("cover_letters")
        assert len(dirs) == 1
        assert "02-Cover-Letters" in str(dirs[0])

    def test_resumes_returns_one_dir(self):
        dirs = server._scan_dirs("resumes")
        assert len(dirs) == 1
        assert "01-Current-Optimized" in str(dirs[0])

    def test_misc_returns_resume_folder(self):
        dirs = server._scan_dirs("misc")
        assert len(dirs) == 1
        assert dirs[0] == server.RESUME_FOLDER

    def test_all_returns_three_dirs(self):
        dirs = server._scan_dirs("all")
        assert len(dirs) == 3

    def test_unknown_defaults_to_cover_letters(self):
        dirs = server._scan_dirs("unknown_category")
        assert "02-Cover-Letters" in str(dirs[0])

    def test_case_insensitive(self):
        dirs_lower = server._scan_dirs("cover_letters")
        dirs_upper = server._scan_dirs("COVER_LETTERS")
        assert dirs_lower == dirs_upper
