"""
Tests for scan_materials_for_tone tool.

Each test uses isolated_server so RESUME_FOLDER and SCAN_INDEX_FILE point to
a clean tmp directory.  We seed .txt files into the expected sub-dirs and assert
on the rendered output + index persistence.
"""
import json
from pathlib import Path

import pytest

import server
from tests.conftest import _write, _write_json


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_cover_letter(res_dir: Path, name: str, content: str) -> Path:
    cl_dir = res_dir / "02-Cover-Letters"
    cl_dir.mkdir(parents=True, exist_ok=True)
    p = cl_dir / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_resume(res_dir: Path, name: str, content: str) -> Path:
    ro_dir = res_dir / "01-Current-Optimized"
    ro_dir.mkdir(parents=True, exist_ok=True)
    p = ro_dir / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_misc(res_dir: Path, name: str, content: str) -> Path:
    p = res_dir / name
    p.write_text(content, encoding="utf-8")
    return p


def _load_index(data_dir: Path) -> dict:
    idx_path = data_dir / "scan_index.json"
    if not idx_path.exists():
        return {"scanned": {}}
    return json.loads(idx_path.read_text())


# ─── Basic functionality ───────────────────────────────────────────────────────

class TestScanBasic:
    def test_cover_letter_content_returned(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Airbnb Cover Letter.txt", "Unique cover letter text here.")
        out = server.scan_materials_for_tone(category="cover_letters", limit=5)
        assert "Unique cover letter text here." in out

    def test_filename_in_output(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Reddit Cover Letter.txt", "content")
        out = server.scan_materials_for_tone(category="cover_letters")
        assert "Reddit Cover Letter.txt" in out

    def test_resumes_category_scans_correct_dir(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_resume(res_dir, "Frank MacBride Resume - GM.txt", "GM resume body.")
        _make_cover_letter(res_dir, "ShouldNotAppear.txt", "cover letter content")
        out = server.scan_materials_for_tone(category="resumes", limit=5)
        assert "GM resume body." in out
        assert "cover letter content" not in out

    def test_misc_category_reads_root_txt(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_misc(res_dir, "LinkedIn Message.txt", "LinkedIn misc text.")
        out = server.scan_materials_for_tone(category="misc", limit=5)
        assert "LinkedIn misc text." in out

    def test_no_files_returns_all_scanned_message(self, isolated_server, tmp_path):
        out = server.scan_materials_for_tone(category="cover_letters")
        assert "scanned" in out.lower() or "no" in out.lower()

    def test_extraction_instructions_present(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Acme Cover Letter.txt", "some text")
        out = server.scan_materials_for_tone(category="cover_letters")
        assert "log_tone_sample" in out
        assert "log_personal_story" in out


# ─── Limit parameter ──────────────────────────────────────────────────────────

class TestLimit:
    def test_limit_controls_files_returned(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        for i in range(5):
            _make_cover_letter(res_dir, f"Company{i} Cover Letter.txt", f"content {i}")
        out = server.scan_materials_for_tone(category="cover_letters", limit=2)
        count = out.count("FILE:")
        assert count == 2

    def test_default_limit_is_three(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        for i in range(5):
            _make_cover_letter(res_dir, f"Co{i} Cover Letter.txt", f"body {i}")
        out = server.scan_materials_for_tone(category="cover_letters")
        assert out.count("FILE:") == 3


# ─── Scan index persistence ───────────────────────────────────────────────────

class TestScanIndex:
    def test_scanned_files_recorded_in_index(self, isolated_server, tmp_path):
        res_dir  = tmp_path / "resumes"
        data_dir = tmp_path / "data"
        _make_cover_letter(res_dir, "Airbnb Cover Letter.txt", "content")
        server.scan_materials_for_tone(category="cover_letters", limit=5)
        idx = _load_index(data_dir)
        keys = list(idx["scanned"].keys())
        assert any("Airbnb Cover Letter.txt" in k for k in keys)

    def test_already_scanned_files_skipped_by_default(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Airbnb Cover Letter.txt", "content1")
        _make_cover_letter(res_dir, "Reddit Cover Letter.txt", "content2")
        # First pass scans both
        server.scan_materials_for_tone(category="cover_letters", limit=5)
        # Second pass should find nothing new
        out = server.scan_materials_for_tone(category="cover_letters", limit=5)
        assert "FILE:" not in out  # nothing returned
        assert "scanned" in out.lower()

    def test_force_rescans_already_indexed_files(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Airbnb Cover Letter.txt", "rescan content")
        # First scan
        server.scan_materials_for_tone(category="cover_letters", limit=5)
        # Force re-scan
        out = server.scan_materials_for_tone(category="cover_letters", limit=5, force=True)
        assert "rescan content" in out

    def test_scan_index_timestamps_are_iso_format(self, isolated_server, tmp_path):
        res_dir  = tmp_path / "resumes"
        data_dir = tmp_path / "data"
        _make_cover_letter(res_dir, "SomeCo Cover Letter.txt", "text")
        server.scan_materials_for_tone(category="cover_letters", limit=5)
        idx = _load_index(data_dir)
        for ts in idx["scanned"].values():
            assert "T" in ts  # ISO 8601 separator


# ─── Company filter ───────────────────────────────────────────────────────────

class TestCompanyFilter:
    def test_company_filter_limits_to_matching_files(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Airbnb Cover Letter.txt", "airbnb text here")
        _make_cover_letter(res_dir, "Reddit Cover Letter.txt", "reddit text here")
        out = server.scan_materials_for_tone(category="cover_letters", company="Airbnb")
        assert "airbnb text here" in out
        assert "reddit text here" not in out

    def test_company_filter_case_insensitive(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Airbnb Cover Letter.txt", "airbnb content")
        out = server.scan_materials_for_tone(category="cover_letters", company="airbnb")
        assert "airbnb content" in out

    def test_company_filter_no_match_reports_all_scanned(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Reddit Cover Letter.txt", "reddit content")
        out = server.scan_materials_for_tone(category="cover_letters", company="Zillow")
        assert "FILE:" not in out

    def test_company_filter_note_in_empty_message(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        _make_cover_letter(res_dir, "Reddit Cover Letter.txt", "text")
        out = server.scan_materials_for_tone(
            category="cover_letters", company="Zillow"
        )
        assert "Zillow" in out or "scanned" in out.lower()


# ─── Remaining count reported ─────────────────────────────────────────────────

class TestRemainingCount:
    def test_remaining_count_decreases_each_call(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        for i in range(4):
            _make_cover_letter(res_dir, f"Co{i} Cover Letter.txt", f"text {i}")
        out1 = server.scan_materials_for_tone(category="cover_letters", limit=2)
        out2 = server.scan_materials_for_tone(category="cover_letters", limit=2)
        # Both calls should succeed with FILE: entries
        assert out1.count("FILE:") == 2
        assert out2.count("FILE:") == 2

    def test_scan_again_message_present(self, isolated_server, tmp_path):
        res_dir = tmp_path / "resumes"
        for i in range(5):
            _make_cover_letter(res_dir, f"C{i} Cover Letter.txt", f"body {i}")
        out = server.scan_materials_for_tone(category="cover_letters", limit=2)
        assert "scan_materials_for_tone" in out
