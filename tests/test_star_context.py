"""
Tests for get_star_story_context().

Covers:
- Direct tag matching
- Related tag traversal
- Resume metric insertion
- Company framing hints
- Unknown tag / empty story graceful fallback
- STAR scaffold always present in output
"""

import pytest

import server as srv


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def seeded(isolated_server):
    """Seed stories for the STAR context tests."""
    srv.log_personal_story(
        story="Fire truck brass threads — quality built in, not bolted on.",
        tags=["ford", "quality", "testing", "craftsmanship", "solo-developer"],
        title="1934 Ford Fire Truck",
    )
    srv.log_personal_story(
        story="Grandfather 50 years at Ford, service manager at 19.",
        tags=["ford", "family", "motivation"],
        people=["Grandpa"],
        title="Grandfather at Ford",
    )
    srv.log_personal_story(
        story="Sean Evans introduced me to Just Like Frank by LTJ.",
        tags=["music", "friendship", "identity"],
        people=["Sean Evans"],
        title="Sean Evans / LTJ",
    )
    return isolated_server


# ──────────────────────────────────────────────────────────────────────────────
# Tag matching
# ──────────────────────────────────────────────────────────────────────────────

class TestTagMatching:
    def test_direct_tag_match_primary(self, seeded):
        result = srv.get_star_story_context("testing")
        assert "PRIMARY STORIES" in result
        assert "1934 Ford Fire Truck" in result

    def test_related_tag_pulls_additional_stories(self, seeded):
        # "quality" is related to "testing" — should bring in fire truck too
        result = srv.get_star_story_context("testing")
        # The fire truck story matches directly, grandfather matches via "ford"
        # which is in the related chain: testing -> craftsmanship -> ford
        assert "Fire Truck" in result

    def test_unrelated_tag_not_included(self, seeded):
        result = srv.get_star_story_context("testing")
        # "Sean Evans / LTJ" has no tags related to testing
        assert "Sean Evans" not in result

    def test_unknown_tag_graceful(self, seeded):
        result = srv.get_star_story_context("absolutely_nonexistent_tag")
        assert "No personal stories found" in result

    def test_no_stories_at_all(self, isolated_server):
        result = srv.get_star_story_context("testing")
        assert "No personal stories found" in result
        assert "log_personal_story" in result  # gives actionable guidance


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_testing_tag_includes_coverage_metric(self, isolated_server):
        result = srv.get_star_story_context("testing")
        assert "80%" in result

    def test_solo_developer_metrics_present(self, isolated_server):
        result = srv.get_star_story_context("solo-developer")
        assert "500K" in result

    def test_ai_tag_includes_adoption_metric(self, isolated_server):
        result = srv.get_star_story_context("ai")
        assert "35%" in result or "3.5x" in result

    def test_cloud_tag_includes_azure(self, isolated_server):
        result = srv.get_star_story_context("cloud")
        assert "Azure" in result

    def test_no_duplicate_metrics(self, isolated_server):
        # testing → quality → craftsmanship → solo-developer all overlap;
        # each metric line should appear at most once
        result = srv.get_star_story_context("testing")
        lines = [l.strip() for l in result.splitlines() if l.strip().startswith("•")]
        seen: set = set()
        for line in lines:
            assert line not in seen, f"Duplicate metric: {line}"
            seen.add(line)


# ──────────────────────────────────────────────────────────────────────────────
# Company framing
# ──────────────────────────────────────────────────────────────────────────────

class TestCompanyFraming:
    def test_ford_framing_present(self, isolated_server):
        result = srv.get_star_story_context("ford", company="ford")
        assert "FORD FRAMING" in result.upper()
        assert "grandfather" in result.lower() or "legacy" in result.lower()

    def test_fanduel_framing_present(self, isolated_server):
        result = srv.get_star_story_context("testing", company="fanduel")
        assert "FANDUEL" in result.upper()

    def test_unknown_company_no_framing_section(self, isolated_server):
        result = srv.get_star_story_context("testing", company="UnknownCorpXYZ")
        assert "FRAMING HINTS" not in result

    def test_company_name_in_header(self, isolated_server):
        result = srv.get_star_story_context("ai", company="microsoft")
        assert "microsoft" in result.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Output structure
# ──────────────────────────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_star_scaffold_always_present(self, isolated_server):
        result = srv.get_star_story_context("testing")
        for label in ("Situation", "Task", "Action", "Result"):
            assert label in result, f"STAR label '{label}' missing from output"

    def test_header_contains_tag(self, isolated_server):
        result = srv.get_star_story_context("leadership")
        assert "tag='leadership'" in result

    def test_header_contains_company_when_provided(self, isolated_server):
        result = srv.get_star_story_context("ai", company="reddit")
        assert "company='reddit'" in result

    def test_role_type_in_header(self, isolated_server):
        result = srv.get_star_story_context("ai", role_type="backend")
        assert "role='backend'" in result
