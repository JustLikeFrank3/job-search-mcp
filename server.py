#!/usr/bin/env python3
"""
Job Search MCP Server for Frank MacBride
-----------------------------------------
Provides tools for:
  - Job hunt status tracking
  - Resume / cover letter context generation
  - Job fitment assessment
  - Interview & LeetCode prep
  - sPiCam project skill scanning
  - Mental health check-in logging
  - Personal story / context library (v3)
  - Tone ingestion + voice profile (v3)
  - PDF export for resumes and cover letters (v4)
"""
from mcp.server.fastmcp import FastMCP

from lib import config
from lib.config import _load_config
from lib.io import _read, _load_json, _save_json, _now
from lib.helpers import (
    _build_story_entry,
    _filter_stories,
    _format_story_list,
    _build_checkin_entry,
    _build_tone_sample_entry,
    _scan_dirs,
)

from tools import (
    job_hunt,
    resume,
    fitment,
    interview,
    spicam,
    health,
    context,
    tone,
    rag,
    star,
    outreach,
    export,
)


def _sync_config_exports() -> None:
    global _cfg
    global RESUME_FOLDER, LEETCODE_FOLDER, SPICAM_FOLDER, DATA_FOLDER
    global STATUS_FILE, HEALTH_LOG_FILE, PERSONAL_CONTEXT_FILE, TONE_FILE, SCAN_INDEX_FILE
    global MASTER_RESUME, LEETCODE_CHEATSHEET, QUICK_REFERENCE

    _cfg = config._cfg

    RESUME_FOLDER = config.RESUME_FOLDER
    LEETCODE_FOLDER = config.LEETCODE_FOLDER
    SPICAM_FOLDER = config.SPICAM_FOLDER
    DATA_FOLDER = config.DATA_FOLDER

    STATUS_FILE = config.STATUS_FILE
    HEALTH_LOG_FILE = config.HEALTH_LOG_FILE
    PERSONAL_CONTEXT_FILE = config.PERSONAL_CONTEXT_FILE
    TONE_FILE = config.TONE_FILE
    SCAN_INDEX_FILE = config.SCAN_INDEX_FILE

    MASTER_RESUME = config.MASTER_RESUME
    LEETCODE_CHEATSHEET = config.LEETCODE_CHEATSHEET
    QUICK_REFERENCE = config.QUICK_REFERENCE


def _reconfigure(cfg: dict) -> None:
    config._reconfigure(cfg)
    _sync_config_exports()


_sync_config_exports()


mcp = FastMCP(
    "job-search-as",
    instructions=(
        "You are Frank MacBride's personal job search assistant. "
        "You have direct filesystem access to his resume materials, job hunt status, "
        "and interview prep files. Use the available tools to retrieve context before "
        "generating resumes, cover letters, prep docs, or assessments. "
        "Always read the master resume before generating any application material."
    ),
)


job_hunt.register(mcp)
resume.register(mcp)
fitment.register(mcp)
interview.register(mcp)
spicam.register(mcp)
health.register(mcp)
context.register(mcp)
tone.register(mcp)
rag.register(mcp)
star.register(mcp)
outreach.register(mcp)
export.register(mcp)


get_job_hunt_status = job_hunt.get_job_hunt_status
update_application = job_hunt.update_application

read_master_resume = resume.read_master_resume
list_existing_materials = resume.list_existing_materials
read_existing_resume = resume.read_existing_resume
read_reference_file = resume.read_reference_file

assess_job_fitment = fitment.assess_job_fitment
get_customization_strategy = fitment.get_customization_strategy

get_interview_quick_reference = interview.get_interview_quick_reference
get_leetcode_cheatsheet = interview.get_leetcode_cheatsheet
generate_interview_prep_context = interview.generate_interview_prep_context
get_existing_prep_file = interview.get_existing_prep_file

scan_spicam_for_skills = spicam.scan_spicam_for_skills

log_mental_health_checkin = health.log_mental_health_checkin
get_mental_health_log = health.get_mental_health_log

log_personal_story = context.log_personal_story
get_personal_context = context.get_personal_context

log_tone_sample = tone.log_tone_sample
get_tone_profile = tone.get_tone_profile
scan_materials_for_tone = tone.scan_materials_for_tone

search_materials = rag.search_materials
reindex_materials = rag.reindex_materials

get_star_story_context = star.get_star_story_context

draft_outreach_message = outreach.draft_outreach_message

export_resume_pdf = export.export_resume_pdf
export_cover_letter_pdf = export.export_cover_letter_pdf


if __name__ == "__main__":
    mcp.run()
