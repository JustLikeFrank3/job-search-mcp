# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.4.2] - 2026-02-21

### Fixed
- `server.py` — unterminated triple-quoted docstring (missing closing `"""`) caused server to crash with `SyntaxError` on startup; all tools appeared disabled in VS Code
- All 22 tool functions across 10 modules (`job_hunt`, `resume`, `fitment`, `interview`, `spicam`, `health`, `star`, `tone`, `rag`, `context`) were missing docstrings, causing VS Code MCP to warn "Tool does not have a description" and refuse to call them; added accurate descriptions to every tool

## [0.4.1] - 2026-02-21

### Fixed
- `templates/resume.html` — increased bottom page margin (`0.42in` → `0.52in`) and changed footer `vertical-align` from `bottom` to `middle` so the `</ROLE>` tag no longer sits flush against the paper edge
- `templates/cover_letter.html` — name sidebar font size bumped from `15pt` to `18pt` for better visual weight
- `tools/export.py` `_parse_cover_letter_txt` — salutation lines (`Dear …`, `To whom …`, `Hello`, `Hi`) were being mistakenly parsed as the author's name; now excluded from name detection and correctly fall back to the hardcoded default
- `tools/export.py` — closing salutation now normalised to `"Kindest regards,"` on its own line with `"Frank Vladmir MacBride III"` as a separate line below; replaces the old single-line `"Best regards, Frank V. MacBride III"` across all cover letters
- `tools/export.py` — full middle name **Vladmir** added to the cover-letter sidebar default (`FRANK VLADMIR MACBRIDE III`) and all generated signatures

## [0.4.0] - 2026-02-21

### Added
- `draft_outreach_message(contact, company, context, message_type?)` — packages tone profile, personal context, application status, and message-type-specific writing instructions so the AI can draft ready-to-send outreach in Frank's voice; supports `linkedin_followup`, `thank_you`, `referral_ask`, `recruiter_nudge`, and `cold_outreach`; auto-detects message type from context when not specified
- `export_resume_pdf(filename, footer_tag?, output_filename?)` — parses a `.txt` resume from `01-Current-Optimized/` and renders a pixel-perfect PDF matching the Courier New / code-aesthetic template (header bracket `<NAME>`, section underlines, `</ROLE>` footer tag); renders via Jinja2 + WeasyPrint
- `export_cover_letter_pdf(filename, output_filename?)` — parses a `.txt` cover letter from `02-Cover-Letters/` and renders a PDF with the two-column sidebar layout (stacked name, bracket aesthetic, contact block); PDFs land in `03-Resume-PDFs/`
- `tools/outreach.py` module — new tool module following the `register(mcp)` pattern
- `tools/export.py` module — resume/cover-letter parser + WeasyPrint renderer; `templates/resume.html` and `templates/cover_letter.html` are Jinja2 templates matching the Canva design
- **System dep:** `pango` (Homebrew) required by WeasyPrint on macOS (`brew install pango`)

## [0.3.0] - 2026-02-21

### Added
- **Modular architecture** — split monolithic `server.py` (1200+ lines) into `lib/` (config, I/O, pure helpers) and `tools/` packages (11 modules), each with a `register(mcp)` pattern; `server.py` is now a thin ~120-line orchestrator
- `log_personal_story(story, tags, people?, title?)` — log personal STAR stories with tags and optional people/title metadata
- `get_personal_context(tag?, person?)` — retrieve stories filtered by tag or person
- `log_tone_sample(text, source, context?)` — ingest a writing sample to teach tone/voice
- `get_tone_profile()` — retrieve all tone samples for AI drafting context
- `scan_materials_for_tone(category?)` — auto-scan resumes, cover letters, and prep files to index new tone samples; persists a scan index to avoid re-ingesting unchanged files
- `get_star_story_context(tag, company?, role_type?)` — retrieve matching STAR stories, derived metric bullets (via tag chains), and company-specific framing hints in one call
- **Implied daily check-in nudge** — `get_job_hunt_status()` appends a reminder to log a mental health check-in if none exists for today
- **Test coverage** — 134 tests, 69% total coverage across `server`, `lib`, and `tools`; `pyproject.toml` now tracks all three sources with `fail_under = 50`

### Changed
- `server.py` re-exports all tool functions + path globals for backward compatibility with existing test fixtures
- `lib/config._load_config()` now falls back to `config.example.json` when `config.json` is absent — clean clones and CI no longer crash on import
- `log_personal_story` `people` parameter changed from mutable default `[]` to `None` (normalized inside the function) — fixes classic Python mutable-default-argument bug

### Fixed
- Resolved all 3 Copilot automated code review suggestions from PR #10: mutable default argument, eager config load in CI, and misleading conftest docstring

## [0.2.0] - 2026-02-20

### Added
- **RAG semantic search** (`rag.py`) — chunks and embeds all job search materials using OpenAI `text-embedding-3-small`
- `search_materials(query, category?, n_results?)` MCP tool — returns ranked chunks across resume, cover letters, LeetCode, interview prep, and reference files
- `reindex_materials()` MCP tool — rebuilds the full semantic index on demand
- Pure numpy cosine similarity search — no external vector database required

### Changed
- Replaced ChromaDB with numpy + JSON — fully compatible with Python 3.14 (ChromaDB uses Pydantic V1, which is incompatible)
- Updated README setup instructions to include `openai` and `numpy` in the install step

### Dependencies
- Added: `openai>=1.0.0`, `numpy>=1.24.0`
- Removed: `chromadb` (incompatible with Python 3.14)

---

## [0.1.0] - 2026-02-19

### Added
- Initial release — 15-tool FastMCP server for persistent job search context
- `get_job_hunt_status()` — live application pipeline
- `update_application()` — add/update applications with status, notes, contacts
- `read_master_resume()` — full master resume with all metrics and achievements
- `assess_job_fitment(company, role, jd)` — packages resume + JD for fitment analysis
- `get_customization_strategy(role_type)` — resume emphasis strategy by role type
- `generate_interview_prep_context(company, role, stage)` — structured interview prep
- `get_leetcode_cheatsheet(section?)` — algorithm patterns reference
- `get_interview_quick_reference()` — STAR stories, system design framework, behavioral talking points
- `scan_spicam_for_skills()` — scans IoT side-project codebase for new resume skills
- `log_mental_health_checkin(mood, energy, ...)` — mood/energy logging
- `get_mental_health_log(days?)` — recent check-in history
- `list_existing_materials(company?)` — list resumes and cover letters
- `read_existing_resume(filename)` — read a specific resume file
- `read_reference_file(filename)` — read reference materials
- `get_existing_prep_file(company)` — read an interview prep document
- `config.json`-based path loading — no hardcoded paths in source
- Sanitized for public GitHub — `config.example.json`, example data templates, `.gitignore`
