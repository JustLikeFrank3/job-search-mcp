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
"""

import json
import os
import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Paths are loaded from config.json (gitignored) sitting next to this file.
# Copy config.example.json → config.json and fill in your own paths.

_HERE = Path(__file__).parent


def _load_config() -> dict:
    config_path = _HERE / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_path}\n"
            "Copy config.example.json → config.json and fill in your paths."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


# Paths are module-level globals so they can be swapped by _reconfigure() in tests.
# Production: _load_config() runs once at startup and populates these.
# Tests: call _reconfigure(fake_cfg_dict) from conftest before executing any tool.

RESUME_FOLDER: Path
LEETCODE_FOLDER: Path
SPICAM_FOLDER: Path
DATA_FOLDER: Path

STATUS_FILE: Path
HEALTH_LOG_FILE: Path
PERSONAL_CONTEXT_FILE: Path
TONE_FILE: Path
SCAN_INDEX_FILE: Path

MASTER_RESUME: Path
LEETCODE_CHEATSHEET: Path
QUICK_REFERENCE: Path


def _reconfigure(cfg: dict) -> None:
    """Re-initialise all path globals from *cfg*.

    Called once at startup (via _load_config()) and can be called again by
    tests to inject a fully-controlled tmp-dir config without touching disk.
    """
    global RESUME_FOLDER, LEETCODE_FOLDER, SPICAM_FOLDER, DATA_FOLDER
    global STATUS_FILE, HEALTH_LOG_FILE, PERSONAL_CONTEXT_FILE, TONE_FILE, SCAN_INDEX_FILE
    global MASTER_RESUME, LEETCODE_CHEATSHEET, QUICK_REFERENCE

    RESUME_FOLDER   = Path(cfg["resume_folder"])
    LEETCODE_FOLDER = Path(cfg["leetcode_folder"])
    SPICAM_FOLDER   = Path(cfg["spicam_folder"])
    DATA_FOLDER     = Path(cfg["data_folder"])

    STATUS_FILE           = DATA_FOLDER / "status.json"
    HEALTH_LOG_FILE       = DATA_FOLDER / "mental_health_log.json"
    PERSONAL_CONTEXT_FILE = DATA_FOLDER / "personal_context.json"
    TONE_FILE             = DATA_FOLDER / "tone_samples.json"
    SCAN_INDEX_FILE       = DATA_FOLDER / "scan_index.json"

    MASTER_RESUME       = RESUME_FOLDER / cfg["master_resume_path"]
    LEETCODE_CHEATSHEET = LEETCODE_FOLDER / cfg["leetcode_cheatsheet_path"]
    QUICK_REFERENCE     = LEETCODE_FOLDER / cfg["quick_reference_path"]


_reconfigure(_load_config())

# ─── SERVER ───────────────────────────────────────────────────────────────────

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

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[Error reading {path.name}: {e}]"


def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


# ─── PURE LOGIC HELPERS (no I/O — fully unit-testable) ───────────────────────

def _build_story_entry(stories: list, story: str, tags: list, people: list, title: str) -> dict:
    """Build a new story dict. Does not touch the filesystem."""
    return {
        "id":        len(stories) + 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "title":     title or (story[:60] + ("..." if len(story) > 60 else "")),
        "story":     story,
        "tags":      [t.lower().strip() for t in tags],
        "people":    list(people),
    }


def _filter_stories(stories: list, tag: str = "", person: str = "") -> list:
    """Return stories matching optional tag / person filters."""
    if tag:
        stories = [s for s in stories if tag.lower() in s.get("tags", [])]
    if person:
        stories = [s for s in stories if
                   any(person.lower() in p.lower() for p in s.get("people", []))]
    return stories


def _format_story_list(stories: list) -> str:
    """Render a list of story dicts as a human-readable string."""
    lines = [f"\u2550\u2550\u2550 PERSONAL CONTEXT ({len(stories)} stories) \u2550\u2550\u2550", ""]
    for s in stories:
        lines.append(f"\u25aa #{s['id']} \u2014 {s['title']}")
        lines.append(f"  Tags:   {', '.join(s.get('tags', []))}")
        if s.get("people"):
            lines.append(f"  People: {', '.join(s['people'])}")
        lines.append(f"  {s['story']}")
        lines.append("")
    return "\n".join(lines)


def _build_checkin_entry(mood: str, energy: int, notes: str, productive: bool) -> tuple:
    """Build a mental-health check-in dict and its guidance string."""
    energy_int = max(1, min(10, int(energy)))
    entry = {
        "timestamp":  datetime.datetime.now().isoformat(),
        "date":       datetime.date.today().isoformat(),
        "mood":       mood,
        "energy":     energy_int,
        "productive": bool(productive),
        "notes":      notes,
    }
    if energy_int <= 3 or mood in ("depressed", "low"):
        guidance = (
            "Low energy logged. Small wins count — "
            "even one LeetCode problem or one email sent is real progress. "
            "You're still moving, even on hard days."
        )
    elif mood == "hyperfocus" or energy_int >= 8:
        guidance = (
            "High energy logged. Good time for deep work. "
            "Just remember to eat, hydrate, and step away before burnout hits."
        )
    else:
        guidance = "Logged. You're doing the work, even when it's hard."
    return entry, guidance


def _build_tone_sample_entry(samples: list, text: str, source: str, context: str) -> dict:
    """Build a tone-sample dict. Does not touch the filesystem."""
    return {
        "id":         len(samples) + 1,
        "timestamp":  datetime.datetime.now().isoformat(),
        "source":     source,
        "context":    context,
        "text":       text,
        "word_count": len(text.split()),
    }


def _scan_dirs(category: str) -> list:
    """Resolve which RESUME_FOLDER subdirectories to scan for a given category."""
    mapping: dict = {
        "cover_letters": [RESUME_FOLDER / "02-Cover-Letters"],
        "resumes":       [RESUME_FOLDER / "01-Current-Optimized"],
        "misc":          [RESUME_FOLDER],
        "all":           [
            RESUME_FOLDER / "02-Cover-Letters",
            RESUME_FOLDER / "01-Current-Optimized",
            RESUME_FOLDER,
        ],
    }
    return mapping.get(category.lower(), [RESUME_FOLDER / "02-Cover-Letters"])


# ─── TOOLS: JOB HUNT STATUS ───────────────────────────────────────────────────

@mcp.tool()
def get_job_hunt_status() -> str:
    """
    Returns the current status of all tracked job applications with
    next steps, contacts, and notes.
    """
    data = _load_json(STATUS_FILE, {"applications": []})
    apps = data.get("applications", [])
    if not apps:
        return "No applications tracked yet. Use update_application() to add one."

    lines = [
        "═══ JOB HUNT STATUS ═══",
        f"Last updated: {data.get('last_updated', 'unknown')}",
        "",
    ]
    for app in apps:
        lines.append(f"■ {app['company']} — {app['role']}")
        lines.append(f"  Status:       {app['status']}")
        lines.append(f"  Last update:  {app.get('last_updated', '—')}")
        if app.get("next_steps"):
            lines.append(f"  Next steps:   {app['next_steps']}")
        if app.get("contact"):
            lines.append(f"  Contact:      {app['contact']}")
        if app.get("notes"):
            lines.append(f"  Notes:        {app['notes']}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def update_application(
    company: str,
    role: str,
    status: str,
    next_steps: str = "",
    contact: str = "",
    notes: str = "",
) -> str:
    """
    Add or update a job application in the tracker.

    status values: applied | phone_screen | technical_screen | coding_assessment
                   system_design | onsite | offer | rejected | withdrawn | waiting
    """
    data = _load_json(STATUS_FILE, {"applications": []})
    apps: list = data.setdefault("applications", [])

    existing = next(
        (a for a in apps
         if a["company"].lower() == company.lower()
         and a["role"].lower() == role.lower()),
        None,
    )
    # Fallback: match on company only if exact role not found
    if existing is None:
        existing = next(
            (a for a in apps if a["company"].lower() == company.lower()), None
        )
    if existing:
        existing.update(
            role=role,
            status=status,
            next_steps=next_steps,
            contact=contact,
            notes=notes,
            last_updated=_now(),
        )
        action = "Updated"
    else:
        apps.append(
            dict(
                company=company,
                role=role,
                status=status,
                next_steps=next_steps,
                contact=contact,
                notes=notes,
                applied_date=_now(),
                last_updated=_now(),
            )
        )
        action = "Added"

    data["last_updated"] = _now()
    _save_json(STATUS_FILE, data)
    return f"✓ {action}: {company} — {role} ({status})"


# ─── TOOLS: RESUME CONTENT ────────────────────────────────────────────────────

@mcp.tool()
def read_master_resume() -> str:
    """
    Returns the full MASTER SOURCE resume with all metrics, achievements, and
    personal context notes. Always call this before generating any resume or
    cover letter.
    """
    return _read(MASTER_RESUME)


@mcp.tool()
def list_existing_materials(company: str = "") -> str:
    """
    Lists resumes and cover letters already generated in this workspace.
    Optionally filter by company name.
    """
    optimized_dir    = RESUME_FOLDER / _cfg["optimized_resumes_dir"]
    cover_letter_dir = RESUME_FOLDER / _cfg["cover_letters_dir"]

    def _list_dir(d: Path, label: str) -> list[str]:
        if not d.exists():
            return [f"  (folder not found: {d.name})"]
        files = sorted(
            f.name for f in d.iterdir()
            if f.suffix in (".txt", ".md") and "MASTER" not in f.name
            and (not company or company.lower() in f.name.lower())
        )
        out = [f"\n══ {label} ({len(files)}) ══"]
        out += [f"  {f}" for f in files] or ["  (none found)"]
        return out

    lines = []
    lines += _list_dir(optimized_dir,    "RESUMES")
    lines += _list_dir(cover_letter_dir, "COVER LETTERS")
    return "\n".join(lines)


@mcp.tool()
def read_existing_resume(filename: str) -> str:
    """
    Read the content of a specific resume file.
    filename should be just the base filename (no path) from 01-Current-Optimized/.
    Use list_existing_materials() to see available files.
    """
    path = RESUME_FOLDER / _cfg["optimized_resumes_dir"] / filename
    if not path.exists():
        return f"Not found: {filename}\nUse list_existing_materials() to see available resumes."
    return _read(path)


@mcp.tool()
def read_reference_file(filename: str) -> str:
    """
    Read a file from the 06-Reference-Materials/ folder.
    Useful files: 'Frank MacBride Resume - Template Format.txt',
                  'Skills - 10% Shorter.txt',
                  'GM Recognition Awards, Feedback_Received.txt'
    """
    ref_dir = RESUME_FOLDER / _cfg["reference_materials_dir"]
    path = ref_dir / filename
    if not path.exists():
        available = sorted(
            f.name for f in ref_dir.iterdir()
        ) if ref_dir.exists() else []
        return f"Not found: {filename}\nAvailable: {available}"
    return _read(path)


# ─── TOOLS: FITMENT & STRATEGY ────────────────────────────────────────────────

@mcp.tool()
def assess_job_fitment(company: str, role: str, job_description: str) -> str:
    """
    Provides everything needed to assess how well Frank's background maps to a
    job description. Returns the master resume + JD packaged for analysis.

    After calling this tool, produce:
      1. Overall fit score (0–100%)
      2. Key strengths matching requirements
      3. Gaps / risks
      4. Suggested resume emphasis (which bullets to lead with)
      5. Top 3–5 STAR stories from Frank's background most relevant here
      6. Questions to probe / clarify before applying
    """
    master = _read(MASTER_RESUME)
    return (
        f"═══ FITMENT ASSESSMENT ═══\n"
        f"Company: {company}\n"
        f"Role:    {role}\n\n"
        f"──── JOB DESCRIPTION ────\n{job_description}\n\n"
        f"──── FRANK'S MASTER RESUME ────\n{master}"
    )


@mcp.tool()
def get_customization_strategy(role_type: str) -> str:
    """
    Returns the recommended resume emphasis strategy for a given role type.

    role_type options:
      testing | cloud | data_engineering | backend | fullstack | ai_innovation | iot
    """
    strategies = {
        "testing": (
            "Lead with JUnit/Mockito/Selenium expertise, 80%+ coverage metrics, TDD practices. "
            "Feature the 'prevented production defects' story. "
            "Highlight Karma/Jest for frontend testing."
        ),
        "cloud": (
            "Lead with Azure Container Apps, Terraform IaC, zero-downtime PCF→OCF→Azure migration. "
            "Emphasize containerization, CI/CD pipelines, and infrastructure-as-code."
        ),
        "data_engineering": (
            "Lead with IBM DataStage, PySpark ETL pipelines, Oracle→PostgreSQL migration. "
            "Emphasize data modeling, multi-source integration, and warehouse work."
        ),
        "backend": (
            "Lead with microservices architecture, event-driven pub/sub, Spring Boot, "
            "98% SLA compliance on production forecasting app. "
            "Emphasize distributed systems debugging, on-call rotation, observability."
        ),
        "fullstack": (
            "Lead with Java/Spring Boot + Angular/TypeScript full ownership. "
            "Emphasize API design, end-to-end modernization, cross-functional product work."
        ),
        "ai_innovation": (
            "Lead with GitHub Copilot champion story: 35% org adoption, 3.5x target. "
            "Emphasize technical evangelism, AI tooling adoption, and measurable team impact."
        ),
        "iot": (
            "Lead with IoT Engineering degree, RetrosPiCam project (FastAPI + Raspberry Pi + servo HAT), "
            "hardware/software integration, and Azure edge/cloud connectivity. "
            "Tie in LiveVox latency work (2.8ms web, 12.7ms iOS) as embedded-adjacent."
        ),
    }
    result = strategies.get(role_type.lower())
    if result:
        return f"Strategy for '{role_type}':\n\n{result}"
    return (
        f"Unknown role type: '{role_type}'\n"
        f"Available options: {', '.join(strategies)}"
    )


# ─── TOOLS: INTERVIEW & LEETCODE PREP ────────────────────────────────────────

@mcp.tool()
def get_interview_quick_reference() -> str:
    """
    Returns Frank's interview day quick reference: STAR stories, system design
    framework, behavioral talking points, questions to ask. Best used before
    any behavioral or system design interview.
    """
    return _read(QUICK_REFERENCE)


@mcp.tool()
def get_leetcode_cheatsheet(section: str = "") -> str:
    """
    Returns the full LeetCode / algorithm cheatsheet, or a specific section.

    section options (leave empty for full cheatsheet):
      arrays | linkedlist | trees | graphs | stacks | dp | system_design | strings
    """
    content = _read(LEETCODE_CHEATSHEET)
    if not section:
        return content

    # Try to isolate the requested section by header matching
    lines   = content.split("\n")
    result  = []
    inside  = False
    target  = section.lower()

    for line in lines:
        stripped = line.lstrip("#").strip().lower()
        is_header = line.startswith("#")

        if is_header and target in stripped:
            inside = True
        elif is_header and inside and target not in stripped:
            # Hit a same-or-higher-level header — stop
            if line.startswith("# ") or (line.startswith("## ") and not target in stripped):
                break

        if inside:
            result.append(line)

    if result:
        return "\n".join(result)
    return f"Section '{section}' not found. Returning full cheatsheet.\n\n{content}"


@mcp.tool()
def generate_interview_prep_context(
    company: str,
    role: str,
    stage: str = "phone_screen",
    job_description: str = "",
) -> str:
    """
    Packages Frank's master resume + quick reference for interview prep.
    Returns structured context for generating a company-specific prep document.

    stage options: phone_screen | technical_screen | coding_assessment
                   system_design | onsite | behavioral
    """
    master    = _read(MASTER_RESUME)
    quick_ref = _read(QUICK_REFERENCE)

    desc_block = f"\n──── JOB DESCRIPTION ────\n{job_description}" if job_description else ""

    return (
        f"═══ INTERVIEW PREP CONTEXT ═══\n"
        f"Company: {company}\n"
        f"Role:    {role}\n"
        f"Stage:   {stage}\n"
        f"{desc_block}\n\n"
        f"──── FRANK'S MASTER RESUME ────\n{master}\n\n"
        f"──── QUICK REFERENCE / STAR STORIES ────\n{quick_ref}\n\n"
        f"Use the above to produce:\n"
        f"  1. Top 5 things Frank must communicate for THIS role/stage\n"
        f"  2. Anticipated questions + suggested STAR responses\n"
        f"  3. Technical topics to review (if applicable)\n"
        f"  4. Smart questions for Frank to ask the interviewer\n"
        f"  5. Any gaps to proactively address\n"
        f"  6. Confidence anchors — Frank's strongest achievements relevant here\n"
    )


@mcp.tool()
def get_existing_prep_file(company: str) -> str:
    """
    Looks for an existing interview prep file for the given company.
    Searches the Resume 2025 folder recursively for matching .txt and .md files.
    """
    matches = sorted(
        f for f in RESUME_FOLDER.rglob("*")
        if f.suffix in (".txt", ".md")
        and company.lower() in f.name.lower()
        and any(kw in f.name.lower() for kw in ("prep", "interview", "call", "assessment"))
    )
    if not matches:
        return f"No existing prep files found for '{company}'."

    lines = [f"Found {len(matches)} prep file(s) for '{company}':\n"]
    for m in matches:
        lines.append(f"──── {m.name} ────")
        lines.append(_read(m))
        lines.append("")
    return "\n".join(lines)


# ─── TOOLS: RETROSPICAM SKILL SCANNER ────────────────────────────────────────

@mcp.tool()
def scan_spicam_for_skills() -> str:
    """
    Scans the RetrosPiCam project codebase and returns:
      - Technologies / patterns detected
      - Suggested resume bullets
      - New skills not yet in the master resume
    """
    if not SPICAM_FOLDER.exists():
        return f"RetrosPiCam folder not found at: {SPICAM_FOLDER}"

    tech_found: set[str] = set()
    file_inventory: list[str] = []

    SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "env", ".expo", "build", "dist"}

    for root, dirs, files in os.walk(SPICAM_FOLDER):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in files:
            fpath = Path(root) / fname
            rel   = str(fpath.relative_to(SPICAM_FOLDER))
            ext   = fpath.suffix.lower()
            file_inventory.append(rel)

            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue

            # ── Python ──
            if ext == ".py":
                tech_found.add("Python")
                if "fastapi"          in text: tech_found.add("FastAPI")
                if "pydantic"         in text: tech_found.add("Pydantic")
                if "azure"            in text: tech_found.add("Azure Blob Storage")
                if "async def"        in text: tech_found.add("Python async/await")
                if "websocket"        in text: tech_found.add("WebSockets")
                if "pytest"           in text: tech_found.add("pytest")
                if "systemd"          in text: tech_found.add("systemd / Linux services")
                if "gpio"             in text or "rpi" in text: tech_found.add("Raspberry Pi GPIO")
                if "servo"            in text or "adafruit" in text: tech_found.add("Servo / Adafruit HAT")
                if "range"            in text and "http" in text: tech_found.add("HTTP Range requests")
                if "jwt"              in text or "bearer" in text: tech_found.add("JWT authentication")
                if "docker"           in text: tech_found.add("Docker")
                if "retention"        in text: tech_found.add("Automated retention policies")

            # ── TypeScript / React Native ──
            elif ext in (".ts", ".tsx", ".js", ".jsx"):
                tech_found.add("TypeScript/JavaScript")
                if "react-native"     in text or "react native" in text: tech_found.add("React Native")
                if "expo"             in text: tech_found.add("Expo")
                if "testflight"       in text: tech_found.add("iOS TestFlight deployment")

            # ── Swift ──
            elif ext == ".swift":
                tech_found.add("Swift / iOS")

            # ── Infrastructure ──
            elif fname == "Dockerfile":
                tech_found.add("Docker")
            elif fname in ("docker-compose.yml", "docker-compose.yaml"):
                tech_found.add("Docker Compose")
            elif ext in (".tf", ".tfvars"):
                tech_found.add("Terraform IaC")

    already_on_resume = {
        "Python", "FastAPI", "Azure Blob Storage", "React Native", "Pydantic",
        "iOS TestFlight deployment", "HTTP Range requests",
    }
    new_skills = sorted(tech_found - already_on_resume)

    lines = [
        "═══ RETROSPICAM SKILL SCAN ═══",
        f"Files scanned: {len(file_inventory)}",
        "",
        "── All Technologies Detected ──",
    ]
    for t in sorted(tech_found):
        marker = "  ✓" if t in already_on_resume else "  ★ NEW"
        lines.append(f"{marker}  {t}")

    lines += [
        "",
        "── New Skills Not Yet on Master Resume ──",
    ]
    lines += [f"  • {s}" for s in new_skills] or ["  (none — master resume is up to date)"]

    lines += [
        "",
        "── Suggested Resume Bullets ──",
        "  • Built production IoT camera system integrating Raspberry Pi hardware, "
        "    servo HAT, Python/FastAPI backend, and React Native mobile app",
    ]
    if "Pydantic"              in tech_found: lines.append("  • Designed type-safe API layer with FastAPI + Pydantic models")
    if "Python async/await"    in tech_found: lines.append("  • Implemented async Python services for concurrent hardware + network I/O")
    if "Azure Blob Storage"    in tech_found: lines.append("  • Integrated Azure Blob Storage with automated 7-day retention management")
    if "HTTP Range requests"   in tech_found: lines.append("  • Enabled on-demand video streaming via HTTP Range request support")
    if "systemd / Linux services" in tech_found: lines.append("  • Configured systemd service for reliable auto-start on embedded Linux")
    if "WebSockets"            in tech_found: lines.append("  • Delivered real-time camera stream via WebSocket connections")
    if "JWT authentication"    in tech_found: lines.append("  • Secured API endpoints with JWT bearer-token authentication")

    return "\n".join(lines)


# ─── TOOLS: MENTAL HEALTH CHECK-INS ──────────────────────────────────────────

@mcp.tool()
def log_mental_health_checkin(
    mood: str,
    energy: int,
    notes: str = "",
    productive: bool = False,
) -> str:
    """
    Log a mental health check-in entry.

    mood:      e.g. 'good' | 'anxious' | 'depressed' | 'hyperfocus' | 'stable' | 'low' | 'motivated'
    energy:    1–10 scale (1 = can barely get out of bed, 10 = hyperfocus mode)
    notes:     optional free-text context
    productive: did you get meaningful work done today?
    """
    data  = _load_json(HEALTH_LOG_FILE, {"entries": []})
    entry, guidance = _build_checkin_entry(mood, energy, notes, productive)
    data.setdefault("entries", []).append(entry)
    _save_json(HEALTH_LOG_FILE, data)
    return f"Check-in saved ({entry['date']}, energy {entry['energy']}/10, mood: {mood}).\n{guidance}"


@mcp.tool()
def get_mental_health_log(days: int = 14) -> str:
    """
    Returns mental health log entries from the last N days (default: 14).
    Includes average energy trend and any notable patterns.
    """
    data    = _load_json(HEALTH_LOG_FILE, {"entries": []})
    entries = data.get("entries", [])

    cutoff  = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    recent  = [e for e in entries if e.get("date", "") >= cutoff]

    if not recent:
        return f"No check-ins logged in the past {days} days."

    lines = [f"═══ MENTAL HEALTH LOG (last {days} days) ═══", ""]
    for e in reversed(recent):
        eng = e.get("energy", 0)
        bar = "🟥" if eng <= 3 else "🟨" if eng <= 6 else "🟩"
        prod = "✓" if e.get("productive") else "–"
        lines.append(
            f"{e['date']}  {bar}  mood: {e['mood']:<12}  energy: {eng}/10  productive: {prod}"
        )
        if e.get("notes"):
            lines.append(f"          ↳ {e['notes']}")

    avg = sum(e.get("energy", 0) for e in recent) / len(recent)
    lines += ["", f"Average energy over {days} days: {avg:.1f}/10"]

    # Trend note
    if avg <= 4:
        lines.append("⚠  Trend: extended low-energy period. Consider reaching out for support.")
    elif avg >= 7:
        lines.append("✓  Trend: strong energy. Keep the momentum going.")

    return "\n".join(lines)


# ─── TOOLS: PERSONAL CONTEXT ────────────────────────────────────────────────

@mcp.tool()
def log_personal_story(
    story: str,
    tags: list[str],
    people: list[str] = [],
    title: str = "",
) -> str:
    """
    Log a personal story, memory, or context detail about Frank.
    These accumulate over time and are retrieved to enrich cover letters,
    dedications, behavioral answers, and any writing that should sound like Frank.

    tags examples: family | motivation | music | career | friendship | identity | humor
    people: names of people relevant to the story (e.g. ["Sean Evans", "Grandpa Frank"])
    title: optional short label for the story (auto-generated from first 60 chars if omitted)
    """
    data  = _load_json(PERSONAL_CONTEXT_FILE, {"stories": []})
    entry = _build_story_entry(data["stories"], story, tags, people, title)
    data["stories"].append(entry)
    _save_json(PERSONAL_CONTEXT_FILE, data)
    return f"\u2713 Story logged (#{entry['id']}): {entry['title']}"


@mcp.tool()
def get_personal_context(tag: str = "", person: str = "") -> str:
    """
    Retrieve Frank's logged personal stories and context.
    Filter by tag or person name, or leave both empty to get all stories.

    Use this before generating cover letters, dedications, or any writing
    that should feel personal rather than generic.
    """
    data    = _load_json(PERSONAL_CONTEXT_FILE, {"stories": []})
    stories = _filter_stories(data.get("stories", []), tag, person)

    if not stories:
        qualifier  = f" for tag '{tag}'"      if tag    else ""
        qualifier += f" for person '{person}'" if person else ""
        return f"No personal stories found{qualifier}."

    return _format_story_list(stories)


# ─── TOOLS: TONE INGESTION ────────────────────────────────────────────────────

@mcp.tool()
def log_tone_sample(
    text: str,
    source: str,
    context: str = "",
) -> str:
    """
    Ingest a writing sample to capture Frank's tone and voice.
    Over time, the AI uses these to write in Frank's style rather than generic AI prose.

    source: where the sample came from
            e.g. "cover letter Ford", "text to Jessica", "LinkedIn message to Cheyenne"
    context: optional note about the situation or mood when written
    """
    data  = _load_json(TONE_FILE, {"samples": []})
    entry = _build_tone_sample_entry(data["samples"], text, source, context)
    data["samples"].append(entry)
    _save_json(TONE_FILE, data)
    return f"\u2713 Tone sample logged (#{entry['id']}, {entry['word_count']} words from '{source}')"


@mcp.tool()
def get_tone_profile() -> str:
    """
    Returns all ingested tone samples so the AI can write in Frank's voice.
    Call this before drafting any email, cover letter, LinkedIn message,
    or other communication that needs to sound like Frank — not like an AI.
    """
    data = _load_json(TONE_FILE, {"samples": []})
    samples = data.get("samples", [])

    if not samples:
        return (
            "No tone samples logged yet.\n"
            "Use log_tone_sample() to ingest writing samples — cover letters, "
            "messages, anything Frank actually wrote."
        )

    total_words = sum(s.get("word_count", 0) for s in samples)
    lines = [
        f"\u2550\u2550\u2550 TONE PROFILE ({len(samples)} samples, {total_words} total words) \u2550\u2550\u2550",
        "Use these samples to calibrate Frank's voice before writing anything.",
        "",
    ]
    for s in samples:
        lines.append(f"\u2500\u2500 Sample #{s['id']} | {s['source']} \u2500\u2500")
        if s.get("context"):
            lines.append(f"Context: {s['context']}")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


# ─── TOOLS: RAG SEARCH ───────────────────────────────────────────────────────

@mcp.tool()
def search_materials(query: str, category: str = "") -> str:
    """
    Semantic search across all indexed job search materials.
    Much smarter than reading whole files — returns only the most relevant chunks.

    category options (leave empty to search all):
      resume | cover_letters | leetcode | interview_prep | reference

    Examples:
      search_materials("PostgreSQL migration zero downtime")
      search_materials("sliding window pattern", category="leetcode")
      search_materials("behavioral leadership story", category="interview_prep")
    """
    try:
        from rag import search, format_results
        hits = search(query, category=category or None, n_results=6)
        return format_results(hits, f'Results for: "{query}"')
    except Exception as e:
        if "openai_api_key" in str(e).lower() or "not set" in str(e).lower():
            return (
                "RAG search requires an OpenAI API key.\n"
                "Add 'openai_api_key' to config.json, then run reindex_materials()."
            )
        return f"Search error: {e}\nTry running reindex_materials() first."


@mcp.tool()
def reindex_materials() -> str:
    """
    (Re)build the RAG index from all job search materials.
    Run this once after setup, and again whenever you add new resumes,
    cover letters, or prep files. Takes ~30-60 seconds.
    """
    try:
        from rag import build_index
        counts = build_index(verbose=False)
        total  = sum(counts.values())
        lines  = [f"✓ Index built. {total} total chunks indexed:", ""]
        for cat, count in counts.items():
            lines.append(f"  {cat:<16} {count} chunks")
        lines += ["", "You can now use search_materials() for semantic search."]
        return "\n".join(lines)
    except Exception as e:
        if "openai_api_key" in str(e).lower() or "not set" in str(e).lower():
            return (
                "Indexing requires an OpenAI API key.\n"
                "Add 'openai_api_key' to config.json and try again."
            )
        return f"Indexing error: {e}"


# ─── TOOLS: MATERIAL SCAN ───────────────────────────────────────────────────

@mcp.tool()
def scan_materials_for_tone(
    category: str = "cover_letters",
    limit: int = 3,
    company: str = "",
    force: bool = False,
) -> str:
    """
    Read a batch of resumes/cover letters and return their full text for
    tone and personal context extraction.

    After reviewing, call log_tone_sample() for prose that sounds like Frank
    and log_personal_story() for any specific anecdotes or personal details.
    Then call scan_materials_for_tone() again to process the next batch —
    the tool tracks what has already been scanned.

    category: cover_letters | resumes | misc | all  (default: cover_letters)
    limit:    files per call (default 3)
    company:  filter to files whose name contains this string
    force:    re-scan already-processed files (default: skip them)
    """
    dirs = _scan_dirs(category)

    # Collect all .txt candidates (non-recursive for misc/root)
    candidates: list = []
    for d in dirs:
        if not d.exists():
            continue
        pattern = "*.txt" if d != RESUME_FOLDER else "*.txt"
        for f in sorted(d.glob("*.txt")):
            rel = str(f.relative_to(RESUME_FOLDER))
            candidates.append((rel, f))

    # De-duplicate (all category lists the same dir twice for misc)
    seen_rels: set = set()
    deduped = []
    for rel, f in candidates:
        if rel not in seen_rels:
            deduped.append((rel, f))
            seen_rels.add(rel)
    candidates = deduped

    # Company filter
    if company:
        cl = company.lower()
        candidates = [(rel, f) for rel, f in candidates if cl in rel.lower()]

    # Skip already-scanned unless force=True
    index   = _load_json(SCAN_INDEX_FILE, {"scanned": {}})
    scanned = index.get("scanned", {})
    if not force:
        candidates = [(rel, f) for rel, f in candidates if rel not in scanned]

    total_remaining = len(candidates)
    batch = candidates[:limit]

    if not batch:
        filter_note = f" (company filter: '{company}')" if company else ""
        return (
            f"All {category} files have been scanned{filter_note}.\n"
            "Use force=True to re-scan, change category, or add new files."
        )

    lines = [
        f"\u2550\u2550\u2550 MATERIAL SCAN \u2014 {category.upper()} \u2550\u2550\u2550",
        f"Returning {len(batch)} of {total_remaining} unscanned files.",
        "",
    ]

    for rel, path in batch:
        content = _read(path)
        lines += [
            "\u2500" * 60,
            f"FILE: {path.name}",
            "\u2500" * 60,
            content,
            "",
        ]
        scanned[rel] = datetime.datetime.now().isoformat()

    index["scanned"] = scanned
    _save_json(SCAN_INDEX_FILE, index)

    remaining_after = total_remaining - len(batch)
    lines += [
        "\u2550\u2550\u2550 EXTRACTION INSTRUCTIONS \u2550\u2550\u2550",
        "",
        "For each file above, extract:",
        "  1. TONE SAMPLES  \u2014 paragraphs that sound distinctly like Frank's voice.",
        "     Best candidates: opening paragraph, closing paragraph, any personal framing.",
        "     Call: log_tone_sample(text='...', source='Cover Letter <Company>')",
        "",
        "  2. PERSONAL STORIES \u2014 any specific anecdote, motivation, or non-resume detail.",
        "     Call: log_personal_story(story='...', tags=[...], title='...')",
        "",
        f"  {remaining_after} file(s) remaining in this category.",
        "  Call scan_materials_for_tone() again to continue.",
    ]

    return "\n".join(lines)


# ─── TOOLS: STAR STORY CONTEXT ────────────────────────────────────────────────

# Curated resume metrics keyed by story/skill theme
_STAR_METRICS: dict[str, list[str]] = {
    "testing": [
        "80%+ code coverage across JUnit, Mockito, Jest, and Selenium",
        "Sole developer — no QA team, so quality was self-enforced from the first commit",
        "TDD across full stack: Postgres → Spring Boot → Angular",
        "Zero production regressions attributed to test gaps during 4-year GM tenure",
    ],
    "quality": [
        "98% SLA compliance on production forecasting app used by senior GM leadership",
        "Self-enforced standards as the only dev — nowhere else to point if it broke",
        "Legacy codebase modernization (500K+ lines) with no service interruptions",
    ],
    "craftsmanship": [
        "Built-in quality, not bolted-on: TDD, clean migration paths, no shortcuts",
        "Java 8→21 + Spring Boot 2.2→3.5.4 while keeping prod healthy throughout",
        "Zero-downtime Oracle→PostgreSQL migration under live traffic",
    ],
    "solo-developer": [
        "Sole developer across 500K+ line codebase for 4 years",
        "Owned backend (Java/Spring Boot), frontend (Angular), database (PostgreSQL), and CI/CD",
        "No QA buffer — testing rigor was personal, not procedural",
        "98% SLA maintained throughout two major migrations and a modernization",
    ],
    "cloud": [
        "Led PCF → OCF → Azure Container Apps migration with zero downtime",
        "Oracle → PostgreSQL migration under live production traffic",
        "Terraform IaC for Azure provisioning",
        "98% SLA maintained throughout cloud transition period",
    ],
    "ai": [
        "Drove 35%+ GitHub Copilot/Claude adoption in engineering org (3.5x the target)",
        "Built AI-augmented workflows and coached peers on prompt engineering",
        "AI adoption recognized by leadership as exceptional contribution",
    ],
    "leadership": [
        "ERG JumpStart President — led without formal authority",
        "Angular Developer Group Admin — drove cross-team knowledge sharing",
        "3.5x AI adoption target through grassroots coaching, not mandate",
    ],
    "modernization": [
        "Java 8→21, Spring Boot 2.2→3.5.4 across 500K+ lines — no feature freeze",
        "Angular 6→18 migration with no regressions to business analysts",
        "Zero-downtime database migration: Oracle → PostgreSQL",
        "98% SLA held throughout all modernization phases",
    ],
    "ford": [
        "Grandfather spent 50 years at Ford — service manager at 19 during the Depression",
        "Grandfather story: 1934 Ford Fire Truck brass threads, machined to tolerances that looked stripped decades later",
        "Quality as inherited value, not process compliance — built in from the start",
    ],
}

_STAR_RELATED: dict[str, list[str]] = {
    "testing":       ["quality", "craftsmanship", "solo-developer"],
    "quality":       ["testing", "craftsmanship", "solo-developer"],
    "craftsmanship": ["quality", "testing", "ford"],
    "cloud":         ["solo-developer", "modernization"],
    "ai":            ["leadership"],
    "solo-developer":["testing", "quality", "modernization"],
    "leadership":    ["ai"],
    "modernization": ["cloud", "solo-developer"],
    "ford":          ["craftsmanship", "quality"],
    "grandfather":   ["ford", "craftsmanship", "quality"],
}

_COMPANY_FRAMING: dict[str, dict[str, str]] = {
    "ford": {
        "connection": "Grandfather's 50-year Ford career + 1934 Ford Fire Truck precision story",
        "values":     "Craftsmanship, durability, legacy, precision under constraint",
        "angle":      "Quality as an inherited value — built in from the Depression era forward",
    },
    "fanduel": {
        "values": "Scale, speed, uptime — real-time odds, millions of concurrent users",
        "angle":  "Testing rigor is what lets you ship fast without destroying trust",
    },
    "mercedes": {
        "values": "Zero defect, German engineering precision, no tolerance for corner-cutting",
        "angle":  "Self-enforced quality under resource constraint parallels the MB engineering ethos",
    },
    "airbnb": {
        "values": "Trust platform — guests and hosts depend on reliability",
        "angle":  "Solo ownership of uptime, because someone is always depending on it",
    },
    "reddit": {
        "values": "Scale, distributed systems, real-time feeds, developer culture",
        "angle":  "Ownership mentality — built like you're the one on-call when it pages",
    },
    "microsoft": {
        "values": "Engineering excellence, AI-first thinking, developer empowerment",
        "angle":  "AI adoption story (3.5x target) maps directly to Microsoft's Copilot ecosystem",
    },
    "google": {
        "values": "Scale, reliability, SRE culture, code quality",
        "angle":  "SLA obsession and testing rigor as cultural fit, not resume line item",
    },
}


@mcp.tool()
def get_star_story_context(
    tag: str,
    company: str = "",
    role_type: str = "",
) -> str:
    """
    Pull raw materials for a STAR behavioral answer.
    Returns matching personal stories + resume metrics relevant to the tag,
    plus optional company-specific framing hints.

    The LLM uses this context to write the actual STAR answer — this tool
    surfaces the ingredients; the AI does the synthesis.

    tag: the story/skill theme to match
         e.g. "testing" | "quality" | "solo-developer" | "cloud" | "ai"
              "leadership" | "craftsmanship" | "ford" | "grandfather"
    company: optional — triggers company-specific framing hints
             e.g. "ford" | "fanduel" | "mercedes" | "airbnb" | "reddit"
    role_type: optional context hint (not yet used for filtering, future use)

    After calling this, ask the AI to write the STAR answer using the returned context.
    """
    tag_lower = tag.lower().strip()

    # ── Personal stories: direct tag match + related tags ──
    story_data = _load_json(PERSONAL_CONTEXT_FILE, {"stories": []})
    all_stories = story_data.get("stories", [])

    related = _STAR_RELATED.get(tag_lower, [])
    search_tags = {tag_lower} | set(related)

    seen_ids: set = set()
    primary_stories, related_stories = [], []
    for s in all_stories:
        story_tags = set(s.get("tags", []))
        if s["id"] in seen_ids:
            continue
        if tag_lower in story_tags:
            primary_stories.append(s)
            seen_ids.add(s["id"])
        elif story_tags & search_tags:
            related_stories.append(s)
            seen_ids.add(s["id"])

    # ── Resume metrics: direct + related ──
    metrics: list[str] = []
    for t in [tag_lower] + related:
        for m in _STAR_METRICS.get(t, []):
            if m not in metrics:
                metrics.append(m)

    # ── Company framing ──
    company_lower = company.lower().strip()
    framing = None
    for key in _COMPANY_FRAMING:
        if key in company_lower:
            framing = _COMPANY_FRAMING[key]
            break

    # ── Build output ──
    header = f"tag='{tag}'"
    if company:
        header += f" | company='{company}'"
    if role_type:
        header += f" | role='{role_type}'"
    lines = [f"═══ STAR STORY CONTEXT: {header} ═══", ""]

    if primary_stories:
        lines.append(f"── PRIMARY STORIES ({len(primary_stories)} direct match) ──")
        for s in primary_stories:
            lines += [f"\n▪ #{s['id']} — {s['title']}",
                      f"  Tags: {', '.join(s.get('tags', []))}",
                      f"  {s['story']}", ""]

    if related_stories:
        lines.append(f"── RELATED STORIES ({len(related_stories)} via related tags) ──")
        for s in related_stories:
            lines += [f"\n▪ #{s['id']} — {s['title']}",
                      f"  Tags: {', '.join(s.get('tags', []))}",
                      f"  {s['story']}", ""]

    if not primary_stories and not related_stories:
        lines.append("No personal stories found for this tag or related tags.")
        lines.append("Log stories with log_personal_story() to enrich future STAR answers.")
        lines.append("")

    if metrics:
        lines.append("── RESUME METRICS TO WEAVE IN ──")
        for m in metrics:
            lines.append(f"  • {m}")
        lines.append("")

    if framing:
        lines.append(f"── {company.upper()} FRAMING HINTS ──")
        for k, v in framing.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines += [
        "── STAR STRUCTURE ──",
        "  Situation: Set the scene — team size, stack, constraints, stakes",
        "  Task:      What you owned and why it mattered",
        "  Action:    Specific decisions — what you built, how you tested, trade-offs made",
        "  Result:    Metrics first, then narrative payoff",
        "",
        "Use the personal stories for humanity. Use the metrics for credibility.",
        "The story is what makes it memorable. The numbers are what makes it land.",
    ]

    return "\n".join(lines)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
