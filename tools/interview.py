from lib import config
from lib.io import _read


def get_interview_quick_reference() -> str:
    """Return the full interview day quick reference: algorithm pattern cheat sheets, system design 5-step framework, testing talking points, and pre-interview checklist."""
    return _read(config.QUICK_REFERENCE)


def get_leetcode_cheatsheet(section: str = "") -> str:
    """Return the LeetCode algorithm cheatsheet. Pass a section name (e.g. 'trees', 'graphs', 'dynamic programming') to get just that section, or leave blank for the full 1400-line reference."""
    content = _read(config.LEETCODE_CHEATSHEET)
    if not section:
        return content

    lines = content.split("\n")
    result = []
    inside = False
    target = section.lower()

    for line in lines:
        stripped = line.lstrip("#").strip().lower()
        is_header = line.startswith("#")

        if is_header and target in stripped:
            inside = True
        elif is_header and inside and target not in stripped:
            if line.startswith("# ") or (line.startswith("## ") and target not in stripped):
                break

        if inside:
            result.append(line)

    if result:
        return "\n".join(result)
    return f"Section '{section}' not found. Returning full cheatsheet.\n\n{content}"


def generate_interview_prep_context(
    company: str,
    role: str,
    stage: str = "phone_screen",
    job_description: str = "",
) -> str:
    """Bundle Frank's master resume and quick reference into a structured context prompt for interview prep. Specify company, role, stage (phone_screen, technical, behavioral, system_design), and optional job description. Returns a prompt instructing the AI to generate top talking points, STAR responses, technical topics, smart questions, and confidence anchors."""
    master = _read(config.MASTER_RESUME)
    quick_ref = _read(config.QUICK_REFERENCE)

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


def get_existing_prep_file(company: str) -> str:
    """Find and return all existing interview prep files for a given company — searches across both the Resume 2025 and LeetCode folders for files containing the company name and prep/interview/call/assessment keywords."""
    matches = sorted(
        f
        for f in config.RESUME_FOLDER.rglob("*")
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


def register(mcp) -> None:
    mcp.tool()(get_interview_quick_reference)
    mcp.tool()(get_leetcode_cheatsheet)
    mcp.tool()(generate_interview_prep_context)
    mcp.tool()(get_existing_prep_file)
