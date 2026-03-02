"""
Session startup tool — v1

get_session_context()
    Returns master resume + tone profile (all samples) + STAR context (stories,
    metrics, company framing) + job hunt status + people/networking log in a
    single call. MUST be the first tool called in every session.
    No exceptions. This is the entire point of the system.
"""

from lib import config
from lib.io import _load_master_context
from tools.tone import get_tone_profile
from tools.job_hunt import get_job_hunt_status
from tools.people import get_people
from tools.star import get_all_star_context


def get_session_context() -> str:
    """
    SESSION STARTUP — call this first, before anything else, every single session.

    Returns Frank's complete context in one shot:
      1. Master resume with all metrics, projects, and personal notes
      2. Tone profile — all writing samples. Do not write a single word for him without this.
      3. STAR context — all personal stories, anecdotes, metric bullets by category, and company framing hints
      4. Live job hunt pipeline — current applications and next steps
      5. People & networking log — every contact, referral, and relationship

    This exists so Frank never has to recontextualize. Honor that.
    """
    _display_name = config._cfg.get("contact", {}).get("name", "User")
    sections = [
        "═" * 60,
        f"SESSION CONTEXT — {_display_name}",
        "═" * 60,
        "",
        "── 1. MASTER RESUME ──────────────────────────────────────",
        "",
        _load_master_context(),
        "",
        "── 2. TONE PROFILE (all samples) ─────────────────────────",
        "",
        get_tone_profile(),
        "",
        "── 3. STAR CONTEXT (stories + metrics + framing) ─────────",
        "",
        get_all_star_context(),
        "",
        "── 4. JOB HUNT STATUS ────────────────────────────────────",
        "",
        get_job_hunt_status(),
        "",
        "── 5. PEOPLE & NETWORKING ────────────────────────────────",
        "",
        get_people(),
        "",
        "═" * 60,
        "You are now fully contextualized. Proceed.",
        "═" * 60,
    ]
    return "\n".join(sections)


def register(mcp) -> None:
    mcp.tool()(get_session_context)
