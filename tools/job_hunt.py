from lib import config
from lib.io import _load_json, _save_json, _now
from tools.health import get_daily_checkin_nudge


def get_job_hunt_status() -> str:
    """Return the current job application pipeline: all tracked companies, roles, statuses, next steps, and contacts. Also nudges a daily mental health check-in if none has been logged today."""
    data = _load_json(config.STATUS_FILE, {"applications": []})
    apps = data.get("applications", [])
    nudge = get_daily_checkin_nudge()

    if not apps:
        base = "No applications tracked yet. Use update_application() to add one."
        return f"{base}\n\n{nudge}" if nudge else base

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

    if nudge:
        lines += ["", nudge]

    return "\n".join(lines)


def update_application(
    company: str,
    role: str,
    status: str,
    next_steps: str = "",
    contact: str = "",
    notes: str = "",
) -> str:
    """Add or update a job application in the pipeline tracker. Pass company, role, and current status (e.g. 'applied', 'phone screen', 'offer'). Optionally include next_steps, contact name, and free-form notes."""
    data = _load_json(config.STATUS_FILE, {"applications": []})
    apps: list = data.setdefault("applications", [])

    existing = next(
        (
            a
            for a in apps
            if a["company"].lower() == company.lower()
            and a["role"].lower() == role.lower()
        ),
        None,
    )
    if existing is None:
        existing = next((a for a in apps if a["company"].lower() == company.lower()), None)

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
    _save_json(config.STATUS_FILE, data)
    return f"✓ {action}: {company} — {role} ({status})"


def register(mcp) -> None:
    mcp.tool()(get_job_hunt_status)
    mcp.tool()(update_application)
