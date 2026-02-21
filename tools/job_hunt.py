from lib import config
from lib.io import _load_json, _save_json, _now


def get_job_hunt_status() -> str:
    data = _load_json(config.STATUS_FILE, {"applications": []})
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


def update_application(
    company: str,
    role: str,
    status: str,
    next_steps: str = "",
    contact: str = "",
    notes: str = "",
) -> str:
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
