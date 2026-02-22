import datetime

from lib import config
from lib.io import _load_json, _save_json
from lib.helpers import _build_checkin_entry


def get_daily_checkin_nudge() -> str:
    data = _load_json(config.HEALTH_LOG_FILE, {"entries": []})
    entries = data.get("entries", [])
    today = datetime.date.today().isoformat()

    if any(e.get("date") == today for e in entries):
        return ""

    if not entries:
        return (
            "⚠ No check-in logged yet today. "
            "Quick check-in: log_mental_health_checkin(mood='stable', energy=5)"
        )

    last_date = max((e.get("date", "") for e in entries), default="")
    return (
        f"⚠ No check-in logged yet today (last check-in: {last_date or 'unknown'}). "
        "Quick check-in: log_mental_health_checkin(mood='stable', energy=5)"
    )


def log_mental_health_checkin(
    mood: str,
    energy: int,
    notes: str = "",
    productive: bool = False,
) -> str:
    """Log a mental health check-in with mood label (e.g. 'good', 'anxious', 'hyperfocus', 'depressed'), energy level 1-10, optional notes, and whether the day felt productive. Returns a saved confirmation with personalized guidance based on the entry."""
    data = _load_json(config.HEALTH_LOG_FILE, {"entries": []})
    entry, guidance = _build_checkin_entry(mood, energy, notes, productive)
    data.setdefault("entries", []).append(entry)
    _save_json(config.HEALTH_LOG_FILE, data)
    return f"Check-in saved ({entry['date']}, energy {entry['energy']}/10, mood: {mood}).\n{guidance}"


def get_mental_health_log(days: int = 14) -> str:
    """Return recent mental health check-in history. Defaults to the last 14 days. Useful for tracking mood/energy trends during the job search."""
    data = _load_json(config.HEALTH_LOG_FILE, {"entries": []})
    entries = data.get("entries", [])

    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    recent = [e for e in entries if e.get("date", "") >= cutoff]

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

    if avg <= 4:
        lines.append("⚠  Trend: extended low-energy period. Consider reaching out for support.")
    elif avg >= 7:
        lines.append("✓  Trend: strong energy. Keep the momentum going.")

    return "\n".join(lines)


def register(mcp) -> None:
    mcp.tool()(log_mental_health_checkin)
    mcp.tool()(get_mental_health_log)
