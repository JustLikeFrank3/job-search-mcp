import os
from pathlib import Path

from lib import config


def scan_spicam_for_skills() -> str:
    """Scan the RetrosPiCam project directory and detect technologies used (FastAPI, React Native, Azure Blob Storage, Raspberry Pi, etc.). Reports newly detected skills not yet on the master resume so they can be added."""
    if not config.SPICAM_FOLDER.exists():
        return f"RetrosPiCam folder not found at: {config.SPICAM_FOLDER}"

    tech_found: set[str] = set()
    file_inventory: list[str] = []

    skip_dirs = {".git", "__pycache__", "node_modules", "venv", ".venv", "env", ".expo", "build", "dist"}

    for root, dirs, files in os.walk(config.SPICAM_FOLDER):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

        for fname in files:
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(config.SPICAM_FOLDER))
            ext = fpath.suffix.lower()
            file_inventory.append(rel)

            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue

            if ext == ".py":
                tech_found.add("Python")
                if "fastapi" in text:
                    tech_found.add("FastAPI")
                if "pydantic" in text:
                    tech_found.add("Pydantic")
                if "azure" in text:
                    tech_found.add("Azure Blob Storage")
                if "async def" in text:
                    tech_found.add("Python async/await")
                if "websocket" in text:
                    tech_found.add("WebSockets")
                if "pytest" in text:
                    tech_found.add("pytest")
                if "systemd" in text:
                    tech_found.add("systemd / Linux services")
                if "gpio" in text or "rpi" in text:
                    tech_found.add("Raspberry Pi GPIO")
                if "servo" in text or "adafruit" in text:
                    tech_found.add("Servo / Adafruit HAT")
                if "range" in text and "http" in text:
                    tech_found.add("HTTP Range requests")
                if "jwt" in text or "bearer" in text:
                    tech_found.add("JWT authentication")
                if "docker" in text:
                    tech_found.add("Docker")
                if "retention" in text:
                    tech_found.add("Automated retention policies")
            elif ext in (".ts", ".tsx", ".js", ".jsx"):
                tech_found.add("TypeScript/JavaScript")
                if "react-native" in text or "react native" in text:
                    tech_found.add("React Native")
                if "expo" in text:
                    tech_found.add("Expo")
                if "testflight" in text:
                    tech_found.add("iOS TestFlight deployment")
            elif ext == ".swift":
                tech_found.add("Swift / iOS")
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

    lines += ["", "── New Skills Not Yet on Master Resume ──"]
    lines += [f"  • {s}" for s in new_skills] or ["  (none — master resume is up to date)"]

    lines += [
        "",
        "── Suggested Resume Bullets ──",
        "  • Built production IoT camera system integrating Raspberry Pi hardware, "
        "    servo HAT, Python/FastAPI backend, and React Native mobile app",
    ]
    if "Pydantic" in tech_found:
        lines.append("  • Designed type-safe API layer with FastAPI + Pydantic models")
    if "Python async/await" in tech_found:
        lines.append("  • Implemented async Python services for concurrent hardware + network I/O")
    if "Azure Blob Storage" in tech_found:
        lines.append("  • Integrated Azure Blob Storage with automated 7-day retention management")
    if "HTTP Range requests" in tech_found:
        lines.append("  • Enabled on-demand video streaming via HTTP Range request support")
    if "systemd / Linux services" in tech_found:
        lines.append("  • Configured systemd service for reliable auto-start on embedded Linux")
    if "WebSockets" in tech_found:
        lines.append("  • Delivered real-time camera stream via WebSocket connections")
    if "JWT authentication" in tech_found:
        lines.append("  • Secured API endpoints with JWT bearer-token authentication")

    return "\n".join(lines)


def register(mcp) -> None:
    mcp.tool()(scan_spicam_for_skills)
