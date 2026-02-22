from pathlib import Path

from lib import config
from lib.io import _read


def read_master_resume() -> str:
    """Read Frank's master source resume — the single source of truth containing all metrics, achievements, projects, and context notes. Always read this before generating any resume or cover letter."""
    return _read(config.MASTER_RESUME)


def list_existing_materials(company: str = "") -> str:
    """List all existing resume and cover letter files. Optionally filter by company name to see materials for a specific target company."""
    optimized_dir = config.RESUME_FOLDER / config._cfg["optimized_resumes_dir"]
    cover_letter_dir = config.RESUME_FOLDER / config._cfg["cover_letters_dir"]

    def _list_dir(d: Path, label: str) -> list[str]:
        if not d.exists():
            return [f"  (folder not found: {d.name})"]
        files = sorted(
            f.name
            for f in d.iterdir()
            if f.suffix in (".txt", ".md")
            and "MASTER" not in f.name
            and (not company or company.lower() in f.name.lower())
        )
        out = [f"\n══ {label} ({len(files)}) ══"]
        out += [f"  {f}" for f in files] or ["  (none found)"]
        return out

    lines = []
    lines += _list_dir(optimized_dir, "RESUMES")
    lines += _list_dir(cover_letter_dir, "COVER LETTERS")
    return "\n".join(lines)


def read_existing_resume(filename: str) -> str:
    """Read the full text of an existing resume from 01-Current-Optimized/. Use list_existing_materials() to find available filenames."""
    path = config.RESUME_FOLDER / config._cfg["optimized_resumes_dir"] / filename
    if not path.exists():
        return f"Not found: {filename}\nUse list_existing_materials() to see available resumes."
    return _read(path)


def read_reference_file(filename: str) -> str:
    """Read a file from 06-Reference-Materials/ (e.g. template format, consolidated resume, skills variants, GM feedback). Pass the filename only — use list_existing_materials() to discover what's available."""
    ref_dir = config.RESUME_FOLDER / config._cfg["reference_materials_dir"]
    path = ref_dir / filename
    if not path.exists():
        available = sorted(f.name for f in ref_dir.iterdir()) if ref_dir.exists() else []
        return f"Not found: {filename}\nAvailable: {available}"
    return _read(path)


def register(mcp) -> None:
    mcp.tool()(read_master_resume)
    mcp.tool()(list_existing_materials)
    mcp.tool()(read_existing_resume)
    mcp.tool()(read_reference_file)
