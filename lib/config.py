import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_cfg: dict


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


def _load_config() -> dict:
    config_path = _HERE / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_path}\n"
            "Copy config.example.json → config.json and fill in your paths."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def _reconfigure(cfg: dict) -> None:
    global _cfg
    global RESUME_FOLDER, LEETCODE_FOLDER, SPICAM_FOLDER, DATA_FOLDER
    global STATUS_FILE, HEALTH_LOG_FILE, PERSONAL_CONTEXT_FILE, TONE_FILE, SCAN_INDEX_FILE
    global MASTER_RESUME, LEETCODE_CHEATSHEET, QUICK_REFERENCE

    _cfg = dict(cfg)

    RESUME_FOLDER = Path(cfg["resume_folder"])
    LEETCODE_FOLDER = Path(cfg["leetcode_folder"])
    SPICAM_FOLDER = Path(cfg["spicam_folder"])
    DATA_FOLDER = Path(cfg["data_folder"])

    STATUS_FILE = DATA_FOLDER / "status.json"
    HEALTH_LOG_FILE = DATA_FOLDER / "mental_health_log.json"
    PERSONAL_CONTEXT_FILE = DATA_FOLDER / "personal_context.json"
    TONE_FILE = DATA_FOLDER / "tone_samples.json"
    SCAN_INDEX_FILE = DATA_FOLDER / "scan_index.json"

    MASTER_RESUME = RESUME_FOLDER / cfg["master_resume_path"]
    LEETCODE_CHEATSHEET = LEETCODE_FOLDER / cfg["leetcode_cheatsheet_path"]
    QUICK_REFERENCE = LEETCODE_FOLDER / cfg["quick_reference_path"]


_reconfigure(_load_config())
