# Codex Task: Split server.py into lib/ + tools/ package

**Repo:** `https://github.com/JustLikeFrank3/job-search-mcp` — branch `v3`

Split `server.py` (1200+ lines, 23 MCP tools) into a proper package.
**No behavior changes. All 111 tests must still pass after.**

---

## Target Structure

```
job-search-mcp/
  server.py              ← entry point only: imports, mcp init, register calls, run()
  lib/
    __init__.py
    config.py            ← _load_config(), _reconfigure(), all Path globals
    io.py                ← _read(), _load_json(), _save_json(), _now()
    helpers.py           ← all pure _build_*/filter/format/_scan_dirs functions
  tools/
    __init__.py
    job_hunt.py          ← get_job_hunt_status, update_application
    resume.py            ← read_master_resume, list_existing_materials,
                            read_existing_resume, read_reference_file
    fitment.py           ← assess_job_fitment, get_customization_strategy
    interview.py         ← get_interview_quick_reference, get_leetcode_cheatsheet,
                            generate_interview_prep_context, get_existing_prep_file
    spicam.py            ← scan_spicam_for_skills
    health.py            ← log_mental_health_checkin, get_mental_health_log
    context.py           ← log_personal_story, get_personal_context
    tone.py              ← log_tone_sample, get_tone_profile, scan_materials_for_tone
    rag.py               ← search_materials, reindex_materials
    star.py              ← get_star_story_context (+ _STAR_METRICS,
                            _STAR_RELATED, _COMPANY_FRAMING dicts)
```

---

## Rules

### 1. `mcp` object + `register()` pattern

The `mcp` object lives in `server.py`. Each tool module exposes a `register(mcp)` function:

```python
# tools/context.py
from mcp.server.fastmcp import FastMCP  # only for type hint if needed

def log_personal_story(...) -> str:
    ...

def get_personal_context(...) -> str:
    ...

def register(mcp) -> None:
    mcp.tool()(log_personal_story)
    mcp.tool()(get_personal_context)
```

`server.py` calls every module's `register`:

```python
# server.py
from fastmcp import FastMCP
from lib.config import _reconfigure  # re-export for tests
import tools.job_hunt, tools.resume, tools.fitment, tools.interview
import tools.spicam, tools.health, tools.context, tools.tone, tools.rag, tools.star

mcp = FastMCP("job-search-as")

tools.job_hunt.register(mcp)
tools.resume.register(mcp)
# ... all modules

if __name__ == "__main__":
    mcp.run()
```

### 2. Path globals

All `Path` globals live only in `lib/config.py`. Tool modules import what they need:

```python
from lib.config import STATUS_FILE, RESUME_FOLDER, PERSONAL_CONTEXT_FILE  # etc.
```

### 3. Import hierarchy (no circular imports)

```
lib/config.py   ← no internal imports
lib/io.py       ← imports from lib/config
lib/helpers.py  ← imports from lib/config (for RESUME_FOLDER in _scan_dirs)
tools/*.py      ← imports from lib/config, lib/io, lib/helpers as needed
server.py       ← imports from lib/config (re-export), imports tools.*
```

`lib/config.py` must NOT import from `tools/`.

### 4. `_reconfigure()` must still work from `server` namespace

The test fixture does `server._reconfigure(fake_cfg)`. After the split:

```python
# server.py
from lib.config import _reconfigure  # makes server._reconfigure available
```

`_reconfigure()` in `lib/config.py` must use `global` and reassign every path global. Nothing changes about its behavior.

### 5. Tests call functions via `server.*` — preserve those names

Tests currently call e.g. `server.log_personal_story(...)`, `server.scan_materials_for_tone(...)`, etc.
After the split, re-export every function that tests call directly:

```python
# server.py — after all register() calls:
from tools.context import log_personal_story, get_personal_context
from tools.health  import log_mental_health_checkin, get_mental_health_log
from tools.tone    import log_tone_sample, get_tone_profile, scan_materials_for_tone
from tools.star    import get_star_story_context
from lib.helpers   import (
    _build_story_entry, _filter_stories, _format_story_list,
    _build_checkin_entry, _build_tone_sample_entry, _scan_dirs,
)
from lib.config    import RESUME_FOLDER  # _scan_dirs tests check this
```

To be safe, grep the test files for `server\.` to find every name that needs re-exporting.

### 6. Verify

```bash
python -m pytest tests/ -v
# Must show: 111 passed

python -m pytest tests/ --cov=server --cov-report=term-missing
```

If coverage drops below 50% because `source = ["server"]` no longer covers lib/tools,
update `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["server", "lib", "tools"]
```

Re-run and confirm `fail_under = 50` still passes.

---

## Current File Map (line numbers in server.py at commit 803a39f)

| Lines       | Contents                              | Destination         |
|-------------|---------------------------------------|---------------------|
| 1–29        | imports, mcp init                     | `server.py`         |
| 30–59       | `_load_config()`                      | `lib/config.py`     |
| 60–101      | `_reconfigure()` + Path globals       | `lib/config.py`     |
| 103–126     | `_read`, `_load_json`, `_save_json`, `_now` | `lib/io.py`   |
| 128–218     | pure helpers block                    | `lib/helpers.py`    |
| 219–309     | job hunt tools                        | `tools/job_hunt.py` |
| 310–378     | resume tools                          | `tools/resume.py`   |
| 380–454     | fitment/strategy tools                | `tools/fitment.py`  |
| 456–563     | interview/leetcode tools              | `tools/interview.py`|
| 565–671     | spicam scanner                        | `tools/spicam.py`   |
| 672–733     | health tools                          | `tools/health.py`   |
| 734–778     | personal context tools                | `tools/context.py`  |
| 779–891     | tone tools + scan_materials           | `tools/tone.py`     |
| 863–891     | RAG search/reindex                    | `tools/rag.py`      |
| 992–1205    | STAR dicts + get_star_story_context   | `tools/star.py`     |
| 1206+       | `mcp.run()` entry point               | `server.py`         |

---

## Commit

```
refactor: split server.py into lib/ + tools/ package (v3)
```

Push to `origin v3`.
