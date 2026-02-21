# Bug Report: VS Code MCP Tool "Disabled by User" Despite Being Toggled On

## Summary

A custom FastMCP stdio server is correctly configured, starts cleanly, and responds to
JSON-RPC requests — but VS Code Copilot Chat refuses to call any of its tools, returning:

```
ERROR: Tool mcp_job-search-as_get_job_hunt_status is currently disabled by the user,
and cannot be called.
```

This happens even though every tool is **visually toggled ON** in the Copilot chat tools
panel. The error persists across window reloads, server restarts, and VS Code restarts.

---

## Environment

| Item | Value |
|---|---|
| OS | macOS (Apple Silicon arm64) |
| VS Code | 1.109.5 (commit `072586267e`) |
| Python | 3.14.2 |
| MCP library | `mcp==1.26.0` (FastMCP) |
| Server type | `stdio` |
| Server config location | `~/Library/Application Support/Code/User/mcp.json` |

---

## Server Configuration

**`~/Library/Application Support/Code/User/mcp.json`**
```jsonc
{
    "servers": {
        "job-search-as": {
            "type": "stdio",
            "command": "/Users/fvm3/Projects/job-search-mcp/.venv/bin/python",
            "args": [
                "/Users/fvm3/Projects/job-search-mcp/server.py"
            ]
        }
    }
}
```

---

## Server Code (relevant excerpt)

**`/Users/fvm3/Projects/job-search-mcp/server.py`**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "job-search-assistant",
    instructions="..."
)

@mcp.tool()
def get_job_hunt_status() -> str:
    """Returns the current status of all tracked job applications..."""
    ...

# 15 more tools decorated with @mcp.tool()

if __name__ == "__main__":
    mcp.run()
```

All 16 tools use `@mcp.tool()` with no extra kwargs.

---

## What Works

1. **Server loads clean** — no import errors, no startup exceptions:
   ```
   $ python -c "import server; print('OK')"
   OK
   ```

2. **Server responds to stdin** — when given a raw JSON-RPC `tools/list` request:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
     | .venv/bin/python server.py 2>&1
   ```
   Output:
   ```
   WARNING  Failed to validate request: Received request before initialization
   {"jsonrpc":"2.0","id":1,"error":{"code":-32602,...}}
   ```
   *(Expected — this is the correct protocol rejection of a pre-init request.)*

3. **VS Code shows the server and tools** in the Copilot chat tools panel.

4. **All tools are checked ON** in the UI — verified multiple times.

---

## What Fails

Every tool call from Copilot Chat returns:
```
ERROR while calling tool: Tool mcp_job-search-as_<tool_name> is currently disabled
by the user, and cannot be called.
```

This has been reproduced with:
- `get_job_hunt_status` (no parameters)
- `read_master_resume` (no parameters)
- `assess_job_fitment` (with parameters)
- Every other tool in the server

---

## What Was Already Tried

- ✅ Toggling each tool checkbox off and back on in the Copilot panel
- ✅ Restarting the MCP server via the gear icon in the Copilot panel
- ✅ `Developer: Reload Window`
- ✅ Full VS Code quit + relaunch
- ✅ Verified `mcp.json` is at the correct path and parses as valid JSON
- ✅ Confirmed the venv Python path exists and is executable
- ✅ Confirmed `server.py` is at the specified path
- ✅ Checked `settings.json` — no `chat.mcp.*` disable overrides found
- ✅ Searched `globalStorage/github.copilot-chat/` — no persisted disable state found
- ✅ Searched `workspaceStorage/` — no disable state found

---

## Suspected Root Cause

The VS Code Copilot Chat extension appears to store per-tool "disabled" state somewhere
that is **not** reset by the UI toggle, window reload, or server restart.

Possible storage locations not yet found:
- A binary/non-JSON storage file in `globalStorage/`
- An in-memory state that is initialized from an unknown source on extension activation
- A workspace-level override in a `.vscode/` file (none present in this workspace)
- A race condition where the server's tool discovery response arrives after the "disabled"
  state is applied from cache

---

## Hypothesis to Investigate

1. **Does nuking the Copilot Chat extension's global storage fix it?**
   Path: `~/Library/Application Support/Code/User/globalStorage/github.copilot-chat/`
   *(Risky — may lose chat history)*

2. **Is there a `chat.mcp.tools.disabled` or similar setting that the UI writes to
   `settings.json` in a namespace not searched yet?**

3. **Does the FastMCP `name` parameter affect tool registration?**
   The server is named `"job-search-assistant"` but registered in `mcp.json` as
   `"job-search-as"`. Could a mismatch between the server's self-reported name and the
   config key cause tools to be silently rejected?

4. **Does FastMCP's `mcp.run()` vs `mcp.run(transport="stdio")` matter here?**

---

## Task for Codex

**Primary goal:** Fix the issue so `get_job_hunt_status` and other tools can be called
from Copilot Chat in VS Code 1.109.5.

**Approach options (in order of preference):**

1. **Diagnose the VS Code storage** — find where the disable state is persisted and
   clear/reset it programmatically without losing extension data.

2. **Fix the server** — if the `name` mismatch (`"job-search-assistant"` vs
   `"job-search-as"`) or any other server-side issue is causing VS Code to reject tools,
   fix it in `server.py`.

3. **Workaround** — if this is a VS Code bug, provide a reliable workaround (e.g. a
   different transport, a wrapper script, or a config change) that makes the tools
   callable.

---

## File Locations

| File | Path |
|---|---|
| `mcp.json` | `~/Library/Application Support/Code/User/mcp.json` |
| `server.py` | `/Users/fvm3/Projects/job-search-mcp/server.py` |
| `config.json` | `/Users/fvm3/Projects/job-search-mcp/config.json` |
| Copilot global storage | `~/Library/Application Support/Code/User/globalStorage/github.copilot-chat/` |
| VS Code user settings | `~/Library/Application Support/Code/User/settings.json` |
