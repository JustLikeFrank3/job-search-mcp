from lib import config
from lib.io import _load_json, _save_json
from lib.helpers import _build_story_entry, _filter_stories, _format_story_list


def log_personal_story(
    story: str,
    tags: list[str],
    people: list[str] = [],
    title: str = "",
) -> str:
    data = _load_json(config.PERSONAL_CONTEXT_FILE, {"stories": []})
    entry = _build_story_entry(data["stories"], story, tags, people, title)
    data["stories"].append(entry)
    _save_json(config.PERSONAL_CONTEXT_FILE, data)
    return f"✓ Story logged (#{entry['id']}): {entry['title']}"


def get_personal_context(tag: str = "", person: str = "") -> str:
    data = _load_json(config.PERSONAL_CONTEXT_FILE, {"stories": []})
    stories = _filter_stories(data.get("stories", []), tag, person)

    if not stories:
        qualifier = f" for tag '{tag}'" if tag else ""
        qualifier += f" for person '{person}'" if person else ""
        return f"No personal stories found{qualifier}."

    return _format_story_list(stories)


def register(mcp) -> None:
    mcp.tool()(log_personal_story)
    mcp.tool()(get_personal_context)
