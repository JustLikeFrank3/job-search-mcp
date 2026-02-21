import datetime

from lib import config


def _build_story_entry(stories: list, story: str, tags: list, people: list, title: str) -> dict:
    return {
        "id": len(stories) + 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "title": title or (story[:60] + ("..." if len(story) > 60 else "")),
        "story": story,
        "tags": [t.lower().strip() for t in tags],
        "people": list(people),
    }


def _filter_stories(stories: list, tag: str = "", person: str = "") -> list:
    if tag:
        stories = [s for s in stories if tag.lower() in s.get("tags", [])]
    if person:
        stories = [
            s
            for s in stories
            if any(person.lower() in p.lower() for p in s.get("people", []))
        ]
    return stories


def _format_story_list(stories: list) -> str:
    lines = [f"═══ PERSONAL CONTEXT ({len(stories)} stories) ═══", ""]
    for s in stories:
        lines.append(f"▪ #{s['id']} — {s['title']}")
        lines.append(f"  Tags:   {', '.join(s.get('tags', []))}")
        if s.get("people"):
            lines.append(f"  People: {', '.join(s['people'])}")
        lines.append(f"  {s['story']}")
        lines.append("")
    return "\n".join(lines)


def _build_checkin_entry(mood: str, energy: int, notes: str, productive: bool) -> tuple:
    energy_int = max(1, min(10, int(energy)))
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "date": datetime.date.today().isoformat(),
        "mood": mood,
        "energy": energy_int,
        "productive": bool(productive),
        "notes": notes,
    }
    if energy_int <= 3 or mood in ("depressed", "low"):
        guidance = (
            "Low energy logged. Small wins count — "
            "even one LeetCode problem or one email sent is real progress. "
            "You're still moving, even on hard days."
        )
    elif mood == "hyperfocus" or energy_int >= 8:
        guidance = (
            "High energy logged. Good time for deep work. "
            "Just remember to eat, hydrate, and step away before burnout hits."
        )
    else:
        guidance = "Logged. You're doing the work, even when it's hard."
    return entry, guidance


def _build_tone_sample_entry(samples: list, text: str, source: str, context: str) -> dict:
    return {
        "id": len(samples) + 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "source": source,
        "context": context,
        "text": text,
        "word_count": len(text.split()),
    }


def _scan_dirs(category: str) -> list:
    mapping: dict = {
        "cover_letters": [config.RESUME_FOLDER / "02-Cover-Letters"],
        "resumes": [config.RESUME_FOLDER / "01-Current-Optimized"],
        "misc": [config.RESUME_FOLDER],
        "all": [
            config.RESUME_FOLDER / "02-Cover-Letters",
            config.RESUME_FOLDER / "01-Current-Optimized",
            config.RESUME_FOLDER,
        ],
    }
    return mapping.get(category.lower(), [config.RESUME_FOLDER / "02-Cover-Letters"])
