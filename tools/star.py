from lib import config
from lib.io import _load_json


_STAR_METRICS: dict[str, list[str]] = {
    "testing": [
        "80%+ code coverage across JUnit, Mockito, Jest, and Selenium",
        "Sole developer — no QA team, so quality was self-enforced from the first commit",
        "TDD across full stack: Postgres → Spring Boot → Angular",
        "Zero production regressions attributed to test gaps during 4-year GM tenure",
    ],
    "quality": [
        "98% SLA compliance on production forecasting app used by senior GM leadership",
        "Self-enforced standards as the only dev — nowhere else to point if it broke",
        "Legacy codebase modernization (500K+ lines) with no service interruptions",
    ],
    "craftsmanship": [
        "Built-in quality, not bolted-on: TDD, clean migration paths, no shortcuts",
        "Java 8→21 + Spring Boot 2.2→3.5.4 while keeping prod healthy throughout",
        "Zero-downtime Oracle→PostgreSQL migration under live traffic",
    ],
    "solo-developer": [
        "Sole developer across 500K+ line codebase for 4 years",
        "Owned backend (Java/Spring Boot), frontend (Angular), database (PostgreSQL), and CI/CD",
        "No QA buffer — testing rigor was personal, not procedural",
        "98% SLA maintained throughout two major migrations and a modernization",
    ],
    "cloud": [
        "Led PCF → OCF → Azure Container Apps migration with zero downtime",
        "Oracle → PostgreSQL migration under live production traffic",
        "Terraform IaC for Azure provisioning",
        "98% SLA maintained throughout cloud transition period",
    ],
    "ai": [
        "Drove 35%+ GitHub Copilot/Claude adoption in engineering org (3.5x the target)",
        "Built AI-augmented workflows and coached peers on prompt engineering",
        "AI adoption recognized by leadership as exceptional contribution",
    ],
    "leadership": [
        "ERG JumpStart President — led without formal authority",
        "Angular Developer Group Admin — drove cross-team knowledge sharing",
        "3.5x AI adoption target through grassroots coaching, not mandate",
    ],
    "modernization": [
        "Java 8→21, Spring Boot 2.2→3.5.4 across 500K+ lines — no feature freeze",
        "Angular 6→18 migration with no regressions to business analysts",
        "Zero-downtime database migration: Oracle → PostgreSQL",
        "98% SLA held throughout all modernization phases",
    ],
    "ford": [
        "Grandfather spent 50 years at Ford — service manager at 19 during the Depression",
        "Grandfather story: 1934 Ford Fire Truck brass threads, machined to tolerances that looked stripped decades later",
        "Quality as inherited value, not process compliance — built in from the start",
    ],
}

_STAR_RELATED: dict[str, list[str]] = {
    "testing": ["quality", "craftsmanship", "solo-developer"],
    "quality": ["testing", "craftsmanship", "solo-developer"],
    "craftsmanship": ["quality", "testing", "ford"],
    "cloud": ["solo-developer", "modernization"],
    "ai": ["leadership"],
    "solo-developer": ["testing", "quality", "modernization"],
    "leadership": ["ai"],
    "modernization": ["cloud", "solo-developer"],
    "ford": ["craftsmanship", "quality"],
    "grandfather": ["ford", "craftsmanship", "quality"],
}

_COMPANY_FRAMING: dict[str, dict[str, str]] = {
    "ford": {
        "connection": "Grandfather's 50-year Ford career + 1934 Ford Fire Truck precision story",
        "values": "Craftsmanship, durability, legacy, precision under constraint",
        "angle": "Quality as an inherited value — built in from the Depression era forward",
    },
    "fanduel": {
        "values": "Scale, speed, uptime — real-time odds, millions of concurrent users",
        "angle": "Testing rigor is what lets you ship fast without destroying trust",
    },
    "mercedes": {
        "values": "Zero defect, German engineering precision, no tolerance for corner-cutting",
        "angle": "Self-enforced quality under resource constraint parallels the MB engineering ethos",
    },
    "airbnb": {
        "values": "Trust platform — guests and hosts depend on reliability",
        "angle": "Solo ownership of uptime, because someone is always depending on it",
    },
    "reddit": {
        "values": "Scale, distributed systems, real-time feeds, developer culture",
        "angle": "Ownership mentality — built like you're the one on-call when it pages",
    },
    "microsoft": {
        "values": "Engineering excellence, AI-first thinking, developer empowerment",
        "angle": "AI adoption story (3.5x target) maps directly to Microsoft's Copilot ecosystem",
    },
    "google": {
        "values": "Scale, reliability, SRE culture, code quality",
        "angle": "SLA obsession and testing rigor as cultural fit, not resume line item",
    },
}


def get_star_story_context(
    tag: str,
    company: str = "",
    role_type: str = "",
) -> str:
    tag_lower = tag.lower().strip()

    story_data = _load_json(config.PERSONAL_CONTEXT_FILE, {"stories": []})
    all_stories = story_data.get("stories", [])

    related = _STAR_RELATED.get(tag_lower, [])
    search_tags = {tag_lower} | set(related)

    seen_ids: set = set()
    primary_stories, related_stories = [], []
    for s in all_stories:
        story_tags = set(s.get("tags", []))
        if s["id"] in seen_ids:
            continue
        if tag_lower in story_tags:
            primary_stories.append(s)
            seen_ids.add(s["id"])
        elif story_tags & search_tags:
            related_stories.append(s)
            seen_ids.add(s["id"])

    metrics: list[str] = []
    for t in [tag_lower] + related:
        for m in _STAR_METRICS.get(t, []):
            if m not in metrics:
                metrics.append(m)

    company_lower = company.lower().strip()
    framing = None
    for key in _COMPANY_FRAMING:
        if key in company_lower:
            framing = _COMPANY_FRAMING[key]
            break

    header = f"tag='{tag}'"
    if company:
        header += f" | company='{company}'"
    if role_type:
        header += f" | role='{role_type}'"
    lines = [f"═══ STAR STORY CONTEXT: {header} ═══", ""]

    if primary_stories:
        lines.append(f"── PRIMARY STORIES ({len(primary_stories)} direct match) ──")
        for s in primary_stories:
            lines += [
                f"\n▪ #{s['id']} — {s['title']}",
                f"  Tags: {', '.join(s.get('tags', []))}",
                f"  {s['story']}",
                "",
            ]

    if related_stories:
        lines.append(f"── RELATED STORIES ({len(related_stories)} via related tags) ──")
        for s in related_stories:
            lines += [
                f"\n▪ #{s['id']} — {s['title']}",
                f"  Tags: {', '.join(s.get('tags', []))}",
                f"  {s['story']}",
                "",
            ]

    if not primary_stories and not related_stories:
        lines.append("No personal stories found for this tag or related tags.")
        lines.append("Log stories with log_personal_story() to enrich future STAR answers.")
        lines.append("")

    if metrics:
        lines.append("── RESUME METRICS TO WEAVE IN ──")
        for m in metrics:
            lines.append(f"  • {m}")
        lines.append("")

    if framing:
        lines.append(f"── {company.upper()} FRAMING HINTS ──")
        for k, v in framing.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines += [
        "── STAR STRUCTURE ──",
        "  Situation: Set the scene — team size, stack, constraints, stakes",
        "  Task:      What you owned and why it mattered",
        "  Action:    Specific decisions — what you built, how you tested, trade-offs made",
        "  Result:    Metrics first, then narrative payoff",
        "",
        "Use the personal stories for humanity. Use the metrics for credibility.",
        "The story is what makes it memorable. The numbers are what makes it land.",
    ]

    return "\n".join(lines)


def register(mcp) -> None:
    mcp.tool()(get_star_story_context)
