
def search_materials(query: str, category: str = "") -> str:
    """Semantic search across all job search materials (resumes, cover letters, interview prep, LeetCode notes, reference files). Optionally filter by category. Requires OpenAI API key and a built index — run reindex_materials() first if needed."""
    try:
        from rag import search, format_results

        hits = search(query, category=category or None, n_results=6)
        return format_results(hits, f'Results for: "{query}"')
    except Exception as e:
        if "openai_api_key" in str(e).lower() or "not set" in str(e).lower():
            return (
                "RAG search requires an OpenAI API key.\n"
                "Add 'openai_api_key' to config.json, then run reindex_materials()."
            )
        return f"Search error: {e}\nTry running reindex_materials() first."


def reindex_materials() -> str:
    """Rebuild the semantic search index over all job search materials. Run this after adding new resumes, cover letters, or prep files. Requires an OpenAI API key set in config.json."""
    try:
        from rag import build_index

        counts = build_index(verbose=False)
        total = sum(counts.values())
        lines = [f"✓ Index built. {total} total chunks indexed:", ""]
        for cat, count in counts.items():
            lines.append(f"  {cat:<16} {count} chunks")
        lines += ["", "You can now use search_materials() for semantic search."]
        return "\n".join(lines)
    except Exception as e:
        if "openai_api_key" in str(e).lower() or "not set" in str(e).lower():
            return (
                "Indexing requires an OpenAI API key.\n"
                "Add 'openai_api_key' to config.json and try again."
            )
        return f"Indexing error: {e}"


def register(mcp) -> None:
    mcp.tool()(search_materials)
    mcp.tool()(reindex_materials)
