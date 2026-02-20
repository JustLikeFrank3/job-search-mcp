#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) module for job-search-mcp.
Indexes all resume, prep, and LeetCode materials into ChromaDB
and provides semantic search via OpenAI embeddings.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import chromadb
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent


def _load_config() -> dict:
    return json.loads((_HERE / "config.json").read_text(encoding="utf-8"))


_cfg        = _load_config()
_CHROMA_DIR = Path(_cfg["data_folder"]) / "chroma"
_OPENAI_KEY = _cfg.get("openai_api_key", "")

CATEGORIES = {
    "resume":        "Resume bullets, achievements, STAR stories, technical skills",
    "cover_letters": "Cover letters tailored to specific companies and roles",
    "leetcode":      "Algorithm patterns, code templates, data structures, complexity analysis",
    "interview_prep":"Company-specific interview prep, behavioral questions, talking points",
    "reference":     "Reference materials, awards, feedback, template guides",
}

# ─── CLIENTS ──────────────────────────────────────────────────────────────────

def _chroma_client() -> chromadb.PersistentClient:
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_CHROMA_DIR))


def _openai_client() -> OpenAI:
    if not _OPENAI_KEY:
        raise ValueError(
            "openai_api_key not set in config.json. "
            "Add it to use RAG search."
        )
    return OpenAI(api_key=_OPENAI_KEY)


# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries."""
    # First split by double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If a single paragraph exceeds max_chars, split by sentences
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) > max_chars and current:
                    chunks.append(current.strip())
                    # Keep overlap from end of last chunk
                    current = current[-overlap:] + " " + sent
                else:
                    current += " " + sent
        else:
            if len(current) + len(para) > max_chars and current:
                chunks.append(current.strip())
                current = current[-overlap:] + "\n\n" + para
            else:
                current += "\n\n" + para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 50]  # filter trivial chunks


def _doc_id(source: str, index: int) -> str:
    return hashlib.md5(f"{source}:{index}".encode()).hexdigest()


# ─── INDEXING ─────────────────────────────────────────────────────────────────

def _embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed a batch of texts using text-embedding-3-small."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def _index_files(
    collection: chromadb.Collection,
    files: list[Path],
    category: str,
    oai: OpenAI,
    label: str = "",
) -> int:
    """Chunk, embed, and upsert a list of files into a collection."""
    all_docs, all_ids, all_metas = [], [], []

    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not text:
            continue

        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_docs.append(chunk)
            all_ids.append(_doc_id(str(fpath), i))
            all_metas.append({
                "source":   fpath.name,
                "category": category,
                "chunk":    i,
            })

    if not all_docs:
        return 0

    # Embed in batches of 100 (OpenAI limit)
    embeddings = []
    for i in range(0, len(all_docs), 100):
        batch = all_docs[i:i + 100]
        embeddings.extend(_embed(batch, oai))

    collection.upsert(
        ids=all_ids,
        documents=all_docs,
        embeddings=embeddings,
        metadatas=all_metas,
    )

    print(f"  ✓ {label or category}: {len(files)} files → {len(all_docs)} chunks")
    return len(all_docs)


def build_index(verbose: bool = True) -> dict[str, int]:
    """
    Index all job search materials into ChromaDB.
    Safe to re-run — upserts by content hash so unchanged chunks are skipped.
    Returns chunk counts per category.
    """
    cfg    = _load_config()
    oai    = _openai_client()
    client = _chroma_client()

    collection = client.get_or_create_collection(
        name="job_search",
        metadata={"hnsw:space": "cosine"},
    )

    resume_folder   = Path(cfg["resume_folder"])
    leetcode_folder = Path(cfg["leetcode_folder"])
    counts: dict[str, int] = {}

    if verbose:
        print("Building RAG index...")

    # ── Master resume ──────────────────────────────────────────────────────────
    master = resume_folder / cfg["master_resume_path"]
    if master.exists():
        counts["resume"] = _index_files(
            collection, [master], "resume", oai, "Master resume"
        )

    # ── All resumes ────────────────────────────────────────────────────────────
    optimized_dir = resume_folder / cfg["optimized_resumes_dir"]
    if optimized_dir.exists():
        resume_files = [
            f for f in optimized_dir.glob("*.txt")
            if "MASTER" not in f.name
        ]
        counts["resume"] = counts.get("resume", 0) + _index_files(
            collection, resume_files, "resume", oai, f"Resumes ({len(resume_files)} files)"
        )

    # ── Cover letters ──────────────────────────────────────────────────────────
    cl_dir = resume_folder / cfg["cover_letters_dir"]
    if cl_dir.exists():
        cl_files = list(cl_dir.glob("*.txt"))
        counts["cover_letters"] = _index_files(
            collection, cl_files, "cover_letters", oai, f"Cover letters ({len(cl_files)} files)"
        )

    # ── Reference materials ────────────────────────────────────────────────────
    ref_dir = resume_folder / cfg["reference_materials_dir"]
    if ref_dir.exists():
        ref_files = list(ref_dir.glob("*.txt"))
        counts["reference"] = _index_files(
            collection, ref_files, "reference", oai, f"Reference materials ({len(ref_files)} files)"
        )

    # ── Interview prep files (top-level .txt in Resume folder) ────────────────
    prep_files = [
        f for f in resume_folder.glob("*.txt")
        if any(kw in f.name.lower() for kw in ("prep", "interview", "call", "cheat"))
    ]
    counts["interview_prep"] = _index_files(
        collection, prep_files, "interview_prep", oai,
        f"Interview prep ({len(prep_files)} files)"
    )

    # ── LeetCode cheatsheet + quick reference ─────────────────────────────────
    lc_files = [
        p for name in (cfg["leetcode_cheatsheet_path"], cfg["quick_reference_path"])
        if (p := leetcode_folder / name).exists()
    ]
    counts["leetcode"] = _index_files(
        collection, lc_files, "leetcode", oai,
        f"LeetCode ({len(lc_files)} files)"
    )

    total = sum(counts.values())
    if verbose:
        print(f"\nIndex complete. {total} total chunks across {len(counts)} categories.")

    return counts


# ─── SEARCH ───────────────────────────────────────────────────────────────────

def search(
    query: str,
    category: Optional[str] = None,
    n_results: int = 6,
) -> list[dict]:
    """
    Semantic search across indexed materials.
    Returns list of {text, source, category, score} dicts.
    """
    oai    = _openai_client()
    client = _chroma_client()

    collection = client.get_or_create_collection(
        name="job_search",
        metadata={"hnsw:space": "cosine"},
    )

    query_embedding = _embed([query], oai)[0]

    where = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "category": meta.get("category", "unknown"),
            "score":    round(1 - dist, 3),  # cosine similarity
        })

    return hits


def format_results(hits: list[dict], header: str = "Search Results") -> str:
    """Format search results for display in MCP tool output."""
    if not hits:
        return "No relevant results found."

    lines = [f"═══ {header} ═══", ""]
    for i, hit in enumerate(hits, 1):
        lines.append(f"[{i}] {hit['source']} (score: {hit['score']}, category: {hit['category']})")
        lines.append(hit["text"])
        lines.append("")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:])
        if not query:
            print("Usage: python rag.py search <query>")
            sys.exit(1)
        hits = search(query)
        print(format_results(hits, f"Results for: {query}"))
    else:
        build_index()
