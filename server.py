"""
PaperPilot AI
-------------
MCP-Powered Semantic Research Platform

Exposes tools that let an MCP client search arXiv,
retrieve papers, perform semantic search, and generate
grounded answers using RAG.
"""

import arxiv
import httpx
import pypdf
import io
import os
import sqlite3
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from fastmcp import FastMCP

load_dotenv()  # reads variables from a .env file in this folder into os.environ

mcp = FastMCP("PaperPilot AI")

# Small, fast embedding model — good enough for a resume project and
# runs on CPU without any API key. Loaded once at server startup.
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# Requires GROQ_API_KEY in a .env file next to this script (see .env.example).
# Get a free key at https://console.groq.com/keys
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not found! Add it to a .env file next to server.py.")

# --- Persistence layer (SQLite) -------------------------------------------
# Fetched papers survive a server restart instead of vanishing from memory.
DB_PATH = os.path.join(os.path.dirname(__file__), "papers.db")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            chunks TEXT,       -- JSON list of strings
            embeddings BLOB,   -- raw float32 bytes
            embed_shape TEXT   -- "rows,cols" to reconstruct the array
        )
    """)
    conn.commit()
    conn.close()


def save_paper(paper_id: str, title: str, chunks: list[str], embeddings: np.ndarray) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?)",
        (
            paper_id,
            title,
            json.dumps(chunks),
            embeddings.astype(np.float32).tobytes(),
            f"{embeddings.shape[0]},{embeddings.shape[1]}",
        ),
    )
    conn.commit()
    conn.close()


def load_paper(paper_id: str) -> tuple[list[str], np.ndarray] | None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT chunks, embeddings, embed_shape FROM papers WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    chunks = json.loads(row[0])
    rows, cols = map(int, row[2].split(","))
    embeddings = np.frombuffer(row[1], dtype=np.float32).reshape(rows, cols)
    return chunks, embeddings


def list_all_papers() -> list[tuple[str, str]]:
    """Return (paper_id, title) for every paper ever fetched, from the DB."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT paper_id, title FROM papers ORDER BY rowid DESC").fetchall()
    conn.close()
    return rows


init_db()

# In-memory cache as a fast layer on top of SQLite: paper_id -> (chunks, embeddings).
# Checked first; falls back to the database if not present (e.g. after a restart).
PAPER_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks so semantic meaning isn't cut
    off mid-sentence at chunk boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def _search_arxiv_raw(query: str, max_results: int = 5) -> list[dict]:
    """Core search logic, returning structured data. Used by both the
    search_arxiv MCP tool (which formats it as text) and the demo UI
    (which renders it directly) — single source of truth."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [
        {
            "id": r.get_short_id(),
            "title": r.title,
            "authors": [a.name for a in r.authors[:3]],
            "published": str(r.published.date()),
            "summary": r.summary,
        }
        for r in client.results(search)
    ]


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers matching a query.

    Args:
        query: Topic or keywords to search for, e.g. "retrieval augmented generation"
        max_results: How many results to return (default 5)
    """
    try:
        papers = _search_arxiv_raw(query, max_results)
        if not papers:
            return "No results found for that query."
        return "\n---\n".join(
            f"ID: {p['id']}\n"
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Published: {p['published']}\n"
            f"Summary: {p['summary'][:300]}...\n"
            for p in papers
        )
    except Exception as e:
        return f"Search failed: {e}. Check your internet connection and try a simpler query."


@mcp.tool()
def fetch_paper(paper_id: str) -> str:
    """Download a paper's PDF by arXiv ID and extract its text into memory
    so it can be queried later with ask_paper.

    Args:
        paper_id: arXiv ID, e.g. "2005.11401"
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        try:
            paper = next(client.results(search))
        except StopIteration:
            return f"No paper found with ID '{paper_id}'. Double-check the ID from search_arxiv."

        resp = httpx.get(paper.pdf_url, timeout=30)
        resp.raise_for_status()

        reader = pypdf.PdfReader(io.BytesIO(resp.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return f"Downloaded '{paper.title}' but couldn't extract any text (the PDF may be scanned images)."

        chunks = chunk_text(text)
        embeddings = EMBEDDER.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

        PAPER_CACHE[paper_id] = (chunks, embeddings)
        save_paper(paper_id, paper.title, chunks, embeddings)

        return (
            f"Fetched '{paper.title}' ({len(text)} characters, {len(chunks)} chunks embedded). "
            f"You can now use ask_paper on ID {paper_id}."
        )
    except httpx.HTTPError as e:
        return f"Couldn't download the PDF: {e}. Try again in a moment."
    except Exception as e:
        return f"Failed to fetch paper {paper_id}: {e}"


def _get_cached_paper(paper_id: str) -> tuple[list[str], np.ndarray] | None:
    """Look up a paper's chunks+embeddings, checking memory first then
    falling back to the DB (covers the case of a server restart)."""
    cached = PAPER_CACHE.get(paper_id)
    if cached is None:
        cached = load_paper(paper_id)
        if cached is not None:
            PAPER_CACHE[paper_id] = cached
    return cached


def _retrieve_top_chunks(chunks: list[str], embeddings: np.ndarray, query: str, top_k: int = 3) -> str:
    """Embed a query and return the top_k most similar chunks, joined as context."""
    q_embedding = EMBEDDER.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = embeddings @ q_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n---\n\n".join(chunks[i] for i in top_indices)


def _ask_paper_raw(paper_id: str, question: str) -> dict:
    """Core RAG logic for ask_paper, returning structured data (answer +
    the actual retrieved context) instead of a formatted string. Used by
    both the ask_paper tool and the demo UI, which shows the context
    directly to prove this is real retrieval, not just an LLM call."""
    cached = _get_cached_paper(paper_id)
    if cached is None:
        return {"error": f"Paper {paper_id} not fetched yet. Call fetch_paper first."}
    chunks, embeddings = cached

    try:
        context = _retrieve_top_chunks(chunks, embeddings, question, top_k=3)
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": (
                    f"Answer the question using ONLY the excerpts below from an "
                    f"academic paper. If the excerpts don't contain the answer, "
                    f"say so clearly instead of guessing.\n\n"
                    f"Excerpts:\n{context}\n\n"
                    f"Question: {question}"
                )
            }]
        )
        return {"answer": response.choices[0].message.content, "context": context, "error": None}
    except Exception as e:
        return {"error": f"Couldn't generate an answer: {e}. The retrieval worked, but the LLM call failed — check your GROQ_API_KEY and rate limits."}


@mcp.tool()
def ask_paper(paper_id: str, question: str) -> str:
    """Answer a question using only the text of a previously fetched paper.

    Args:
        paper_id: arXiv ID that was already passed to fetch_paper
        question: The question to answer using the paper's content
    """
    result = _ask_paper_raw(paper_id, question)
    if result.get("error"):
        return result["error"]
    return f"{result['answer']}\n\n---\n(Answer generated from paper {paper_id}, grounded in 3 retrieved excerpts.)"


def _compare_papers_raw(paper_id_1: str, paper_id_2: str) -> dict:
    """Core comparison logic, returning structured data (comparison text
    + both papers' retrieved context) instead of a formatted string."""
    cached_1 = _get_cached_paper(paper_id_1)
    cached_2 = _get_cached_paper(paper_id_2)
    missing = [pid for pid, c in [(paper_id_1, cached_1), (paper_id_2, cached_2)] if c is None]
    if missing:
        return {"error": f"Paper(s) not fetched yet: {', '.join(missing)}. Call fetch_paper on each first."}

    try:
        probe = "core contribution, problem statement, and method"
        context_1 = _retrieve_top_chunks(*cached_1, probe, top_k=3)
        context_2 = _retrieve_top_chunks(*cached_2, probe, top_k=3)

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=700,
            messages=[{
                "role": "user",
                "content": (
                    f"Compare these two academic papers using ONLY the excerpts below.\n\n"
                    f"Paper A ({paper_id_1}) excerpts:\n{context_1}\n\n"
                    f"Paper B ({paper_id_2}) excerpts:\n{context_2}\n\n"
                    f"Structure your answer as:\n"
                    f"1. Paper A's problem and method (2-3 sentences)\n"
                    f"2. Paper B's problem and method (2-3 sentences)\n"
                    f"3. Key differences in approach\n"
                    f"4. Which seems stronger for which use case, and why"
                )
            }]
        )
        return {
            "comparison": response.choices[0].message.content,
            "context_1": context_1,
            "context_2": context_2,
            "error": None,
        }
    except Exception as e:
        return {"error": f"Couldn't generate a comparison: {e}. Check your GROQ_API_KEY and rate limits."}


@mcp.tool()
def compare_fetched_papers(paper_id_1: str, paper_id_2: str) -> str:
    """Compare two already-fetched papers: their core contribution, method,
    and how their approaches differ. Both papers must have been fetched
    with fetch_paper first.

    Args:
        paper_id_1: arXiv ID of the first paper
        paper_id_2: arXiv ID of the second paper
    """
    result = _compare_papers_raw(paper_id_1, paper_id_2)
    if result.get("error"):
        return result["error"]
    return result["comparison"]


# --- Resources --------------------------------------------------------
# Unlike tools (actions the client calls), resources are data the client
# can browse directly — like files. A client such as Claude Desktop can
# show these as attachable context without you calling a tool first.

@mcp.resource("papers://list")
def papers_list() -> str:
    """A browsable list of every paper that has been fetched so far."""
    papers = list_all_papers()
    if not papers:
        return "No papers fetched yet. Use fetch_paper to add one."
    return "\n".join(f"{pid}: {title}" for pid, title in papers)


@mcp.resource("papers://{paper_id}")
def paper_full_text(paper_id: str) -> str:
    """The full extracted text of a specific fetched paper, for browsing
    or attaching as context without going through ask_paper."""
    cached = _get_cached_paper(paper_id)
    if cached is None:
        return f"Paper {paper_id} has not been fetched yet."
    chunks, _ = cached
    return "\n\n".join(chunks)


# --- Prompts ------------------------------------------------------------
# Reusable prompt templates the client can surface directly (e.g. as a
# slash command), rather than the user having to write the same
# instructions to the model every time.

@mcp.prompt()
def literature_review(topic: str, num_papers: int = 5) -> str:
    """Scaffolds a literature-review workflow: search, fetch, and
    synthesize findings across multiple papers on a topic."""
    return (
        f"I want a short literature review on: {topic}\n\n"
        f"Please:\n"
        f"1. Use search_arxiv to find the {num_papers} most relevant papers.\n"
        f"2. Use fetch_paper on each one.\n"
        f"3. Use ask_paper on each to extract their core contribution and method.\n"
        f"4. Summarize the common themes, disagreements, and open problems "
        f"across all the papers in a short synthesis at the end."
    )


@mcp.prompt()
def compare_papers(paper_id_1: str, paper_id_2: str) -> str:
    """Scaffolds a side-by-side comparison of two already-fetched papers."""
    return (
        f"Compare paper {paper_id_1} and paper {paper_id_2}. "
        f"Use ask_paper on each to determine: (1) the problem each paper solves, "
        f"(2) their core method, (3) how their approaches differ, and "
        f"(4) which seems stronger and why."
    )


if __name__ == "__main__":
    mcp.run()