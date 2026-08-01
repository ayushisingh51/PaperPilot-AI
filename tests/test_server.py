"""
Tests for the research-assistant MCP server.

Run with:  pytest -v

Network calls (arXiv, PDF download) and the Groq LLM call are mocked —
these tests never hit the real internet or spend API credits, and the
embedding model runs locally so it's exercised for real.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import server


# --- chunk_text: pure function, no mocking needed -------------------------

def test_chunk_text_splits_long_text():
    text = "a" * 2000
    chunks = server.chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    # every chunk should respect the requested size
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    text = "short text"
    chunks = server.chunk_text(text, chunk_size=800, overlap=100)
    assert chunks == ["short text"]


def test_chunk_text_empty_string():
    assert server.chunk_text("") == []


# --- search_arxiv -----------------------------------------------------------

def _fake_arxiv_result(short_id="2005.11401"):
    r = MagicMock()
    r.get_short_id.return_value = short_id
    r.title = "Fake Paper Title"
    author_a, author_b = MagicMock(), MagicMock()
    author_a.name = "Author A"
    author_b.name = "Author B"
    r.authors = [author_a, author_b]
    r.published.date.return_value = "2020-05-22"
    r.summary = "A summary. " * 30
    return r


def test_search_arxiv_returns_formatted_results():
    fake_result = _fake_arxiv_result()
    with patch.object(server.arxiv, "Client") as MockClient:
        MockClient.return_value.results.return_value = [fake_result]
        output = server.search_arxiv("retrieval augmented generation")
    assert "2005.11401" in output
    assert "Fake Paper Title" in output


def test_search_arxiv_no_results():
    with patch.object(server.arxiv, "Client") as MockClient:
        MockClient.return_value.results.return_value = []
        output = server.search_arxiv("a query with no matches")
    assert "No results found" in output


def test_search_arxiv_network_failure_returns_friendly_message():
    with patch.object(server.arxiv, "Client", side_effect=ConnectionError("no internet")):
        output = server.search_arxiv("anything")
    assert "Search failed" in output
    assert "no internet" in output


# --- fetch_paper --------------------------------------------------------

def _fake_pdf_bytes():
    # Minimal valid-enough PDF bytes aren't needed — we mock pypdf too,
    # so any bytes object stands in for the HTTP response content.
    return b"%PDF-1.4 fake content"


def test_fetch_paper_success_caches_and_persists():
    fake_paper = _fake_arxiv_result("2005.11401")
    fake_paper.pdf_url = "https://arxiv.org/pdf/2005.11401"

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "This is the extracted paper text. " * 50

    with patch.object(server.arxiv, "Client") as MockClient, \
         patch.object(server.httpx, "get") as mock_get, \
         patch.object(server.pypdf, "PdfReader") as MockReader:

        MockClient.return_value.results.return_value = iter([fake_paper])
        mock_get.return_value = MagicMock(content=_fake_pdf_bytes())
        MockReader.return_value.pages = [fake_page]

        output = server.fetch_paper("2005.11401")

    assert "Fetched 'Fake Paper Title'" in output
    assert "2005.11401" in server.PAPER_CACHE
    # persisted to DB too, not just memory
    assert server.load_paper("2005.11401") is not None


def test_fetch_paper_invalid_id_returns_friendly_message():
    with patch.object(server.arxiv, "Client") as MockClient:
        MockClient.return_value.results.return_value = iter([])  # no matches -> StopIteration
        output = server.fetch_paper("not-a-real-id")
    assert "No paper found" in output
    assert "not-a-real-id" not in server.PAPER_CACHE


def test_fetch_paper_download_failure_returns_friendly_message():
    fake_paper = _fake_arxiv_result("2005.11401")
    fake_paper.pdf_url = "https://arxiv.org/pdf/2005.11401"

    with patch.object(server.arxiv, "Client") as MockClient, \
         patch.object(server.httpx, "get", side_effect=server.httpx.HTTPError("timed out")):
        MockClient.return_value.results.return_value = iter([fake_paper])
        output = server.fetch_paper("2005.11401")

    assert "Couldn't download the PDF" in output


# --- ask_paper ------------------------------------------------------------

def test_ask_paper_not_fetched_yet():
    output = server.ask_paper("9999.99999", "what is this about?")
    assert "not fetched yet" in output


def test_ask_paper_generates_grounded_answer():
    chunks = ["chunk about transformers", "chunk about optimizers", "chunk about datasets"]
    embeddings = np.random.rand(3, 384).astype(np.float32)
    server.PAPER_CACHE["2005.11401"] = (chunks, embeddings)

    fake_response = MagicMock()
    fake_response.choices[0].message.content = "The paper uses a transformer architecture."

    with patch.object(server.groq_client.chat.completions, "create", return_value=fake_response):
        output = server.ask_paper("2005.11401", "what architecture is used?")

    assert "transformer architecture" in output
    assert "2005.11401" in output  # source paper referenced in the footer


def test_ask_paper_llm_failure_returns_friendly_message():
    chunks = ["some chunk"]
    embeddings = np.random.rand(1, 384).astype(np.float32)
    server.PAPER_CACHE["2005.11401"] = (chunks, embeddings)

    with patch.object(server.groq_client.chat.completions, "create",
                       side_effect=RuntimeError("rate limited")):
        output = server.ask_paper("2005.11401", "any question")

    assert "Couldn't generate an answer" in output


def test_ask_paper_falls_back_to_db_after_cache_clear():
    """Simulates a server restart: paper was saved to the DB but the
    in-memory cache is empty; ask_paper should still find it."""
    chunks = ["persisted chunk one", "persisted chunk two"]
    embeddings = np.random.rand(2, 384).astype(np.float32)
    server.save_paper("2005.11401", "Fake Title", chunks, embeddings)
    server.PAPER_CACHE.clear()  # simulate restart wiping memory

    fake_response = MagicMock()
    fake_response.choices[0].message.content = "Answer from persisted data."

    with patch.object(server.groq_client.chat.completions, "create", return_value=fake_response):
        output = server.ask_paper("2005.11401", "what does it say?")

    assert "Answer from persisted data" in output


# --- persistence layer directly --------------------------------------------

def test_save_and_load_paper_roundtrip():
    chunks = ["a", "b", "c"]
    embeddings = np.random.rand(3, 8).astype(np.float32)
    server.save_paper("test-id", "Test Title", chunks, embeddings)

    loaded_chunks, loaded_embeddings = server.load_paper("test-id")
    assert loaded_chunks == chunks
    assert np.allclose(loaded_embeddings, embeddings)


def test_load_paper_returns_none_for_missing_id():
    assert server.load_paper("does-not-exist") is None


def test_list_all_papers_reflects_saved_papers():
    server.save_paper("id-1", "Title One", ["x"], np.random.rand(1, 4).astype(np.float32))
    server.save_paper("id-2", "Title Two", ["y"], np.random.rand(1, 4).astype(np.float32))

    papers = server.list_all_papers()
    ids = [p[0] for p in papers]
    assert "id-1" in ids and "id-2" in ids


# --- resources and prompts -------------------------------------------------

def test_papers_list_resource_empty():
    output = server.papers_list()
    assert "No papers fetched yet" in output


def test_papers_list_resource_with_data():
    server.save_paper("id-1", "My Paper", ["x"], np.random.rand(1, 4).astype(np.float32))
    output = server.papers_list()
    assert "id-1" in output and "My Paper" in output


def test_literature_review_prompt_includes_topic():
    output = server.literature_review("diffusion models", num_papers=3)
    assert "diffusion models" in output
    assert "3" in output


def test_compare_papers_prompt_includes_both_ids():
    output = server.compare_papers("1111.1111", "2222.2222")
    assert "1111.1111" in output
    assert "2222.2222" in output