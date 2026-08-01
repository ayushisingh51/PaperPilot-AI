"""Shared pytest setup for the research-assistant test suite."""
import os
import sys

# server.py raises at import time if GROQ_API_KEY is missing — provide a
# dummy one so tests can import it and mock the actual API calls instead.
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import server


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the paper database at a throwaway temp file for every test,
    so tests never touch your real papers.db, and clear the in-memory
    cache so tests don't leak state into each other."""
    db_path = tmp_path / "test_papers.db"
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    server.init_db()
    server.PAPER_CACHE.clear()
    yield str(db_path)