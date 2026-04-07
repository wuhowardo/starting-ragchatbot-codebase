"""
Shared fixtures for the RAG chatbot test suite.

The production app.py mounts StaticFiles and initialises ChromaDB at import
time, so we build a lightweight test app here that wires the same routes to a
mocked RAGSystem — no real filesystem or vector store needed.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from helpers import create_test_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag():
    """RAGSystem mock with sensible defaults for the happy path."""
    rag = MagicMock()

    # session_manager behaviour
    rag.session_manager.create_session.return_value = "generated-session-id"
    rag.session_manager.clear_session.return_value = None

    # query returns (answer, sources)
    rag.query.return_value = ("Here is the answer.", ["source1.txt", "source2.txt"])

    # course analytics
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Intro to Python", "Machine Learning", "Data Engineering"],
    }

    return rag


@pytest.fixture
def client(mock_rag):
    """TestClient backed by the lightweight test app."""
    app = create_test_app(mock_rag)
    return TestClient(app)


@pytest.fixture
def sample_query_payload():
    return {"query": "What is machine learning?"}


@pytest.fixture
def sample_query_payload_with_session():
    return {"query": "What is machine learning?", "session_id": "existing-session-123"}
