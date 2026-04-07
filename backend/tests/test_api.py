"""
API endpoint tests for the RAG chatbot FastAPI app.

Uses the lightweight test app and fixtures defined in conftest.py — no
ChromaDB, no Anthropic API calls, no frontend static files required.
"""

import pytest
from unittest.mock import MagicMock
from helpers import create_test_app
from fastapi.testclient import TestClient


# ===========================================================================
# POST /api/query
# ===========================================================================

class TestQueryEndpoint:
    def test_returns_200_with_valid_payload(self, client, sample_query_payload):
        response = client.post("/api/query", json=sample_query_payload)
        assert response.status_code == 200

    def test_response_shape(self, client, sample_query_payload):
        response = client.post("/api/query", json=sample_query_payload)
        body = response.json()
        assert "answer" in body
        assert "sources" in body
        assert "session_id" in body

    def test_answer_and_sources_come_from_rag(self, client, mock_rag, sample_query_payload):
        mock_rag.query.return_value = ("RAG answer", ["doc_a.txt"])
        response = client.post("/api/query", json=sample_query_payload)
        body = response.json()
        assert body["answer"] == "RAG answer"
        assert body["sources"] == ["doc_a.txt"]

    def test_session_id_generated_when_not_provided(self, client, mock_rag, sample_query_payload):
        mock_rag.session_manager.create_session.return_value = "new-session-abc"
        response = client.post("/api/query", json=sample_query_payload)
        assert response.json()["session_id"] == "new-session-abc"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_provided_session_id_is_used(self, client, mock_rag, sample_query_payload_with_session):
        response = client.post("/api/query", json=sample_query_payload_with_session)
        assert response.json()["session_id"] == "existing-session-123"
        mock_rag.session_manager.create_session.assert_not_called()

    def test_rag_query_called_with_correct_args(self, client, mock_rag, sample_query_payload_with_session):
        client.post("/api/query", json=sample_query_payload_with_session)
        mock_rag.query.assert_called_once_with(
            "What is machine learning?", "existing-session-123"
        )

    def test_returns_500_when_rag_raises(self, mock_rag):
        mock_rag.query.side_effect = RuntimeError("vector store unavailable")
        app = create_test_app(mock_rag)
        bad_client = TestClient(app, raise_server_exceptions=False)
        response = bad_client.post("/api/query", json={"query": "anything"})
        assert response.status_code == 500
        assert "vector store unavailable" in response.json()["detail"]

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={"session_id": "abc"})
        assert response.status_code == 422

    def test_empty_query_string_is_accepted(self, client):
        response = client.post("/api/query", json={"query": ""})
        assert response.status_code == 200

    def test_sources_list_can_be_empty(self, client, mock_rag):
        mock_rag.query.return_value = ("answer with no sources", [])
        response = client.post("/api/query", json={"query": "hi"})
        assert response.json()["sources"] == []


# ===========================================================================
# GET /api/courses
# ===========================================================================

class TestCoursesEndpoint:
    def test_returns_200(self, client):
        response = client.get("/api/courses")
        assert response.status_code == 200

    def test_response_shape(self, client):
        body = client.get("/api/courses").json()
        assert "total_courses" in body
        assert "course_titles" in body

    def test_returns_correct_stats(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }
        body = client.get("/api/courses").json()
        assert body["total_courses"] == 2
        assert body["course_titles"] == ["Course A", "Course B"]

    def test_returns_500_when_analytics_raises(self, mock_rag):
        mock_rag.get_course_analytics.side_effect = Exception("db error")
        app = create_test_app(mock_rag)
        bad_client = TestClient(app, raise_server_exceptions=False)
        response = bad_client.get("/api/courses")
        assert response.status_code == 500
        assert "db error" in response.json()["detail"]

    def test_empty_catalog_returns_zero_count(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        body = client.get("/api/courses").json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []


# ===========================================================================
# DELETE /api/session/{session_id}
# ===========================================================================

class TestDeleteSessionEndpoint:
    def test_returns_200(self, client):
        response = client.delete("/api/session/my-session-id")
        assert response.status_code == 200

    def test_response_body(self, client):
        response = client.delete("/api/session/my-session-id")
        assert response.json() == {"status": "cleared"}

    def test_clear_session_called_with_correct_id(self, client, mock_rag):
        client.delete("/api/session/target-session-99")
        mock_rag.session_manager.clear_session.assert_called_once_with("target-session-99")
