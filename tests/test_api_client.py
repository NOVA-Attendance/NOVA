#!/usr/bin/env python3
"""
tests/test_api_client.py - Unit tests for scripts/api_client.py

Run from the repository root:
    python -m pytest tests/test_api_client.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import api_client


def mock_response(status, body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body or {}
    resp.text = json.dumps(body or {})
    return resp


class TestGetStudentByRfid:

    def test_success_with_embedding(self):
        embedding = [0.1] * 512
        body = {"student_id": 42, "name": "Alice", "face_embedding": embedding}
        with patch("requests.get", return_value=mock_response(200, body)):
            result = api_client.get_student_by_rfid("ABCD1234")
        assert result["student_id"] == 42
        assert len(result["face_embedding"]) == 512

    def test_success_without_embedding(self):
        body = {"student_id": 7, "name": "Bob", "face_embedding": None}
        with patch("requests.get", return_value=mock_response(200, body)):
            result = api_client.get_student_by_rfid("ABCD5678")
        assert result is not None
        assert result["face_embedding"] is None

    def test_404_returns_none(self):
        with patch("requests.get", return_value=mock_response(404)):
            result = api_client.get_student_by_rfid("UNKNOWN")
        assert result is None

    def test_connection_error_returns_none(self):
        import requests as req
        with patch("requests.get", side_effect=req.ConnectionError):
            result = api_client.get_student_by_rfid("ABCD1234")
        assert result is None

    def test_timeout_returns_none(self):
        import requests as req
        with patch("requests.get", side_effect=req.Timeout):
            result = api_client.get_student_by_rfid("ABCD1234")
        assert result is None

    def test_500_returns_none(self):
        with patch("requests.get", return_value=mock_response(500)):
            result = api_client.get_student_by_rfid("ABCD1234")
        assert result is None


class TestPostFaceResult:

    base = dict(
        rfid_tag="ABCD1234",
        student_id=42,
        class_id=1,
        confidence=0.85,
        matched=True,
        timestamp=datetime(2026, 2, 21, 16, 0, 0),
    )

    def test_success_201(self):
        with patch("requests.post", return_value=mock_response(201, {"log_id": 1})):
            assert api_client.post_face_result(**self.base) is True

    def test_success_200(self):
        with patch("requests.post", return_value=mock_response(200, {"log_id": 1})):
            assert api_client.post_face_result(**self.base) is True

    def test_server_error_returns_false(self):
        with patch("requests.post", return_value=mock_response(500)):
            assert api_client.post_face_result(**self.base) is False

    def test_connection_error_returns_false(self):
        import requests as req
        with patch("requests.post", side_effect=req.ConnectionError):
            assert api_client.post_face_result(**self.base) is False

    def test_payload_contains_required_keys(self):
        captured = {}

        def capture(url, json=None, **kw):
            captured.update(json or {})
            return mock_response(201, {"log_id": 1})

        with patch("requests.post", side_effect=capture):
            api_client.post_face_result(**self.base)

        for key in ("rfid_tag", "student_id", "class_id", "confidence", "matched", "timestamp"):
            assert key in captured
        assert 0.0 <= captured["confidence"] <= 1.0


class TestServerHealth:

    def test_healthy(self):
        with patch("requests.get", return_value=mock_response(200, {"status": "ok"})):
            assert api_client.check_server_health() is True

    def test_unhealthy(self):
        with patch("requests.get", return_value=mock_response(503)):
            assert api_client.check_server_health() is False

    def test_unreachable(self):
        import requests as req
        with patch("requests.get", side_effect=req.ConnectionError):
            assert api_client.check_server_health() is False
