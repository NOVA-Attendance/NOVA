#!/usr/bin/env python3
"""
api_client.py - HTTP wrapper for the NOVA backend server.

All functions return safe defaults (None / False) on failure so the caller
can fall through to the offline path without needing try/except blocks.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import requests

SERVER_URL = "http://192.168.0.100:5001"
REQUEST_TIMEOUT = 5

logger = logging.getLogger(__name__)


def _get(path: str) -> Optional[requests.Response]:
    url = f"{SERVER_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        return requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        logger.warning("GET %s timed out.", path)
    except requests.ConnectionError:
        logger.warning("GET %s - server unreachable.", path)
    except requests.RequestException as e:
        logger.error("GET %s - %s", path, e)
    return None


def _post(path: str, payload: dict) -> Optional[requests.Response]:
    url = f"{SERVER_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        return requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        logger.warning("POST %s timed out.", path)
    except requests.ConnectionError:
        logger.warning("POST %s - server unreachable.", path)
    except requests.RequestException as e:
        logger.error("POST %s - %s", path, e)
    return None


def check_server_health() -> bool:
    """Return True if the backend is reachable and healthy."""
    resp = _get("/health")
    return resp is not None and resp.status_code == 200


def get_student_by_rfid(rfid_tag: str) -> Optional[dict]:
    """Fetch student info and stored face embedding for the given RFID tag.

    Returns a dict with student_id, name, student_number, and face_embedding
    (a list of floats, or None if no embedding has been stored).
    Returns None if the tag is unknown or the server is unreachable.
    """
    resp = _get(f"/students/rfid/{rfid_tag}")
    if resp is None:
        logger.warning("Could not reach server for RFID %s.", rfid_tag)
        return None
    if resp.status_code == 404:
        logger.warning("RFID tag %s not found on server.", rfid_tag)
        return None
    if resp.status_code != 200:
        logger.error("GET /students/rfid/%s returned %d.", rfid_tag, resp.status_code)
        return None

    try:
        data = resp.json()
        embedding = data.get("face_embedding")
        # Handle double-encoded JSON strings from some server versions
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        data["face_embedding"] = embedding
        return data
    except (ValueError, KeyError) as e:
        logger.error("Malformed response for RFID %s: %s", rfid_tag, e)
        return None


def post_face_result(rfid_tag: str, student_id, class_id: int, confidence: float,
                     matched: bool, timestamp: Optional[datetime] = None) -> bool:
    """POST the face verification result to the backend attendance log.

    Returns True on success, False on any failure.
    """
    payload = {
        "rfid_tag":   rfid_tag,
        "student_id": student_id,
        "class_id":   class_id,
        "confidence": round(float(confidence), 4),
        "matched":    matched,
        "timestamp":  (timestamp or datetime.now()).isoformat(),
    }
    resp = _post("/attendance/face-verify", payload)
    if resp is None:
        return False
    if resp.status_code in (200, 201):
        logger.info("Attendance posted for student %s (confidence=%.2f, matched=%s).",
                    student_id, confidence, matched)
        return True
    logger.error("POST /attendance/face-verify returned %d.", resp.status_code)
    return False


def rfid_scan(rfid_tag: str, course_code: Optional[str] = None) -> Optional[dict]:
    """Notify the server of an RFID tap via the existing /rfid/scan endpoint.

    Keeps the dashboard tap monitor working alongside the new face-verify flow.
    Returns the response dict on success, None on failure.
    """
    payload = {"rfid_tag": rfid_tag}
    if course_code:
        payload["course_code"] = course_code
    resp = _post("/rfid/scan", payload)
    if resp is not None and resp.status_code in (200, 201):
        try:
            return resp.json()
        except ValueError:
            return {}
    return None
