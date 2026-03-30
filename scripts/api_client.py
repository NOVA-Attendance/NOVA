#!/usr/bin/env python3
"""
api_client.py - HTTP wrapper for the NOVA backend server.

Paths match NOVA-Server (source of truth). All functions return safe defaults
(None / False) on failure so the caller can fall through to the offline path.
"""

import json
import logging
from datetime import datetime
from typing import Optional
from urllib.parse import quote

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


def get_rfid_face_embedding(rfid_tag: str) -> Optional[dict]:
    """GET /rfid/face-embedding — student row and stored embedding for this card (no attendance).

    Returns a dict with student_id, name, student_number, face_embedding (list or None).
    Returns None if the tag is unknown or the server is unreachable.
    """
    q = quote(str(rfid_tag), safe="")
    resp = _get(f"/rfid/face-embedding?rfid_id={q}")
    if resp is None:
        logger.warning("Could not reach server for RFID %s.", rfid_tag)
        return None
    if resp.status_code == 404:
        logger.warning("RFID tag %s not found on server.", rfid_tag)
        return None
    if resp.status_code != 200:
        logger.error("GET /rfid/face-embedding returned %d.", resp.status_code)
        return None

    try:
        data = resp.json()
        embedding = data.get("face_embedding")
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        data["face_embedding"] = embedding
        return data
    except (ValueError, KeyError) as e:
        logger.error("Malformed response for RFID %s: %s", rfid_tag, e)
        return None


def post_attendance_face_verify(
    rfid_tag: str,
    student_id,
    class_id: int,
    confidence: float,
    matched: bool,
    timestamp: Optional[datetime] = None,
) -> bool:
    """POST /attendance/face-verify — log RFID + face verification outcome."""

    if isinstance(timestamp, str):
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now()
    elif timestamp is not None:
        ts = timestamp
    else:
        ts = datetime.now()

    payload = {
        "rfid_tag": rfid_tag,
        "student_id": student_id,
        "class_id": class_id,
        "confidence": round(float(confidence), 4),
        "matched": matched,
        "timestamp": ts.isoformat(),
    }
    resp = _post("/attendance/face-verify", payload)
    if resp is None:
        return False
    if resp.status_code in (200, 201):
        logger.info(
            "Attendance posted for student %s (confidence=%.2f, matched=%s).",
            student_id,
            confidence,
            matched,
        )
        return True
    logger.error("POST /attendance/face-verify returned %d.", resp.status_code)
    return False
