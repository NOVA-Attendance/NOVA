#!/usr/bin/env python3
"""
tests/smoke_test.py - End-to-end integration test against a running Flask backend.

Simulates a full RFID scan cycle without needing physical hardware.

Usage:
    python tests/smoke_test.py --rfid-tag ABCD1234 --server http://localhost:5001
    python tests/smoke_test.py --rfid-tag ABCD1234 --image data/image_database/test.jpg
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import api_client
from recognize_image import compare_against_embedding


def run(args):
    api_client.SERVER_URL = args.server
    errors = 0

    print(f"\nStep 1: Health check ({args.server})")
    if api_client.check_server_health():
        print("  Server is healthy.")
    else:
        print("  Server unreachable. Is the Flask app running?")
        sys.exit(1)

    print(f"\nStep 2: GET /students/rfid/{args.rfid_tag}")
    student = api_client.get_student_by_rfid(args.rfid_tag)
    if student:
        print(f"  Student found: {student.get('name')} (id={student.get('student_id')})")
        embedding = student.get("face_embedding")
        if embedding:
            print(f"  Embedding received ({len(embedding)} dims).")
        else:
            print("  No embedding stored for this student on the server.")
    else:
        print(f"  Student not found for RFID '{args.rfid_tag}'. Register the student first.")
        errors += 1
        embedding = None

    confidence = 0.0
    matched = False
    image_path = Path(args.image) if args.image else None

    if image_path and image_path.exists() and embedding:
        print(f"\nStep 3: Face comparison ({image_path.name} vs server embedding)")
        result = compare_against_embedding(image_path, embedding, threshold=0.4)
        if "error" in result:
            print(f"  Recognition error: {result['error']}")
            errors += 1
        else:
            confidence = result["confidence"]
            matched = result["matched"]
            print(f"  Confidence: {confidence:.1%} | Matched: {matched}")
    else:
        reason = "no --image provided" if not image_path else (
            "image not found" if not image_path.exists() else "no server embedding"
        )
        print(f"\nStep 3: Face comparison skipped ({reason})")

    student_id = (student or {}).get("student_id", args.rfid_tag)
    print(f"\nStep 4: POST /attendance/face-verify")
    ok = api_client.post_face_result(
        rfid_tag   = args.rfid_tag,
        student_id = student_id,
        class_id   = args.class_id,
        confidence = confidence,
        matched    = matched,
        timestamp  = datetime.now(),
    )
    if ok:
        print("  Attendance logged successfully.")
    else:
        print("  POST failed. Check server logs.")
        errors += 1

    print(f"\nSmoke test {'PASSED' if errors == 0 else 'FAILED'} ({errors} error(s)).")
    return errors


def main():
    parser = argparse.ArgumentParser(description="NOVA server integration smoke test")
    parser.add_argument("--rfid-tag", required=True, help="RFID tag string to test")
    parser.add_argument("--image", help="Path to a test face image (optional)")
    parser.add_argument("--server", default="http://localhost:5001", help="Server base URL")
    parser.add_argument("--class-id", type=int, default=1, help="class_id to report")
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
