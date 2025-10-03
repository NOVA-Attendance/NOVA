#!/usr/bin/env python3
"""
Live webcam demo to display a deterministic facial user ID derived
from DeepFace embeddings. Press 'q' to quit.

Requirements:
  - OpenCV available via opencv-python
  - DeepFace installed in the active environment

Usage:
  python scripts/facial_user_id_webcam.py --model Facenet512 --interval 10 --camera 0
"""

import argparse
import hashlib
import json
import time

import cv2
from deepface import DeepFace


def hash_embedding(embedding: list[float], length: int = 16) -> str:
    serialized = json.dumps(embedding, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest[:length]


def try_compute_embedding(frame_bgr, model_name: str):
    try:
        # DeepFace expects RGB images
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        reps = DeepFace.represent(
            img_path=frame_rgb,
            model_name=model_name,
            detector_backend="retinaface",
            enforce_detection=True,
        )
        rep = reps[0] if isinstance(reps, list) and reps else reps
        embedding = rep.get("embedding") if isinstance(rep, dict) else rep
        if not embedding:
            return None, "no embedding in response"
        return embedding, None
    except Exception as error:  # noqa: BLE001
        return None, str(error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam facial user ID demo")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model name")
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Compute embedding every N frames (reduce CPU usage)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (0 is default; try 1 if 0 fails)",
    )
    args = parser.parse_args()

    # Prefer AVFoundation backend on macOS if available
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Error: Could not open webcam.")

    last_id = ""
    last_error = ""
    frame_count = 0
    fps_last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()
        dt = now - fps_last_time
        if dt >= 0.5:
            fps = (1.0 / dt)
            fps_last_time = now

        if frame_count % max(args.interval, 1) == 0:
            embedding, err = try_compute_embedding(frame, args.model)
            if embedding is not None:
                last_id = hash_embedding([float(x) for x in embedding])
                last_error = ""
            else:
                last_error = err or "embedding failed"

        # Overlay info
        overlay_lines = [
            f"Model: {args.model}",
            f"FPS~: {fps:.1f}",
            f"Facial ID: {last_id or 'N/A'}",
        ]
        if last_error:
            overlay_lines.append(f"Error: {last_error[:60]}")

        y = 25
        for line in overlay_lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 25

        cv2.imshow("NOVA - Facial User ID (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


