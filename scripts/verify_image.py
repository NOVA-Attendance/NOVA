#!/usr/bin/env python3
"""
Identify a face in an input image by comparing to a local embeddings index
produced by scripts/enroll_faces.py. No database required.

Usage:
  python scripts/verify_image.py --image path/to/test.jpg --model Facenet512 --metric cos
"""

import argparse
import json
from pathlib import Path

import numpy as np
from deepface import DeepFace


def compute_embedding(image_path: Path, model_name: str) -> np.ndarray:
    """Compute one embedding for the given image path using DeepFace."""
    reps = DeepFace.represent(
        img_path=str(image_path),
        model_name=model_name,
        detector_backend="opencv",  # Optimized for Jetson
        enforce_detection=True,
    )
    rep = reps[0] if isinstance(reps, list) and reps else reps
    embedding = rep.get("embedding") if isinstance(rep, dict) else rep
    if not embedding:
        raise RuntimeError("embedding missing")
    return np.array(embedding, dtype=np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1 - cosine similarity (smaller means closer)."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    # cos sim -> [ -1..1 ], convert to distance [0..2]
    cos_sim = float(np.dot(a_norm, b_norm))
    return float(1.0 - cos_sim)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify/identify face from local index")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model")
    parser.add_argument("--metric", choices=["cos", "l2"], default="cos", help="Distance metric")
    args = parser.parse_args()

    data_dir = Path("data")
    index_path = data_dir / "face_index.npz"
    labels_path = data_dir / "labels.json"
    students_path = data_dir / "students.json"

    if not index_path.exists() or not labels_path.exists():
        raise SystemExit("Index not found. Run scripts/enroll_faces.py first.")

    npz = np.load(index_path)
    embeddings = npz["embeddings"].astype(np.float32)
    labels = json.loads(labels_path.read_text())
    students = {}
    if students_path.exists():
        try:
            for row in json.loads(students_path.read_text()):
                # Map label -> student_id for friendly reporting
                students[row.get("label")] = row.get("student_id")
        except Exception:
            students = {}

    query_emb = compute_embedding(Path(args.image), args.model)

    if args.metric == "cos":
        dists = [cosine_distance(query_emb, e) for e in embeddings]
    else:
        dists = [l2_distance(query_emb, e) for e in embeddings]

    best_idx = int(np.argmin(dists))
    best_label = labels[best_idx]
    best_student_id = students.get(best_label)
    best_dist = float(dists[best_idx])

    print(json.dumps({
        "label": best_label,
        "student_id": best_student_id,
        "distance": best_dist,
        "metric": args.metric,
        "index_path": str(index_path),
    }, indent=2))


if __name__ == "__main__":
    main()


