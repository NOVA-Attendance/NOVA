#!/usr/bin/env python3
"""
Generate a deterministic facial user ID from an image by hashing its embedding.

Usage:
  python scripts/facial_user_id.py --image path/to/image.jpg [--model Facenet512]

This script computes a face embedding using DeepFace and outputs a short,
stable identifier derived from the embedding.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from deepface import DeepFace


def compute_face_embedding(image_path: Path, model_name: str) -> list[float]:
    """
    Compute a face embedding for the given image using a specified model.

    Returns the embedding as a list of floats. Raises SystemExit on errors.
    """
    try:
        representations = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend="retinaface",
            enforce_detection=True,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Error: failed to compute embedding: {error}", file=sys.stderr)
        raise SystemExit(1)

    if not representations:
        print("Error: no face representation returned", file=sys.stderr)
        raise SystemExit(1)

    # DeepFace.represent may return list of dicts or a single dict depending on version
    rep = representations[0] if isinstance(representations, list) else representations

    embedding = rep.get("embedding") if isinstance(rep, dict) else rep
    if not embedding:
        print("Error: embedding not found in representation", file=sys.stderr)
        raise SystemExit(1)

    return [float(x) for x in embedding]


def hash_embedding(embedding: list[float]) -> str:
    """
    Hash the embedding to a short, URL-safe ID.

    Uses SHA-256 over the JSON-serialized embedding and returns the first
    16 hex characters (64-bit fingerprint). Adjust length as needed.
    """
    serialized = json.dumps(embedding, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate facial user ID from image")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument(
        "--model",
        default="Facenet512",
        help="DeepFace model to use (e.g., Facenet512, ArcFace, VGG-Face)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        raise SystemExit(1)

    embedding = compute_face_embedding(image_path=image_path, model_name=args.model)
    facial_user_id = hash_embedding(embedding)

    print(json.dumps({
        "image": str(image_path),
        "model": args.model,
        "facial_user_id": facial_user_id,
    }))


if __name__ == "__main__":
    main()


