#!/usr/bin/env python3
"""
Compare two face images and output similarity percentage.

Usage:
  python scripts/compare_faces.py --reference path/to/reference.jpg --test path/to/test.jpg
  python scripts/compare_faces.py --reference data/image_database/jeff_bezos_demo_images.webp --test captured_face.jpg
"""

import argparse
from pathlib import Path

import numpy as np
from deepface import DeepFace


def compute_embedding(image_path: Path, model_name: str) -> np.ndarray:
    """Compute face embedding for a single image."""
    try:
        reps = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend="opencv",
            enforce_detection=True,
        )
        rep = reps[0] if isinstance(reps, list) and reps else reps
        embedding = rep.get("embedding") if isinstance(rep, dict) else rep
        if not embedding:
            raise RuntimeError("No face detected in image")
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {e}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings (0-1, higher is better)."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    similarity = float(np.dot(a_norm, b_norm))
    return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]


def compare_faces(reference_path: Path, test_path: Path, model_name: str = "Facenet512") -> dict:
    """
    Compare two face images and return similarity percentage.
    
    Returns:
        dict with keys: similarity_percentage, match (bool), reference_image, test_image
    """
    print(f"Loading reference image: {reference_path.name}")
    reference_embedding = compute_embedding(reference_path, model_name)
    
    print(f"Loading test image: {test_path.name}")
    test_embedding = compute_embedding(test_path, model_name)
    
    print("Comparing faces...")
    similarity = cosine_similarity(reference_embedding, test_embedding)
    similarity_percentage = similarity * 100.0
    
    # Typical threshold is 0.4 distance = 0.6 similarity = 60%
    is_match = similarity >= 0.6
    
    return {
        "reference_image": str(reference_path),
        "test_image": str(test_path),
        "similarity_percentage": round(similarity_percentage, 2),
        "match": is_match,
        "model": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two face images")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--test", required=True, help="Path to test image")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model name")
    args = parser.parse_args()
    
    reference_path = Path(args.reference)
    test_path = Path(args.test)
    
    if not reference_path.exists():
        print(f"Error: Reference image not found: {reference_path}")
        return
    
    if not test_path.exists():
        print(f"Error: Test image not found: {test_path}")
        return
    
    try:
        result = compare_faces(reference_path, test_path, args.model)
        
        print("\n" + "="*60)
        print("FACE COMPARISON RESULT")
        print("="*60)
        print(f"Reference: {result['reference_image']}")
        print(f"Test:      {result['test_image']}")
        print(f"Similarity: {result['similarity_percentage']}%")
        print(f"Match:     {'✓ YES (Same person)' if result['match'] else '✗ NO (Different person)'}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

