#!/usr/bin/env python3
"""
Recognize faces in saved images by comparing to enrolled face database.

This script can process single images or batch process multiple images
to identify faces against the enrolled face database.

Usage:
  python scripts/recognize_image.py --image path/to/image.jpg
  python scripts/recognize_image.py --batch data/images/ --threshold 0.4
  python scripts/recognize_image.py --folder data/images/300123456/
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from deepface import DeepFace
import cv2


def compute_embedding(image_path: Path, model_name: str) -> np.ndarray:
    """Compute a face embedding for one image using the selected model."""
    try:
        # Universal optimization: opencv detector is fastest across all platforms
        reps = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend="opencv",  # 3-5x faster than retinaface on all devices
            enforce_detection=True,
        )
        rep = reps[0] if isinstance(reps, list) and reps else reps
        embedding = rep.get("embedding") if isinstance(rep, dict) else rep
        if not embedding:
            raise RuntimeError("No face detected in image")
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to compute embedding: {e}")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1 - cosine similarity (smaller is better match)."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    cos_sim = float(np.dot(a_norm, b_norm))
    return float(1.0 - cos_sim)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two embedding vectors."""
    return float(np.linalg.norm(a - b))


def load_face_database(data_dir: Path) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Load embeddings, human-readable labels, and optional student-id map."""
    index_path = data_dir / "face_index.npz"
    labels_path = data_dir / "labels.json"
    students_path = data_dir / "students.json"

    if not index_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Face database not found. Run scripts/enroll_faces.py first.")

    # Load embeddings and labels
    npz = np.load(index_path)
    embeddings = npz["embeddings"].astype(np.float32)
    labels = json.loads(labels_path.read_text())

    # Load student mapping
    students = {}
    if students_path.exists():
        try:
            for student in json.loads(students_path.read_text()):
                students[student["label"]] = student["student_id"]
        except Exception:
            pass

    return embeddings, labels, students


def find_best_match(query_embedding: np.ndarray, database_embeddings: np.ndarray, 
                   database_labels: List[str], metric: str = "cos") -> Dict:
    """Return the closest label with distance and confidence score."""
    if metric == "cos":
        distances = [cosine_distance(query_embedding, emb) for emb in database_embeddings]
    else:
        distances = [l2_distance(query_embedding, emb) for emb in database_embeddings]
    
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    best_label = database_labels[best_idx]
    
    return {
        "label": best_label,
        "distance": best_distance,
        "confidence": max(0, 1 - best_distance) if metric == "cos" else max(0, 1 / (1 + best_distance))
    }


def recognize_single_image(image_path: Path, model_name: str, threshold: float = 0.4, 
                         metric: str = "cos") -> Dict:
    """Compare one image against the DB and mark recognized if >= threshold."""
    data_dir = Path("data")
    
    try:
        database_embeddings, database_labels, students = load_face_database(data_dir)
        query_embedding = compute_embedding(image_path, model_name)
        match = find_best_match(query_embedding, database_embeddings, database_labels, metric)
        student_id = students.get(match["label"])
        match["student_id"] = student_id
        match["image_path"] = str(image_path)
        match["model"] = model_name
        match["metric"] = metric
        match["threshold"] = threshold
        match["recognized"] = match["confidence"] >= threshold
        return match
        
    except Exception as e:
        return {
            "image_path": str(image_path),
            "error": str(e),
            "recognized": False
        }


def process_image_batch(image_paths: List[Path], model_name: str, threshold: float = 0.4,
                       metric: str = "cos") -> List[Dict]:
    """Loop through images, run recognition, and print a short summary."""
    results = []
    
    print(f"Processing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"Processing {i}/{len(image_paths)}: {image_path.name}")
        result = recognize_single_image(image_path, model_name, threshold, metric)
        results.append(result)
        
        if result.get("recognized", False):
            print(f"  ✓ Recognized: {result.get('student_id', 'Unknown')} (confidence: {result['confidence']:.3f})")
        else:
            print(f"  ✗ Not recognized (confidence: {result.get('confidence', 0):.3f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Recognize faces in saved images")
    parser.add_argument("--image", help="Path to single image to recognize")
    parser.add_argument("--batch", help="Path to directory with images to process")
    parser.add_argument("--folder", help="Path to specific student folder to process")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model name")
    parser.add_argument("--threshold", type=float, default=0.4, help="Recognition threshold (0-1)")
    parser.add_argument("--metric", choices=["cos", "l2"], default="cos", help="Distance metric")
    parser.add_argument("--output", help="Output JSON file for batch results")
    args = parser.parse_args()

    if not any([args.image, args.batch, args.folder]):
        parser.error("Must specify --image, --batch, or --folder")

    # Determine which images to process based on flags
    image_paths = []
    
    if args.image:
        image_paths = [Path(args.image)]
    elif args.batch:
        batch_dir = Path(args.batch)
        image_paths = (
            list(batch_dir.rglob("*.jpg"))
            + list(batch_dir.rglob("*.jpeg"))
            + list(batch_dir.rglob("*.png"))
            + list(batch_dir.rglob("*.webp"))
        )
    elif args.folder:
        folder_dir = Path(args.folder)
        image_paths = (
            list(folder_dir.glob("*.jpg"))
            + list(folder_dir.glob("*.jpeg"))
            + list(folder_dir.glob("*.png"))
            + list(folder_dir.glob("*.webp"))
        )

    if not image_paths:
        print("No images found to process")
        return

    # Process images
    if len(image_paths) == 1:
        # Single image processing
        result = recognize_single_image(image_paths[0], args.model, args.threshold, args.metric)
        
        print(f"\nRecognition Result:")
        print(f"Image: {result['image_path']}")
        
        if result.get("recognized", False):
            print(f"✓ RECOGNIZED: {result.get('student_id', 'Unknown')}")
            print(f"  Label: {result['label']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Distance: {result['distance']:.3f}")
        else:
            print(f"✗ NOT RECOGNIZED")
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Confidence: {result.get('confidence', 0):.3f} (threshold: {args.threshold})")
    else:
        # Batch processing
        results = process_image_batch(image_paths, args.model, args.threshold, args.metric)
        
        # Summary
        recognized_count = sum(1 for r in results if r.get("recognized", False))
        print(f"\nBatch Processing Summary:")
        print(f"Total images: {len(results)}")
        print(f"Recognized: {recognized_count}")
        print(f"Not recognized: {len(results) - recognized_count}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

