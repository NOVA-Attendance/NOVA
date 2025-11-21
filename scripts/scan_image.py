#!/usr/bin/env python3
"""
Simple interactive image scanner for face recognition.

Easy-to-use script for scanning individual images and recognizing faces.

Usage:
  python scripts/scan_image.py
  python scripts/scan_image.py --image path/to/image.jpg
"""

import argparse
import json
from pathlib import Path

import numpy as np
from deepface import DeepFace


def compute_embedding(image_path: Path, model_name: str) -> np.ndarray:
    """Compute a face embedding for the provided image path."""
    try:
        reps = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend="opencv",  # Fastest across all platforms
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
    """Compute cosine distance between two embeddings."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    cos_sim = float(np.dot(a_norm, b_norm))
    return float(1.0 - cos_sim)


def load_face_database():
    """Load stored embeddings, labels and student-id mapping from disk."""
    data_dir = Path("data")
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


def scan_image(image_path: Path, model_name: str = "Facenet512", threshold: float = 0.4):
    """Recognize a single image and print a short, readable result."""
    print(f"Scanning image: {image_path}")
    print("-" * 50)
    
    try:
        # Load face database
        database_embeddings, database_labels, students = load_face_database()
        print(f"Loaded database with {len(database_embeddings)} enrolled faces")
        
        # Compute embedding for query image
        print("Computing face embedding...")
        query_embedding = compute_embedding(image_path, model_name)
        print("✓ Face detected and embedded")
        
        # Find best match
        print("Searching database...")
        distances = [cosine_distance(query_embedding, emb) for emb in database_embeddings]
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        best_label = database_labels[best_idx]
        confidence = max(0, 1 - best_distance)
        
        # Get student info
        student_id = students.get(best_label, "Unknown")
        
        print("\n" + "="*50)
        print("RECOGNITION RESULT")
        print("="*50)
        
        if confidence >= threshold:
            print(f"✓ FACE RECOGNIZED!")
            print(f"  Student ID: {student_id}")
            print(f"  Label: {best_label}")
            print(f"  Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            print(f"  Distance: {best_distance:.3f}")
        else:
            print(f"✗ FACE NOT RECOGNIZED")
            print(f"  Best match: {student_id} ({best_label})")
            print(f"  Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            print(f"  Threshold: {threshold:.3f} ({threshold*100:.1f}%)")
            print(f"  Distance: {best_distance:.3f}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")


def interactive_mode():
    """Simple CLI loop to scan one image or all images in data/images."""
    print("NOVA Image Scanner - Interactive Mode")
    print("="*40)
    
    # Check if database exists
    try:
        load_face_database()
        print("✓ Face database loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("Please run 'python3 scripts/enroll_faces.py' first to create the database.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Scan an image")
        print("2. Scan all images in data/images/")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if image_path and Path(image_path).exists():
                scan_image(Path(image_path))
            else:
                print("Invalid image path")
        
        elif choice == "2":
            images_dir = Path("data/images")
            if not images_dir.exists():
                print("No images directory found")
                continue
            
            # Find all images (support jpg, jpeg, png, webp)
            image_files = (
                list(images_dir.rglob("*.jpg"))
                + list(images_dir.rglob("*.jpeg"))
                + list(images_dir.rglob("*.png"))
                + list(images_dir.rglob("*.webp"))
            )
            if not image_files:
                print("No images found in data/images/")
                continue
            
            print(f"Found {len(image_files)} images to scan")
            for image_file in image_files:
                scan_image(image_file)
                print()
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")


def main():
    parser = argparse.ArgumentParser(description="Simple image scanner for face recognition")
    parser.add_argument("--image", help="Path to image to scan")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model name")
    parser.add_argument("--threshold", type=float, default=0.4, help="Recognition threshold (0-1)")
    args = parser.parse_args()

    if args.image:
        # Single image mode
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            return
        scan_image(Path(args.image), args.model, args.threshold)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()

