#!/usr/bin/env python3
"""
Enroll faces from images directory into a searchable index.

This script processes all images in the data/images/ directory and creates
a face index that can be used for fast face recognition.

Usage:
  python scripts/enroll_faces.py --model Facenet512
"""

import argparse
import json
from pathlib import Path

import numpy as np
from deepface import DeepFace


def compute_embedding(image_path: Path, model_name: str):
    """Compute and return a face embedding for one image path."""
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
            return None
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Enroll faces into searchable index")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model name")
    parser.add_argument(
        "--images-dir",
        default="data/images",
        help="Directory of images. Supports subfolders per person or one flat folder.",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    images_dir = Path(args.images_dir)
    students_path = data_dir / "students.json"
    
    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}")
        return

    # Load student_id -> label map so results are human friendly
    students = {}
    if students_path.exists():
        try:
            for student in json.loads(students_path.read_text()):
                students[student["student_id"]] = student["label"]
        except Exception as e:
            print(f"Warning: Could not load students.json: {e}")

    embeddings = []
    labels = []
    
    print("Processing images...")
    processed_count = 0
    error_count = 0

    # Determine if directory is hierarchical (subfolders) or flat
    subdirs = [p for p in images_dir.iterdir() if p.is_dir()]

    if subdirs:
        # Hierarchical mode: each subfolder is a person
        for student_dir in subdirs:
            student_id = student_dir.name
            label = students.get(student_id, f"student_{student_id}")
            print(f"Processing {student_id}...")
            image_files = (
                list(student_dir.glob("*.jpg"))
                + list(student_dir.glob("*.jpeg"))
                + list(student_dir.glob("*.png"))
                + list(student_dir.glob("*.webp"))
            )
            for image_path in image_files:
                embedding = compute_embedding(image_path, args.model)
                if embedding is not None:
                    embeddings.append(embedding)
                    labels.append(label)
                    processed_count += 1
                    print(f"  ✓ {image_path.name}")
                else:
                    error_count += 1
                    print(f"  ✗ {image_path.name}")
    else:
        # Flat mode: enroll every image file in this single folder
        print(f"Processing flat folder: {images_dir}")
        image_files = (
            list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.jpeg"))
            + list(images_dir.glob("*.png"))
            + list(images_dir.glob("*.webp"))
        )
        for image_path in image_files:
            # Use filename stem as label when no subfolders are present
            label = image_path.stem
            embedding = compute_embedding(image_path, args.model)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(label)
                processed_count += 1
                print(f"  ✓ {image_path.name}")
            else:
                error_count += 1
                print(f"  ✗ {image_path.name}")

    if not embeddings:
        print("No valid face embeddings found!")
        return

    embeddings_array = np.array(embeddings)
    index_path = data_dir / "face_index.npz"
    np.savez_compressed(index_path, embeddings=embeddings_array)
    labels_path = data_dir / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"\nEnrollment complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Errors: {error_count} images")
    print(f"  Index saved to: {index_path}")
    print(f"  Labels saved to: {labels_path}")


if __name__ == "__main__":
    main()
