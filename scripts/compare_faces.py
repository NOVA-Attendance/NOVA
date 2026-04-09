#!/usr/bin/env python3
"""
Compare two images to determine if they show the same person.

This script takes two image paths, extracts face embeddings using Facenet512,
and compares them to determine if they're the same person. It returns a
confidence score showing how similar the faces are.

Usage:
  python scripts/compare_faces.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
  python scripts/compare_faces.py --image1 img1.jpg --image2 img2.jpg --threshold 0.4
  python scripts/compare_faces.py --image1 img1.jpg --image2 img2.jpg --metric l2
"""

import argparse
import json
from pathlib import Path

import numpy as np
from deepface import DeepFace


def compute_embedding(image_path: Path, model_name: str) -> np.ndarray:
    """
    Compute a face embedding (numerical representation) from an image.
    
    This function uses DeepFace to detect a face in the image and convert it
    into a 512-dimensional vector that represents the facial features.
    The same person will have similar embeddings, different people will have
    different embeddings.
    
    Args:
        image_path: Path to the image file containing a face
        model_name: Name of the AI model to use (e.g., "Facenet512")
    
    Returns:
        A numpy array of 512 numbers representing the face
    
    Raises:
        RuntimeError: If no face is detected or embedding fails
    """
    try:
        # Use DeepFace to analyze the image and extract face features
        # detector_backend="opencv" is faster than other options
        # enforce_detection=True means it will fail if no face is found
        reps = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend="opencv",  # Fast face detector
            enforce_detection=True,  # Require a face to be found
        )
        
        # DeepFace may return a list or single dict depending on version
        # Handle both cases to be compatible
        rep = reps[0] if isinstance(reps, list) and reps else reps
        
        # Extract the embedding vector from the response
        # The embedding is a list of numbers representing facial features
        embedding = rep.get("embedding") if isinstance(rep, dict) else rep
        
        if not embedding:
            raise RuntimeError("No embedding found in DeepFace response")
        
        # Convert to numpy array for easier math operations
        return np.array(embedding, dtype=np.float32)
        
    except Exception as e:
        raise RuntimeError(f"Failed to compute embedding from {image_path}: {e}")


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine distance between two face embeddings.
    
    Cosine distance measures the angle between two vectors. Smaller distance
    means the faces are more similar. Distance ranges from 0 (identical) to 2
    (completely different).
    
    How it works:
    1. Normalize both embeddings (make them unit length)
    2. Calculate dot product (measures angle between vectors)
    3. Convert similarity to distance (1 - similarity)
    
    Args:
        embedding1: First face embedding (512 numbers)
        embedding2: Second face embedding (512 numbers)
    
    Returns:
        Distance value between 0 and 2 (smaller = more similar)
    """
    # Normalize embeddings to unit length (divide by their magnitude)
    # This makes the comparison angle-based, not magnitude-based
    # Adding 1e-8 prevents division by zero
    norm1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    norm2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
    
    # Calculate cosine similarity (dot product of normalized vectors)
    # This gives a value between -1 and 1
    # 1 = same direction (same person), -1 = opposite direction (different person)
    cosine_similarity = float(np.dot(norm1, norm2))
    
    # Convert similarity to distance: distance = 1 - similarity
    # This gives us: 0 = identical, 2 = completely different
    distance = float(1.0 - cosine_similarity)
    
    return distance


def l2_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate Euclidean (L2) distance between two face embeddings.
    
    This measures the straight-line distance between two points in 512-dimensional
    space. Smaller distance means more similar faces.
    
    Args:
        embedding1: First face embedding (512 numbers)
        embedding2: Second face embedding (512 numbers)
    
    Returns:
        Euclidean distance (smaller = more similar)
    """
    # Calculate the straight-line distance between the two embedding vectors
    # This is the standard distance formula: sqrt(sum((a-b)^2))
    return float(np.linalg.norm(embedding1 - embedding2))


def compare_two_images(image1_path: Path, image2_path: Path, 
                      model_name: str = "Facenet512", 
                      metric: str = "cos",
                      threshold: float = 0.4) -> dict:
    """
    Compare two images to see if they show the same person.
    
    This is the main function that:
    1. Extracts face embeddings from both images
    2. Calculates the distance between them
    3. Converts distance to confidence percentage
    4. Determines if they're the same person based on threshold
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        model_name: AI model to use (default: Facenet512)
        metric: Distance metric - "cos" for cosine, "l2" for Euclidean
        threshold: Minimum confidence to consider same person (0-1)
    
    Returns:
        Dictionary with comparison results including:
        - same_person: True/False if they match
        - confidence: Similarity percentage (0-100%)
        - distance: Raw distance value
        - metric: Which metric was used
    """
    # Step 1: Extract face embeddings from both images
    # These are numerical representations of facial features
    print(f"Processing image 1: {image1_path}")
    embedding1 = compute_embedding(image1_path, model_name)
    print(f"✓ Face detected in image 1")
    
    print(f"Processing image 2: {image2_path}")
    embedding2 = compute_embedding(image2_path, model_name)
    print(f"✓ Face detected in image 2")
    
    # Step 2: Calculate distance between the two embeddings
    # Smaller distance = more similar faces
    if metric == "cos":
        distance = cosine_distance(embedding1, embedding2)
    else:  # l2
        distance = l2_distance(embedding1, embedding2)
    
    # Step 3: Convert distance to confidence percentage
    # For cosine: confidence = 1 - distance (since distance is already 1-similarity)
    # For L2: confidence = 1 / (1 + distance) to normalize to 0-1 range
    if metric == "cos":
        confidence = max(0.0, 1.0 - distance)
    else:  # l2
        confidence = max(0.0, 1.0 / (1.0 + distance))
    
    # Step 4: Determine if they're the same person
    # If confidence is above threshold, consider them the same person
    same_person = confidence >= threshold
    
    # Return all the results
    return {
        "same_person": same_person,
        "confidence": float(confidence),
        "confidence_percent": float(confidence * 100),
        "distance": float(distance),
        "metric": metric,
        "threshold": threshold,
        "image1": str(image1_path),
        "image2": str(image2_path),
        "model": model_name,
    }


def main() -> None:
    """
    Main function that handles command-line arguments and runs the comparison.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Compare two images to see if they show the same person"
    )
    
    # Required arguments: paths to both images
    parser.add_argument(
        "--image1",
        required=True,
        help="Path to first image file"
    )
    parser.add_argument(
        "--image2",
        required=True,
        help="Path to second image file"
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--model",
        default="Facenet512",
        help="DeepFace model to use (default: Facenet512)"
    )
    parser.add_argument(
        "--metric",
        choices=["cos", "l2"],
        default="cos",
        help="Distance metric: 'cos' for cosine similarity (default), 'l2' for Euclidean"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold to consider same person (0-1, default: 0.4 = 40%%)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text"
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Convert string paths to Path objects and check if files exist
    image1_path = Path(args.image1)
    image2_path = Path(args.image2)
    
    if not image1_path.exists():
        print(f"Error: Image 1 not found: {image1_path}", file=__import__("sys").stderr)
        return
    
    if not image2_path.exists():
        print(f"Error: Image 2 not found: {image2_path}", file=__import__("sys").stderr)
        return
    
    # Run the comparison
    try:
        result = compare_two_images(
            image1_path=image1_path,
            image2_path=image2_path,
            model_name=args.model,
            metric=args.metric,
            threshold=args.threshold
        )
        
        # Display results
        if args.json:
            # JSON output for programmatic use
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print("\n" + "=" * 60)
            print("FACE COMPARISON RESULT")
            print("=" * 60)
            print(f"Image 1: {result['image1']}")
            print(f"Image 2: {result['image2']}")
            print(f"Model: {result['model']}")
            print(f"Metric: {result['metric']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Confidence: {result['confidence']:.3f} ({result['confidence_percent']:.1f}%)")
            print(f"Threshold: {result['threshold']:.3f} ({result['threshold']*100:.1f}%)")
            print("-" * 60)
            
            if result['same_person']:
                print("✓ SAME PERSON - Faces match!")
            else:
                print("✗ DIFFERENT PERSONS - Faces do not match")
                print(f"  (Confidence {result['confidence_percent']:.1f}% is below threshold {result['threshold']*100:.1f}%)")
            
            print("=" * 60)
    
    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        return


if __name__ == "__main__":
    main()
