#!/usr/bin/env python3
"""
Live webcam demo to display a deterministic facial user ID derived
from DeepFace embeddings with face enrollment capability. Press 'q' to quit.

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
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace


def hash_embedding(embedding: list[float], length: int = 16) -> str:
    """Turn an embedding into a short, stable text identifier."""
    serialized = json.dumps(embedding, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest[:length]


def try_compute_embedding(frame_bgr, model_name: str):
    """Detect a face in the frame and compute its embedding if found."""
    try:
        # Resize large frames for better performance on all devices
        height, width = frame_bgr.shape[:2]
        if width > 800:  # Resize if too large - improves speed on all platforms
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            frame_bgr = cv2.resize(frame_bgr, (new_width, new_height))
        
        # DeepFace expects RGB images
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Use opencv detector - fastest across all platforms (Mac, Windows, Linux, Jetson)
        reps = DeepFace.represent(
            img_path=frame_rgb,
            model_name=model_name,
            detector_backend="opencv",  # Universal optimization: 3-5x faster than retinaface
            enforce_detection=True,
        )
        rep = reps[0] if isinstance(reps, list) and reps else reps
        embedding = rep.get("embedding") if isinstance(rep, dict) else rep
        if not embedding:
            return None, "no embedding in response"
        return embedding, None
    except Exception as error:  # noqa: BLE001
        return None, str(error)


def save_face_to_enrollment(face_image, facial_id, student_id=None):
    """Persist the current face crop under data/images for later enrollment."""
    try:
        data_dir = Path("data")
        images_dir = data_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Create student directory if student_id provided, otherwise use facial_id
        student_dir = images_dir / (student_id or facial_id)
        student_dir.mkdir(exist_ok=True)
        
        # Save the face image
        timestamp = int(time.time())
        image_path = student_dir / f"face_{timestamp}.jpg"
        cv2.imwrite(str(image_path), face_image)
        
        # Update students.json
        students_path = data_dir / "students.json"
        students = []
        if students_path.exists():
            try:
                students = json.loads(students_path.read_text())
            except:
                students = []
        
        # Check if student already exists
        student_exists = any(s.get("student_id") == (student_id or facial_id) for s in students)
        if not student_exists:
            students.append({
                "student_id": student_id or facial_id,
                "label": f"student_{student_id or facial_id}",
                "images_dir": str(student_dir)
            })
            
            with open(students_path, 'w') as f:
                json.dump(students, f, indent=2)
        
        return str(image_path), True
    except Exception as e:
        return str(e), False


def draw_face_rectangle(frame, face_region=None):
    """Draw rectangle around detected face."""
    if face_region is not None:
        x, y, w, h = face_region
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame[y:y+h, x:x+w]  # Return cropped face
    return None


def scan_saved_image(image_path: str, model_name: str):
    """Scan a saved image for face recognition - new feature for image processing."""
    # Import the recognition function directly to avoid module path issues
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from recognize_image import recognize_single_image
    
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"Scanning saved image: {image_file.name}")
    print("-" * 50)
    
    result = recognize_single_image(image_file, model_name)
    
    if result.get("recognized", False):
        print(f"✓ FACE RECOGNIZED!")
        print(f"  Student ID: {result.get('student_id', 'Unknown')}")
        print(f"  Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
    else:
        print(f"✗ FACE NOT RECOGNIZED")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Confidence: {result.get('confidence', 0):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam facial user ID demo with enrollment")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model name")
    parser.add_argument(
        "--interval",
        type=int,
        default=12,  # Universal optimization: balanced performance for all devices
        help="Compute embedding every N frames (reduce CPU usage)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (0 is default; try 1 if 0 fails)",
    )
    # removed recognition threshold/metric (restore original simple mode)
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Enable face enrollment mode (saves detected faces)",
    )
    parser.add_argument(
        "--student-id",
        type=str,
        help="Student ID for enrollment (required if --enroll is used)",
    )
    parser.add_argument(
        "--scan-image",
        type=str,
        help="Scan a saved image instead of using webcam",
    )
    # New: optional live recognition against enrolled DB (non-breaking default off)
    parser.add_argument(
        "--recognize-live",
        action="store_true",
        help="Compare webcam face to enrolled database and overlay recognition result",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Recognition threshold (0-1) used when --recognize-live is set",
    )
    parser.add_argument(
        "--metric",
        choices=["cos", "l2"],
        default="cos",
        help="Distance metric used when --recognize-live is set",
    )
    parser.add_argument(
        "--backend",
        choices=["deepface", "lbph"],
        default="deepface",
        help="Recognition backend",
    )
    args = parser.parse_args()

    if args.enroll and not args.student_id:
        print("Error: --student-id is required when using --enroll")
        return
    
    # New feature: Scan saved image instead of webcam
    if args.scan_image:
        scan_saved_image(args.scan_image, args.model)
        return

    # Universal camera optimization - works on Mac, Windows, Linux, Jetson
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)  # Best for Mac
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # Best for Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)  # Fallback for Linux/Jetson
    if not cap.isOpened():
        raise SystemExit("Error: Could not open webcam.")

    # Universal camera settings - balanced performance for all devices
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Optimized resolution: good quality + speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 4:3 aspect ratio works well everywhere
    cap.set(cv2.CAP_PROP_FPS, 30)            # Standard FPS for smooth video
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce latency on all platforms

    last_id = ""
    last_error = ""
    frame_count = 0
    fps_last_time = time.time()
    fps = 0.0
    face_detected = False
    face_saved = False
    last_save_time = 0
    save_cooldown = 3  # Don't save more than once every 3 seconds

    print("Starting webcam... Press 'q' to quit, 's' to save current face (enrollment mode)")
    if args.recognize_live:
        print("Live recognition enabled: will compare to enrolled database")

    # Load database for live recognition (only if requested)
    database_embeddings = None
    database_labels = None
    if args.recognize_live:
        try:
            from pathlib import Path as _P
            import numpy as _np
            import json as _json
            data_dir = _P("data")
            npz = _np.load(data_dir / "face_index.npz")
            database_embeddings = np.array(npz["embeddings"], dtype=np.float32)
            database_labels = _json.loads((data_dir / "labels.json").read_text())
        except Exception as _e:
            print(f"Warning: live recognition disabled (DB not available): {_e}")
            args.recognize_live = False
    
    # Restore original simple mode (no live DB recognition here)
    if args.enroll:
        print(f"Enrollment mode: Will save faces for student ID: {args.student_id}")

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

        # Process face detection every N frames
        if frame_count % max(args.interval, 1) == 0:
            embedding, err = try_compute_embedding(frame, args.model)
            if embedding is not None:
                last_id = hash_embedding([float(x) for x in embedding])
                last_error = ""
                face_detected = True
                recognized = False
                recognized_label = None
                recognized_conf = 0.0
                if args.recognize_live and database_embeddings is not None:
                    query = np.array(embedding, dtype=np.float32)
                    if args.metric == "cos":
                        q = query / (np.linalg.norm(query) + 1e-8)
                        dists = [1.0 - float(np.dot(q, e / (np.linalg.norm(e) + 1e-8))) for e in database_embeddings]
                        confs = [max(0.0, 1.0 - d) for d in dists]
                    else:
                        dists = [float(np.linalg.norm(query - e)) for e in database_embeddings]
                        confs = [max(0.0, 1.0 / (1.0 + d)) for d in dists]
                    idx = int(np.argmin(dists))
                    recognized_conf = float(confs[idx])
                    recognized_label = database_labels[idx] if database_labels else None
                    recognized = recognized_conf >= args.threshold
            else:
                last_error = err or "embedding failed"
                face_detected = False

        # Overlay info with better visual feedback
        overlay_lines = [
            f"Model: {args.model}",
            f"FPS: {fps:.1f}",
            f"Facial ID: {last_id or 'N/A'}",
        ]
        
        if face_detected:
            if args.recognize_live and database_embeddings is not None:
                if recognized:
                    overlay_lines.append(f"✓ Recognized in system ({recognized_conf:.2f})")
                    if recognized_label:
                        overlay_lines.append(f"Label: {recognized_label}")
                else:
                    overlay_lines.append("Face detected (unknown)")
            else:
                overlay_lines.append("✓ Face Detected")
        else:
            overlay_lines.append("✗ No Face")
            
        if last_error:
            overlay_lines.append(f"Error: {last_error[:50]}")
            
        if args.enroll:
            overlay_lines.append(f"Enrollment: {args.student_id}")
            if face_saved:
                overlay_lines.append("✓ Face Saved!")

        # Draw overlay with better visibility
        y = 30
        for i, line in enumerate(overlay_lines):
            color = (0, 255, 0) if "✓" in line else (0, 255, 255) if "✗" in line else (255, 255, 255)
            thickness = 2 if "✓" in line or "✗" in line else 1
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += 30

        # Add instructions
        cv2.putText(
            frame,
            "Press 'q' to quit, 's' to save face",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("NOVA - Facial User ID (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("s") and args.enroll and face_detected and last_id:
            # Save current face
            if now - last_save_time > save_cooldown:
                image_path, success = save_face_to_enrollment(frame, last_id, args.student_id)
                if success:
                    print(f"Face saved to: {image_path}")
                    face_saved = True
                    last_save_time = now
                else:
                    print(f"Failed to save face: {image_path}")
            else:
                print("Please wait before saving another face...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()