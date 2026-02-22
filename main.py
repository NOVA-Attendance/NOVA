#!/usr/bin/env python3
"""
NOVA Edge Device - Main entry point

Attendance system for the NVIDIA Jetson Nano with RFID + camera.

The RFID reader thread captures an image immediately on each card tap and
places a task on a queue. A worker thread processes the queue by fetching
the student's face embedding from the server, running on-device comparison,
and reporting the result back. This decoupling means the reader is never
blocked by recognition and no scans are dropped.

If the server is unreachable the worker falls back to the local face index
and writes failed attendance records to disk for retry on reconnect.
"""

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    import api_client
    from recognize_image import compare_against_embedding, recognize_single_image
except ImportError as e:
    print(f"Error importing required scripts: {e}")
    sys.exit(1)

SERVER_URL   = "http://192.168.0.100:5001"
CLASS_ID     = 1
MODEL_NAME   = "Facenet512"
THRESHOLD    = 0.4
SCAN_DIR     = Path("scans")
OFFLINE_LOG  = Path("offline_queue.jsonl")
WORKER_COUNT = 1

scan_queue     = queue.Queue()
offline_mode   = threading.Event()
shutdown_event = threading.Event()
reader         = None
logger         = None

ScanTask = namedtuple("ScanTask", ["rfid_tag", "image_path", "timestamp"])


def setup_logging():
    log = logging.getLogger("nova")
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler("attendance.log")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(console_handler)
    return log


def capture_image(rfid_tag: str, timestamp: datetime) -> Path:
    """Capture one frame via GStreamer and save it to a unique path.

    Returns the output Path on success, or None on failure.
    """
    SCAN_DIR.mkdir(exist_ok=True)
    safe_tag = rfid_tag.replace("/", "_").replace("\\", "_")
    out_path = SCAN_DIR / f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{safe_tag}.jpg"
    out_path.unlink(missing_ok=True)

    cmd = [
        "gst-launch-1.0",
        "nvarguscamerasrc", "sensor-id=0", "num-buffers=1", "!",
        "video/x-raw(memory:NVMM),", "width=1280,", "height=720,", "format=NV12", "!",
        "nvvidconv", "!",
        "video/x-raw,", "format=I420", "!",
        "jpegenc", "!",
        "filesink", f"location={out_path}",
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if out_path.exists():
            logger.info(f"Image captured and saved to {out_path}")
            return out_path
        logger.error("gst-launch finished but output file is missing.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run gst-launch: {e}")
    return None


def save_offline_record(record: dict):
    """Append an attendance record to disk for later upload."""
    try:
        with open(OFFLINE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.error(f"Could not write offline record: {e}")


def flush_offline_queue():
    """Attempt to upload any records stored while the server was down."""
    if not OFFLINE_LOG.exists():
        return
    try:
        lines = OFFLINE_LOG.read_text().strip().splitlines()
    except OSError:
        return
    if not lines:
        return

    logger.info(f"Flushing {len(lines)} offline record(s) to server.")
    remaining = []
    for line in lines:
        try:
            rec = json.loads(line)
            ok = api_client.post_face_result(
                rfid_tag   = rec["rfid_tag"],
                student_id = rec["student_id"],
                class_id   = rec["class_id"],
                confidence = rec["confidence"],
                matched    = rec["matched"],
                timestamp  = datetime.fromisoformat(rec["timestamp"]),
            )
            if not ok:
                remaining.append(line)
        except Exception as e:
            logger.warning(f"Could not replay offline record: {e}")
            remaining.append(line)

    if remaining:
        OFFLINE_LOG.write_text("\n".join(remaining) + "\n")
    else:
        OFFLINE_LOG.unlink(missing_ok=True)
        logger.info("Offline queue fully flushed.")


def process_task(task: ScanTask):
    """Process a single scan task on the worker thread."""
    logger.info(f"Processing scan - RFID: {task.rfid_tag}")

    # Notify the dashboard via the existing tap endpoint
    api_client.rfid_scan(task.rfid_tag)

    student = None
    if not offline_mode.is_set():
        student = api_client.get_student_by_rfid(task.rfid_tag)
        if student is None:
            logger.warning(f"RFID {task.rfid_tag} not found on server. Skipping recognition.")
            task.image_path.unlink(missing_ok=True)
            return

    server_embedding = student.get("face_embedding") if student else None

    if server_embedding:
        result = compare_against_embedding(
            image_path          = task.image_path,
            reference_embedding = server_embedding,
            model_name          = MODEL_NAME,
            threshold           = THRESHOLD,
        )
    else:
        # No server embedding available, fall back to local database
        logger.info(f"No server embedding for RFID {task.rfid_tag}, using local database.")
        result = recognize_single_image(
            image_path = task.image_path,
            model_name = MODEL_NAME,
            threshold  = THRESHOLD,
        )

    if "error" in result:
        logger.warning(f"Face recognition error: {result['error']}")

    confidence = result.get("confidence", 0.0)
    matched    = result.get("matched", False)
    student_id = (student or {}).get("student_id") or result.get("student_id")

    logger.info(f"Result - student: {student_id} | confidence: {confidence:.2%} | matched: {matched}")

    record = {
        "rfid_tag":   task.rfid_tag,
        "student_id": student_id,
        "class_id":   CLASS_ID,
        "confidence": confidence,
        "matched":    matched,
        "timestamp":  task.timestamp.isoformat(),
    }

    if offline_mode.is_set():
        save_offline_record(record)
    else:
        ok = api_client.post_face_result(**record)
        if not ok:
            save_offline_record(record)
        else:
            flush_offline_queue()

    task.image_path.unlink(missing_ok=True)


def worker_thread_fn():
    """Worker thread that pops scan tasks from the queue and processes them."""
    logger.info("Worker thread started.")
    while not shutdown_event.is_set():
        try:
            task = scan_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            process_task(task)
        except Exception as e:
            logger.error(f"Unhandled error in worker: {e}", exc_info=True)
        finally:
            scan_queue.task_done()
    logger.info("Worker thread stopped.")


def init():
    global reader, logger

    logger = setup_logging()
    logger.info("System initialization started.")

    api_client.SERVER_URL = SERVER_URL

    if os.getuid() != 0 and os.geteuid() != 0:
        logger.error("Insufficient permissions: program must be run as root or with sudo.")
        sys.exit(1)
    logger.debug("INIT: Root permissions confirmed.")

    try:
        subprocess.run(
            ["gst-launch-1.0", "--version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        logger.debug("INIT: GStreamer tools found.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("Missing dependency: 'gst-launch-1.0' not found. Install with: sudo apt install gstreamer1.0-tools")
        sys.exit(1)

    if not os.path.exists("/dev/video0"):
        logger.error("Hardware Error: Camera device '/dev/video0' not found. Check ribbon cable connection.")
        sys.exit(1)
    logger.debug("INIT: Camera device (/dev/video0) detected.")

    try:
        status = subprocess.call(["systemctl", "is-active", "--quiet", "nvargus-daemon"])
        if status != 0:
            logger.warning("Camera Service Issue: 'nvargus-daemon' is not active. Attempting to restart...")
            subprocess.run(["systemctl", "restart", "nvargus-daemon"], check=True)
            time.sleep(2)
            logger.info("Camera service restarted successfully.")
        logger.debug("INIT: nvargus-daemon service is active.")
    except Exception as e:
        logger.warning(f"Could not check/restart nvargus-daemon: {e}")

    try:
        from Jetson_MFRC522 import SimpleMFRC522
        logger.debug("INIT: RFID library imported successfully.")
    except ModuleNotFoundError as e:
        logger.error(f"RFID library not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error importing RFID library: {e}")
        sys.exit(1)

    try:
        reader = SimpleMFRC522()
        logger.debug("INIT: RFID reader initialized successfully.")
    except FileNotFoundError as e:
        logger.error(f"RFID reader initialization failed: {e}\nEnsure spidev driver is loaded.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error initializing RFID reader: {e}")
        sys.exit(1)

    if api_client.check_server_health():
        logger.info(f"INIT: Server connection established at {SERVER_URL}.")
    else:
        logger.warning("Unable to reach server. Entering offline mode.")
        offline_mode.set()

    logger.info("Initialization complete.")


def main():
    init()

    for i in range(WORKER_COUNT):
        t = threading.Thread(target=worker_thread_fn, name=f"worker-{i}", daemon=True)
        t.start()

    logger.info("Waiting for RFID...")
    try:
        while True:
            rfid_tag, _text = reader.read()
            timestamp = datetime.now()
            logger.debug(f"RFID read: {rfid_tag}")

            image_path = capture_image(str(rfid_tag), timestamp)
            if image_path is None:
                logger.warning(f"Image capture failed for RFID {rfid_tag}, skipping.")
                continue

            task = ScanTask(rfid_tag=str(rfid_tag), image_path=image_path, timestamp=timestamp)
            scan_queue.put(task)
            logger.info(f"Task queued. Queue depth: {scan_queue.qsize()}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        raise
    finally:
        shutdown_event.set()
        scan_queue.join()


if __name__ == "__main__":
    main()