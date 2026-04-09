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
import contextlib
import io
from collections import namedtuple
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    import api_client
    from recognize_image import compare_against_embedding, recognize_single_image
except ImportError as e:
    # Avoid UnicodeEncodeError on Jetson terminals when the underlying
    # ImportError message contains broken byte sequences.
    print("Error importing required scripts:", repr(e))
    sys.exit(1)

SERVER_URL   = "http://192.168.0.164:5001"
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


def disable_console_logging(log: logging.Logger):
    """Keep file logging, remove console output (for TUI)."""
    for h in list(log.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            log.removeHandler(h)


def capture_image(rfid_tag: str, timestamp: datetime) -> Path:
    """Capture one frame via GStreamer and save it to a unique path.

    Returns the output Path on success, or None on failure.
    """
    SCAN_DIR.mkdir(exist_ok=True)
    safe_tag = rfid_tag.replace("/", "_").replace("\\", "_")
    out_path = SCAN_DIR / f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{safe_tag}.jpg"
    # out_path.unlink(missing_ok=True)

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
            ok = api_client.post_attendance_face_verify(
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
        # OFFLINE_LOG.unlink(missing_ok=True)
        logger.info("Offline queue fully flushed.")


def process_task(task: ScanTask):
    """Process a single scan task on the worker thread."""
    logger.info(f"Processing scan - RFID: {task.rfid_tag}")

    student = None
    if not offline_mode.is_set():
        student = api_client.get_rfid_face_embedding(task.rfid_tag)
        if student is None:
            logger.warning(f"RFID {task.rfid_tag} not found on server. Skipping recognition.")
            # task.image_path.unlink(missing_ok=True)
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
    matched    = result.get("matched", result.get("recognized", False))
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
    elif student_id is None:
        logger.warning("No student_id resolved; skipping server POST.")
        save_offline_record(record)
    else:
        ok = api_client.post_attendance_face_verify(**record, image_path=task.image_path)
        if not ok:
            save_offline_record(record)
        else:
            flush_offline_queue()

    # task.image_path.unlink(missing_ok=True)


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


def rfid_reader_thread_fn(event_queue: "queue.Queue[tuple]"):
    """Blocking RFID read loop that emits UI events and enqueues scan tasks."""
    import contextlib
    import io
    import os

    @contextlib.contextmanager
    def _suppress_terminal_output():
        """
        Suppress both Python-level stdout/stderr and OS-level fd(1/2) output.
        Some RFID libraries write AUTH ERROR directly to the terminal, which
        will corrupt curses.
        """
        buf = io.StringIO()
        devnull_fd = None
        saved_out = None
        saved_err = None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                saved_out = os.dup(1)
                saved_err = os.dup(2)
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                yield buf
        finally:
            try:
                if saved_out is not None:
                    os.dup2(saved_out, 1)
            except Exception:
                pass
            try:
                if saved_err is not None:
                    os.dup2(saved_err, 2)
            except Exception:
                pass
            for fd in (saved_out, saved_err, devnull_fd):
                try:
                    if fd is not None:
                        os.close(fd)
                except Exception:
                    pass

    logger.info("RFID reader thread started.")
    while not shutdown_event.is_set():
        try:
            # Capture and discard any RFID library terminal output (AUTH ERROR, etc).
            with _suppress_terminal_output() as buf:
                rfid_tag, _text = reader.read()
            noise = (buf.getvalue() or "").strip()
            if noise:
                # Hide all library output from the user; log to file for debugging.
                logger.debug(f"RFID library output suppressed: {noise!r}")

            timestamp = datetime.now()
            event_queue.put(("rfid_read", str(rfid_tag), timestamp))

            image_path = capture_image(str(rfid_tag), timestamp)
            if image_path is None:
                logger.warning(f"Image capture failed for RFID {rfid_tag}, skipping.")
                event_queue.put(("capture_failed", str(rfid_tag), timestamp))
                continue

            event_queue.put(("image_captured", str(rfid_tag), timestamp, str(image_path)))
            task = ScanTask(rfid_tag=str(rfid_tag), image_path=image_path, timestamp=timestamp)
            scan_queue.put(task)
            event_queue.put(("task_queued", str(rfid_tag), timestamp, scan_queue.qsize()))
        except Exception as e:
            logger.error(f"Unexpected error in RFID reader loop: {e}", exc_info=True)
            event_queue.put(("reader_error", repr(e), datetime.now()))
            time.sleep(0.5)
    logger.info("RFID reader thread stopped.")


def run_tui(event_queue: "queue.Queue[tuple]"):
    """Curses TUI that shows state and scales to terminal size."""
    import curses

    state = {
        "status": "Initializing...",
        "detail": "",
        "last_rfid": None,
        "last_image": None,
        "last_event_at": datetime.now(),
        "queue_depth": 0,
        "offline": offline_mode.is_set(),
        "error": None,
    }

    def _ascii(s: str) -> str:
        # Enforce ASCII-only output to avoid mojibake on SSH terminals.
        try:
            return (s or "").encode("ascii", "replace").decode("ascii")
        except Exception:
            return ""

    def _safe_addstr(stdscr, y, x, s, attr=0):
        try:
            h, w = stdscr.getmaxyx()
            if y < 0 or y >= h or x >= w:
                return
            if x < 0:
                s = s[-x:]
                x = 0
            if not s:
                return
            stdscr.addnstr(y, x, _ascii(str(s)), max(0, w - x - 1), attr)
        except curses.error:
            pass

    def _draw(stdscr):
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        title = "NOVA Attendance (Jetson Edge)"
        right = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _safe_addstr(stdscr, 0, 0, title, curses.color_pair(1) | curses.A_BOLD)
        _safe_addstr(stdscr, 0, max(0, w - len(right) - 1), right, curses.color_pair(1) | curses.A_BOLD)

        online_txt = "OFFLINE" if state["offline"] else "ONLINE"
        online_attr = curses.color_pair(3) | curses.A_BOLD if state["offline"] else curses.color_pair(2) | curses.A_BOLD
        _safe_addstr(stdscr, 2, 0, f"Server: {online_txt}", online_attr)
        _safe_addstr(stdscr, 2, 18, f"Queue: {state['queue_depth']}", curses.color_pair(1))

        status_attr = curses.color_pair(2) | curses.A_BOLD
        if state["error"]:
            status_attr = curses.color_pair(3) | curses.A_BOLD
        _safe_addstr(stdscr, 4, 0, f"Status: {state['status']}", status_attr)
        if state["detail"]:
            _safe_addstr(stdscr, 5, 0, state["detail"], curses.color_pair(1))

        if state["last_rfid"]:
            _safe_addstr(stdscr, 7, 0, f"Last RFID: {state['last_rfid']}", curses.color_pair(1))
        if state["last_image"]:
            _safe_addstr(stdscr, 8, 0, f"Last image: {state['last_image']}", curses.color_pair(1))

        help_txt = "Ctrl+C to exit"
        _safe_addstr(stdscr, h - 1, 0, help_txt, curses.color_pair(1))

        if w < 50 or h < 10:
            warn = "Terminal too small - enlarge for full UI"
            _safe_addstr(stdscr, 1, 0, warn, curses.color_pair(3) | curses.A_BOLD)

        stdscr.refresh()

    def _tui_main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(200)

        # Start RFID reading only after curses has initialized the terminal.
        # This avoids interactions where RFID libs (or our suppression) touch
        # stdout/stderr during curses startup.
        reader_thread = threading.Thread(
            target=rfid_reader_thread_fn,
            args=(event_queue,),
            name="rfid-reader",
            daemon=True,
        )
        reader_thread.start()

        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, -1)   # default
            curses.init_pair(2, curses.COLOR_GREEN, -1)   # success/ready
            curses.init_pair(3, curses.COLOR_RED, -1)     # error/offline
            curses.init_pair(4, curses.COLOR_CYAN, -1)    # info

        state["status"] = "Waiting for RFID scan..."
        state["detail"] = "Tap your card on the reader."
        state["error"] = None
        state["offline"] = offline_mode.is_set()
        state["queue_depth"] = scan_queue.qsize()
        _draw(stdscr)

        last_ready_at = time.monotonic()
        while not shutdown_event.is_set():
            state["offline"] = offline_mode.is_set()
            state["queue_depth"] = scan_queue.qsize()

            # Consume as many events as available between draws.
            drained = False
            while True:
                try:
                    ev = event_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                kind = ev[0]
                state["last_event_at"] = datetime.now()

                if kind == "rfid_read":
                    _kind, tag, ts = ev
                    state["last_rfid"] = tag
                    state["last_image"] = None
                    state["error"] = None
                    state["status"] = "RFID scanned."
                    state["detail"] = f"Capturing image for {tag}..."
                    last_ready_at = time.monotonic()
                elif kind == "image_captured":
                    _kind, tag, ts, img = ev
                    state["last_rfid"] = tag
                    state["last_image"] = img
                    state["error"] = None
                    state["status"] = "Image captured."
                    state["detail"] = "Queued for recognition/attendance."
                    last_ready_at = time.monotonic()
                elif kind == "task_queued":
                    _kind, tag, ts, qd = ev
                    state["queue_depth"] = int(qd)
                    state["error"] = None
                    state["status"] = "Scan complete."
                    state["detail"] = "Ready for next user..."
                    last_ready_at = time.monotonic()
                elif kind == "capture_failed":
                    _kind, tag, ts = ev
                    state["error"] = "capture_failed"
                    state["status"] = "Image capture failed."
                    state["detail"] = "Please try scanning again."
                    last_ready_at = time.monotonic()
                elif kind == "reader_error":
                    _kind, msg, ts = ev
                    state["error"] = "reader_error"
                    state["status"] = "RFID reader error."
                    state["detail"] = msg
                    last_ready_at = time.monotonic()
                # NOTE: we intentionally do not surface RFID library stdout/stderr in the TUI.

                event_queue.task_done()

            # Auto-return to "waiting" after a brief success window.
            if state["status"] in ("Scan complete.", "Image captured.", "RFID scanned.") and (time.monotonic() - last_ready_at) > 2.0:
                state["status"] = "Waiting for RFID scan..."
                state["detail"] = "Tap your card on the reader."
                state["error"] = None

            try:
                ch = stdscr.getch()
                if ch == curses.KEY_RESIZE:
                    drained = True
            except curses.error:
                pass

            if drained:
                _draw(stdscr)
            else:
                # Periodic redraw for clock / queue depth changes
                _draw(stdscr)

    curses.wrapper(_tui_main)


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

    event_queue: "queue.Queue[tuple]" = queue.Queue()
    # Switch from console logs to a TUI after initialization.
    disable_console_logging(logger)
    logger.info("TUI mode enabled (console logging disabled).")

    try:
        try:
            run_tui(event_queue)
        except Exception as e:
            # If curses fails (e.g., non-interactive environment), fall back to logs.
            logger.error(f"TUI failed to start; falling back to console logging. Error: {e}", exc_info=True)
            # Re-enable console handler by re-running setup if needed
            # (simplest: add a new console StreamHandler).
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.info("Waiting for RFID (no TUI)...")
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