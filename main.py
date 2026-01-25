#!/usr/bin/env python3

from time import sleep
import sys
import os
import requests
import logging
import subprocess # Added for calling gst-launch
from pathlib import Path

# Add scripts folder to path to allow importing recognition logic
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Try to import the recognition function from the existing script
try:
    from recognize_image import recognize_single_image
except ImportError as e:
    print(f"Error importing recognition script: {e}")
    recognize_single_image = None

SERVER_URL = "http://192.168.0.100" # Fixed missing schema
OFFLINE_MODE = False
reader = None
logger = None

def capture_and_save_image(output_path="last_scan.jpg"):
    """
    Captures an image using the system's gst-launch-1.0 utility.
    This bypasses the need for OpenCV to have GStreamer support compiled in.
    """
    # Remove existing file if it exists to ensure we don't process old data
    if os.path.exists(output_path):
        os.remove(output_path)

    # GStreamer pipeline to capture 1 frame and save as JPEG
    # We capture 10 buffers (num-buffers=10) to allow auto-exposure to settle, 
    # but only save the last one by using the 'jpegenc' snapshot behavior or just overwriting.
    # Actually, simpler approach: Use nvjpegenc which is optimized for Jetson.
    
    cmd = [
        "gst-launch-1.0",
        "nvarguscamerasrc", "sensor-id=0", "num-buffers=1", "!" ,
        "video/x-raw(memory:NVMM),", "width=1280,", "height=720,", "format=NV12", "!",
        "nvvidconv", "!",
        "video/x-raw,", "format=I420", "!",
        "jpegenc", "!",
        "filesink", f"location={output_path}"
    ]

    try:
        # Run the command and wait for it to finish
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(output_path):
            logger.info(f"Image captured and saved to {output_path}")
            return True
        else:
            logger.error("gst-launch finished but output file is missing.")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run gst-launch: {e}")
        return False

def init():
    global reader, OFFLINE_MODE, logger

    # Initialize logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("attendance.log")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("System initialization started.")

    # 1. Check Root Permissions
    if os.getuid() != 0 and os.geteuid() != 0:
        logger.error("Insufficient permissions: program must be run as root or with sudo.")
        sys.exit(1)
    logger.debug("INIT: Root permissions confirmed.")

    # 2. Check for GStreamer tools
    try:
        subprocess.run(
            ["gst-launch-1.0", "--version"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        logger.debug("INIT: GStreamer tools found.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("Missing dependency: 'gst-launch-1.0' not found. Install with: sudo apt install gstreamer1.0-tools")
        sys.exit(1)

    # 3. Check if Camera Device Exists
    if not os.path.exists("/dev/video0"):
        logger.error("Hardware Error: Camera device '/dev/video0' not found. Check ribbon cable connection.")
        sys.exit(1)
    logger.debug("INIT: Camera device (/dev/video0) detected.")

    # 4. Check if nvargus-daemon is running
    try:
        status = subprocess.call(["systemctl", "is-active", "--quiet", "nvargus-daemon"])
        if status != 0:
            logger.warning("Camera Service Issue: 'nvargus-daemon' is not active. Attempting to restart...")
            subprocess.run(["systemctl", "restart", "nvargus-daemon"], check=True)
            sleep(2)
            logger.info("Camera service restarted successfully.")
        logger.debug("INIT: nvargus-daemon service is active.")
    except Exception as e:
        logger.warning(f"Could not check/restart nvargus-daemon: {e}")

    # 5. Check RFID Library
    try:
        from Jetson_MFRC522 import SimpleMFRC522
        logger.debug("INIT: RFID library imported successfully.")
    except ModuleNotFoundError as e:
        logger.error(f"RFID library not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error importing RFID library: {e}")
        sys.exit(1)

    # 6. Initialize RFID Reader
    try:
        reader = SimpleMFRC522()
        logger.debug("INIT: RFID reader initialized successfully.")
    except FileNotFoundError as e:
        logger.error(f"RFID reader initialization failed: {e}\nEnsure spidev driver is loaded.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error initializing RFID reader: {e}")
        sys.exit(1)

    # 7. Check Server Connectivity
    try:
        response = requests.get(SERVER_URL, timeout=5)
        if response.status_code == 200:
            logger.debug("INIT: Server connection established.")
        else:
            logger.warning(f"Server returned status {response.status_code}. Entering offline mode.")
            OFFLINE_MODE = True
    except requests.RequestException:
        logger.warning("Unable to reach server. Entering offline mode.")
        OFFLINE_MODE = True

def main():
    init()
    try:
        while True:
            logger.info("Waiting for RFID...")
            id, text = reader.read()
            logger.debug(f"ID: {id} | Text: {text}")
            
            # --- Start: Camera & Recognition Logic ---
            capture_path = "captured_scan.jpg"
            logger.info("RFID read complete. Activating camera...")
            
            # Now using the subprocess version
            if capture_and_save_image(capture_path):
                if recognize_single_image:
                    logger.info("Processing face recognition...")
                    result = recognize_single_image(Path(capture_path), model_name="Facenet512")
                    
                    if result.get("recognized"):
                        student_id = result.get("student_id", "Unknown")
                        conf = result.get("confidence", 0)
                        logger.info(f"FACE MATCHED! Student ID: {student_id} (Confidence: {conf:.2f})")
                    else:
                        logger.info("Face not recognized in database.")
                else:
                    logger.warning("Recognition function not available. Skipping face check.")
            else:
                logger.warning("Skipping recognition due to camera failure.")
            # --- End: Camera & Recognition Logic ---

            send_attendance(id)
            sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        raise

if __name__ == "__main__":
    main()