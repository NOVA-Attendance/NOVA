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

    # Check for root/sudo
    if os.getuid() != 0 and os.geteuid() != 0:
        logger.error("Insufficient permissions: program must be run as root or with sudo.")
        sys.exit(1)

    # Try to import RFID library
    try:
        from Jetson_MFRC522 import SimpleMFRC522
    except ModuleNotFoundError as e:
        logger.error(f"RFID library not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error importing RFID library: {e}")
        sys.exit(1)

    # Initialize RFID reader
    try:
        reader = SimpleMFRC522()
        logger.info("RFID reader initialized successfully.")
    except FileNotFoundError as e:
        logger.error(f"RFID reader initialization failed: {e}\nEnsure spidev driver is loaded.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error initializing RFID reader: {e}")
        sys.exit(1)

    # Check server connectivity
    try:
        response = requests.get(SERVER_URL, timeout=5)
        logger.debug(response)
        if response.status_code == 200:
            logger.info("Connected to server successfully.")
        else:
            logger.warning(f"Server returned status {response.status_code}. Entering offline mode.")
            OFFLINE_MODE = True
    except requests.RequestException:
        logger.warning("Unable to reach server. Entering offline mode.")
        OFFLINE_MODE = True

def send_attendance(id_value):
    global OFFLINE_MODE
    if OFFLINE_MODE:
        logger.info(f"Offline mode: recorded attendance locally for ID {id_value}")
        return

    try:
        response = requests.post(
            SERVER_URL,
            json={"id": id_value},
            timeout=5
        )
        if response.status_code == 200:
            logger.info(f"Attendance recorded successfully for ID {id_value}")
        else:
            logger.warning(f"Server returned status {response.status_code}: {response.text}")
    except requests.RequestException as e:
        logger.error(f"Failed to send attendance: {e}")
        logger.warning("Switching to offline mode.")
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