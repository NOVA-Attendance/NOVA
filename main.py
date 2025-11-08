#!/usr/bin/env python3

from time import sleep
import sys
import os
import requests
import logging

SERVER_URL = "192.168.0.100"
OFFLINE_MODE = False
reader = None
logger = None

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

    # Try to import RFID library, most likely to fail...
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
            logger.info("Waiting for RFID")
            id, text = reader.read()
            logger.debug(f"ID: {id} | Text: {text}")
            send_attendance(id)
            sleep(1)
    except KeyboardInterrupt:
        raise

if __name__ == "__main__":
    main()
