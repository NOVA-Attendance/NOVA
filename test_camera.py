#!/usr/bin/env python3
"""Quick camera test - press 'q' to quit"""
import cv2
import time

print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

print("Camera opened successfully!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame")
        break
    
    # Add text overlay
    cv2.putText(frame, "Camera Test - Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("NOVA Camera Test", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done")

