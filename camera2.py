import cv2
import os
import time
from datetime import datetime

CAMERA_INDEX = 0
SAVE_DIR = "captured_frames"
INTERVAL = 5  # seconds

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: cannot open camera with index {CAMERA_INDEX}")
    exit()

print(f"Capturing one frame every {INTERVAL} seconds. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to read frame.")
            break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
