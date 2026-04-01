import cv2
import os
import time
from datetime import datetime


class CameraManager:
    def __init__(self, camera_index=0, save_dir="captured_frames", interval=5):
        self.camera_index = camera_index
        self.save_dir = save_dir
        self.interval = interval
        self.frame_count = 0

        os.makedirs(self.save_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Error: cannot open camera with index {self.camera_index}"
            )

        print(f"Camera {self.camera_index} initialized.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: failed to read frame.")
        return frame

    def get_encoded_frame(self, format=".jpg", quality=90):
        frame = self.get_frame()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        success, buffer = cv2.imencode(format, frame, encode_param)
        if not success:
            raise RuntimeError("Error: failed to encode frame.")

        return buffer.tobytes()

    def save_frame(self, filename=None):
        frame = self.get_frame()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.save_dir,
                f"frame_{timestamp}_{self.frame_count}.jpg"
            )

        success = cv2.imwrite(filename, frame)
        if not success:
            raise RuntimeError(f"Error: failed to save frame to {filename}")

        self.frame_count += 1
        print(f"Saved: {filename}")
        return filename

    def capture_periodically(self):
        print(
            f"Capturing one frame every {self.interval} seconds. Press Ctrl+C to stop."
        )

        try:
            while True:
                self.save_frame()
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("Stopped by user.")

    def release(self):
        if self.cap is not None:
            self.cap.release()
            print("Camera released.")