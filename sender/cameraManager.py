import cv2
import os
import time
from datetime import datetime






class CameraManager:
    def __init__(self, camera_index=0, frame_interval=30, save_dir="captured_frames"):
        self.camera_index = camera_index
        self.save_dir = save_dir
        self.frame_interval = frame_interval
        os.makedirs(self.save_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: cannot open camera with index {self.camera_index}")

        print(f"Camera {self.camera_index} initialized.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: failed to read frame.")
        return frame

    def get_frame_size(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0 or fps != fps:
            fps = 20.0

        return width, height, fps

    def release(self):
        if self.cap is not None:
            self.cap.release()
            print("Camera released.")