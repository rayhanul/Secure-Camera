import os
import cv2


class VideoManager:
    def __init__(self, video_path, frame_interval=2):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def get_frame_size(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        return width, height, fps

    def get_source_metadata(self):
        return {
            "input_source": "video",
            "video_path": self.video_path,
            "video_name": os.path.basename(self.video_path),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "current_video_frame": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
        }

    def release(self):
        if self.cap is not None:
            self.cap.release()