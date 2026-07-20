import os
import cv2


class VideoManager:
    def __init__(self, video_path, frame_interval=0.0):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.last_capture_time_s = None

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        # Obtain the timestamp of the current frame from the video.
        position_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        if position_ms > 0:
            self.last_capture_time_s = position_ms / 1000.0
        else:
            # Fallback for codecs that do not provide CAP_PROP_POS_MSEC.
            current_frame = max(
                0.0,
                self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1.0,
            )
            self.last_capture_time_s = current_frame / fps

        return frame

    def get_capture_time_s(self):
        """Return the source-video time of the most recently read frame."""
        return self.last_capture_time_s

    def get_frame_size(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        return width, height, fps

    def get_source_metadata(self):
        return {
            "input_source": "video",
            "video_path": self.video_path,
            "video_name": os.path.basename(self.video_path),
            "total_frames": int(
                self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            ),
            "current_video_frame": int(
                self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ),
            "video_timestamp_s": self.last_capture_time_s,
        }

    def release(self):
        if self.cap is not None:
            self.cap.release()