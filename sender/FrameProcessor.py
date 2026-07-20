

import os
import cv2
import time
import math
import base64
import numpy as np
from datetime import datetime

from PIL import Image
from torchvision import transforms as T
from ultralytics import YOLO



class FrameProcessor:
    def __init__(
        self,
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        jpeg_quality=70,
        calibration=None,
        tracker="bytetrack.yaml",
        speed_alpha=0.35,
        stationary_threshold_mps=0.30,
    ):
        self.yolo_model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.jpeg_quality = jpeg_quality
        self.tracker = tracker

        # Per-track state used to calculate velocity between frames.
        self.track_state = {}
        self.speed_alpha = float(speed_alpha)
        self.stationary_threshold_mps = float(stationary_threshold_mps)

        # Camera-to-road calibration. The homography converts the bottom-center
        # point of a bounding box from image pixels to road-plane meters.
        self.homography = None
        self.camera_world_xy_m = np.array([0.0, 0.0], dtype=np.float32)
        self.camera_height_m = 0.0

        if calibration is not None:
            self._configure_calibration(calibration)

        self.person_transform = T.Compose(
            [
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.vehicle_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _configure_calibration(self, calibration):
        """Create an image-to-road homography from measured point pairs."""
        image_points = np.asarray(
            calibration["image_points_px"], dtype=np.float32
        )
        world_points = np.asarray(
            calibration["world_points_m"], dtype=np.float32
        )

        if image_points.ndim != 2 or image_points.shape[1] != 2:
            raise ValueError("image_points_px must be an N x 2 array")
        if world_points.ndim != 2 or world_points.shape[1] != 2:
            raise ValueError("world_points_m must be an N x 2 array")
        if len(image_points) != len(world_points):
            raise ValueError(
                "image_points_px and world_points_m must have equal lengths"
            )
        if len(image_points) < 4:
            raise ValueError("At least four calibration point pairs are required")

        if len(image_points) > 4:
            # The reprojection threshold is expressed in the target-plane unit,
            # which is meters for world_points_m.
            self.homography, _ = cv2.findHomography(
                image_points,
                world_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=0.25,
            )
        else:
            self.homography, _ = cv2.findHomography(
                image_points,
                world_points,
                method=0,
            )

        if self.homography is None:
            raise ValueError("Unable to calculate the camera homography")

        self.camera_world_xy_m = np.asarray(
            calibration.get("camera_world_xy_m", [0.0, 0.0]),
            dtype=np.float32,
        )
        if self.camera_world_xy_m.shape != (2,):
            raise ValueError("camera_world_xy_m must contain [x, y]")

        self.camera_height_m = float(
            calibration.get("camera_height_m", 0.0)
        )

    def detect_objects(self, frame):
        """Detect and track persons and vehicles using YOLO."""
        results = self.yolo_model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf_threshold,
            classes=[0, 2, 3, 5, 7],
            verbose=False,
        )
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                if confidence < self.conf_threshold:
                    continue

                if class_id == 0:
                    class_name = "person"
                elif class_id in [2, 3, 5, 7]:
                    class_name = "vehicle"
                else:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_name": class_name,
                        "confidence": confidence,
                        "track_id": track_id,
                    }
                )

        return detections, results

    def crop_and_preprocess(self, frame, detection):
        """
        Keep preprocessing available in the class for compatibility,
        though current network payload sends crop_jpg_b64 instead.
        """
        x1, y1, x2, y2 = detection["bbox"]
        class_name = detection["class_name"]

        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None, None

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None, None

        pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        processed = None
        if class_name == "person":
            processed = self.person_transform(pil_image)
        elif class_name == "vehicle":
            processed = self.vehicle_transform(pil_image)

        return cropped, processed

    def _pixel_to_world(self, pixel_x, pixel_y):
        """Convert one image point to local road-plane coordinates in meters."""
        if self.homography is None:
            return None

        image_point = np.array(
            [[[float(pixel_x), float(pixel_y)]]], dtype=np.float32
        )
        world_point = cv2.perspectiveTransform(
            image_point, self.homography
        )[0, 0]

        return float(world_point[0]), float(world_point[1])

    def _classify_motion(
        self,
        velocity_x,
        velocity_y,
        relative_x,
        relative_y,
        speed_mps,
    ):
        """Describe motion relative to the camera."""
        if speed_mps < self.stationary_threshold_mps:
            return "stationary"

        ground_distance = math.hypot(relative_x, relative_y)
        raw_speed = math.hypot(velocity_x, velocity_y)

        if ground_distance <= 1e-6 or raw_speed <= 1e-6:
            return "unknown"

        radial_motion = (
            velocity_x * relative_x + velocity_y * relative_y
        ) / (raw_speed * ground_distance)

        if radial_motion < -0.35:
            return "approaching"
        if radial_motion > 0.35:
            return "moving_away"
        if velocity_x > 0:
            return "crossing_right"
        return "crossing_left"

    def _estimate_kinematics(self, bbox, track_id, capture_time_s):
        """Estimate position, distance, bearing, movement direction and speed."""
        x1, _, x2, y2 = bbox

        # For a road-plane homography, the bottom-center of the box is the best
        # approximation of the object's contact point with the ground.
        ground_x_px = (x1 + x2) / 2.0
        ground_y_px = float(y2)

        kinematics = {
            "kinematics_available": False,
            "ground_point_px": [
                round(ground_x_px, 2),
                round(ground_y_px, 2),
            ],
            "world_position_m": None,
            "ground_distance_m": None,
            "distance_m": None,
            "bearing_angle_deg": None,
            "velocity_mps": None,
            "motion_direction_deg": None,
            "motion_direction": "unknown",
            "speed_mps": None,
            "speed_kmh": None,
        }

        world_position = self._pixel_to_world(ground_x_px, ground_y_px)
        if world_position is None:
            return kinematics

        world_x, world_y = world_position
        camera_x = float(self.camera_world_xy_m[0])
        camera_y = float(self.camera_world_xy_m[1])
        relative_x = world_x - camera_x
        relative_y = world_y - camera_y

        ground_distance_m = math.hypot(relative_x, relative_y)
        distance_m = math.sqrt(
            ground_distance_m ** 2 + self.camera_height_m ** 2
        )

        # Calibration convention: +x is camera-right and +y is forward.
        # Therefore, 0 degrees is straight ahead and positive angles are right.
        bearing_angle_deg = math.degrees(
            math.atan2(relative_x, relative_y)
        )

        kinematics.update(
            {
                "kinematics_available": True,
                "world_position_m": [
                    round(world_x, 3),
                    round(world_y, 3),
                ],
                "ground_distance_m": round(ground_distance_m, 3),
                "distance_m": round(distance_m, 3),
                "bearing_angle_deg": round(bearing_angle_deg, 2),
            }
        )

        # Distance and bearing require only one observation. Speed and movement
        # direction require the same tracked object in at least two frames.
        if track_id is None:
            return kinematics

        previous = self.track_state.get(track_id)
        smoothed_speed_mps = None

        if previous is not None:
            elapsed_s = capture_time_s - previous["time_s"]

            if elapsed_s > 1e-6:
                velocity_x = (world_x - previous["world_x"]) / elapsed_s
                velocity_y = (world_y - previous["world_y"]) / elapsed_s
                raw_speed_mps = math.hypot(velocity_x, velocity_y)
                previous_speed_mps = previous.get("speed_mps")

                if previous_speed_mps is None:
                    smoothed_speed_mps = raw_speed_mps
                else:
                    smoothed_speed_mps = (
                        self.speed_alpha * raw_speed_mps
                        + (1.0 - self.speed_alpha) * previous_speed_mps
                    )

                motion_direction_deg = math.degrees(
                    math.atan2(velocity_x, velocity_y)
                )
                motion_direction = self._classify_motion(
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    relative_x=relative_x,
                    relative_y=relative_y,
                    speed_mps=smoothed_speed_mps,
                )

                kinematics.update(
                    {
                        "velocity_mps": [
                            round(velocity_x, 3),
                            round(velocity_y, 3),
                        ],
                        "motion_direction_deg": round(
                            motion_direction_deg, 2
                        ),
                        "motion_direction": motion_direction,
                        "speed_mps": round(smoothed_speed_mps, 3),
                        "speed_kmh": round(smoothed_speed_mps * 3.6, 3),
                    }
                )

        self.track_state[track_id] = {
            "world_x": world_x,
            "world_y": world_y,
            "time_s": float(capture_time_s),
            "speed_mps": smoothed_speed_mps,
        }

        return kinematics

    def process_frame(
        self,
        frame,
        frame_id,
        camera_id="cam_1",
        camera_location="unknown",
        save_images=False,
        capture_time_s=None,
    ):
        """Detect and build payload-ready objects."""
        if capture_time_s is None:
            capture_time_s = time.monotonic()

        detections, results = self.detect_objects(frame)
        processed_objects = []

        for i, detection in enumerate(detections):
            cropped, processed_object = self.crop_and_preprocess(frame, detection)
            if cropped is None:
                continue

            success, crop_buf = cv2.imencode(
                ".jpg",
                cropped,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not success:
                continue

            crop_bytes = crop_buf.tobytes()
            crop_b64 = base64.b64encode(crop_bytes).decode("utf-8")

            track_id = detection["track_id"]
            kinematics = self._estimate_kinematics(
                bbox=detection["bbox"],
                track_id=track_id,
                capture_time_s=float(capture_time_s),
            )

            processed_objects.append(
                {
                    # object_id preserves the existing per-frame numbering.
                    "object_id": i,
                    "track_id": track_id,
                    "global_track_id": (
                        f"{camera_id}:{track_id}"
                        if track_id is not None
                        else None
                    ),
                    "bbox": detection["bbox"],
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "crop_jpg_b64": crop_b64,
                    **kinematics,
                }
            )

            if save_images and processed_object is not None:
                save_directory = "/tmp/processed_objects"
                os.makedirs(save_directory, exist_ok=True)
                save_path = os.path.join(
                    save_directory, f"{detection['class_name']}_{frame_id}_{i}.jpg"
                )

                unnormalized = processed_object.clone()
                unnormalized[0] = unnormalized[0] * 0.229 + 0.485
                unnormalized[1] = unnormalized[1] * 0.224 + 0.456
                unnormalized[2] = unnormalized[2] * 0.225 + 0.406
                unnormalized = unnormalized.clamp(0, 1)

                to_pil = T.ToPILImage()
                pil_image = to_pil(unnormalized)
                pil_image.save(save_path)

        metadata = {
            "frame_id": frame_id,
            "camera_id": camera_id,
            "camera_location": camera_location,
            "timestamp": datetime.now().isoformat(),
            "capture_time_s": float(capture_time_s),
            "coordinate_convention": (
                "+x camera-right, +y forward, bearing 0 deg forward"
            ),
        }

        payload = {
            "metadata": metadata,
            "objects": processed_objects,
        }

        return payload, results