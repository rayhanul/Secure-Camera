

import os
import cv2
import time
import json
import zlib
import socket
import base64
import argparse
from datetime import datetime

from PIL import Image
from torchvision import transforms as T
from ultralytics import YOLO



class FrameProcessor:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, jpeg_quality=70):
        self.yolo_model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.jpeg_quality = jpeg_quality

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

    def detect_objects(self, frame):
        """Detect persons and vehicles using YOLO."""
        results = self.yolo_model(frame, verbose=False)
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
                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_name": class_name,
                        "confidence": confidence,
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

    def process_frame(self, frame, frame_id, camera_id="cam_1", save_images=False):
        """Detect and build payload-ready objects."""
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

            processed_objects.append(
                {
                    "object_id": i,
                    "bbox": detection["bbox"],
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "crop_jpg_b64": crop_b64,
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
            "timestamp": datetime.now().isoformat(),
        }

        payload = {
            "metadata": metadata,
            "objects": processed_objects,
        }

        return payload, results

