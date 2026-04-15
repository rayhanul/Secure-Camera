import argparse
import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import redis
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO


class FrameProcessor:
    def __init__(
        self, redis_host="localhost", redis_port=6379, model_path="yolov8n.pt"
    ):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)

        # Initialize YOLO model
        self.yolo_model = YOLO(model_path)

        # Person ReID preprocessing
        self.person_transform = T.Compose(
            [
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Vehicle ReID preprocessing
        self.vehicle_transform = T.Compose(
            [
                T.Resize((224, 224)),  # Different size for vehicles
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def detect_objects(self, frame):
        """Detect persons and vehicles using YOLO"""
        results = self.yolo_model(frame)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)

                    # Class 0 is 'person', classes 2,3,5,7 are vehicles (car, motorcycle, bus, truck)
                    if confidence > 0.5:
                        if class_id == 0:  # person
                            class_name = "person"
                        elif class_id in [2, 3, 5, 7]:  # vehicles
                            class_name = "vehicle"
                        else:
                            continue  # Skip other classes

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append(
                            {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "class_name": class_name,
                                "confidence": confidence,
                            }
                        )

        return detections

    def crop_and_preprocess(self, frame, detection):
        """Crop object from frame and preprocess for ReID"""
        bbox = detection["bbox"]
        class_name = detection["class_name"]

        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        # Apply appropriate transform based on object type
        processed = None
        if class_name == "person":
            processed = self.person_transform(pil_image)
        # else:  # vehicle
        #     processed = self.vehicle_transform(pil_image)

        # processed object should have

        return processed

    def process_frame(self, frame, save_images=False):
        """Process frame: detect persons and vehicles and prepare for ReID"""
        detections = self.detect_objects(frame)
        processed_objects = []

        for i, detection in enumerate(detections):
            processed_object = self.crop_and_preprocess(frame, detection)
            if processed_object is not None:
                processed_objects.append(
                    {
                        "object_id": i,
                        "bbox": detection["bbox"],
                        "class_name": detection["class_name"],
                        "confidence": detection["confidence"],
                        "processed_image": processed_object.numpy().tolist(),
                    }
                )

                if save_images:
                    # saving this extracted images locally in /tmp/processed_objects
                    save_directory = "/tmp/processed_objects"
                    os.makedirs(save_directory, exist_ok=True)
                    save_path = os.path.join(
                        save_directory, f"{detection['class_name']}_{i}.jpg"
                    )

                    unnormalized = processed_object.clone()
                    unnormalized[0] = unnormalized[0] * 0.229 + 0.485
                    unnormalized[1] = unnormalized[1] * 0.224 + 0.456
                    unnormalized[2] = unnormalized[2] * 0.225 + 0.406
                    unnormalized = unnormalized.clamp(0, 1)

                    # Convert to PIL and save
                    to_pil = T.ToPILImage()
                    pil_image = to_pil(unnormalized)
                    pil_image.save(save_path)

        return processed_objects

    def send_to_queue(self, processed_objects, metadata):
        """Send processed objects to Redis queue"""
        data = {"objects": processed_objects, "metadata": metadata}
        # check status if failing
        try:
            self.redis_client.lpush("object_queue", json.dumps(data))
        except Exception as e:
            print(f"Failed to send to queue: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Frame Processing for ReID")
    parser.add_argument(
        "--camera_id", type=int, default=0, help="Camera ID for video stream"
    )
    parser.add_argument(
        "--redis_host", type=str, default="localhost", help="Redis host"
    )
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis port")
    # parser.add_argument('--redis_queue', type=str, default='reid_queue', help="Redis queue name")
    parser.add_argument(
        "--use_camera", action="store_true", help="Use camera for live video stream"
    )
    parser.add_argument("--save_images", default=False, action="store_true")
    parser.add_argument("--video_path", type=str, default="", help="Path to video file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/models/yolov8n.pt",
        help="Path to YOLO model",
    )
    return parser.parse_args()


def main():
    # Parse the input
    args = parse_args()

    print("Frame Processor Starting ....")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # processor = FrameProcessor(redis_host=args.redis_host, redis_port=args.redis_port)
    processor = FrameProcessor(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        model_path=args.model_path,
    )

    if args.use_camera:
        # Live camera processing
        cap = cv2.VideoCapture(args.camera_id)
        frame_id = 0

        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera_id}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            processed_frame = processor.process_frame(
                frame, save_images=args.save_images
            )

            if processed_frame:  # Only send if objects detected
                metadata = {
                    "frame_id": frame_id,
                    "camera_id": args.camera_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M%S"),
                    "object_count": len(processed_frame),
                }

                print(metadata)
                print(processed_frame)

                processor.send_to_queue(processed_frame, metadata)
                persons = [
                    obj for obj in processed_frame if obj["class_name"] == "person"
                ]
                vehicles = [
                    obj for obj in processed_frame if obj["class_name"] == "vehicle"
                ]
            print(f"Processed frame {frame_id} from camera {args.camera_id}")

            frame_id += 1

        cap.release()

    elif args.video_path:
        # Video file processing
        cap = cv2.VideoCapture(args.video_path)
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = processor.process_frame(
                frame, save_images=args.save_images
            )

            if processed_frame:  # Only send if objects detected
                metadata = {
                    "frame_id": frame_id,
                    "video_path": args.video_path,
                    "camera_id": 1,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M%S"),
                    "object_count": len(processed_frame),
                }

                processor.send_to_queue(processed_frame, metadata)
                persons = [
                    obj for obj in processed_frame if obj["class_name"] == "person"
                ]
                vehicles = [
                    obj for obj in processed_frame if obj["class_name"] == "vehicle"
                ]
                print(
                    f"Frame {frame_id}: Detected {len(persons)} persons, {len(vehicles)} vehicles"
                )

            frame_id += 1
        cap.release()

    print("Frame processing completed.")


if __name__ == "__main__":
    main()
