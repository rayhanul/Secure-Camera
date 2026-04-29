
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

from cameraManager import CameraManager
from networkManager import NetworkManager
from FrameProcessor import FrameProcessor




# def main():
#     cam = CameraManager(camera_index=0, save_dir="captured_frames", interval=5)

#     try:
#         cam.capture_periodically()
#     finally:
#         cam.release()

# if __name__ == "__main__":
#     main()



# def main():
#     camera = CameraManager(camera_index=0, save_dir="captured_frames", interval=5)
#     network = None
#     frame_id = 0

#     try:
#         network = NetworkManager(
#             dest_ip="192.168.10.2",
#             dest_port=12345,
#             priority=6,
#             interval=0.1,
#         )

#         print("Sending camera frames over VLAN UDP. Press Ctrl+C to stop.")

#         while True:
#             frame_bytes = camera.get_encoded_frame(format=".jpg", quality=80)
#             network.send_frame(frame_bytes, frame_id)

#             print(f"Sent frame {frame_id}, size={len(frame_bytes)} bytes")
#             frame_id += 1

#             time.sleep(network.interval)

#     except PermissionError:
#         print("Error: run with sudo if SO_PRIORITY requires elevated permission.")
#     except KeyboardInterrupt:
#         print("\nStopped by user.")
#     finally:
#         camera.release()
#         if network is not None:
#             network.close()


# if __name__ == "__main__":
#     main()


def draw_boxes(frame, results):
    annotated = frame.copy()

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id == 0:
                label = "person"
            elif cls_id in [2, 3, 5, 7]:
                label = "vehicle"
            else:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return annotated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous camera capture, YOLO detection, and TSN/UDP sender"
    )
    parser.add_argument("--camera_index", type=int, default=0, help="Camera index")
    parser.add_argument("--save_dir", type=str, default="captured_frames", help="Directory to save captured video/frames")
    parser.add_argument("--model_path", type=str, default="/home/ubuntu/Documents/secure-camera/models/yolov8n.pt", help="Path to YOLO model")
    # parser.add_argument("--dest_ip", type=str, default="192.168.10.11", help="Receiver IP")
    parser.add_argument("--dest_port", type=int, default=12345, help="Receiver UDP port")
    # parser.add_argument("--priority", type=int, default=7, help="Socket priority for TSN/PCP mapping")
    # parser.add_argument("--interface_name", type=str, default=None, help="Optional interface name, e.g. enp1s0.10")
    parser.add_argument("--camera_id", type=str, default="cam_1", help="Camera ID in metadata")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--jpeg_quality", type=int, default=70, help="JPEG quality for transmitted crops")
    parser.add_argument("--save_images", action="store_true", default=False, help="Save processed objects under /tmp/processed_objects")
    parser.add_argument("--save_annotated_every", type=int, default=30, help="Save annotated frame every N frames; 0 disables")
    parser.add_argument("--send_empty", action="store_true", default=True, help="Send metadata even when no object is detected")
    
    
    # route for raw captured frames
    parser.add_argument("--frame_vlan_interface", type=str, default="enp1s0.10")
    parser.add_argument("--frame_dest_ip", type=str, default="192.168.10.11")
    parser.add_argument("--frame_vlan_id", type=int, default=10)
    parser.add_argument("--frame_priority", type=int, default=7)

    # route for YOLO detected objects
    parser.add_argument("--object_vlan_interface", type=str, default="enp1s0.20")
    parser.add_argument("--object_dest_ip", type=str, default="192.168.20.11")
    parser.add_argument("--object_vlan_id", type=int, default=20)
    parser.add_argument("--object_priority", type=int, default=7)
    
    return parser.parse_args()


def main():
    args = parse_args()

    print("===== ARGUMENTS =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=====================")

    camera = CameraManager(camera_index=args.camera_index, save_dir=args.save_dir)
    frame_network = None
    object_network = None 
    frame_id = 0
    video_writer = None

    processor = FrameProcessor(
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        jpeg_quality=args.jpeg_quality,
    )

    try:
        # network = NetworkManager(
        #     dest_ip=args.dest_ip,
        #     dest_port=args.dest_port,
        #     priority=args.priority,
        #     interface_name=args.interface_name,
        # )

        # VLAN 10 route for captured frame
        frame_network = NetworkManager(
            dest_ip=args.frame_dest_ip,
            dest_port=args.dest_port,
            priority=args.frame_priority,
            interface_name=args.frame_vlan_interface,
            vlan_id=args.frame_vlan_id,
        )

        # VLAN 20 route for detected objects
        object_network = NetworkManager(
            dest_ip=args.object_dest_ip,
            dest_port=args.dest_port,
            priority=args.object_priority,
            interface_name=args.object_vlan_interface,
            vlan_id=args.object_vlan_id,
        )

        print(
            f"Frame route: VLAN={args.frame_vlan_id}, "
            f"iface={args.frame_vlan_interface}, dest={args.frame_dest_ip}"
        )

        print(
            f"Object route: VLAN={args.object_vlan_id}, "
            f"iface={args.object_vlan_interface}, dest={args.object_dest_ip}"
        )

        width, height, fps = camera.get_frame_size()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(args.save_dir, f"session_{timestamp}.avi")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")

        print("Recording video, running YOLO, and sending detections to server. Press Ctrl+C to stop.")

        # while True:
        #     frame = camera.get_frame()
        #     video_writer.write(frame)

        #     payload, results = processor.process_frame(
        #         frame=frame,
        #         frame_id=frame_id,
        #         camera_id=args.camera_id,
        #         save_images=args.save_images,
        #     )

        #     annotated = draw_boxes(frame, results)

        #     if args.save_annotated_every > 0 and frame_id % args.save_annotated_every == 0:
        #         jpg_name = os.path.join(args.save_dir, f"frame_{frame_id}.jpg")
        #         cv2.imwrite(jpg_name, annotated)

        #     print(
        #         f"[SENDING] frame_id={frame_id}, detected_objects={len(payload['objects'])}"
        #     )

        #     if payload["objects"]:
        #         # safest for UDP: one object per packet
        #         for obj in payload["objects"]:
        #             small_payload = {
        #                 "metadata": payload["metadata"],
        #                 "objects": [obj],
        #             }
        #             network.send_json(small_payload)
        #     elif args.send_empty:
        #         network.send_json(payload)

        #     frame_id += 1

        while True:
            frame = camera.get_frame()
            video_writer.write(frame)

            payload, results = processor.process_frame(
                frame=frame,
                frame_id=frame_id,
                camera_id=args.camera_id,
                save_images=args.save_images,
            )

            annotated = draw_boxes(frame, results)

            if args.save_annotated_every > 0 and frame_id % args.save_annotated_every == 0:
                jpg_name = os.path.join(args.save_dir, f"frame_{frame_id}.jpg")
                cv2.imwrite(jpg_name, annotated)

            # Send raw captured frame through VLAN 10 / frame route
            success, frame_buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )

            if success:
                frame_b64 = base64.b64encode(frame_buf.tobytes()).decode("utf-8")

                frame_payload = {
                    "type": "raw_frame",
                    "metadata": {
                        "frame_id": frame_id,
                        "camera_id": args.camera_id,
                        "timestamp": datetime.now().isoformat(),
                        "vlan_id": args.frame_vlan_id,
                        "vlan_interface": args.frame_vlan_interface,
                        "priority": args.frame_priority,
                    },
                    "frame_jpg_b64": frame_b64,
                }

                print(
                    f"[Frame] type=raw_frame "
                    f"frame_id={frame_id} "
                    f"vlan={args.frame_vlan_id} "
                    f"prio={args.frame_priority} "
                    f"iface={args.frame_vlan_interface} "
                    f"dest={args.frame_dest_ip}"
                )

                frame_network.send_json(frame_payload)

            # Send YOLO detected objects through VLAN 20 / object route
            payload["metadata"]["payload_type"] = "detected_objects"
            payload["metadata"]["vlan_id"] = args.object_vlan_id
            payload["metadata"]["vlan_interface"] = args.object_vlan_interface
            payload["metadata"]["priority"] = args.object_priority 
            payload["metadata"]["num_objects"] = len(payload["objects"])

            print(
                f"[OBJECTS] type=detected_objects "
                f"frame_id={frame_id} "
                f"detected objects={len(payload['objects'])} "
                f"vlan={args.object_vlan_id} "
                f"prio={args.object_priority} "
                f"iface={args.object_vlan_interface} "
                f"dest={args.object_dest_ip}"
            )

            if payload["objects"]:
                for obj in payload["objects"]:
                    object_payload = {
                        "type": "detected_objects",
                        "metadata": payload["metadata"],
                        "objects": [obj],
                    }
                    object_network.send_json(object_payload)
            elif args.send_empty:
                object_network.send_json(
                    {
                        "type": "detected_objects",
                        "metadata": payload["metadata"],
                        "objects": [],
                    }
                )

            frame_id += 1



    except PermissionError:
        print("Error: run with sudo if SO_PRIORITY requires elevated permission.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.release()

        if video_writer is not None:
            video_writer.release()
            print("Video saved.")


        # Close both VLAN sockets
        if "frame_network" in locals() and frame_network is not None:
            frame_network.close()

        if "object_network" in locals() and object_network is not None:
            object_network.close()


if __name__ == "__main__":
    main()
