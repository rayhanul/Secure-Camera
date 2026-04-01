import time 
import cv2
from datetime import datetime

from cameraManager import CameraManager
from networkManager import NetworkManager




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



def main():
    camera = CameraManager(camera_index=0, save_dir="captured_frames", interval=5)
    network = None
    frame_id = 0

    try:
        network = NetworkManager(
            dest_ip="192.168.10.2",
            dest_port=12345,
            priority=7,
            interval=0.1,
        )

        print("Sending camera frames over VLAN UDP. Press Ctrl+C to stop.")

        while True:
            # Capture frame
            frame = camera.get_frame()

            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera.save_dir}/frame_{timestamp}_{frame_id}.jpg"
            cv2.imwrite(filename, frame)

            # Encode frame
            success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                print("Encoding failed")
                continue

            frame_bytes = buffer.tobytes()

            # 🔹 Print sending info
            print(f"[SENDING] frame_id={frame_id}, file={filename}, size={len(frame_bytes)} bytes")

            # Send frame
            # total_chunks = network.send_frame(frame_bytes, frame_id)
            total_chunks =network.send_frame(frame_bytes, frame_id, filename)

            # 🔹 Print completion info
            print(f"[COMPLETE] frame_id={frame_id}, chunks={total_chunks}")

            frame_id += 1
            time.sleep(network.interval)

    except PermissionError:
        print("Error: run with sudo if SO_PRIORITY requires elevated permission.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.release()
        if network is not None:
            network.close()


if __name__ == "__main__":
    main()