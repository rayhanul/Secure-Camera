import socket
import struct
import os
import time
from datetime import datetime

import cv2
import numpy as np


LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 12345
SAVE_DIR = "received_frames"
MAX_PACKET_SIZE = 65535
FRAME_TIMEOUT = 5.0

# Header format:
# frame_id      -> unsigned int   (4 bytes)
# total_chunks  -> unsigned short (2 bytes)
# chunk_index   -> unsigned short (2 bytes)
# filename_len  -> unsigned short (2 bytes)
HEADER_FORMAT = "!IHHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class FrameReceiver:
    def __init__(self, listen_ip=LISTEN_IP, listen_port=LISTEN_PORT, save_dir=SAVE_DIR):
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.listen_port))

        # frame_buffer[frame_id] = {
        #   "total_chunks": int,
        #   "chunks": {chunk_index: bytes},
        #   "timestamp": float,
        #   "filename": str or None
        # }
        self.frame_buffer = {}

        print(f"Listening on {self.listen_ip}:{self.listen_port}")

    def cleanup_old_frames(self):
        now = time.time()
        expired = []

        for frame_id, info in self.frame_buffer.items():
            if now - info["timestamp"] > FRAME_TIMEOUT:
                expired.append(frame_id)
        for frame_id, info in self.frame_buffer.items():
            if now - info["timestamp"] > FRAME_TIMEOUT:
                expired.append(frame_id)

        for frame_id in expired:
            info = self.frame_buffer[frame_id]
            print(
                f"[TIMEOUT] Dropping incomplete frame {frame_id} "
                f"({len(info['chunks'])}/{info['total_chunks']} chunks received)"
            )
            del self.frame_buffer[frame_id]

    def receive_packets(self):
        while True:
            packet, addr = self.sock.recvfrom(MAX_PACKET_SIZE)

            self.cleanup_old_frames()

            if len(packet) < HEADER_SIZE:
                print("Received packet too small, ignored.")
                continue

            header = packet[:HEADER_SIZE]
            payload = packet[HEADER_SIZE:]

            frame_id, total_chunks, chunk_index, filename_len = struct.unpack(
                HEADER_FORMAT, header
            )

            if frame_id not in self.frame_buffer:
                self.frame_buffer[frame_id] = {
                    "total_chunks": total_chunks,
                    "chunks": {},
                    "timestamp": time.time(),
                    "filename": None,
                }

            info = self.frame_buffer[frame_id]
            info["timestamp"] = time.time()

            # First chunk contains filename bytes first, then image bytes
            if chunk_index == 0 and filename_len > 0:
                if len(payload) < filename_len:
                    print(f"Frame {frame_id}: invalid filename length in first chunk")
                    del self.frame_buffer[frame_id]
                    continue
                filename_bytes = payload[:filename_len]
                image_payload = payload[filename_len:]

                try:
                    info["filename"] = filename_bytes.decode()
                except UnicodeDecodeError:
                    print(f"Frame {frame_id}: failed to decode filename")
                    del self.frame_buffer[frame_id]
                    continue

                info["chunks"][chunk_index] = image_payload
            else:
                info["chunks"][chunk_index] = payload

            print(
                f"Received chunk {chunk_index + 1}/{total_chunks} "
                f"for frame {frame_id} from {addr}"
            )

            if self.is_frame_complete(frame_id):
                self.process_complete_frame(frame_id)

    def is_frame_complete(self, frame_id):
        info = self.frame_buffer[frame_id]
        return len(info["chunks"]) == info["total_chunks"]

    def process_complete_frame(self, frame_id):
        info = self.frame_buffer[frame_id]
        total_chunks = info["total_chunks"]
        chunks = info["chunks"]
        sender_filename = info["filename"]

        try:
            frame_bytes = b"".join(chunks[i] for i in range(total_chunks))
        except KeyError:
            print(f"Frame {frame_id} missing chunks during reassembly")
            del self.frame_buffer[frame_id]
            return

        np_data = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"Failed to decode frame {frame_id}")
            del self.frame_buffer[frame_id]
            return

        if sender_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sender_filename = f"frame_{frame_id}_{timestamp}.jpg"

        # Keep only the file name part, not sender directories
        safe_filename = os.path.basename(sender_filename)
        save_path = os.path.join(self.save_dir, safe_filename)

        ok = cv2.imwrite(save_path, frame)

        if ok:
            print(f"[COMPLETE] Saved frame {frame_id} as {save_path}")
        else:
            print(f"Failed to save frame {frame_id}")

        del self.frame_buffer[frame_id]

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    receiver = FrameReceiver()

    try:
        receiver.receive_packets()
    except KeyboardInterrupt:
        print("\nReceiver stopped by user.")
    finally:
        receiver.close()
                                       