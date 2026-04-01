import socket
import struct
import time
import os
from datetime import datetime

from database_manager import DatabaseManager


LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 12345
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
    def __init__(self, listen_ip=LISTEN_IP, listen_port=LISTEN_PORT, db_path="received_images.db"):
        self.listen_ip = listen_ip
        self.listen_port = listen_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.listen_port))

        self.db = DatabaseManager(db_path=db_path)

        # frame_buffer[frame_id] = {
        #   "total_chunks": int,
        #   "chunks": {chunk_index: bytes},
        #   "timestamp": float,
        #   "filename": str or None
        # }
        self.frame_buffer = {}

        print(f"Listening on {self.listen_ip}:{self.listen_port}")
        print(f"Saving received images into SQLite database: {db_path}")

    def cleanup_old_frames(self):
        now = time.time()
        expired = []

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

            # First chunk contains filename + first image bytes
            if chunk_index == 0 and filename_len > 0:
                if len(payload) < filename_len:
                    print(f"Frame {frame_id}: invalid filename length")
                    del self.frame_buffer[frame_id]
                    continue

                filename_bytes = payload[:filename_len]
                image_payload = payload[filename_len:]

                try:
                    info["filename"] = filename_bytes.decode()
                except UnicodeDecodeError:
                    print(f"Frame {frame_id}: filename decode failed")
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
            image_bytes = b"".join(chunks[i] for i in range(total_chunks))
        except KeyError:
            print(f"Frame {frame_id} missing chunks during reassembly")
            del self.frame_buffer[frame_id]
            return

        if sender_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sender_filename = f"frame_{frame_id}_{timestamp}.jpg"

        safe_filename = os.path.basename(sender_filename)

        self.db.save_image(frame_id, safe_filename, image_bytes)
        print(f"[COMPLETE] Stored frame {frame_id} in database with filename {safe_filename}")

        del self.frame_buffer[frame_id]

    def close(self):
        self.sock.close()
        self.db.close()


if __name__ == "__main__":
    receiver = FrameReceiver()

    try:
        receiver.receive_packets()
    except KeyboardInterrupt:
        print("\nReceiver stopped by user.")
    finally:
        receiver.close()