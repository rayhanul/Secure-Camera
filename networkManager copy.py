import socket
import time
import struct


DEST_IP = "192.168.10.2"
DEST_PORT = 12345
VLAN_PRIORITY = 6
INTERVAL = 0.1

# Keep payload small enough for UDP
MAX_DGRAM = 60000

# Header format:
# frame_id      -> unsigned int   (4 bytes)
# total_chunks  -> unsigned short (2 bytes)
# chunk_index   -> unsigned short (2 bytes)
HEADER_FORMAT = "!IHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class NetworkManager:
    def __init__(
        self,
        dest_ip=DEST_IP,
        dest_port=DEST_PORT,
        priority=VLAN_PRIORITY,
        interval=INTERVAL,
    ):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.priority = priority
        self.interval = interval

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, self.priority)

    def send_data(self, data: bytes):
        self.sock.sendto(data, (self.dest_ip, self.dest_port))

    def send_frame(self, frame_bytes: bytes, frame_id: int):
        max_payload = MAX_DGRAM - HEADER_SIZE
        total_chunks = (len(frame_bytes) + max_payload - 1) // max_payload

        for chunk_index in range(total_chunks):
            start = chunk_index * max_payload
            end = start + max_payload
            chunk = frame_bytes[start:end]

            header = struct.pack(
                HEADER_FORMAT,
                frame_id,
                total_chunks,
                chunk_index,
            )

            packet = header + chunk
            self.send_data(packet)

    def close(self):
        self.sock.close()





