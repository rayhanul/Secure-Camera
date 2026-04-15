import socket
import struct


import os
import cv2
import time
import json
import zlib
import socket
import base64
import argparse
from datetime import datetime

# DEST_IP = "192.168.10.2"
# DEST_PORT = 12345
# VLAN_PRIORITY = 6
# INTERVAL = 0.1

# Keep payload small enough for UDP
MAX_DGRAM = 60000

# Header format:
# frame_id      -> unsigned int   (4 bytes)
# total_chunks  -> unsigned short (2 bytes)
# chunk_index   -> unsigned short (2 bytes)
# HEADER_FORMAT = "!IHH"
HEADER_FORMAT = "!IHHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class NetworkManager:
    def __init__(
        self,
        dest_ip="192.168.10.11",
        dest_port=12345,
        priority=7,
        interface_name=None,
    ):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.priority = priority

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if interface_name:
            self.sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_BINDTODEVICE,
                interface_name.encode(),
            )

        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, self.priority)

    def send_data(self, data: bytes):
        self.sock.sendto(data, (self.dest_ip, self.dest_port))

    def send_json(self, payload: dict):
        raw = json.dumps(payload).encode("utf-8")
        compressed = zlib.compress(raw)

        # helpful for debugging payload size
        print(f"[DEBUG] raw={len(raw)} bytes, compressed={len(compressed)} bytes")

        self.send_data(compressed)

    def close(self):
        self.sock.close()




