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

import uuid

MAX_DGRAM = 60000

# Packet type:
# 0 = single compressed JSON packet
# 1 = chunked compressed JSON packet
SINGLE_PACKET_TYPE = 0
CHUNK_PACKET_TYPE = 1

HEADER_FORMAT = "!HIHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


# class NetworkManager:
#     def __init__(
#         self,
#         dest_ip="192.168.10.11",
#         dest_port=12345,
#         priority=7,
#         interface_name=None,
#     ):
#         self.dest_ip = dest_ip
#         self.dest_port = dest_port
#         self.priority = priority

#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#         if interface_name:
#             self.sock.setsockopt(
#                 socket.SOL_SOCKET,
#                 socket.SO_BINDTODEVICE,
#                 interface_name.encode(),
#             )

#         self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, self.priority)

#     def send_data(self, data: bytes):
#         self.sock.sendto(data, (self.dest_ip, self.dest_port))

#     def send_json(self, payload: dict):
#         raw = json.dumps(payload).encode("utf-8")
#         compressed = zlib.compress(raw)

#         # helpful for debugging payload size
#         print(f"[DEBUG] raw={len(raw)} bytes, compressed={len(compressed)} bytes")

#         self.send_data(compressed)

#     def close(self):
#         self.sock.close()


class NetworkManager:
    def __init__(
        self,
        dest_ip="192.168.10.11",
        dest_port=12345,
        priority=7,
        interface_name=None,
        vlan_id=None,
    ):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.priority = priority
        self.interface_name = interface_name
        self.vlan_id = vlan_id

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
        """
        Send JSON payload.
        If payload fits in one UDP datagram, send one packet.
        If too large, automatically chunk it.
        """
        raw = json.dumps(payload).encode("utf-8")
        compressed = zlib.compress(raw)

        max_single_payload = MAX_DGRAM - HEADER_SIZE

        print(
            f"[DEBUG] type={payload.get('type', 'unknown')}, "
            f"raw={len(raw)} bytes, compressed={len(compressed)} bytes, "
            f"vlan={self.vlan_id}, iface={self.interface_name}"
        )

        if len(compressed) <= max_single_payload:
            self._send_single_packet(compressed)
        else:
            self._send_chunked_packets(compressed, payload)

    def _send_single_packet(self, compressed: bytes):
        """
        Send a small compressed JSON payload as one packet.
        """
        message_id = uuid.uuid4().int & 0xFFFFFFFF

        header = struct.pack(
            HEADER_FORMAT,
            SINGLE_PACKET_TYPE,
            message_id,
            1,
            0,
        )

        packet = header + compressed
        self.send_data(packet)

        print(
            f"[SENT SINGLE] message_id={message_id}, "
            f"size={len(packet)} bytes, vlan={self.vlan_id}"
        )

    def _send_chunked_packets(self, compressed: bytes, payload: dict):
        """
        Send a large compressed JSON payload by splitting it into chunks.
        """
        message_id = uuid.uuid4().int & 0xFFFFFFFF
        max_chunk_payload = MAX_DGRAM - HEADER_SIZE
        total_chunks = (len(compressed) + max_chunk_payload - 1) // max_chunk_payload

        print(
            f"[CHUNKING] message_id={message_id}, "
            f"type={payload.get('type', 'unknown')}, "
            f"compressed={len(compressed)} bytes, chunks={total_chunks}, "
            f"vlan={self.vlan_id}, iface={self.interface_name}"
        )

        for chunk_index in range(total_chunks):
            start = chunk_index * max_chunk_payload
            end = start + max_chunk_payload
            chunk = compressed[start:end]

            header = struct.pack(
                HEADER_FORMAT,
                CHUNK_PACKET_TYPE,
                message_id,
                total_chunks,
                chunk_index,
            )

            packet = header + chunk
            self.send_data(packet)

        print(
            f"[SENT CHUNKED] message_id={message_id}, "
            f"chunks={total_chunks}, vlan={self.vlan_id}"
        )

    def close(self):
        self.sock.close()





