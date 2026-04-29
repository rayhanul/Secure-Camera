import os

import argparse
import base64
import json
import socket
import time
import zlib
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np


import torch
from torchvision import transforms as T

from utils.reid_result import ReIDResult  # Just a data class
from utils.results_saver import ReIDResultsSaver
from utils.storage import SecureReIDStorage
from utils.util import objects_to_tensor
from utils.weaviate import ReIDVectorStore

import struct

MAX_DGRAM = 60000

SINGLE_PACKET_TYPE = 0
CHUNK_PACKET_TYPE = 1

HEADER_FORMAT = "!HIHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)



import sys
sys.path.insert(0, '/home/jdg24001/Documents/github/Secure-Camera/C2')


# class TSNReceiver:
#     def __init__(self, listen_ip="0.0.0.0", port=12345, buffer_size=65535):
#         self.listen_ip = listen_ip
#         self.port = port
#         self.buffer_size = buffer_size

#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         self.sock.bind((self.listen_ip, self.port))

#         print(f"Listening on {self.listen_ip}:{self.port}")

#     def get_data(self) -> Dict:
#         packet, addr = self.sock.recvfrom(self.buffer_size)
#         raw = zlib.decompress(packet)
#         data = json.loads(raw.decode("utf-8"))
#         return data

#     def close(self):
#         self.sock.close()

class TSNReceiver:
    def __init__(self, listen_ip="0.0.0.0", port=12345, buffer_size=65535):
        self.listen_ip = listen_ip
        self.port = port
        self.buffer_size = buffer_size
        self.chunk_buffers = {}

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.port))

        print(f"Listening on {self.listen_ip}:{self.port}")

    def get_data(self) -> Optional[Dict]:
        packet, addr = self.sock.recvfrom(self.buffer_size)

        if len(packet) < HEADER_SIZE:
            print("Packet too small, ignoring")
            return None

        packet_type, message_id, total_chunks, chunk_index = struct.unpack(
            HEADER_FORMAT,
            packet[:HEADER_SIZE],
        )

        payload_bytes = packet[HEADER_SIZE:]

        if packet_type == SINGLE_PACKET_TYPE:
            raw = zlib.decompress(payload_bytes)
            data = json.loads(raw.decode("utf-8"))
            data["_source_addr"] = addr[0]
            return data

        if packet_type == CHUNK_PACKET_TYPE:
            return self._handle_chunk(
                message_id=message_id,
                total_chunks=total_chunks,
                chunk_index=chunk_index,
                payload_bytes=payload_bytes,
                addr=addr,
            )

        print(f"Unknown packet type: {packet_type}")
        return None

    def _handle_chunk(self, message_id, total_chunks, chunk_index, payload_bytes, addr):
        if message_id not in self.chunk_buffers:
            self.chunk_buffers[message_id] = {
                "total_chunks": total_chunks,
                "chunks": {},
                "source_addr": addr[0],
                "start_time": time.time(),
            }

        self.chunk_buffers[message_id]["chunks"][chunk_index] = payload_bytes

        received = len(self.chunk_buffers[message_id]["chunks"])
        print(f"[CHUNK RX] message_id={message_id}, chunk={chunk_index + 1}/{total_chunks}")

        if received < total_chunks:
            return None

        chunks = self.chunk_buffers[message_id]["chunks"]
        compressed = b"".join(chunks[i] for i in range(total_chunks))

        del self.chunk_buffers[message_id]

        raw = zlib.decompress(compressed)
        data = json.loads(raw.decode("utf-8"))
        data["_source_addr"] = addr[0]

        print(f"[CHUNK COMPLETE] message_id={message_id}, type={data.get('type')}")
        return data

    def close(self):
        self.sock.close()



class TransReIDProcessor:
    def __init__(
        self,
        model_path="/home/jdg24001/Documents/github/Secure-Camera/weights-models/transformer_best.pth",
        config_path="/home/jdg24001/Documents/github/Secure-Camera/weights-models/vit_transreid_stride.yml",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device selected: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.load_model(model_path, config_path)

    def load_model(self, model_path=None, config_file=None):
        if model_path is None:
            model_path = "/home/jdg24001/Documents/github/Secure-Camera/weights-models/weights/transformer_best.pth"

        try:
            print("Loading TransReID model...")
            from TransReID.config import cfg
            from TransReID.model import make_model

            cfg.MODEL.NAME = "transformer"
            cfg.MODEL.TRANSFORMER_TYPE = "vit_base_patch16_224_TransReID"
            cfg.MODEL.SIE_CAMERA = False
            cfg.MODEL.SIE_VIEW = False
            cfg.MODEL.JPM = True
            cfg.MODEL.PRETRAIN_CHOICE = "self"
            cfg.MODEL.PRETRAIN_PATH = "/home/jdg24001/Documents/github/Secure-Camera/weights-models/weights/transformer_best.pth"
            cfg.TEST.WEIGHT = model_path
            cfg.INPUT.SIZE_TEST = [256, 128]
            cfg.INPUT.SIZE_TRAIN = [256, 128]
            cfg.MODEL.DEVICE = self.device
            cfg.freeze()

            self.model = make_model(cfg, num_class=751, camera_num=0, view_num=0)

            if os.path.exists(model_path):
                try:
                    self.model.load_param(model_path)
                    print(f"Successfully loaded weights from {model_path}")
                except Exception:
                    print("Standard loading failed, trying partial loading...")
                    self._load_weights_partially(model_path)
                    print("Partially loaded weights")
            else:
                print(f"Warning: Weights not found at {model_path}")

            self.model.to(self.device)
            self.model.eval()
            print(f"TransReID model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading TransReID model: {e}")
            self.model = None

    def _load_weights_partially(self, model_path):
        param_dict = torch.load(model_path, map_location="cpu")
        model_dict = self.model.state_dict()

        compatible_dict = {}
        for key in param_dict:
            clean_key = key.replace("module.", "")
            if clean_key in model_dict and param_dict[key].shape == model_dict[clean_key].shape:
                compatible_dict[clean_key] = param_dict[key]

        model_dict.update(compatible_dict)
        self.model.load_state_dict(model_dict, strict=False)

    def extract_features(self, person_images):
        if self.model is None:
            print("TransReID model not loaded, using fallback normalization")
            # return torch.nn.functional.normalize(person_images, dim=1, p=2)

            pooled = person_images.mean(dim=(2, 3))

            # optional: repeat to make a slightly larger vector, e.g. 384 dims
            pooled = pooled.repeat(1, 128)   # [N, 384]

            return torch.nn.functional.normalize(pooled, dim=1, p=2)

        try:
            with torch.no_grad():
                person_images = person_images.to(self.device)

                if person_images.dim() != 4:
                    raise ValueError(f"Expected 4D tensor, got {person_images.dim()}D")
                if person_images.size(1) != 3:
                    raise ValueError(f"Expected 3 channels, got {person_images.size(1)}")
                if person_images.size(2) != 256 or person_images.size(3) != 128:
                    raise ValueError(
                        f"Expected 256x128 images, got {person_images.size(2)}x{person_images.size(3)}"
                    )

                features = self.model(person_images)
                return features.cpu()
        except Exception as e:
            print(f"Feature extraction fallback due to error: {e}")
            # return torch.nn.functional.normalize(person_images, dim=1, p=2)

            pooled = torch.nn.functional.adaptive_avg_pool2d(person_images, (16, 16))
            pooled = pooled.flatten(start_dim=1)
            return torch.nn.functional.normalize(pooled, dim=1, p=2)


class WeaviateReIDManager:
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        collection_name: str = "reid_collection",
        similarity_threshold: float = 0.7,
        max_gallery_size: int = 10000,
        store_crops: bool = False,
    ):
        self.vector_store = ReIDVectorStore(weaviate_url, collection_name)
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        self.person_id_counter = 0
        self.store_crops = store_crops

        self.mongo_storage = SecureReIDStorage()
        print("🔒 MongoDB storage initialized for gallery data")

    def process_and_identify(
        self, objects_data: Dict, reid_features: torch.Tensor
    ) -> List[Dict]:
        results = []

        for i, obj in enumerate(objects_data["objects"]):
            person_feature = reid_features[i : i + 1]

            similar_persons = self.find_similar_persons(
                person_feature,
                obj.get("class_name", "person"),
                objects_data["metadata"]["camera_id"],
            )

            person_identity = self.determine_identity(
                similar_persons, person_feature, obj
            )

            storage_result = self.store_person_data(
                objects_data, obj, person_feature, person_identity, i, self.store_crops
            )

            result = {
                "detection_id": obj.get("object_id", i),
                "person_id": person_identity["person_id"],
                "confidence": person_identity["confidence"],
                "is_new_person": person_identity["is_new"],
                "similar_detections": len(similar_persons),
                "bbox": obj.get("bbox", [0, 0, 0, 0]),
                "camera_id": objects_data["metadata"]["camera_id"],
                "frame_id": objects_data["metadata"]["frame_id"],
                "timestamp": objects_data["metadata"]["timestamp"],
                "cross_camera_matches": person_identity.get("cross_camera_matches", []),
                "weaviate_id": storage_result[0] if storage_result and len(storage_result) > 0 else None,
            }
            results.append(result)

        return results

    def find_similar_persons(
        self, query_feature: torch.Tensor, class_filter: str, current_camera: str
    ) -> List[Dict]:
        try:
            similar_results = self.vector_store.search_similar(
                query_feature,
                top_k=20,
                class_filter=class_filter,
                confidence_threshold=0.3,
                distance_threshold=2.0,
            )

            filtered_results = []
            for result in similar_results:
                similarity_score = self.calculate_similarity(query_feature, result)

                if similarity_score >= self.similarity_threshold:
                    result["similarity_score"] = similarity_score
                    result["is_cross_camera"] = result.get("camera_id") != current_camera
                    filtered_results.append(result)

            return filtered_results

        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def determine_identity(
        self, similar_persons: List[Dict], person_feature: torch.Tensor, obj: Dict
    ) -> Dict:
        if not similar_persons:
            self.person_id_counter += 1
            return {
                "person_id": f"person_{self.person_id_counter:06d}",
                "confidence": 1.0,
                "is_new": True,
                "matched_detection": None,
            }

        best_match = max(similar_persons, key=lambda x: x.get("similarity_score", 0))

        if best_match["similarity_score"] >= self.similarity_threshold:
            cross_camera_matches = [
                p for p in similar_persons
                if p.get("is_cross_camera", False)
                and p.get("similarity_score", 0) >= self.similarity_threshold
            ]

            return {
                "person_id": best_match.get(
                    "person_id", f"unknown_{best_match.get('object_id')}"
                ),
                "confidence": best_match["similarity_score"],
                "is_new": False,
                "matched_detection": best_match,
                "cross_camera_matches": cross_camera_matches[:5],
            }

        self.person_id_counter += 1
        return {
            "person_id": f"person_{self.person_id_counter:06d}",
            "confidence": 0.6,
            "is_new": True,
            "matched_detection": None,
        }

    def store_person_data(
        self,
        objects_data: Dict,
        obj: Dict,
        person_feature: torch.Tensor,
        person_identity: Dict,
        index: int,
        store_crops: bool = False,
    ):
        try:
            person_crop_bytes = b""

            if "processed_image" in obj:
                img_list = obj["processed_image"]
                tensor = torch.tensor(img_list)

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                unnormalized = tensor * std + mean
                unnormalized = unnormalized.clamp(0, 1)

                to_pil = T.ToPILImage()
                pil_image = to_pil(unnormalized)

                import io
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format="PNG")
                person_crop_bytes = img_buffer.getvalue()

            mongo_result = ReIDResult(
                object_id=str(obj.get("object_id", f"obj_{index}")),
                class_name=obj.get("class_name", "unknown"),
                confidence=float(obj.get("confidence", 0.0)),
                bbox=obj.get("bbox", [0, 0, 0, 0]),
                camera_id=str(objects_data["metadata"].get("camera_id", "unknown")),
                frame_id=int(objects_data["metadata"].get("frame_id", 0)),
                timestamp=str(objects_data["metadata"].get("timestamp", 0)),
                embedding_method="TransReID",
                reid_confidence=person_identity["confidence"],
                person_id=person_identity["person_id"],
                is_new_person=person_identity["is_new"],
                image=person_crop_bytes if person_crop_bytes else b"",
            )

            embedding_vector = person_feature.cpu().numpy()
            self.mongo_storage.store_reid_result(embedding_vector, mongo_result)

            objects_data_with_person_id = {
                "metadata": objects_data["metadata"],
                "objects": [
                    {
                        "person_id": person_identity["person_id"],
                        "reid_confidence": person_identity["confidence"],
                        "is_new_person": person_identity["is_new"],
                    }
                ],
            }

            result = self.vector_store.store_embeddings(
                objects_data_with_person_id, person_feature
            )
            return result if result else None

        except Exception as e:
            print(f"Error storing person data: {e}")
            return None

    def calculate_similarity(self, query_feature: torch.Tensor, result: Dict) -> float:
        try:
            distance = result.get("_additional", {}).get("distance", 1.0)
            return max(0, 1.0 - (distance / 2.0))
        except Exception:
            return 0.5


class C2Processor:
    def __init__(self, args):
        self.args = args

        self.receiver = TSNReceiver(args.listen_ip, args.port)
        self.transreid_processor = TransReIDProcessor(model_path=args.transreid_model_path)
        self.weaviate_manager = WeaviateReIDManager(
            args.weaviate_url,
            "PersonReID",
            similarity_threshold=args.similarity_threshold,
            store_crops=getattr(args, "store_crops", False),
        )

        self.results_saver = None
        if args.save_results:
            self.results_saver = ReIDResultsSaver(self.weaviate_manager.mongo_storage)
            print("Results saver enabled - will save to 'results' folder")

        self.person_transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print("TSN receiver with Weaviate ReID initialized")

    def crop_b64_to_processed_image(self, crop_b64: str) -> Optional[List]:
        try:
            crop_bytes = base64.b64decode(crop_b64)
            arr = np.frombuffer(crop_bytes, dtype=np.uint8)
            crop_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if crop_bgr is None:
                return None

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.person_transform(crop_rgb)
            return tensor.tolist()

        except Exception as e:
            print(f"Failed to decode crop_jpg_b64: {e}")
            return None

    def normalize_incoming_data(self, data: Dict) -> Dict:
        normalized_objects = []

        for i, obj in enumerate(data.get("objects", [])):
            new_obj = dict(obj)

            if "processed_image" not in new_obj and "crop_jpg_b64" in new_obj:
                processed = self.crop_b64_to_processed_image(new_obj["crop_jpg_b64"])
                if processed is not None:
                    new_obj["processed_image"] = processed

            if "object_id" not in new_obj:
                new_obj["object_id"] = i

            normalized_objects.append(new_obj)

        data["objects"] = normalized_objects
        return data

    def process_detection(self, data: Dict) -> List[Dict]:
        try:
            data = self.normalize_incoming_data(data)

            frame_id = data["metadata"].get("frame_id", "unknown")
            camera_id = data["metadata"].get("camera_id", "unknown")
            print(f"Processing frame {frame_id} from camera {camera_id}")

            valid_objects = [
                obj for obj in data.get("objects", [])
                if "processed_image" in obj and obj.get("class_name") == "person"
            ]

            if not valid_objects:
                print("No valid person objects with processed_image found")
                return []

            data["objects"] = valid_objects

            raw_features = objects_to_tensor(data["objects"])
            reid_features = self.transreid_processor.extract_features(raw_features)
            print(f"Extracted features for {len(data.get('objects', []))} objects")

            reid_features = torch.nn.functional.normalize(reid_features, dim=1, p=2)

            reid_results = self.weaviate_manager.process_and_identify(
                data, reid_features
            )

            if self.args.save_results:
                self.save_simple_results(data, reid_results)

            return reid_results

        except Exception as e:
            print(f"Error processing detection: {e}")
            return []

    def log_rx(self, data: Dict, status: str = "received"):
        metadata = data.get("metadata", {})

        payload_type = data.get("type") or metadata.get("payload_type", "unknown")
        frame_id = metadata.get("frame_id", "NA")
        camera_id = metadata.get("camera_id", "NA")
        vlan_id = metadata.get("vlan_id", "NA")
        vlan_interface = metadata.get("vlan_interface", "NA")
        source_addr = data.get("_source_addr", "NA")
        num_objects = len(data.get("objects", [])) if "objects" in data else 0

        print(
            f"[RX] status={status} "
            f"type={payload_type} "
            f"frame_id={frame_id} "
            f"camera={camera_id} "
            f"objects={num_objects} "
            f"vlan={vlan_id} "
            f"iface={vlan_interface} "
            f"src={source_addr}"
        )


    def handle_raw_frame(self, data: Dict):
        try:
            metadata = data.get("metadata", {})
            frame_id = metadata.get("frame_id", "unknown")
            camera_id = metadata.get("camera_id", "unknown")

            frame_b64 = data.get("frame_jpg_b64")
            if not frame_b64:
                print(f"Raw frame packet missing frame_jpg_b64, frame_id={frame_id}")
                return

            frame_bytes = base64.b64decode(frame_b64)
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"Could not decode raw frame, frame_id={frame_id}")
                return

            save_dir = "received_raw_frames"
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.join(
                save_dir,
                f"{camera_id}_frame_{frame_id}.jpg",
            )

            cv2.imwrite(filename, frame)

            # print(
            #     f"[RAW FRAME RX] frame_id={frame_id}, "
            #     f"camera={camera_id}, saved={filename}, "
            #     f"source={data.get('_source_addr')}"
            # )
            self.log_rx(data, status=f"raw_frame_saved saved={filename}")

        except Exception as e:
            print(f"Error handling raw frame: {e}")



    # def run(self):
    #     print("Starting continuous TSN packet processing...")

    #     while True:
    #         try:
    #             data = self.receiver.get_data()
    #             if data:
    #                 results = self.process_detection(data)

    #                 if results and self.args.save_results and self.args.save_json:
    #                     self.save_results_summary(results)

    #         except KeyboardInterrupt:
    #             print("\nStopping C2 processor...")
    #             break
    #         except Exception as e:
    #             print(f"Unexpected error: {e}")
    #             time.sleep(1)
    #             continue


    def run(self):
        print("Starting continuous TSN packet processing...")

        while True:
            try:
                data = self.receiver.get_data()

                if data is None:
                    continue

                payload_type = data.get("type") or data.get("metadata", {}).get("payload_type")

                if payload_type == "raw_frame":
                    # self.log_rx(data, status="raw_frame_received")
                    self.handle_raw_frame(data)
                    continue

                if payload_type == "detected_objects":
                    if not data.get("objects"):
                        # frame_id = data.get("metadata", {}).get("frame_id", "unknown")
                        # vlan_id = data.get("metadata", {}).get("vlan_id", "unknown")
                        # print(f"[EMPTY OBJECT RX] frame_id={frame_id}, vlan={vlan_id}")

                        self.log_rx(data, status="empty_object_packet")
                        continue

                    self.log_rx(data, status="object_packet_received")

                    results = self.process_detection(data)

                    if results and self.args.save_results and self.args.save_json:
                        self.save_results_summary(results)

                    continue

                # print(f"Unknown payload type: {payload_type}")
                self.log_rx(data, status="unknown_payload")

            except KeyboardInterrupt:
                print("\nStopping C2 processor...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(1)
                continue



    def save_results_summary(self, results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reid_results_{timestamp}.json"

        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def save_simple_results(self, data: Dict, reid_results: List[Dict]):
        try:
            frame_id = str(data["metadata"].get("frame_id", "unknown"))
            camera_id = str(data["metadata"].get("camera_id", "unknown"))

            if not self.results_saver:
                print("Results saver not initialized.")
                return

            saver = self.results_saver
            print(f"Saving results for {len(reid_results)} detected persons...")

            for result in reid_results:
                person_id = result.get("person_id", "unknown")
                detection_id = result.get("detection_id")

                person_obj = None
                for obj in data.get("objects", []):
                    if str(obj.get("object_id", "")) == str(detection_id):
                        person_obj = obj
                        break

                if person_obj is None:
                    continue

                try:
                    obj_tensor = objects_to_tensor([person_obj])
                    obj_features = self.transreid_processor.extract_features(obj_tensor)
                    obj_features = torch.nn.functional.normalize(obj_features, dim=1, p=2)
                    query_embedding = obj_features.cpu().numpy().flatten()

                    query_image_data = None
                    if "processed_image" in person_obj:
                        img_list = person_obj["processed_image"]
                        tensor = torch.tensor(img_list)

                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                        unnormalized = tensor * std + mean
                        unnormalized = unnormalized.clamp(0, 1)

                        to_pil = T.ToPILImage()
                        pil_image = to_pil(unnormalized)

                        import io
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format="PNG")
                        query_image_data = img_buffer.getvalue()

                except Exception as e:
                    print(f"Could not extract features for person {person_id}: {e}")
                    continue

                similar_embeddings = []
                if result.get("similar_detections", 0) > 0:
                    try:
                        similar_matches = self.weaviate_manager.vector_store.search_similar(
                            obj_features, top_k=5, distance_threshold=0.8
                        )

                        for match in similar_matches:
                            if isinstance(match, dict) and "vector" in match:
                                raw_vector = match["vector"]
                                vector_array = None

                                if isinstance(raw_vector, dict):
                                    for key in ["values", "data", "vector", "default"]:
                                        if key in raw_vector:
                                            vector_array = np.array(raw_vector[key], dtype=np.float32)
                                            break
                                elif isinstance(raw_vector, (list, np.ndarray)):
                                    vector_array = np.array(raw_vector, dtype=np.float32)

                                if vector_array is not None and vector_array.size > 0:
                                    similar_embeddings.append(vector_array)

                    except Exception as e:
                        print(f"Error getting similar embeddings from Weaviate: {e}")

                query_name = f"person_{person_id}_frame_{frame_id}_cam_{camera_id}"

                try:
                    saver.save_query_results_with_image(
                        query_embedding=query_embedding,
                        similar_embeddings=similar_embeddings,
                        query_name=query_name,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        query_image_data=query_image_data,
                    )
                except Exception as e:
                    print(f"Error saving results for person {person_id}: {e}")

        except Exception as e:
            print(f"Error in save_simple_results: {e}")

    def close(self):
        self.receiver.close()

    def stop(self):
        print("\nShutting down gracefully...")
        # Close Weaviate
        if hasattr(self, 'weaviate_manager') and self.weaviate_manager.client:
            self.weaviate_manager.client.close()
            print("✓ Weaviate connection closed.")
        
        # Close MongoDB
        if hasattr(self, 'mongo_storage') and self.mongo_storage.client:
            self.mongo_storage.client.close()
            print("✓ MongoDB connection closed.")


def parse_args():
    parser = argparse.ArgumentParser(description="C2 ReID with TSN/UDP input and Weaviate")
    parser.add_argument("--listen_ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--weaviate_url", type=str, default="http://localhost:8080")
    parser.add_argument(
        "--transreid_model_path",
        type=str,
        default="/home/jdg24001/Documents/github/Secure-Camera/weights-models/weights/transformer_best.pth",
    )
    parser.add_argument("--similarity_threshold", type=float, default=0.7)
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--store_crops", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("===== ARGUMENTS =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=====================")

    processor = None
    try:
        processor = C2Processor(args)
        processor.run()
    except KeyboardInterrupt:
        print("\nReceiver stopped by user.")
    finally:
        if processor is not None:
            processor.close()


if __name__ == "__main__":
    main()