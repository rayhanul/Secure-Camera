import argparse
import json
import os
import time
import socket
import zlib
import base64
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torchvision import transforms as T
from utils.image_utils import encode_image_to_base64
from utils.reid_result import ReIDResult  # Just a data class
from utils.results_saver import ReIDResultsSaver
from utils.storage import SecureReIDStorage

# TransReID imports will be handled within the TransReIDProcessor class
from utils.util import objects_to_tensor
from utils.weaviate import ReIDVectorStore

# ================= PREDICTION MODIFICATION 1 =================
# Keep object-history storage, model feeding, and prediction logic in a
# separate module placed in the same directory as this main file.
from prediction_module.object_prediction import ExistingObjectPredictor

# Prediction is internal program behavior; it does not require CLI arguments.
# Change values here if the camera or trained model configuration changes.
PREDICTION_CONFIG = {
    "enabled": True,
    # True: also predict frame N+1 after complete frame N data is received.
    # False: predict only when an object or one of its properties is missing.
    "continuous_prediction": True,
    # Evaluation mode: deliberately hide selected fields from some known
    # objects, predict them, and compare against the untouched received values.
    "simulate_missing_fields": True,
    "simulation_every_n_frames": 1,
    "simulation_object_stride": 2,
    "simulation_min_history": 3,
    "simulation_fields": [
        "bbox",
        "distance_m",
        "bearing_deg",
        "direction",
        "speed_kmh",
    ],
    "camera_fps": 25.0,
    "history_size": 10,
    "max_missed_frames": 3,
    "max_speed_kmh": 180.0,
    "frame_width": 1920,
    "frame_height": 1080,
    "max_distance_m": 500.0,
    "max_bearing_deg": 45.0,
    # Set this to a trained ObjectMotionGRU checkpoint when available.
    "model_path": None,
    "model_device": "auto",
    "model_hidden_size": 128,
    "model_num_layers": 2,
    # Store only actual observations here. Predictions are stored separately.
    "history_store_path": os.path.join(
        "prediction_data",
        "actual_object_history.jsonl",
    ),
    "save_predictions": True,
    "prediction_output_dir": os.path.join(
        "prediction_data",
        "predictions",
    ),
}
# =============== END PREDICTION MODIFICATION 1 ===============


import struct

MAX_DGRAM = 60000

SINGLE_PACKET_TYPE = 0
CHUNK_PACKET_TYPE = 1

HEADER_FORMAT = "!HIHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


import pdb; 


import sys

TRANSREID_ROOT = "/home/jdg24001/Documents/github/Secure-Camera/C2/TransReID"

if TRANSREID_ROOT not in sys.path:
    sys.path.insert(0, TRANSREID_ROOT)
    
    

def parse_timestamp(timestamp_value):
    if timestamp_value is None:
        return time.time()

    if isinstance(timestamp_value, int) or isinstance(timestamp_value, float):
        return float(timestamp_value)

    if isinstance(timestamp_value, str):
        try:
            return float(timestamp_value)
        except ValueError:
            try:
                return datetime.fromisoformat(timestamp_value).timestamp()
            except ValueError:
                return time.time()

    return time.time()






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
    """Proper TransReID model for feature extraction"""

    def __init__(
        self,
        model_path="/home/jdg24001/Documents/github/Secure-Camera/weights-models/jx_vit_base_p16_224-80ecf9dd.pth",
        config_path="/home/jdg24001/Documents/github/Secure-Camera/weights-models/vit_transreid_stride.yml",
    ):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Device selected: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        else:
            print("No GPU available, using CPU")
        self.load_model(model_path, config_path)

    def load_model(self, model_path=None, config_file=None):
        """Load TransReID model with multiple fallback strategies"""

        # Set default model path

        
        if model_path is None:
            model_path = "/home/jdg24001/Documents/github/Secure-Camera/weights-models/jx_vit_base_p16_224-80ecf9dd.pth"

        try:
            print("Loading TransReID model...")

            # Import TransReID config
            from TransReID.config import cfg

            # Configure the model
            cfg.MODEL.NAME = "transformer"
            cfg.MODEL.TRANSFORMER_TYPE = "vit_base_patch16_224_TransReID"
            cfg.MODEL.SIE_CAMERA = False
            cfg.MODEL.SIE_VIEW = False
            cfg.MODEL.JPM = True
            cfg.MODEL.PRETRAIN_CHOICE = "self"
            cfg.MODEL.PRETRAIN_PATH = (
                "/home/jdg24001/Documents/github/Secure-Camera/weights-models/jx_vit_base_p16_224-80ecf9dd.pth"
            )
            cfg.TEST.WEIGHT = model_path
            cfg.INPUT.SIZE_TEST = [256, 128]
            cfg.INPUT.SIZE_TRAIN = [256, 128]
            cfg.MODEL.DEVICE = self.device
            cfg.freeze()

            # Import and create model
            from TransReID.model import make_model

            # Create model without SIE (no camera/view settings needed)
            self.model = make_model(cfg, num_class=751, camera_num=0, view_num=0)

            # Try to load weights
            if os.path.exists(model_path):
                try:
                    self.model.load_param(model_path)
                    print(f" Successfully loaded weights from {model_path}")
                except Exception as e:
                    print(f"Standard loading failed: {e}")
                    print("Trying partial weight loading...")
                    self._load_weights_partially(model_path)
                    print(" Partially loaded weights")
            else:
                print(f"Warning: Weights not found at {model_path}")

            self.model.to(self.device)
            self.model.eval()
            print(f" TransReID model loaded successfully on {self.device}")

        except Exception as e:
            print(f" Error loading TransReID model: {e}")
            import traceback

            traceback.print_exc()
            print("Falling back to feature passthrough mode")
            self.model = None

    def _load_weights_partially(self, model_path):
        """Load weights partially, skipping incompatible layers"""
        import torch

        try:
            param_dict = torch.load(model_path, map_location="cpu")
            model_dict = self.model.state_dict()

            # Filter out incompatible parameters
            compatible_dict = {}
            incompatible_keys = []

            for key in param_dict:
                clean_key = key.replace("module.", "")
                if clean_key in model_dict:
                    if param_dict[key].shape == model_dict[clean_key].shape:
                        compatible_dict[clean_key] = param_dict[key]
                    else:
                        incompatible_keys.append(f"{clean_key} (shape mismatch)")
                else:
                    incompatible_keys.append(f"{clean_key} (missing in model)")

            # Load compatible parameters
            model_dict.update(compatible_dict)
            self.model.load_state_dict(model_dict, strict=False)

            print(f"Loaded {len(compatible_dict)}/{len(param_dict)} parameters")
            if incompatible_keys:
                print(
                    f"Skipped incompatible keys: {incompatible_keys[:5]}..."
                )  # Show first 5

        except Exception as e:
            print(f"Partial loading also failed: {e}")
            raise e

    def extract_features(self, person_images):
        """Extract ReID features from person images"""

        # if self.model is None:
        #     # Fallback: just normalize input features
        #     print(
        #         "TransReID model failed to load for feature extraction. "
        #         "Falling back to feature passthrough mode"
        #     )
        #     return torch.nn.functional.normalize(person_images, dim=1, p=2)
        if self.model is None:
            print(
                "TransReID model failed to load for feature extraction. "
                "Using fallback 384-dim embedding"
            )

            if person_images.dim() == 3:
                person_images = person_images.unsqueeze(0)

            flat = person_images.flatten(start_dim=1)

            if flat.shape[1] >= 384:
                features = flat[:, :384]
            else:
                pad_size = 384 - flat.shape[1]
                features = torch.nn.functional.pad(flat, (0, pad_size))

            return torch.nn.functional.normalize(features, dim=1, p=2)

        try:
            """Context-manager that disables gradient calculation.
            Disabling gradient calculation is useful for inference, when you are sure
            that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.
            """
            with torch.no_grad():
                print(f"Input tensor shape: {person_images.shape}")
                # print(f"Input tensor dtype: {person_images.dtype}")
                print(f"Input tensor device: {person_images.device}")

                person_images = person_images.to(self.device)
                print(f"🚀 Running inference on: {self.device}")
                if self.device == "cuda":
                    print(
                        f"⚡ GPU Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
                    )

                # Add input validation
                if person_images.dim() != 4:
                    raise ValueError(f"Expected 4D tensor, got {person_images.dim()}D")
                if person_images.size(1) != 3:
                    raise ValueError(
                        f"Expected 3 channels, got {person_images.size(1)}"
                    )
                if person_images.size(2) != 256 or person_images.size(3) != 128:
                    raise ValueError(
                        f"Expected 256x128 images, got {person_images.size(2)}x{person_images.size(3)}"
                    )

                # Calling the model - Step 7
                # print("Calling TransReID model...")
                features = self.model(person_images)
                # print(f"Model output shape: {features.shape}")
                if self.device == "cuda":
                    print(
                        f"📊 GPU Memory after inference: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
                    )
                return features.cpu()
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            import traceback

            traceback.print_exc()
            # print("Falling back to feature passthrough mode")
            # return torch.nn.functional.normalize(person_images, dim=1, p=2)
            print("Using fallback 384-dim embedding")

            if person_images.dim() == 3:
                person_images = person_images.unsqueeze(0)

            flat = person_images.flatten(start_dim=1)

            if flat.shape[1] >= 384:
                features = flat[:, :384]
            else:
                pad_size = 384 - flat.shape[1]
                features = torch.nn.functional.pad(flat, (0, pad_size))

            return torch.nn.functional.normalize(features.cpu(), dim=1, p=2)


class WeaviateReIDManager:
    """Enhanced Weaviate manager for complete ReID operations"""

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

        # Initialize MongoDB Storage - Encrypted
        self.mongo_storage = SecureReIDStorage()
        print("🔒 MongoDB storage initialized for gallery data")

    def process_and_identify(
        self, objects_data: Dict, reid_features: torch.Tensor
    ) -> List[Dict]:
        """
        Core ReID functionality:
        1. Store new features in Weaviate
        2. Find similar existing persons
        3. Assign person IDs
        4. Update person profiles
        """
        results = []
        print(f"\nTotal number of received objects: {len(objects_data["objects"])}\n")

        for i, obj in enumerate(objects_data["objects"]):
            person_feature = reid_features[i : i + 1]  # Shape: [1, feature_dim]

            # 1. Search for similar persons in database
            similar_persons = self.find_similar_persons(
                person_feature,
                obj.get("class_name", "person"),
                objects_data["metadata"]["camera_id"],
            )

            # 2. Determine person identity
            person_identity = self.determine_identity(
                similar_persons, person_feature, obj
            )

            print(
                f"[IDENTITY] object_id={obj.get('object_id', i)} "
                f"person_id={person_identity['person_id']} "
                f"is_new={person_identity['is_new']} "
                f"confidence={person_identity['confidence']:.3f} "
                f"similar_matches={len(similar_persons)}"
            )

            # 3. Store/Update in Weaviate
            store_crops = getattr(self, "store_crops", False)
            print(f"🔧 store_crops={store_crops} for object {i}")
            if store_crops:
                print(f"🖼️  Storing with image crops enabled for object {i}")
            storage_result = self.store_person_data(
                objects_data, obj, person_feature, person_identity, i, store_crops
            )

            # 4. Compile result
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
                "weaviate_id": storage_result[0]
                if storage_result and len(storage_result) > 0
                else None,
            }

            results.append(result)

        return results

    def find_similar_persons(
        self, query_feature: torch.Tensor, class_filter: str, current_camera: str
    ) -> List[Dict]:
        """Find similar persons in the vector database"""
        try:
            # Search in Weaviate for similar embeddings
            similar_results = self.vector_store.search_similar(
                query_feature,
                top_k=20,  # Get more results for better matching
                class_filter=class_filter,
                confidence_threshold=0.3,  # Lower threshold for initial search
                distance_threshold=2.0,  # Euclidean distance threshold
            )

            # Filter by similarity threshold and add distance scores
            filtered_results = []
            for result in similar_results:
                # Calculate actual similarity score (you might need to implement this)
                similarity_score = self.calculate_similarity(query_feature, result)

                if similarity_score >= self.similarity_threshold:
                    result["similarity_score"] = similarity_score
                    result["is_cross_camera"] = result["camera_id"] != current_camera
                    filtered_results.append(result)

            print(f"done with find_similar_persons, and found: {len(filtered_results)}")
            return filtered_results

        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def determine_identity(
        self, similar_persons: List[Dict], person_feature: torch.Tensor, obj: Dict
    ) -> Dict:
        """Determine if this is a new person or matches existing identity"""

        if not similar_persons:
            # New person - assign new ID
            self.person_id_counter += 1
            return {
                "person_id": f"person_{self.person_id_counter:06d}",
                "confidence": 1.0,
                "is_new": True,
                "matched_detection": None,
            }

        # Find best match
        best_match = max(similar_persons, key=lambda x: x.get("similarity_score", 0))

        if best_match["similarity_score"] >= self.similarity_threshold:
            # Existing person identified
            cross_camera_matches = [
                p
                for p in similar_persons
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
                "cross_camera_matches": cross_camera_matches[
                    :5
                ],  # Top 5 cross-camera matches
            }

            # object_id = best_match.get("person_id")

            # if object_id is None or object_id == "":
            #     self.person_id_counter += 1
            #     matched_person_id = f"person_{self.person_id_counter:06d}"
            # else:
            #     matched_person_id = f"unknown_{object_id}"
                

            # return {
            #     "person_id": matched_person_id,
            #     "confidence": best_match["similarity_score"],
            #     "is_new": False,
            #     "matched_detection": best_match,
            #     "cross_camera_matches": cross_camera_matches[:5],
            # }


        else:
            # No confident match - new person
            print("new person detected!")
            self.person_id_counter += 1
            return {
                "person_id": f"person_{self.person_id_counter:06d}",
                "confidence": 0.6,  # Lower confidence for borderline cases
                "is_new": True,
                "matched_detection": None,
            }
        print("done with identity detemination")

    def store_person_data(
        self,
        objects_data: Dict,
        obj: Dict,
        person_feature: torch.Tensor,
        person_identity: Dict,
        index: int,
        store_crops: bool = False,
    ):
        """Store person data with assigned identity in Weaviate"""
        try:
            # Use processed_image - it's a normalized tensor that needs unnormalization
            person_crop = None
            if "processed_image" in obj:
                img_list = obj["processed_image"]

                print(f"📷 Raw tensor shape: {np.array(img_list).shape}")

                # Convert JSON list -> torch tensor
                tensor = torch.tensor(img_list)  # shape: (3, H, W)

                # Undo ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                unnormalized = tensor * std + mean  # reverse normalization
                unnormalized = unnormalized.clamp(0, 1)

                # print(
                #     f"🔄 Unnormalized tensor range: [{unnormalized.min():.3f}, {unnormalized.max():.3f}]"
                # )

                # Convert to PIL image
                to_pil = T.ToPILImage()
                pil_image = to_pil(unnormalized)

                # Convert PIL to numpy array for base64 encoding
                person_crop_array = np.array(pil_image)

                # convert to bytes (PNG format for storage)
                import io

                from PIL import Image

                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format="PNG")
                person_crop_bytes = img_buffer.getvalue()

            if person_identity.get("person_id") is None:
                self.person_id_counter += 1
                person_identity["person_id"] = f"person_{self.person_id_counter:06d}"

            mongo_result = ReIDResult(
                object_id=str(obj.get("object_id", f"obj_{index}")),
                class_name=obj.get("class_name", "unknown"),
                confidence=float(obj.get("confidence", 0.0)),
                bbox=obj.get("bbox", [0, 0, 0, 0]),
                camera_id=str(objects_data["metadata"].get("camera_id", "unknown")),
                camera_location=str(objects_data["metadata"].get("camera_location", "unknown")),
                frame_id=int(objects_data["metadata"].get("frame_id", 0)),
                # timestamp=float(objects_data["metadata"].get("timestamp", 0)),
                timestamp=parse_timestamp(objects_data["metadata"].get("timestamp")),
                embedding_method="TransReID",
                reid_confidence=person_identity["confidence"],
                person_id=person_identity["person_id"],
                is_new_person=person_identity["is_new"],
                image=person_crop_bytes
                if person_crop_bytes
                else b"",  # Raw image bytes
            )

            # store data in mongo
            embedding_vector = person_feature.cpu().numpy()
            self.mongo_storage.store_reid_result(embedding_vector, mongo_result)
            print(
                f"Stored encrypted data in mongo for person {person_identity['person_id']} "
            )

            # Process detection to include image crop if available and enabled
            # Store the person crop directly as base64 if available
            # enhanced_obj = obj.copy()
            # if store_crops and person_crop is not None:
            #     try:
            #         base64_crop = encode_image_to_base64(person_crop)
            #         if base64_crop:
            #             enhanced_obj["image_crop_base64"] = base64_crop
            #             print(f" Encoded person crop to base64 for object {index}")
            #         else:
            #             print(f"⚠️ Failed to encode person crop for object {index}")
            #     except Exception as e:
            #         print(f" Error encoding person crop for object {index}: {e}")
            # elif store_crops:
            #     print(f"⚠️ No processed_image found for object {index}")

            # Prepare data object with person identity to store in weaviate - minimal information
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

            # Store in Weaviate with image crops if available
            result = self.vector_store.store_embeddings(
                objects_data_with_person_id, person_feature
            )

            return result if result else None

        except Exception as e:
            print(f"Error storing person data: {e}")
            import traceback

            traceback.print_exc()
            return None

    def calculate_similarity(self, query_feature: torch.Tensor, result: Dict) -> float:
        """Calculate similarity score between query and result"""
        # This is a placeholder - you might need to implement proper similarity calculation
        # based on the distance returned by Weaviate
        try:
            # Weaviate returns distance, convert to similarity (0-1 scale)
            distance = result.get("_additional", {}).get("distance", 1.0)
            similarity = max(
                0, 1.0 - (distance / 2.0)
            )  # Normalize distance to similarity
            return similarity
        except:
            return 0.5  # Default similarity

    # def get_person_history(self, person_id: str, limit: int = 10) -> List[Dict]:
    #     """Get detection history for a specific person"""
    #     try:
    #         # Query Weaviate for all detections of this person
    #         query_builder = (
    #             self.vector_store.client.query.get(
    #                 self.vector_store.collection_name,
    #                 [
    #                     "object_id",
    #                     "camera_id",
    #                     "frame_id",
    #                     "timestamp",
    #                     "bbox",
    #                     "confidence",
    #                 ],
    #             )
    #             .with_where(
    #                 {
    #                     "path": ["person_id"],
    #                     "operator": "Equal",
    #                     "valueString": person_id,
    #                 }
    #             )
    #             .with_limit(limit)
    #             .with_sort([{"path": ["timestamp"], "order": "desc"}])
    #         )

    #         results = query_builder.do()
    #         return results["data"]["Get"][self.vector_store.collection_name]

    #     except Exception as e:
    #         print(f"Error getting person history: {e}")
    #         return []
    def get_person_history(self, person_id: str, limit: int = 10) -> List[Dict]:
        try:
            if person_id is None or person_id == "":
                print("Cannot get history because person_id is None")
                return []

            from weaviate.classes.query import Filter

            collection = self.vector_store.client.collections.get(
                self.vector_store.collection_name
            )

            response = collection.query.fetch_objects(
                filters=Filter.by_property("person_id").equal(person_id),
                limit=limit,
            )

            history = []

            for obj in response.objects:
                item = dict(obj.properties)
                item["weaviate_id"] = str(obj.uuid)
                history.append(item)

            return history

        except Exception as e:
            print(f"Error getting person history: {e}")
            return []


class C2Processor:
    """Main C2 processor with complete Pose2ID and Weaviate integration"""

    def __init__(self, args):
        self.args = args
        # Open the UDP receiver only after every other dependency has
        # initialized successfully.  If predictor or Weaviate startup fails,
        # no bound socket is left behind.
        self.receiver = None

        # ================= PREDICTION MODIFICATION 2 =================
        # Create one predictor for all cameras. It internally separates history
        # by (camera_id, camera_location, track_id). A trained GRU checkpoint is
        # optional; without it, robust history extrapolation is used.
        self.object_predictor = None
        if PREDICTION_CONFIG["enabled"]:
            self.object_predictor = ExistingObjectPredictor(
                fps=PREDICTION_CONFIG["camera_fps"],
                history_size=PREDICTION_CONFIG["history_size"],
                max_missed_frames=PREDICTION_CONFIG["max_missed_frames"],
                max_radial_speed_kmh=PREDICTION_CONFIG["max_speed_kmh"],
                frame_width=PREDICTION_CONFIG["frame_width"],
                frame_height=PREDICTION_CONFIG["frame_height"],
                model_path=PREDICTION_CONFIG["model_path"],
                model_device=PREDICTION_CONFIG["model_device"],
                model_hidden_size=PREDICTION_CONFIG["model_hidden_size"],
                model_num_layers=PREDICTION_CONFIG["model_num_layers"],
                max_distance_m=PREDICTION_CONFIG["max_distance_m"],
                max_bearing_deg=PREDICTION_CONFIG["max_bearing_deg"],
                history_store_path=PREDICTION_CONFIG["history_store_path"],
                continuous_prediction=PREDICTION_CONFIG[
                    "continuous_prediction"
                ],
            )
            prediction_method = (
                "trained GRU"
                if self.object_predictor.model_enabled
                else "robust history fallback"
            )
            print(
                "Existing-object prediction enabled: "
                f"method={prediction_method}, "
                f"history={PREDICTION_CONFIG['history_size']}, "
                f"max_missed={PREDICTION_CONFIG['max_missed_frames']}, "
                f"continuous={PREDICTION_CONFIG['continuous_prediction']}, "
                f"input_store={PREDICTION_CONFIG['history_store_path']}"
            )
        # =============== END PREDICTION MODIFICATION 2 ===============
        
        self.transreid_processor = TransReIDProcessor(model_path=args.transreid_model_path)
        self.weaviate_manager = WeaviateReIDManager(
            args.weaviate_url,
            "PersonReID",
            similarity_threshold=args.similarity_threshold,
            store_crops=getattr(args, "store_crops", False),
        )

        # Initialize results saver if saving is enabled
        if args.save_results:
            print("Results saver enabled - will save to 'results' folder")

        self.person_transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Runtime statistics... ... ... 

        self.start_time = time.time()
        self.frames_processed = 0
        self.persons_detected = 0
        self.new_persons = 0
        self.existing_persons = 0
        self.stats_interval = 15

        # Keep this last: C2Processor.__init__ must not leave an open UDP
        # socket when an earlier dependency raises an exception.
        self.receiver = TSNReceiver(args.listen_ip, args.port)
        print("TSN receiver with Weaviate ReID initialized")

    # ================= PREDICTION MODIFICATION 3 =================
    @staticmethod
    def _first_packet_value(mapping: Dict, *names, default=None):
        for name in names:
            value = mapping.get(name)
            if value is not None:
                return value
        return default

    def print_received_object_properties(self, data: Dict) -> None:
        """Print every received object before prediction or ReID filtering.

        A track is labelled ``existing`` when its ID already has real
        observations in the server-side prediction history. This check is
        performed before the current packet is written to history.
        """
        metadata = data.get("metadata") or {}
        objects = data.get("objects") or []

        def first_value(obj: Dict, *names, default=None):
            for name in names:
                value = obj.get(name)
                if value is not None:
                    return value
            return default

        frame_id = first_value(
            metadata, "frame_id", default=data.get("frame_id", "unknown")
        )
        camera_id = first_value(
            metadata,
            "camera_id",
            "camera",
            default=data.get("camera_id", "unknown"),
        )
        camera_location = first_value(
            metadata,
            "camera_location",
            "location",
            default=data.get("camera_location", "unknown"),
        )

        track_states = []
        existing_count = 0
        new_count = 0
        for obj in objects:
            if not isinstance(obj, dict):
                track_states.append("unknown")
                continue
            track_id = first_value(
                obj, "track_id", "object_track_id", "object_id"
            )
            is_existing = bool(
                track_id is not None
                and self.object_predictor is not None
                and self.object_predictor.has_prior_history(
                    camera_id,
                    camera_location,
                    track_id,
                    frame_id,
                )
            )
            track_state = "existing" if is_existing else "new"
            track_states.append(track_state)
            if is_existing:
                existing_count += 1
            else:
                new_count += 1

        print(
            f"\n[RECEIVED OBJECTS] frame_id={frame_id} "
            f"camera={camera_id} location={camera_location} "
            f"count={len(objects)} existing={existing_count} new={new_count}",
            flush=True,
        )

        if not objects:
            print("  No object records were received in this packet.", flush=True)
            return

        for index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                print(
                    f"  received object {index}: invalid_record={obj!r}",
                    flush=True,
                )
                continue

            explicit_has_crop = obj.get("has_crop")
            has_crop = (
                bool(explicit_has_crop)
                if explicit_has_crop is not None
                else any(
                    obj.get(name) not in (None, "")
                    for name in (
                        "crop_jpg_b64",
                        "image_crop_base64",
                        "crop",
                    )
                )
            )

            explicit_has_processed = obj.get("has_processed")
            has_processed = (
                bool(explicit_has_processed)
                if explicit_has_processed is not None
                else obj.get("processed_image") is not None
            )

            print(
                f"  received {track_states[index]} object {index}: "
                f"class={first_value(obj, 'class_name', 'class', default='unknown')} "
                f"conf={first_value(obj, 'confidence', 'conf')} "
                f"track_id={first_value(obj, 'track_id', 'object_track_id', 'object_id')} "
                f"bbox={obj.get('bbox')} "
                f"distance_m={first_value(obj, 'distance_m', 'distance')} "
                f"bearing_deg={first_value(obj, 'bearing_deg', 'bearing_angle')} "
                f"direction={first_value(obj, 'direction', 'motion_direction')} "
                f"speed_kmh={first_value(obj, 'speed_kmh', 'speed')} "
                f"privacy_level={obj.get('privacy_level')} "
                f"jpeg_crop_size={obj.get('jpeg_crop_size')} "
                f"serialized_object_size={obj.get('serialized_object_size')} "
                f"has_crop={has_crop} "
                f"has_processed={has_processed}",
                flush=True,
            )

    @classmethod
    def _minimal_prediction_observation(cls, obj: Dict) -> Optional[Dict]:
        """Remove crops, embeddings, encryption, and detailed metadata.

        The returned record contains only fields needed to identify a track,
        update its numerical history, or preserve an actually received motion
        value while the remaining properties are predicted.
        """
        track_id = cls._first_packet_value(
            obj, "track_id", "object_track_id", "object_id"
        )
        if track_id is None:
            return None

        minimal = {"track_id": track_id}
        aliases = {
            "class_name": ("class_name", "class"),
            "class_id": ("class_id",),
            "confidence": ("confidence", "conf"),
            "bbox": ("bbox",),
            "distance_m": ("distance_m", "distance"),
            "bearing_deg": ("bearing_deg", "bearing_angle"),
            "direction": ("direction", "motion_direction"),
            "speed_kmh": ("speed_kmh", "speed"),
        }
        for output_name, input_names in aliases.items():
            value = cls._first_packet_value(obj, *input_names)
            if value is not None:
                minimal[output_name] = value
        return minimal

    @classmethod
    def _packet_requests_prediction(cls, data: Dict) -> bool:
        metadata = data.get("metadata", {})
        mode = str(
            cls._first_packet_value(
                metadata,
                "transmission_mode",
                default=data.get("transmission_mode", ""),
            )
        ).lower()
        reduced_modes = {
            "new_objects_only",
            "reduced_existing_objects",
            "prediction_only",
            "gcl_reconfiguration",
            "empty_existing_data",
            "minimal_existing_data",
            "metadata_only",
        }
        return bool(
            metadata.get("existing_objects_omitted")
            or metadata.get("prediction_required")
            or metadata.get("object_data_empty")
            or metadata.get("existing_object_data_empty")
            or metadata.get("existing_track_ids")
            or data.get("existing_objects_omitted")
            or data.get("prediction_required")
            or data.get("object_data_empty")
            or data.get("existing_object_data_empty")
            or data.get("existing_track_ids")
            or mode in reduced_modes
        )

    def process_object_predictions(self, data: Dict) -> List[Dict]:
        """Adapt a full packet to the predictor's minimal-data interface.

        Predictions are attached to ``predicted_existing_objects`` instead of
        ``objects`` because they have no real crop or ReID embedding and must
        not be stored as actual observations.
        """
        # Print the real packet at the prediction boundary, before history is
        # updated and before process_detection() applies its person/ReID filter.
        self.print_received_object_properties(data)

        if self.object_predictor is None:
            return []

        metadata = data.get("metadata", {})
        frame_id = self._first_packet_value(metadata, "frame_id", default=0)
        try:
            frame_id = int(frame_id)
        except (TypeError, ValueError):
            frame_id = 0
        timestamp = self._first_packet_value(
            metadata,
            "capture_timestamp",
            "timestamp",
            "timestamp_s",
            default=None,
        )
        camera_id = metadata.get("camera_id", "unknown")
        camera_location = metadata.get("camera_location", "unknown")
        frame_width = self._first_packet_value(
            metadata,
            "frame_width",
            "image_width",
            default=PREDICTION_CONFIG["frame_width"],
        )
        frame_height = self._first_packet_value(
            metadata,
            "frame_height",
            "image_height",
            default=PREDICTION_CONFIG["frame_height"],
        )

        # The full packet stays in this server adapter. Only this sanitized
        # list crosses into the prediction module.
        minimal_observations = [
            observation
            for observation in (
                self._minimal_prediction_observation(obj)
                for obj in (data.get("objects") or [])
            )
            if observation is not None
        ]

        # Freeze new/existing status before storing anything from the current
        # frame. A track is existing only if it appeared in an earlier frame.
        prior_existing_track_ids = set()
        new_track_ids = set()
        for observation in minimal_observations:
            track_key = str(observation["track_id"])
            if self.object_predictor.has_prior_history(
                camera_id,
                camera_location,
                observation["track_id"],
                frame_id,
            ):
                prior_existing_track_ids.add(track_key)
            else:
                new_track_ids.add(track_key)

        existing_full_observations = []
        partial_descriptors = []
        fully_received_track_ids = set()
        for observation in minimal_observations:
            track_id = str(observation["track_id"])
            was_seen_in_previous_frame = (
                track_id in prior_existing_track_ids
            )
            missing_fields = self.object_predictor.missing_prediction_fields(
                observation
            )
            if not missing_fields:
                fully_received_track_ids.add(track_id)
                if was_seen_in_previous_frame:
                    existing_full_observations.append(observation)
            elif was_seen_in_previous_frame:
                partial_descriptors.append(observation)

        print(
            f"[PREDICTION ELIGIBILITY] frame_id={frame_id} "
            f"existing_track_ids={sorted(prior_existing_track_ids)} "
            f"new_track_ids={sorted(new_track_ids)} "
            f"new_objects_excluded={len(new_track_ids)}",
            flush=True,
        )

        # ================= SIMULATED MISSING-DATA EVALUATION =================
        # Keep the untouched values as ground truth, but remove configured
        # fields from every Nth known object before calling predict_tracks().
        simulation_ground_truth = []
        simulation_descriptors = []
        simulation_track_ids = set()
        simulation_enabled = bool(
            PREDICTION_CONFIG["simulate_missing_fields"]
        )
        simulation_every = max(
            1, int(PREDICTION_CONFIG["simulation_every_n_frames"])
        )
        simulation_stride = max(
            1, int(PREDICTION_CONFIG["simulation_object_stride"])
        )
        simulation_min_history = max(
            1, int(PREDICTION_CONFIG["simulation_min_history"])
        )
        simulation_fields = list(
            PREDICTION_CONFIG["simulation_fields"]
        )

        if simulation_enabled and frame_id % simulation_every == 0:
            for object_index, observation in enumerate(
                existing_full_observations
            ):
                if object_index % simulation_stride != 0:
                    continue
                track_id = observation["track_id"]
                if self.object_predictor.history_length(
                    camera_id, camera_location, track_id
                ) < simulation_min_history:
                    continue

                masked_descriptor = dict(observation)
                removed_fields = []
                for field in simulation_fields:
                    if field in masked_descriptor:
                        del masked_descriptor[field]
                        removed_fields.append(field)
                if not removed_fields:
                    continue

                track_key = str(track_id)
                simulation_track_ids.add(track_key)
                simulation_ground_truth.append(observation)
                simulation_descriptors.append(masked_descriptor)
                print(
                    f"[SIMULATED MISSING DATA] frame_id={frame_id} "
                    f"object_index={object_index} track_id={track_id} "
                    f"removed_fields={removed_fields} "
                    f"model_input_fields={sorted(masked_descriptor.keys())}"
                )

        # Compare ordinary cached predictions now. Ground-truth records chosen
        # for simulation are compared only after their masked prediction runs,
        # preventing duplicate comparisons for the same object and frame.
        ordinary_comparison_observations = [
            observation
            for observation in minimal_observations
            if str(observation["track_id"]) not in simulation_track_ids
        ]
        comparisons = self.object_predictor.compare_received_observations(
            camera_id=camera_id,
            camera_location=camera_location,
            frame_id=frame_id,
            observations=ordinary_comparison_observations,
        )
        # =============== END SIMULATED MISSING-DATA EVALUATION ===============

        prediction_requested = self._packet_requests_prediction(data)
        omitted_ids = self._first_packet_value(
            metadata,
            "omitted_track_ids",
            "existing_track_ids",
            "tracks_to_predict",
            default=self._first_packet_value(
                data,
                "omitted_track_ids",
                "existing_track_ids",
                "tracks_to_predict",
            ),
        )

        # Current-frame inference descriptors contain only the track ID unless
        # a partial lightweight record supplied one or more real properties.
        current_descriptors_by_id = {
            str(item["track_id"]): item for item in partial_descriptors
        }
        if isinstance(omitted_ids, (list, tuple, set)):
            for track_id in omitted_ids:
                track_key = str(track_id)
                if (
                    track_key not in fully_received_track_ids
                    and self.object_predictor.has_prior_history(
                        camera_id,
                        camera_location,
                        track_id,
                        frame_id,
                    )
                ):
                    current_descriptors_by_id.setdefault(
                        track_key, {"track_id": track_id}
                    )
        elif prediction_requested:
            for track_id in self.object_predictor.known_track_ids(
                camera_id, camera_location
            ):
                if (
                    track_id not in fully_received_track_ids
                    and self.object_predictor.has_prior_history(
                        camera_id,
                        camera_location,
                        track_id,
                        frame_id,
                    )
                ):
                    current_descriptors_by_id.setdefault(
                        track_id, {"track_id": track_id}
                    )

        predictions = []
        if current_descriptors_by_id:
            target_frame = self._first_packet_value(
                metadata,
                "prediction_target_frame_id",
                default=frame_id,
            )
            predictions.extend(
                self.object_predictor.predict_tracks(
                    camera_id=camera_id,
                    camera_location=camera_location,
                    reference_frame_id=frame_id,
                    reference_timestamp=timestamp,
                    target_frame_id=target_frame,
                    track_descriptors=list(
                        current_descriptors_by_id.values()
                    ),
                    frame_width=frame_width,
                    frame_height=frame_height,
                    prediction_reason="omitted_existing_object",
                    merge_partial_values=True,
                    actual_payload_pending=True,
                )
            )

        # Predict the deliberately hidden fields for the current frame using
        # history ending at frame N-1, then compare against the untouched
        # values that were actually received for frame N.
        if simulation_descriptors:
            simulated_predictions = self.object_predictor.predict_tracks(
                camera_id=camera_id,
                camera_location=camera_location,
                reference_frame_id=frame_id,
                reference_timestamp=timestamp,
                target_frame_id=frame_id,
                track_descriptors=simulation_descriptors,
                frame_width=frame_width,
                frame_height=frame_height,
                prediction_reason="simulated_missing_fields",
                merge_partial_values=True,
                actual_payload_pending=False,
            )
            predictions.extend(simulated_predictions)

            simulation_comparisons = (
                self.object_predictor.compare_received_observations(
                    camera_id=camera_id,
                    camera_location=camera_location,
                    frame_id=frame_id,
                    observations=simulation_ground_truth,
                )
            )
            comparisons.extend(simulation_comparisons)

        if comparisons:
            data["prediction_comparisons"] = comparisons
            self.print_prediction_comparisons(comparisons)

        # History receives only the sanitized numerical observations. An
        # ID-only descriptor has no bbox and is therefore not stored as fact.
        self.object_predictor.update_history(
            camera_id=camera_id,
            camera_location=camera_location,
            frame_id=frame_id,
            timestamp=timestamp,
            observations=minimal_observations,
        )

        # For complete objects, update history with frame N first and then ask
        # for frame N+1 using only track_id and optional class_name.
        if (
            PREDICTION_CONFIG["continuous_prediction"]
            and existing_full_observations
        ):
            next_frame = self._first_packet_value(
                metadata,
                "continuous_prediction_target_frame_id",
                default=int(frame_id) + 1,
            )
            next_frame_descriptors = [
                {
                    "track_id": observation["track_id"],
                    **(
                        {"class_name": observation["class_name"]}
                        if observation.get("class_name") is not None
                        else {}
                    ),
                }
                for observation in existing_full_observations
            ]
            predictions.extend(
                self.object_predictor.predict_tracks(
                    camera_id=camera_id,
                    camera_location=camera_location,
                    reference_frame_id=frame_id,
                    reference_timestamp=timestamp,
                    target_frame_id=next_frame,
                    track_descriptors=next_frame_descriptors,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    prediction_reason="continuous_next_frame",
                    merge_partial_values=False,
                    actual_payload_pending=False,
                )
            )

        data["predicted_existing_objects"] = predictions

        if not predictions:
            if prediction_requested:
                print(
                    "[PREDICTION] Prediction was requested, but no eligible "
                    "existing tracks were found in server history"
                )
            return []

        print("\n" + "=" * 80)
        print(
            f"[SERVER PREDICTION] frame_id={frame_id} "
            f"camera={camera_id} location={camera_location} "
            f"predicted_existing_objects={len(predictions)}"
        )
        for index, obj in enumerate(predictions):
            existing = obj.get("existing_object_used_for_prediction") or {}

            print(
                f"  received existing object {index}: "
                f"class={existing.get('class_name')} "
                f"track_id={existing.get('track_id')} "
                f"source_frame={existing.get('frame_id')} "
                f"bbox={existing.get('bbox')} "
                f"distance_m={existing.get('distance_m')} "
                f"bearing_deg={existing.get('bearing_deg')} "
                f"direction={existing.get('direction')} "
                f"speed_kmh={existing.get('speed_kmh')} "
                f"confidence={existing.get('confidence')}"
            )
            print(
                f"  predicted object {index}: "
                f"class={obj.get('class_name')} "
                f"predicted_track_id={obj.get('track_id')} "
                f"track_status={obj.get('track_status', 'existing')} "
                f"target_frame={obj.get('predicted_for_frame_id')} "
                f"bbox={obj.get('bbox')} "
                f"distance_m={obj.get('distance_m')} "
                f"bearing_deg={obj.get('bearing_deg')} "
                f"direction={obj.get('direction')} "
                f"speed_kmh={obj.get('speed_kmh')} "
                f"reason={obj.get('prediction_reason')} "
                f"predicted_fields={obj.get('predicted_fields')} "
                f"held_out_fields="
                f"{obj.get('simulated_missing_fields', [])} "
                f"source={obj.get('prediction_source')} "
                f"confidence={obj.get('prediction_confidence')} "
                f"inference_input_fields="
                f"{obj.get('inference_input_fields')}"
            )
        print("=" * 80)

        if PREDICTION_CONFIG["save_predictions"]:
            self.save_predicted_objects(data, predictions)
        return predictions

    @staticmethod
    def print_prediction_comparisons(comparisons: List[Dict]):
        """Print predicted, received, and error values side by side."""
        print("\n" + "=" * 100)
        print("[PREDICTED VS RECEIVED OBJECT PROPERTIES]")
        for comparison in comparisons:
            predicted = comparison["predicted"]
            received = comparison["received"]
            error = comparison["error"]
            print(
                f"  frame_id={comparison['frame_id']} "
                f"camera={comparison['camera_id']} "
                f"location={comparison['camera_location']} "
                f"track_id={comparison['track_id']} "
                f"source={comparison['prediction_source']} "
                f"reason={comparison.get('prediction_reason')} "
                f"held_out_fields="
                f"{comparison.get('simulated_missing_fields', [])} "
                f"prediction_confidence="
                f"{comparison['prediction_confidence']}"
            )

            print(
                f"    class:      predicted={predicted['class_name']} "
                f"received={received['class_name']} "
                f"match={error['class_match']}"
            )
            confidence_error = error["confidence"]
            confidence_error_text = (
                "not_available"
                if confidence_error is None
                else (
                    f"signed_error={confidence_error['signed_error']} "
                    f"absolute_error={confidence_error['absolute_error']}"
                )
            )
            print(
                f"    confidence: predicted={predicted['confidence']} "
                f"received={received['confidence']} "
                f"{confidence_error_text}"
            )

            bbox_error = error["bbox"]
            bbox_error_text = (
                "not_available"
                if bbox_error is None
                else (
                    f"coordinate_error_px="
                    f"{bbox_error['coordinate_error_px']} "
                    f"mae_px={bbox_error['mean_absolute_error_px']} "
                    f"center_error_px={bbox_error['center_error_px']} "
                    f"iou={bbox_error['iou']}"
                )
            )
            print(
                f"    bbox:       predicted={predicted['bbox']} "
                f"received={received['bbox']} {bbox_error_text}"
            )

            for field, unit in (
                ("distance_m", "m"),
                ("bearing_deg", "deg"),
                ("speed_kmh", "km/h"),
            ):
                field_error = error[field]
                error_text = (
                    "not_available"
                    if field_error is None
                    else (
                        f"signed_error={field_error['signed_error']} {unit} "
                        f"absolute_error={field_error['absolute_error']} {unit}"
                    )
                )
                print(
                    f"    {field + ':':11} "
                    f"predicted={predicted[field]} "
                    f"received={received[field]} {error_text}"
                )

            print(
                f"    direction:  predicted={predicted['direction']} "
                f"received={received['direction']} "
                f"match={error['direction_match']}"
            )
        print("=" * 100)

    def save_predicted_objects(self, data: Dict, predictions: List[Dict]):
        metadata = data.get("metadata", {})
        camera_id = str(metadata.get("camera_id", "unknown"))
        camera_location = str(metadata.get("camera_location", "unknown"))
        frame_id = metadata.get(
            "prediction_target_frame_id",
            metadata.get("frame_id", "unknown"),
        )
        safe_camera = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in f"{camera_location}_{camera_id}"
        )
        prediction_output_dir = PREDICTION_CONFIG["prediction_output_dir"]
        os.makedirs(prediction_output_dir, exist_ok=True)
        filename = os.path.join(
            prediction_output_dir,
            f"{safe_camera}_frame_{frame_id}_predicted.json",
        )
        output = {
            "type": "predicted_existing_objects",
            "metadata": {
                "camera_id": camera_id,
                "camera_location": camera_location,
                "frame_id": frame_id,
                "generated_at": datetime.now().isoformat(),
                "actual_payload_pending": True,
            },
            "objects": predictions,
            "comparisons": data.get("prediction_comparisons", []),
        }
        try:
            with open(filename, "w") as prediction_file:
                json.dump(output, prediction_file, indent=2, default=str)
            print(f"[PREDICTION SAVED] {filename}")
        except Exception as e:
            print(f"Could not save object predictions: {e}")
    # =============== END PREDICTION MODIFICATION 3 ===============


    def print_statistics(self):
        runtime = time.time() - self.start_time

        if runtime > 0:
            fps = self.frames_processed / runtime
        else:
            fps = 0.0

        print("\n" + "-" * 80)
        print(f"STATISTICS (Runtime: {runtime:.1f}s):")
        print(f"  Frames processed: {self.frames_processed} ({fps:.2f} FPS)")
        print(f"  Persons detected: {self.persons_detected}")
        print(f"  New persons: {self.new_persons}")
        print(f"  Existing persons: {self.existing_persons}")
        print("-" * 80)

        
    def update_statistics(self, reid_results: List[Dict]):
        """
        Update runtime statistics after processing one detected_objects packet.
        """

        self.frames_processed += 1

        persons_in_frame = len(reid_results)
        self.persons_detected += persons_in_frame

        new_in_frame = sum(1 for r in reid_results if r.get("is_new_person", False))
        existing_in_frame = persons_in_frame - new_in_frame

        self.new_persons += new_in_frame
        self.existing_persons += existing_in_frame

        if self.frames_processed % self.stats_interval == 0:
            self.print_statistics()







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
    
    def print_person_history(self, person_id: str, limit: int = 50):
        history = self.weaviate_manager.get_person_history(person_id, limit=limit)

        if not history:
            print(f"No history found for {person_id}")
            return
        
        history = sorted(
            history,
            key=lambda x: str(x.get("timestamp", "")),
            reverse=True,
        )


        from collections import Counter
        camera_counts = Counter(item.get("camera_id") for item in history)

        print(f"\nHistory for {person_id}:")
        print("-" * 80)

        print("Camera summary:")
        for cam, count in camera_counts.items():
            print(f"  {cam}: {count} records")

        print("-" * 80)

        for item in history:
            print(
                f"person_id={item.get('person_id')} "
                f"camera={item.get('camera_id')} "
                f"location={item.get('camera_location')} "
                f"frame={item.get('frame_id')} "
                f"time={item.get('timestamp')} "
            )

        print("-" * 80)


    def process_detection(self, data: Dict) -> List[Dict]:

        
        """
        
        
        Complete detection processing pipeline
        
        
        """
        try:
            data = self.normalize_incoming_data(data)

            frame_id = data["metadata"].get("frame_id", "unknown")
            camera_id = data["metadata"].get("camera_id", "unknown")
            camera_location = data["metadata"].get("camera_location", "unknown")
            print(f"Processing frame {frame_id} from camera {camera_id} at location {camera_location}")
            valid_objects = [
                obj for obj in data.get("objects", [])
                if "processed_image" in obj and obj.get("class_name") == "person"
            ]

            if not valid_objects:
                print(
                    "[REID SKIP] No person objects with processed_image; "
                    "all received objects were printed above"
                )
                return []

            data["objects"] = valid_objects
            
            print(f"the number of valid objects: {len(valid_objects)}")

            # processed_images_count = 0
            # for i, obj in enumerate(data.get("objects", [])):
            #     if "processed_image" in obj:
            #         processed_shape = np.array(obj["processed_image"]).shape
            #         print(f"📷 Object {i}: raw image shape={processed_shape}")
            #         processed_images_count += 1

            # if processed_images_count == 0:
            #     print("No raw image data found in any objects!")
            # else:
            #     print(f"Found {processed_images_count} objects with raw image data")

            # 1. Load trained ReID Model (TransReID, Resnet50, etc..)
            # We already have the model loaded in the self.transreid_processor

            # 1. Convert objects to tensor (Step 5)
            raw_features = objects_to_tensor(data["objects"])
            print(f"Raw features shape: {raw_features.shape}")

            # 2. Extract proper ReID features using TransReID
            # (step 6)
            reid_features = self.transreid_processor.extract_features(raw_features)
            print(f"ReID features shape: {reid_features.shape}")

            # if reid_features.dim() > 2:
            #     print(f"Fixing invalid ReID feature shape: {reid_features.shape}")

            #     reid_features = reid_features.flatten(start_dim=1)

            #     if reid_features.shape[1] >= 384:
            #         reid_features = reid_features[:, :384]
            #     else:
            #         pad_size = 384 - reid_features.shape[1]
            #         reid_features = torch.nn.functional.pad(reid_features, (0, pad_size))

            #     print(f"Fixed ReID features shape: {reid_features.shape}")
            
            # FIX: make feature vector compatible with current Weaviate index dimension = 384
            if reid_features.dim() > 2:
                print(f"Flattening invalid ReID feature shape: {reid_features.shape}")
                reid_features = reid_features.flatten(start_dim=1)

            # if reid_features.shape[1] > 384:
            #     print(f"Truncating ReID feature from {reid_features.shape[1]} to 384")
            #     reid_features = reid_features[:, :384]

            # elif reid_features.shape[1] < 384:
            #     print(f"Padding ReID feature from {reid_features.shape[1]} to 384")
            #     pad_size = 384 - reid_features.shape[1]
            #     reid_features = torch.nn.functional.pad(reid_features, (0, pad_size))

            print(f"Fixed ReID features shape: {reid_features.shape}")




            # NOTE: We don't need to use Pose2ID's NFC since we can use weaviate's functionality for finding neighbors for current object

            # 3. Final normalization
            reid_features = torch.nn.functional.normalize(reid_features, dim=1, p=2)

            # 4. Process through Weaviate ReID system
            reid_results = self.weaviate_manager.process_and_identify(
                data, reid_features
            )

            # 5. Display results
            self.display_results(reid_results)

            # shows person history

            for result in reid_results:
                self.print_person_history(result["person_id"])
                

            # 6. Save results if enabled
            if self.args.save_results:
                self.save_simple_results(data, reid_results)

            return reid_results

        except Exception as e:
            print(f"Error processing detection: {e}")
            return []

    def display_results(self, results: List[Dict]):
        """Display ReID results"""
        print(f"\nReID Results ({len(results)} persons detected):")
        print("-" * 80)

        for result in results:
            status = "NEW" if result["is_new_person"] else "EXISTING"
            cross_cam = (
                f", {len(result.get('cross_camera_matches', []))} cross-camera"
                if result.get("cross_camera_matches")
                else ""
            )

            print(
                f"{status} | ID: {result['person_id']} | "
                f"Conf: {result['confidence']:.3f} | "
                f"Camera: {result['camera_id']} | "
                f"Similar: {result['similar_detections']}{cross_cam}"
            )

        print("-" * 80)

    def log_rx(self, data: Dict, status: str = "received"):
        metadata = data.get("metadata", {})

        payload_type = data.get("type") or metadata.get("payload_type", "unknown")
        frame_id = metadata.get("frame_id", "NA")
        camera_id = metadata.get("camera_id", "NA")
        camera_location = metadata.get("camera_location", "unknown")
        vlan_id = metadata.get("vlan_id", "NA")
        vlan_interface = metadata.get("vlan_interface", "NA")
        source_addr = data.get("_source_addr", "NA")
        num_objects = len(data.get("objects", [])) if "objects" in data else 0

        print(
            f"[RX] status={status} "
            f"type={payload_type} "
            f"frame_id={frame_id} "
            f"camera={camera_id} "
            f"location={camera_location} "
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
            camera_location = metadata.get("camera_location", "unknown")
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

            doc = {
                "frame_id": frame_id,
                "camera_id": camera_id,
                "camera_location": camera_location,
                "timestamp": metadata.get("timestamp"),
                "image_format": "jpg",
                "image_bytes": frame_bytes,
            }
            
            try:
                mongo_id = self.weaviate_manager.mongo_storage.store_raw_frame_doc(doc)
            except Exception as e:
                print(f"Error storing raw frame doc in MongoDB: {e}")
                mongo_id = None
    
    
            save_dir = "received_raw_frames"
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.join(
                save_dir,
                f"{camera_location}_{camera_id}_frame_{frame_id}.jpg",
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

    def run(self):

        print("Starting continuous processing...")

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
                    # ================= PREDICTION MODIFICATION 4 =================
                    # This prints the unmodified packet first, then predicts
                    # before process_detection() applies ReID-only filtering.
                    predictions = self.process_object_predictions(data)
                    # =============== END PREDICTION MODIFICATION 4 ===============

                    if not data.get("objects"):
                        status = "empty_object_packet"
                        if predictions:
                            status += f" predicted_existing={len(predictions)}"
                        self.log_rx(data, status=status)
                        continue

                    status = "object_packet_received"
                    if predictions:
                        status += f" predicted_existing={len(predictions)}"
                    self.log_rx(data, status=status)
                    
                    
                    results = self.process_detection(data)

                    # Update runtime statistics
                    self.update_statistics(results)


                    # Optional: Save results summary
                    if results and self.args.save_results:
                        self.save_results_summary(results)
                    continue
                
                self.log_rx(data, status="unknown_payload")


            except KeyboardInterrupt:
                print("\nStopping C2 processor...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

    def save_results_summary(self, results: List[Dict]):
        """Save processing results for analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reid_results_{timestamp}.json"

        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def save_simple_results(self, data: Dict, reid_results: List[Dict]):
        """Save results using simple file operations - one folder per person"""
        try:
            # Extract metadata
            frame_id = str(data["metadata"].get("frame_id", "unknown"))
            camera_id = str(data["metadata"].get("camera_id", "unknown"))

            # Process each person separately
            for result in reid_results:
                person_id = result.get("person_id", "unknown")
                detection_id = result.get("detection_id")

                print(f"\n🧑 Processing person {person_id} (detection {detection_id})")

                # Find the corresponding object in data
                person_obj = None
                for obj in data.get("objects", []):
                    if str(obj.get("object_id", "")) == str(detection_id):
                        person_obj = obj
                        break

                if person_obj is None:
                    print(f"⚠️ Could not find object data for detection {detection_id}")
                    continue

                # Get the person's original image
                original_image = None
                if "processed_image" in person_obj:
                    img_list = person_obj["processed_image"]

                    # Convert JSON list -> torch tensor
                    tensor = torch.tensor(img_list)  # shape: (3, H, W)

                    # Undo ImageNet normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                    unnormalized = tensor * std + mean
                    unnormalized = unnormalized.clamp(0, 1)

                    # Convert to PIL then numpy
                    to_pil = T.ToPILImage()
                    pil_image = to_pil(unnormalized)
                    processed = np.array(pil_image)

                    # Convert RGB to BGR for OpenCV
                    if len(processed.shape) == 3 and processed.shape[2] == 3:
                        original_image = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    else:
                        original_image = processed

                    print(f"📷 Extracted original image for person {person_id}")
                else:
                    print(f"⚠️ No processed_image found for person {person_id}")
                    continue

                # Get similar images for this person
                similar_results = []
                if result.get("similar_detections", 0) > 0:
                    try:
                        # Extract features and search for similar
                        obj_tensor = objects_to_tensor([person_obj])
                        obj_features = self.transreid_processor.extract_features(
                            obj_tensor
                        )
                        obj_features = torch.nn.functional.normalize(
                            obj_features, dim=1, p=2
                        )

                        # Search for similar embeddings
                        similar_matches = (
                            self.weaviate_manager.vector_store.search_similar(
                                obj_features, top_k=5, distance_threshold=0.8
                            )
                        )

                        # Add matches to results (exclude self)
                        for match in similar_matches:
                            match_object_id = match.get("object_id")
                            if match_object_id != str(detection_id):
                                similar_results.append(match)

                    except Exception as e:
                        print(f"Could not fetch similar images for {person_id}: {e}")

                # Debug: Check base64 data in similar results
                has_base64_images = 0
                for similar in similar_results:
                    if similar.get("image_crop_base64", "").strip():
                        has_base64_images += 1
                print(
                    f"🔍 Found {has_base64_images}/{len(similar_results)} similar results with base64 image data"
                )

                # Save individual person results
                saver = ReIDResultsSaver("results")
                image_name = f"person_{person_id}_frame_{frame_id}_cam_{camera_id}"

                results_folder = saver.save_complete_results(
                    image_name=image_name,
                    original_image=original_image,
                    reid_results=[result],  # Only this person's result
                    similar_results=similar_results,
                )
                print(f"Saved person {person_id} results to: {results_folder}")

        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def close(self):
        receiver = getattr(self, "receiver", None)
        if receiver is not None:
            try:
                receiver.close()
            except OSError:
                pass
            self.receiver = None

        weaviate_manager = getattr(self, "weaviate_manager", None)
        vector_store = getattr(weaviate_manager, "vector_store", None)
        client = getattr(vector_store, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception as error:
                print(f"Warning: could not close Weaviate client: {error}")

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
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.98,
        help="Similarity threshold for person matching",
    )
    
    parser.add_argument(
        "--save_results", action="store_true", help="Save processing results to file"
    )
    parser.add_argument(
        "--save_json", action="store_true", help="Save reid_results JSON files"
    )
    parser.add_argument(
        "--store_crops",
        action="store_true",
        help="Store image crops in Weaviate as base64",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug output"
    )

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