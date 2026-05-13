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

        # Initialize components
        self.receiver = TSNReceiver(args.listen_ip, args.port)
        
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

        print("TSN receiver with Weaviate ReID initialized")


        # Runtime statistics... ... ... 

        self.start_time = time.time()
        self.frames_processed = 0
        self.persons_detected = 0
        self.new_persons = 0
        self.existing_persons = 0
        self.stats_interval = 15


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
    
    def print_person_history(self, person_id: str):
        history = self.weaviate_manager.get_person_history(person_id, limit=10)

        if not history:
            print(f"No history found for {person_id}")
            return

        print(f"\nHistory for {person_id}:")
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
                print("No valid person objects with processed_image found")
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
                    if not data.get("objects"):
                        
                        self.log_rx(data, status="empty_object_packet")
                        continue

                    self.log_rx(data, status="object_packet_received")
                    
                    
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
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.95,
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
