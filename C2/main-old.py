# import argparse
# import json
# import os
# import time
# from datetime import datetime
# from typing import Dict, List

# import numpy as np
# import redis
# import torch
# from torchvision import transforms as T
# from utils.reid_result import ReIDResult  # Just a data class
# from utils.results_saver import ReIDResultsSaver
# from utils.storage import SecureReIDStorage

# # TransReID imports will be handled within the TransReIDProcessor class
# from utils.util import objects_to_tensor
# from utils.weaviate import ReIDVectorStore


# class TransReIDProcessor:
#     """Proper TransReID model for feature extraction"""

#     def __init__(
#         self,
#         model_path="/app/TransReID/weights/vit_transreid_market.pth",
#         config_path="/app/TransReID/configs/Market/vit_transreid_stride.yml",
#     ):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"🎯 Device selected: {self.device}")
#         if torch.cuda.is_available():
#             print(f"GPU: {torch.cuda.get_device_name(0)}")
#         self.load_model(model_path, config_path)

#     def load_model(self, model_path=None, config_file=None):
#         """Load TransReID model with multiple fallback strategies"""

#         # Set default model path
#         if model_path is None:
#             model_path = "/app/TransReID/weights/vit_transreid_market.pth"

#         try:
#             print("Loading TransReID model...")
#             # Import TransReID config
#             from TransReID.config import cfg

#             # Configure the model
#             cfg.MODEL.NAME = "transformer"
#             cfg.MODEL.TRANSFORMER_TYPE = "vit_base_patch16_224_TransReID"
#             cfg.MODEL.SIE_CAMERA = False
#             cfg.MODEL.SIE_VIEW = False
#             cfg.MODEL.JPM = True
#             cfg.MODEL.PRETRAIN_CHOICE = "self"
#             cfg.MODEL.PRETRAIN_PATH = (
#                 "/app/TransReID/weights/jx_vit_base_p16_224-80ecf9dd.pth"
#             )
#             cfg.TEST.WEIGHT = model_path
#             cfg.INPUT.SIZE_TEST = [256, 128]
#             cfg.INPUT.SIZE_TRAIN = [256, 128]
#             cfg.MODEL.DEVICE = self.device
#             cfg.freeze()

#             # Import and create model
#             from TransReID.model import make_model

#             # Create model without SIE (no camera/view settings needed)
#             self.model = make_model(cfg, num_class=751, camera_num=0, view_num=0)

#             # Try to load weights
#             if os.path.exists(model_path):
#                 try:
#                     self.model.load_param(model_path)
#                     print(f"✅ Successfully loaded weights from {model_path}")
#                 except Exception as e:
#                     print(f"⚠️ Standard loading failed, trying partial loading...")
#                     self._load_weights_partially(model_path)
#                     print("✅ Partially loaded weights")
#             else:
#                 print(f"❌ Warning: Weights not found at {model_path}")

#             self.model.to(self.device)
#             self.model.eval()
#             print(f"✅ TransReID model loaded successfully on {self.device}")

#         except Exception as e:
#             print(f"❌ Error loading TransReID model: {e}")
#             self.model = None

#     def _load_weights_partially(self, model_path):
#         """Load weights partially, skipping incompatible layers"""
#         import torch

#         try:
#             param_dict = torch.load(model_path, map_location="cpu")
#             model_dict = self.model.state_dict()

#             # Filter out incompatible parameters
#             compatible_dict = {}
#             incompatible_keys = []

#             for key in param_dict:
#                 clean_key = key.replace("module.", "")
#                 if clean_key in model_dict:
#                     if param_dict[key].shape == model_dict[clean_key].shape:
#                         compatible_dict[clean_key] = param_dict[key]
#                     else:
#                         incompatible_keys.append(f"{clean_key} (shape mismatch)")
#                 else:
#                     incompatible_keys.append(f"{clean_key} (missing in model)")

#             # Load compatible parameters
#             model_dict.update(compatible_dict)
#             self.model.load_state_dict(model_dict, strict=False)

#         except Exception as e:
#             raise e

#     def extract_features(self, person_images):
#         """Extract ReID features from person images"""

#         if self.model is None:
#             # Fallback: just normalize input features
#             print("⚠️ TransReID model not loaded, using fallback normalization")
#             return torch.nn.functional.normalize(person_images, dim=1, p=2)

#         try:
#             """Context-manager that disables gradient calculation.
#             Disabling gradient calculation is useful for inference, when you are sure
#             that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.
#             """
#             with torch.no_grad():
#                 person_images = person_images.to(self.device)

#                 # Add input validation
#                 if person_images.dim() != 4:
#                     raise ValueError(f"Expected 4D tensor, got {person_images.dim()}D")
#                 if person_images.size(1) != 3:
#                     raise ValueError(
#                         f"Expected 3 channels, got {person_images.size(1)}"
#                     )
#                 if person_images.size(2) != 256 or person_images.size(3) != 128:
#                     raise ValueError(
#                         f"Expected 256x128 images, got {person_images.size(2)}x{person_images.size(3)}"
#                     )

#                 # Calling the model - Step 7
#                 features = self.model(person_images)
#                 return features.cpu()
#         except Exception as e:
#             return torch.nn.functional.normalize(person_images, dim=1, p=2)


# class WeaviateReIDManager:
#     """Enhanced Weaviate manager for complete ReID operations"""

#     def __init__(
#         self,
#         weaviate_url: str = "http://localhost:8080",
#         collection_name: str = "reid_collection",
#         similarity_threshold: float = 0.7,
#         max_gallery_size: int = 10000,
#         store_crops: bool = False,
#     ):
#         self.vector_store = ReIDVectorStore(weaviate_url, collection_name)
#         self.similarity_threshold = similarity_threshold
#         self.max_gallery_size = max_gallery_size
#         self.person_id_counter = 0
#         self.store_crops = store_crops

#         # Initialize MongoDB Storage - Encrypted
#         self.mongo_storage = SecureReIDStorage()
#         print("🔒 MongoDB storage initialized for gallery data")

#     def process_and_identify(
#         self, objects_data: Dict, reid_features: torch.Tensor
#     ) -> List[Dict]:
#         """
#         Core ReID functionality:
#         1. Store new features in Weaviate
#         2. Find similar existing persons
#         3. Assign person IDs
#         4. Update person profiles
#         """
#         results = []

#         for i, obj in enumerate(objects_data["objects"]):
#             person_feature = reid_features[i : i + 1]  # Shape: [1, feature_dim]

#             # 1. Search for similar persons in database
#             similar_persons = self.find_similar_persons(
#                 person_feature,
#                 obj.get("class_name", "person"),
#                 objects_data["metadata"]["camera_id"],
#             )

#             # 2. Determine person identity
#             person_identity = self.determine_identity(
#                 similar_persons, person_feature, obj
#             )

#             # 3. Store/Update in Weaviate
#             store_crops = getattr(self, "store_crops", False)
#             if store_crops:
#                 pass
#             storage_result = self.store_person_data(
#                 objects_data, obj, person_feature, person_identity, i, store_crops
#             )

#             # 4. Compile result
#             result = {
#                 "detection_id": obj.get("object_id", i),
#                 "person_id": person_identity["person_id"],
#                 "confidence": person_identity["confidence"],
#                 "is_new_person": person_identity["is_new"],
#                 "similar_detections": len(similar_persons),
#                 "bbox": obj.get("bbox", [0, 0, 0, 0]),
#                 "camera_id": objects_data["metadata"]["camera_id"],
#                 "frame_id": objects_data["metadata"]["frame_id"],
#                 "timestamp": objects_data["metadata"]["timestamp"],
#                 "cross_camera_matches": person_identity.get("cross_camera_matches", []),
#                 "weaviate_id": storage_result[0]
#                 if storage_result and len(storage_result) > 0
#                 else None,
#             }

#             results.append(result)

#         return results

#     def find_similar_persons(
#         self, query_feature: torch.Tensor, class_filter: str, current_camera: str
#     ) -> List[Dict]:
#         """Find similar persons in the vector database"""
#         try:
#             # Search in Weaviate for similar embeddings
#             similar_results = self.vector_store.search_similar(
#                 query_feature,
#                 top_k=20,  # Get more results for better matching
#                 class_filter=class_filter,
#                 confidence_threshold=0.3,  # Lower threshold for initial search
#                 distance_threshold=2.0,  # Euclidean distance threshold
#             )

#             # Filter by similarity threshold and add distance scores
#             filtered_results = []
#             for result in similar_results:
#                 # Calculate actual similarity score (you might need to implement this)
#                 similarity_score = self.calculate_similarity(query_feature, result)

#                 if similarity_score >= self.similarity_threshold:
#                     result["similarity_score"] = similarity_score
#                     result["is_cross_camera"] = result["camera_id"] != current_camera
#                     filtered_results.append(result)

#             return filtered_results

#         except Exception as e:
#             print(f"❌ Error in similarity search: {e}")
#             return []

#     def determine_identity(
#         self, similar_persons: List[Dict], person_feature: torch.Tensor, obj: Dict
#     ) -> Dict:
#         """Determine if this is a new person or matches existing identity"""

#         if not similar_persons:
#             # New person - assign new ID
#             self.person_id_counter += 1
#             return {
#                 "person_id": f"person_{self.person_id_counter:06d}",
#                 "confidence": 1.0,
#                 "is_new": True,
#                 "matched_detection": None,
#             }

#         # Find best match
#         best_match = max(similar_persons, key=lambda x: x.get("similarity_score", 0))

#         if best_match["similarity_score"] >= self.similarity_threshold:
#             # Existing person identified
#             cross_camera_matches = [
#                 p
#                 for p in similar_persons
#                 if p.get("is_cross_camera", False)
#                 and p.get("similarity_score", 0) >= self.similarity_threshold
#             ]

#             return {
#                 "person_id": best_match.get(
#                     "person_id", f"unknown_{best_match.get('object_id')}"
#                 ),
#                 "confidence": best_match["similarity_score"],
#                 "is_new": False,
#                 "matched_detection": best_match,
#                 "cross_camera_matches": cross_camera_matches[
#                     :5
#                 ],  # Top 5 cross-camera matches
#             }
#         else:
#             # No confident match - new person
#             print(
#                 f"🆕 New person detected! Assigning ID: person_{self.person_id_counter + 1:06d}"
#             )
#             self.person_id_counter += 1
#             return {
#                 "person_id": f"person_{self.person_id_counter:06d}",
#                 "confidence": 0.6,  # Lower confidence for borderline cases
#                 "is_new": True,
#                 "matched_detection": None,
#             }

#     def store_person_data(
#         self,
#         objects_data: Dict,
#         obj: Dict,
#         person_feature: torch.Tensor,
#         person_identity: Dict,
#         index: int,
#         store_crops: bool = False,
#     ):
#         """Store person data with assigned identity in Weaviate"""
#         try:
#             # Use processed_image - it's a normalized tensor that needs unnormalization
#             person_crop = None
#             if "processed_image" in obj:
#                 img_list = obj["processed_image"]

#                 # Convert JSON list -> torch tensor
#                 tensor = torch.tensor(img_list)  # shape: (3, H, W)

#                 # Undo ImageNet normalization
#                 mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#                 std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#                 unnormalized = tensor * std + mean
#                 unnormalized = unnormalized.clamp(0, 1)

#                 # Convert to PIL image
#                 to_pil = T.ToPILImage()
#                 pil_image = to_pil(unnormalized)

#                 # Convert PIL to numpy array for base64 encoding
#                 person_crop_array = np.array(pil_image)

#                 # convert to bytes (PNG format for storage)
#                 import io

#                 img_buffer = io.BytesIO()
#                 pil_image.save(img_buffer, format="PNG")
#                 person_crop_bytes = img_buffer.getvalue()

#             mongo_result = ReIDResult(
#                 object_id=str(obj.get("object_id", f"obj_{index}")),
#                 class_name=obj.get("class_name", "unknown"),
#                 confidence=float(obj.get("confidence", 0.0)),
#                 bbox=obj.get("bbox", [0, 0, 0, 0]),
#                 camera_id=str(objects_data["metadata"].get("camera_id", "unknown")),
#                 frame_id=int(objects_data["metadata"].get("frame_id", 0)),
#                 timestamp=str(objects_data["metadata"].get("timestamp", 0)),
#                 embedding_method="TransReID",
#                 reid_confidence=person_identity["confidence"],
#                 person_id=person_identity["person_id"],
#                 is_new_person=person_identity["is_new"],
#                 image=person_crop_bytes
#                 if person_crop_bytes
#                 else b"",  # Raw image bytes
#             )

#             # STORE data in mongo (encrypted)
#             embedding_vector = person_feature.cpu().numpy()
#             self.mongo_storage.store_reid_result(embedding_vector, mongo_result)

#             # Process detection to include image crop if available and enabled
#             # Store the person crop directly as base64 if available
#             # enhanced_obj = obj.copy()
#             # if store_crops and person_crop is not None:
#             #     try:
#             #         base64_crop = encode_image_to_base64(person_crop)
#             #         if base64_crop:
#             #             enhanced_obj["image_crop_base64"] = base64_crop
#             #             print(f" Encoded person crop to base64 for object {index}")
#             #         else:
#             #             print(f"⚠️ Failed to encode person crop for object {index}")
#             #     except Exception as e:
#             #         print(f" Error encoding person crop for object {index}: {e}")
#             # elif store_crops:
#             #     print(f"⚠️ No processed_image found for object {index}")

#             # Prepare data object with person identity to store in weaviate - minimal information
#             # STORE in Weaviate (minimal data for similarity search)
#             objects_data_with_person_id = {
#                 "metadata": objects_data["metadata"],
#                 "objects": [
#                     {
#                         "person_id": person_identity["person_id"],
#                         "reid_confidence": person_identity["confidence"],
#                         "is_new_person": person_identity["is_new"],
#                     }
#                 ],
#             }

#             # Store in Weaviate with image crops if available
#             result = self.vector_store.store_embeddings(
#                 objects_data_with_person_id, person_feature
#             )

#             return result if result else None

#         except Exception as e:
#             print(f"❌ Error storing person data: {e}")
#             return None

#     def calculate_similarity(self, query_feature: torch.Tensor, result: Dict) -> float:
#         """Calculate similarity score between query and result"""
#         # This is a placeholder - you might need to implement proper similarity calculation
#         # based on the distance returned by Weaviate
#         try:
#             # Weaviate returns distance, convert to similarity (0-1 scale)
#             distance = result.get("_additional", {}).get("distance", 1.0)
#             similarity = max(
#                 0, 1.0 - (distance / 2.0)
#             )  # Normalize distance to similarity
#             return similarity
#         except:
#             return 0.5  # Default similarity

#     def get_person_history(self, person_id: str, limit: int = 10) -> List[Dict]:
#         """Get detection history for a specific person"""
#         try:
#             # Query Weaviate for all detections of this person
#             query_builder = (
#                 self.vector_store.client.query.get(
#                     self.vector_store.collection_name,
#                     [
#                         "object_id",
#                         "camera_id",
#                         "frame_id",
#                         "timestamp",
#                         "bbox",
#                         "confidence",
#                     ],
#                 )
#                 .with_where(
#                     {
#                         "path": ["person_id"],
#                         "operator": "Equal",
#                         "valueString": person_id,
#                     }
#                 )
#                 .with_limit(limit)
#                 .with_sort([{"path": ["timestamp"], "order": "desc"}])
#             )

#             results = query_builder.do()
#             return results["data"]["Get"][self.vector_store.collection_name]

#         except Exception as e:
#             print(f"❌ Error getting person history: {e}")
#             return []


# class RedisConnection:
#     def __init__(
#         self, redis_host="localhost", redis_port=6379, queue_name="object_queue"
#     ):
#         self.redis_host = redis_host
#         self.redis_port = redis_port
#         self.queue_name = queue_name
#         self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port)

#     def get_data(self):
#         data = self.redis_client.lpop(self.queue_name)
#         if data:
#             return json.loads(data)
#         else:
#             return None

#     def connection_active(self):
#         return self.redis_client.ping()

#     def queue_size(self):
#         return self.redis_client.llen(self.queue_name)


# class C2Processor:
#     """Main C2 processor with complete Pose2ID and Weaviate integration"""

#     def __init__(self, args):
#         self.args = args

#         # Initialize components
#         self.redis_conn = RedisConnection(
#             args.redis_host, args.redis_port, args.queue_name
#         )
#         self.transreid_processor = TransReIDProcessor()
#         self.weaviate_manager = WeaviateReIDManager(
#             args.weaviate_url,
#             "PersonReID",
#             similarity_threshold=args.similarity_threshold,
#             store_crops=getattr(args, "store_crops", False),
#         )

#         # Initialize results saver if saving is enabled
#         self.results_saver = None
#         if args.save_results:
#             self.results_saver = ReIDResultsSaver(self.weaviate_manager.mongo_storage)
#             print("📁 Results saver enabled - will save to 'results' folder")

#         print("🚀 Container running with Weaviate ReID initialized")

#     def process_detection(self, data: Dict) -> List[Dict]:
#         # You get the data from the Redis queue
#         """Complete detection processing pipeline"""
#         try:
#             frame_id = data["metadata"].get("frame_id", "unknown")
#             camera_id = data["metadata"].get("camera_id", "unknown")
#             print(f"🎬 Processing frame {frame_id} from camera {camera_id}")

#             # 1. Load trained ReID Model (TransReID, Resnet50, etc..)
#             # We already have the model loaded in the self.transreid_processor

#             # 1. Convert objects to tensor (Step 5)
#             raw_features = objects_to_tensor(data["objects"])

#             # 2. Extract proper ReID features using TransReID
#             # (step 6)
#             reid_features = self.transreid_processor.extract_features(raw_features)
#             print(f"🔍 Extracted features for {len(data.get('objects', []))} objects")

#             # NOTE: We don't need to use Pose2ID's NFC since we can use weaviate's functionality for finding neighbors for current object

#             # 3. Final normalization
#             reid_features = torch.nn.functional.normalize(reid_features, dim=1, p=2)

#             # 4. Process through Weaviate ReID system
#             reid_results = self.weaviate_manager.process_and_identify(
#                 data, reid_features
#             )

#             # 5. Display results
#             self.display_results(reid_results)

#             # 6. Save results if enabled
#             if self.args.save_results:
#                 self.save_simple_results(data, reid_results)

#             return reid_results

#         except Exception as e:
#             print(f"❌ Error processing detection: {e}")
#             return []

#     def display_results(self, results: List[Dict]):
#         """Display ReID results"""
#         pass

#     def run(self):
#         """Main processing loop"""
#         if not self.redis_conn.connection_active():
#             print("❌ Redis connection failed. Make sure Redis is running.")
#             return

#         print(f"✅ Redis connected. Queue size: {self.redis_conn.queue_size()}")
#         print("🔄 Starting continuous processing...")

#         while True:
#             try:
#                 data = self.redis_conn.get_data()

#                 if data:
#                     # print(f"data from Queue: {data}")
#                     results = self.process_detection(data)

#                     # Optional: Save results summary
#                     if results and self.args.save_results and self.args.save_json:
#                         self.save_results_summary(results)
#                 else:
#                     print("Queue empty")
#                     time.sleep(3)
#                     continue

#             except KeyboardInterrupt:
#                 print("\nStopping C2 processor...")
#                 break
#             except Exception as e:
#                 print(f"Unexpected error: {e}")
#                 continue

#     def save_results_summary(self, results: List[Dict]):
#         """Save processing results to JSON file for analysis (controlled by --save_json flag)"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"reid_results_{timestamp}.json"

#         try:
#             with open(filename, "w") as f:
#                 json.dump(results, f, indent=2, default=str)
#             print(f"Results saved to {filename}")
#         except Exception as e:
#             print(f"❌ Error saving results: {e}")

#     def save_simple_results(self, data: Dict, reid_results: List[Dict]):
#         """Save results using MongoDB-based results saver with decryption"""
#         try:
#             # Extract metadata
#             frame_id = str(data["metadata"].get("frame_id", "unknown"))
#             camera_id = str(data["metadata"].get("camera_id", "unknown"))

#             # Use the pre-initialized results saver
#             if not self.results_saver:
#                 print(
#                     "⚠️ Results saver not initialized. Enable save_results in arguments."
#                 )
#                 return
#             saver = self.results_saver

#             print(f"💾 Saving results for {len(reid_results)} detected persons...")

#             # Process each person separately
#             for result in reid_results:
#                 person_id = result.get("person_id", "unknown")
#                 detection_id = result.get("detection_id")

#                 # Find the corresponding object in data
#                 person_obj = None
#                 for obj in data.get("objects", []):
#                     if str(obj.get("object_id", "")) == str(detection_id):
#                         person_obj = obj
#                         break

#                 if person_obj is None:
#                     continue

#                 # Extract features for this person (QUERY EMBEDDING)
#                 try:
#                     obj_tensor = objects_to_tensor([person_obj])
#                     obj_features = self.transreid_processor.extract_features(obj_tensor)
#                     obj_features = torch.nn.functional.normalize(
#                         obj_features, dim=1, p=2
#                     )
#                     query_embedding = obj_features.cpu().numpy().flatten()

#                     # Extract the current query image from processed_image
#                     query_image_data = None
#                     if "processed_image" in person_obj:
#                         img_list = person_obj["processed_image"]

#                         # Convert JSON list -> torch tensor
#                         tensor = torch.tensor(img_list)  # shape: (3, H, W)

#                         # Undo ImageNet normalization
#                         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#                         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#                         unnormalized = tensor * std + mean
#                         unnormalized = unnormalized.clamp(0, 1)

#                         # Convert to PIL image
#                         to_pil = T.ToPILImage()
#                         pil_image = to_pil(unnormalized)

#                         # Convert PIL to bytes (PNG format for storage)
#                         import io

#                         img_buffer = io.BytesIO()
#                         pil_image.save(img_buffer, format="PNG")
#                         query_image_data = img_buffer.getvalue()

#                 except Exception as e:
#                     print(f"⚠️ Could not extract features for person {person_id}: {e}")
#                     continue

#                 # Get similar embeddings from Weaviate (NEIGHBOR VECTORS)
#                 similar_embeddings = []
#                 if result.get("similar_detections", 0) > 0:
#                     try:
#                         # Search for similar embeddings in Weaviate
#                         similar_matches = (
#                             self.weaviate_manager.vector_store.search_similar(
#                                 obj_features, top_k=5, distance_threshold=0.8
#                             )
#                         )

#                         # Extract embedding vectors from search results
#                         for i, match in enumerate(similar_matches):
#                             if isinstance(match, dict):
#                                 if "vector" in match:
#                                     raw_vector = match["vector"]

#                                     # Handle different vector formats
#                                     vector_array = None
#                                     if isinstance(raw_vector, dict):
#                                         # Try common keys for vector data
#                                         if "values" in raw_vector:
#                                             vector_array = np.array(
#                                                 raw_vector["values"], dtype=np.float32
#                                             )
#                                         elif "data" in raw_vector:
#                                             vector_array = np.array(
#                                                 raw_vector["data"], dtype=np.float32
#                                             )
#                                         elif "vector" in raw_vector:
#                                             vector_array = np.array(
#                                                 raw_vector["vector"], dtype=np.float32
#                                             )
#                                         elif "default" in raw_vector:
#                                             vector_array = np.array(
#                                                 raw_vector["default"], dtype=np.float32
#                                             )
#                                     elif isinstance(raw_vector, (list, np.ndarray)):
#                                         vector_array = np.array(
#                                             raw_vector, dtype=np.float32
#                                         )

#                                     if (
#                                         vector_array is not None
#                                         and vector_array.size > 0
#                                     ):
#                                         similar_embeddings.append(vector_array)

#                     except Exception as e:
#                         print(f"⚠️ Error getting similar embeddings from Weaviate: {e}")

#                 # Save query results using actual embedding vectors
#                 query_name = f"person_{person_id}_frame_{frame_id}_cam_{camera_id}"

#                 try:
#                     results_folder = saver.save_query_results_with_image(
#                         query_embedding=query_embedding,
#                         similar_embeddings=similar_embeddings,
#                         query_name=query_name,
#                         camera_id=camera_id,
#                         frame_id=frame_id,
#                         query_image_data=query_image_data,
#                     )

#                 except Exception as e:
#                     print(f"❌ Error saving results for person {person_id}: {e}")

#         except Exception as e:
#             print(f"❌ Error in save_simple_results: {e}")


# def parse_args():
#     parser = argparse.ArgumentParser(description="Enhanced C2 ReID with Weaviate")
#     parser.add_argument("--redis_host", type=str, default="redis")
#     parser.add_argument("--redis_port", type=str, default="6379")
#     parser.add_argument("--queue_name", type=str, default="object_queue")
#     parser.add_argument("--weaviate_url", type=str, default="http://weaviate_db:8080")
#     parser.add_argument("--use_nfc", action="store_true", help="Apply NFC processing")
#     parser.add_argument(
#         "--similarity_threshold",
#         type=float,
#         default=0.7,
#         help="Similarity threshold for person matching",
#     )
#     parser.add_argument(
#         "--save_results", action="store_true", help="Save processing results to file"
#     )
#     parser.add_argument(
#         "--save_json", action="store_true", help="Save reid_results JSON files"
#     )
#     parser.add_argument(
#         "--store_crops",
#         action="store_true",
#         help="Store image crops in Weaviate as base64",
#     )
#     parser.add_argument(
#         "--verbose", action="store_true", help="Enable verbose debug output"
#     )

#     return parser.parse_args()


# def main():
#     args = parse_args()
#     processor = C2Processor(args)
#     processor.run()


# if __name__ == "__main__":
#     main()


