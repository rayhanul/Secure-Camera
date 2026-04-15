import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import weaviate
from numpy.core.multiarray import where
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth


class ReIDVectorStore:
    def __init__(
        self,
        # weaviate_url: str = "http://localhost:8080",
        weaviate_url: str = "weaviate_db",
        collection_name: str = "reid_collection",
    ):
        """
        Initialize Weaviate Client for ReID embeddings storage
        """
        # Try to get from environment variables first
        env_weaviate_url = os.environ.get("WEAVIATE_URL")
        weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")

        # Use environment URL if available, otherwise use parameter
        if env_weaviate_url:
            weaviate_url = env_weaviate_url

        # Determine connection type based on URL and API key
        if weaviate_api_key and (
            "weaviate.cloud" in weaviate_url or "wcs" in weaviate_url
        ):
            # Cloud connection
            print(f"🌐 Connecting to Weaviate Cloud: {weaviate_url}")
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=Auth.api_key(weaviate_api_key),
            )
        else:
            # Local connection
            print(f"🔗 Connecting to local Weaviate: {weaviate_url}")
            # Handle Docker service names
            if weaviate_url == "weaviate_db":
                weaviate_url = "http://weaviate_db:8080"
            elif not weaviate_url.startswith("http"):
                weaviate_url = f"http://{weaviate_url}:8080"

            self.client = weaviate.connect_to_local(
                host=weaviate_url.replace("http://", "").replace(":8080", "")
            )

        print(f"✅ Weaviate ready: {self.client.is_ready()}")
        self.collection_name = collection_name
        self.setup_schema()
        print("📋 Schema setup complete")

    def close(self):
        """Close the Weaviate connection"""
        if hasattr(self, "client") and self.client:
            self.client.close()

    def setup_schema(self):
        """
        Create the collection schema if it doesn't exist
        """
        try:
            # Check if the collection already exists
            existing_collection = self.client.collections.use(
                self.collection_name
            ).config.get()
            if existing_collection:
                print(f"📂 Collection '{self.collection_name}' already exists.")
                return
        except Exception:
            # Collection doesn't exist, create it
            print(f"🔧 Creating collection '{self.collection_name}'...")

        try:
            # Create new collection (v4 API)
            _ = self.client.collections.create(
                name=self.collection_name,
                description="Person/Vehicle ReID embeddings with metadata from multiple cameras and frames.",
                properties=[
                    # Property(
                    #     name="object_id",
                    #     data_type=DataType.TEXT,
                    #     description="Unique object identifier assigned by the detection system.",
                    # ),
                    Property(
                        name="class_name",
                        data_type=DataType.TEXT,
                        description="Detected class label such as 'person', 'car', etc.",
                    ),
                    # Property(
                    #     name="confidence",
                    #     data_type=DataType.NUMBER,
                    #     description="Detection confidence score (0–1).",
                    # ),
                    # Property(
                    #     name="bbox",
                    #     data_type=DataType.NUMBER_ARRAY,
                    #     description="Bounding box coordinates [x1, y1, x2, y2].",
                    # ),
                    # Property(
                    #     name="camera_id",
                    #     data_type=DataType.TEXT,
                    #     description="Identifier for the source camera capturing this detection.",
                    # ),
                    # Property(
                    #     name="frame_id",
                    #     data_type=DataType.INT,
                    #     description="Frame number in which this object appeared.",
                    # ),
                    Property(
                        name="timestamp",
                        data_type=DataType.DATE,
                        description="UTC timestamp of detection.",
                    ),
                    # Property(
                    #     name="embedding_method",
                    #     data_type=DataType.TEXT,
                    #     description="Feature extraction model used (e.g. TransReID, OSNet, etc.).",
                    # ),
                    # Property(
                    #     name="person_id",
                    #     data_type=DataType.TEXT,
                    #     description="Assigned person ID for ReID tracking.",
                    # ),
                    Property(
                        name="reid_confidence",
                        data_type=DataType.NUMBER,
                        description="ReID matching confidence score.",
                    ),
                    Property(
                        name="is_new_person",
                        data_type=DataType.BOOL,
                        description="Whether this is a newly identified person.",
                    ),
                    # Property(
                    #     name="image_crop_base64",
                    #     data_type=DataType.TEXT,
                    #     description="Base64 encoded image crop of the detected person (optional).",
                    # ),
                ],
            )
            print(f"✅ Created Weaviate collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            raise

    def store_embeddings(self, objects_data: Dict, embeddings: torch.Tensor):
        """
        Store ReID embeddings with metadata in weaviate
        Arguments:
            objects_data (Dict): Dictionary from C1 containing object metadata
            embeddings (torch.Tensor): Tensor of shape [N, embedding_dim] from C2 ReID Model
        """

        if len(objects_data["objects"]) != embeddings.shape[0]:
            raise ValueError("Number of objects must match number of embeddings")

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        try:
            # collection = self.client.collections.get(self.collection_name)
            collection = self.client.collections.use(self.collection_name)
            results = []

            for i, obj in enumerate(objects_data["objects"]):
                embedding_vector = embeddings[i].tolist()

                # Debug timestamp haI'll send a teams indling
                timestamp_raw = objects_data["metadata"].get(
                    "timestamp", datetime.now().timestamp()
                )

                # Handle timestamp conversion more safely
                if isinstance(timestamp_raw, str):
                    # If it's already a string, try to parse it
                    try:
                        from datetime import datetime as dt

                        if timestamp_raw.endswith("Z"):
                            timestamp_obj = dt.fromisoformat(timestamp_raw[:-1])
                        else:
                            timestamp_obj = dt.fromisoformat(timestamp_raw)
                        timestamp_iso = timestamp_obj.isoformat() + "Z"
                    except (ValueError, TypeError):
                        timestamp_iso = datetime.now().isoformat() + "Z"
                else:
                    # Convert numeric timestamp to ISO format
                    timestamp_iso = (
                        datetime.fromtimestamp(float(timestamp_raw)).isoformat() + "Z"
                    )

                # data_object = {
                #     "object_id": int(obj.get("object_id", f"obj_{i}")),
                #     "class_name": obj.get("class_name", "unknown"),
                #     "confidence": float(obj.get("confidence", 0.0)),  # Ensure float
                #     "bbox": obj.get("bbox", [0, 0, 0, 0]),
                #     "camera_id": str(
                #         objects_data["metadata"].get("camera_id", "unknown")
                #     ),  # Ensure string
                #     "frame_id": int(
                #         objects_data["metadata"].get("frame_id", 0)
                #     ),  # Ensure int
                #     "timestamp": timestamp_iso,
                #     "embedding_method": "Pose2ID_TransReID",
                #     "person_id": str(obj.get("person_id", "unknown")),
                #     "reid_confidence": float(obj.get("reid_confidence", 0.0)),
                #     "is_new_person": bool(obj.get("is_new_person", True)),
                #     "image_crop_base64": obj.get("image_crop_base64", ""),
                # }
                #

                data_object = {
                    "class_name": obj.get("class_name", "unknown"),
                    "timestamp": timestamp_iso,
                    "reid_confidence": float(obj.get("reid_confidence", 0.0)),
                    "is_new_person": bool(obj.get("is_new_person", True)),
                }

                # Insert single object with vector
                result = collection.data.insert(
                    properties=data_object, vector=embedding_vector
                )
                results.append(result)

            print(f"📊 Stored {len(results)} embeddings in Weaviate")
            return results

        except Exception as e:
            print(f"❌ Error storing embeddings: {e}")
            return None

    def search_similar(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 10,
        class_filter: Optional[str] = None,
        confidence_threshold: float = 0.0,
        distance_threshold: float = 1.0,
    ) -> List[Dict]:
        """
        search for similar embeddings using Euclidian distance

        Args:
            query_embedding (torch.Tensor): The query embedding tensor [1, embedding_dim]
            top_k (int): The number of top results to return.
            class_filter (Optional[str]): Filter results by objectclass name (person, car, etc).
            confidence_threshold (float): Minimum detection confidence score.
            distance_threshold (float): Maximum Euclidean distance threshold
        """

        if isinstance(query_embedding, torch.Tensor):
            query_vector = query_embedding.detach().cpu().numpy().flatten().tolist()
        else:
            query_vector = query_embedding.flatten().tolist()

        try:
            # Use v4 API syntax
            collection = self.client.collections.get(self.collection_name)

            # Build where filter using v4 Filter API
            from weaviate.classes.query import Filter

            where_filter = None

            if class_filter and confidence_threshold > 0:
                where_filter = Filter.by_property("class_name").equal(
                    class_filter
                ) & Filter.by_property("confidence").greater_than(confidence_threshold)
            elif class_filter:
                where_filter = Filter.by_property("class_name").equal(class_filter)
            elif confidence_threshold > 0:
                where_filter = Filter.by_property("confidence").greater_than(
                    confidence_threshold
                )

            # Perform vector search using correct v4 API syntax

            # For now, skip filters until we resolve the v4 API syntax
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                # distance=distance_threshold,
                # where=where_filter,
                return_metadata=["distance"],
                include_vector=True,  # Include vectors in response
            )

            # Convert response to expected format
            results = []
            for obj in response.objects:
                # Get distance from metadata
                distance = obj.metadata.distance if obj.metadata else None

                # Apply distance threshold filter if specified
                if distance is not None and distance > distance_threshold:
                    continue

                result = {
                    "object_id": obj.properties.get("object_id"),
                    "class_name": obj.properties.get("class_name"),
                    "confidence": obj.properties.get("confidence"),
                    "bbox": obj.properties.get("bbox"),
                    "camera_id": obj.properties.get("camera_id"),
                    "frame_id": obj.properties.get("frame_id"),
                    "timestamp": obj.properties.get("timestamp"),
                    "embedding_method": obj.properties.get("embedding_method"),
                    "person_id": obj.properties.get("person_id"),
                    "reid_confidence": obj.properties.get("reid_confidence"),
                    "is_new_person": obj.properties.get("is_new_person"),
                    "image_crop_base64": obj.properties.get("image_crop_base64", ""),
                    "vector": obj.vector,  # Include the embedding vector
                    "_additional": {"distance": distance},
                }
                results.append(result)

            return results

        except Exception as e:
            print(f"❌ Error searching embeddings: {e}")
            return []

    def get_statistics(self):
        """Get statistics about stored embeddings"""
        try:
            collection = self.client.collections.get(self.collection_name)

            # Get total count using v4 API
            response = collection.aggregate.over_all(total_count=True)
            total_objects = response.total_count

            # For class distribution, we'd need to query differently in v4
            # This is a simplified version
            return {
                "total_objects": total_objects,
                "class_distribution": "Not implemented in v4 API yet",
            }
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {"total_objects": 0, "class_distribution": {}}


# Usage example function
# def extract_reid_features(objects_tensor, reid_model=None):
#     """
#     Extract ReID features using your existing model
#     This would integrate with your TransReID or other ReID model
#     """
#     # Placeholder - replace with your actual ReID model
#     if reid_model is None:
#         # Simulate feature extraction
#         batch_size = objects_tensor.shape[0]
#         feature_dim = 2048  # Typical ReID feature dimension
#         features = torch.randn(batch_size, feature_dim)
#         return torch.nn.functional.normalize(features, dim=1, p=2)
#     else:
#         with torch.no_grad():
#             features = reid_model(objects_tensor)
#             return torch.nn.functional.normalize(features, dim=1, p=2)
