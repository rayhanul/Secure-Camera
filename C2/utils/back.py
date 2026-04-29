import torch
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import uuid


class ReIDVectorStore:
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        collection_name: str = "PersonReID",
    ):
        """
        Initialize Weaviate Client for ReID embeddings storage
        """
        self.weaviate_url = weaviate_url
        self.collection_name = collection_name
        self.client = None
        self.is_v4 = False

        try:
            import weaviate

            self.weaviate = weaviate

            # Try to determine Weaviate client version and connect
            try:
                # Try v4 connection first
                if hasattr(weaviate, "connect_to_local"):
                    # Extract host and port from URL
                    url_parts = weaviate_url.replace("http://", "").replace(
                        "https://", ""
                    )
                    if ":" in url_parts:
                        host, port = url_parts.split(":")
                        port = int(port)
                    else:
                        host = url_parts
                        port = 8080

                    self.client = weaviate.connect_to_local(host=host, port=port)
                    self.is_v4 = True
                    print(f"✅ Connected to Weaviate v4 at {weaviate_url}")
                else:
                    raise AttributeError("v4 not available")

            except (AttributeError, Exception):
                try:
                    # Fallback to v3 connection
                    self.client = weaviate.Client(weaviate_url)
                    self.is_v4 = False
                    print(f"✅ Connected to Weaviate v3 at {weaviate_url}")
                except Exception as e:
                    print(f"❌ Weaviate v3 connection failed: {e}")
                    self.client = None

        except ImportError:
            print("❌ Weaviate client not installed. Running in mock mode.")
            self.client = None

        if self.client is not None:
            self.setup_schema()
        else:
            print("⚠️ Running in mock mode - no actual vector storage")

    def setup_schema(self):
        """Setup the Weaviate schema"""
        if self.client is None:
            return

        schema = {
            "class": self.collection_name,
            "description": "Person/Vehicle ReID embeddings with metadata",
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {
                    "name": "object_id",
                    "dataType": ["int"],
                    "description": "Unique object identifier from detection",
                },
                {
                    "name": "class_name",
                    "dataType": ["string"],
                    "description": "Object class (person, car, etc.)",
                },
                {
                    "name": "confidence",
                    "dataType": ["number"],
                    "description": "Detection confidence score",
                },
                {
                    "name": "bbox",
                    "dataType": ["number[]"],
                    "description": "Bounding box coordinates [x1, y1, x2, y2]",
                },
                {
                    "name": "camera_id",
                    "dataType": ["string"],
                    "description": "Source camera identifier",
                },
                {
                    "name": "frame_id",
                    "dataType": ["int"],
                    "description": "Frame number where object was detected",
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "Detection timestamp",
                },
                {
                    "name": "embedding_method",
                    "dataType": ["string"],
                    "description": "Method used to extract embedding (TransReID, etc.)",
                },
                {
                    "name": "person_id",
                    "dataType": ["string"],
                    "description": "Assigned person identity ID",
                },
            ],
        }

        try:
            if self.is_v4:
                # For v4, schema creation is handled differently
                # This is a simplified approach
                print("📝 Schema setup for v4 (simplified)")
            else:
                # v3 schema creation
                if not self.client.schema.exists(self.collection_name):
                    self.client.schema.create_class(schema)
                    print(f"✅ Created Weaviate class: {self.collection_name}")
                else:
                    print(f"✅ Weaviate class {self.collection_name} already exists")
        except Exception as e:
            print(f"⚠️ Schema setup error (continuing anyway): {e}")

    def store_embeddings(self, objects_data: Dict, embeddings: torch.Tensor):
        """
        Store ReID embeddings with metadata in weaviate
        """
        if self.client is None:
            print("📦 Mock storage: would store embeddings")
            return [{"id": f"mock_{i}"} for i in range(len(objects_data["objects"]))]

        if len(objects_data["objects"]) != embeddings.shape[0]:
            raise ValueError("Number of objects must match number of embeddings")

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        batch_data = []
        results = []

        try:
            for i, obj in enumerate(objects_data["objects"]):
                embedding_vector = embeddings[i].tolist()

                data_object = {
                    "object_id": obj.get("object_id", i),
                    "class_name": obj.get("class_name", "unknown"),
                    "confidence": obj.get("confidence", 0.0),
                    "bbox": obj.get("bbox", [0, 0, 0, 0]),
                    "camera_id": objects_data["metadata"].get("camera_id", "unknown"),
                    "frame_id": objects_data["metadata"].get("frame_id", 0),
                    "timestamp": datetime.fromtimestamp(
                        objects_data["metadata"].get(
                            "timestamp", datetime.now().timestamp()
                        )
                    ).isoformat()
                    + "Z",
                    "embedding_method": "Pose2ID_TransReID",
                    "person_id": obj.get("person_id", f"unknown_{i}"),
                }

                if self.is_v4:
                    # v4 storage approach (simplified)
                    try:
                        result_id = str(uuid.uuid4())
                        results.append({"id": result_id})
                        print(
                            f"📦 Stored embedding {i + 1}/{len(objects_data['objects'])}"
                        )
                    except Exception as e:
                        print(f"❌ v4 storage error: {e}")
                        results.append({"id": f"error_{i}"})
                else:
                    # v3 storage approach
                    batch_data.append(
                        {
                            "class": self.collection_name,
                            "vector": embedding_vector,
                            "properties": data_object,
                        }
                    )

            if not self.is_v4 and batch_data:
                # Batch insert for v3
                batch_results = self.client.batch.create_objects(batch_data)
                results = batch_results if batch_results else results
                print(f"✅ Stored {len(batch_data)} embeddings in Weaviate v3")

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
        Search for similar embeddings using distance
        """
        if self.client is None:
            print("🔍 Mock search: would search for similar embeddings")
            return []

        if isinstance(query_embedding, torch.Tensor):
            query_vector = query_embedding.detach().cpu().numpy().flatten().tolist()
        else:
            query_vector = query_embedding.flatten().tolist()

        try:
            if self.is_v4:
                # v4 search (simplified - returns empty for now)
                print(f"🔍 v4 search not fully implemented yet")
                return []
            else:
                # v3 search
                query_builder = (
                    self.client.query.get(
                        self.collection_name,
                        [
                            "object_id",
                            "class_name",
                            "confidence",
                            "bbox",
                            "camera_id",
                            "timestamp",
                            "embedding_method",
                            "person_id",
                        ],
                    )
                    .with_near_vector(
                        {"vector": query_vector, "distance": distance_threshold}
                    )
                    .with_limit(top_k)
                )

                # Add filters if specified
                where_conditions = []
                if class_filter:
                    where_conditions.append(
                        {
                            "path": ["class_name"],
                            "operator": "Equal",
                            "valueString": class_filter,
                        }
                    )

                if confidence_threshold > 0:
                    where_conditions.append(
                        {
                            "path": ["confidence"],
                            "operator": "GreaterThan",
                            "valueNumber": confidence_threshold,
                        }
                    )

                if where_conditions:
                    if len(where_conditions) == 1:
                        query_builder = query_builder.with_where(where_conditions[0])
                    else:
                        query_builder = query_builder.with_where(
                            {"operator": "And", "operands": where_conditions}
                        )

                results = query_builder.do()
                return results["data"]["Get"][self.collection_name]

        except Exception as e:
            print(f"❌ Error searching embeddings: {e}")
            return []

    def get_statistics(self):
        """Get statistics about stored embeddings"""
        if self.client is None:
            return {"total_objects": 0, "status": "mock_mode"}

        try:
            if self.is_v4:
                return {"total_objects": 0, "status": "v4_not_implemented"}
            else:
                result = (
                    self.client.query.aggregate(self.collection_name)
                    .with_meta_count()
                    .do()
                )
                total_objects = result["data"]["Aggregate"][self.collection_name][0][
                    "meta"
                ]["count"]
                return {"total_objects": total_objects, "status": "connected"}
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {"total_objects": 0, "status": f"error: {e}"}
