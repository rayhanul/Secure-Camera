#!/usr/bin/env python3
"""
Results Saver Utility for ReID System
Organizes processed images and their similar matches into structured folders
Uses MongoDB with decryption to retrieve full image data
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


class ReIDResultsSaver:
    """
    Simple ReID results saver that works with MongoDB encrypted storage

    Results structure:
    results/
    ├── session_timestamp/
    │   ├── query_1/
    │   │   ├── query_image.jpg
    │   │   ├── similar_1.jpg
    │   │   ├── similar_2.jpg
    │   │   └── metadata.json
    │   └── query_2/
    │       └── ...
    """

    def __init__(self, mongo_storage, results_base_dir: str = "results"):
        """
        Initialize the results saver

        Args:
            mongo_storage: SecureReIDStorage instance for decrypting data
            results_base_dir: Base directory for saving results
        """
        self.mongo_storage = mongo_storage
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(exist_ok=True)

        # Create session folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.results_base_dir / f"session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)

        print(f"📁 Results will be saved to: {self.session_dir}")

    def save_query_results_with_image(
        self,
        query_embedding: np.ndarray,
        similar_embeddings: List[np.ndarray],
        query_name: str = None,
        camera_id: str = "unknown",
        frame_id: str = "unknown",
        query_image_data: bytes = None,
    ) -> Path:
        """
        Save query results with current frame query image

        Args:
            query_embedding: The query embedding vector
            similar_embeddings: List of similar embedding vectors from Weaviate
            query_name: Optional name for the query folder
            camera_id: Camera ID for original image naming
            frame_id: Frame ID for original image naming
            query_image_data: Current query image as bytes

        Returns:
            Path to created query folder
        """
        try:
            # Create query folder
            if query_name:
                folder_name = self._sanitize_filename(query_name)
            else:
                folder_name = f"query_{datetime.now().strftime('%H%M%S')}"

            query_folder = self.session_dir / folder_name
            query_folder.mkdir(exist_ok=True)

            # Save current query image directly from frame data
            if query_image_data:
                query_image_path = self._save_current_query_image(
                    query_folder, query_image_data, camera_id, frame_id
                )
                if query_image_path:
                    print(f"💾 Saved original image: {query_image_path.name}")
                else:
                    print(f"⚠️ Failed to save original image")
            else:
                print("⚠️ No query image data provided")

            # Get and save similar images from MongoDB (not including query)
            similar_paths = []
            metadata_list = []

            for i, similar_embedding in enumerate(similar_embeddings):
                similar_results = self.mongo_storage.get_similar_results(
                    similar_embedding
                )

                for j, result in enumerate(similar_results):
                    # Save image
                    image_path = self._save_similar_image(query_folder, result, i, j)
                    if image_path:
                        similar_paths.append(image_path)

                    # Collect metadata
                    metadata_list.append(
                        {
                            "embedding_index": i,
                            "result_index": j,
                            "person_id": result.person_id,
                            "camera_id": result.camera_id,
                            "timestamp": result.timestamp,
                            "confidence": result.confidence,
                            "reid_confidence": result.reid_confidence,
                            "class_name": result.class_name,
                            "bbox": result.bbox,
                            "is_new_person": result.is_new_person,
                            "saved_image": image_path.name if image_path else None,
                        }
                    )

            # Save metadata
            self._save_metadata(query_folder, metadata_list)

            # Save summary file
            self._save_query_summary(query_folder, metadata_list)

            print(f"✅ Saved {len(similar_paths)} similar images")
            return query_folder

        except Exception as e:
            print(f"❌ Error saving query results: {e}")
            return None

    def save_query_results(
        self,
        query_embedding: np.ndarray,
        similar_embeddings: List[np.ndarray],
        query_name: str = None,
        camera_id: str = "unknown",
        frame_id: str = "unknown",
    ) -> Path:
        """
        Save query results: query image + similar images from MongoDB

        Args:
            query_embedding: The query embedding vector
            similar_embeddings: List of similar embedding vectors from Weaviate
            query_name: Optional name for the query folder
            camera_id: Camera ID for original image naming
            frame_id: Frame ID for original image naming

        Returns:
            Path to created query folder
        """
        try:
            # Create query folder
            if query_name:
                folder_name = self._sanitize_filename(query_name)
            else:
                folder_name = f"query_{datetime.now().strftime('%H%M%S')}"

            query_folder = self.session_dir / folder_name
            query_folder.mkdir(exist_ok=True)

            # Skip MongoDB query image lookup - we already saved the current query image above

            # Get and save similar images from MongoDB
            similar_paths = []
            metadata_list = []

            for i, similar_embedding in enumerate(similar_embeddings):
                similar_results = self.mongo_storage.get_similar_results(
                    similar_embedding
                )

                for j, result in enumerate(similar_results):
                    # Save image
                    image_path = self._save_similar_image(query_folder, result, i, j)
                    if image_path:
                        similar_paths.append(image_path)

                    # Collect metadata
                    metadata_list.append(
                        {
                            "embedding_index": i,
                            "result_index": j,
                            "person_id": result.person_id,
                            "camera_id": result.camera_id,
                            "timestamp": result.timestamp,
                            "confidence": result.confidence,
                            "reid_confidence": result.reid_confidence,
                            "class_name": result.class_name,
                            "bbox": result.bbox,
                            "is_new_person": result.is_new_person,
                            "saved_image": image_path.name if image_path else None,
                        }
                    )

            # Save metadata
            self._save_metadata(query_folder, metadata_list)

            # Save summary file
            self._save_query_summary(query_folder, metadata_list)
            return query_folder

        except Exception as e:
            return None

    def save_query_results_by_hash(
        self,
        query_hash: str,
        similar_hashes: List[str],
        query_name: str = None,
        camera_id: str = "unknown",
        frame_id: str = "unknown",
    ) -> Path:
        """
        Save query results using embedding hashes instead of vectors

        Args:
            query_hash: The query embedding hash
            similar_hashes: List of similar embedding hashes
            query_name: Optional name for the query folder
            camera_id: Camera ID for original image naming
            frame_id: Frame ID for original image naming

        Returns:
            Path to created query folder
        """
        try:
            # Create query folder
            if query_name:
                folder_name = self._sanitize_filename(query_name)
            else:
                folder_name = f"query_{datetime.now().strftime('%H%M%S')}"

            query_folder = self.session_dir / folder_name
            query_folder.mkdir(exist_ok=True)

            # Get and save query image from MongoDB by hash
            query_results = self.mongo_storage.get_similar_results_by_hash(query_hash)
            if query_results:
                query_image_path = self._save_query_image(
                    query_folder, query_results[0], camera_id, frame_id
                )
                print(f"💾 Saved original image: {query_image_path.name}")
            else:
                print(f"⚠️ No query image found in MongoDB for hash {query_hash[:8]}...")

            # Get and save similar images from MongoDB by hashes
            similar_paths = []
            metadata_list = []

            for i, similar_hash in enumerate(similar_hashes):
                similar_results = self.mongo_storage.get_similar_results_by_hash(
                    similar_hash
                )

                for j, result in enumerate(similar_results):
                    # Save image
                    image_path = self._save_similar_image(query_folder, result, i, j)
                    if image_path:
                        similar_paths.append(image_path)

                    # Collect metadata
                    metadata_list.append(
                        {
                            "embedding_hash": similar_hash,
                            "result_index": j,
                            "person_id": result.person_id,
                            "camera_id": result.camera_id,
                            "timestamp": result.timestamp,
                            "confidence": result.confidence,
                            "reid_confidence": result.reid_confidence,
                            "class_name": result.class_name,
                            "bbox": result.bbox,
                            "is_new_person": result.is_new_person,
                            "saved_image": image_path.name if image_path else None,
                        }
                    )

            # Save metadata
            self._save_metadata(query_folder, metadata_list)

            # Save summary file
            self._save_query_summary(query_folder, metadata_list)

            print(f"✅ Saved {len(similar_paths)} similar images using hashes")
            return query_folder

        except Exception as e:
            print(f"❌ Error saving query results by hash: {e}")
            return None

    def _save_current_query_image(
        self,
        query_folder: Path,
        query_image_data: bytes,
        camera_id: str = "unknown",
        frame_id: str = "unknown",
    ) -> Optional[Path]:
        """Save the current query image from frame data"""
        try:
            if not query_image_data:
                return None

            # Create filename with timestamp, camera, and frame info
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"original_{timestamp_str}_camera{camera_id}_frame{frame_id}.png"
            filename = self._sanitize_filename(filename)
            image_path = query_folder / filename

            # Save image bytes directly as PNG
            with open(image_path, "wb") as f:
                f.write(query_image_data)

            return image_path

        except Exception as e:
            print(f"⚠️ Error saving current query image: {e}")
            return None

    def _save_query_image(
        self,
        query_folder: Path,
        reid_result,
        camera_id: str = "unknown",
        frame_id: str = "unknown",
    ) -> Optional[Path]:
        """Save the query image from ReIDResult with original naming format"""
        try:
            if not reid_result.image:
                return None

            # Create filename with timestamp, camera, and frame info
            try:
                if isinstance(reid_result.timestamp, str):
                    timestamp_str = (
                        reid_result.timestamp.replace(":", "")
                        .replace("-", "")
                        .replace("T", "")
                        .replace("Z", "")[:14]
                    )
                else:
                    timestamp_str = datetime.fromtimestamp(
                        reid_result.timestamp
                    ).strftime("%Y%m%d_%H%M%S")
            except Exception as e:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"original_{timestamp_str}_camera{camera_id}_frame{frame_id}.png"
            filename = self._sanitize_filename(filename)
            image_path = query_folder / filename

            # Convert bytes back to image
            image = Image.open(io.BytesIO(reid_result.image))

            # Save as PNG directly using PIL
            image.save(str(image_path), format="PNG")
            return image_path

        except Exception as e:
            return None

    def _save_similar_image(
        self, query_folder: Path, reid_result, embedding_idx: int, result_idx: int
    ) -> Optional[Path]:
        """Save a similar image from ReIDResult"""
        try:
            if not reid_result.image:
                return None

            # Create filename
            try:
                # Handle both string and float timestamps
                if isinstance(reid_result.timestamp, str):
                    timestamp_str = reid_result.timestamp.replace(":", "").replace(
                        "-", ""
                    )[:6]
                else:
                    timestamp_str = datetime.fromtimestamp(
                        reid_result.timestamp
                    ).strftime("%H%M%S")
            except Exception as e:
                timestamp_str = datetime.now().strftime("%H%M%S")
            filename = f"similar_{embedding_idx}_{result_idx}_{reid_result.person_id}_{timestamp_str}.jpg"
            filename = self._sanitize_filename(filename)
            image_path = query_folder / filename

            # Convert bytes back to image
            image = Image.open(io.BytesIO(reid_result.image))
            image_array = np.array(image)

            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            success = cv2.imwrite(str(image_path), image_array)
            if success:
                return image_path
            else:
                print(f"⚠️ Failed to save similar image {filename}")
                return None

        except Exception as e:
            print(f"⚠️ Error saving similar image: {e}")
            return None

    def _save_metadata(self, query_folder: Path, metadata_list: List[Dict]) -> Path:
        """Save metadata JSON file"""
        metadata_path = query_folder / "metadata.json"

        full_metadata = {
            "session_timestamp": datetime.now().isoformat(),
            "query_folder": query_folder.name,
            "total_similar_images": len([m for m in metadata_list if m["saved_image"]]),
            "results": metadata_list,
        }

        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        return metadata_path

    def _save_query_summary(
        self, query_folder: Path, metadata_list: List[Dict]
    ) -> Path:
        """Save human-readable summary file for the query folder"""
        summary_path = query_folder / "summary.txt"

        # Count various statistics
        total_similar = len([m for m in metadata_list if m["saved_image"]])
        unique_persons = len(
            set(m["person_id"] for m in metadata_list if m["person_id"] is not None)
        )
        unique_cameras = len(set(m["camera_id"] for m in metadata_list))
        avg_confidence = (
            sum(m["reid_confidence"] for m in metadata_list) / len(metadata_list)
            if metadata_list
            else 0
        )

        # Get time range
        timestamps = [m["timestamp"] for m in metadata_list if m["timestamp"]]
        time_range = f"{min(timestamps)} to {max(timestamps)}" if timestamps else "N/A"

        # Group by confidence levels
        high_conf = len([m for m in metadata_list if m["reid_confidence"] > 0.9])
        medium_conf = len(
            [m for m in metadata_list if 0.7 <= m["reid_confidence"] <= 0.9]
        )
        low_conf = len([m for m in metadata_list if m["reid_confidence"] < 0.7])

        summary_content = f"""ReID Query Summary
==================
Query Folder: {query_folder.name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

STATISTICS
----------
Total Similar Images: {total_similar}
Unique Person IDs: {unique_persons if unique_persons > 0 else "No identified persons"}
Unique Cameras: {unique_cameras}
Average ReID Confidence: {avg_confidence:.3f}
Time Range: {time_range}

CONFIDENCE DISTRIBUTION
-----------------------
High Confidence (>0.9): {high_conf} images
Medium Confidence (0.7-0.9): {medium_conf} images
Low Confidence (<0.7): {low_conf} images

DETAILED RESULTS
----------------
"""

        # Add detailed results
        for i, result in enumerate(metadata_list, 1):
            summary_content += f"{i}. {result['saved_image'] or 'N/A'}\n"
            summary_content += f"   Person ID: {result['person_id'] or 'Unknown'}\n"
            summary_content += f"   Camera: {result['camera_id']}\n"
            summary_content += f"   ReID Confidence: {result['reid_confidence']:.3f}\n"
            summary_content += f"   Timestamp: {result['timestamp']}\n"
            if result["bbox"]:
                summary_content += f"   BBox: {result['bbox']}\n"
            summary_content += "\n"

        with open(summary_path, "w") as f:
            f.write(summary_content)

        print(f"📄 Saved summary: {summary_path.name}")
        return summary_path

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility"""
        # Remove/replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Limit length
        if len(filename) > 100:
            filename = filename[:100]

        return filename

    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        query_folders = [d for d in self.session_dir.iterdir() if d.is_dir()]

        total_images = 0
        total_queries = len(query_folders)

        for folder in query_folders:
            images = [
                f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".png"]
            ]
            total_images += len(images)

        return {
            "session_directory": str(self.session_dir),
            "total_queries": total_queries,
            "total_images_saved": total_images,
            "query_folders": [f.name for f in query_folders],
        }


def create_simple_results_saver(
    mongo_storage, base_dir: str = "results"
) -> ReIDResultsSaver:
    """
    Factory function to create a simple results saver

    Args:
        mongo_storage: SecureReIDStorage instance
        base_dir: Base directory for results

    Returns:
        ReIDResultsSaver instance
    """
    return ReIDResultsSaver(mongo_storage, base_dir)


# Example usage function
def save_reid_session(
    mongo_storage,
    query_embedding: np.ndarray,
    similar_embeddings: List[np.ndarray],
    session_name: str = None,
):
    """
    Simple function to save a complete ReID session

    Args:
        mongo_storage: SecureReIDStorage instance
        query_embedding: Query embedding vector
        similar_embeddings: List of similar embeddings from Weaviate search
        session_name: Optional name for the session
    """
    saver = ReIDResultsSaver(mongo_storage)

    query_folder = saver.save_query_results(
        query_embedding=query_embedding,
        similar_embeddings=similar_embeddings,
        query_name=session_name,
    )

    if query_folder:
        summary = saver.get_session_summary()
        print(f"📊 Session Summary:")
        print(f"   Queries processed: {summary['total_queries']}")
        print(f"   Images saved: {summary['total_images_saved']}")
        print(f"   Results directory: {summary['session_directory']}")

    return saver
