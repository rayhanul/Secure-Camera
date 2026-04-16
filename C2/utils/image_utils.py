#!/usr/bin/env python3
"""
Image utilities for ReID system
Handles image cropping, encoding, and processing operations
"""

import cv2
import numpy as np
import base64
from typing import Tuple, Optional, Dict, List
import io
from PIL import Image


def extract_person_crop(
    image: np.ndarray, bbox: List[int], padding: int = 10
) -> Optional[np.ndarray]:
    """
    Extract person crop from image using bounding box

    Args:
        image: Input image as numpy array
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Additional padding around bounding box

    Returns:
        Cropped image or None if extraction fails
    """
    try:
        if len(bbox) != 4:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]

        # Add padding and ensure coordinates are within image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Ensure valid crop dimensions
        if x2 <= x1 or y2 <= y1:
            return None

        # Extract crop
        crop = image[y1:y2, x1:x2]

        # Ensure minimum size
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        return crop

    except Exception as e:
        print(f"Error extracting person crop: {e}")
        return None


def encode_image_to_base64(image: np.ndarray, format: str = "JPEG") -> Optional[str]:
    """
    Encode image to base64 string

    Args:
        image: Image as numpy array (assumed to be in RGB format)
        format: Image format (JPEG, PNG)

    Returns:
        Base64 encoded string or None if encoding fails
    """
    try:
        # Convert to PIL Image - assume input is RGB format from tensor
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume RGB format (from tensor conversion)
            pil_image = Image.fromarray(image, mode="RGB")
        elif len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode="L")
        else:
            # Fallback - let PIL figure it out
            pil_image = Image.fromarray(image)

        # Encode to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=85)
        buffer.seek(0)

        # Convert to base64
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")

        return base64_string

    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None


def decode_base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Decode base64 string to image

    Args:
        base64_string: Base64 encoded image string

    Returns:
        Image as numpy array (BGR format for OpenCV) or None if decoding fails
    """
    try:
        if not base64_string:
            return None

        # Decode base64
        image_bytes = base64.b64decode(base64_string)

        # Convert to PIL Image
        buffer = io.BytesIO(image_bytes)
        pil_image = Image.open(buffer)

        # Convert to numpy array
        if pil_image.mode == "RGB":
            image_rgb = np.array(pil_image)
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr
        elif pil_image.mode == "L":
            return np.array(pil_image)
        else:
            # Convert to RGB first, then to BGR
            pil_image = pil_image.convert("RGB")
            image_rgb = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr

    except Exception as e:
        print(f"Error decoding base64 to image: {e}")
        return None


def resize_image_maintain_aspect(
    image: np.ndarray, target_height: int = 256, target_width: int = 128
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio (common for ReID models)

    Args:
        image: Input image
        target_height: Target height
        target_width: Target width

    Returns:
        Resized image
    """
    try:
        h, w = image.shape[:2]

        # Calculate aspect ratios
        aspect_ratio = w / h
        target_aspect_ratio = target_width / target_height

        if aspect_ratio > target_aspect_ratio:
            # Width is the limiting factor
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height))

        # Create canvas of target size and center the resized image
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Calculate position to center the image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        # Place resized image on canvas
        canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
            resized
        )

        return canvas

    except Exception as e:
        print(f"Error resizing image: {e}")
        return image


def process_detection_for_storage(
    original_image: np.ndarray,
    detection: Dict,
    store_crops: bool = True,
    crop_size: Tuple[int, int] = (256, 128),
) -> Dict:
    """
    Process a detection for storage, optionally extracting and encoding image crop

    Args:
        original_image: Original full image
        detection: Detection dictionary with bbox and other metadata
        store_crops: Whether to extract and store image crops
        crop_size: Target size for crops (height, width)

    Returns:
        Enhanced detection dictionary with optional image_crop_base64
    """
    try:
        enhanced_detection = detection.copy()

        if store_crops and original_image is not None:
            bbox = detection.get("bbox")
            if bbox:
                # Extract person crop
                crop = extract_person_crop(original_image, bbox, padding=5)

                if crop is not None:
                    # Resize crop to standard size
                    resized_crop = resize_image_maintain_aspect(
                        crop, crop_size[0], crop_size[1]
                    )

                    # Encode to base64
                    base64_crop = encode_image_to_base64(resized_crop)

                    if base64_crop:
                        enhanced_detection["image_crop_base64"] = base64_crop

        return enhanced_detection

    except Exception as e:
        print(f"Error processing detection for storage: {e}")
        return detection


def create_similarity_visualization(
    query_image: np.ndarray, similar_results: List[Dict], max_results: int = 5
) -> np.ndarray:
    """
    Create a visualization showing query image and similar matches

    Args:
        query_image: Query image
        similar_results: List of similar match results with image data
        max_results: Maximum number of results to show

    Returns:
        Combined visualization image
    """
    try:
        # Standardize image size
        img_height, img_width = 200, 120
        query_resized = cv2.resize(query_image, (img_width, img_height))

        # Start with query image
        visualization = query_resized.copy()

        # Add query label
        cv2.putText(
            visualization,
            "Query",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Process similar results
        similar_images = []
        for i, result in enumerate(similar_results[:max_results]):
            try:
                # Try to get image from base64
                base64_data = result.get("image_crop_base64", "")
                if base64_data:
                    similar_img = decode_base64_to_image(base64_data)
                    if similar_img is not None:
                        similar_resized = cv2.resize(
                            similar_img, (img_width, img_height)
                        )

                        # Add distance label
                        distance = result.get("_additional", {}).get("distance", "N/A")
                        cv2.putText(
                            similar_resized,
                            f"Dist: {distance:.3f}"
                            if isinstance(distance, float)
                            else str(distance),
                            (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 255),
                            1,
                        )

                        # Add person ID
                        person_id = result.get("person_id", "Unknown")
                        cv2.putText(
                            similar_resized,
                            f"ID: {person_id}",
                            (5, img_height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 0),
                            1,
                        )

                        similar_images.append(similar_resized)

            except Exception as e:
                print(f"Error processing similar image {i}: {e}")
                continue

        # Combine all images horizontally
        if similar_images:
            visualization = np.hstack([visualization] + similar_images)

        return visualization

    except Exception as e:
        print(f"Error creating similarity visualization: {e}")
        return query_image


def validate_image_for_reid(image: np.ndarray) -> bool:
    """
    Validate if image is suitable for ReID processing

    Args:
        image: Input image

    Returns:
        True if image is valid for ReID processing
    """
    try:
        if image is None:
            return False

        # Check dimensions
        if len(image.shape) not in [2, 3]:
            return False

        h, w = image.shape[:2]

        # Check minimum size
        if h < 32 or w < 16:
            return False

        # Check aspect ratio (people are typically taller than wide)
        aspect_ratio = w / h
        if aspect_ratio > 2.0:  # Too wide
            return False

        # Check data type
        if image.dtype != np.uint8:
            return False

        return True

    except Exception:
        return False


# Utility functions for batch processing
def process_batch_detections(
    original_image: np.ndarray, detections: List[Dict], store_crops: bool = True
) -> List[Dict]:
    """
    Process multiple detections from a single image

    Args:
        original_image: Original full image
        detections: List of detection dictionaries
        store_crops: Whether to extract and store crops

    Returns:
        List of enhanced detection dictionaries
    """
    enhanced_detections = []

    for detection in detections:
        enhanced_detection = process_detection_for_storage(
            original_image, detection, store_crops
        )
        enhanced_detections.append(enhanced_detection)

    return enhanced_detections


def get_image_info(image: np.ndarray) -> Dict:
    """
    Get detailed information about an image

    Args:
        image: Input image

    Returns:
        Dictionary with image information
    """
    try:
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes,
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
            "min_value": image.min(),
            "max_value": image.max(),
            "mean_value": image.mean(),
            "is_valid_for_reid": validate_image_for_reid(image),
        }
    except Exception as e:
        return {"error": str(e)}
