#!/usr/bin/env python3
"""
Test script for ReIDResultsSaver functionality
Demonstrates how to save ReID processing results with organized image folders
"""

import os
import sys
import cv2
import torch
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.results_saver import ReIDResultsSaver, integrate_with_reid_processor


def create_test_image(width=640, height=480, text="Test Image", color=(100, 150, 200)):
    """Create a test image with text overlay"""
    image = np.full((height, width, 3), color, dtype=np.uint8)

    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Center the text
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2

    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    # Add some random rectangles to make it more interesting
    for i in range(5):
        pt1 = (np.random.randint(0, width // 2), np.random.randint(0, height // 2))
        pt2 = (
            np.random.randint(width // 2, width),
            np.random.randint(height // 2, height),
        )
        color_rect = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(image, pt1, pt2, color_rect, 2)

    return image


def create_sample_reid_results():
    """Create sample ReID processing results"""
    return [
        {
            "detection_id": "det_001",
            "person_id": "person_001",
            "confidence": 0.92,
            "is_new_person": True,
            "similar_detections": 2,
            "bbox": [100, 150, 300, 400],
            "camera_id": "camera_01",
            "frame_id": 12345,
            "timestamp": "2024-01-15T14:30:22Z",
            "cross_camera_matches": ["camera_02_person_001"],
            "weaviate_id": "uuid-123-456-789",
            "reid_confidence": 0.87,
        },
        {
            "detection_id": "det_002",
            "person_id": "person_002",
            "confidence": 0.89,
            "is_new_person": False,
            "similar_detections": 1,
            "bbox": [400, 120, 580, 380],
            "camera_id": "camera_01",
            "frame_id": 12345,
            "timestamp": "2024-01-15T14:30:22Z",
            "cross_camera_matches": [],
            "weaviate_id": "uuid-987-654-321",
            "reid_confidence": 0.94,
        },
        {
            "detection_id": "det_003",
            "person_id": "person_003",
            "confidence": 0.78,
            "is_new_person": True,
            "similar_detections": 0,
            "bbox": [50, 300, 180, 470],
            "camera_id": "camera_01",
            "frame_id": 12345,
            "timestamp": "2024-01-15T14:30:22Z",
            "cross_camera_matches": [],
            "weaviate_id": "uuid-111-222-333",
            "reid_confidence": 0.65,
        },
    ]


def create_sample_similar_results():
    """Create sample similar image results"""
    return [
        {
            "object_id": "det_historical_001",
            "person_id": "person_001",
            "confidence": 0.91,
            "bbox": [110, 160, 290, 390],
            "camera_id": "camera_02",
            "frame_id": 11800,
            "timestamp": "2024-01-15T12:15:30Z",
            "weaviate_id": "uuid-hist-001",
            "reid_confidence": 0.88,
            "_additional": {"distance": 0.23},
        },
        {
            "object_id": "det_historical_002",
            "person_id": "person_001",
            "confidence": 0.85,
            "bbox": [95, 140, 285, 410],
            "camera_id": "camera_03",
            "frame_id": 11200,
            "timestamp": "2024-01-15T11:45:15Z",
            "weaviate_id": "uuid-hist-002",
            "reid_confidence": 0.82,
            "_additional": {"distance": 0.31},
        },
        {
            "object_id": "det_historical_003",
            "person_id": "person_002",
            "confidence": 0.93,
            "bbox": [390, 100, 570, 360],
            "camera_id": "camera_02",
            "frame_id": 11950,
            "timestamp": "2024-01-15T13:20:45Z",
            "weaviate_id": "uuid-hist-003",
            "reid_confidence": 0.91,
            "_additional": {"distance": 0.18},
        },
    ]


def test_basic_functionality():
    """Test basic results saver functionality"""
    print("🧪 Testing Basic Results Saver Functionality")
    print("=" * 50)

    # Initialize results saver
    saver = ReIDResultsSaver("test_results_basic")

    # Create test image
    test_image = create_test_image(text="Original Frame 12345")

    # Create sample results
    reid_results = create_sample_reid_results()

    # Save complete results
    try:
        results_folder = saver.save_complete_results(
            image_name="test_frame_12345",
            original_image=test_image,
            reid_results=reid_results,
            original_filename="frame_12345.jpg",
        )

        print(f"✅ Basic test completed successfully")
        print(f"📁 Results saved to: {results_folder}")

        # List created files
        files = list(results_folder.iterdir())
        print(f"📄 Created {len(files)} files:")
        for file in files:
            print(f"   - {file.name}")

        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def test_with_similar_images():
    """Test saving with similar images"""
    print("\n🧪 Testing Results Saver with Similar Images")
    print("=" * 50)

    # Initialize results saver
    saver = ReIDResultsSaver("test_results_similar")

    # Create original image
    original_image = create_test_image(
        text="Camera 01 - Frame 12345", color=(120, 180, 100)
    )

    # Create similar images cache
    similar_images = {}
    similar_results = create_sample_similar_results()

    # Generate similar images
    colors = [(150, 100, 180), (180, 150, 100), (100, 180, 150)]
    for i, result in enumerate(similar_results):
        weaviate_id = result["weaviate_id"]
        camera_id = result["camera_id"]
        frame_id = result["frame_id"]

        similar_image = create_test_image(
            text=f"{camera_id} - Frame {frame_id}", color=colors[i % len(colors)]
        )
        similar_images[weaviate_id] = similar_image
        result["image_data"] = similar_image  # Add image data to result

    # Create ReID results
    reid_results = create_sample_reid_results()

    try:
        results_folder = saver.save_complete_results(
            image_name="multi_camera_frame_12345",
            original_image=original_image,
            reid_results=reid_results,
            similar_results=similar_results,
            original_filename="frame_12345_cam01.jpg",
            processing_info={
                "camera_count": 3,
                "total_similar_matches": len(similar_results),
                "processing_time_ms": 1250,
            },
            image_cache=similar_images,
        )

        print(f"✅ Similar images test completed successfully")
        print(f"📁 Results saved to: {results_folder}")

        # Count different types of files
        files = list(results_folder.iterdir())
        similar_count = len([f for f in files if f.name.startswith("similar_")])

        print(f"📄 Created {len(files)} files total:")
        print(f"   - 1 original image")
        print(f"   - {similar_count} similar images")
        print(f"   - 1 metadata.json")
        print(f"   - 1 summary.txt")

        return True

    except Exception as e:
        print(f"❌ Similar images test failed: {e}")
        return False


def test_integration_function():
    """Test the integration function"""
    print("\n🧪 Testing Integration with ReID Processor")
    print("=" * 50)

    # Initialize results saver
    saver = ReIDResultsSaver("test_results_integration")

    # Create test data as it would come from ReID processor
    original_image = create_test_image(
        text="Integration Test Frame", color=(200, 120, 80)
    )

    reid_results = create_sample_reid_results()

    try:
        # Test integration function
        results_folder = integrate_with_reid_processor(
            results_saver=saver,
            reid_results=reid_results,
            original_image_data=original_image,
            frame_id="12345",
            camera_id="camera_01",
        )

        print(f"✅ Integration test completed successfully")
        print(f"📁 Results saved to: {results_folder}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_results_summary():
    """Test results summary functionality"""
    print("\n🧪 Testing Results Summary")
    print("=" * 50)

    # Initialize results saver (should find previously created folders)
    saver = ReIDResultsSaver("test_results_basic")

    try:
        summary = saver.get_results_summary()

        print(f"✅ Summary generated successfully")
        print(f"📊 Summary statistics:")
        print(f"   - Total folders: {summary['total_folders']}")
        print(f"   - Total images: {summary['total_images']}")
        print(f"   - Similar matches: {summary['total_similar_matches']}")

        if summary["folders"]:
            print(f"📁 Folders found:")
            for folder in summary["folders"]:
                print(f"   - {folder['name']} ({len(folder['files'])} files)")

        return True

    except Exception as e:
        print(f"❌ Summary test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)

    saver = ReIDResultsSaver("test_results_edge_cases")

    tests_passed = 0
    total_tests = 3

    # Test 1: No image data
    try:
        results_folder = saver.save_complete_results(
            image_name="no_image_test",
            original_image=np.zeros((100, 100, 3), dtype=np.uint8),
            reid_results=[],
            similar_results=None,
        )
        print("✅ Test 1 passed: Handled empty results")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    # Test 2: Invalid characters in filename
    try:
        results_folder = saver.save_complete_results(
            image_name="test<>:|?*/image",  # Invalid characters
            original_image=create_test_image(text="Invalid Name Test"),
            reid_results=create_sample_reid_results()[:1],
        )
        print("✅ Test 2 passed: Handled invalid filename characters")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    # Test 3: Missing similar image data
    try:
        similar_results_no_images = [
            {
                "object_id": "missing_img_001",
                "weaviate_id": "uuid-missing-001",
                "timestamp": "2024-01-15T14:30:22Z",
                "_additional": {"distance": 0.5},
            }
        ]

        results_folder = saver.save_complete_results(
            image_name="missing_images_test",
            original_image=create_test_image(text="Missing Images Test"),
            reid_results=create_sample_reid_results()[:1],
            similar_results=similar_results_no_images,
        )
        print("✅ Test 3 passed: Handled missing image data")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")

    print(f"📊 Edge cases: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def main():
    """Run all tests"""
    print("🔬 ReID Results Saver Test Suite")
    print("=" * 60)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Similar Images", test_with_similar_images),
        ("Integration Function", test_integration_function),
        ("Results Summary", test_results_summary),
        ("Edge Cases", test_edge_cases),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} encountered an error: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")

    print(f"\n🎯 Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Results saver is working correctly.")
        print("\n📁 Check the following folders for saved results:")
        for folder in [
            "test_results_basic",
            "test_results_similar",
            "test_results_integration",
            "test_results_edge_cases",
        ]:
            if os.path.exists(folder):
                print(f"   - {folder}/")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
