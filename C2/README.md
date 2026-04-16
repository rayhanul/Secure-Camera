# Enhanced ReID + Weaviate System

A complete Person Re-Identification system that combines deep learning ReID features with Weaviate vector database for persistent identity tracking across multiple cameras.

## Features

 - **TransReID Model Integration** - Advanced transformer-based ReID feature extraction
 - **Weaviate Vector Database** - Persistent storage and similarity search
 - **Cross-Camera Tracking** - Track persons across multiple camera feeds
 - **Real-time Processing** - Redis queue-based processing pipeline
 - **Modular Design** - Enable/disable ReID or Weaviate independently
 - **NFC Processing** - Optional Nearest Feature Center enhancement
 - **Runtime Statistics** - Performance monitoring and analytics

## System Architecture

```
Redis Queue → C2 Processor → ReID Features → Weaviate DB → Person Identity
     ↑              ↓              ↓              ↓              ↓
 Detection      Raw Objects   Feature Vectors  Similarity    Tracking
   Data        (Images/bbox)                    Search       Results
```

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Redis Server**
   ```bash
   # Ubuntu/Debian
   sudo apt install redis-server
   sudo systemctl start redis-server

   # macOS with Homebrew
   brew install redis
   brew services start redis
   ```

3. **Weaviate Server**
   ```bash
   # Using Docker (recommended)
   docker run -p 8080:8080 semitechnologies/weaviate:latest

   # Or use docker-compose (see docker-compose.yml in repo)
   docker-compose up weaviate
   ```

4. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Command Line Options

```bash
python main.py [OPTIONS]
```

**Options:**
- `--redis_host HOST`
  Redis server host (default: `localhost`)
- `--redis_port PORT`
  Redis server port (default: `6379`)
- `--queue_name QUEUE`
  Redis queue name (default: `object_queue`)
- `--weaviate_url URL`
  Weaviate server URL (default: `http://localhost:8080`)
- `--use_nfc`
  Apply NFC processing to features (not recommended; Weaviate already handles neighbor search)
- `--similarity_threshold THR`
  Similarity threshold for person matching (default: `0.7`)
- `--save_results`
  Save processing results to JSON files

**Note:**
There are no `--use_reid` or `--use_weaviate` flags in the current main.py. Both ReID and Weaviate are always used if configured.

### Usage Examples

### Application Startup Examples

1. **Standard Full System (ReID + Weaviate, default settings)**
   ```bash
   python main.py
   ```

2. **Custom Redis/Weaviate Hosts**
   ```bash
   python main.py --redis_host 192.168.1.100 --weaviate_url http://192.168.1.101:8080
   ```

3. **Change Similarity Threshold**
   ```bash
   python main.py --similarity_threshold 0.8
   ```

4. **Enable NFC (not recommended)**
   ```bash
   python main.py --use_nfc
   ```

5. **Save Results to JSON**
   ```bash
   python main.py --save_results
   ```

You can combine options as needed:
```bash
python main.py --redis_host redis --weaviate_url http://weaviate_db:8080 --similarity_threshold 0.75 --save_results
```

## Data Format

### Input Data (Redis Queue)

The system expects detection data in this format:

```json
{
  "metadata": {
    "frame_id": 12345,
    "camera_id": "cam_01",
    "timestamp": "2025-10-06T10:30:00",
    "video_path": "/path/to/video.mp4"
  },
  "objects": [
    {
      "object_id": "person_1_12345",
      "class_name": "person",
      "bbox": [100, 150, 200, 400],
      "confidence": 0.95,
      "features": [0.1, 0.2, ...] // Optional pre-extracted features
    }
  ]
}
```

### Output Results

```json
{
  "detection_id": "person_1_12345",
  "person_id": "person_000042",
  "confidence": 0.87,
  "is_new_person": false,
  "similar_detections": 5,
  "bbox": [100, 150, 200, 400],
  "camera_id": "cam_01",
  "frame_id": 12345,
  "timestamp": "2025-10-06T10:30:00",
  "cross_camera_matches": [...],
  "weaviate_id": "uuid-string"
}
```

## System Status

The system provides comprehensive status monitoring:

```
SYSTEM STATUS CHECK
============================================================
Redis: ✓ CONNECTED (Queue size: 42)
ReID Model: ✓ LOADED (Device: cuda)
Weaviate: ✓ CONNECTED (Person counter: 157)

Configuration:
  - Similarity threshold: 0.7
  - NFC enabled: False
  - Save results: True
============================================================
```

## Performance Monitoring

Runtime statistics are displayed every 10 frames:

```
STATISTICS (Runtime: 125.3s):
  Frames processed: 1250 (9.97 FPS)
  Persons detected: 3847
  New persons: 234
  Existing persons: 3613
--------------------------------------------------------------------------------
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check if Redis is running
   redis-cli ping
   # Should return: PONG
   ```

2. **Weaviate Connection Failed**
   ```bash
   # Check if Weaviate is running
   curl http://localhost:8080/v1/meta
   # Should return JSON metadata
   ```

3. **ReID Model Loading Error**
   - Ensure model files are in `./Pose2ID/IPG/pretrained/`
   - Check CUDA availability for GPU acceleration
   - System falls back to feature passthrough if model fails

4. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install torch torchvision redis-py weaviate-client numpy
   ```

### Testing

Run the test script to verify system functionality:

```bash
python test_system.py
```

This will test:
- Basic module imports
- Mock data processing
- Component initialization
- System status checks

## Architecture Details

### Components

1. **TransReIDProcessor**
   - Loads transformer-based ReID model
   - Extracts 2048-dimensional feature vectors
   - GPU acceleration when available
   - Fallback to feature normalization

2. **WeaviateReIDManager**
   - Vector similarity search
   - Person identity assignment
   - Cross-camera matching
   - Persistent storage

3. **EnhancedC2Processor**
   - Main processing pipeline
   - Redis queue management
   - Component orchestration
   - Statistics tracking

### Data Flow

1. **Detection Input** → Redis queue receives detection data
2. **Feature Extraction** → TransReID processes person crops
3. **Similarity Search** → Weaviate finds similar persons
4. **Identity Assignment** → New ID or match existing person
5. **Storage** → Update Weaviate with new data
6. **Results Output** → Return tracking results

## API Integration

For integration with other systems:

```python
from main import EnhancedC2Processor

# Initialize processor
args = parse_args()
processor = EnhancedC2Processor(args)

# Process single detection
results = processor.process_detection(detection_data)

# Get person history
if processor.weaviate_manager:
    history = processor.weaviate_manager.get_person_history("person_000042")
```

## License

This project extends the Pose2ID framework. See LICENSE files for details.
