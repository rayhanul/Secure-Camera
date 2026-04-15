#### Live Camera Processing

```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  --network reid_network \
  -v $(pwd)/logs:/app/logs \
  c1_processor \
  python main.py --use_camera --camera_id 0
```

#### Video File Processing
```bash
docker run --rm \
  --network reid_network \
  -v $(pwd)/videos:/data \
  -v $(pwd)/logs:/app/logs \
  c1_processor \
  python main.py --video_path /data/sample_video.mp4
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--camera_id` | Camera device ID | 0 |
| `--redis_host` | Redis server hostname | localhost |
| `--redis_port` | Redis server port | 6379 |
| `--use_camera` | Use live camera feed | False |
| `--video_path` | Path to video file | '' |
| `--model_path` | Path to YOLO model weights | /models/yolov8n.pt |
| `--log_level` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--log_file` | Custom log file path | /app/logs/c1_processor.log |

### Examples

#### Process Live Camera Feed
```bash
python main.py --use_camera --camera_id 0 --redis_host redis
```

#### Process Video File
```bash
python main.py --video_path --redis-host=redis /data/surveillance_video.mp4
```

#### Debug Mode with Verbose Logging
```bash
python main.py --use_camera --log_level DEBUG
```

## Configuration

### Environment Variables
```bash
# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# YOLO Model Configuration
MODEL_PATH=/models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.5

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=/app/logs/c1_processor.log
```

## Data Flow

### Input Processing
1. **Frame Capture**: From camera or video file
2. **Object Detection**: YOLO detects persons and vehicles
3. **Object Cropping**: Extract detected objects from frame
4. **Preprocessing**: Apply ReID-specific transforms
5. **Queue Transmission**: Send to Redis for C2 processing

### Output Format
```json
{
  "objects": [
    {
      "object_id": 0,
      "bbox": [x1, y1, x2, y2],
      "class_name": "person",
      "confidence": 0.85,
      "processed_image": [tensor_data]
    }
  ],
  "metadata": {
    "frame_id": 123,
    "camera_id": 0,
    "timestamp": 1642678890.123,
    "object_count": 3
  }
}
```

## Logging

### Log Levels
- **DEBUG**: Detailed processing information
- **INFO**: General operational messages
- **WARNING**: Non-critical issues
- **ERROR**: Error conditions

### Log Access
```bash
# View Docker logs
docker logs c1_processor

# View persistent log files
tail -f logs/c1_processor.log

# Search for errors
grep "ERROR" logs/c1_processor.log
```

### Log Rotation
- Automatic rotation when files exceed 10MB
- Keeps 5 backup files
- Configurable via logging settings

## Performance Tuning

### GPU Acceleration
Ensure NVIDIA Docker runtime is installed:
```bash
# Check GPU availability
docker run --gpus all c1_processor python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Optimization
- Use smaller YOLO models for edge devices
- Adjust batch processing size
- Configure Redis memory limits

### Network Optimization
- Use local Redis instance
- Compress image data if bandwidth limited
- Batch multiple objects per Redis message

## Troubleshooting

### Common Issues

#### Camera Access
```bash
# Permission denied
sudo chmod 666 /dev/video0

# Camera busy
lsof /dev/video0
```

#### Redis Connection
```bash
# Test Redis connectivity
docker exec c1_processor redis-cli -h redis ping
```

#### YOLO Model Loading
```bash
# Check model file exists
docker exec c1_processor ls -la /models/

# Download manually
docker exec c1_processor wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O /models/yolov8n.pt
```

#### GPU Issues
```bash
# Check CUDA availability
docker run --gpus all c1_processor nvidia-smi

# Verify PyTorch CUDA
docker exec c1_processor python -c "import torch; print(torch.cuda.is_available())"
```

### Debug Mode
Enable verbose logging for troubleshooting:
```bash
python main.py --log_level DEBUG --use_camera
```

## Monitoring

### Health Check
```bash
# Container status
docker ps | grep c1_processor

# Redis queue status
redis-cli -h redis LLEN object_queue

# Processing statistics
tail -f logs/c1_processor.log | grep "Detected"
```

### Performance Metrics
- Frames per second (FPS)
- Detection accuracy
- Redis queue depth
- Memory usage
- GPU utilization

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py --redis_host localhost --use_camera
```

### Testing
```bash
# Test with sample video
python main.py --redis_host redis --video_path test_video.mp4

# Test Redis connection
python -c "import redis; r=redis.Redis(); print(r.ping())"
```
