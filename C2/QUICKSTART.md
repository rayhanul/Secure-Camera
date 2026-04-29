# Quick Start Guide - ReID System with External Services

## Your Current Command

You're using:
```bash
python main.py --redis_host redis --weaviate_url weaviate_db --use_nfc --save_results
```

## What This Command Does

✅ **Connects to Redis** at host `redis:6379`  
✅ **Connects to Weaviate** at `http://weaviate_db:8080`  
✅ **Enables ReID features** (default behavior)  
✅ **Enables Weaviate storage** (default behavior)  
✅ **Applies NFC processing** to improve feature quality  
✅ **Saves results** to JSON files with timestamps  

## Pre-Flight Checklist

### 1. Test Service Connections
```bash
# Test Redis and Weaviate connectivity
python check_services.py
```

Expected output:
```
SYSTEM READINESS CHECK
============================================================
Testing Redis connection to redis:6379...
✓ Redis connected successfully
✓ Redis queue operations working
Testing Weaviate connection to http://weaviate_db:8080...
✓ Weaviate connected successfully
  Version: 1.xx.x
============================================================
✓ ALL SYSTEMS READY
```

### 2. Generate Test Data (Optional)
```bash
# Send a single test detection
python generate_test_data.py --single --camera cam_01 --persons 3

# Or simulate continuous stream for 30 seconds
python generate_test_data.py --duration 30 --fps 2
```

## Running the System

### Full Command with All Options
```bash
python main.py \
  --redis_host redis \
  --weaviate_url weaviate_db \
  --use_nfc \
  --save_results \
  --similarity_threshold 0.7 \
  --queue_name object_queue
```

### Alternative Configurations

**High Precision Mode** (fewer false matches):
```bash
python main.py --redis_host redis --weaviate_url weaviate_db --similarity_threshold 0.85
```

**Performance Mode** (no NFC, no saving):
```bash
python main.py --redis_host redis --weaviate_url weaviate_db
```

**ReID Only** (no persistent storage):
```bash
python main.py --redis_host redis --no_weaviate --use_nfc
```

**Weaviate Only** (use pre-extracted features):
```bash
python main.py --redis_host redis --weaviate_url weaviate_db --no_reid
```

## Expected Output

When running successfully, you should see:

```
SYSTEM STATUS CHECK
============================================================
Redis: ✓ CONNECTED (Queue size: 0)
ReID Model: ⚠ FALLBACK MODE (Device: cpu)
Weaviate: ✓ CONNECTED (Person counter: 0)

Configuration:
  - Redis: redis:6379 (queue: object_queue)
  - Weaviate: weaviate_db
  - Similarity threshold: 0.7
  - NFC enabled: True
  - Save results: True
============================================================
Enhanced C2 processor initialized with: ReID, Weaviate

Starting continuous processing...
Press Ctrl+C to stop

Processing frame 1 from cam_01
Raw features shape: torch.Size([2, 2048])
ReID features shape: torch.Size([2, 2048])
NFC applied

ReID Results (2 persons detected):
--------------------------------------------------------------------------------
NEW | ID: person_000001 | Conf: 1.000 | Camera: cam_01 | Similar: 0
NEW | ID: person_000002 | Conf: 1.000 | Camera: cam_01 | Similar: 0
--------------------------------------------------------------------------------
```

## Troubleshooting

### Redis Connection Issues
- Ensure Redis service is running and accessible at `redis:6379`
- Check network connectivity between containers
- Verify queue name matches between producer and consumer

### Weaviate Connection Issues  
- Ensure Weaviate is running at `http://weaviate_db:8080`
- Check Weaviate logs for initialization errors
- Verify Weaviate schema is created properly

### ReID Model Issues
- System falls back to feature normalization if TransReID model fails to load
- This is normal for the first run - features will still work for similarity matching
- For full ReID functionality, ensure model files are in `./Pose2ID/IPG/pretrained/`

### No Data Received
- Check if data is being sent to the Redis queue
- Use the test data generator to send sample data
- Verify queue name matches in both producer and consumer

## Performance Monitoring

The system displays statistics every 10 frames:

```
STATISTICS (Runtime: 125.3s):
  Frames processed: 100 (0.80 FPS)
  Persons detected: 247
  New persons: 23
  Existing persons: 224
--------------------------------------------------------------------------------
```

## Output Files

With `--save_results`, the system creates files like:
- `reid_results_20251006_143022.json`

These contain detailed tracking results for analysis.

## Docker Integration

If running in Docker, ensure your docker-compose.yml has proper service names:

```yaml
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  weaviate_db:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
  
  reid_system:
    build: .
    depends_on:
      - redis
      - weaviate_db
    command: python main.py --redis_host redis --weaviate_url weaviate_db --use_nfc --save_results
```

Your command should work perfectly with this setup! 🚀
