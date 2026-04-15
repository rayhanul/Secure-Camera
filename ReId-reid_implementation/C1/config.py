class Config:
    # Redis Configuration
    REDIS_HOST = 'redis'
    REDIS_PORT = 6379
    REDIS_QUEUE_NAME = 'reid_queue'

    # Image processing
    IMAGE_SIZE = (256, 128)  # Height, Width
    BATCH_SIZE = 32

    # Camera Configuration
    CAMERA_FPS = 30
    CAMERA_RESOLUTION = (640, 480)  # Width, Height