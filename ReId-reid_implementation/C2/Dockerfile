FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Set working Dir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Dev only
RUN  apt update -y && apt install vim -y

# Install Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . /app/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs

# Set Python path for Pose2ID
ENV PYTHONPATH="${PYTHONPATH}:/app/Pose2ID:/app/Pose2ID/IPG/reidmodel"

# Default command - can be overridden in docker-compose
CMD ["python", "enhanced_c2_weaviate.py", "--redis_host", "redis", "--weaviate_url", "http://weaviate_db:8080", "--use_nfc"]
