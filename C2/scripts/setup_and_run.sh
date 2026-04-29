#!/bin/bash

# Enhanced C2 ReID System Setup and Run Script
# This script fixes all dependencies and starts the ReID system

set -e  # Exit on any error

echo "🚀 Enhanced C2 ReID System Setup"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Step 1: Check current environment
print_info "Step 1: Checking environment..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Available GPU: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Step 2: Install missing dependencies
print_info "Step 2: Installing missing dependencies..."
pip install --no-cache-dir yacs thop timm einops omegaconf scikit-learn matplotlib

# Fix Weaviate client version
print_info "Step 3: Fixing Weaviate client..."
pip uninstall weaviate-client -y 2>/dev/null || true
pip install "weaviate-client>=3.26.7,<4.0.0"

# Step 4: Set up Python paths
print_info "Step 4: Setting up Python paths..."
export PYTHONPATH="/app/Pose2ID:/app/Pose2ID/IPG:/app/Pose2ID/IPG/reidmodel:/app:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Step 5: Verify file structure
print_info "Step 5: Verifying file structure..."
if [ -f "/app/Pose2ID/NFC.py" ]; then
    print_status "NFC.py found"
else
    print_error "NFC.py not found at /app/Pose2ID/NFC.py"
    ls -la /app/Pose2ID/ || echo "Pose2ID directory not found"
fi

if [ -f "/app/Pose2ID/ID2.py" ]; then
    print_status "ID2.py found"
else
    print_error "ID2.py not found at /app/Pose2ID/ID2.py"
fi

if [ -f "/app/utils/weaviate.py" ]; then
    print_status "weaviate.py found"
else
    print_error "weaviate.py not found"
fi

# Step 6: Test imports
print_info "Step 6: Testing imports..."
python -c "
import sys
sys.path.insert(0, '/app/Pose2ID')
try:
    from NFC import NFC
    print('✅ NFC import successful')
except ImportError as e:
    print('❌ NFC import failed:', e)
    exit(1)

try:
    from ID2 import ID2
    print('✅ ID2 import successful')
except ImportError as e:
    print('❌ ID2 import failed:', e)
    exit(1)

try:
    import torch
    print('✅ PyTorch available')
    print('   CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print('❌ PyTorch import failed:', e)
    exit(1)

try:
    import redis
    print('✅ Redis client available')
except ImportError as e:
    print('❌ Redis import failed:', e)
    exit(1)

print('✅ All critical imports successful!')
"

if [ $? -ne 0 ]; then
    print_error "Import test failed. Please check the errors above."
    exit 1
fi

# Step 7: Test service connections
print_info "Step 7: Testing service connections..."

# Test Redis connection
echo "Testing Redis connection..."
python -c "
import redis
try:
    r = redis.Redis(host='redis', port=6379, socket_connect_timeout=5)
    r.ping()
    print('✅ Redis connection successful')
    print('   Queue size:', r.llen('object_queue'))
except Exception as e:
    print('⚠️ Redis connection failed:', e)
    print('   Make sure Redis container is running')
"

# Test Weaviate connection
echo "Testing Weaviate connection..."
python -c "
try:
    import weaviate
    try:
        client = weaviate.Client('http://weaviate_db:8080', timeout_config=(5, 15))
        if client.is_ready():
            print('✅ Weaviate connection successful')
        else:
            print('⚠️ Weaviate not ready')
    except Exception as e:
        print('⚠️ Weaviate connection failed:', e)
        print('   Make sure Weaviate container is running')
except ImportError:
    print('⚠️ Weaviate client not installed')
"

# Step 8: Create run configuration
print_info "Step 8: Creating run configurations..."

# Create different run modes
cat > /app/run_basic.sh << 'EOF'
#!/bin/bash
export PYTHONPATH="/app/Pose2ID:/app/Pose2ID/IPG:/app/Pose2ID/IPG/reidmodel:/app:$PYTHONPATH"
python main.py --redis_host redis --weaviate_url http://weaviate_db:8080
EOF

cat > /app/run_full.sh << 'EOF'
#!/bin/bash
export PYTHONPATH="/app/Pose2ID:/app/Pose2ID/IPG:/app/Pose2ID/IPG/reidmodel:/app:$PYTHONPATH"
python main.py \
    --redis_host redis \
    --redis_port 6379 \
    --queue_name object_queue \
    --weaviate_url http://weaviate_db:8080 \
    --use_nfc \
    --similarity_threshold 0.7 \
    --save_results
EOF

cat > /app/run_debug.sh << 'EOF'
#!/bin/bash
export PYTHONPATH="/app/Pose2ID:/app/Pose2ID/IPG:/app/Pose2ID/IPG/reidmodel:/app:$PYTHONPATH"
python -u main.py \
    --redis_host redis \
    --weaviate_url http://weaviate_db:8080 \
    --use_nfc \
    --save_results 2>&1 | tee debug_output.log
EOF

# Make scripts executable
chmod +x /app/run_*.sh

print_status "Run scripts created:"
print_info "  - /app/run_basic.sh   (basic functionality)"
print_info "  - /app/run_full.sh    (all features enabled)"
print_info "  - /app/run_debug.sh   (debug mode with logging)"

# Step 9: Create a minimal test script
print_info "Step 9: Creating test script..."
cat > /app/test_system.py << 'EOF'
#!/usr/bin/env python
"""
Minimal system test script
"""
import sys
import os

# Set up paths
sys.path.insert(0, '/app/Pose2ID')

def test_imports():
    print("🧪 Testing imports...")
    try:
        from NFC import NFC
        from ID2 import ID2
        import torch
        import redis
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_redis():
    print("🧪 Testing Redis...")
    try:
        import redis
        r = redis.Redis(host='redis', port=6379, socket_connect_timeout=5)
        r.ping()
        queue_size = r.llen('object_queue')
        print(f"✅ Redis connected, queue size: {queue_size}")
        return True
    except Exception as e:
        print(f"❌ Redis failed: {e}")
        return False

def test_weaviate():
    print("🧪 Testing Weaviate...")
    try:
        import weaviate
        client = weaviate.Client('http://weaviate_db:8080', timeout_config=(5, 15))
        if client.is_ready():
            print("✅ Weaviate connected")
            return True
        else:
            print("⚠️ Weaviate not ready")
            return False
    except Exception as e:
        print(f"❌ Weaviate failed: {e}")
        return False

def main():
    print("🔬 System Test Suite")
    print("==================")

    tests = [
        ("Imports", test_imports),
        ("Redis", test_redis),
        ("Weaviate", test_weaviate)
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
        print()

    print("📊 Test Results:")
    print("================")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:15} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("💡 Try running: ./run_basic.sh")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
EOF

chmod +x /app/test_system.py

# Step 10: Run system test
print_info "Step 10: Running system test..."
python /app/test_system.py

# Step 11: Final instructions
echo
print_status "Setup Complete!"
echo "================="
print_info "Available commands:"
print_info "  ./run_basic.sh     - Start basic ReID processing"
print_info "  ./run_full.sh      - Start with all features"
print_info "  ./run_debug.sh     - Start with debug logging"
print_info "  python test_system.py - Run system tests"
echo
print_info "Direct command (if you prefer):"
echo 'export PYTHONPATH="/app/Pose2ID:/app/Pose2ID/IPG:/app/Pose2ID/IPG/reidmodel:/app:$PYTHONPATH"'
echo "python main.py --redis_host redis --weaviate_url http://weaviate_db:8080 --use_nfc --save_results"
echo

# Parse command line arguments for auto-run
if [ "$1" = "--run" ]; then
    mode="${2:-basic}"
    print_info "Auto-starting in $mode mode..."
    case $mode in
        "basic")
            ./run_basic.sh
            ;;
        "full")
            ./run_full.sh
            ;;
        "debug")
            ./run_debug.sh
            ;;
        *)
            print_warning "Unknown mode: $mode. Available: basic, full, debug"
            print_info "Starting in basic mode..."
            ./run_basic.sh
            ;;
    esac
else
    print_warning "Setup complete. Run with:"
    print_info "  ./setup_and_run.sh --run basic"
    print_info "  ./setup_and_run.sh --run full"
    print_info "  ./setup_and_run.sh --run debug"
fi
