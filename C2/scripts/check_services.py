#!/usr/bin/env python3
"""
Quick connection test script for ReID system services
"""
import redis
import requests
import json
import sys

def test_redis_connection(host='redis', port=6379):
    """Test Redis connection"""
    print(f"Testing Redis connection to {host}:{port}...")
    try:
        client = redis.Redis(host=host, port=port, socket_timeout=5)
        client.ping()
        print(f"✓ Redis connected successfully")
        
        # Test queue operations
        test_queue = 'test_connection_queue'
        client.lpush(test_queue, json.dumps({'test': 'data'}))
        data = client.lpop(test_queue)
        if data:
            print(f"✓ Redis queue operations working")
        return True
        
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False

def test_weaviate_connection(url='weaviate_db'):
    """Test Weaviate connection"""
    # Format URL
    if not url.startswith('http'):
        url = f"http://{url}:8080"
    elif ':' not in url.replace('http://', '').replace('https://', ''):
        url += ':8080'
    
    print(f"Testing Weaviate connection to {url}...")
    try:
        # Test basic connectivity
        response = requests.get(f"{url}/v1/meta", timeout=10)
        if response.status_code == 200:
            print(f"✓ Weaviate connected successfully")
            meta = response.json()
            print(f"  Version: {meta.get('version', 'unknown')}")
            return True
        else:
            print(f"✗ Weaviate returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Weaviate connection failed: {e}")
        return False

def test_system_readiness():
    """Test overall system readiness"""
    print("=" * 60)
    print("SYSTEM READINESS CHECK")
    print("=" * 60)
    
    redis_ok = test_redis_connection('redis', 6379)
    weaviate_ok = test_weaviate_connection('weaviate_db')
    
    print("\n" + "=" * 60)
    if redis_ok and weaviate_ok:
        print("✓ ALL SYSTEMS READY")
        print("\nYou can now run:")
        print("python main.py --redis_host redis --weaviate_url weaviate_db --use_nfc --save_results")
        return True
    else:
        print("✗ SYSTEM NOT READY")
        print("\nIssues found:")
        if not redis_ok:
            print("- Redis service not accessible")
        if not weaviate_ok:
            print("- Weaviate service not accessible")
        return False

if __name__ == "__main__":
    success = test_system_readiness()
    sys.exit(0 if success else 1)
