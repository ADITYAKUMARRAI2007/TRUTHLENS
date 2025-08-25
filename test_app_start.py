#!/usr/bin/env python3
"""
Test if the app can start locally
"""

import sys
import os
sys.path.insert(0, 'deepfake_detection_agent')

try:
    print("Testing app import...")
    from app import app
    print("✅ App imported successfully")
    
    print("Testing app attributes...")
    print(f"App title: {app.title}")
    print(f"App version: {app.version}")
    
    print("Testing routes...")
    routes = [route.path for route in app.routes]
    print(f"Found {len(routes)} routes")
    
    print("✅ App is ready to start!")
    
except Exception as e:
    print(f"❌ App test failed: {e}")
    import traceback
    traceback.print_exc()
