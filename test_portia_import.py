#!/usr/bin/env python3
"""
Test Portia import and environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=== Portia Import Test ===")
print(f"PORTIA_API_KEY: {os.getenv('PORTIA_API_KEY')}")
print(f"PORTIA_API_KEY length: {len(os.getenv('PORTIA_API_KEY', ''))}")

# Test Portia import
try:
    print("\n=== Testing Portia Import ===")
    from portia_agent import run_through_portia
    print("✅ Portia import successful")
    print(f"run_through_portia function: {run_through_portia}")
except Exception as e:
    print(f"❌ Portia import failed: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# Test if portia_agent.py exists
import os
portia_file = "portia_agent.py"
if os.path.exists(portia_file):
    print(f"\n✅ {portia_file} exists")
    print(f"File size: {os.path.getsize(portia_file)} bytes")
else:
    print(f"\n❌ {portia_file} not found")
