#!/usr/bin/env python3
"""
Test Gmail configuration locally
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=== Gmail Configuration Test ===")
print(f"ADMIN_EMAIL: {os.getenv('ADMIN_EMAIL')}")
print(f"GMAIL_SENDER: {os.getenv('GMAIL_SENDER')}")
print(f"GOOGLE_CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID')[:20] if os.getenv('GOOGLE_CLIENT_ID') else 'None'}...")
print(f"GOOGLE_CLIENT_SECRET: {os.getenv('GOOGLE_CLIENT_SECRET')[:10] if os.getenv('GOOGLE_CLIENT_SECRET') else 'None'}...")
print(f"GMAIL_REFRESH_TOKEN: {os.getenv('GMAIL_REFRESH_TOKEN')[:20] if os.getenv('GMAIL_REFRESH_TOKEN') else 'None'}...")

# Test Gmail imports
try:
    from email.mime.text import MIMEText
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    import io
    print("✅ Gmail libraries imported successfully")
    
    # Test creating credentials
    refresh_token = os.getenv('GMAIL_REFRESH_TOKEN')
    if refresh_token:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET')
        )
        print("✅ Gmail credentials created successfully")
        
        # Test building service
        service = build('gmail', 'v1', credentials=creds)
        print("✅ Gmail service built successfully")
        
    else:
        print("❌ GMAIL_REFRESH_TOKEN not found")
        
except Exception as e:
    print(f"❌ Gmail test failed: {e}")
    import traceback
    traceback.print_exc()
