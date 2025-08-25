#!/usr/bin/env python3
"""
Production startup script for TruthLens backend on Render.
This script ensures proper environment setup and starts the FastAPI app.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup production environment variables and directories."""
    try:
        # Ensure output directory exists
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir.absolute()}")
        
        # Ensure jobs directory exists
        jobs_dir = output_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Jobs directory ready: {jobs_dir.absolute()}")
        
        # Set default environment variables if not present
        if not os.getenv("APP_SECRET"):
            os.environ["APP_SECRET"] = "render-production-secret-2024"
            logger.info("Set default APP_SECRET")
            
        if not os.getenv("MAX_UPLOAD_MB"):
            os.environ["MAX_UPLOAD_MB"] = "100"
            logger.info("Set default MAX_UPLOAD_MB to 100")
            
        # Log environment info
        logger.info(f"PORT: {os.getenv('PORT', '8001')}")
        logger.info(f"FRONTEND_ORIGIN: {os.getenv('FRONTEND_ORIGIN', 'Not set')}")
        logger.info(f"MAX_UPLOAD_MB: {os.getenv('MAX_UPLOAD_MB', '50')}")
        
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        raise

def main():
    """Main startup function."""
    try:
        logger.info("🚀 Starting TruthLens Backend...")
        
        # Setup environment
        setup_environment()
        
        # Import and start the app
        from app import app
        logger.info("✅ App imported successfully")
        
        # Get port from environment
        port = int(os.getenv("PORT", "8001"))
        
        logger.info(f"🎯 Starting server on port {port}")
        logger.info(f"📱 App title: {app.title}")
        logger.info(f"🔢 App version: {app.version}")
        
        # Import uvicorn and run
        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
