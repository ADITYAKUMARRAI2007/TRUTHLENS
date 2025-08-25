"""
Simple fallback replay service for when the main reality replay fails.
This provides basic video stabilization and enhancement.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

def run_simple_replay(video_path: str) -> str:
    """
    Simple video enhancement using ffmpeg filters.
    This is a fallback when the main reality replay service fails.
    """
    try:
        # Create output path
        output_path = str(Path(video_path).parent / f"enhanced_{Path(video_path).name}")
        
        # Basic enhancement filters
        filters = [
            "scale=960:960:force_original_aspect_ratio=decrease",  # Scale down
            "fps=15",  # Reduce FPS for stability
            "unsharp=5:5:0.5:5:5:0.0",  # Sharpen
            "eq=contrast=1.1:brightness=0.05",  # Enhance contrast
            "hqdn3d=4:3:6:4.5",  # Denoise
        ]
        
        filter_string = ",".join(filters)
        
        # Run ffmpeg
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", filter_string,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            # If enhancement failed, return original
            return video_path
            
    except Exception as e:
        print(f"Simple replay failed: {e}")
        # Return original video if enhancement fails
        return video_path

def run_reality_replay(video_path: str) -> str:
    """
    Main entry point that tries the simple replay if available.
    This function signature matches what the main app expects.
    """
    return run_simple_replay(video_path)
