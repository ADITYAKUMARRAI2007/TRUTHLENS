# -*- coding: utf-8 -*-
"""
Simplified AI-content detection for Images, PDFs and Video/Audio.

This version is optimized for reliability and simplicity.
"""

import os
import sys
import cv2
import math
import json
import tempfile
import subprocess
from typing import Optional, Dict, Any, Set

import torch
import numpy as np
import librosa
from PIL import Image, ImageOps, ImageFile

# ───── Small-instance tuning ──────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

# ───── File type helpers ─────────────────────────────────────────────────────
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
PDF_EXTS:   Set[str] = {".pdf"}

def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def is_image_path(path: str) -> bool: return _ext(path) in IMAGE_EXTS
def is_video_path(path: str) -> bool: return _ext(path) in VIDEO_EXTS
def is_pdf_path(path: str)   -> bool: return _ext(path) in PDF_EXTS

# ───── Simple Detection Functions ────────────────────────────────────────────

def detect_image_simple(path: str) -> Dict[str, Any]:
    """Simple image detection using basic heuristics"""
    try:
        img = Image.open(path)
        img_array = np.array(img)
        
        # Basic heuristics for AI-generated images
        # 1. Check for unusual patterns in pixel distribution
        # 2. Look for artifacts that are common in AI-generated images
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate basic statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Simple heuristic: AI images often have more uniform pixel distributions
        uniformity_score = 1.0 - (std_val / 255.0)
        
        # Edge detection for artifact analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine scores
        ai_probability = (uniformity_score * 0.6 + (1 - edge_density) * 0.4)
        ai_probability = max(0.0, min(1.0, ai_probability))
        
        return {
            "result": "FAKE" if ai_probability > 0.7 else "REAL" if ai_probability < 0.3 else "UNCERTAIN",
            "ai_probability": float(ai_probability),
            "image_score": float(ai_probability),
            "signals": {
                "uniformity_score": float(uniformity_score),
                "edge_density": float(edge_density),
                "mean_pixel_value": float(mean_val),
                "std_pixel_value": float(std_val)
            }
        }
    except Exception as e:
        return {
            "result": "UNKNOWN",
            "ai_probability": 0.5,
            "image_score": 0.5,
            "error": f"image_detection_failed: {str(e)}"
        }

def detect_video_simple(path: str) -> Dict[str, Any]:
    """Simple video detection using basic heuristics"""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {
                "result": "UNKNOWN",
                "ai_probability": 0.5,
                "video_score": 0.5,
                "error": "video_cannot_be_opened"
            }
        
        frame_count = 0
        total_score = 0.0
        scores = []
        
        # Analyze first 30 frames or all frames if video is shorter
        max_frames = min(30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Basic analysis similar to image detection
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            uniformity_score = 1.0 - (std_val / 255.0)
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Frame score
            frame_score = (uniformity_score * 0.6 + (1 - edge_density) * 0.4)
            scores.append(frame_score)
            total_score += frame_score
            frame_count += 1
        
        cap.release()
        
        if frame_count == 0:
            return {
                "result": "UNKNOWN",
                "ai_probability": 0.5,
                "video_score": 0.5,
                "error": "no_frames_analyzed"
            }
        
        # Calculate average score
        avg_score = total_score / frame_count
        
        return {
            "result": "FAKE" if avg_score > 0.7 else "REAL" if avg_score < 0.3 else "UNCERTAIN",
            "ai_probability": float(avg_score),
            "video_score": float(avg_score),
            "signals": {
                "frames_analyzed": frame_count,
                "avg_uniformity": float(np.mean([s for s in scores])),
                "score_variance": float(np.var(scores))
            }
        }
    except Exception as e:
        return {
            "result": "UNKNOWN",
            "ai_probability": 0.5,
            "video_score": 0.5,
            "error": f"video_detection_failed: {str(e)}"
        }

def detect_audio_simple(path: str) -> Optional[float]:
    """Simple audio detection using basic heuristics"""
    try:
        # Load audio file
        y, sr = librosa.load(path, sr=None)
        
        # Basic audio analysis
        # 1. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # 2. MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 3. Calculate statistics
        centroid_mean = np.mean(spectral_centroids)
        centroid_std = np.std(spectral_centroids)
        rolloff_mean = np.mean(spectral_rolloff)
        mfcc_mean = np.mean(mfccs)
        mfcc_std = np.std(mfccs)
        
        # Simple heuristic: AI-generated audio often has more uniform spectral characteristics
        uniformity_score = 1.0 - (centroid_std / centroid_mean if centroid_mean > 0 else 0)
        uniformity_score = max(0.0, min(1.0, uniformity_score))
        
        return float(uniformity_score)
    except Exception as e:
        print(f"Audio detection failed: {e}")
        return None

def detect_pdf_simple(path: str) -> Dict[str, Any]:
    """Simple PDF detection"""
    try:
        # For now, return a basic result
        # In a real implementation, you would analyze PDF content
        return {
            "result": "UNCERTAIN",
            "ai_probability": 0.5,
            "error": "pdf_analysis_not_implemented"
        }
    except Exception as e:
        return {
            "result": "UNKNOWN",
            "ai_probability": 0.5,
            "error": f"pdf_detection_failed: {str(e)}"
        }

# ───── Unified entrypoint ────────────────────────────────────────────────────
def detect_ai_content(path: str) -> dict:
    """
    Routes images, PDFs, and videos to the right detector.
    Always returns JSON; never throws so /analyze won't 500.
    """
    if is_image_path(path):
        try:
            result = detect_image_simple(path)
            return {
                "result": result["result"],
                "video_score": None,
                "audio_score": None,
                "image_score": result.get("ai_probability"),
                "ai_probability": result.get("ai_probability"),
                "signals": result.get("signals", {}),
            }
        except Exception as e:
            return {
                "result": "UNKNOWN",
                "video_score": None,
                "audio_score": None,
                "image_score": None,
                "ai_probability": None,
                "error": f"image_pathway_failed: {e.__class__.__name__}",
            }

    if is_pdf_path(path):
        try:
            return detect_pdf_simple(path)
        except Exception as e:
            return {
                "result": "UNKNOWN",
                "ai_probability": None,
                "error": f"pdf_pathway_failed: {e.__class__.__name__}",
            }

    # Video/Audio pathway
    try:
        video_result = detect_video_simple(path)
        video_score = video_result.get("ai_probability", 0.5)
    except Exception as e:
        return {
            "result": "UNKNOWN",
            "video_score": None,
            "audio_score": None,
            "ai_probability": None,
            "error": f"video_model_failed: {e.__class__.__name__}",
        }

    # Try audio analysis
    audio_score = None
    try:
        audio_score = detect_audio_simple(path)
    except Exception as e:
        # non-fatal; keep going with video only
        audio_score = None

    # Combine video and audio scores
    if audio_score is None:
        final_score = video_score
    else:
        # Weight video more heavily than audio
        final_score = 0.7 * video_score + 0.3 * audio_score

    # Determine result
    if final_score > 0.7:
        label = "FAKE"
    elif final_score < 0.3:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    return {
        "result": label,
        "video_score": float(video_score) if video_score is not None else None,
        "audio_score": float(audio_score) if audio_score is not None else None,
        "ai_probability": float(final_score),
    }

# ───── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m deepfake_detection_agent.backend.services.detection <media_path>")
        sys.exit(1)
    print(json.dumps(detect_ai_content(sys.argv[1]), indent=2))