# -*- coding: utf-8 -*-
"""
Unified AI-content detection for Images, PDFs and Video/Audio.

Hardening for small instances:
- Lazy-load HF models on first use (no heavy work at import).
- Force CPU + single thread for stability.
- Prefer safetensors & low-memory options for Transformers.
- Throttle video frames (stride + max clips) to avoid long requests.
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

# ───── Transformers (import-only; from_pretrained happens lazily) ────────────
from transformers import (
    VideoMAEImageProcessor,
    AutoModelForVideoClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

# ───── Optional deps (guarded) ───────────────────────────────────────────────
def _optional_import(modname: str):
    try:
        return __import__(modname)
    except Exception:
        return None

timm        = _optional_import("timm")
c2pa        = _optional_import("c2pa")
exifread    = _optional_import("exifread")
iw          = _optional_import("invisible_watermark")
noiseprint  = _optional_import("noiseprint")
PyPDF2      = _optional_import("PyPDF2")
pdf2img_mod = _optional_import("pdf2image")
pytesseract = _optional_import("pytesseract")
convert_from_path = getattr(pdf2img_mod, "convert_from_path", None)

# transformers for text AI-detector (PDF), optional
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

# ───── File type helpers ─────────────────────────────────────────────────────
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
PDF_EXTS:   Set[str] = {".pdf"}

def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def is_image_path(path: str) -> bool: return _ext(path) in IMAGE_EXTS
def is_video_path(path: str) -> bool: return _ext(path) in VIDEO_EXTS
def is_pdf_path(path: str)   -> bool: return _ext(path) in PDF_EXTS

# ───── Image detection settings ──────────────────────────────────────────────
IMG_MODEL       = os.getenv("IMAGE_MODEL", "fatformer").lower()   # "fatformer" | "effb4"
FATFORMER_CKPT  = os.getenv("FATFORMER_CKPT")
EFFB4_CKPT      = os.getenv("EFFB4_CKPT")
USE_NOISEPRINT  = os.getenv("USE_NOISEPRINT", "0") == "1"
USE_C2PA        = os.getenv("USE_C2PA", "0") == "1"
USE_SD_WM       = os.getenv("USE_SD_WATERMARK", "0") == "1"

# ───── PDF detection settings ────────────────────────────────────────────────
PDF_DETECTOR_MODEL_NAME = os.getenv("PDF_TEXT_DETECTOR", "roberta-base-openai-detector")
_pdf_pipeline_cache = None  # lazy pipeline

# ───── Helpers: images ──────────────────────────────────────────────────────
def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _exif_features(path: str) -> Dict[str, Any]:
    out = {"has_camera_tags": False, "camera_make": None, "camera_model": None}
    if exifread is None:
        return out
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        make  = str(tags.get("Image Make", "") or "").strip()
        model = str(tags.get("Image Model", "") or "").strip()
        out["camera_make"]  = make or None
        out["camera_model"] = model or None
        out["has_camera_tags"] = bool(make or model)
    except Exception:
        pass
    return out

def _c2pa_ok(path: str) -> Optional[bool]:
    if not USE_C2PA or c2pa is None:
        return None
    try:
        res = c2pa.read_file(path)
        return bool(res and getattr(res, "active_manifest", None))
    except Exception:
        return None

def _sd_watermark_present(img: Image.Image) -> Optional[bool]:
    if not USE_SD_WM or iw is None:
        return None
    try:
        arr = np.array(img)
        try:
            pred = iw.decode(arr)  # API varies by version
            return bool(pred)
        except Exception:
            return None
    except Exception:
        return None

def _noiseprint_score(path: str) -> Optional[float]:
    """Return [0..1] ‘realness’ score from PRNU (higher → more camera-like)."""
    if not USE_NOISEPRINT or noiseprint is None:
        return None
    try:
        img = _load_image(path)
        arr = np.array(img)
        resid = noiseprint.estimate(arr)  # library-specific; best-effort
        if resid is None:
            return None
        e = float(np.abs(resid).mean())   # crude global energy proxy
        conf = 1.0 - math.exp(-3.0 * e)   # squash to 0..1
        return max(0.0, min(1.0, conf))
    except Exception:
        return None

# ───── Image classifiers (FatFormer/EfficientNet-B4) ────────────────────────
_fat_model = None
_eff_model = None

def _prep(img: Image.Image, size=384):
    img = ImageOps.fit(img, (size, size))
    arr = np.array(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    std  = np.array([0.229, 0.224, 0.225])[None, None, :]
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def _load_fatformer():
    global _fat_model
    if _fat_model is not None or timm is None:
        return _fat_model
    try:
        model = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=1)
        if FATFORMER_CKPT and os.path.isfile(FATFORMER_CKPT):
            state = torch.load(FATFORMER_CKPT, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        model.eval()
        _fat_model = model
    except Exception:
        _fat_model = None
    return _fat_model

def _load_effb4():
    global _eff_model
    if _eff_model is not None or timm is None:
        return _eff_model
    try:
        model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1)
        if EFFB4_CKPT and os.path.isfile(EFFB4_CKPT):
            state = torch.load(EFFB4_CKPT, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        _eff_model = model.eval()
    except Exception:
        _eff_model = None
    return _eff_model

@torch.inference_mode()
def _model_prob_fake(img: Image.Image) -> Optional[float]:
    if timm is None:
        return None
    x = _prep(img, size=384 if IMG_MODEL == "fatformer" else 380)
    m = _load_fatformer() if IMG_MODEL == "fatformer" else _load_effb4()
    if m is None:
        return None
    y = m(x)  # [1, 1] logit
    logit = float(y.flatten()[0])
    prob = 1.0 / (1.0 + math.exp(-logit))
    return float(prob)

def score_image_fake_probability(image_path: str) -> Dict[str, Any]:
    img = _load_image(image_path)

    p_cls = _model_prob_fake(img)
    p_real = _noiseprint_score(image_path)     # 0..1 (None if unavailable)
    p_fake_prnu = (1.0 - p_real) if (p_real is not None) else None

    c2 = _c2pa_ok(image_path)                  # True → likely real
    sd = _sd_watermark_present(img)            # True → likely SD-generated
    ex = _exif_features(image_path).get("has_camera_tags", False)

    prob_fake = p_cls if p_cls is not None else 0.5
    if p_fake_prnu is not None:
        prob_fake = 0.65 * prob_fake + 0.35 * p_fake_prnu
    if c2 is True:
        prob_fake = max(0.0, prob_fake - 0.25)
    if sd is True:
        prob_fake = min(1.0, prob_fake + 0.25)
    if ex is True:
        prob_fake = max(0.0, prob_fake - 0.05)

    return {
        "prob_fake": float(prob_fake),
        "signals": {
            "classifier": p_cls,
            "noiseprint_realness": p_real,
            "c2pa_present": c2,
            "sd_watermark": sd,
            "has_camera_exif": bool(ex),
        }
    }

# ───── PDF helpers & detection ───────────────────────────────────────────────
def _load_pdf_text(file_path: str) -> str:
    text = ""
    # 1) Selectable text
    if PyPDF2 is not None:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
        except Exception:
            pass

    # 2) OCR fallback
    if (not text.strip()) and (convert_from_path is not None) and (pytesseract is not None):
        try:
            pages = convert_from_path(file_path)  # requires poppler installed
            for img in pages:
                text += pytesseract.image_to_string(img) + "\n"
        except Exception:
            pass

    return text.strip()

def _pdf_metadata(file_path: str) -> Dict[str, Any]:
    meta = {}
    if PyPDF2 is None:
        return meta
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            meta = dict(reader.metadata or {})
    except Exception:
        pass
    return meta

def _analyze_pdf_metadata(meta: Dict[str, Any]) -> list:
    issues = []
    if not meta:
        issues.append("Missing metadata")
        return issues

    def has(k: str) -> bool:
        return (k in meta) and bool(str(meta.get(k) or "").strip())

    if not has("/Author"):
        issues.append("Missing author field")
    if not has("/Creator"):
        issues.append("Missing creator field")

    return issues

def _load_pdf_ai_pipeline():
    global _pdf_pipeline_cache
    if _pdf_pipeline_cache is not None:
        return _pdf_pipeline_cache
    if hf_pipeline is None:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(PDF_DETECTOR_MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            PDF_DETECTOR_MODEL_NAME,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        _pdf_pipeline_cache = hf_pipeline("text-classification", model=mdl, tokenizer=tok)
        return _pdf_pipeline_cache
    except Exception:
        return None

def _pdf_text_ai_scores(text: str) -> Dict[str, Any]:
    det = _load_pdf_ai_pipeline()
    if det is None or not text.strip():
        return {"fake_prob": None, "chunks": []}

    chunks = [text[i:i+800] for i in range(0, len(text), 800)]
    results, fake_probs = [], []

    for ch in chunks:
        try:
            out = det(ch)[0]  # {'label': ..., 'score': ...}
            label = str(out.get("label", "")).lower()
            score = float(out.get("score", 0.0))
            if "fake" in label or "generated" in label or label.endswith("1"):
                p_fake = score
            elif "real" in label or "human" in label or label.endswith("0"):
                p_fake = 1.0 - score
            else:
                p_fake = 0.5
            results.append({"label": out.get("label"), "score": score, "p_fake": p_fake})
            fake_probs.append(p_fake)
        except Exception:
            results.append({"label": "ERROR", "score": 0.0, "p_fake": 0.5})
            fake_probs.append(0.5)

    avg_fake = float(np.mean(fake_probs)) if fake_probs else None
    return {"fake_prob": avg_fake, "chunks": results}

def detect_pdf_fake(file_path: str) -> Dict[str, Any]:
    text = _load_pdf_text(file_path)
    meta = _pdf_metadata(file_path)
    meta_issues = _analyze_pdf_metadata(meta)
    scores = _pdf_text_ai_scores(text)

    p_fake = scores["fake_prob"]
    if p_fake is None:
        label, p_out = "UNCERTAIN", 0.5
    else:
        if p_fake >= 0.75:
            label = "FAKE"
        elif p_fake <= 0.35:
            label = "REAL"
        else:
            label = "UNCERTAIN"
        p_out = p_fake

    return {
        "result": label,
        "ai_probability": p_out,
        "pdf": {
            "metadata_issues": meta_issues,
            "has_text": bool(text.strip()),
            "chunks_evaluated": len(scores["chunks"]),
        },
    }

# ───── Video/Audio models (lazy) ────────────────────────────────────────────
VIDEO_MODEL_ID = "muneeb1812/videomae-base-fake-video-classification"
AUDIO_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

_video_bundle = {"processor": None, "model": None}
_audio_bundle = {"extractor": None, "model": None}

def _ensure_video_bundle():
    if _video_bundle["processor"] is None or _video_bundle["model"] is None:
        proc = VideoMAEImageProcessor.from_pretrained(VIDEO_MODEL_ID)
        mdl  = AutoModelForVideoClassification.from_pretrained(
            VIDEO_MODEL_ID,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        ).to(DEVICE).eval()
        _video_bundle["processor"] = proc
        _video_bundle["model"]    = mdl
    return _video_bundle["processor"], _video_bundle["model"]

def _ensure_audio_bundle():
    if _audio_bundle["extractor"] is None or _audio_bundle["model"] is None:
        ext = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
        mdl = AutoModelForAudioClassification.from_pretrained(
            AUDIO_MODEL_ID,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        ).to(DEVICE).eval()
        _audio_bundle["extractor"] = ext
        _audio_bundle["model"]     = mdl
    return _audio_bundle["extractor"], _audio_bundle["model"]

print("✅ detection.py loaded")

# ───── Subprocess helpers with timeouts ──────────────────────────────────────
def _run(cmd, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)

# ───── Video & audio inference ───────────────────────────────────────────────
@torch.inference_mode()
def predict_video_fake(video_path: str, clip_len: int = 16, frame_stride: int = 2, max_clips: int = 12) -> float:
    """
    Downsamples frames (frame_stride) and caps total clips (max_clips) for speed.
    """
    video_processor, video_model = _ensure_video_bundle()

    cap = cv2.VideoCapture(video_path)
    frames, scores = [], []
    kept = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx % frame_stride) != 0:
            idx += 1
            continue
        idx += 1

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)

        if len(frames) == clip_len:
            try:
                inputs = video_processor(frames, return_tensors="pt").to(DEVICE)
                logits = video_model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                scores.append(probs[1])  # class 1 = AI-generated
            except Exception as e:
                print("⚠️ video clip inference failed:", e)
            frames.clear()
            kept += 1
            if kept >= max_clips:
                break

    cap.release()
    return float(np.mean(scores)) if scores else 0.5

def _has_audio_stream(video_path: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
            text=True, timeout=15,
        ).strip()
        return bool(out)
    except Exception:
        return False

def extract_audio(video_path: str, sr: int = 16000):
    if not _has_audio_stream(video_path):
        print("ℹ️ No audio stream present; skipping audio extraction.")
        return None, sr
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", video_path, "-vn", "-map", "a:0?",
               "-ar", str(sr), "-ac", "1", "-f", "wav", tmp_path]
        _run(cmd, timeout=90)

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            print("ℹ️ Audio extraction produced no data; continuing without audio.")
            try: os.remove(tmp_path)
            except Exception: pass
            return None, sr

        y, sr = librosa.load(tmp_path, sr=sr)
        os.remove(tmp_path)
        return y, sr
    except Exception as e:
        print("⚠️ extract_audio failed (non-fatal):", e)
        return None, sr

@torch.inference_mode()
def predict_audio_fake(video_path: str) -> Optional[float]:
    try:
        y, sr = extract_audio(video_path, sr=16000)
    except Exception as e:
        print("⚠️ Audio load failed:", e)
        return None

    if y is None or (hasattr(y, "size") and getattr(y, "size", 0) < sr):
        return None

    audio_extractor, audio_model = _ensure_audio_bundle()

    try:
        inputs = audio_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True).to(DEVICE)
        logits = audio_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    except Exception as e:
        print("⚠️ audio inference failed:", e)
        return None

    fake_idx = [k for k, v in audio_model.config.id2label.items() if "fake" in v.lower()]
    fake_idx = fake_idx[0] if fake_idx else 1
    return float(probs[fake_idx])

def fuse_predictions(video_path, video_score: float, audio_score: Optional[float]):
    """
    Return both the label and the fused probability in [0..1].
    """
    if audio_score is None:
        final_score = video_score
    else:
        # favor voice authenticity to fight lip-swap artifacts
        final_score = 0.3 * video_score + 0.7 * (1 - audio_score)

    if final_score > 0.7:
        label = "FAKE"
    elif final_score < 0.3:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    print(f"🎬 Video: {video_path}")
    if audio_score is None:
        print("Audio score unavailable, decision based only on video score.")
        print(f"Video Score: {video_score:.4f}")
    else:
        print(f"Final Score: {final_score:.4f}")
        print(f"Scores → Video AI: {video_score:.4f}, Audio AI: {audio_score:.4f}")
    print(f"Result: {label}")
    return label, float(final_score)

# ───── Unified entrypoint ────────────────────────────────────────────────────
# ========= Unified entrypoint =========
def detect_ai_content(path: str) -> dict:
    """
    Routes images, PDFs, and videos to the right detector.
    Always returns JSON; never throws so /analyze won't 500.
    """
    if is_image_path(path):
        try:
            r = score_image_fake_probability(path)
            p = float(r["prob_fake"])
            if p >= 0.75:
                label = "FAKE"
            elif p <= 0.35:
                label = "REAL"
            else:
                label = "UNCERTAIN"
            return {
                "result": label,
                "video_score": None,
                "audio_score": None,
                "image_score": p,
                "ai_probability": p,
                "signals": r.get("signals", {}),
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
            return detect_pdf_fake(path)
        except Exception as e:
            return {
                "result": "UNKNOWN",
                "ai_probability": None,
                "error": f"pdf_pathway_failed: {e.__class__.__name__}",
            }

    # ---- Video/Audio pathway (never crash) ----
    try:
        video_score = predict_video_fake(path)
    except Exception as e:
        return {
            "result": "UNKNOWN",
            "video_score": None,
            "audio_score": None,
            "ai_probability": None,
            "error": f"video_model_failed: {e.__class__.__name__}",
        }

    audio_score = None
    try:
        audio_score = predict_audio_fake(path)
    except Exception as e:
        # non-fatal; keep going with video only
        audio_score = None

    label, final_score = fuse_predictions(path, video_score, audio_score)
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