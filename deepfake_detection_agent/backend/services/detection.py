import sys
import os
import cv2
import math
import json
import tempfile
import subprocess
from typing import Optional, Dict, Any, Set

import torch
import librosa
import numpy as np
from PIL import Image, ImageOps, ImageFile

from transformers import (
    VideoMAEImageProcessor,
    AutoModelForVideoClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========= Optional deps (best-effort) =========
try:
    import timm
except Exception:
    timm = None

try:
    import c2pa
except Exception:
    c2pa = None

try:
    import invisible_watermark as iw
except Exception:
    iw = None

try:
    import exifread
except Exception:
    exifread = None

try:
    import noiseprint
except Exception:
    noiseprint = None

# ========= Optional deps (PDF) =========
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# transformers for text AI-detector (PDF)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

# ========= File type helpers =========
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
PDF_EXTS: Set[str] = {".pdf"}

def is_image_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def is_video_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS

def is_pdf_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in PDF_EXTS

# ========= Image detection settings =========
IMG_MODEL       = os.getenv("IMAGE_MODEL", "fatformer").lower()   # "fatformer" | "effb4"
FATFORMER_CKPT  = os.getenv("FATFORMER_CKPT")  # path to real AIGC/Deepfake ckpt (optional)
EFFB4_CKPT      = os.getenv("EFFB4_CKPT")      # path to real AIGC/Deepfake ckpt (optional)
USE_NOISEPRINT  = os.getenv("USE_NOISEPRINT", "0") == "1"         # default OFF for stability
USE_C2PA        = os.getenv("USE_C2PA", "0") == "1"
USE_SD_WM       = os.getenv("USE_SD_WATERMARK", "0") == "1"

# ========= PDF detection settings =========
PDF_DETECTOR_MODEL_NAME = os.getenv("PDF_TEXT_DETECTOR", "roberta-base-openai-detector")
_pdf_detector = None  # lazy-loaded pipeline

# ========= Helpers: images =========
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
        out["camera_make"] = make or None
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
            pred = iw.decode(arr)  # API differs by version; best-effort
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

# ========= Image classifiers (FatFormer/EfficientNet-B4) =========
_fat_model = None
_eff_model = None

def _prep(img: Image.Image, size=384):
    img = ImageOps.fit(img, (size, size))
    arr = np.array(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406])[None,None,:]
    std  = np.array([0.229, 0.224, 0.225])[None,None,:]
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def _load_fatformer():
    global _fat_model
    if _fat_model is not None or timm is None:
        return _fat_model
    try:
        model = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=1)
        if FATFORMER_CKPT and os.path.isfile(FATFORMER_CKPT):
            state = torch.load(FATFORMER_CKPT, map_location="cpu")
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
            state = torch.load(EFFB4_CKPT, map_location="cpu")
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

    # classifier (requires a real deepfake checkpoint to be meaningful)
    p_cls = _model_prob_fake(img)

    # PRNU (as realness)
    p_real = _noiseprint_score(image_path)     # 0..1 (None if unavailable)
    p_fake_prnu = (1.0 - p_real) if (p_real is not None) else None

    # provenance/meta
    c2 = _c2pa_ok(image_path)                  # True → likely real
    sd = _sd_watermark_present(img)            # True → likely SD-generated
    ex = _exif_features(image_path).get("has_camera_tags", False)

    # fusion
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

# ========= PDF helpers & detection =========
def _load_pdf_text(file_path: str) -> str:
    """
    Try extracting selectable text with PyPDF2. If no text and OCR stack
    is present, OCR each page image via pdf2image + pytesseract.
    """
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
    global _pdf_detector
    if _pdf_detector is not None:
        return _pdf_detector
    if hf_pipeline is None:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(PDF_DETECTOR_MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(PDF_DETECTOR_MODEL_NAME)
        _pdf_detector = hf_pipeline("text-classification", model=mdl, tokenizer=tok)
        return _pdf_detector
    except Exception:
        return None

def _pdf_text_ai_scores(text: str) -> Dict[str, Any]:
    """
    Chunk text, run AI-vs-Human classifier.
    Return average fake probability and chunk details.
    """
    det = _load_pdf_ai_pipeline()
    if det is None or not text.strip():
        return {"fake_prob": None, "chunks": []}

    chunks = [text[i:i+800] for i in range(0, len(text), 800)]
    results = []
    fake_probs = []

    for ch in chunks:
        try:
            out = det(ch)[0]  # {'label': ..., 'score': ...}
            label = str(out.get("label", "")).lower()
            score = float(out.get("score", 0.0))

            # Map labels to fake-prob
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
    """
    Full PDF authenticity analysis: text + metadata (OCR fallback if needed).
    Returns 'result' + 'ai_probability' so UI can render seamlessly.
    """
    text = _load_pdf_text(file_path)
    meta = _pdf_metadata(file_path)
    meta_issues = _analyze_pdf_metadata(meta)
    scores = _pdf_text_ai_scores(text)

    p_fake = scores["fake_prob"]
    if p_fake is None:
        label = "UNCERTAIN"
        p_out = 0.5
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

# ========= Video/Audio pipeline =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_MODEL_ID = "muneeb1812/videomae-base-fake-video-classification"
video_processor = VideoMAEImageProcessor.from_pretrained(VIDEO_MODEL_ID)
video_model = AutoModelForVideoClassification.from_pretrained(VIDEO_MODEL_ID).to(DEVICE).eval()

AUDIO_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
audio_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_ID).to(DEVICE).eval()

print("✅ detection.py loaded")

def predict_video_fake(video_path: str, clip_len: int = 16) -> float:
    cap = cv2.VideoCapture(video_path)
    frames = []
    scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)

        if len(frames) == clip_len:
            inputs = video_processor(frames, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = video_model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            scores.append(probs[1])  # class 1 = AI-generated
            frames = []

    cap.release()
    return float(np.mean(scores)) if scores else 0.5

def _has_audio_stream(video_path: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
            text=True,
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
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

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

def predict_audio_fake(video_path: str) -> Optional[float]:
    try:
        y, sr = extract_audio(video_path, sr=16000)
    except Exception as e:
        print("⚠️ Audio load failed:", e)
        return None

    if y is None or (hasattr(y, "size") and y.size < sr):
        return None

    inputs = audio_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        logits = audio_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

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
    return label, float(final_score)   # <-- CHANGED: return the fused probability

# ========= Unified entrypoint =========
def detect_ai_content(path: str) -> dict:
    """
    Routes images, PDFs, and videos to the right detector.
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
                "ai_probability": p,              # for UI confidence
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

    # Video/Audio pathway
    video_score = predict_video_fake(path)
    audio_score = predict_audio_fake(path)
    label, final_score = fuse_predictions(path, video_score, audio_score)  # <-- CHANGED
    return {
        "result": label,        # 'REAL' | 'FAKE' | 'UNCERTAIN'
        "video_score": float(video_score) if video_score is not None else None,
        "audio_score": float(audio_score) if audio_score is not None else None,
        "ai_probability": float(final_score),  # <-- CHANGED: send unified probability
    }

# ========= CLI =========
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.services.detection <media_path>")
        sys.exit(1)
    detect_ai_content(sys.argv[1])