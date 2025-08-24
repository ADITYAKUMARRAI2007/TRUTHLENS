# deepfake_detection_agent/backend/services/reality_replay.py

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ======================== Speed / Quality Tunables ============================
# Overridable via environment variables.
FAST_MODE = os.getenv("TL_FAST_MODE", "1") not in {"0", "false", "False"}
TARGET_MAX_W = int(os.getenv("TL_TARGET_MAX_W", "960"))   # max bound for width/height (not upscaling)
TARGET_FPS   = int(os.getenv("TL_TARGET_FPS", "15"))
JPEG_QUALITY = int(os.getenv("TL_JPEG_QUALITY", "85"))   # 70–90 is fine
USE_OPTICAL_FLOW_ENV = os.getenv("TL_USE_FLOW", "auto")  # "auto" | "1" | "0"
USE_OPTICAL_FLOW = (USE_OPTICAL_FLOW_ENV not in {"0", "false", "False"}) if USE_OPTICAL_FLOW_ENV != "auto" else (not FAST_MODE)

FILM_GRAIN_SIGMA = 1.5 if FAST_MODE else 2.5

# Common scale filter (portable, no conditional expressions)
# Caps BOTH dimensions to TARGET_MAX_W, preserves AR, and never upscales.
def _scale_filter(max_dim: int) -> str:
    return f"scale={max_dim}:{max_dim}:force_original_aspect_ratio=decrease:flags=lanczos"

# ===================== Helpers: quality & stability ===========================

def _downscale_lanczos(img: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    h, w = target_wh[1], target_wh[0]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

def _match_color_lab(prev_bgr: np.ndarray, cur_bgr: np.ndarray, mix: float = 0.3) -> np.ndarray:
    """Stabilize exposure/tones frame-to-frame by matching L channel CDF, blended."""
    prev_lab = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2LAB)
    cur_lab  = cv2.cvtColor(cur_bgr,  cv2.COLOR_BGR2LAB)
    pl, pa, pb = cv2.split(prev_lab)
    cl, ca, cb = cv2.split(cur_lab)

    prev_hist, _ = np.histogram(pl.flatten(), 256, [0, 256])
    cur_hist,  _ = np.histogram(cl.flatten(), 256, [0, 256])

    prev_cdf = np.cumsum(prev_hist).astype(np.float64)
    cur_cdf  = np.cumsum(cur_hist).astype(np.float64)
    prev_cdf /= prev_cdf[-1] if prev_cdf[-1] else 1
    cur_cdf  /= cur_cdf[-1]  if cur_cdf[-1]  else 1

    map_l = np.interp(cur_cdf, prev_cdf, np.arange(256)).astype(np.uint8)
    clm = map_l[cl]
    matched = cv2.merge([clm, ca, cb])
    matched_bgr = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(cur_bgr, 1.0 - mix, matched_bgr, mix, 0)

def _optical_flow_blend(prev_bgr: np.ndarray, cur_bgr: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Warp previous output to current via optical flow and blend to reduce shimmer."""
    pgray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cgray = cv2.cvtColor(cur_bgr,  cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pgray, cgray, None, 0.5, 3, 21, 3, 5, 1.1, 0)
    h, w = cgray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    prev_warp = cv2.remap(prev_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(cur_bgr, 1.0 - alpha, prev_warp, alpha, 0)

def _add_film_grain(bgr: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    noise = np.random.normal(0, sigma, bgr.shape).astype(np.float32)
    out = np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

# ===================== Optional heavy models (if installed) ====================

_HAS_GFPGAN = False
_HAS_REALESRGAN = False
GFPGANer = None
RealESRGANer = None

try:
    from gfpgan import GFPGANer as _GFPGANer
    GFPGANer = _GFPGANer
    _HAS_GFPGAN = True
except Exception:
    _HAS_GFPGAN = False

try:
    from realesrgan import RealESRGANer as _RealESRGANer
    RealESRGANer = _RealESRGANer
    _HAS_REALESRGAN = True
except Exception:
    _HAS_REALESRGAN = False

# ============================ System helpers ==================================

def _run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    print("▶", " ".join(shlex.quote(str(x)) for x in cmd))
    return subprocess.run(cmd, check=check)

def _probe_fps(video_path: str) -> float:
    """Probe FPS; default 25 if unknown/out of range."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            text=True,
        ).strip()
        if "/" in out:
            a, b = out.split("/")
            fps = float(a) / float(b)
        else:
            fps = float(out)
        return 25.0 if fps <= 0 or fps > 240 else fps
    except Exception:
        return 25.0

def _has_audio_stream(video_path: str) -> bool:
    """Return True if the input has at least one audio stream."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                video_path,
            ],
            text=True,
        ).strip()
        return bool(out)
    except Exception:
        return False

def _extract_audio(src: str, dst_wav: str) -> bool:
    """Extract mono 16k WAV when audio exists (no-throw if not)."""
    if not _has_audio_stream(src):
        print("ℹ️ No audio stream present; skipping audio extraction.")
        return False
    try:
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", src,
            "-vn", "-map", "a:0?",
            "-ar", "16000", "-ac", "1",
            "-f", "wav", dst_wav,
        ])
        return Path(dst_wav).exists() and Path(dst_wav).stat().st_size > 0
    except Exception as e:
        print(f"⚠️ extract_audio failed (non-fatal): {e}")
        return False

def _reattach_audio(video_no_audio: str, audio_wav: str, out_mp4: str) -> bool:
    try:
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_no_audio,
            "-i", audio_wav,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            out_mp4,
        ])
        return Path(out_mp4).exists() and Path(out_mp4).stat().st_size > 0
    except Exception as e:
        print(f"⚠️ reattach_audio failed: {e}")
        return False

# ============================ Restoration helpers =============================

def _lightweight_restore(bgr: np.ndarray) -> np.ndarray:
    """CPU-only: unsharp mask + light denoise + CLAHE."""
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    sharp = cv2.addWeighted(bgr, 1.4, blur, -0.4, 0)
    den = cv2.fastNlMeansDenoisingColored(sharp, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    return out

def _init_gfpgan() -> Optional[object]:
    if not _HAS_GFPGAN:
        return None
    try:
        return GFPGANer(model_path=None, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
    except Exception as e:
        print(f"⚠️ GFPGAN init failed: {e}")
        return None

def _init_realesrgan() -> Optional[object]:
    if not _HAS_REALESRGAN:
        return None
    try:
        return RealESRGANer(scale=2, model_path=None, tile=0, tile_pad=10, pre_pad=0, half=True)
    except Exception as e:
        print(f"⚠️ RealESRGAN init failed: {e}")
        return None

def _apply_gfpgan(restorer, bgr: np.ndarray) -> np.ndarray:
    try:
        _, _, restored = restorer.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True)
        return bgr if restored is None else restored
    except Exception as e:
        print(f"⚠️ GFPGAN enhance failed: {e}")
        return bgr

def _apply_realesrgan(upsampler, bgr: np.ndarray) -> np.ndarray:
    """Upscale 2× to recover micro-detail, then Lanczos downscale to original size."""
    try:
        h, w = bgr.shape[:2]
        up, _ = upsampler.enhance(bgr, outscale=2)
        out = _downscale_lanczos(up, (w, h))
        return out
    except Exception as e:
        print(f"⚠️ RealESRGAN enhance failed: {e}")
        return bgr

# ============================== Main entrypoint ===============================

def run_reality_replay(video_path: str) -> str:
    """
    Clean/humanize a video. If audio exists, extract and reattach it.
    Always returns a playable MP4; if no audio present, returns video-only MP4.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    frames_dir = tmp_dir / "frames"
    restored_dir = tmp_dir / "restored"
    frames_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)

    # Probe (even if we output at TARGET_FPS)
    _ = _probe_fps(video_path)

    # ---------- Super-fast path (single-pass) ----------
    if FAST_MODE and not _HAS_GFPGAN and not _HAS_REALESRGAN:
        restored_video_fast = str(tmp_dir / "reconstructed.mp4")
        vf = f"{_scale_filter(TARGET_MAX_W)},fps={TARGET_FPS},unsharp=5:5:0.5:5:5:0.0"
        try:
            _run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-sws_flags", "lanczos",
                "-i", video_path,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                "-pix_fmt", "yuv420p",
                restored_video_fast,
            ])
            # Attach audio if present
            audio_wav = str(tmp_dir / "audio.wav")
            if _extract_audio(video_path, audio_wav):
                out_mp4 = str(tmp_dir / "reconstructed_with_audio.mp4")
                if _reattach_audio(restored_video_fast, audio_wav, out_mp4):
                    print(f"✅ Reality Replay ready: {out_mp4}")
                    return out_mp4
            print(f"✅ Reality Replay ready: {restored_video_fast}")
            return restored_video_fast
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Fast path failed, falling back to frame pipeline: {e}")

    # ---------- Frame pipeline ----------
    frame_pattern = str(frames_dir / "frame_%06d.jpg")
    vf = f"{_scale_filter(TARGET_MAX_W)},fps={TARGET_FPS}"
    # JPEG quality: map 0..100 to FFmpeg -q:v (2..31 ish). We’ll use 2..8 range for high quality.
    qv = max(2, min(8, int(round((100 - JPEG_QUALITY) / 15)) + 2))

    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-sws_flags", "lanczos",
        "-i", video_path,
        "-vf", vf,
        "-q:v", str(qv),
        frame_pattern,
    ])

    # Extract audio if present
    audio_wav = str(tmp_dir / "audio.wav")
    has_audio = _extract_audio(video_path, audio_wav)

    # Init models
    gfpgan = _init_gfpgan()
    realesr = _init_realesrgan()  # <-- typo fix below (see next line)

    # (typo safe) correct init call:
    realesr = _init_realesrgan()

    # Process frames
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError("No frames extracted; cannot reconstruct video.")

    prev_frame: Optional[np.ndarray] = None
    for fp in frame_paths:
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        out = bgr
        if gfpgan is not None:
            out = _apply_gfpgan(gfpgan, out)
        if realesr is not None:
            out = _apply_realesrgan(realesr, out)
        if gfpgan is None and realesr is None:
            out = _lightweight_restore(out)

        if USE_OPTICAL_FLOW and prev_frame is not None:
            out = _match_color_lab(prev_frame, out, mix=0.20)
            out = _optical_flow_blend(prev_frame, out, alpha=0.18)

        out = _add_film_grain(out, sigma=FILM_GRAIN_SIGMA)

        cv2.imwrite(str(restored_dir / fp.name), out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        prev_frame = out

    # Rebuild video (no audio yet)
    video_no_audio = str(tmp_dir / "reconstructed_noaudio.mp4")
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(TARGET_FPS),
        "-i", str(restored_dir / "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        video_no_audio,
    ])

    # Reattach audio if we had it
    restored_video = str(tmp_dir / "reconstructed.mp4")
    if has_audio and Path(audio_wav).exists():
        ok = _reattach_audio(video_no_audio, audio_wav, restored_video)
        if not ok:
            os.replace(video_no_audio, restored_video)
    else:
        os.replace(video_no_audio, restored_video)

    print(f"✅ Reality Replay ready: {restored_video}")
    return restored_video