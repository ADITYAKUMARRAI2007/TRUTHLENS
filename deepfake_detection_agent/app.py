# deepfake_detection_agent/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn, os, aiofiles, base64, json, traceback, mimetypes, hmac, hashlib, time, shutil, logging
from typing import Optional, Set
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truthlens")

# ---- detection / replay (soft-optional for replay) ----
try:
    from backend.services.detection import detect_ai_content
except Exception as e:
    logger.warning(f"Detection service not available: {e}")
    def detect_ai_content(video_path: str):
        return {
            "result": "UNKNOWN",
            "ai_probability": 0.5,
            "error": "detection_service_not_available"
        }

try:
    from backend.services.detection import (
        _ensure_video_bundle, _ensure_audio_bundle
    )
except Exception:
    _ensure_video_bundle = None
    _ensure_audio_bundle = None

try:
    from backend.services.reality_replay import run_reality_replay  # type: ignore
except Exception:
    try:
        from backend.services.simple_replay import run_reality_replay  # type: ignore
    except Exception:
        def run_reality_replay(video_path: str) -> str:
            return video_path

# ---- optional deps ----
try:
    import requests
except Exception:
    requests = None  # type: ignore

# Email functionality completely removed

# All external integrations removed

# =================== Config ===================
PORT = int(os.getenv("PORT", "8001"))

# CORS: explicit list or regex
origins_env = os.getenv("FRONTEND_ORIGIN", "")
ALLOWED_ORIGINS = [o.strip() for o in origins_env.split(",") if o.strip()]
ALLOWED_ORIGIN_REGEX = os.getenv("FRONTEND_ORIGIN_REGEX")
if not ALLOWED_ORIGINS and not ALLOWED_ORIGIN_REGEX:
    ALLOWED_ORIGINS = ["https://chainbreaker.netlify.app", "http://localhost:5173"]

# Max upload guard (MiB)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Email functionality completely removed

# All external integrations removed

# Admin
APP_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
# Admin email removed
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "dev-admin-key")

# Prefer Render external URL so media/email links are correct
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    os.getenv("RENDER_EXTERNAL_URL", f"http://127.0.0.1:{PORT}")
)

# Storage dir
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =================== File-type helpers ===================
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def _is_image(path: str) -> bool:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return True
    mt, _ = mimetypes.guess_type(str(path))
    return (mt or "").startswith("image/")

def _is_video(path: str) -> bool:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return True
    mt, _ = mimetypes.guess_type(str(path))
    return (mt or "").startswith("video/")

# =================== FastAPI ===================
app = FastAPI(title="TruthLens API (HIL + Background + CORS)", version="3.8")

cors_kwargs = dict(
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if ALLOWED_ORIGIN_REGEX:
    app.add_middleware(CORSMiddleware, allow_origin_regex=ALLOWED_ORIGIN_REGEX, **cors_kwargs)
    print("CORS allow_origin_regex:", ALLOWED_ORIGIN_REGEX)
else:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, **cors_kwargs)
    print("CORS allow_origins:", ALLOWED_ORIGINS)

# serve anything in ./output at /media/*
app.mount("/media", StaticFiles(directory=str(OUTPUT_DIR), html=False), name="media")

@app.get("/")
def root():
    return {"message": "TruthLens API is running 🚀"}

@app.get("/config")
def config():
    return {
        "allowed_origins": ALLOWED_ORIGINS,
        "allowed_origin_regex": ALLOWED_ORIGIN_REGEX,
        "max_upload_mb": MAX_UPLOAD_MB,
        "public_base_url": PUBLIC_BASE_URL,
    }

# Warmup after cold start
@app.get("/warmup")
def warmup():
    warmed = {"video": False, "audio": False}
    try:
        if callable(_ensure_video_bundle):
            _ensure_video_bundle()
            warmed["video"] = True
    except Exception as e:
        logger.warning("Video warmup skipped: %s", e)
    try:
        if callable(_ensure_audio_bundle):
            _ensure_audio_bundle()
            warmed["audio"] = True
    except Exception as e:
        logger.warning("Audio warmup skipped: %s", e)
    return {"ok": True, "warmed": warmed}

@app.get("/health")
def health():
    integrations = {
        "gmail_env_ready": False,
        "drive_env_ready": False,
        "notion_env_ready": False,
        "portia_ready": False,
        "gemini_ready": False,
        "admin_email_set": False,
        "app_secret_set": APP_SECRET != "dev-secret-change-me",
    }
    return {"ok": True, **integrations}

# Email functionality completely removed

@app.get("/debug/notification")
def debug_notification():
    """Test notification system"""
    try:
        webhook_url = os.getenv("WEBHOOK_URL")
        if webhook_url and requests:
            payload = {
                "content": "🔔 **TruthLens Notification Test**\n\nThis is a test notification from TruthLens!",
                "username": "TruthLens Bot"
            }
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                return {"ok": True, "message": "Webhook notification sent successfully"}
            else:
                return {"ok": False, "error": f"Webhook failed: {response.status_code}"}
        else:
            return {"ok": True, "message": "Notification system ready (webhook not configured)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# All external integrations removed

# =================== HIL / Jobs ===================
PENDING_JOBS = {}  # {job_id: {...}}

def _sign_job(job_id: str) -> str:
    return hmac.new(APP_SECRET.encode(), job_id.encode(), hashlib.sha256).hexdigest()

def _verify_sig(job_id: str, sig: str) -> bool:
    return hmac.compare_digest(sig, _sign_job(job_id))

def _color(status: str) -> str:
    return "#22c55e" if status == "REAL" else "#ef4444" if status == "FAKE" else "#f59e0b"

def _public_media_url(filename: str) -> str:
    return f"{PUBLIC_BASE_URL}/media/{filename}"

# ---------- Disk-backed persistence for jobs (survives Render sleeps) ----------
JOBS_DIR = OUTPUT_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

def _save_job(job_id: str) -> None:
    try:
        job = PENDING_JOBS.get(job_id)
        if not job:
            return
        with open(JOBS_DIR / f"{job_id}.json", "w") as f:
            json.dump(job, f)
    except Exception:
        logger.warning("save_job failed:\n%s", traceback.format_exc())

def _load_job(job_id: str):
    j = PENDING_JOBS.get(job_id)
    if j:
        return j
    path = JOBS_DIR / f"{job_id}.json"
    if path.exists():
        try:
            with open(path) as f:
                j = json.load(f)
            PENDING_JOBS[job_id] = j  # rehydrate in-memory cache
            return j
        except Exception:
            logger.warning("load_job failed:\n%s", traceback.format_exc())
    return None

# ---------- background workers ----------
def _process_job(job_id: str, public_path: Path):
    logger.info("BG start: detection for job %s", job_id)
    try:
        detection = detect_ai_content(str(public_path))
        logger.info("BG done: detection for job %s -> %s", job_id, detection)
    except Exception as e:
        logger.error("detect_ai_content crashed for %s:\n%s", job_id, traceback.format_exc())
        detection = {
            "result": "UNKNOWN",
            "ai_probability": None,
            "error": f"detect_ai_content_crashed: {e.__class__.__name__}",
        }

    job = _load_job(job_id)
    if not job:
        logger.warning("BG save skipped: job %s not found", job_id)
        return

    # persist fields the frontend polls
    job["result"] = detection
    job["ai_probability"] = detection.get("ai_probability")
    job["status"] = detection.get("result", "UNKNOWN")
    # optional convenience copies (if the detector exposes them)
    if "video_score" in detection:
        job["video_score"] = detection["video_score"]
    if "audio_score" in detection:
        job["audio_score"] = detection["audio_score"]

    _save_job(job_id)

def _finalize_after_approve(job_id: str):
    """
    Runs after approve (in background). If result is still PROCESSING/UNKNOWN,
    compute detection and build replay.
    """
    job = _load_job(job_id)
    if not job:
        return

    src_path = job.get("file", "")
    # Run detection only if needed
    status_now = (job.get("result") or {}).get("result")
    if status_now in (None, "PROCESSING", "UNKNOWN"):
        try:
            job["result"] = detect_ai_content(src_path)
        except Exception:
            logger.error("detect_ai_content (finalize) crashed:\n%s", traceback.format_exc())
            job["result"] = {"result": "UNKNOWN", "ai_probability": None, "error": "detect_ai_content_crashed"}

    # Generate replay (best-effort)
    replay_public_filename = f"replay_{job_id}.mp4"
    replay_public_path = OUTPUT_DIR / replay_public_filename
    try:
        replay_tmp = run_reality_replay(src_path)  # may equal src_path
        if replay_tmp != str(replay_public_path):
            shutil.copyfile(replay_tmp, replay_public_path)
        job["replay_url"] = _public_media_url(replay_public_filename)
    except Exception:
        logger.error("Replay generation/copy failed:\n%s", traceback.format_exc())
        job["replay_url"] = None

    _save_job(job_id)

# =================== Endpoints ===================

@app.post("/analyze")
async def analyze(background: BackgroundTasks, file: UploadFile = File(...)):
    """
    1) Stream upload to ./output (served at /media)
    2) Queue detection in background (prevents 502)
    3) Best-effort email to admin
    4) Always 200 with PENDING + job_id
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(int(time.time() * 1000))
    safe_name = f"{job_id}_{Path(file.filename or 'upload').name}"
    public_path = OUTPUT_DIR / safe_name

    # Stream write with size guard
    written = 0
    try:
        async with aiofiles.open(public_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    try: await f.flush()
                    except Exception: pass
                    try: public_path.unlink(missing_ok=True)
                    except Exception: pass
                    return JSONResponse({"ok": False, "error": f"file_too_large>{MAX_UPLOAD_MB}MB"}, status_code=200)
                await f.write(chunk)
    except Exception:
        try: public_path.unlink(missing_ok=True)
        except Exception: pass
        return JSONResponse({"ok": False, "error": "upload_failed"}, status_code=200)

    original_url = _public_media_url(safe_name)

    # init job
    PENDING_JOBS[job_id] = {
        "status": "PENDING",
        "approved": False,
        "file": str(public_path),
        "result": {"result": "PROCESSING"},
        "created_at": int(time.time()),
        "preview_link": None,
        "original_url": original_url,
        "replay_url": None,
    }
    _save_job(job_id)

    # queue background detection
    background.add_task(_process_job, job_id, public_path)

    # Simple notification without email
    try:
        logger.info(f"New file uploaded: {Path(public_path).name}")
        send_webhook_notification(job_id, Path(public_path).name, original_url)
    except Exception as e:
        logger.error(f"Notification failed: {e}")
        # Fallback to simple notification
        try:
            send_simple_notification(job_id, Path(public_path).name, original_url)
        except Exception as simple_error:
            logger.error(f"All notification methods failed: {simple_error}")

    return {
        "ok": True,
        "job_id": job_id,
        "status": "PENDING",
        "message": "Queued for analysis.",
        "prelim_result": "PROCESSING",
        "ai_probability": None,
        "error": None,
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _load_job(job_id)
    if not job or job["status"] != "APPROVED":
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "ok": True,
        "job_id": job_id,
        "approved": True,
        "status": "APPROVED",
        "result": job.get("result", {}),
        "original_url": job.get("original_url"),
        "replay_url": job.get("replay_url"),
        "original_link": job.get("original_url"),
        "replay_link": job.get("replay_url"),
        "ai_probability": job.get("result", {}).get("ai_probability"),
        "file_name": Path(job.get("file", "")).name,
    }
def _run_replay_and_save(job_id: str, src_path: str, replay_public_path: Path):
    """
    Generate replay video and save it to the public directory.
    This runs in a background thread after job approval.
    """
    try:
        logger.info("Starting replay generation for job %s", job_id)
        
        # Generate replay using the reality replay service
        replay_tmp = run_reality_replay(src_path)
        
        # If replay generation succeeded and we got a different file
        if replay_tmp and replay_tmp != str(replay_public_path):
            # Copy the generated replay to our public directory
            shutil.copyfile(replay_tmp, replay_public_path)
            logger.info("Replay copied from %s to %s", replay_tmp, replay_public_path)
            
            # Update the job with the replay URL
            job = _load_job(job_id)
            if job:
                job["replay_url"] = _public_media_url(replay_public_path.name)
                _save_job(job_id)
                logger.info("Replay URL saved for job %s: %s", job_id, job["replay_url"])
        else:
            # If replay generation failed or returned the same path, create a simple copy
            logger.warning("Replay generation returned same path, creating copy for job %s", job_id)
            shutil.copyfile(src_path, replay_public_path)
            
            job = _load_job(job_id)
            if job:
                job["replay_url"] = _public_media_url(replay_public_path.name)
                _save_job(job_id)
                logger.info("Fallback replay URL saved for job %s", job_id)
                
    except Exception as e:
        logger.error("Replay generation failed for job %s: %s", job_id, str(e))
        # Try to create a fallback replay (just copy the original)
        try:
            shutil.copyfile(src_path, replay_public_path)
            job = _load_job(job_id)
            if job:
                job["replay_url"] = _public_media_url(replay_public_path.name)
                _save_job(job_id)
                logger.info("Fallback replay created for job %s after error", job_id)
        except Exception as fallback_error:
            logger.error("Fallback replay also failed for job %s: %s", job_id, str(fallback_error))
@app.get("/jobs/{job_id}/approve")
def approve_job(job_id: str, sig: str):
    if not _verify_sig(job_id, sig):
        raise HTTPException(status_code=403, detail="Invalid signature")
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "APPROVED":
        return {"ok": True, "job_id": job_id, "status": "APPROVED"}

    src_path = job["file"]

    # All external integrations removed

    # APPROVE IMMEDIATELY
    job["status"] = "APPROVED"
    job["approved"] = True
    _save_job(job_id)  # so GET /jobs/{id} starts returning data immediately

    # Now schedule replay asynchronously
    try:
        replay_public_filename = f"replay_{job_id}.mp4"
        replay_public_path = OUTPUT_DIR / replay_public_filename

        import threading
        threading.Thread(
            target=_run_replay_and_save,
            args=(job_id, src_path, replay_public_path),
            daemon=True
        ).start()
    except Exception:
        logger.warning("Could not schedule replay for %s", job_id)

    return {"ok": True, "job_id": job_id, "status": "APPROVED"}

@app.get("/jobs/{job_id}/deny")
def deny_job(job_id: str, sig: str):
    if not _verify_sig(job_id, sig):
        raise HTTPException(status_code=403, detail="Invalid signature")
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["status"] = "DENIED"
    job["approved"] = False
    _save_job(job_id)
    return {"ok": True, "job_id": job_id, "status": "DENIED"}

@app.get("/admin/jobs")
def list_jobs(request: Request):
    if request.headers.get("x-admin-key") != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return PENDING_JOBS

@app.post("/reality-replay")
async def reality_replay(file: UploadFile = File(...)):
    filename = (file.filename or "upload").lower()
    if (file.content_type or "").startswith("image/") or any(filename.endswith(e) for e in IMAGE_EXTS):
        raise HTTPException(status_code=400, detail="Reality Replay is only for videos.")

    # Stream to /tmp
    tmp_name = filename if any(filename.endswith(e) for e in VIDEO_EXTS) else "upload.mp4"
    temp_path = f"/tmp/{tmp_name}"
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await f.write(chunk)
    except Exception:
        try: os.remove(temp_path)
        except Exception: pass
        raise HTTPException(status_code=500, detail="Upload failed.")

    if not _is_video(temp_path):
        try: os.remove(temp_path)
        except Exception: pass
        raise HTTPException(status_code=400, detail="Provided file is not a video.")

    try:
        restored_path = run_reality_replay(temp_path)
        return FileResponse(restored_path, filename="reconstructed.mp4")
    except Exception:
        raise HTTPException(status_code=500, detail="Replay failed.")

# =================== Simple Email Fallback ===================
def send_simple_notification(job_id: str, filename: str, original_url: str):
    """
    Simple notification method when Gmail is not available.
    This logs the approval URLs so you can manually approve.
    """
    try:
        sig = _sign_job(job_id)
        approve_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/approve?sig={sig}"
        deny_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/deny?sig={sig}"
        
        logger.info("=" * 80)
        logger.info("📧 ADMIN REVIEW REQUIRED - Gmail not available")
        logger.info("=" * 80)
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Filename: {filename}")
        logger.info(f"Original URL: {original_url}")
        logger.info(f"✅ Approve: {approve_url}")
        logger.info(f"❌ Deny: {deny_url}")
        logger.info("=" * 80)
        
        # Also log to a file for easy access
        with open("admin_reviews.log", "a") as f:
            f.write(f"\n{'-'*50}\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Filename: {filename}\n")
            f.write(f"Approve: {approve_url}\n")
            f.write(f"Deny: {deny_url}\n")
            f.write(f"{'-'*50}\n")
            
    except Exception as e:
        logger.error(f"Simple notification failed: {e}")

# =================== Webhook Notification System ===================
def send_webhook_notification(job_id: str, filename: str, original_url: str):
    """
    Send notification via webhook (Discord, Slack, etc.) when Gmail fails.
    This provides immediate notification without email setup.
    """
    try:
        sig = _sign_job(job_id)
        approve_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/approve?sig={sig}"
        deny_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/deny?sig={sig}"
        
        # Create a simple notification message
        message = f"""
🔍 **NEW TRUTHLENS SUBMISSION REQUIRES REVIEW**

📁 **File:** {filename}
🆔 **Job ID:** {job_id}
🔗 **Preview:** {original_url}

✅ **Approve:** {approve_url}
❌ **Deny:** {deny_url}

---
*Sent via TruthLens Webhook System*
        """
        
        # Log the notification
        logger.info("=" * 80)
        logger.info("🔔 WEBHOOK NOTIFICATION - Admin Review Required")
        logger.info("=" * 80)
        logger.info(message)
        logger.info("=" * 80)
        
        # You can add webhook URLs here for Discord/Slack/etc.
        webhook_url = os.getenv("WEBHOOK_URL")
        if webhook_url and requests:
            try:
                payload = {
                    "content": message,
                    "username": "TruthLens Bot"
                }
                response = requests.post(webhook_url, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info("Webhook notification sent successfully")
                else:
                    logger.warning(f"Webhook failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Webhook error: {e}")
        
        # Also save to file for manual review
        with open("pending_reviews.txt", "a") as f:
            f.write(f"\n{'-'*60}\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"File: {filename}\n")
            f.write(f"Preview: {original_url}\n")
            f.write(f"Approve: {approve_url}\n")
            f.write(f"Deny: {deny_url}\n")
            f.write(f"{'-'*60}\n")
            
    except Exception as e:
        logger.error(f"Webhook notification failed: {e}")

# Local dev:
if __name__ == "__main__":
    uvicorn.run("deepfake_detection_agent.app:app", host="0.0.0.0", port=PORT, reload=True)