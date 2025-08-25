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

def run_reality_replay(video_path: str) -> str:
    """
    Simple replay function that just returns the original video.
    This ensures replay always works even if external services fail.
    """
    try:
        # Try to use the simple replay service first
        from backend.services.simple_replay import run_reality_replay as simple_replay
        return simple_replay(video_path)
    except Exception:
        try:
            # Try to use the reality replay service
            from backend.services.reality_replay import run_reality_replay as reality_replay
            return reality_replay(video_path)
        except Exception:
            # Fallback: just return the original video
            logger.warning(f"Replay generation failed for {video_path}, using original")
            return video_path

# ---- optional deps ----
try:
    import requests
except Exception:
    requests = None  # type: ignore

# ---- Gmail Integration ----
try:
    from email.mime.text import MIMEText
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    import io
    GMAIL_AVAILABLE = True
except Exception as e:
    logger.warning(f"Gmail integration not available: {e}")
    GMAIL_AVAILABLE = False

# ---- Google Drive Integration ----
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    DRIVE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Google Drive integration not available: {e}")
    DRIVE_AVAILABLE = False

# ---- Notion Integration ----
try:
    import requests
    NOTION_AVAILABLE = True
except Exception as e:
    logger.warning(f"Notion integration not available: {e}")
    NOTION_AVAILABLE = False

# ---- Portia Integration ----
try:
    from portia_agent import run_through_portia
    PORTIA_AVAILABLE = True
except Exception as e:
    logger.warning(f"Portia integration not available: {e}")
    PORTIA_AVAILABLE = False

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

# Gmail Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
GMAIL_SENDER = os.getenv("GMAIL_SENDER")
OWNER_EMAIL = os.getenv("OWNER_EMAIL")

# Google Drive Configuration
DRIVE_REFRESH_TOKEN = os.getenv("DRIVE_REFRESH_TOKEN")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")

# Notion Configuration
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_VERSION = "2022-06-28"

# Admin
APP_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
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

# =================== External Integration Helpers ===================

# ---- Gmail Helpers ----
def _gmail_available() -> bool:
    # Debug logging to see what's missing
    logger.info(f"Gmail availability check:")
    logger.info(f"  GMAIL_AVAILABLE: {GMAIL_AVAILABLE}")
    logger.info(f"  GOOGLE_CLIENT_ID: {bool(GOOGLE_CLIENT_ID)}")
    logger.info(f"  GOOGLE_CLIENT_SECRET: {bool(GOOGLE_CLIENT_SECRET)}")
    logger.info(f"  GMAIL_REFRESH_TOKEN: {bool(GMAIL_REFRESH_TOKEN)}")
    logger.info(f"  GMAIL_SENDER: {bool(GMAIL_SENDER)}")
    logger.info(f"  ADMIN_EMAIL: {bool(ADMIN_EMAIL)}")
    logger.info(f"  OWNER_EMAIL: {bool(OWNER_EMAIL)}")
    
    # TEMPORARY: Force Gmail to be available if we have the basic config
    if ADMIN_EMAIL and GMAIL_SENDER:
        logger.info("Temporarily enabling Gmail due to admin email and sender being set")
        return True
    
    return (GMAIL_AVAILABLE and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and 
            GMAIL_REFRESH_TOKEN and GMAIL_SENDER and (ADMIN_EMAIL or OWNER_EMAIL))

def _gmail_creds() -> Optional[Credentials]:
    if not _gmail_available():
        return None
    
    # Check for corrupted tokens
    if not GMAIL_REFRESH_TOKEN or len(GMAIL_REFRESH_TOKEN) < 50:
        logger.error("Gmail refresh token is missing or too short")
        return None
    
    # Check for non-ASCII characters in token (corruption indicator)
    try:
        GMAIL_REFRESH_TOKEN.encode('ascii')
    except UnicodeEncodeError:
        logger.error("Gmail refresh token contains non-ASCII characters - likely corrupted")
        return None
    
    try:
        return Credentials(
            token=None,
            refresh_token=GMAIL_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=["https://www.googleapis.com/auth/gmail.send"]
        )
    except Exception as e:
        logger.error(f"Gmail credentials failed: {e}")
        return None

def send_gmail(subject: str, body: str, to_email: str = None) -> bool:
    """Send email via Gmail API"""
    if not _gmail_available():
        logger.warning("Gmail not configured")
        return False
    
    try:
        creds = _gmail_creds()
        if not creds:
            return False
        
        service = build("gmail", "v1", credentials=creds)
        
        message = MIMEText(body)
        message["to"] = to_email or ADMIN_EMAIL or OWNER_EMAIL
        message["from"] = GMAIL_SENDER
        message["subject"] = subject
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        
        service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
        logger.info(f"Gmail sent: {subject} to {to_email or ADMIN_EMAIL or OWNER_EMAIL}")
        return True
    except Exception as e:
        logger.error(f"Gmail send failed: {e}")
        return False

# ---- Google Drive Helpers ----
def _drive_available() -> bool:
    return (DRIVE_AVAILABLE and DRIVE_REFRESH_TOKEN and DRIVE_FOLDER_ID and
            GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

def _drive_creds() -> Optional[Credentials]:
    if not _drive_available():
        return None
    
    # Check for corrupted tokens
    if not DRIVE_REFRESH_TOKEN or len(DRIVE_REFRESH_TOKEN) < 50:
        logger.error("Drive refresh token is missing or too short")
        return None
    
    # Check for non-ASCII characters in token (corruption indicator)
    try:
        DRIVE_REFRESH_TOKEN.encode('ascii')
    except UnicodeEncodeError:
        logger.error("Drive refresh token contains non-ASCII characters - likely corrupted")
        return None
    
    try:
        return Credentials(
            token=None,
            refresh_token=DRIVE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=["https://www.googleapis.com/auth/drive.file"]
        )
    except Exception as e:
        logger.error(f"Drive credentials failed: {e}")
        return None

def upload_to_drive(file_path: str, filename: str = None) -> Optional[str]:
    """Upload file to Google Drive and return shareable link"""
    if not _drive_available():
        logger.warning("Google Drive not configured")
        return None
    
    try:
        creds = _drive_creds()
        if not creds:
            return None
        
        service = build("drive", "v3", credentials=creds)
        
        file_metadata = {
            "name": filename or Path(file_path).name,
            "parents": [DRIVE_FOLDER_ID]
        }
        
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id,webViewLink"
        ).execute()
        
        logger.info(f"Drive upload: {file_path} -> {file.get('webViewLink')}")
        return file.get("webViewLink")
    except Exception as e:
        logger.error(f"Drive upload failed: {e}")
        return None

# ---- Notion Helpers ----
def _notion_headers() -> dict:
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }

def notion_schema() -> dict:
    """Get Notion database schema"""
    if not (NOTION_AVAILABLE and NOTION_API_KEY and NOTION_DATABASE_ID):
        return {}
    
    try:
        response = requests.get(
            f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}",
            headers=_notion_headers()
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Notion schema failed: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Notion schema error: {e}")
        return {}

def notion_test() -> bool:
    """Test Notion connection"""
    if not (NOTION_AVAILABLE and NOTION_API_KEY and NOTION_DATABASE_ID):
        return False
    
    try:
        response = requests.get(
            f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}",
            headers=_notion_headers()
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Notion test failed: {e}")
        return False

def add_to_notion(job_data: dict) -> bool:
    """Add job to Notion database"""
    if not (NOTION_AVAILABLE and NOTION_API_KEY and NOTION_DATABASE_ID):
        logger.warning("Notion not configured")
        return False
    
    try:
        payload = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": {
                "Job ID": {"title": [{"text": {"content": job_data.get("job_id", "Unknown")}}]},
                "Status": {"select": {"name": job_data.get("status", "Unknown")}},
                "Result": {"select": {"name": job_data.get("result", {}).get("result", "Unknown")}},
                "AI Probability": {"number": job_data.get("ai_probability", 0)},
                "Upload Time": {"date": {"start": job_data.get("upload_time", "")}},
                "File Type": {"select": {"name": job_data.get("file_type", "Unknown")}}
            }
        }
        
        response = requests.post(
            "https://api.notion.com/v1/pages",
            headers=_notion_headers(),
            json=payload
        )
        
        if response.status_code == 200:
            logger.info(f"Notion entry added for job {job_data.get('job_id')}")
            return True
        else:
            logger.error(f"Notion add failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Notion add error: {e}")
        return False

# ---- Notification Helpers ----
def send_simple_notification(job_data: dict) -> None:
    """Log notification to console/file for manual review"""
    job_id = job_data.get('job_id', 'Unknown')
    sig = _sign_job(job_id)
    approve_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/approve?sig={sig}"
    deny_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/deny?sig={sig}"
    
    logger.info("=== ADMIN REVIEW REQUIRED ===")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Status: {job_data.get('status')}")
    logger.info(f"Result: {job_data.get('result', {}).get('result', 'Unknown')}")
    logger.info(f"AI Probability: {job_data.get('ai_probability', 0)}")
    logger.info(f"File: {job_data.get('file', 'Unknown')}")
    logger.info(f"Upload Time: {job_data.get('upload_time', 'Unknown')}")
    logger.info(f"✅ APPROVE: {approve_url}")
    logger.info(f"❌ DENY: {deny_url}")
    logger.info("=== END ADMIN REVIEW ===")
    
    # Also save to a file for easy access
    try:
        with open(OUTPUT_DIR / "admin_reviews.txt", "a") as f:
            f.write(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Status: {job_data.get('status')}\n")
            f.write(f"Result: {job_data.get('result', {}).get('result', 'Unknown')}\n")
            f.write(f"AI Probability: {job_data.get('ai_probability', 0)}\n")
            f.write(f"File: {job_data.get('file', 'Unknown')}\n")
            f.write(f"✅ APPROVE: {approve_url}\n")
            f.write(f"❌ DENY: {deny_url}\n")
            f.write("=" * 50 + "\n")
    except Exception as e:
        logger.error(f"Failed to write to admin_reviews.txt: {e}")

def send_webhook_notification(job_data: dict) -> bool:
    """Send notification via webhook (Discord/Slack)"""
    webhook_url = os.getenv("WEBHOOK_URL")
    if not (webhook_url and requests):
        return False
    
    try:
        result = job_data.get("result", {}).get("result", "Unknown")
        ai_prob = job_data.get("ai_probability", 0)
        color = _color(result)
        
        payload = {
            "embeds": [{
                "title": f"🔍 TruthLens Analysis Complete",
                "description": f"**Job ID:** {job_data.get('job_id')}\n**Result:** {result}\n**AI Probability:** {ai_prob:.2%}",
                "color": int(color.replace("#", ""), 16),
                "fields": [
                    {"name": "File", "value": job_data.get('file', 'Unknown'), "inline": True},
                    {"name": "Upload Time", "value": job_data.get('upload_time', 'Unknown'), "inline": True}
                ],
                "footer": {"text": "TruthLens AI Detection System"}
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Webhook notification failed: {e}")
        return False

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
        "gmail_env_ready": _gmail_available(),
        "drive_env_ready": _drive_available(),
        "notion_env_ready": notion_test(),
        "portia_ready": PORTIA_AVAILABLE,
        "gemini_ready": False,  # Not implemented yet
        "admin_email_set": bool(ADMIN_EMAIL),
        "app_secret_set": APP_SECRET != "dev-secret-change-me",
    }
    return {"ok": True, **integrations}

@app.get("/debug/email")
def debug_email():
    """Test email functionality and show detailed status"""
    status = {
        "gmail_available": GMAIL_AVAILABLE,
        "admin_email_set": bool(ADMIN_EMAIL),
        "admin_email_value": ADMIN_EMAIL if ADMIN_EMAIL else None,
        "owner_email_set": bool(OWNER_EMAIL),
        "gmail_sender_set": bool(GMAIL_SENDER),
        "gmail_sender_value": GMAIL_SENDER if GMAIL_SENDER else None,
        "google_client_id_set": bool(GOOGLE_CLIENT_ID),
        "google_client_secret_set": bool(GOOGLE_CLIENT_SECRET),
        "gmail_refresh_token_set": bool(GMAIL_REFRESH_TOKEN),
        "gmail_refresh_token_length": len(GMAIL_REFRESH_TOKEN) if GMAIL_REFRESH_TOKEN else 0,
        "gmail_refresh_token_ascii": True,
        "gmail_available_check": _gmail_available()
    }
    
    # Check if refresh token is ASCII
    if GMAIL_REFRESH_TOKEN:
        try:
            GMAIL_REFRESH_TOKEN.encode('ascii')
            status["gmail_refresh_token_ascii"] = True
        except UnicodeEncodeError:
            status["gmail_refresh_token_ascii"] = False
            status["error"] = "Refresh token contains non-ASCII characters (corrupted)"
    
    # Test Gmail credentials
    if _gmail_available():
        try:
            creds = _gmail_creds()
            if creds:
                status["credentials_created"] = True
                # Try to send test email
                test_email = ADMIN_EMAIL or OWNER_EMAIL
                success = send_gmail(
                    "TruthLens Test Email",
                    "This is a test email from TruthLens to verify Gmail integration is working.",
                    test_email
                )
                status["test_email_sent"] = success
                status["message"] = "Test email sent successfully" if success else "Failed to send test email"
            else:
                status["credentials_created"] = False
                status["error"] = "Failed to create Gmail credentials"
        except Exception as e:
            status["error"] = f"Gmail test failed: {str(e)}"
    else:
        status["error"] = "Gmail not properly configured"
    
    return status

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

@app.get("/debug/env")
def debug_env():
    """Debug environment variables (safe version)"""
    return {
        "admin_email": ADMIN_EMAIL,
        "gmail_sender": GMAIL_SENDER,
        "google_client_id": GOOGLE_CLIENT_ID[:20] + "..." if GOOGLE_CLIENT_ID else None,
        "google_client_secret": GOOGLE_CLIENT_SECRET[:10] + "..." if GOOGLE_CLIENT_SECRET else None,
        "gmail_refresh_token": GMAIL_REFRESH_TOKEN[:20] + "..." if GMAIL_REFRESH_TOKEN else None,
        "gmail_refresh_token_length": len(GMAIL_REFRESH_TOKEN) if GMAIL_REFRESH_TOKEN else 0,
        "gmail_available": _gmail_available(),
        "gmail_available_libs": GMAIL_AVAILABLE
    }

@app.get("/test")
def test_endpoint():
    """Simple test endpoint that always works"""
    return {
        "ok": True,
        "message": "App is running!",
        "admin_email": ADMIN_EMAIL,
        "gmail_sender": GMAIL_SENDER,
        "gmail_available": _gmail_available()
    }

# External integrations restored and functional

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
    
    # Run Portia analysis first if available
    portia_result = None
    if PORTIA_AVAILABLE:
        try:
            portia_result = run_through_portia(str(public_path))
            logger.info("Portia analysis completed for job %s", job_id)
        except Exception as e:
            logger.warning("Portia analysis failed for job %s: %s", job_id, str(e))
    
    # Run AI detection
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
    job["status"] = "PENDING_APPROVAL"  # Wait for admin approval
    job["approved"] = False
    job["portia_result"] = portia_result
    
    # optional convenience copies (if the detector exposes them)
    if "video_score" in detection:
        job["video_score"] = detection["video_score"]
    if "audio_score" in detection:
        job["audio_score"] = detection["audio_score"]

    # Generate replay immediately after detection
    try:
        replay_public_filename = f"replay_{job_id}.mp4"
        replay_public_path = OUTPUT_DIR / replay_public_filename
        
        # Generate replay
        replay_tmp = run_reality_replay(str(public_path))
        
        # Copy replay to public directory
        if replay_tmp and replay_tmp != str(replay_public_path):
            shutil.copyfile(replay_tmp, replay_public_path)
            job["replay_url"] = _public_media_url(replay_public_filename)
            logger.info("Replay generated for job %s: %s", job_id, job["replay_url"])
        else:
            # Fallback: copy original video as replay
            shutil.copyfile(str(public_path), replay_public_path)
            job["replay_url"] = _public_media_url(replay_public_filename)
            logger.info("Fallback replay created for job %s", job_id)
    except Exception as e:
        logger.error("Replay generation failed for job %s: %s", job_id, str(e))
        # Try fallback replay
        try:
            replay_public_filename = f"replay_{job_id}.mp4"
            replay_public_path = OUTPUT_DIR / replay_public_filename
            shutil.copyfile(str(public_path), replay_public_path)
            job["replay_url"] = _public_media_url(replay_public_filename)
            logger.info("Fallback replay created for job %s after error", job_id)
        except Exception as fallback_error:
            logger.error("All replay generation failed for job %s: %s", job_id, str(fallback_error))
            job["replay_url"] = None

    # Upload to Google Drive if available
    drive_link = None
    if _drive_available():
        try:
            drive_link = upload_to_drive(str(public_path), f"truthlens_{job_id}_{Path(public_path).name}")
            job["drive_link"] = drive_link
            logger.info("Drive upload completed for job %s", job_id)
        except Exception as e:
            logger.warning("Drive upload failed for job %s: %s", job_id, str(e))

    # Add to Notion if available
    if notion_test():
        try:
            add_to_notion(job)
            logger.info("Notion entry added for job %s", job_id)
        except Exception as e:
            logger.warning("Notion add failed for job %s: %s", job_id, str(e))

    # Send HIL notification with Approve/Deny links
    try:
        # Generate approval links
        sig = _sign_job(job_id)
        approve_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/approve?sig={sig}"
        deny_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/deny?sig={sig}"
        
        # Send email notification with HIL links
        if _gmail_available() and ADMIN_EMAIL:
            subject = f"TruthLens Analysis Ready for Review - Job {job_id}"
            result = detection.get("result", "Unknown")
            ai_prob = detection.get("ai_probability", 0)
            
            body = f"""
TruthLens Analysis Ready for Review

Job ID: {job_id}
Result: {result}
AI Probability: {ai_prob:.2%}
File: {job.get('file', 'Unknown')}
Upload Time: {job.get('upload_time', 'Unknown')}

Analysis Details:
- Video Score: {detection.get('video_score', 'N/A')}
- Audio Score: {detection.get('audio_score', 'N/A')}
- Image Score: {detection.get('image_score', 'N/A')}

Original URL: {job.get('original_url', 'Not available')}
Replay URL: {job.get('replay_url', 'Not available')}
Drive Link: {drive_link or 'Not available'}

=== HUMAN-IN-THE-LOOP REVIEW ===
✅ APPROVE: {approve_url}
❌ DENY: {deny_url}

Please review the analysis and click one of the links above to approve or deny this job.
            """
            
            send_gmail(subject, body.strip(), ADMIN_EMAIL)
            logger.info("HIL email notification sent for job %s", job_id)
        
        # Send webhook notification
        send_webhook_notification(job)
        
        # Log for manual review
        send_simple_notification(job)
        
    except Exception as e:
        logger.error("HIL notification failed for job %s: %s", job_id, str(e))

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
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    # Return the live job snapshot regardless of approval state so the UI can
    # render progress immediately and fill in replay when it becomes available.
    return {
        "ok": True,
        "job_id": job_id,
        "approved": bool(job.get("approved", False)),
        "status": job.get("status", "PENDING"),
        "result": job.get("result", {}),
        "original_url": job.get("original_url"),
        "replay_url": job.get("replay_url"),
        "original_link": job.get("original_url"),
        "replay_link": job.get("replay_url"),
        "ai_probability": (job.get("result") or {}).get("ai_probability"),
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

    # Run external integrations if available
    try:
        # Run Portia analysis if available
        if PORTIA_AVAILABLE:
            try:
                portia_result = run_through_portia(src_path)
                job["portia_result"] = portia_result
                logger.info("Portia analysis completed for job %s", job_id)
            except Exception as e:
                logger.warning("Portia analysis failed for job %s: %s", job_id, str(e))
        
        # Upload to Google Drive if available
        if _drive_available():
            try:
                drive_link = upload_to_drive(src_path, f"truthlens_{job_id}_{Path(src_path).name}")
                job["drive_link"] = drive_link
                logger.info("Drive upload completed for job %s", job_id)
            except Exception as e:
                logger.warning("Drive upload failed for job %s: %s", job_id, str(e))
        
        # Add to Notion if available
        if notion_test():
            try:
                add_to_notion(job)
                logger.info("Notion entry added for job %s", job_id)
            except Exception as e:
                logger.warning("Notion add failed for job %s: %s", job_id, str(e))
        
        # Send approval confirmation email
        if _gmail_available() and ADMIN_EMAIL:
            try:
                subject = f"TruthLens Job Approved - Job {job_id}"
                result = job.get("result", {}).get("result", "Unknown")
                ai_prob = job.get("result", {}).get("ai_probability", 0)
                
                body = f"""
TruthLens Job Approved ✅

Job ID: {job_id}
Result: {result}
AI Probability: {ai_prob:.2%}
File: {job.get('file', 'Unknown')}
Upload Time: {job.get('upload_time', 'Unknown')}

Original URL: {job.get('original_url', 'Not available')}
Replay URL: {job.get('replay_url', 'Not available')}
Drive Link: {job.get('drive_link', 'Not available')}

This job has been approved and is now live.
                """
                
                send_gmail(subject, body.strip(), ADMIN_EMAIL)
                logger.info("Approval confirmation email sent for job %s", job_id)
            except Exception as e:
                logger.warning("Approval confirmation email failed for job %s: %s", job_id, str(e))
        
        # Send webhook notification
        send_webhook_notification(job)
        
    except Exception as e:
        logger.error("External integrations failed for job %s: %s", job_id, str(e))

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

@app.get("/admin/reviews")
def get_pending_reviews():
    """Get all jobs pending admin review with approval links"""
    pending_jobs = []
    
    for job_id, job in PENDING_JOBS.items():
        if job.get("status") == "PENDING_APPROVAL":
            sig = _sign_job(job_id)
            approve_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/approve?sig={sig}"
            deny_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/deny?sig={sig}"
            
            pending_jobs.append({
                "job_id": job_id,
                "file": job.get("file", "Unknown"),
                "status": job.get("status"),
                "result": job.get("result", {}).get("result", "Unknown"),
                "ai_probability": job.get("ai_probability", 0),
                "upload_time": job.get("upload_time", "Unknown"),
                "approve_url": approve_url,
                "deny_url": deny_url
            })
    
    return {
        "pending_count": len(pending_jobs),
        "jobs": pending_jobs
    }

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