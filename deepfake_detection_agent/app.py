# deepfake_detection_agent/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn, os, aiofiles, base64, json, traceback, mimetypes, hmac, hashlib, time, shutil
from typing import Optional, Set
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---- detection (always) ----
from deepfake_detection_agent.backend.services.detection import detect_ai_content

# ---- replay (soft optional; we provide a fallback so it never crashes) ----
try:
    from deepfake_detection_agent.backend.services.reality_replay import run_reality_replay  # type: ignore
except Exception:
    def run_reality_replay(video_path: str) -> str:
        return video_path

# ---- optional deps; guard imports so deploys never crash ----
try:
    import requests
except Exception:
    requests = None  # type: ignore

try:
    from email.mime.text import MIMEText
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except Exception:
    Credentials = None  # type: ignore
    build = None        # type: ignore
    MediaFileUpload = None  # type: ignore
    MIMEText = None     # type: ignore

# ---- optional orchestrator ----
try:
    from deepfake_detection_agent.portia_agent import run_through_portia  # type: ignore
except Exception:
    def run_through_portia(path: str):
        return {"skipped": True, "reason": "portia_agent not available"}

# =================== Config ===================
PORT = int(os.getenv("PORT", "8001"))

# CORS: either explicit list via FRONTEND_ORIGIN, or regex via FRONTEND_ORIGIN_REGEX
origins_env = os.getenv("FRONTEND_ORIGIN", "")
ALLOWED_ORIGINS = [o.strip() for o in origins_env.split(",") if o.strip()]
ALLOWED_ORIGIN_REGEX = os.getenv("FRONTEND_ORIGIN_REGEX")  # e.g. r"^https://([a-z0-9-]+\.)?chainbreaker\.netlify\.app$"

# If nothing configured, default to local dev and your Netlify app (no "*", because allow_credentials=True)
if not ALLOWED_ORIGINS and not ALLOWED_ORIGIN_REGEX:
    ALLOWED_ORIGINS = ["http://localhost:5173", "https://chainbreaker.netlify.app"]

# Max upload size (MB). Default 50 MB to be Render-friendly.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Gmail OAuth (optional)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
GMAIL_SENDER = os.getenv("GMAIL_SENDER")

# Drive OAuth (optional)
DRIVE_REFRESH_TOKEN = os.getenv("DRIVE_REFRESH_TOKEN")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")

# Notion (optional)
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# Owner/admin & HIL
OWNER_EMAIL = os.getenv("OWNER_EMAIL", GMAIL_SENDER or "")
APP_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", OWNER_EMAIL)
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "dev-admin-key")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", f"http://127.0.0.1:{PORT}")

# storage dir for public media
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
app = FastAPI(title="TruthLens API (HIL + CORS + lazy models)", version="3.4")

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

@app.get("/health")
def health():
    integrations = {
        "gmail_env_ready": all([GMAIL_SENDER, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GMAIL_REFRESH_TOKEN]),
        "drive_env_ready": all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, DRIVE_REFRESH_TOKEN, DRIVE_FOLDER_ID]),
        "notion_env_ready": bool(NOTION_API_KEY and NOTION_DATABASE_ID),
        "portia_ready": bool(os.getenv("PORTIA_API_KEY")),
        "gemini_ready": bool(os.getenv("GOOGLE_API_KEY")),
        "admin_email_set": bool(ADMIN_EMAIL),
        "app_secret_set": APP_SECRET != "dev-secret-change-me",
    }
    return {"ok": True, **integrations}

# =================== Gmail helpers (optional) ===================
TOKEN_URI = "https://oauth2.googleapis.com/token"

def _gmail_available() -> bool:
    return all([
        GMAIL_SENDER, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GMAIL_REFRESH_TOKEN,
        Credentials is not None, build is not None, MIMEText is not None
    ])

def _gmail_creds():
    if not _gmail_available():
        raise RuntimeError("Gmail not configured or google libs missing.")
    return Credentials(  # type: ignore
        token=None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/gmail.send"],
    )

def send_gmail(to_addr: str, subject: str, html_body: str):
    if not _gmail_available():
        print("Gmail not available; skipping email.")
        return
    creds = _gmail_creds()
    service = build("gmail", "v1", credentials=creds)  # type: ignore
    msg = MIMEText(html_body, "html")  # type: ignore
    msg["to"] = to_addr
    msg["from"] = GMAIL_SENDER
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    service.users().messages().send(userId="me", body={"raw": raw}).execute()

# =================== Drive helpers (optional) ===================
def _drive_available() -> bool:
    return all([
        GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, DRIVE_REFRESH_TOKEN,
        build is not None, MediaFileUpload is not None, Credentials is not None
    ])

def _drive_creds():
    if not _drive_available():
        raise RuntimeError("Drive not configured or google libs missing.")
    return Credentials(  # type: ignore
        token=None,
        refresh_token=DRIVE_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )

def upload_to_drive(file_path: str, folder_id: Optional[str]) -> Optional[str]:
    if not _drive_available():
        return None
    if not file_path or not os.path.isfile(file_path):
        print(f"Drive upload skipped (file missing): {file_path}")
        return None
    try:
        creds = _drive_creds()
        service = build("drive", "v3", credentials=creds)  # type: ignore
        fname = os.path.basename(file_path)
        media = MediaFileUpload(file_path, resumable=True)  # type: ignore
        meta = {"name": fname}
        if folder_id:
            meta["parents"] = [folder_id]
        created = service.files().create(body=meta, media_body=media, fields="id, webViewLink").execute()
        file_id = created["id"]
        try:
            service.permissions().create(fileId=file_id, body={"role": "reader", "type": "anyone"}).execute()
            created = service.files().get(fileId=file_id, fields="webViewLink").execute()
        except Exception as e:
            print("Drive permission set failed (non-fatal):", e)
        return created.get("webViewLink")
    except Exception:
        print("Drive upload failed:", traceback.format_exc())
        return None

# =================== Notion helpers (optional/debug) ===================
NOTION_VERSION = "2022-06-28"

def _notion_headers():
    if not NOTION_API_KEY or requests is None:
        raise RuntimeError("NOTION_API_KEY missing or requests not installed")
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

@app.get("/debug/notion/schema")
def notion_schema():
    if requests is None:
        return {"error": "requests not installed"}
    try:
        url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}"
        r = requests.get(url, headers=_notion_headers(), timeout=10)
        return {"status": r.status_code, "json": r.json()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug/notion/test")
def notion_test():
    if requests is None:
        return {"error": "requests not installed"}
    try:
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%dT%H:%M:%S")
        url = "https://api.notion.com/v1/pages"
        body = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": {
                "Name": {"title": [{"text": {"content": "Test row from API"}}]},
                "Status": {"status": {"name": "REAL"}},
                "Run At": {"date": {"start": now_ist, "time_zone": "Asia/Kolkata"}},
                "Video score": {"number": 0.42},
                "Audio score": {"number": 0.88},
                "Original link": {"url": "https://example.com/original"},
                "Replay link": {"url": "https://example.com/replay"},
            },
        }
        r = requests.post(url, headers=_notion_headers(), data=json.dumps(body), timeout=10)
        return {"status": r.status_code, "json": r.json()}
    except Exception:
        return {"error": traceback.format_exc()}

# =================== Human-in-the-loop (Admin) ===================
PENDING_JOBS = {}  # {job_id: {...}}

def _sign_job(job_id: str) -> str:
    return hmac.new(APP_SECRET.encode(), job_id.encode(), hashlib.sha256).hexdigest()

def _verify_sig(job_id: str, sig: str) -> bool:
    return hmac.compare_digest(sig, _sign_job(job_id))

def _color(status: str) -> str:
    return "#22c55e" if status == "REAL" else "#ef4444" if status == "FAKE" else "#f59e0b"

def _public_media_url(filename: str) -> str:
    return f"{PUBLIC_BASE_URL}/media/{filename}"

# =================== Background worker ===================
def _process_job(job_id: str, public_path: Path):
    # 1) Detection with hard guard
    try:
        detection = detect_ai_content(str(public_path))
    except Exception:
        print("detect_ai_content crashed:", traceback.format_exc())
        detection = {"result": "UNKNOWN", "ai_probability": None, "error": "detect_ai_content_crashed"}

    job = PENDING_JOBS.get(job_id)
    if not job:
        return

    job["result"] = detection

    # 2) Optional: Drive upload (best-effort)
    preview_link = None
    try:
        preview_link = upload_to_drive(str(public_path), DRIVE_FOLDER_ID) if _drive_available() else None
    except Exception:
        preview_link = None
    job["preview_link"] = preview_link

    # (Keep status PENDING for HIL; if you want auto-approve in dev, uncomment)
    # job["status"] = "APPROVED"
    # job["approved"] = True
    # try:
    #     replay_tmp = run_reality_replay(str(public_path))
    #     replay_public_filename = f"replay_{job_id}.mp4"
    #     replay_public_path = OUTPUT_DIR / replay_public_filename
    #     if replay_tmp != str(replay_public_path):
    #         shutil.copyfile(replay_tmp, replay_public_path)
    #     job["replay_url"] = _public_media_url(replay_public_filename)
    # except Exception:
    #     job["replay_url"] = None

# =================== Endpoints ===================

@app.post("/analyze")
async def analyze(background: BackgroundTasks, file: UploadFile = File(...)):
    """
    1) Save upload to ./output
    2) Queue detection in background (fast return)
    3) Email admin (best-effort)
    4) Always return JSON (never raise 500)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(int(time.time() * 1000))
    safe_name = f"{job_id}_{Path(file.filename or 'upload').name}"
    public_path = OUTPUT_DIR / safe_name

    # Enforce max size (rough check based on chunks)
    bytes_limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0

    try:
        async with aiofiles.open(public_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > bytes_limit:
                    raise ValueError(f"file_too_large>{MAX_UPLOAD_MB}MB")
                await f.write(chunk)
    except Exception as e:
        print("Upload write failed:", traceback.format_exc())
        return JSONResponse({"ok": False, "error": f"upload_failed: {str(e)}"}, status_code=200)

    original_url = _public_media_url(safe_name)

    # Initialize job immediately
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

    # Kick heavy work to background (prevents proxy timeouts/502)
    background.add_task(_process_job, job_id, public_path)

    # best-effort email (never crash)
    try:
        if _gmail_available() and ADMIN_EMAIL:
            sig = _sign_job(job_id)
            approve_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/approve?sig={sig}"
            deny_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/deny?sig={sig}"
            color = _color("PROCESSING")
            html = f"""
            <div style="font-family:system-ui,sans-serif">
                <h2>New Submission Pending Review</h2>
                <p><b>Filename:</b> {Path(public_path).name}</p>
                <p><b>Preliminary Result:</b> <span style="color:{color}">PROCESSING</span></p>
                <pre style="background:#0b1020;color:#e6edf3;padding:12px;border-radius:8px">{{}}</pre>
                <p><a href="{original_url}">Local preview</a></p>
                <p><a href="{approve_url}">✅ Approve</a> &nbsp;&nbsp; <a href="{deny_url}">❌ Deny</a></p>
            </div>"""
            send_gmail(ADMIN_EMAIL, "[TruthLens] Review Required", html)
    except Exception:
        print("Admin email send failed:", traceback.format_exc())

    # Always 200 so the browser never sees “Network error/CORS”
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
    job = PENDING_JOBS.get(job_id)
    if not job or job["status"] != "APPROVED":
        # If you'd prefer to return the current state instead of 404, replace this block:
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

@app.get("/jobs/{job_id}/approve")
def approve_job(job_id: str, sig: str):
    if not _verify_sig(job_id, sig):
        raise HTTPException(status_code=403, detail="Invalid signature")
    job = PENDING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "APPROVED":
        return {"ok": True, "job_id": job_id, "status": "APPROVED"}

    src_path = job["file"]

    # orchestrator (best-effort)
    try:
        job["portia_result"] = run_through_portia(src_path)
    except Exception:
        print("Portia pipeline failed:", traceback.format_exc())
        job["portia_result"] = {"error": "portia_failed"}

    # replay -> expose via /media
    replay_public_filename = f"replay_{job_id}.mp4"
    replay_public_path = OUTPUT_DIR / replay_public_filename
    try:
        replay_tmp = run_reality_replay(src_path)  # may equal src_path in fallback
        if replay_tmp != str(replay_public_path):
            shutil.copyfile(replay_tmp, replay_public_path)
        job["replay_url"] = _public_media_url(replay_public_filename)
    except Exception:
        print("Replay generation/copy failed:", traceback.format_exc())
        job["replay_url"] = None

    job["status"] = "APPROVED"
    job["approved"] = True
    return {"ok": True, "job_id": job_id, "status": "APPROVED"}

@app.get("/jobs/{job_id}/deny")
def deny_job(job_id: str, sig: str):
    if not _verify_sig(job_id, sig):
        raise HTTPException(status_code=403, detail="Invalid signature")
    job = PENDING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["status"] = "DENIED"
    job["approved"] = False
    return {"ok": True, "job_id": job_id, "status": "DENIED"}

@app.get("/admin/jobs")
def list_jobs(request: Request):
    if request.headers.get("x-admin-key") != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return PENDING_JOBS

@app.post("/reality-replay")
async def reality_replay(file: UploadFile = File(...)):
    # refuse images here
    filename = (file.filename or "upload").lower()
    if (file.content_type or "").startswith("image/") or any(filename.endswith(e) for e in IMAGE_EXTS):
        raise HTTPException(status_code=400, detail="Reality Replay is only for videos.")

    # Save to /tmp
    temp_path = f"/tmp/{filename if any(filename.endswith(e) for e in VIDEO_EXTS) else 'upload.mp4'}"
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

# Local dev:
if __name__ == "__main__":
    uvicorn.run("deepfake_detection_agent.app:app", host="0.0.0.0", port=PORT, reload=True)