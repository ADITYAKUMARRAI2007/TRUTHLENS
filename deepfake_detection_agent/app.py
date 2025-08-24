# deepfake_detection_agent/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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

# ---- detection / replay ----
from deepfake_detection_agent.backend.services.detection import detect_ai_content
from deepfake_detection_agent.backend.services.reality_replay import run_reality_replay

# ---- Google / Notion deps (helpers kept for Portia tools too) ----
import requests
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ---- Portia orchestrator (used on APPROVE) ----
from deepfake_detection_agent.portia_agent import run_through_portia

# =================== Config ===================
PORT = int(os.getenv("PORT", 8001))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# Gmail OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
GMAIL_SENDER = os.getenv("GMAIL_SENDER")

# Drive OAuth
DRIVE_REFRESH_TOKEN = os.getenv("DRIVE_REFRESH_TOKEN")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")

# Notion
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# Owner email
OWNER_EMAIL = os.getenv("OWNER_EMAIL", GMAIL_SENDER)

# Human-in-the-loop envs
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
app = FastAPI(title="Deepfake Detection API (Portia-powered + HIL)", version="3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve anything in ./output at /media/*
app.mount("/media", StaticFiles(directory=str(OUTPUT_DIR), html=False), name="media")

@app.get("/")
def root():
    return {"message": "TruthLens API is running (Portia + Admin Approval) 🚀"}

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

# =================== Gmail helpers ===================
TOKEN_URI = "https://oauth2.googleapis.com/token"

def _gmail_creds() -> Credentials:
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GMAIL_REFRESH_TOKEN):
        raise RuntimeError("Gmail OAuth ENV not set (GOOGLE_CLIENT_ID/SECRET, GMAIL_REFRESH_TOKEN)")
    return Credentials(
        token=None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/gmail.send"],
    )

def send_gmail(to_addr: str, subject: str, html_body: str):
    creds = _gmail_creds()
    service = build("gmail", "v1", credentials=creds)
    msg = MIMEText(html_body, "html")
    msg["to"] = to_addr
    msg["from"] = GMAIL_SENDER
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    service.users().messages().send(userId="me", body={"raw": raw}).execute()

# =================== Drive helpers ===================
def _drive_creds() -> Credentials:
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and DRIVE_REFRESH_TOKEN):
        raise RuntimeError("Drive OAuth ENV not set (GOOGLE_CLIENT_ID/SECRET, DRIVE_REFRESH_TOKEN)")
    return Credentials(
        token=None,
        refresh_token=DRIVE_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )

def upload_to_drive(file_path: str, folder_id: Optional[str]) -> Optional[str]:
    if not file_path or not os.path.isfile(file_path):
        print(f"Drive upload skipped (file missing): {file_path}")
        return None
    try:
        creds = _drive_creds()
        service = build("drive", "v3", credentials=creds)
        fname = os.path.basename(file_path)
        media = MediaFileUpload(file_path, resumable=True)
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

# =================== Notion helpers (debug) ===================
NOTION_VERSION = "2022-06-28"

def _notion_headers():
    if not NOTION_API_KEY:
        raise RuntimeError("NOTION_API_KEY is missing")
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

@app.get("/debug/notion/schema")
def notion_schema():
    try:
        url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}"
        r = requests.get(url, headers=_notion_headers(), timeout=10)
        return {"status": r.status_code, "json": r.json()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug/notion/test")
def notion_test():
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
PENDING_JOBS = {}  # {job_id: {status, file, result, created_at, preview_link?, original_url?, replay_url?, approved?}}

def _sign_job(job_id: str) -> str:
    return hmac.new(APP_SECRET.encode(), job_id.encode(), hashlib.sha256).hexdigest()

def _verify_sig(job_id: str, sig: str) -> bool:
    return hmac.compare_digest(sig, _sign_job(job_id))

def _color(status: str) -> str:
    return "#22c55e" if status == "REAL" else "#ef4444" if status == "FAKE" else "#f59e0b"

def _public_media_url(filename: str) -> str:
    return f"{PUBLIC_BASE_URL}/media/{filename}"

# =================== Endpoints ===================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Human-in-the-loop flow:
      1) Save upload (publicly served at /media)
      2) Run light detection immediately (local)
      3) Email ADMIN with Approve/Deny links
      4) Return job_id with PENDING status
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(int(time.time() * 1000))  # create id first so we can name file
    safe_name = f"{job_id}_{Path(file.filename).name}"
    public_path = OUTPUT_DIR / safe_name

    async with aiofiles.open(public_path, "wb") as f:
        await f.write(await file.read())

    try:
        # Run quick local detection (supports images, pdfs & videos)
        detection = detect_ai_content(str(public_path))
        status = detection.get("result", "Unknown")

        # Optional: upload original to Drive so admin can preview externally
        preview_link = upload_to_drive(str(public_path), DRIVE_FOLDER_ID) if DRIVE_FOLDER_ID else None

        # Save pending job with public original_url
        original_url = _public_media_url(safe_name)
        PENDING_JOBS[job_id] = {
            "status": "PENDING",
            "approved": False,
            "file": str(public_path),
            "result": detection,
            "created_at": int(time.time()),
            "preview_link": preview_link,
            "original_url": original_url,
            "replay_url": None,
        }

        # Build approver links
        sig = _sign_job(job_id)
        approve_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/approve?sig={sig}"
        deny_url = f"{PUBLIC_BASE_URL}/jobs/{job_id}/deny?sig={sig}"

        # Email admin
        try:
            color = _color(status)
            html = f"""
            <div style="font-family:system-ui,sans-serif">
              <h2>New Submission Pending Review</h2>
              <p><b>Filename:</b> {Path(public_path).name}</p>
              <p><b>Preliminary Result:</b> <span style="color:{color}">{status}</span></p>
              <p><b>Scores:</b> {json.dumps({k:v for k,v in detection.items() if 'score' in k or k=='result'}, indent=2)}</p>
              <p><a href="{original_url}">Local preview</a></p>
              {'<p><a href="'+preview_link+'">Drive preview</a></p>' if preview_link else ''}
              <p>
                <a href="{approve_url}">✅ Approve</a> &nbsp;&nbsp;
                <a href="{deny_url}">❌ Deny</a>
              </p>
            </div>
            """
            send_gmail(ADMIN_EMAIL, "[TruthLens] Review Required", html)
        except Exception:
            print("Admin email send failed:", traceback.format_exc())

        return {
            "ok": True,
            "job_id": job_id,
            "status": "PENDING",
            "message": "Sent to admin for approval.",
            "prelim_result": detection.get("result"),
            "ai_probability": detection.get("ai_probability"),
        }
    except Exception:
        print("Analyze error:", traceback.format_exc())
        # best-effort cleanup if something exploded immediately
        try: public_path.unlink(missing_ok=True)
        except Exception: pass
        raise HTTPException(status_code=500, detail="Analyze failed")

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Public: only APPROVED jobs visible; others 404 (keeps gate)."""
    job = PENDING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job["status"] != "APPROVED":
        raise HTTPException(status_code=404, detail="Not found")
    # include a flat, UI-friendly shape
    return {
        "ok": True,
        "job_id": job_id,
        "approved": True,
        "status": "APPROVED",
        "result": job.get("result", {}),
        "original_url": job.get("original_url"),
        "replay_url": job.get("replay_url"),
        # compatibility aliases some UIs expect:
        "original_link": job.get("original_url"),
        "replay_link": job.get("replay_url"),
        "ai_probability": job.get("result", {}).get("ai_probability"),
        "file_name": Path(job.get("file", "")).name,
    }

@app.get("/jobs/{job_id}/approve")
def approve_job(job_id: str, sig: str):
    """
    Admin clicks secure link → run heavy pipeline → mark APPROVED.
    Ensures both original_url and replay_url are public /media links.
    """
    if not _verify_sig(job_id, sig):
        raise HTTPException(status_code=403, detail="Invalid signature")
    job = PENDING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "APPROVED":
        return {"ok": True, "job_id": job_id, "status": "APPROVED"}

    src_path = job["file"]

    # (A) Run your orchestrator (Drive, Gmail, Notion, etc) – best-effort
    try:
        portia_result = run_through_portia(src_path)
        job["portia_result"] = portia_result
    except Exception:
        print("Portia pipeline after approval failed:", traceback.format_exc())
        job["portia_result"] = {"error": "portia_failed"}

    # (B) Build a replay and copy it into ./output for public serving
    replay_tmp = None
    replay_public_filename = f"replay_{job_id}.mp4"
    replay_public_path = OUTPUT_DIR / replay_public_filename
    try:
        replay_tmp = run_reality_replay(src_path)  # returns a tmp file path
        # copy to ./output so it is served at /media
        shutil.copyfile(replay_tmp, replay_public_path)
        replay_url = _public_media_url(replay_public_filename)
    except Exception:
        print("Replay generation/copy failed:", traceback.format_exc())
        replay_url = None

    job["replay_url"] = replay_url
    job["status"] = "APPROVED"
    job["approved"] = True

    return {"ok": True, "job_id": job_id, "status": "APPROVED"}

@app.get("/jobs/{job_id}/deny")
def deny_job(job_id: str, sig: str):
    """Admin denies → mark DENIED. (We do not run heavy pipeline.)"""
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
    """Admin listing endpoint (requires x-admin-key header)."""
    if request.headers.get("x-admin-key") != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return PENDING_JOBS

@app.post("/reality-replay")
async def reality_replay(file: UploadFile = File(...)):
    # refuse images here
    if (file.content_type or "").startswith("image/") or any(
        file.filename.lower().endswith(e) for e in IMAGE_EXTS
    ):
        raise HTTPException(status_code=400, detail="Reality Replay is only for videos.")

    # Save uploaded video temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # extra safety
    if not _is_video(temp_path):
        try: os.remove(temp_path)
        except Exception: pass
        raise HTTPException(status_code=400, detail="Provided file is not a video.")

    # Run replay pipeline
    restored_path = run_reality_replay(temp_path)
    return FileResponse(restored_path, filename="reconstructed.mp4")

# Run
if __name__ == "__main__":
    uvicorn.run("deepfake_detection_agent.app:app", host="0.0.0.0", port=PORT, reload=True)