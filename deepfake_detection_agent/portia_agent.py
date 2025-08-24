# deepfake_detection_agent/portia_agent.py
# A lightweight “Portia-style” orchestrator with NO external dependency.
# It provides run_through_portia(file_path) which performs:
# detect -> (replay if FAKE/UNCERTAIN) -> Drive upload(s) -> Gmail -> Notion

import os, base64, json, time, traceback
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# ---- Your existing detectors ----
from deepfake_detection_agent.backend.services.detection import detect_ai_content
from deepfake_detection_agent.backend.services.reality_replay import run_reality_replay

# ---- Google / Notion deps ----
import requests
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# =================== ENV ===================
TOKEN_URI = "https://oauth2.googleapis.com/token"

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Gmail
GMAIL_REFRESH_TOKEN  = os.getenv("GMAIL_REFRESH_TOKEN")
GMAIL_SENDER         = os.getenv("GMAIL_SENDER")
OWNER_EMAIL          = os.getenv("OWNER_EMAIL", GMAIL_SENDER)

# Drive
DRIVE_REFRESH_TOKEN  = os.getenv("DRIVE_REFRESH_TOKEN")
DRIVE_FOLDER_ID      = os.getenv("DRIVE_FOLDER_ID")

# Notion
NOTION_API_KEY       = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID   = os.getenv("NOTION_DATABASE_ID")
NOTION_VERSION       = "2022-06-28"

# =================== Helpers: Gmail ===================
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

def gmail_send_html(to_addr: str, subject: str, html_body: str):
    if not to_addr:
        return
    creds = _gmail_creds()
    service = build("gmail", "v1", credentials=creds)
    msg = MIMEText(html_body, "html")
    msg["to"] = to_addr
    msg["from"] = GMAIL_SENDER
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    service.users().messages().send(userId="me", body={"raw": raw}).execute()

# =================== Helpers: Drive ===================
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

def drive_upload(file_path: Optional[str], folder_id: Optional[str]) -> Optional[str]:
    if not file_path or not os.path.isfile(file_path):
        return None
    creds = _drive_creds()
    service = build("drive", "v3", credentials=creds)
    media = MediaFileUpload(file_path, resumable=True)
    meta = {"name": os.path.basename(file_path)}
    if folder_id:
        meta["parents"] = [folder_id]
    created = service.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()
    fid = created["id"]
    try:
        service.permissions().create(fileId=fid, body={"role": "reader", "type": "anyone"}).execute()
        created = service.files().get(fileId=fid, fields="webViewLink").execute()
    except Exception:
        pass
    return created.get("webViewLink")

# =================== Helpers: Notion ===================
def _notion_headers():
    if not NOTION_API_KEY:
        raise RuntimeError("NOTION_API_KEY is missing")
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

def notion_log_row(title: str, status: str, video_score, audio_score, original_link, replay_link, ist_dt: datetime):
    if not (NOTION_API_KEY and NOTION_DATABASE_ID):
        return

    # Build properties matching your DB (Status = status type, Run At = date, Name = title)
    body = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Name": {"title": [{"text": {"content": title}}]},
            "Status": {"status": {"name": status}},
            "Run At": {"date": {"start": ist_dt.strftime("%Y-%m-%dT%H:%M:%S"), "time_zone": "Asia/Kolkata"}},
            "Video score": {"number": float(video_score) if video_score is not None else None},
            "Audio score": {"number": float(audio_score) if audio_score is not None else None},
            "Original link": {"url": original_link} if original_link else {"url": None},
            "Replay link": {"url": replay_link} if replay_link else {"url": None},
        },
    }
    requests.post("https://api.notion.com/v1/pages", headers=_notion_headers(), data=json.dumps(body), timeout=10)

# =================== Orchestrator ===================
def run_through_portia(file_path: str):
    """
    Deterministic orchestration (no external 'portia' library needed):
      1) Detect
      2) If FAKE/UNCERTAIN -> build replay
      3) Upload original (and replay if exists) to Drive
      4) Email summary via Gmail
      5) Log to Notion
    Returns a summary dict that your API can return to the frontend.
    """
    summary = {
        "file": os.path.basename(file_path),
        "result": "Unknown",
        "video_score": None,
        "audio_score": None,
        "original_drive": None,
        "replay_drive": None,
        "emailed": False,
        "notion_logged": False,
    }

    try:
        # 1) Detect
        det = detect_ai_content(file_path)
        if isinstance(det, dict):
            summary["result"] = det.get("result", "Unknown")
            summary["video_score"] = det.get("video_score")
            summary["audio_score"] = det.get("audio_score")
        else:
            summary["result"] = str(det)

        # 2) Replay if needed
        replay_path = None
        if summary["result"] in {"FAKE", "UNCERTAIN"}:
            try:
                replay_path = run_reality_replay(file_path)
            except Exception:
                print("Replay failed:", traceback.format_exc())

        # 3) Drive uploads
        try:
            summary["original_drive"] = drive_upload(file_path, DRIVE_FOLDER_ID)
        except Exception:
            print("Drive upload (original) failed:", traceback.format_exc())
        try:
            if replay_path:
                summary["replay_drive"] = drive_upload(replay_path, DRIVE_FOLDER_ID)
        except Exception:
            print("Drive upload (replay) failed:", traceback.format_exc())

        # 4) Gmail
        if OWNER_EMAIL and GMAIL_SENDER:
            color = "#22c55e" if summary["result"] == "REAL" else "#ef4444" if summary["result"] == "FAKE" else "#f59e0b"
            subject = f"[TruthLens] {summary['result']} — {summary['file']}"
            html = f"""
            <div style="font-family:system-ui,sans-serif">
              <h2>TruthLens Result: <span style="color:{color}">{summary['result']}</span></h2>
              <p><strong>Video score:</strong> {summary['video_score']}</p>
              <p><strong>Audio score:</strong> {summary['audio_score']}</p>
              <p>
                {(f'📂 <a href="{summary["original_drive"]}">Original (Drive)</a><br/>' if summary["original_drive"] else '')}
                {(f'🎬 <a href="{summary["replay_drive"]}">Replay (Drive)</a><br/>' if summary["replay_drive"] else '')}
              </p>
            </div>
            """
            try:
                gmail_send_html(OWNER_EMAIL, subject, html)
                summary["emailed"] = True
            except Exception:
                print("Gmail send failed:", traceback.format_exc())

        # 5) Notion
        try:
            ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
            notion_log_row(
                title=summary["file"],
                status=summary["result"],
                video_score=summary["video_score"],
                audio_score=summary["audio_score"],
                original_link=summary["original_drive"],
                replay_link=summary["replay_drive"],
                ist_dt=ist_now,
            )
            summary["notion_logged"] = True
        except Exception:
            print("Notion log failed:", traceback.format_exc())

        return summary

    except Exception:
        print("Orchestrator error:", traceback.format_exc())
        return {**summary, "error": "Orchestration failed"}