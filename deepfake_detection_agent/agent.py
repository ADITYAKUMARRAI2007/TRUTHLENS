# agent.py
import os
import re
import json
import base64
import mimetypes
from typing import ClassVar, Tuple, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel

# Porter AI (Portia)
from portia import Portia, Config, Tool, ToolRunContext

# Your existing detection
from backend.services.detection import detect_ai_content

# Google SDKs
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Notion
import requests

# ---------------------------
# Load env
# ---------------------------
load_dotenv()

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")

GMAIL_ADDRESS        = os.getenv("GMAIL_ADDRESS")  # sender (must be the authorized Google account)
DRIVE_FOLDER_ID      = os.getenv("DRIVE_FOLDER_ID")  # target folder id in Drive

NOTION_API_KEY       = os.getenv("NOTION_API_KEY")
NOTION_PAGE_ID       = os.getenv("NOTION_PAGE_ID")  # a page (or block) to append results into

OWNER_EMAIL          = os.getenv("OWNER_EMAIL")  # fallback recipient for reports (optional)

# ---------------------------
# Google OAuth helper
# ---------------------------
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def google_creds(scopes: list[str]) -> Credentials:
    """
    Server-side OAuth using a refresh_token. Make sure the refresh token was minted
    with *these exact scopes* (gmail.send / drive.file).
    """
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REFRESH_TOKEN):
        raise RuntimeError("Missing GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET/GOOGLE_REFRESH_TOKEN in .env")
    creds = Credentials(
        None,
        refresh_token=GOOGLE_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=scopes,
    )
    creds.refresh(Request())
    return creds

# ---------------------------
# Schemas
# ---------------------------
class DetectContentInput(BaseModel):
    file_path: str

class DetectContentOutput(BaseModel):
    result: str
    video_score: float | None = None
    audio_score: float | None = None

# ---------------------------
# Tool 1: Deepfake Detection
# ---------------------------
class DetectContentTool(Tool):
    id: ClassVar[str] = "detect_content"
    name: ClassVar[str] = "detect_content"
    description: ClassVar[str] = "Detect whether the given file is AI-generated or deepfake."

    input_schema: ClassVar[Tuple[str, str]] = (
        json.dumps({
            "title": "DetectContentInput",
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Full path of the file to analyze"}
            },
            "required": ["file_path"]
        }),
        "Path to the file to analyze",
    )

    output_schema: ClassVar[Tuple[str, str]] = (
        json.dumps({
            "title": "DetectContentOutput",
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "video_score": {"type": "number"},
                "audio_score": {"type": "number"}
            },
            "required": ["result"]
        }),
        "Result of deepfake detection",
    )

    def run(self, ctx: ToolRunContext, **kwargs) -> DetectContentOutput:
        file_path = kwargs.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return DetectContentOutput(result="❌ file not found")
        raw = detect_ai_content(file_path)
        if isinstance(raw, dict):
            return DetectContentOutput(
                result=raw.get("result", "Unknown"),
                video_score=raw.get("video_score"),
                audio_score=raw.get("audio_score"),
            )
        return DetectContentOutput(result=str(raw))

# ---------------------------
# Tool 2: Google Drive Upload
# ---------------------------
class DriveTool(Tool):
    id: ClassVar[str] = "upload_drive"
    name: ClassVar[str] = "upload_drive"
    description: ClassVar[str] = "Upload analyzed videos & reports to Google Drive."

    input_schema: ClassVar[Tuple[str, str]] = (
        json.dumps({
            "title": "DriveInput",
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "folder_id": {"type": "string"}
            },
            "required": ["file_path"]
        }),
        "Upload file to Drive"
    )

    output_schema: ClassVar[Tuple[str, str]] = (
        json.dumps({
            "title": "DriveOutput",
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "link": {"type": "string"}
            },
            "required": ["file_id", "link"]
        }),
        "Drive file link"
    )

    def run(self, ctx: ToolRunContext, **kwargs):
        file_path = kwargs.get("file_path")
        folder_id = kwargs.get("folder_id") or DRIVE_FOLDER_ID
        if not file_path or not os.path.exists(file_path):
            raise RuntimeError("upload_drive: file_path invalid or not found")
        if not folder_id:
            raise RuntimeError("upload_drive: missing folder_id (set DRIVE_FOLDER_ID in .env or pass it)")

        creds = google_creds(DRIVE_SCOPES)
        drive = build("drive", "v3", credentials=creds)

        fname = os.path.basename(file_path)
        mime, _ = mimetypes.guess_type(file_path)
        if not mime:
            mime = "application/octet-stream"

        media = MediaFileUpload(file_path, mimetype=mime, resumable=True)
        body = {"name": fname, "parents": [folder_id]}
        file = drive.files().create(body=body, media_body=media, fields="id, webViewLink").execute()

        # Make link-viewable (optional)
        try:
            drive.permissions().create(
                fileId=file["id"],
                body={"role": "reader", "type": "anyone"},
            ).execute()
        except Exception:
            pass

        return {"file_id": file["id"], "link": file.get("webViewLink", "")}

# ---------------------------
# Tool 3: Gmail Sender
# ---------------------------
class GmailTool(Tool):
    id: ClassVar[str] = "send_gmail"
    name: ClassVar[str] = "send_gmail"
    description: ClassVar[str] = "Send detection results via Gmail."

    input_schema: ClassVar[Tuple[str, str]] = (
        json.dumps({
            "title": "GmailInput",
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
        }),
        "Email details"
    )

    output_schema: ClassVar[Tuple[str, str]] = (
        json.dumps({
            "title": "GmailOutput",
            "type": "object",
            "properties": {"status": {"type": "string"}, "id": {"type": "string"}}
        }),
        "Status of email send"
    )

    def run(self, ctx: ToolRunContext, **kwargs):
        to = kwargs.get("to") or OWNER_EMAIL
        subject = kwargs.get("subject")
        html_body = kwargs.get("body")

        if not to:
            raise RuntimeError("send_gmail: missing 'to' and OWNER_EMAIL not set")
        if not (subject and html_body):
            raise RuntimeError("send_gmail: subject/body required")
        if not GMAIL_ADDRESS:
            raise RuntimeError("send_gmail: GMAIL_ADDRESS not set")

        creds = google_creds(GMAIL_SCOPES)
        service = build("gmail", "v1", credentials=creds)

        raw_mime = (
            f"From: {GMAIL_ADDRESS}\r\n"
            f"To: {to}\r\n"
            f"Subject: {subject}\r\n"
            "Content-Type: text/html; charset=UTF-8\r\n\r\n"
            f"{html_body}"
        ).encode("utf-8")

        raw_b64 = base64.urlsafe_b64encode(raw_mime).decode("utf-8")
        sent = service.users().messages().send(userId="me", body={"raw": raw_b64}).execute()
        return {"status": "sent", "id": sent.get("id", "")}

# ---------------------------
# Tool 4: Notion Logger (append to a Page)
# ---------------------------
class NotionTool(Tool):
    id: ClassVar[str] = "log_notion"
    name: ClassVar[str] = "log_notion"
    description: ClassVar[str] = "Log detection results into a Notion page (append block)."

    input_schema: ClassVar[Tuple[str, str]] = (
        json.dumps({
            "title": "NotionInput",
            "type": "object",
            "properties": {
                "page_id": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["page_id", "content"]
        }),
        "Log content into Notion"
    )

    output_schema: ClassVar[Tuple[str, str]] = (
        json.dumps({"title": "NotionOutput", "type": "object", "properties": {"status": {"type": "string"}}}),
        "Status of Notion log"
    )

    def run(self, ctx: ToolRunContext, **kwargs):
        page_id = kwargs.get("page_id") or NOTION_PAGE_ID
        content = kwargs.get("content") or ""
        if not (NOTION_API_KEY and page_id):
            return {"status": "skipped (missing NOTION_API_KEY or page_id)"}

        headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }
        payload = {
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content}}]
                    }
                }
            ]
        }
        r = requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=headers, json=payload, timeout=30
        )
        return {"status": "logged" if r.ok else f"failed ({r.status_code})"}

# ---------------------------
# Register Agent (Porter AI)
# ---------------------------
agent = Portia(
    tools=[DetectContentTool(), DriveTool(), GmailTool(), NotionTool()],
    config=Config(
        instructions=(
            "You are TruthLens, a Porter AI agent.\n"
            "Workflow for any path like '/path/video.mp4':\n"
            "1) Call detect_content with {file_path}.\n"
            "2) Upload the *same file* to Drive via upload_drive (use env DRIVE_FOLDER_ID if none given).\n"
            "3) Email a short HTML report via send_gmail to OWNER_EMAIL (or provided 'to').\n"
            "4) Append a summary line to Notion via log_notion.\n"
            "Always include tool args; never leave them empty."
        ),
        llm_provider="GOOGLE_GENERATIVE_AI",
        default_model="gemini-1.5-flash",
    ),
)

# ---------------------------
# (Optional) Small preprocessor so typing:
# "analyze /tmp/clip.mp4" triggers the flow.
# ---------------------------
def maybe_extract_path(q: str) -> Optional[str]:
    m = re.search(r"(/[^ \n\t]+\.(mp4|mov|avi|mkv|mp3|wav))", q, re.IGNORECASE)
    return m.group(1) if m else None

if __name__ == "__main__":
    print("✅ TruthLens (Porter AI) ready — type: analyze /full/path/video.mp4")
    while True:
        q = input("User: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        path = maybe_extract_path(q)
        if path:
            # 1) detect
            det = DetectContentTool().run(None, file_path=path)
            verdict = det.result
            vs = det.video_score
            aus = det.audio_score

            # 2) drive
            drv = DriveTool().run(None, file_path=path)
            link = drv["link"]

            # 3) email
            body = f"""
            <h2>TruthLens Report</h2>
            <p><b>File:</b> {os.path.basename(path)}</p>
            <p><b>Verdict:</b> {verdict}</p>
            <p><b>Video score:</b> {vs}</p>
            <p><b>Audio score:</b> {aus}</p>
            <p><b>Drive:</b> <a href="{link}">{link}</a></p>
            """
            GmailTool().run(None,
                to=OWNER_EMAIL or (GMAIL_ADDRESS or ""),
                subject=f"TruthLens: {os.path.basename(path)} → {verdict}",
                body=body
            )

            # 4) notion
            if NOTION_API_KEY and NOTION_PAGE_ID:
                NotionTool().run(None,
                    page_id=NOTION_PAGE_ID,
                    content=f"{os.path.basename(path)} → {verdict} | video={vs} audio={aus} | {link}"
                )

            print(json.dumps({
                "result": verdict,
                "video_score": vs,
                "audio_score": aus,
                "drive_link": link
            }, indent=2))
        else:
            # Let the agent handle free-form chat / tool calls
            print("Agent:", agent.run(q))