from fastapi import FastAPI, UploadFile, File
from services.detection import detect_deepfake
from services.slack import send_slack_alert
from services.replay import generate_replay
import shutil, os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Deepfake Detection Agent Running 🚀"}

@app.post("/scan-video/")
async def scan_video(file: UploadFile = File(...)):
    # Save video temporarily
    contents = await file.read()
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(contents)

    # Run detection
    result = detect_deepfake(video_path)

    # If fake → send Slack alert
    if result["label"] == "FAKE":
        send_slack_alert(file.filename, result["confidence"])

    return {"filename": file.filename, "result": result}

@app.post("/reality-replay/")
async def reality_replay(file: UploadFile = File(...)):
    temp_path = f"./output/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    replay_path = f"./output/replay_{file.filename}"
    generate_replay(temp_path, replay_path)

    return {"replay_video": replay_path}
