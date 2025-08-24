# backend/services/gradcam.py
import os
import cv2
import torch
import numpy as np
import subprocess
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from backend.utils.gradcam_utils import VitAttentionRollout, overlay_heatmap

# Load CLIP once
print("🔁 Loading CLIP (openai/clip-vit-base-patch32)…")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("✅ CLIP loaded.")


def detect_deepfake(video_path: str, output_dir: str = "output", num_frames: int = 64):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0  # safe default

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W == 0 or H == 0:
        raise ValueError(f"❌ Could not read frame size from {video_path}")

    rollout = VitAttentionRollout(discard_ratio=0.0)

    temp_dir = os.path.join(output_dir, "frames_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    scores = []
    frame_idx = 0

    for target in tqdm(idxs, desc="Infer+CAM"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"⚠️ Skipping empty frame at index {target}")
            continue

        inputs = clip_processor(images=frame[:, :, ::-1], return_tensors="pt")
        pixel_values = inputs.get("pixel_values", None)

        if pixel_values is None or pixel_values.ndim != 4:
            print(f"⚠️ Invalid pixel_values shape at frame {target}")
            continue

        with torch.no_grad():
            vision_out = clip_model.vision_model(
                pixel_values=pixel_values, output_attentions=True
            )
            pooled = vision_out.pooler_output
            score = torch.sigmoid(pooled.mean()).item()
            scores.append(score)

            try:
                cam = rollout.generate(clip_model.vision_model, pixel_values)
                overlay = overlay_heatmap(frame, cam)

                if overlay is None:
                    continue
                if overlay.shape[1] != W or overlay.shape[0] != H:
                    overlay = cv2.resize(overlay, (W, H))

                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
                cv2.imwrite(frame_path, overlay)
                frame_idx += 1

            except Exception as e:
                print(f"⚠️ CAM generation failed at frame {target}: {e}")

    cap.release()

    out_path = os.path.join(
        output_dir,
        os.path.basename(video_path).replace(".mp4", "_with_heatmap.mp4"),
    )

    # Use ffmpeg to stitch PNGs
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    subprocess.run(cmd, check=False)

    # Cleanup temp frames
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"❌ ffmpeg failed, no video written at {out_path}")

    if len(scores) == 0:
        probs = {"real": 0.0, "fake": 1.0}
    else:
        mean_score = float(np.mean(scores))
        probs = {"real": mean_score, "fake": 1.0 - mean_score}

    return probs, out_path
