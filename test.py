import cv2
import sys

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Could not open video: {video_path}")
    sys.exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ Video opened successfully!")
print(f"Frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")

cap.release()
