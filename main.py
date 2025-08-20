from ultralytics import YOLO
import cv2, os, subprocess, uuid, re

# Load custom YOLOv8 model
model = YOLO("weights/best.pt")

# -----------------------------
# Run YOLO on an image
# -----------------------------
def process_image(file_path, output_folder):
    """Run YOLO on an image and save result"""
    results = model(file_path)
    output_img = results[0].plot()

    name, ext = os.path.splitext(os.path.basename(file_path))
    output_path = os.path.join(output_folder, f"{name}_detected{ext}")
    cv2.imwrite(output_path, output_img)

    return output_path


# -----------------------------
# Run YOLO on a video
# -----------------------------
def process_video(file_path, output_folder):
    """Run YOLO on video and re-encode for web"""
    cap = cv2.VideoCapture(file_path)

    # Ensure even dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) & ~1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) & ~1
    fps = cap.get(cv2.CAP_PROP_FPS) or 24

    name, ext = os.path.splitext(os.path.basename(file_path))
    temp_path = os.path.join(output_folder, f"{name}_temp{ext}")
    output_path = os.path.join(output_folder, f"{name}_detected.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        output_frame = results[0].plot()
        out.write(output_frame)

    cap.release()
    out.release()

    # -----------------------------
    # Re-encode with FFmpeg (browser-ready H.264)
    # -----------------------------
    cmd = [
        "ffmpeg", "-y", "-i", temp_path,
        "-c:v", "libx264", "-crf", "23",
        "-preset", "fast",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)

    os.remove(temp_path)
    return output_path
