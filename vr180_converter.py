import os
import cv2
import torch
import shutil
import subprocess
import numpy as np
import json
import math
from torchvision import transforms

# -----------------------------
# Load MiDaS Depth Model (AI)
# -----------------------------
def load_midas_model(model_type="DPT_Large"):
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, transform

midas_model, midas_transform = load_midas_model()

# -----------------------------
# Frame Extraction
# -----------------------------
def extract_frames(video_path, output_dir="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > 3840:  # upscale control
            scale = 3840 / w
            frame = cv2.resize(frame, (3840, int(h * scale)))
        cv2.imwrite(os.path.join(images_dir, f"frame_{count:04d}.png"), frame)
        count += 1

    cap.release()
    return images_dir, fps

# -----------------------------
# Depth Estimation (MiDaS)
# -----------------------------
def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = midas_model(input_batch)
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    return depth

# -----------------------------
# Stereo Generation
# -----------------------------
def generate_stereo(frame, depth, offset=40):
    h, w = frame.shape[:2]
    x = np.arange(w)

    left = np.zeros_like(frame)
    right = np.zeros_like(frame)

    disp = (depth * offset).astype(np.int32)

    for y in range(h):
        dx = disp[y]
        lx = np.clip(x + dx // 2, 0, w - 1)
        rx = np.clip(x - dx // 2, 0, w - 1)
        left[y] = frame[y, lx]
        right[y] = frame[y, rx]

    return left, right

# -----------------------------
# Panini Projection (Fisheye)
# -----------------------------
def panini_projection(img, d=1.0):
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    cx, cy = w // 2, h // 2
    fov = math.pi / 2
    for y in range(h):
        for x in range(w):
            nx = (x - cx) / w
            ny = (y - cy) / h
            r = math.sqrt(nx * nx + ny * ny)
            if r == 0: continue
            theta = math.atan(r)
            k = (theta / r) * (1 + d * r * r)
            sx = int(cx + nx * k * w)
            sy = int(cy + ny * k * h)
            if 0 <= sx < w and 0 <= sy < h:
                out[y, x] = img[sy, sx]
    return out

# -----------------------------
# Foveated Rendering Blur
# -----------------------------
def foveated_blur(img, strength=25):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (w//2, h//2), min(h,w)//3, 1, -1)
    blur = cv2.GaussianBlur(img, (0,0), strength)
    out = (mask[...,None]*img + (1-mask[...,None])*blur).astype(np.uint8)
    return out

# -----------------------------
# Render VR180 Views
# -----------------------------
def render_vr180(images_dir, output_dir="vr180_renders"):
    os.makedirs(output_dir, exist_ok=True)
    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])

    for i, f in enumerate(frame_files):
        frame = cv2.imread(os.path.join(images_dir, f))
        if frame is None: continue

        depth = estimate_depth(frame)
        left, right = generate_stereo(frame, depth)

        # Apply fisheye + foveated blur
        left = foveated_blur(panini_projection(left))
        right = foveated_blur(panini_projection(right))

        cv2.imwrite(os.path.join(left_dir, f"frame_{i:04d}.png"), left)
        cv2.imwrite(os.path.join(right_dir, f"frame_{i:04d}.png"), right)

    return output_dir

# -----------------------------
# Combine into Video
# -----------------------------
def combine_vr180(render_dir, fps=30, output="vr180_output.mp4"):
    left = os.path.join(render_dir, "left", "frame_%04d.png")
    right = os.path.join(render_dir, "right", "frame_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps), "-i", left,
        "-framerate", str(fps), "-i", right,
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", output
    ]
    subprocess.run(cmd, check=True)
    return os.path.abspath(output)

# -----------------------------
# Metadata Injection
# -----------------------------
def inject_metadata(video_path):
    out = video_path.replace(".mp4", "_vr180.mp4")
    cmd = [
        "python", "-m", "spatialmedia",
        "-i", "--stereo=left-right", "--projection=rectilinear",
        video_path, out
    ]
    try:
        subprocess.run(cmd, check=True)
    except:
        print("[WARN] Metadata injection failed")
        return video_path
    return out

# -----------------------------
# 8K Upscaling (Real-ESRGAN)
# -----------------------------
def upscale_to_8k(input_video, output_video="vr180_8k.mp4"):
    cmd = [
        "realesrgan-ncnn-vulkan",  # must be installed separately
        "-i", input_video,
        "-o", output_video,
        "-s", "4"  # upscale 4x
    ]
    try:
        subprocess.run(cmd, check=True)
    except:
        print("[WARN] 8K upscale skipped (Real-ESRGAN not found)")
        return input_video
    return os.path.abspath(output_video)

# -----------------------------
# Main Pipeline
# -----------------------------
def convert_to_vr180(video_path, upscale=False):
    images_dir, fps = extract_frames(video_path)
    render_dir = render_vr180(images_dir)
    output_video = combine_vr180(render_dir, fps=fps)
    tagged = inject_metadata(output_video)

    if upscale:
        tagged = upscale_to_8k(tagged)

    shutil.rmtree(images_dir)
    shutil.rmtree(render_dir)
    return tagged