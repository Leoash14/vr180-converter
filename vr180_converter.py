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
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if "DPT" in model_type:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

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
def estimate_depth(frame, model, transform):
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:  # RGBA → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).unsqueeze(0)  # [1,3,H,W]

    if input_batch.ndim == 5:  
        input_batch = input_batch.squeeze(1)   # Fix accidental [1,1,3,H,W]

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    return depth_map

# -----------------------------
# Stereo Generation (DIBR)
# -----------------------------
def generate_stereo(frame, depth, max_disp=40):
    h, w = frame.shape[:2]
    x = np.arange(w)

    left = np.zeros_like(frame)
    right = np.zeros_like(frame)

    disp = (depth * max_disp).astype(np.int32)

    for y in range(h):
        dx = disp[y]
        lx = np.clip(x + dx // 2, 0, w - 1)
        rx = np.clip(x - dx // 2, 0, w - 1)
        left[y] = frame[y, lx]
        right[y] = frame[y, rx]

    return left, right

# -----------------------------
# Projection Mixing (Panini + Stereographic)
# -----------------------------
def projection_mix(img, panini_w=0.7, stereo_w=0.2):
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    cx, cy = w // 2, h // 2

    for y in range(h):
        for x in range(w):
            nx = (x - cx) / w
            ny = (y - cy) / h
            r = math.sqrt(nx * nx + ny * ny)
            if r == 0: continue
            theta = math.atan(r)
            k_panini = (theta / r) * (1 + panini_w * r * r)
            k_stereo = math.tan(theta) / r
            k = (1 - stereo_w) * k_panini + stereo_w * k_stereo
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

        depth = estimate_depth(frame, midas_model, midas_transform)
        left, right = generate_stereo(frame, depth)

        # Projection + foveated blur
        left = foveated_blur(projection_mix(left))
        right = foveated_blur(projection_mix(right))

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
# Metadata Injection (VR180)
# -----------------------------
def inject_metadata(video_path):
    out = video_path.replace(".mp4", "_vr180.mp4")
    cmd = [
        "python", "spatialmedia/spatialmedia.py",
        "-i", "--stereo=left-right", "--projection=equirectangular",
        video_path, out
    ]
    try:
        subprocess.run(cmd, check=True)
    except:
        print("[WARN] Metadata injection failed")
        return video_path
    return out

# -----------------------------
# Main Pipeline
# -----------------------------
def convert_to_vr180(video_path, upscale=False):
    images_dir, fps = extract_frames(video_path)
    render_dir = render_vr180(images_dir)
    output_video = combine_vr180(render_dir, fps=fps)
    tagged = inject_metadata(output_video)

    shutil.rmtree(images_dir)
    shutil.rmtree(render_dir)
    return tagged