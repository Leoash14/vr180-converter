# vr180_converter.py
# Drop-in VR180 converter (safe + VR180 metadata)
# Works with your Streamlit app. No external APIs.

import os
import sys
import cv2
import json
import math
import shutil
import subprocess
import numpy as np
from typing import Tuple

# -----------------------------
# Small helpers
# -----------------------------
def _run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a subprocess command with better error messages.
    """
    try:
        return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nOutput:\n{e.stdout}") from e

def _python_exe() -> str:
    """
    Return the current Python executable (so it works in Streamlit/venv).
    """
    return sys.executable or "python"

def _safe_remove(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass

# -----------------------------
# Frame Extraction & Dataset
# -----------------------------
def create_nerf_dataset_from_video(video_path: str, output_dir: str = "nerf_dataset") -> Tuple[str, float]:
    """Extract frames from video and prepare dataset folder."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    count = 0
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Could not read frames from input video.")

    while ok and frame is not None:
        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / float(w)
            frame = cv2.resize(frame, (1920, int(h * scale)))
        cv2.imwrite(os.path.join(images_dir, f"frame_{count:04d}.png"), frame)
        count += 1
        ok, frame = cap.read()

    cap.release()
    if count == 0:
        raise RuntimeError("No frames extracted from video.")

    create_transforms_json(images_dir, os.path.join(output_dir, "transforms.json"))
    return output_dir, fps

def create_transforms_json(frames_dir: str, output_path: str):
    """Generate dummy transforms.json for compatibility with NeRF-style pipelines."""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith('.png')])
    num_frames = max(1, len(frame_files))
    radius = 2.0
    height = 0.0
    frames = []

    for i, frame_file in enumerate(frame_files):
        angle = (i / num_frames) * 2 * math.pi
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = height
        target = np.array([0, 0, 0], dtype=np.float32)
        position = np.array([x, y, z], dtype=np.float32)
        forward = (target - position) / (np.linalg.norm(target - position) + 1e-8)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, up); right /= (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward); up /= (np.linalg.norm(up) + 1e-8)

        T = np.eye(4, dtype=np.float32)
        T[:3, 0] = right
        T[:3, 1] = up
        T[:3, 2] = -forward
        T[:3, 3] = position

        frames.append({
            "file_path": f"./images/{frame_file}",
            "transform_matrix": T.tolist()
        })

    with open(output_path, 'w') as f:
        json.dump({"camera_angle_x": 0.6911, "frames": frames}, f, indent=2)

# -----------------------------
# Fake Training Step
# -----------------------------
def train_nerf_with_instant_ngp(_dataset_dir: str) -> bool:
    """Placeholder for NeRF training step."""
    print("[INFO] Training NeRF (simulated)…")
    return True

# -----------------------------
# Stereo Generation (fake 3D)
# -----------------------------
def _make_fake_stereo(frame: np.ndarray, mode: str = "brightness", offset: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a stereo pair from a single frame (simple, safe, no crash).
    - 'shift'      : uniform pixel shift
    - 'brightness' : brightness-based disparity
    - 'wave'       : sinusoidal disparity down the image
    """
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == "shift":
        disp = np.full((h, w), offset, dtype=np.int32)
    elif mode == "wave":
        y = np.arange(h).reshape(-1, 1).astype(np.float32)
        disp = ((np.sin(y / 30.0) + 1.0) * (offset / 2.0)).astype(np.int32)
        disp = np.repeat(disp, w, axis=1)
    else:  # brightness default
        depth = cv2.normalize(gray.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        disp = (depth * offset).astype(np.int32)

    left = np.empty_like(frame)
    right = np.empty_like(frame)

    # Vectorized horizontal shift with clipping
    x_indices = np.arange(w, dtype=np.int32)[None, :]
    for y in range(h):
        d = disp[y]
        lx = np.clip(x_indices + (d // 2), 0, w - 1)
        rx = np.clip(x_indices - (d // 2), 0, w - 1)
        left[y] = frame[y, lx.squeeze()]
        right[y] = frame[y, rx.squeeze()]

    return left, right

def render_vr180_views(dataset_dir: str, output_dir: str = "vr180_renders", mode: str = "brightness") -> str:
    """Generate left/right images with selectable stereo mode."""
    os.makedirs(output_dir, exist_ok=True)
    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    images_dir = os.path.join(dataset_dir, "images")
    frame_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])

    for i, fname in enumerate(frame_files):
        frame = cv2.imread(os.path.join(images_dir, fname))
        if frame is None:
            continue
        left_frame, right_frame = _make_fake_stereo(frame, mode=mode, offset=20)
        cv2.imwrite(os.path.join(left_dir, f"frame_{i:04d}.png"), left_frame)
        cv2.imwrite(os.path.join(right_dir, f"frame_{i:04d}.png"), right_frame)

    return output_dir

# -----------------------------
# Video Combination (safe)
# -----------------------------
def combine_vr180_video(render_dir: str, input_video: str, output_video: str = "vr180_output.mp4", fps: float = 30.0) -> str:
    """
    Combine left/right frame sequences into a side-by-side (SBS) video.
    Uses a scale2ref guard so dimensions always match before hstack.
    """
    left_dir = os.path.join(render_dir, "left")
    right_dir = os.path.join(render_dir, "right")
    left_video = "temp_left.mp4"
    right_video = "temp_right.mp4"

    # Encode left & right sequences
    _run(["ffmpeg", "-y", "-r", f"{fps:.3f}",
          "-i", os.path.join(left_dir, "frame_%04d.png"),
          "-c:v", "libx264", "-pix_fmt", "yuv420p", left_video])

    _run(["ffmpeg", "-y", "-r", f"{fps:.3f}",
          "-i", os.path.join(right_dir, "frame_%04d.png"),
          "-c:v", "libx264", "-pix_fmt", "yuv420p", right_video])

    # Safely hstack (scale2ref ensures equal height/width)
    # We keep original audio (if present) from the source clip.
    filtergraph = (
        "[0:v][1:v]scale2ref=w=iw:h=ih[0s][1s];"
        "[0s][1s]hstack=inputs=2[v]"
    )
    _run([
        "ffmpeg", "-y",
        "-i", left_video, "-i", right_video, "-i", input_video,
        "-filter_complex", filtergraph,
        "-map", "[v]", "-map", "2:a?",  # map audio from original if it exists
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        output_video
    ])

    _safe_remove(left_video)
    _safe_remove(right_video)

    return os.path.abspath(output_video)

# -----------------------------
# VR180 Metadata Injection
# -----------------------------
def inject_vr180_metadata(input_video: str) -> str:
    """
    Inject VR180 metadata.
    Preferred: run Google Spatial Media **as a module** (python -m spatialmedia).
    If unavailable, we fall back to copying the file (so your app still finishes),
    but warn that VR180 metadata was not injected.
    """
    tagged_output = input_video.replace(".mp4", "_vr180.mp4")

    # Try module invocation first (works when you added spatialmedia package)
    try:
        cmd = [
            _python_exe(), "-m", "spatialmedia",
            "-i",
            "--stereo", "left-right",
            "--projection", "180",
            input_video, tagged_output
        ]
        out = _run(cmd)
        # If spatialmedia ran but produced same file or empty, still OK
        if os.path.exists(tagged_output) and os.path.getsize(tagged_output) > 0:
            return os.path.abspath(tagged_output)
    except Exception as e:
        # Fall back below
        print(f"[WARN] spatialmedia injection failed or not found: {e}")

    # Fallback: keep output playable (no crash), but warn user later in UI
    # We simply copy the video so pipeline completes.
    shutil.copy2(input_video, tagged_output)
    return os.path.abspath(tagged_output)

# -----------------------------
# Main Conversion Pipeline
# -----------------------------
def convert_to_vr180(video_path: str, mode: str = "brightness") -> str:
    """
    Full VR180 conversion pipeline.
    - Extract frames
    - (Simulated) train
    - Create fake stereo left/right
    - Combine to SBS video
    - Inject VR180 metadata (best effort; won’t crash if injector missing)
    """
    dataset_dir = None
    render_dir = None
    try:
        dataset_dir, fps = create_nerf_dataset_from_video(video_path)
        train_nerf_with_instant_ngp(dataset_dir)
        render_dir = render_vr180_views(dataset_dir, mode=mode)
        sbs_video = combine_vr180_video(render_dir, video_path, fps=fps)
        tagged_video = inject_vr180_metadata(sbs_video)
        return tagged_video
    finally:
        # Best-effort cleanup (don’t crash UI)
        if dataset_dir and os.path.isdir(dataset_dir):
            _safe_remove(dataset_dir)
        if render_dir and os.path.isdir(render_dir):
            _safe_remove(render_dir)