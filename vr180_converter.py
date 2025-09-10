import os
import cv2
import subprocess
import numpy as np
import json
import math
import shutil


def create_nerf_dataset_from_video(video_path, output_dir="nerf_dataset"):
"""Extract frames from video and prepare dataset folder."""
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
    height, width = frame.shape[:2]  
    if width > 1920:  
        scale = 1920 / width  
        frame = cv2.resize(frame, (1920, int(height * scale)))  
    cv2.imwrite(os.path.join(images_dir, f"frame_{count:04d}.png"), frame)  
    count += 1  

cap.release()  
create_transforms_json(images_dir, os.path.join(output_dir, "transforms.json"), fps)  
return output_dir, fps

def create_transforms_json(frames_dir, output_path, fps=30):
"""Generate dummy transforms.json for compatibility with NeRF pipelines."""
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
num_frames = len(frame_files)
radius = 2.0
height = 0.0
frames = []

for i, frame_file in enumerate(frame_files):  
    angle = (i / num_frames) * 2 * math.pi  
    x = radius * math.cos(angle)  
    z = radius * math.sin(angle)  
    y = height  
    target = np.array([0, 0, 0])  
    position = np.array([x, y, z])  
    forward = (target - position) / np.linalg.norm(target - position)  
    up = np.array([0, 1, 0])  
    right = np.cross(forward, up)  
    right /= np.linalg.norm(right)  
    up = np.cross(right, forward)  
    up /= np.linalg.norm(up)  

    transform_matrix = np.eye(4)  
    transform_matrix[:3, 0] = right  
    transform_matrix[:3, 1] = up  
    transform_matrix[:3, 2] = -forward  
    transform_matrix[:3, 3] = position  

    frames.append({  
        "file_path": f"./images/{frame_file}",  
        "transform_matrix": transform_matrix.tolist()  
    })  

with open(output_path, 'w') as f:  
    json.dump({"camera_angle_x": 0.6911, "frames": frames}, f, indent=2)

Fake Training Step

def train_nerf_with_instant_ngp(dataset_dir):
"""Placeholder for NeRF training step."""
print("[INFO] Training NeRF (simulated)...")
return True

Stereo Generation Helpers


def create_views(frame, mode="brightness", offset=15):
"""Generate stereo pair using different fake-3D modes (vectorized)."""
h, w, _ = frame.shape
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

if mode == "shift":  
    disp = np.full_like(gray, offset)  
elif mode == "brightness":  
    depth = cv2.normalize(gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)  
    disp = (depth * offset).astype(np.int32)  
elif mode == "wave":  
    y_indices = np.arange(h).reshape(-1, 1)  
    disp = ((np.sin(y_indices / 30.0) + 1) * offset / 2).astype(np.int32)  
    disp = np.repeat(disp, w, axis=1)  
else:  
    disp = np.zeros_like(gray)  

x = np.arange(w)  
left = np.zeros_like(frame)  
right = np.zeros_like(frame)  
for y in range(h):  
    dx = disp[y]  
    lx = np.clip(x + dx // 2, 0, w - 1)  
    rx = np.clip(x - dx // 2, 0, w - 1)  
    left[y] = frame[y, lx]  
    right[y] = frame[y, rx]  

return left, right

def render_vr180_views(dataset_dir, output_dir="vr180_renders", mode="brightness"):
"""Generate left/right images with selectable stereo mode."""
os.makedirs(output_dir, exist_ok=True)
left_dir = os.path.join(output_dir, "left")
right_dir = os.path.join(output_dir, "right")
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

images_dir = os.path.join(dataset_dir, "images")  
frame_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])  

for i, frame_file in enumerate(frame_files):  
    frame = cv2.imread(os.path.join(images_dir, frame_file))  
    if frame is None:  
        continue  

    left_frame, right_frame = create_views(frame, mode=mode, offset=20)  

    cv2.imwrite(os.path.join(left_dir, f"frame_{i:04d}.png"), left_frame)  
    cv2.imwrite(os.path.join(right_dir, f"frame_{i:04d}.png"), right_frame)  

return output_dir

Video Combination

def combine_vr180_video(render_dir, input_video, output_video="vr180_output.mp4", fps=30):
"""Combine left/right frames into side-by-side VR180 video."""
left_dir = os.path.join(render_dir, "left")
right_dir = os.path.join(render_dir, "right")
left_video = "temp_left.mp4"
right_video = "temp_right.mp4"

ffmpeg_cmd_left = [  
    "ffmpeg", "-y", "-r", str(fps),  
    "-i", os.path.join(left_dir, "frame_%04d.png"),  
    "-c:v", "libx264", "-pix_fmt", "yuv420p", left_video  
]  
ffmpeg_cmd_right = [  
    "ffmpeg", "-y", "-r", str(fps),  
    "-i", os.path.join(right_dir, "frame_%04d.png"),  
    "-c:v", "libx264", "-pix_fmt", "yuv420p", right_video  
]  
subprocess.run(ffmpeg_cmd_left, check=True)  
subprocess.run(ffmpeg_cmd_right, check=True)  

output_cmd = [  
    "ffmpeg", "-y", "-i", left_video, "-i", right_video, "-i", input_video,  
    "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",  
    "-map", "[v]", "-map", "2:a?",  
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", output_video  
]  
subprocess.run(output_cmd, check=True)  

os.remove(left_video)  
os.remove(right_video)  

return os.path.abspath(output_video)

VR180 Metadata Injection

def inject_vr180_metadata(input_video):
"""
Inject VR180 metadata using Google's Spatial Media Metadata Injector.
"""
tagged_output = input_video.replace(".mp4", "_vr180.mp4")

cmd = [  
    "python", "-m", "spatialmedia.__main__",  
    "-i", "--stereo=left-right", "--projection=equirectangular",  
    input_video, tagged_output  
]  

try:  
    subprocess.run(cmd, check=True)  
except subprocess.CalledProcessError as e:  
    print("[WARN] Metadata injection failed:", e)  
    return input_video  # fallback if injection fails  

return os.path.abspath(tagged_output)


Main Conversion Pipeline

def convert_to_vr180(video_path, mode="brightness"):
"""Full VR180 conversion pipeline."""
dataset_dir, fps = create_nerf_dataset_from_video(video_path)
train_nerf_with_instant_ngp(dataset_dir)
render_dir = render_vr180_views(dataset_dir, mode=mode)
output_video = combine_vr180_video(render_dir, video_path, fps=fps)
tagged_output = inject_vr180_metadata(output_video)

shutil.rmtree(dataset_dir)  
shutil.rmtree(render_dir)  

return tagged_output



