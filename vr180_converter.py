import cv2
import numpy as np
import torch
import torchvision.transforms as T
import subprocess
import os
import json

# ---------------------------
# Load MiDaS Depth Model
# ---------------------------
def load_midas_model(model_type="DPT_Large"):
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

    if "DPT" in model_type:
        return model, transform.dpt_transform
    else:
        return model, transform.small_transform

# ---------------------------
# Depth estimation
# ---------------------------
def estimate_depth(frame, model, transform):
    input_batch = transform(frame).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    return depth_map

# ---------------------------
# Stereo rendering (DIBR)
# ---------------------------
def generate_stereo(frame, depth_map, ipd=6.3, disparity_cap=1.5):
    h, w, _ = frame.shape
    max_shift = int(w * 0.01 * disparity_cap)  

    left = np.zeros_like(frame)
    right = np.zeros_like(frame)

    for y in range(h):
        shift = (depth_map[y] * max_shift).astype(np.int32)
        for x in range(w):
            dx = shift[x]
            new_x_l = x - dx
            new_x_r = x + dx
            if 0 <= new_x_l < w:
                left[y, new_x_l] = frame[y, x]
            if 0 <= new_x_r < w:
                right[y, new_x_r] = frame[y, x]

    left = cv2.inpaint(left, cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) == 0, 3, cv2.INPAINT_NS)
    right = cv2.inpaint(right, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) == 0, 3, cv2.INPAINT_NS)

    return left, right

# ---------------------------
# Projection: Equidistant fisheye
# ---------------------------
def apply_fisheye(image):
    h, w = image.shape[:2]
    K = np.array([[w/2, 0, w/2],
                  [0, w/2, h/2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([0.2, -0.05, 0, 0], dtype=np.float32)  # mild fisheye
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

# ---------------------------
# Metadata injection (VR180 fisheye)
# ---------------------------
def inject_vr180_metadata(video_path):
    out = video_path.replace(".mp4", "_vr180.mp4")
    metadata = {
        "stereo_mode": "left-right",
        "projection": "fisheye_equidistant",
        "180_mode": True
    }

    meta_file = "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f)

    cmd = ["mp4box", "-add", video_path, "-udta", meta_file, out]
    try:
        subprocess.run(cmd, check=True)
        print("[INFO] VR180 metadata injected.")
        return os.path.abspath(out)
    except Exception as e:
        print("[WARN] Metadata injection failed:", e)
        return video_path

# ---------------------------
# Main converter
# ---------------------------
def convert_to_vr180(input_path, output_path="output_vr180.mp4"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w, target_h = 7680, 3840
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    model, transform = load_midas_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_map = estimate_depth(frame_rgb, model, transform)

        left, right = generate_stereo(frame, depth_map)
        left_f = apply_fisheye(left)
        right_f = apply_fisheye(right)

        sbs = np.concatenate((left_f, right_f), axis=1)
        sbs = cv2.resize(sbs, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        out.write(sbs)

    cap.release()
    out.release()

    final_video = inject_vr180_metadata(output_path)
    return final_video