import cv2
import torch
import numpy as np
import ffmpeg
from torchvision.transforms import Compose
from torchvision.transforms import transforms as T

# ✅ Load MiDaS depth model (small for speed)
def load_midas_model(model_type="MiDaS_small"):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        midas_transform = transform.dpt_transform
    else:
        midas_transform = transform.small_transform

    return midas, midas_transform


midas_model, midas_transform = load_midas_model()


# ✅ Depth estimation
def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        prediction = midas_model(input_batch)

        # Fix: only unsqueeze if 3D
        if prediction.ndim == 3:  # [1, H, W]
            prediction = prediction.unsqueeze(1)  # → [1, 1, H, W]

        depth = torch.nn.functional.interpolate(
            prediction,
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    return depth


# ✅ Create stereo pair using depth-based shift
def create_stereo_pair(frame, depth, ipd=0.06, max_disp=1.5):
    h, w, _ = frame.shape
    disparity = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    disparity = disparity * (max_disp * w / 100)

    left_img = np.zeros_like(frame)
    right_img = np.zeros_like(frame)

    for y in range(h):
        for x in range(w):
            disp = int(disparity[y, x] * ipd * 100)
            if x - disp >= 0:
                left_img[y, x - disp] = frame[y, x]
            if x + disp < w:
                right_img[y, x + disp] = frame[y, x]

    return left_img, right_img


# ✅ Projection: Panini + stereographic mix → equidistant fisheye
def panini_stretch(img, d=0.7, s=0.2):
    h, w = img.shape[:2]
    fov = np.pi  # 180°
    out = np.zeros_like(img)
    cx, cy = w // 2, h // 2

    for y in range(h):
        for x in range(w):
            nx = (x - cx) / cx
            ny = (y - cy) / cy
            r = np.sqrt(nx * nx + ny * ny)
            if r == 0:
                out[y, x] = img[cy, cx]
                continue

            theta = np.arctan(r)
            scale = (np.sin(theta) / r) * (d + (1 - d) * np.cos(theta))
            src_x = int(cx + nx * scale * cx * (1 - s))
            src_y = int(cy + ny * scale * cy * (1 - s))
            if 0 <= src_x < w and 0 <= src_y < h:
                out[y, x] = img[src_y, src_x]

    return out


# ✅ Foveated blur for periphery
def foveated_blur(img, start_deg=70):
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    mask = np.zeros((h, w), np.float32)

    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask[y, x] = dist / max_radius

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15)
    blurred = cv2.GaussianBlur(img, (25, 25), 15)

    out = np.uint8(img * (1 - mask[..., None]) + blurred * mask[..., None])
    return out


# ✅ VR180 Converter
def convert_to_vr180(input_path, output_path, upscale=True):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    target_w, target_h = 7680, 3840
    out = cv2.VideoWriter(output_path, fourcc, 30, (target_w, target_h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        depth = estimate_depth(frame)
        left, right = create_stereo_pair(frame, depth)

        left = panini_stretch(left)
        right = panini_stretch(right)

        left = foveated_blur(left)
        right = foveated_blur(right)

        stereo = np.hstack((left, right))

        if upscale:
            stereo = cv2.resize(stereo, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        out.write(stereo)

    cap.release()
    out.release()

    # ✅ Inject VR180 metadata
    ffmpeg.input(output_path).output(
        output_path.replace(".mp4", "_vr180.mp4"),
        vf="v360=input=equirect:output=hequirect",
        metadata="stereo_mode=left_right",
    ).run(overwrite_output=True)