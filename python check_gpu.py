import os
import cv2
import math
import shutil
import subprocess
import numpy as np

# Optional heavy dependency
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# -----------------------
# CONFIG DEFAULTS
# -----------------------
DEFAULT_WORK_W = 1280
DEFAULT_WORK_H = 720
TARGET_W = 7680
TARGET_H = 3840
DISPARITY_CAP_PCT = 0.02
PANINI_D = 0.7
PANINI_S = 0.2
FOVEA_RADIUS_FRAC = 0.33
VIGNETTE_STRENGTH = 0.22

# -----------------------
# MiDaS loader
# -----------------------
MIDAS_MODEL = None
MIDAS_TRANSFORM = None
MIDAS_DEVICE = None

def try_load_midas(model_type="MiDaS_small"):
    global MIDAS_MODEL, MIDAS_TRANSFORM, MIDAS_DEVICE
    if not HAVE_TORCH:
        return
    try:
        if MIDAS_MODEL is None:
            MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", model_type)
            MIDAS_MODEL.eval()
            MIDAS_TRANSFORM = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type in ["DPT_Large", "DPT_Hybrid"]:
                MIDAS_TRANSFORM = MIDAS_TRANSFORM.dpt_transform
            else:
                MIDAS_TRANSFORM = MIDAS_TRANSFORM.small_transform
            MIDAS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MIDAS_MODEL.to(MIDAS_DEVICE)
    except Exception as e:
        print("[WARN] MiDaS load failed:", e)
        MIDAS_MODEL = None
        MIDAS_TRANSFORM = None
        MIDAS_DEVICE = None

try_load_midas()

# -----------------------
# Depth estimation
# -----------------------
def estimate_depth_midas(frame, downsize=(384, 216)):
    if MIDAS_MODEL is None or MIDAS_TRANSFORM is None:
        return None
    try:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, downsize, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        input_batch = MIDAS_TRANSFORM(img)
        while input_batch.ndim > 4:
            input_batch = input_batch.squeeze(0)
        if input_batch.ndim == 3:
            input_batch = input_batch.unsqueeze(0)

        input_batch = input_batch.to(MIDAS_DEVICE)

        with torch.no_grad():
            pred = MIDAS_MODEL(input_batch)

        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0, :, :]
        if pred.ndim == 4 and pred.shape[1] != 1:
            pred = pred[:, 0, :, :]
        if pred.ndim == 3:
            pred = pred.squeeze(0)

        depth = pred.cpu().numpy()
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth
    except Exception as e:
        print("[WARN] MiDaS depth estimation error:", e)
        return None

def estimate_depth_fast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.bilateralFilter(gray, 7, 75, 75)
    depth = cv2.normalize(small.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return 1.0 - depth

def estimate_depth(frame):
    mid = estimate_depth_midas(frame)
    if mid is not None:
        return mid
    return estimate_depth_fast(frame)

# -----------------------
# Stereo generation
# -----------------------
def compute_disparity_map(depth, base_percent=0.08, cap_pct=DISPARITY_CAP_PCT):
    h, w = depth.shape
    base_shift = max(1, int(w * base_percent))
    disp = (depth * base_shift).astype(np.int32)
    cap_px = max(1, int(w * cap_pct))
    return np.clip(disp, -cap_px, cap_px)

def create_stereo_pair_vectorized(frame, depth):
    h, w = frame.shape[:2]
    disp = compute_disparity_map(depth)
    row_shift = disp.mean(axis=1).astype(np.int32)

    left = np.empty_like(frame)
    right = np.empty_like(frame)

    for y in range(h):
        s = int(row_shift[y])
        if s == 0:
            left[y] = frame[y]
            right[y] = frame[y]
        else:
            left[y] = np.roll(frame[y], -s, axis=0)
            right[y] = np.roll(frame[y], s, axis=0)

    return left, right

# -----------------------
# Inpaint occlusions
# -----------------------
def inpaint_occlusions(img):
    if img is None:
        return img
    mask = np.all(img == 0, axis=2).astype(np.uint8) * 255
    if mask.sum() == 0:
        return img
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    try:
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    except Exception as e:
        print("[WARN] inpaint failed:", e)
        return img

# -----------------------
# Panini projection
# -----------------------
def panini_projection(img, d=PANINI_D, s=PANINI_S):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y_i, x_i = np.indices((h, w), dtype=np.float32)
    nx = (x_i - cx) / cx
    ny = (y_i - cy) / cy
    r = np.sqrt(nx * nx + ny * ny) + 1e-8
    theta = np.arctan(r)
    scale = (np.sin(theta) / r) * (d + (1 - d) * np.cos(theta))
    src_x = (cx + nx * scale * cx * (1 - s)).astype(np.float32)
    src_y = (cy + ny * scale * cy * (1 - s)).astype(np.float32)
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    return cv2.remap(img, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# -----------------------
# Foveated blur + vignette
# -----------------------
def foveated_blur_vignette(img, fovea_frac=FOVEA_RADIUS_FRAC, vignette_strength=VIGNETTE_STRENGTH):
    h, w = img.shape[:2]
    y, x = np.indices((h, w))
    cx, cy = w//2, h//2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    norm = dist / (max_r + 1e-8)
    mask = np.clip((norm - fovea_frac) / (1.0 - fovea_frac + 1e-8), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0,0), sigmaX=25)
    mask = mask[..., None]
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=12)
    out = (img * (1 - mask) + blurred * mask).astype(np.uint8)
    vign = 1.0 - (mask.squeeze() * vignette_strength)
    vign = vign[..., None]
    return np.clip(out.astype(np.float32) * vign + 0.5, 0, 255).astype(np.uint8)

# -----------------------
# Metadata injection
# -----------------------
def inject_vr180_metadata(mp4_path):
    tagged_path = mp4_path.replace(".mp4", "_vr180.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-c", "copy",
        "-metadata:s:v:0", "stereo_mode=left_right",
        "-metadata:s:v:0", "ProjectionType=equirectangular",
        "-metadata:s:v:0", "spherical_video=1",
        tagged_path
    ]
    subprocess.run(cmd, check=True)
    return tagged_path

# -----------------------
# Combine frames + upscale
# -----------------------
def combine_and_upscale(tmp_dir, fps=30, upscale_to=(TARGET_W, TARGET_H), use_bicubic=True):
    left_glob = os.path.join(tmp_dir, "left", "frame_%04d.png")
    right_glob = os.path.join(tmp_dir, "right", "frame_%04d.png")
    temp_left = os.path.join(tmp_dir, "temp_left.mp4")
    temp_right = os.path.join(tmp_dir, "temp_right.mp4")
    work_side = os.path.join(tmp_dir, "work_sbs.mp4")

    subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i", left_glob,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", temp_left], check=True)
    subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i", right_glob,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", temp_right], check=True)

    subprocess.run(["ffmpeg", "-y", "-i", temp_left, "-i", temp_right,
                    "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", work_side], check=True)

    final = work_side.replace(".mp4", "_final.mp4")
    tw, th = upscale_to
    if use_bicubic:
        subprocess.run(["ffmpeg", "-y", "-i", work_side,
                        "-vf", f"scale={tw}:{th}:flags=bicubic",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", final], check=True)
    else:
        shutil.copy(work_side, final)

    # Inject VR180 metadata
    tagged = inject_vr180_metadata(final)

    for f in (temp_left, temp_right, work_side, final):
        try:
            os.remove(f)
        except:
            pass

    return os.path.abspath(tagged)

# -----------------------
# Main conversion
# -----------------------
def convert_to_vr180(input_path, output_path=None, work_w=DEFAULT_WORK_W, work_h=DEFAULT_WORK_H, max_seconds=None, fps=None, upscale=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError("Input not found: " + input_path)

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        output_path = os.path.join(outputs_dir, f"{base}_vr180.mp4")

    tmp_dir = "tmp_vr180_work"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    left_dir = os.path.join(tmp_dir, "left")
    right_dir = os.path.join(tmp_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps is None:
        fps = int(input_fps)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = min(frame_count, int(max_seconds * input_fps)) if max_seconds else frame_count

    frame_idx = 0
    saved = 0
    print(f"[INFO] convert_to_vr180: work_res={work_w}x{work_h}, fps={fps}, max_frames={max_frames}")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        work = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)
        depth = estimate_depth(work)

        left, right = create_stereo_pair_vectorized(work, depth)
        left = inpaint_occlusions(left)
        right = inpaint_occlusions(right)

        left = panini_projection(left)
        right = panini_projection(right)

        left = foveated_blur_vignette(left)
        right = foveated_blur_vignette(right)

        cv2.imwrite(os.path.join(left_dir, f"frame_{saved:04d}.png"), left)
        cv2.imwrite(os.path.join(right_dir, f"frame_{saved:04d}.png"), right)

        frame_idx += 1
        saved += 1

    cap.release()

    if saved == 0:
        raise RuntimeError("No frames processed; check input or work resolution.")

    final_tagged = combine_and_upscale(tmp_dir, fps=fps, upscale_to=(TARGET_W, TARGET_H) if upscale else (work_w*2, work_h), use_bicubic=True)
    shutil.copy(final_tagged, output_path)

    try:
        shutil.rmtree(tmp_dir)
    except:
        pass

    print("[INFO] Conversion complete ->", output_path)

    # -----------------------
    # Metadata verification
    # -----------------------
    print("\n[INFO] Checking metadata tags with ffprobe...")
    subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags",
        "-of", "default=nw=1:p=0", output_path
    ])

    return os.path.abspath(output_path)

# -----------------------
# CLI entry
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input video")
    parser.add_argument("--output", help="output video path", default=None)
    parser.add_argument("--work_w", type=int, default=DEFAULT_WORK_W)
    parser.add_argument("--work_h", type=int, default=DEFAULT_WORK_H)
    parser.add_argument("--seconds", type=float, default=None, help="process first N seconds")
    parser.add_argument("--upscale", action="store_true", help="upscale to final target")
    args = parser.parse_args()
    out = convert_to_vr180(args.input, output_path=args.output, work_w=args.work_w, work_h=args.work_h, max_seconds=args.seconds, upscale=args.upscale)
    print("Done:", out)

python -m spatialmedia -i       
--stereo=left-right        
--proj=equirectangular 
tmp_vr180_work\work_sbs_final.mp4