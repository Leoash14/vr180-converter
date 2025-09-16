import os
import cv2
import math
import shutil
import subprocess
import numpy as np
import sys
import tempfile
import time
import soundfile as sf

# Optional heavy dependencies
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    from realesrgan import RealESRGANer
    HAVE_REALESRGAN = True
except Exception:
    HAVE_REALESRGAN = False

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
FOVEA_DEG = 70
VIGNETTE_STRENGTH = 0.22

# -----------------------
# MiDaS loader (optional)
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
# Temp folder helper (unique)
# -----------------------
def prepare_tmp_dir(base="tmp_vr180_work"):
    uniq = f"{base}_{int(time.time())}_{os.getpid()}"
    tmp_dir = os.path.join(tempfile.gettempdir(), uniq)
    if os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"[WARN] Could not fully delete {tmp_dir}: {e}")
    os.makedirs(os.path.join(tmp_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "right"), exist_ok=True)
    return tmp_dir

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
        if pred.ndim == 4 and pred.shape[1] >= 1:
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
# Panini + Stereographic blend
# -----------------------
def panini_stereographic_projection(img, d=PANINI_D, s=PANINI_S, blend=0.2, zoom_out=0.95):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y_i, x_i = np.indices((h, w), dtype=np.float32)
    nx = (x_i - cx) / cx
    ny = (y_i - cy) / cy
    r = np.sqrt(nx * nx + ny * ny) + 1e-8
    theta = np.arctan(r)

    # Panini
    scale_panini = (np.sin(theta) / r) * (d + (1 - d) * np.cos(theta))
    panini_x = (cx + nx * scale_panini * cx * (1 - s)).astype(np.float32)
    panini_y = (cy + ny * scale_panini * cy * (1 - s)).astype(np.float32)

    # Stereographic
    scale_stereo = np.tan(theta / 2) / (r / 2 + 1e-8)
    stereo_x = (cx + nx * scale_stereo * cx).astype(np.float32)
    stereo_y = (cy + ny * scale_stereo * cy).astype(np.float32)

    # Blend
    src_x = (1 - blend) * panini_x + blend * stereo_x
    src_y = (1 - blend) * panini_y + blend * stereo_y

    # Zoom out for comfort
    src_x = (src_x - cx) * zoom_out + cx
    src_y = (src_y - cy) * zoom_out + cy

    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    return cv2.remap(img, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# -----------------------
# Foveated blur + vignette
# -----------------------
def foveated_blur_vignette(img, fovea_deg=FOVEA_DEG, vignette_strength=VIGNETTE_STRENGTH):
    h, w = img.shape[:2]
    y, x = np.indices((h, w))
    cx, cy = w//2, h//2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    norm = dist / (max_r + 1e-8)
    fovea_frac = fovea_deg / 90.0
    mask = np.clip((norm - fovea_frac) / (1.0 - fovea_frac + 1e-8), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0,0), sigmaX=25)[..., None]
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=12)
    out = (img * (1 - mask) + blurred * mask).astype(np.uint8)
    vign = 1.0 - (mask.squeeze() * vignette_strength)
    vign = vign[..., None]
    return np.clip(out.astype(np.float32) * vign + 0.5, 0, 255).astype(np.uint8)

# -----------------------
# FOA audio conversion
# -----------------------
def stereo_to_foa(stereo_path, foa_path):
    # reads stereo WAV (or other) and writes 4-channel FOA (ACN/SN3D) float file
    data, sr = sf.read(stereo_path)
    if data.ndim == 1:
        data = np.stack([data, data], axis=-1)
    if data.ndim == 2 and data.shape[1] < 2:
        data = np.tile(data[:, :1], (1, 2))

    L = data[:, 0]
    R = data[:, 1]

    W = (L + R) / np.sqrt(2)
    X = (L - R) / np.sqrt(2)
    Y = np.zeros_like(W)
    Z = np.zeros_like(W)

    foa = np.stack([W, X, Y, Z], axis=-1)
    sf.write(foa_path, foa, sr)
    print(f"[INFO] Converted {stereo_path} -> FOA {foa_path}")

def add_foa_to_video(input_video, output_video):
    # output_video is the final FOA container we intend to produce
    base = os.path.splitext(output_video)[0]
    stereo_audio = base + "_temp_stereo.wav"
    foa_audio = base + "_temp_foa.wav"

    # Probe video for any audio stream
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,channels",
        "-select_streams", "a",
        "-of", "default=noprint_wrappers=1", input_video
    ], capture_output=True, text=True)

    if not probe.stdout.strip():
        print("[WARN] No audio stream found in input for FOA mux; copying input to output.")
        shutil.copy(input_video, output_video)
        return

    # Extract first audio stream and force stereo WAV (downmix any channel layout)
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_video,
            "-map", "0:a:0",
            "-ac", "2", "-ar", "48000",
            "-f", "wav", stereo_audio
        ], capture_output=True, text=True)
    except Exception as e:
        print("[WARN] ffmpeg extraction exception:", e)
        result = None

    if (result is None) or (result.returncode != 0) or (not os.path.exists(stereo_audio)) or (os.path.getsize(stereo_audio) < 500):
        print("[WARN] Stereo audio extraction failed or output missing; skipping FOA mux.")
        shutil.copy(input_video, output_video)
        try:
            if os.path.exists(stereo_audio):
                os.remove(stereo_audio)
        except:
            pass
        return

    # Convert stereo -> FOA (ACN/SN3D)
    try:
        stereo_to_foa(stereo_audio, foa_audio)
    except Exception as e:
        print("[WARN] FOA conversion failed:", e)
        shutil.copy(input_video, output_video)
        # cleanup
        for f in (stereo_audio, foa_audio):
            try:
                if os.path.exists(f): os.remove(f)
            except: pass
        return

    # Mux video + FOA audio (4 channels)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_video, "-i", foa_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-ar", "48000", "-ac", "4",
            output_video
        ], check=True)
    except Exception as e:
        print("[WARN] ffmpeg mux FOA failed:", e)
        shutil.copy(input_video, output_video)
        # cleanup
        for f in (stereo_audio, foa_audio):
            try:
                if os.path.exists(f): os.remove(f)
            except: pass
        return

    print(f"[INFO] Final VR180 video with FOA audio saved: {output_video}")

    # cleanup
    for f in (stereo_audio, foa_audio):
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass

# -----------------------
# Metadata injection (spatialmedia python module required)
# -----------------------
def inject_vr180_metadata(mp4_path):
    base_noext = os.path.splitext(mp4_path)[0]
    tagged_path = base_noext + "_vr180.mp4"

    # detect audio presence for spatialmedia flag
    probe = subprocess.run(
        ["ffprobe", "-i", mp4_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"],
        capture_output=True, text=True
    )
    has_audio = bool(probe.stdout.strip())

    cmd = [
        sys.executable, "-m", "spatialmedia",
        "-i",
        "--stereo=left-right",
        "--projection=equirectangular","--spatial-audio",
        mp4_path, tagged_path
    ]

    print("[INFO] Running spatialmedia for metadata injection:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not os.path.exists(tagged_path):
        raise RuntimeError("spatialmedia did not produce expected output: " + tagged_path)
    print(f"[INFO] Injected VR180 metadata: {tagged_path}")
    return os.path.abspath(tagged_path)
   

# -----------------------
# Combine frames + upscale (safe mux)
# -----------------------
def combine_and_upscale(tmp_dir, fps, upscale_to, orig_audio=None, use_bicubic=True, use_realesrgan=False):
    left_glob = os.path.join(tmp_dir, "left", "frame_%04d.png")
    right_glob = os.path.join(tmp_dir, "right", "frame_%04d.png")
    temp_left = os.path.join(tmp_dir, "temp_left.mp4")
    temp_right = os.path.join(tmp_dir, "temp_right.mp4")
    work_side = os.path.join(tmp_dir, "work_sbs.mp4")

    # Left and right videos (video-only)
    subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i", left_glob,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", temp_left], check=True)
    subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i", right_glob,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", temp_right], check=True)

    # stack side-by-side
    subprocess.run(["ffmpeg", "-y", "-i", temp_left, "-i", temp_right,
                    "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", work_side], check=True)

    final = work_side.replace(".mp4", "_final.mp4")
    tw, th = upscale_to

    if use_realesrgan and HAVE_REALESRGAN:
        print("[INFO] Upscaling with RealESRGAN (ffmpeg fallback)...")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", work_side,
                            "-vf", f"scale={tw}:{th}:flags=lanczos",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", final], check=True)
        except Exception as e:
            print("[WARN] RealESRGAN/lanczos failed, falling back to bicubic:", e)
            subprocess.run(["ffmpeg", "-y", "-i", work_side,
                            "-vf", f"scale={tw}:{th}:flags=bicubic",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", final], check=True)
    elif use_bicubic:
        subprocess.run(["ffmpeg", "-y", "-i", work_side,
                        "-vf", f"scale={tw}:{th}:flags=bicubic",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", final], check=True)
    else:
        shutil.copy(work_side, final)

    # mux original audio if present
    final_with_audio = final
    if orig_audio and os.path.exists(orig_audio):
    # First mux original stereo audio
     temp_with_audio = final.replace(".mp4", "_withaudio.mp4")
     result = subprocess.run(
        ["ffmpeg", "-y", "-i", final, "-i", orig_audio,
         "-c:v", "copy", "-c:a", "aac", "-shortest", temp_with_audio]
    )

    if result.returncode == 0:
        print(f"[INFO] Audio mux successful -> {temp_with_audio}")

        # Now extract stereo, convert to FOA, and replace
        stereo_wav = temp_with_audio.replace(".mp4", "_stereo.wav")
        foa_wav = temp_with_audio.replace(".mp4", "_foa.wav")

        subprocess.run([
            "ffmpeg", "-y", "-i", temp_with_audio,
            "-map", "0:a:0", "-ac", "2", "-ar", "48000",
            "-f", "wav", stereo_wav
        ], check=True)

        stereo_to_foa(stereo_wav, foa_wav)

        final_with_audio = final.replace(".mp4", "_withfoa.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_with_audio, "-i", foa_wav,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "4",
            final_with_audio
        ], check=True)

        print(f"[INFO] FOA mux successful -> {final_with_audio}")

        for f in (stereo_wav, foa_wav, temp_with_audio):
            try: os.remove(f)
            except: pass
    else:
        print("[WARN] Audio mux failed, using video without audio.")
        final_with_audio = final

    # Inject VR180 metadata (fisheye)
    tagged = inject_vr180_metadata(final_with_audio)

    # If we had original audio, attempt FOA conversion on the tagged file
    tagged_foa = None
    if orig_audio and os.path.exists(orig_audio):
        tagged_foa = tagged.replace(".mp4", "_foa.mp4")
        # ensure tagged file exists before FOA
        if os.path.exists(tagged):
            try:
                add_foa_to_video(tagged, tagged_foa)
                if os.path.exists(tagged_foa):
                    print("[INFO] FOA mux complete ->", tagged_foa)
                else:
                    print("[WARN] FOA mux reported success but output missing")
                    tagged_foa = None
            except Exception as e:
                print("[WARN] FOA mux failed:", e)
                tagged_foa = None
        else:
            print("[WARN] Tagged file missing, cannot do FOA:", tagged)
            tagged_foa = None

    # cleanup intermediates but keep final tagged outputs
    for f in (temp_left, temp_right, work_side, final, final_with_audio, orig_audio):
        try:
            if f and os.path.exists(f):
                if "_vr180" in os.path.basename(f) or f.endswith("_foa.mp4"):
                    continue
                os.remove(f)
        except Exception:
            pass

    return os.path.abspath(tagged_foa if (tagged_foa and os.path.exists(tagged_foa)) else tagged)

# -----------------------
# Main conversion (full pipeline)
# -----------------------
def convert_to_vr180(input_path, output_path=None, work_w=DEFAULT_WORK_W, work_h=DEFAULT_WORK_H,
                     max_seconds=None, fps=None, upscale=False, use_realesrgan=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError("Input not found: " + input_path)

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        output_path = os.path.join(outputs_dir, f"{base}_vr180.mp4")

    tmp_dir = prepare_tmp_dir()
    left_dir = os.path.join(tmp_dir, "left")
    right_dir = os.path.join(tmp_dir, "right")

    # Extract original audio (if any) to a reliable file (AAC .m4a)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ], capture_output=True, text=True)
    has_audio = bool(probe.stdout.strip())
    audio_tmp = None
    if has_audio:
        audio_tmp = os.path.join(tmp_dir, "orig_stereo.m4a")
        print("[INFO] Extracting stereo audio from input to:", audio_tmp)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-vn", "-ac", "2", "-ar", "48000",
                "-c:a", "aac", "-b:a", "192k",
                audio_tmp
            ], check=True)
            if not os.path.exists(audio_tmp) or os.path.getsize(audio_tmp) < 100:
                print("[WARN] Extracted audio file missing or tiny – treating as no audio.")
                has_audio = False
                audio_tmp = None
        except Exception as e:
            print("[WARN] Audio extraction failed:", e)
            has_audio = False
            audio_tmp = None
    else:
        print("[INFO] No audio stream detected in input.")

    # Open video and process frames into left/right
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

        left = panini_stereographic_projection(left, d=0.7, s=0.2, blend=0.2, zoom_out=0.95)
        right = panini_stereographic_projection(right, d=0.7, s=0.2, blend=0.2, zoom_out=0.95)

        left = foveated_blur_vignette(left)
        right = foveated_blur_vignette(right)

        cv2.imwrite(os.path.join(left_dir, f"frame_{saved:04d}.png"), left)
        cv2.imwrite(os.path.join(right_dir, f"frame_{saved:04d}.png"), right)

        frame_idx += 1
        saved += 1

    cap.release()

    if saved == 0:
        raise RuntimeError("No frames processed; check input or work resolution.")

    final_tagged = combine_and_upscale(
        tmp_dir,
        fps=fps,
        upscale_to=(TARGET_W, TARGET_H) if upscale else (work_w*2, work_h),
        orig_audio=audio_tmp,
        use_bicubic=True,
        use_realesrgan=use_realesrgan
    )

    # copy final to requested output path
    shutil.copy(final_tagged, output_path)

    # cleanup tmp dir
    try:
        shutil.rmtree(tmp_dir)
    except:
        pass

    print("[INFO] Conversion complete ->", output_path)

    # show metadata
    subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags",
        "-of", "default=noprint_wrappers=1:nokey=0", output_path
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
    parser.add_argument("--real_esrgan", action="store_true", help="Use RealESRGAN for upscaling")
    args = parser.parse_args()

    out = convert_to_vr180(
        args.input,
        output_path=args.output,
        work_w=args.work_w,
        work_h=args.work_h,
        max_seconds=args.seconds,
        fps=None,
        upscale=args.upscale,
        use_realesrgan=args.real_esrgan
    )
    print("Done:", out)
