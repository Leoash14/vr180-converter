import cv2
import numpy as np
import math
import io
import ffmpeg
from typing import Tuple

def create_nerf_dataset_from_video_bytes(video_bytes: bytes) -> Tuple[list, int]:
    """
    Extract frames from in-memory video bytes and return as list of NumPy arrays.
    Returns frames and FPS.
    """
    # Write video bytes to OpenCV VideoCapture using temporary in-memory buffer
    temp_file = 'temp_video.mp4'
    with open(temp_file, 'wb') as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        if width > 1920:
            scale = 1920 / width
            frame = cv2.resize(frame, (1920, int(height * scale)))
        frames.append(frame)
    
    cap.release()
    return frames, fps

def train_nerf_with_instant_ngp(frames: list):
    """
    Dummy NeRF training simulation on in-memory frames.
    """
    # Here you can add any in-memory enhancements
    # Currently just prints info
    if not frames:
        return False
    print(f"[INFO] Training on {len(frames)} frames (in-memory)")
    return True

def render_vr180_views(frames: list) -> Tuple[list, list]:
    """
    Create left and right eye frames in-memory.
    Returns two lists: left_frames, right_frames
    """
    left_frames, right_frames = [], []
    
    for frame in frames:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (height * width)
        base_offset = 12
        stereo_offset = int(base_offset + edge_density * 8)
        stereo_offset = min(max(stereo_offset, 5), 20)

        # Left eye
        left_crop = frame[:, stereo_offset:]
        left_padded = np.pad(left_crop, ((0,0),(stereo_offset,0),(0,0)), mode='edge')
        left_matrix = np.float32([[1,0,-2],[0,1,0]])
        left_corrected = cv2.warpAffine(left_padded, left_matrix, (width,height))
        left_frames.append(left_corrected)

        # Right eye
        right_crop = frame[:, :-stereo_offset]
        right_padded = np.pad(right_crop, ((0,0),(0,stereo_offset),(0,0)), mode='edge')
        right_matrix = np.float32([[1,0,2],[0,1,0]])
        right_corrected = cv2.warpAffine(right_padded, right_matrix, (width,height))
        right_frames.append(right_corrected)
    
    return left_frames, right_frames

def combine_vr180_video(left_frames: list, right_frames: list, fps=30) -> bytes:
    """
    Combine left/right frames into a VR180 side-by-side video in-memory.
    Returns video bytes ready for download.
    """
    if not left_frames or not right_frames:
        raise ValueError("No frames provided for VR180 video")

    width, height = left_frames[0].shape[1], left_frames[0].shape[0]

    # Create FFmpeg input streams
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width*2,height), framerate=fps)
        .output('pipe:', format='mp4', pix_fmt='yuv420p', vcodec='libx264', crf=18)
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    # Write frames as side-by-side
    for l_frame, r_frame in zip(left_frames, right_frames):
        combined = np.concatenate((l_frame, r_frame), axis=1)
        process.stdin.write(combined.astype(np.uint8).tobytes())

    process.stdin.close()
    out, err = process.communicate()
    
    return out

def convert_to_vr180(video_file) -> bytes:
    """
    Main in-memory VR180 conversion.
    video_file: file-like object (Streamlit uploaded_file)
    Returns bytes of final MP4 video.
    """
    video_bytes = video_file.read()
    frames, fps = create_nerf_dataset_from_video_bytes(video_bytes)
    train_nerf_with_instant_ngp(frames)
    left_frames, right_frames = render_vr180_views(frames)
    vr180_bytes = combine_vr180_video(left_frames, right_frames, fps=fps)
    return vr180_bytes
