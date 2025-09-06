from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import traceback
import subprocess
import cv2
import numpy as np
from pathlib import Path
import time

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"[INFO] Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
print(f"[INFO] Output folder: {os.path.abspath(OUTPUT_FOLDER)}")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "VR180 Converter Backend is running",
        "upload_folder": os.path.abspath(UPLOAD_FOLDER),
        "output_folder": os.path.abspath(OUTPUT_FOLDER)
    })

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "message": "No file selected"}), 400

        # Create a safe filename
        filename = file.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        
        print(f"[INFO] Attempting to save file: {filename} to {input_path}")
        file.save(input_path)
        
        # Verify the file was saved
        if os.path.exists(input_path):
            file_size = os.path.getsize(input_path)
            print(f"[INFO] Successfully uploaded file: {filename} ({file_size} bytes)")
            return jsonify({
                "success": True,
                "message": "File uploaded successfully",
                "filename": filename,
                "size": file_size
            })
        else:
            print(f"[ERROR] File was not saved: {input_path}")
            return jsonify({"success": False, "message": "Failed to save file"}), 500
            
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500

def simple_vr180_conversion(input_path, output_path):
    """Simplified VR180 conversion using basic OpenCV operations"""
    try:
        print(f"[INFO] Starting VR180 conversion: {input_path} -> {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = width * 2  # Double width for side-by-side stereo
        out_height = height
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create stereo effect by duplicating and shifting the frame
            # Left eye: original frame
            left_eye = frame.copy()
            
            # Right eye: slightly shifted frame for stereo effect
            right_eye = frame.copy()
            
            # Add simple stereo offset (horizontal shift)
            stereo_offset = 10
            if right_eye.shape[1] > stereo_offset:
                right_eye[:, :-stereo_offset] = right_eye[:, stereo_offset:]
            
            # Combine left and right eyes side by side
            stereo_frame = np.hstack((left_eye, right_eye))
            
            # Write frame
            out.write(stereo_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"[INFO] Processed {frame_count}/{total_frames} frames")
        
        # Release everything
        cap.release()
        out.release()
        
        print(f"[INFO] VR180 conversion completed: {frame_count} frames processed")
        return True
        
    except Exception as e:
        print(f"[ERROR] VR180 conversion failed: {str(e)}")
        print(traceback.format_exc())
        return False

@app.route("/convert", methods=["POST"])
def convert_video():
    try:
        data = request.get_json()
        if not data or "filename" not in data:
            return jsonify({"success": False, "message": "No filename provided"}), 400
        
        filename = data["filename"]
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(input_path):
            return jsonify({"success": False, "message": "File not found"}), 404
        
        # Create output filename
        name, ext = os.path.splitext(filename)
        output_filename = f"vr180_{name}{ext}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Remove existing output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        
        print(f"[INFO] Starting conversion: {input_path} -> {output_path}")
        
        # Perform VR180 conversion
        success = simple_vr180_conversion(input_path, output_path)
        
        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"[INFO] Conversion successful: {output_filename} ({file_size} bytes)")
            
            return jsonify({
                "success": True,
                "message": "Video converted successfully",
                "filename": output_filename,
                "url": f"/outputs/{output_filename}",
                "size": file_size
            })
        else:
            return jsonify({"success": False, "message": "Conversion failed"}), 500
            
    except Exception as e:
        print(f"[ERROR] Conversion failed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": f"Conversion failed: {str(e)}"}), 500

@app.route("/outputs/<filename>", methods=["GET"])
def get_output(filename):
    try:
        path = os.path.join(OUTPUT_FOLDER, filename)
        
        if not os.path.exists(path):
            return jsonify({"error": "File not found"}), 404
        
        # Determine MIME type
        if filename.lower().endswith('.mp4'):
            mimetype = 'video/mp4'
        elif filename.lower().endswith('.avi'):
            mimetype = 'video/x-msvideo'
        else:
            mimetype = 'application/octet-stream'
        
        print(f"[INFO] Serving file: {filename} ({mimetype})")
        return send_file(path, mimetype=mimetype, as_attachment=False)
        
    except Exception as e:
        print(f"[ERROR] Failed to serve file {filename}: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to serve file: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
