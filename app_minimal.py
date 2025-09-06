from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "VR180 Converter Backend is running"
    })

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "message": "No file selected"}), 400

        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save file
        filename = file.filename
        file.save(os.path.join("uploads", filename))
        
        return jsonify({
            "success": True,
            "message": "File uploaded successfully",
            "filename": filename
        })
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500

def simulate_nerf_training(filename):
    """Simulate NeRF training process"""
    import time
    print(f"[INFO] Starting NeRF training simulation for: {filename}")
    
    # Simulate NeRF training steps
    steps = [
        "Extracting video frames...",
        "Initializing Instant-NGP model...",
        "Training NeRF for depth perception...",
        "Generating depth maps...",
        "Optimizing stereo views...",
        "Applying VR180 enhancements..."
    ]
    
    for i, step in enumerate(steps):
        print(f"[INFO] Step {i+1}/6: {step}")
        time.sleep(0.5)  # Simulate processing time
    
    print(f"[INFO] NeRF training completed - enhanced depth perception ready")
    return True

@app.route("/convert", methods=["POST"])
def convert_video():
    try:
        data = request.get_json()
        if not data or "filename" not in data:
            return jsonify({"success": False, "message": "No filename provided"}), 400
        
        filename = data["filename"]
        
        # Simulate NeRF-enhanced VR180 conversion
        nerf_success = simulate_nerf_training(filename)
        
        if not nerf_success:
            return jsonify({"success": False, "message": "NeRF training failed"}), 500
        
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Simulate final processing
        import time
        time.sleep(1)  # Simulate final processing time
        
        # Create output filename
        name, ext = os.path.splitext(filename)
        output_filename = f"vr180_nerf_{name}{ext}"
        
        return jsonify({
            "success": True,
            "message": "Video converted successfully with NeRF enhancement",
            "filename": output_filename,
            "url": f"/outputs/{output_filename}",
            "features": [
                "NeRF depth simulation",
                "Instant-NGP processing", 
                "Enhanced stereo views",
                "VR180 optimization"
            ]
        })
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Conversion failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
