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

@app.route("/convert", methods=["POST"])
def convert_video():
    try:
        data = request.get_json()
        if not data or "filename" not in data:
            return jsonify({"success": False, "message": "No filename provided"}), 400
        
        filename = data["filename"]
        
        # Simulate conversion process
        import time
        time.sleep(2)  # Simulate processing time
        
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # For now, just return success with a mock URL
        output_filename = f"vr180_{filename}"
        
        return jsonify({
            "success": True,
            "message": "Video converted successfully (simulated)",
            "filename": output_filename,
            "url": f"/outputs/{output_filename}"
        })
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Conversion failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
