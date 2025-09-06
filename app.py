from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import traceback
from vr180_converter import convert_to_vr180

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
            return jsonify({"success": True, "filename": filename, "size": file_size})
        else:
            print(f"[ERROR] File was not saved: {input_path}")
            return jsonify({"success": False, "message": "Failed to save file"}), 500
            
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500


@app.route("/convert", methods=["POST"])
def convert_video():
    try:
        data = request.json
        filename = data.get("filename")
        if not filename:
            return jsonify({"success": False, "message": "No filename provided"}), 400

        input_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(input_path):
            return jsonify({"success": False, "message": f"File not found: {input_path}"}), 404

        print(f"[INFO] Starting VR180 conversion for: {filename}")
        
        # Use real VR180 conversion
        output_file = convert_to_vr180(input_path)
        
        # Move the output to the outputs folder with a proper name
        output_name = f"vr180_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        
        # Remove existing file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        
        os.rename(output_file, output_path)
        
        if os.path.exists(output_path):
            print(f"[INFO] Conversion completed: {output_name}")
            return jsonify({
                "success": True,
                "url": f"http://127.0.0.1:5000/outputs/{output_name}",
                "filename": output_name
            })
        else:
            return jsonify({"success": False, "message": "Conversion failed - output file not created"}), 500
            
    except Exception as e:
        print(f"[ERROR] Conversion failed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": f"Conversion failed: {str(e)}"}), 500

@app.route("/outputs/<filename>")
def serve_output(filename):
    try:
        path = os.path.join(OUTPUT_FOLDER, filename)
        print(f"[INFO] Serving file: {path}")
        
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            return jsonify({"error": "File not found"}), 404
            
        # Determine MIME type based on file extension
        if filename.lower().endswith('.mp4'):
            mimetype = "video/mp4"
        elif filename.lower().endswith('.avi'):
            mimetype = "video/x-msvideo"
        elif filename.lower().endswith('.mov'):
            mimetype = "video/quicktime"
        else:
            mimetype = "video/mp4"  # default
            
        print(f"[INFO] Serving {filename} with MIME type: {mimetype}")
        return send_file(path, mimetype=mimetype, as_attachment=False)
        
    except Exception as e:
        print(f"[ERROR] Failed to serve file {filename}: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to serve file: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

