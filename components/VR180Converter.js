import { useState } from "react";
import { convertVideo } from "../lib/api.js";

export default function VR180Converter() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [isConverting, setIsConverting] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError(""); // Clear any previous errors
    }
  };

  const handleConvert = async () => {
    if (!file) {
      setError("Please select a video first!");
      return;
    }
    
    setIsConverting(true);
    setError("");
    
    try {
      const url = await convertVideo(file);
      setVideoUrl(url);
    } catch (err) {
      console.error(err);
      setError(err.message || "Conversion failed!");
    } finally {
      setIsConverting(false);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>VR180 Video Converter</h1>
      <input type="file" accept="video/*" onChange={handleFileChange} />
      <button 
        onClick={handleConvert} 
        disabled={isConverting}
        style={{ 
          marginLeft: "10px",
          padding: "8px 16px",
          backgroundColor: isConverting ? "#ccc" : "#007bff",
          color: "white",
          border: "none",
          borderRadius: "4px",
          cursor: isConverting ? "not-allowed" : "pointer"
        }}
      >
        {isConverting ? "Converting..." : "Convert to VR180"}
      </button>

      {error && (
        <div style={{ 
          marginTop: "10px", 
          padding: "10px", 
          backgroundColor: "#f8d7da", 
          color: "#721c24", 
          border: "1px solid #f5c6cb",
          borderRadius: "4px"
        }}>
          Error: {error}
        </div>
      )}

      {videoUrl && (
        <div style={{ marginTop: "20px" }}>
          <h2>VR180 Video Preview</h2>
          <video src={videoUrl} controls width="600" />
          <br />
          <a 
            href={videoUrl} 
            download={`vr180_${file.name}`}
            style={{
              display: "inline-block",
              marginTop: "10px",
              padding: "8px 16px",
              backgroundColor: "#28a745",
              color: "white",
              textDecoration: "none",
              borderRadius: "4px"
            }}
          >
            Download VR180 Video
          </a>
        </div>
      )}
    </div>
  );
}
