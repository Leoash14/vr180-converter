const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/upload`, {
    method: "POST",
    body: formData,
  });
  return res.json();
}

export async function convertVideo(file) {
  // First upload the file
  const uploadResult = await uploadVideo(file);
  if (!uploadResult.success) {
    throw new Error(uploadResult.message || "Upload failed");
  }

  // Then convert the uploaded file
  const res = await fetch(`${API_URL}/convert`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: uploadResult.filename }),
  });
  const data = await res.json();
  
  if (!data.success) {
    throw new Error(data.message || "Conversion failed");
  }
  
  return data.url; // URL of the converted video
}