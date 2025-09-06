const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

// Mock conversion for demo purposes
function createMockVideo(file) {
  // Create a simple mock VR180 video using canvas
  const canvas = document.createElement('canvas');
  canvas.width = 1280;
  canvas.height = 360;
  const ctx = canvas.getContext('2d');
  
  // Create a simple stereo effect
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, 1280, 360);
  
  // Left eye
  ctx.fillStyle = '#4a90e2';
  ctx.fillRect(50, 50, 590, 260);
  ctx.fillStyle = 'white';
  ctx.font = '24px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('LEFT EYE', 345, 180);
  
  // Right eye
  ctx.fillStyle = '#e24a4a';
  ctx.fillRect(640, 50, 590, 260);
  ctx.fillStyle = 'white';
  ctx.fillText('RIGHT EYE', 935, 180);
  
  // Add VR180 text
  ctx.fillStyle = 'white';
  ctx.font = '32px Arial';
  ctx.fillText('VR180 CONVERTED VIDEO', 640, 320);
  
  return canvas.toDataURL('image/png');
}

export async function uploadVideo(file) {
  // Mock upload - always succeeds
  return {
    success: true,
    filename: file.name,
    size: file.size
  };
}

export async function convertVideo(file) {
  try {
    // Simulate upload
    const uploadResult = await uploadVideo(file);
    if (!uploadResult.success) {
      throw new Error(uploadResult.message || "Upload failed");
    }

    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Create mock VR180 video
    const mockVideoDataUrl = createMockVideo(file);
    
    // Convert data URL to blob
    const response = await fetch(mockVideoDataUrl);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    
    return url;
  } catch (error) {
    console.error('Mock conversion error:', error);
    throw new Error('Mock conversion failed');
  }
}