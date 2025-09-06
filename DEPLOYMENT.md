# VR180 Converter - Deployment Guide

## Quick Deployment Options

### Option 1: Vercel (Frontend) + Railway (Backend) - RECOMMENDED

#### Frontend (Vercel):
1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Import your GitHub repository
4. Set environment variable: `NEXT_PUBLIC_API_URL` = your backend URL
5. Deploy

#### Backend (Railway):
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway will automatically detect the Python app
4. Add environment variables if needed
5. Deploy

### Option 2: Vercel (Frontend) + Render (Backend)

#### Frontend (Vercel):
Same as above

#### Backend (Render):
1. Go to [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app:app`
6. Deploy

### Option 3: Full Vercel Deployment (Frontend Only)

If you want to deploy just the frontend and use a local backend:
1. Deploy frontend to Vercel
2. Set `NEXT_PUBLIC_API_URL` to your local backend URL
3. Keep backend running locally

## Environment Variables

### Frontend:
- `NEXT_PUBLIC_API_URL`: Backend API URL (e.g., https://your-backend.railway.app)

### Backend:
- `PORT`: Port number (automatically set by hosting platform)

## File Structure for Deployment

```
vr180-converter/
├── app.py                 # Flask backend
├── vr180_converter.py     # VR180 conversion logic
├── requirements.txt       # Python dependencies
├── Procfile              # Backend deployment config
├── app/                  # Next.js frontend
├── components/           # React components
├── lib/                  # API utilities
└── package.json          # Node.js dependencies
```

## Testing Deployment

1. Deploy backend first
2. Get the backend URL
3. Deploy frontend with the backend URL as environment variable
4. Test the full application

## Notes

- The backend requires FFmpeg for video processing
- Some hosting platforms may have limitations on file upload sizes
- Consider using cloud storage (AWS S3, etc.) for large video files in production
