"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Upload, Play, Download, RotateCcw } from "lucide-react"
import { convertVideo } from "@/lib/api"

type AppState = "login" | "upload" | "processing" | "result"

export default function VR180Converter() {
  const [currentState, setCurrentState] = useState<AppState>("login")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [progress, setProgress] = useState(0)
  const [isConverting, setIsConverting] = useState(false)
  const [error, setError] = useState("")
  const [videoUrl, setVideoUrl] = useState("")

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    // Simple validation - in real app would authenticate
    if (email && password) {
      setCurrentState("upload")
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && (file.type === "video/mp4" || file.type === "video/quicktime" || file.type === "video/avi")) {
      setUploadedFile(file)
      setError("") // Clear any previous errors
    } else {
      setError("Please select a valid video file (MP4, MOV, or AVI)")
    }
  }

  const handleUpload = async () => {
    if (!uploadedFile) {
      setError("Please select a video file first!")
      return
    }
    
    setCurrentState("processing")
    setIsConverting(true)
    setError("")
    
    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 500)

      // Call the real backend API
      const url = await convertVideo(uploadedFile)
      setVideoUrl(url)
      
      clearInterval(progressInterval)
      setProgress(100)
      
      setTimeout(() => {
        setCurrentState("result")
        setIsConverting(false)
      }, 1000)
      
    } catch (err: any) {
      console.error(err)
      setError(err.message || "Conversion failed!")
      setCurrentState("upload")
      setIsConverting(false)
      setProgress(0)
    }
  }

  const handleConvertAnother = () => {
    setUploadedFile(null)
    setProgress(0)
    setVideoUrl("")
    setError("")
    setCurrentState("upload")
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {currentState === "login" && (
          <Card className="shadow-lg">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl font-bold text-foreground">VR180 Converter</CardTitle>
              <CardDescription className="text-muted-foreground">Login to Continue</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleLogin} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email" className="text-sm font-medium text-foreground">
                    Email
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="rounded-lg"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password" className="text-sm font-medium text-foreground">
                    Password
                  </Label>
                  <Input
                    id="password"
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="rounded-lg"
                    required
                  />
                </div>
                <Button type="submit" className="w-full rounded-lg h-12 text-lg font-semibold" size="lg">
                  Login
                </Button>
              </form>
            </CardContent>
          </Card>
        )}

        {currentState === "upload" && (
          <Card className="shadow-lg">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl font-bold text-foreground">Upload your 2D Clip</CardTitle>
              <CardDescription className="text-muted-foreground">
                Select a video file to convert to VR180 format using NeRF technology
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
                <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                <div className="space-y-2">
                  <Label htmlFor="file-upload" className="cursor-pointer">
                    <span className="text-sm font-medium text-foreground">Click to upload or drag and drop</span>
                    <br />
                    <span className="text-xs text-muted-foreground">MP4, MOV, AVI files supported</span>
                  </Label>
                  <Input
                    id="file-upload"
                    type="file"
                    accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </div>
              </div>

              {uploadedFile && (
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm font-medium text-foreground">Selected: {uploadedFile.name}</p>
                  <p className="text-xs text-muted-foreground">{(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              )}

              {error && (
                <div className="text-center p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <p className="text-sm text-destructive">{error}</p>
                </div>
              )}

              <Button
                onClick={handleUpload}
                disabled={!uploadedFile || isConverting}
                className="w-full rounded-lg h-12 text-lg font-semibold"
                size="lg"
              >
                {isConverting ? "Converting..." : "Upload & Convert with NeRF"}
              </Button>
            </CardContent>
          </Card>
        )}

        {currentState === "processing" && (
          <Card className="shadow-lg">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl font-bold text-foreground">NeRF Processing</CardTitle>
              <CardDescription className="text-muted-foreground">Converting your video to VR180 using Neural Radiance Fields...</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex flex-col items-center space-y-4">
                <div className="animate-spin rounded-full h-16 w-16 border-4 border-muted border-t-primary"></div>
                <div className="w-full bg-muted rounded-full h-3">
                  <div
                    className="bg-primary h-3 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="text-sm font-medium text-foreground">{progress}% Complete</p>
                <div className="text-xs text-muted-foreground text-center">
                  <p>• Extracting frames and creating NeRF dataset</p>
                  <p>• Training neural radiance field</p>
                  <p>• Rendering VR180 stereo views</p>
                  <p>• Generating final video</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {currentState === "result" && (
          <Card className="shadow-lg">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl font-bold text-foreground">VR180 Conversion Complete</CardTitle>
              <CardDescription className="text-muted-foreground">Your NeRF-generated VR180 video is ready</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="bg-muted rounded-lg p-6 text-center">
                <Play className="mx-auto h-16 w-16 text-primary mb-4" />
                <p className="text-sm font-medium text-foreground mb-2">VR180 Video Preview</p>
                <p className="text-xs text-muted-foreground">{uploadedFile?.name.replace(/\.[^/.]+$/, "")}_VR180.mp4</p>
                {videoUrl && (
                  <div className="mt-4">
                    <video 
                      src={videoUrl} 
                      controls 
                      className="w-full max-w-sm mx-auto rounded-lg"
                      style={{ maxHeight: '300px' }}
                      preload="metadata"
                      crossOrigin="anonymous"
                      onError={(e) => {
                        console.error('Video load error:', e);
                        setError('Failed to load video preview. You can still download the file.');
                      }}
                      onLoadStart={() => console.log('Video loading started')}
                      onCanPlay={() => console.log('Video can play')}
                    >
                      Your browser does not support the video tag.
                    </video>
                    <div className="mt-2 space-y-2">
                      <a 
                        href={videoUrl} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-xs text-muted-foreground hover:text-primary underline block"
                      >
                        Open video in new tab
                      </a>
                      <p className="text-xs text-muted-foreground">
                        Video URL: {videoUrl}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <Button 
                  className="w-full rounded-lg h-12 text-lg font-semibold" 
                  size="lg"
                  onClick={async () => {
                    try {
                      console.log('Download button clicked, URL:', videoUrl);
                      
                      // Method 1: Direct download using fetch
                      const response = await fetch(videoUrl);
                      const blob = await response.blob();
                      const url = window.URL.createObjectURL(blob);
                      
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = `vr180_${uploadedFile?.name || 'video.mp4'}`;
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                      
                      // Clean up the blob URL
                      window.URL.revokeObjectURL(url);
                      
                      console.log('Download initiated successfully');
                    } catch (error) {
                      console.error('Download failed:', error);
                      // Fallback: open in new tab
                      window.open(videoUrl, '_blank');
                    }
                  }}
                >
                  <Download className="mr-2 h-5 w-5" />
                  Download VR180 Video
                </Button>

                <Button
                  variant="outline"
                  onClick={handleConvertAnother}
                  className="w-full rounded-lg h-12 text-lg font-semibold bg-transparent"
                  size="lg"
                >
                  <RotateCcw className="mr-2 h-5 w-5" />
                  Convert Another Video
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
